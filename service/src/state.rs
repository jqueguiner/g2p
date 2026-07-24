//! Shared application state: the models directory, the set of available
//! `.g2p` blobs, and a lazily-populated in-memory cache of parsed models.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use g2p::Model;

use crate::calib::{self, Analyzed, Calibration};
use crate::error::ApiError;

/// Grammatical gender of a first name, from the corpus annotation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gender {
    Male,
    Female,
    Unisex,
}

impl Gender {
    /// Parse a corpus tag (`m`/`f`/anything-else → unisex).
    pub fn parse(s: &str) -> Gender {
        match s.trim().to_ascii_lowercase().chars().next() {
            Some('m') => Gender::Male,
            Some('f') => Gender::Female,
            _ => Gender::Unisex,
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Gender::Male => "m",
            Gender::Female => "f",
            Gender::Unisex => "u",
        }
    }

    /// Whether a candidate of this gender passes a requested `filter`.
    /// Unisex names pass a male or female filter (usable for either).
    pub fn passes(self, filter: Gender) -> bool {
        match filter {
            Gender::Male => self == Gender::Male || self == Gender::Unisex,
            Gender::Female => self == Gender::Female || self == Gender::Unisex,
            Gender::Unisex => self == Gender::Unisex,
        }
    }
}

/// A corpus first name: its text, gender, census frequency, and precomputed
/// phonetic structure.
pub struct NameEntry {
    pub name: String,
    pub gender: Gender,
    pub freq: u32,
    pub phon: Analyzed,
}

/// A language's name corpus, bucketed by onset (first phoneme) so a query only
/// scans candidates sharing its onset instead of the whole corpus.
pub struct NameIndex {
    pub by_onset: HashMap<Option<char>, Vec<NameEntry>>,
}

impl NameIndex {
    /// Candidates sharing `onset` (empty slice if none).
    pub fn bucket(&self, onset: Option<char>) -> &[NameEntry] {
        self.by_onset.get(&onset).map(Vec::as_slice).unwrap_or(&[])
    }
    pub fn is_empty(&self) -> bool {
        self.by_onset.values().all(Vec::is_empty)
    }
}

/// Process-wide state, shared behind an `Arc` across all requests.
pub struct AppState {
    /// Directory holding `<whisper>.g2p` model blobs.
    pub models_dir: PathBuf,
    /// Whisper codes for which a `.g2p` file exists on disk (scanned at boot).
    pub available: BTreeSet<String>,
    /// Parsed models, loaded on first use and kept resident.
    cache: RwLock<HashMap<String, Arc<Model>>>,
    /// Fallback language when detection fails and none is requested.
    pub default_lang: String,
    /// First-name corpus per language: `<lang>.txt`, one `name<TAB>gender` per
    /// line (scanned at boot from the names dir; gender defaults to unisex).
    names: HashMap<String, Arc<Vec<(String, Gender, u32, Option<String>)>>>,
    /// Per-language name index, built on first use. Each entry's phonetic
    /// structure (segments, syllable count, onset) is precomputed once here,
    /// not per comparison.
    name_index: RwLock<HashMap<String, Arc<NameIndex>>>,
    /// Surname corpus + lazily-built index (mirrors `names`/`name_index`;
    /// surnames carry no gender, stored as unisex).
    surnames: HashMap<String, Arc<Vec<(String, Gender, u32, Option<String>)>>>,
    surname_index: RwLock<HashMap<String, Arc<NameIndex>>>,
    /// Per-language similarity calibration, loaded from `<lang>.json`.
    calibrations: HashMap<String, Arc<Calibration>>,
    /// Fallback calibration (`default.json`, else built-in defaults).
    default_calibration: Arc<Calibration>,
}

impl AppState {
    /// Scan `models_dir` for `*.g2p`, the names dir for `*.txt`, and the
    /// calibration dir for `<lang>.json`.
    pub fn new(
        models_dir: PathBuf,
        names_dir: PathBuf,
        surnames_dir: PathBuf,
        calib_dir: PathBuf,
        default_lang: String,
    ) -> Self {
        let available = scan_models(&models_dir);
        let names = scan_names(&names_dir);
        let surnames = scan_names(&surnames_dir);
        let (calibrations, default_calibration) = scan_calibrations(&calib_dir);
        Self {
            models_dir,
            available,
            cache: RwLock::new(HashMap::new()),
            default_lang,
            names,
            name_index: RwLock::new(HashMap::new()),
            surnames,
            surname_index: RwLock::new(HashMap::new()),
            calibrations,
            default_calibration,
        }
    }

    /// Calibration for `lang`, falling back to the default profile.
    pub fn calibration(&self, lang: &str) -> Arc<Calibration> {
        self.calibrations
            .get(lang)
            .cloned()
            .unwrap_or_else(|| self.default_calibration.clone())
    }

    /// The fallback calibration (used when no language is resolved).
    pub fn calibration_default(&self) -> Arc<Calibration> {
        self.default_calibration.clone()
    }

    /// All calibration profiles (`"default"` plus each language), for inspection.
    pub fn all_calibrations(&self) -> std::collections::BTreeMap<String, Calibration> {
        let mut m = std::collections::BTreeMap::new();
        m.insert("default".to_string(), (*self.default_calibration).clone());
        for (k, v) in &self.calibrations {
            m.insert(k.clone(), (**v).clone());
        }
        m
    }

    /// Languages that have a first-name corpus loaded.
    pub fn name_langs(&self) -> BTreeSet<String> {
        self.names.keys().cloned().collect()
    }

    /// Languages that have a surname corpus loaded.
    pub fn surname_langs(&self) -> BTreeSet<String> {
        self.surnames.keys().cloned().collect()
    }

    /// Entry list for `lang` from `corpus`; falls back to the union of all lists
    /// when the language has no dedicated file.
    fn entries_for(
        corpus: &HashMap<String, Arc<Vec<(String, Gender, u32, Option<String>)>>>,
        lang: &str,
    ) -> Vec<(String, Gender, u32, Option<String>)> {
        if let Some(v) = corpus.get(lang) {
            return v.as_ref().clone();
        }
        let mut all: Vec<(String, Gender, u32, Option<String>)> =
            corpus.values().flat_map(|v| v.iter().cloned()).collect();
        all.sort_by(|a, b| a.0.cmp(&b.0));
        all.dedup_by(|a, b| a.0 == b.0);
        all
    }

    /// Phonemize + analyze a corpus into a cached index, keyed by `lang`.
    fn build_index(
        &self,
        corpus: &HashMap<String, Arc<Vec<(String, Gender, u32, Option<String>)>>>,
        cache: &RwLock<HashMap<String, Arc<NameIndex>>>,
        lang: &str,
        model: &Model,
    ) -> Arc<NameIndex> {
        if let Some(v) = cache.read().unwrap().get(lang) {
            return v.clone();
        }
        // Diphthong set is a language property → use the language's base profile
        // when precomputing syllable counts for the corpus.
        let diph = self.calibration(lang);
        let index: Vec<NameEntry> = Self::entries_for(corpus, lang)
            .into_iter()
            .map(|(name, gender, freq, ipa)| {
                // Use the precomputed IPA when present (fast); else phonemize.
                let ipa = ipa.unwrap_or_else(|| {
                    name.split_whitespace()
                        .map(|w| g2p::phonemize(model, w))
                        .collect::<String>()
                });
                NameEntry {
                    name,
                    gender,
                    freq,
                    phon: calib::analyze(&ipa, &diph.diphthongs),
                }
            })
            .collect();
        // Bucket by onset so queries scan only their onset group.
        let mut by_onset: HashMap<Option<char>, Vec<NameEntry>> = HashMap::new();
        for e in index {
            by_onset.entry(e.phon.onset()).or_default().push(e);
        }
        let arc = Arc::new(NameIndex { by_onset });
        cache.write().unwrap().insert(lang.to_string(), arc.clone());
        arc
    }

    /// First-name index for `lang`, built with `model` and cached.
    pub fn name_index(&self, lang: &str, model: &Model) -> Arc<NameIndex> {
        self.build_index(&self.names, &self.name_index, lang, model)
    }

    /// Surname index for `lang`, built with `model` and cached.
    pub fn surname_index(&self, lang: &str, model: &Model) -> Arc<NameIndex> {
        self.build_index(&self.surnames, &self.surname_index, lang, model)
    }

    /// `true` if a model blob for this Whisper code is present on disk.
    pub fn has_model(&self, lang: &str) -> bool {
        self.available.contains(lang)
    }

    /// Get a parsed model for `lang`, loading and caching it on first request.
    pub fn model(&self, lang: &str) -> Result<Arc<Model>, ApiError> {
        if let Some(m) = self.cache.read().unwrap().get(lang) {
            return Ok(m.clone());
        }
        if !self.available.contains(lang) {
            return Err(ApiError::no_model(lang, &self.available));
        }
        let path = self.models_dir.join(format!("{lang}.g2p"));
        let bytes = std::fs::read(&path)
            .map_err(|e| ApiError::internal(format!("read {}: {e}", path.display())))?;
        // `Model::from_bytes` asserts on malformed input; treat a bad blob as a
        // 500 rather than crashing the worker thread.
        let model = std::panic::catch_unwind(|| Model::from_bytes(&bytes))
            .map_err(|_| ApiError::internal(format!("corrupt model blob: {lang}.g2p")))?;
        let arc = Arc::new(model);
        self.cache
            .write()
            .unwrap()
            .insert(lang.to_string(), arc.clone());
        Ok(arc)
    }

    /// Whisper codes currently resident in the in-memory cache.
    pub fn loaded(&self) -> BTreeSet<String> {
        self.cache.read().unwrap().keys().cloned().collect()
    }
}

fn scan_models(dir: &Path) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return set;
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(code) = name.strip_suffix(".g2p") {
            set.insert(code.to_string());
        }
    }
    set
}

/// Load `<lang>.txt` name lists from the names dir. Each non-blank, non-`#` line
/// is `name` optionally followed by TAB `gender` (`m`/`f`/`u`) and TAB
/// `frequency`; a missing gender means unisex, a missing frequency means 0.
fn scan_names(dir: &Path) -> HashMap<String, Arc<Vec<(String, Gender, u32, Option<String>)>>> {
    let mut map = HashMap::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return map;
    };
    for entry in rd.flatten() {
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();
        let Some(code) = fname.strip_suffix(".txt") else {
            continue;
        };
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        // Each line: `name` [TAB gender] [TAB frequency]; gender/freq optional.
        let list: Vec<(String, Gender, u32, Option<String>)> = content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| {
                let mut it = l.split('\t');
                let name = it.next().unwrap_or("").trim().to_string();
                let gender = it.next().map(Gender::parse).unwrap_or(Gender::Unisex);
                let freq = it.next().and_then(|f| f.trim().parse().ok()).unwrap_or(0);
                // optional 4th column: precomputed IPA (skips runtime phonemize)
                let ipa = it.next().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
                (name, gender, freq, ipa)
            })
            .collect();
        if !list.is_empty() {
            map.insert(code.to_string(), Arc::new(list));
        }
    }
    map
}

/// Load `<lang>.json` calibration profiles. `default.json` becomes the fallback;
/// every other `<lang>.json` is keyed by its stem. Bad files are logged and skipped.
fn scan_calibrations(dir: &Path) -> (HashMap<String, Arc<Calibration>>, Arc<Calibration>) {
    let mut map = HashMap::new();
    let mut default = Calibration::default();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let fname = entry.file_name();
            let fname = fname.to_string_lossy();
            let Some(stem) = fname.strip_suffix(".json") else {
                continue;
            };
            let Ok(txt) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            match serde_json::from_str::<Calibration>(&txt) {
                Ok(c) => {
                    let c = c.sanitize();
                    if stem == "default" {
                        default = c;
                    } else {
                        map.insert(stem.to_string(), Arc::new(c));
                    }
                }
                Err(e) => eprintln!("calibration: skipping {fname}: {e}"),
            }
        }
    }
    (map, Arc::new(default))
}

#[cfg(test)]
mod tests {
    use super::Gender;

    #[test]
    fn gender_parse_and_filter() {
        assert_eq!(Gender::parse("m"), Gender::Male);
        assert_eq!(Gender::parse("f"), Gender::Female);
        assert_eq!(Gender::parse("u"), Gender::Unisex);
        assert_eq!(Gender::parse(""), Gender::Unisex);
        // unisex passes both male and female filters; opposite gender does not
        assert!(Gender::Unisex.passes(Gender::Male));
        assert!(Gender::Unisex.passes(Gender::Female));
        assert!(Gender::Male.passes(Gender::Male));
        assert!(!Gender::Male.passes(Gender::Female));
        assert!(!Gender::Female.passes(Gender::Male));
    }
}
