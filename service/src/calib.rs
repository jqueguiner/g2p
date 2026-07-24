//! Per-language calibrated phonetic similarity.
//!
//! Core g2p2 ships two fixed metrics: `weighted` (articulatory feature L1) and
//! `levenshtein` (0/1 per differing phoneme). Neither is right for every
//! language — weighted over-scores coincidental feature overlap (French
//! `Jean /ʒɑ̃/` vs `Guy /ɡi/` → 0.8), levenshtein throws away real phonetic
//! closeness (`p`~`b`).
//!
//! This module blends the two with **per-language** weights, plus penalties
//! **per phoneme class** (nasality, tone, length, vowel↔consonant), all loaded
//! from editable JSON (`calibration/<lang>.json`) and overridable per request.
//!
//! It also fixes a core blind spot: a nasalized vowel like `ɑ̃` is scored on its
//! base `ɑ` alone in core; here the nasal diacritic is read off the segment.

use serde::{Deserialize, Serialize};

use g2p::similarity::segments;

const NF: usize = 10; // feature dimensions, matching core's table

/// A language's similarity calibration. Every field has a sensible default so a
/// partial JSON (or a partial request override) still deserializes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Calibration {
    /// Human label for the profile (e.g. the language code); informational.
    pub lang: String,
    /// Mix of the two signals in the base substitution cost:
    /// `0.0` = pure exact-mismatch (levenshtein-like), `1.0` = pure feature
    /// distance. Per language.
    pub blend: f32,
    /// Insertion / deletion cost (0..1).
    pub gap: f32,
    /// Insertion / deletion cost for a **reducible** vowel (schwa / e-muet).
    /// French deletes these freely — "keguiner" is just Quéguiner /keɡine/ with
    /// the mute-e dropped, and countless `-ue-` spellings hide a `-ué-`. Much
    /// lower than `gap` so a name typed without the schwa still matches. See
    /// [`Calibration::reducible_vowels`].
    pub schwa_gap: f32,
    /// Extra cost when exactly one segment is nasalized (`~`, or a nasal base).
    pub nasal_penalty: f32,
    /// Extra cost when the tone marks differ (matters for tonal languages).
    pub tone_penalty: f32,
    /// Extra cost when length (`ː`) differs on one side only.
    pub length_penalty: f32,
    /// Extra cost when one segment is a vowel and the other a consonant.
    pub vowel_consonant_penalty: f32,
    /// Multiplier on the feature distance for vowel↔vowel substitutions.
    pub vowel_scale: f32,
    /// Multiplier on the feature distance for consonant↔consonant substitutions.
    pub consonant_scale: f32,
    /// Word-level penalty added when the two words' first phonemes differ.
    /// For names the onset dominates perceived similarity ("starts the same").
    pub onset_penalty: f32,
    /// Word-level penalty scaled by the relative length difference
    /// `|n-m| / max(n,m)` — a short name and a long one rarely "sound alike".
    pub length_ratio_penalty: f32,
    /// Word-level penalty scaled by the relative **syllable-count** difference
    /// (number of vowel nuclei): Michel (2) vs Mickaël (3) are structurally
    /// different even though their phoneme counts are close.
    pub syllable_penalty: f32,
    /// Vowel sequences that form a **single** syllable nucleus in this language
    /// (base-char pairs, e.g. `["aɪ","eɪ","aʊ"]` for English). Empty for
    /// languages without phonemic diphthongs (French: `a-ɛ` is a hiatus = two
    /// syllables). This makes vowel-nucleus counting language-correct.
    #[serde(default)]
    pub diphthongs: Vec<String>,
    /// Vowel bases treated as **reducible** (schwa / e-muet): inserting or
    /// deleting one costs [`Calibration::schwa_gap`] instead of `gap`, and such
    /// a vowel does not count toward the syllable-count penalty. Default `["ə"]`;
    /// French adds `"e"` (Quéguiner /keɡine/ ~ "keguiner" /kɡine/), so the two
    /// differ only by a free schwa, not by real syllable structure — while a
    /// full vowel like /a/ (Michel /miʃɛl/ vs Mickaël /mikaɛl/) stays penalized.
    #[serde(default = "default_reducible")]
    pub reducible_vowels: Vec<String>,
}

/// Default reducible-vowel set: the schwa. French overrides to add e-muet.
fn default_reducible() -> Vec<String> {
    vec!["ə".to_string()]
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            lang: "default".into(),
            blend: 0.4,
            gap: 1.0,
            schwa_gap: 0.35,
            nasal_penalty: 0.5,
            tone_penalty: 0.6,
            length_penalty: 0.15,
            vowel_consonant_penalty: 0.8,
            vowel_scale: 1.2,
            consonant_scale: 1.0,
            onset_penalty: 0.3,
            length_ratio_penalty: 0.2,
            syllable_penalty: 0.5,
            diphthongs: Vec::new(),
            reducible_vowels: default_reducible(),
        }
    }
}

impl Calibration {
    /// Clamp every knob into a sane range after loading / overriding, so a bad
    /// config can't make the normalized distance leave `0..1`.
    pub fn sanitize(mut self) -> Self {
        self.blend = self.blend.clamp(0.0, 1.0);
        self.gap = self.gap.clamp(0.0, 1.0);
        self.schwa_gap = self.schwa_gap.clamp(0.0, 1.0);
        self.nasal_penalty = self.nasal_penalty.clamp(0.0, 2.0);
        self.tone_penalty = self.tone_penalty.clamp(0.0, 2.0);
        self.length_penalty = self.length_penalty.clamp(0.0, 2.0);
        self.vowel_consonant_penalty = self.vowel_consonant_penalty.clamp(0.0, 2.0);
        self.vowel_scale = self.vowel_scale.clamp(0.0, 4.0);
        self.consonant_scale = self.consonant_scale.clamp(0.0, 4.0);
        self.onset_penalty = self.onset_penalty.clamp(0.0, 2.0);
        self.length_ratio_penalty = self.length_ratio_penalty.clamp(0.0, 2.0);
        self.syllable_penalty = self.syllable_penalty.clamp(0.0, 2.0);
        self
    }

    /// Apply a partial override (from a request body) on top of this base.
    pub fn merged(&self, ov: &CalibrationOverride) -> Calibration {
        let mut c = self.clone();
        if let Some(v) = ov.blend {
            c.blend = v;
        }
        if let Some(v) = ov.gap {
            c.gap = v;
        }
        if let Some(v) = ov.schwa_gap {
            c.schwa_gap = v;
        }
        if let Some(v) = ov.nasal_penalty {
            c.nasal_penalty = v;
        }
        if let Some(v) = ov.tone_penalty {
            c.tone_penalty = v;
        }
        if let Some(v) = ov.length_penalty {
            c.length_penalty = v;
        }
        if let Some(v) = ov.vowel_consonant_penalty {
            c.vowel_consonant_penalty = v;
        }
        if let Some(v) = ov.vowel_scale {
            c.vowel_scale = v;
        }
        if let Some(v) = ov.consonant_scale {
            c.consonant_scale = v;
        }
        if let Some(v) = ov.onset_penalty {
            c.onset_penalty = v;
        }
        if let Some(v) = ov.length_ratio_penalty {
            c.length_ratio_penalty = v;
        }
        if let Some(v) = ov.syllable_penalty {
            c.syllable_penalty = v;
        }
        if let Some(v) = &ov.diphthongs {
            c.diphthongs = v.clone();
        }
        if let Some(v) = &ov.reducible_vowels {
            c.reducible_vowels = v.clone();
        }
        c.sanitize()
    }
}

/// Partial calibration sent in a request to tune a single call live. Every field
/// optional; unset fields keep the language default.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct CalibrationOverride {
    pub blend: Option<f32>,
    pub gap: Option<f32>,
    pub schwa_gap: Option<f32>,
    pub nasal_penalty: Option<f32>,
    pub tone_penalty: Option<f32>,
    pub length_penalty: Option<f32>,
    pub vowel_consonant_penalty: Option<f32>,
    pub vowel_scale: Option<f32>,
    pub consonant_scale: Option<f32>,
    pub onset_penalty: Option<f32>,
    pub length_ratio_penalty: Option<f32>,
    pub syllable_penalty: Option<f32>,
    pub diphthongs: Option<Vec<String>>,
    pub reducible_vowels: Option<Vec<String>>,
}

/// An IPA string with its phonetic structure computed **once, up front** — the
/// segmentation, syllable-nucleus count, and onset. Build it with [`analyze`]
/// and reuse it across many comparisons (e.g. one query vs a whole name corpus)
/// so segmentation and vowel counting are not repeated per pair.
#[derive(Clone, Debug)]
pub struct Analyzed {
    /// The original IPA string.
    pub ipa: String,
    /// Phoneme segments (base + diacritics).
    segs: Vec<Box<str>>,
    /// Base char of the first segment (the onset), if any.
    onset: Option<char>,
}

impl Analyzed {
    /// Base char of the first segment (the onset), if any.
    pub fn onset(&self) -> Option<char> {
        self.onset
    }
    /// Number of phoneme segments.
    pub fn nsegs(&self) -> usize {
        self.segs.len()
    }
}

/// Precompute the phonetic structure of an IPA string. `diphthongs` are the
/// language's single-nucleus vowel sequences (see [`Calibration::diphthongs`]);
/// they make the syllable count language-correct.
pub fn analyze(ipa: &str, _diphthongs: &[String]) -> Analyzed {
    let segs = segments(ipa);
    let onset = segs.first().and_then(|s| s.chars().next());
    Analyzed {
        ipa: ipa.to_string(),
        segs,
        onset,
    }
}

/// True if `seg` is a reducible vowel (its base is in the calibration's
/// `reducible_vowels`) — a schwa / e-muet that French drops freely.
fn is_reducible(seg: &str, reducible: &[String]) -> bool {
    match base_char(seg) {
        Some(b) => reducible.iter().any(|r| r.starts_with(b)),
        None => false,
    }
}

/// Gap (insertion / deletion) cost for skipping `seg`: the cheap `schwa_gap`
/// when it is a reducible vowel, otherwise the normal `gap`.
fn gap_cost(seg: &str, c: &Calibration) -> f32 {
    if is_reducible(seg, &c.reducible_vowels) {
        c.schwa_gap
    } else {
        c.gap
    }
}

/// Like [`count_nuclei`] but reducible vowels (schwa / e-muet) do not open a
/// nucleus — the "core" syllable skeleton used by the syllable-count penalty.
fn count_core_nuclei(segs: &[Box<str>], diphthongs: &[String], reducible: &[String]) -> usize {
    let mut nuclei = 0usize;
    let mut prev_vowel_base: Option<char> = None;
    for seg in segs {
        if is_vowel(seg) {
            let base = seg.chars().next().unwrap();
            let non_syllabic = seg.chars().any(|c| c == '\u{032F}');
            let forms_diphthong = match prev_vowel_base {
                Some(p) => {
                    let pair: String = [p, base].iter().collect();
                    diphthongs.iter().any(|d| d.as_str() == pair)
                }
                None => false,
            };
            if !non_syllabic && !forms_diphthong && !is_reducible(seg, reducible) {
                nuclei += 1;
            }
            prev_vowel_base = Some(base);
        } else {
            prev_vowel_base = None;
        }
    }
    nuclei
}

/// Calibrated similarity between two raw IPA strings (analyzes both on the fly
/// using the calibration's diphthong set).
pub fn similarity(a: &str, b: &str, c: &Calibration) -> f32 {
    similarity_of(&analyze(a, &c.diphthongs), &analyze(b, &c.diphthongs), c)
}

/// Calibrated distance between two raw IPA strings (analyzes both on the fly).
#[allow(dead_code)] // public convenience mirror of `similarity`; used in tests
pub fn distance(a: &str, b: &str, c: &Calibration) -> f32 {
    distance_of(&analyze(a, &c.diphthongs), &analyze(b, &c.diphthongs), c)
}

/// Calibrated similarity between two pre-[`analyze`]d strings, in `0..1`.
pub fn similarity_of(a: &Analyzed, b: &Analyzed, c: &Calibration) -> f32 {
    (1.0 - distance_of(a, b, c)).max(0.0)
}

/// Calibrated distance between two pre-[`analyze`]d strings, in `0..1`.
/// Needleman-Wunsch over the (precomputed) segments plus word-level penalties
/// from the (precomputed) onset and syllable count.
pub fn distance_of(a: &Analyzed, b: &Analyzed, c: &Calibration) -> f32 {
    let (sa, sb) = (&a.segs, &b.segs);
    if sa.is_empty() && sb.is_empty() {
        return 0.0;
    }
    let (n, m) = (sa.len(), sb.len());
    let mut d = vec![vec![0f32; m + 1]; n + 1];
    // Border paths are all-gaps; a reducible (schwa) vowel costs `schwa_gap`.
    for i in 1..=n {
        d[i][0] = d[i - 1][0] + gap_cost(&sa[i - 1], c);
    }
    for j in 1..=m {
        d[0][j] = d[0][j - 1] + gap_cost(&sb[j - 1], c);
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = d[i - 1][j - 1] + sub_cost(&sa[i - 1], &sb[j - 1], c);
            let del = d[i - 1][j] + gap_cost(&sa[i - 1], c);
            let ins = d[i][j - 1] + gap_cost(&sb[j - 1], c);
            d[i][j] = sub.min(del).min(ins);
        }
    }
    // sub_cost and gap are both in 0..1, so the worst path costs max(n,m).
    let denom = (n.max(m) as f32).max(1.0);
    let mut dist = d[n][m] / denom;

    // Word-level, name-oriented adjustments from the precomputed structure: the
    // onset and the syllable count matter more to "does this name sound like
    // that one" than a single shared interior phoneme (Jean~Roland share only
    // the final nasal; Michel(2 syll) vs Mickaël(3 syll) are structurally apart).
    if a.onset != b.onset {
        dist += c.onset_penalty;
    }
    dist += c.length_ratio_penalty * (n.abs_diff(m) as f32 / denom);
    // Syllable-count penalty on the *core* nuclei — reducible vowels (schwa /
    // e-muet) are excluded, so a dropped mute-e ("keguiner" 2 syll ~ Quéguiner
    // 3 syll, both 1 core nucleus) is not treated as a structural difference,
    // while a full extra vowel (Michel 2 ~ Mickaël 3, cores 2 vs 3) still is.
    let (ca, cb) = (
        count_core_nuclei(sa, &c.diphthongs, &c.reducible_vowels),
        count_core_nuclei(sb, &c.diphthongs, &c.reducible_vowels),
    );
    if ca != cb {
        let sd = ca.abs_diff(cb) as f32;
        dist += c.syllable_penalty * (sd / ca.max(cb).max(1) as f32);
    }

    dist.min(1.0)
}

/// Substitution cost between two phoneme segments, in `0..1`.
fn sub_cost(a: &str, b: &str, c: &Calibration) -> f32 {
    if a == b {
        return 0.0;
    }
    let (va, vb) = (is_vowel(a), is_vowel(b));
    let scale = if va && vb {
        c.vowel_scale
    } else if !va && !vb {
        c.consonant_scale
    } else {
        1.0
    };
    let feat = (feature_distance(a, b) * scale).min(1.0);
    let exact = 1.0; // they differ
    let mut cost = c.blend * feat + (1.0 - c.blend) * exact;

    if nasalized(a) != nasalized(b) {
        cost += c.nasal_penalty;
    }
    if long(a) != long(b) {
        cost += c.length_penalty;
    }
    if tone(a) != tone(b) {
        cost += c.tone_penalty;
    }
    if va != vb {
        cost += c.vowel_consonant_penalty;
    }
    cost.clamp(0.0, 1.0)
}

// ---- segment analysis ----

fn base_char(seg: &str) -> Option<char> {
    seg.chars().next()
}

/// Vowel iff the base symbol's `syllabic` feature is +1; unknowns are consonants.
fn is_vowel(seg: &str) -> bool {
    base_char(seg)
        .and_then(features)
        .map(|f| f[0] == 1)
        .unwrap_or(false)
}

/// Nasalized: a combining tilde on the segment, or a nasal base consonant.
fn nasalized(seg: &str) -> bool {
    seg.chars().any(|c| c == '\u{0303}')
        || base_char(seg)
            .and_then(features)
            .map(|f| f[2] == 1)
            .unwrap_or(false)
}

fn long(seg: &str) -> bool {
    seg.contains('\u{02D0}') || seg.contains('\u{02D1}')
}

/// Tone signature: the tone letters / superscript digits carried by the segment
/// (empty when none). Two segments "match on tone" iff these are equal.
fn tone(seg: &str) -> String {
    seg.chars().filter(|&c| is_tone_mark(c)).collect()
}

#[inline]
fn is_tone_mark(c: char) -> bool {
    let u = c as u32;
    (0x02E5..=0x02E9).contains(&u)       // ˥˦˧˨˩ tone letters
        || (0x2070..=0x209F).contains(&u) // superscripts/subscripts
        || matches!(c, '\u{00B2}' | '\u{00B3}' | '\u{00B9}') // ² ³ ¹
        || c.is_numeric() // Chao tone digits
}

/// Feature vector for a segment: its base symbol's vector, with the nasal
/// dimension forced on when a nasal diacritic is present. `None` if the base is
/// not in the table.
fn seg_features(seg: &str) -> Option<[i8; NF]> {
    let mut f = features(base_char(seg)?)?;
    if seg.chars().any(|c| c == '\u{0303}') {
        f[2] = 1; // nasal
    }
    Some(f)
}

/// Feature distance in `0..1`; falls back like core for unknown symbols.
fn feature_distance(a: &str, b: &str) -> f32 {
    match (seg_features(a), seg_features(b)) {
        (Some(fa), Some(fb)) => {
            let l1: i32 = fa
                .iter()
                .zip(fb.iter())
                .map(|(x, y)| (*x as i32 - *y as i32).abs())
                .sum();
            l1 as f32 / (2.0 * NF as f32)
        }
        _ if base_char(a) == base_char(b) => 0.2,
        _ => 1.0,
    }
}

/// Articulatory features per base IPA symbol (mirrors core's table).
/// Dims: syllabic, voiced, nasal, continuant, labial, coronal, dorsal, high, back, round.
#[rustfmt::skip]
fn features(c: char) -> Option<[i8; NF]> {
    Some(match c {
        'p' => [-1,-1,-1,-1, 1, 0, 0, 0, 0, 0],
        'b' => [-1, 1,-1,-1, 1, 0, 0, 0, 0, 0],
        't' => [-1,-1,-1,-1, 0, 1, 0, 0, 0, 0],
        'd' => [-1, 1,-1,-1, 0, 1, 0, 0, 0, 0],
        'k' => [-1,-1,-1,-1, 0, 0, 1, 1, 0, 0],
        'g' | 'ɡ' => [-1, 1,-1,-1, 0, 0, 1, 1, 0, 0],
        'q' => [-1,-1,-1,-1, 0, 0, 1, 0, 1, 0],
        'ʔ' => [-1,-1,-1,-1, 0, 0, 0, 0, 0, 0],
        'm' => [-1, 1, 1,-1, 1, 0, 0, 0, 0, 0],
        'ɱ' => [-1, 1, 1,-1, 1, 0, 0, 0, 0, 0],
        'n' => [-1, 1, 1,-1, 0, 1, 0, 0, 0, 0],
        'ŋ' => [-1, 1, 1,-1, 0, 0, 1, 1, 0, 0],
        'ɲ' => [-1, 1, 1,-1, 0, 1, 1, 1, 0, 0],
        'ɴ' => [-1, 1, 1,-1, 0, 0, 1, 0, 1, 0],
        'f' => [-1,-1,-1, 1, 1, 0, 0, 0, 0, 0],
        'v' => [-1, 1,-1, 1, 1, 0, 0, 0, 0, 0],
        'θ' => [-1,-1,-1, 1, 0, 1, 0, 0, 0, 0],
        'ð' => [-1, 1,-1, 1, 0, 1, 0, 0, 0, 0],
        's' => [-1,-1,-1, 1, 0, 1, 0, 0, 0, 0],
        'z' => [-1, 1,-1, 1, 0, 1, 0, 0, 0, 0],
        'ʃ' => [-1,-1,-1, 1, 0, 1, 0, 1, 0, 0],
        'ʒ' => [-1, 1,-1, 1, 0, 1, 0, 1, 0, 0],
        'ç' => [-1,-1,-1, 1, 0, 1, 1, 1, 0, 0],
        'x' => [-1,-1,-1, 1, 0, 0, 1, 1, 0, 0],
        'ɣ' => [-1, 1,-1, 1, 0, 0, 1, 1, 0, 0],
        'χ' => [-1,-1,-1, 1, 0, 0, 1, 0, 1, 0],
        'ʁ' => [-1, 1,-1, 1, 0, 0, 1, 0, 1, 0],
        'ħ' => [-1,-1,-1, 1, 0, 0, 1, 0, 1, 0],
        'ʕ' => [-1, 1,-1, 1, 0, 0, 1, 0, 1, 0],
        'h' => [-1,-1,-1, 1, 0, 0, 0, 0, 0, 0],
        'ɸ' => [-1,-1,-1, 1, 1, 0, 0, 0, 0, 0],
        'l' => [-1, 1,-1, 1, 0, 1, 0, 0, 0, 0],
        'ɭ' => [-1, 1,-1, 1, 0, 1, 0, 0, 0, 0],
        'r' => [-1, 1,-1, 1, 0, 1, 0, 0, 0, 0],
        'ɾ' => [-1, 1,-1, 1, 0, 1, 0, 0, 0, 0],
        'ɹ' => [-1, 1,-1, 1, 0, 1, 0, 0, 0, 0],
        'w' => [-1, 1,-1, 1, 1, 0, 1, 1, 1, 1],
        'j' => [-1, 1,-1, 1, 0, 1, 1, 1, 0, 0],
        'ɥ' => [-1, 1,-1, 1, 1, 1, 1, 1, 0, 1],
        'i' => [ 1, 1,-1, 1, 0, 0, 0, 1,-1,-1],
        'y' => [ 1, 1,-1, 1, 0, 0, 0, 1,-1, 1],
        'ɪ' => [ 1, 1,-1, 1, 0, 0, 0, 1,-1,-1],
        'e' => [ 1, 1,-1, 1, 0, 0, 0, 0,-1,-1],
        'ø' => [ 1, 1,-1, 1, 0, 0, 0, 0,-1, 1],
        'ɛ' => [ 1, 1,-1, 1, 0, 0, 0,-1,-1,-1],
        'œ' => [ 1, 1,-1, 1, 0, 0, 0,-1,-1, 1],
        'æ' => [ 1, 1,-1, 1, 0, 0, 0,-1,-1,-1],
        'a' => [ 1, 1,-1, 1, 0, 0, 0,-1, 0,-1],
        'ə' => [ 1, 1,-1, 1, 0, 0, 0, 0, 0,-1],
        'ɐ' => [ 1, 1,-1, 1, 0, 0, 0,-1, 0,-1],
        'ɑ' => [ 1, 1,-1, 1, 0, 0, 0,-1, 1,-1],
        'ɔ' => [ 1, 1,-1, 1, 0, 0, 0,-1, 1, 1],
        'o' => [ 1, 1,-1, 1, 0, 0, 0, 0, 1, 1],
        'ʊ' => [ 1, 1,-1, 1, 0, 0, 0, 1, 1, 1],
        'u' => [ 1, 1,-1, 1, 0, 0, 0, 1, 1, 1],
        'ɯ' => [ 1, 1,-1, 1, 0, 0, 0, 1, 1,-1],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr() -> Calibration {
        Calibration {
            lang: "fr".into(),
            blend: 0.35,
            nasal_penalty: 0.6,
            ..Default::default()
        }
    }

    #[test]
    fn identical_is_one() {
        assert_eq!(similarity("ʒɑ̃", "ʒɑ̃", &fr()), 1.0);
    }

    #[test]
    fn jean_guy_is_low_jeanne_beats_guy() {
        let c = fr();
        let guy = similarity("ʒɑ̃", "ɡi", &c);
        let jeanne = similarity("ʒɑ̃", "ʒan", &c);
        assert!(guy < 0.3, "Jean~Guy should be low, got {guy}");
        assert!(jeanne > guy, "Jeanne ({jeanne}) should beat Guy ({guy})");
    }

    #[test]
    fn onset_match_beats_shared_final_rhyme() {
        // Jean /ʒɑ̃/: Jeanne /ʒan/ shares the ʒ onset; Roland /ʁɔlɑ̃/ shares only
        // the final nasal. The onset match must rank higher.
        let c = fr();
        let jeanne = similarity("ʒɑ̃", "ʒan", &c);
        let roland = similarity("ʒɑ̃", "ʁɔlɑ̃", &c);
        assert!(
            jeanne > roland,
            "Jeanne ({jeanne}) should beat Roland ({roland})"
        );
    }

    #[test]
    fn diphthongs_are_language_specific() {
        // English merges aɪ/eɪ into one nucleus; French keeps a-ɛ as a hiatus.
        let en = vec!["aɪ".to_string(), "eɪ".to_string()];
        let fr_none: Vec<String> = vec![];
        let syl = |ipa: &str, d: &[String]| count_core_nuclei(&analyze(ipa, d).segs, d, &[]);
        // Michael /maɪkəl/ → EN: 2 nuclei (aɪ + ə)
        assert_eq!(syl("maɪkəl", &en), 2);
        // Mickaël /mikaɛl/ → FR: 3 nuclei (i, a, ɛ — no diphthongs)
        assert_eq!(syl("mikaɛl", &fr_none), 3);
        // same string, EN diphthong set would fold a+ɛ only if listed; it isn't,
        // so aɛ stays 2 nuclei there too — the point is the set is per language.
        assert_eq!(syl("keɪtlaɪn", &en), 2); // Caitlin: eɪ + aɪ
    }

    #[test]
    fn same_syllable_count_beats_different() {
        // Michel /miʃɛl/ (2 nuclei): Michèle /miʃɛl/ (2) must beat Mickaël
        // /mikaɛl/ (3) — same onset, but the syllable structure differs.
        let c = fr();
        let michele = similarity("miʃɛl", "miʃɛl", &c);
        let mickael = similarity("miʃɛl", "mikaɛl", &c);
        assert!(michele > mickael);
        // and the 3-syllable form is pushed down by the syllable penalty
        assert!(mickael < 0.7, "Mickaël should be penalized, got {mickael}");
    }

    #[test]
    fn reducible_vowel_insertion_is_cheap() {
        // French e-muet: "keguiner" /kɡine/ is Quéguiner /keɡine/ with the mute-e
        // dropped. With `e` reducible they must be near-identical, and must beat a
        // real consonant swap like Criner /kʁine/ (same 2 syllables, so it used to
        // win only because it kept the syllable count).
        let mut c = fr();
        c.schwa_gap = 0.3;
        c.reducible_vowels = vec!["ə".into(), "e".into()];
        let queguiner = similarity("kɡine", "keɡine", &c);
        let criner = similarity("kɡine", "kʁine", &c);
        assert!(
            queguiner > 0.85,
            "keguiner~Quéguiner should be high, got {queguiner}"
        );
        assert!(
            queguiner > criner,
            "Quéguiner ({queguiner}) must beat Criner ({criner})"
        );
    }

    #[test]
    fn full_vowel_insertion_stays_penalized() {
        // The reducible-vowel discount must NOT rescue a full extra vowel: Mickaël
        // /mikaɛl/ inserts an /a/ (not reducible) vs Michel /miʃɛl/, so it stays
        // clearly apart even though French has e-muet.
        let mut c = fr();
        c.schwa_gap = 0.3;
        c.reducible_vowels = vec!["ə".into(), "e".into()];
        let mickael = similarity("miʃɛl", "mikaɛl", &c);
        assert!(
            mickael < 0.7,
            "Mickaël should stay penalized, got {mickael}"
        );
    }

    #[test]
    fn nasal_diacritic_is_read() {
        // ɑ̃ vs ɑ must cost more than ɑ vs ɑ (0), because of the nasal diacritic.
        let c = Calibration::default();
        assert!(distance("ɑ̃", "ɑ", &c) > 0.0);
    }

    #[test]
    fn tone_mismatch_penalized_when_configured() {
        let mut c = Calibration::default();
        c.tone_penalty = 0.9;
        // same base vowel, different Chao tone digits
        let d_tone = distance("a˥", "a˩", &c);
        assert!(d_tone > 0.0);
    }

    #[test]
    fn blend_extremes() {
        // isolate the substitution cost from the word-level onset/length terms
        let mut c = Calibration {
            onset_penalty: 0.0,
            length_ratio_penalty: 0.0,
            ..Default::default()
        };
        c.blend = 0.0; // pure exact: any differing single segment -> distance 1
        assert!((distance("p", "b", &c) - 1.0).abs() < 1e-6);
        c.blend = 1.0; // pure feature: p~b differ in one dim only -> small
        assert!(distance("p", "b", &c) < 0.2);
    }

    #[test]
    fn override_merges() {
        let base = Calibration::default();
        let ov = CalibrationOverride {
            blend: Some(1.0),
            ..Default::default()
        };
        let merged = base.merged(&ov);
        assert_eq!(merged.blend, 1.0);
        assert_eq!(merged.gap, base.gap); // untouched
    }

    #[test]
    fn override_clamps_out_of_range() {
        let ov = CalibrationOverride {
            blend: Some(5.0),
            gap: Some(-1.0),
            ..Default::default()
        };
        let m = Calibration::default().merged(&ov);
        assert!((0.0..=1.0).contains(&m.blend), "blend clamped: {}", m.blend);
        assert!(m.gap >= 0.0, "gap clamped: {}", m.gap);
    }

    #[test]
    fn length_mark_penalized() {
        let c = Calibration {
            length_penalty: 0.5,
            onset_penalty: 0.0,
            length_ratio_penalty: 0.0,
            syllable_penalty: 0.0,
            ..Default::default()
        };
        // aː vs a: same base vowel, only length differs -> non-zero distance
        assert!(distance("aː", "a", &c) > 0.0);
    }

    #[test]
    fn non_syllabic_glide_is_not_a_nucleus() {
        // aɪ̯ (ɪ carries U+032F) is one nucleus; ai (two full vowels) is two
        assert_eq!(count_core_nuclei(&analyze("aɪ̯", &[]).segs, &[], &[]), 1);
        assert_eq!(count_core_nuclei(&analyze("ai", &[]).segs, &[], &[]), 2);
    }

    #[test]
    fn analyzed_and_str_paths_agree() {
        let c = Calibration::default();
        let s_str = similarity("aba", "abo", &c);
        let s_an = similarity_of(
            &analyze("aba", &c.diphthongs),
            &analyze("abo", &c.diphthongs),
            &c,
        );
        assert!((s_str - s_an).abs() < 1e-6);
    }
}
