//! Convert a kaikki.org wiktextract dump into the WikiPron-style TSV the
//! builder already eats (`word <TAB> space-separated IPA`).
//!
//! Why bother, when `fetch` already pulls WikiPron: WikiPron scrapes the
//! *English* Wiktionary, so a language's coverage there is whatever English
//! editors happened to document. Its own edition is usually far richer --
//! French is 84k words via WikiPron against ~2M entries on fr.wiktionary. The
//! gap is mostly proper nouns and toponyms, which is exactly what a G2P model
//! cannot guess by rule.
//!
//! Two properties of the source drive the code below.
//!
//! Notation: editions disagree. fr.wiktionary writes the phonemic form as
//! `\a.kœj\` and a phonetic one as `[ɛ̃.n‿a.kœj]`; most other editions use
//! `/.../` for phonemic and `[...]` for phonetic. The phonemic form is a
//! property of the word on its own -- the phonetic one is a contextual
//! realisation (that fr example is "un accueil", article and liaison included).
//! So the broad form (`\...\` or `/.../`) is preferred and the narrow `[...]`
//! is taken only when a word has nothing else. See `phonemic`.
//!
//! Segmentation: kaikki writes a syllabified string, the builder wants
//! phonemes. Dots and stress marks are not phonemes, and a combining mark
//! belongs to its base -- ɑ + U+0303 is one nasal vowel, not two segments.
//!
//! No JSON crate: the runtime is zero-dependency and xtask keeps to it, so the
//! scanner below reads only the fields this needs and skips the rest. Likewise
//! gzip is shelled out to, as `fetch` does with curl.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// One kaikki record, reduced to the fields the builder needs.
pub struct Entry {
    pub word: String,
    pub lang_code: String,
    pub ipas: Vec<String>,
}

/// Cursor over one JSON line.
///
/// Not a general-purpose parser: it materialises the strings this module asks
/// for and structurally skips everything else, which keeps a 3 GB dump cheap to
/// walk. It does handle the parts of the grammar that bite -- escapes and
/// surrogate pairs inside strings, and braces/brackets nested inside strings.
struct Json<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> Json<'a> {
    fn new(s: &'a str) -> Self {
        Json { s, i: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.s[self.i..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.i += c.len_utf8();
        Some(c)
    }

    fn ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn eat(&mut self, want: char) -> Option<()> {
        self.ws();
        (self.peek()? == want).then(|| {
            self.i += want.len_utf8();
        })
    }

    /// `\uXXXX`, joining a surrogate pair into one char.
    fn unicode_escape(&mut self) -> Option<char> {
        let hex = self.s.get(self.i..self.i + 4)?;
        let hi = u32::from_str_radix(hex, 16).ok()?;
        self.i += 4;
        if !(0xD800..0xDC00).contains(&hi) {
            return char::from_u32(hi);
        }
        // High surrogate: the low half must follow as its own \uXXXX.
        if !self.s[self.i..].starts_with("\\u") {
            return Some('\u{FFFD}');
        }
        self.i += 2;
        let hex = self.s.get(self.i..self.i + 4)?;
        let lo = u32::from_str_radix(hex, 16).ok()?;
        self.i += 4;
        char::from_u32(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00))
    }

    fn string(&mut self) -> Option<String> {
        self.eat('"')?;
        let mut out = String::new();
        loop {
            match self.bump()? {
                '"' => return Some(out),
                '\\' => match self.bump()? {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000C}'),
                    'u' => out.push(self.unicode_escape()?),
                    // \" \\ \/ and anything else: the char stands for itself.
                    c => out.push(c),
                },
                c => out.push(c),
            }
        }
    }

    /// Advance past one value without building it.
    fn skip_value(&mut self) -> Option<()> {
        self.ws();
        match self.peek()? {
            '"' => {
                self.string()?;
            }
            '{' => self.skip_nest('{', '}')?,
            '[' => self.skip_nest('[', ']')?,
            _ => {
                // number / true / false / null: run to the next structural char.
                while let Some(c) = self.peek() {
                    if matches!(c, ',' | '}' | ']') {
                        break;
                    }
                    self.i += c.len_utf8();
                }
            }
        }
        Some(())
    }

    /// Skip a balanced container. Strings are consumed whole so that a brace
    /// inside a gloss cannot unbalance the count.
    fn skip_nest(&mut self, open: char, close: char) -> Option<()> {
        self.eat(open)?;
        let mut depth = 1usize;
        while depth > 0 {
            self.ws();
            match self.peek()? {
                '"' => {
                    self.string()?;
                }
                c if c == open => {
                    depth += 1;
                    self.i += c.len_utf8();
                }
                c if c == close => {
                    depth -= 1;
                    self.i += c.len_utf8();
                }
                c => self.i += c.len_utf8(),
            }
        }
        Some(())
    }

    /// `sounds: [{ipa: "..."}, ...]` -> every `ipa` in it.
    fn sounds_ipas(&mut self) -> Option<Vec<String>> {
        let mut out = Vec::new();
        self.eat('[')?;
        loop {
            self.ws();
            match self.peek()? {
                ']' => {
                    self.i += 1;
                    return Some(out);
                }
                ',' => {
                    self.i += 1;
                }
                '{' => {
                    self.eat('{')?;
                    loop {
                        self.ws();
                        match self.peek()? {
                            '}' => {
                                self.i += 1;
                                break;
                            }
                            ',' => {
                                self.i += 1;
                            }
                            '"' => {
                                let k = self.string()?;
                                self.eat(':')?;
                                if k == "ipa" {
                                    out.push(self.string()?);
                                } else {
                                    self.skip_value()?;
                                }
                            }
                            _ => return Some(out), // malformed: take what we have
                        }
                    }
                }
                _ => self.skip_value()?,
            }
        }
    }
}

/// Pull `word`, `lang_code` and `sounds[].ipa` out of one kaikki line.
/// `None` when the line is not a JSON object or carries no word.
pub fn parse_line(line: &str) -> Option<Entry> {
    let mut p = Json::new(line);
    let mut e = Entry {
        word: String::new(),
        lang_code: String::new(),
        ipas: Vec::new(),
    };
    p.eat('{')?;
    loop {
        p.ws();
        match p.peek()? {
            '}' => break,
            ',' => {
                p.i += 1;
            }
            '"' => {
                let k = p.string()?;
                p.eat(':')?;
                match k.as_str() {
                    "word" => e.word = p.string()?,
                    "lang_code" => e.lang_code = p.string()?,
                    "sounds" => e.ipas = p.sounds_ipas()?,
                    _ => p.skip_value()?,
                }
            }
            _ => return None,
        }
    }
    (!e.word.is_empty()).then_some(e)
}

/// Keep the phonemic (`\...\`) readings and drop the phonetic (`[...]`) ones.
pub fn phonemic(ipas: &[String]) -> Vec<&str> {
    // Broad first, narrow only as a fallback. Editions disagree on notation:
    // fr.wiktionary writes phonemic as `\...\`, most others use `/.../` (broad)
    // and `[...]` (narrow, allophonic). Slashes and backslashes are both the
    // citation form we want; brackets carry contextual detail and are taken only
    // when a word has nothing else.
    let broad: Vec<&str> = ipas
        .iter()
        .filter_map(|s| {
            unwrap_delim(s.trim(), '\\', '\\').or_else(|| unwrap_delim(s.trim(), '/', '/'))
        })
        .filter(|s| looks_like_ipa(s))
        .collect();
    if !broad.is_empty() {
        return broad;
    }
    let narrow: Vec<&str> = ipas
        .iter()
        .filter_map(|s| unwrap_delim(s.trim(), '[', ']'))
        .filter(|s| looks_like_ipa(s))
        .collect();
    if !narrow.is_empty() {
        return narrow;
    }
    // Some editions write the reading bare, no delimiters at all -- el has
    // `ˈle.ksi` (790k of its 797k entries carried nothing else). Last resort,
    // same junk guard, and it must contain a real letter to be a reading --
    // stress/length marks are Unicode Lm "letters" (ˈ, ː), so is_alphabetic
    // alone would accept a string of bare marks.
    ipas.iter()
        .map(|s| s.trim())
        .filter(|s| {
            !s.is_empty()
                && s.chars()
                    .any(|c| c.is_alphabetic() && !matches!(c as u32, 0x02B0..=0x02FF))
                && looks_like_ipa(s)
        })
        .collect()
}

/// Content between matching single-char delimiters, or None.
fn unwrap_delim(s: &str, open: char, close: char) -> Option<&str> {
    let inner = s.strip_prefix(open)?.strip_suffix(close)?.trim();
    (!inner.is_empty()).then_some(inner)
}

/// Reject readings that are not IPA. Some editions carry X-SAMPA
/// (`e b o 4 e ~ s i`, `@` for schwa), ASCII approximations (`e."sEx.tu`),
/// bookkeeping placeholders (`…`, `*`), or two forms packed into one field with
/// a `/` separator. Any of these poisons the whole reading, so drop it and let
/// another `sounds` entry (or the n-gram) cover the word.
///
/// `:` and ASCII `'` are NOT rejected here -- they are fixable stand-ins for `ː`
/// and stress, normalised in `segment`.
fn looks_like_ipa(s: &str) -> bool {
    !s.chars().any(|c| {
        c.is_ascii_uppercase()
            || c.is_ascii_digit()
            || matches!(c, '"' | '@' | '~' | '_' | '…' | '*' | ';' | '/')
    })
}

/// Not phonemes: syllable breaks and stress. ASCII `'` (U+0027) is a stress
/// stand-in some editions use for `ˈ` -- dropped like the real mark. The IPA
/// ejective `ʼ` (U+02BC) is a different character and is kept.
fn is_drop(c: char) -> bool {
    matches!(c, '.' | 'ˈ' | 'ˌ' | '\'' | ' ' | '|' | '‖' | '(' | ')')
}

/// Rides with the preceding base rather than standing alone.
fn is_attach(c: char) -> bool {
    matches!(c as u32, 0x0300..=0x035B | 0x035D..=0x036F)
        || matches!(
            c,
            'ː' | 'ˑ' | 'ʰ' | 'ʷ' | 'ʲ' | 'ˠ' | 'ˤ' | '\u{0329}' | '\u{032F}'
        )
}

/// Ties the char before it AND the char after it into one unit -- the affricate
/// / double-articulation bars (t͡s, d͡ʒ, k͡p). Excluded from `is_attach` so the
/// following base is pulled in rather than left as its own phoneme.
fn is_tie(c: char) -> bool {
    matches!(c, '\u{0361}' | '\u{035C}')
}

/// Split a continuous IPA reading into phoneme units.
///
/// The liaison tie `‿` stands alone, matching how WikiPron segments it, so that
/// a merged corpus stays consistent. The affricate tie bar `͡`, by contrast,
/// binds its two flanking symbols into one phoneme: `t͡s` is one token, not
/// `t͡` + `s`.
pub fn segment(ipa: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // Set when the previous char was an affricate tie bar: the next base joins
    // the same token as the second half of the affricate.
    let mut tie_pending = false;
    for c in ipa.chars() {
        // Some editions write length as an ASCII colon; fold it to the IPA mark
        // so it attaches to its vowel like any other length.
        let c = if c == ':' { 'ː' } else { c };
        if is_drop(c) {
            continue;
        }
        if tie_pending && out.last().is_some() {
            out.last_mut().unwrap().push(c);
            tie_pending = false;
        } else if c == '‿' {
            out.push(c.to_string());
        } else if is_tie(c) && out.last().is_some_and(|l| l != "‿") {
            out.last_mut().unwrap().push(c);
            tie_pending = true;
        } else if is_attach(c) && out.last().is_some_and(|l| l != "‿") {
            out.last_mut().unwrap().push(c);
        } else {
            out.push(c.to_string());
        }
    }
    out
}

/// Read `path` (plain or `.gz`), keep `lang` rows, write a WikiPron-style TSV.
///
/// Multi-word entries are skipped: the builder is word-level, and training it
/// on phrases teaches it to spell out what it cannot align.
pub fn convert(lang: &str, path: &str, out_path: &str) {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));

    // gzip via the system tool, for the same reason fetch shells out to curl.
    let mut child = None;
    let reader: Box<dyn BufRead> = if path.ends_with(".gz") {
        let mut c = Command::new("gzip")
            .args(["-dc"])
            .stdin(Stdio::from(file))
            .stdout(Stdio::piped())
            .spawn()
            .expect("gzip not found");
        let out = c.stdout.take().expect("gzip stdout");
        child = Some(c);
        Box::new(BufReader::new(out))
    } else {
        Box::new(BufReader::new(file))
    };

    // Insertion-ordered: keeps the output stable across runs.
    let mut words: Vec<(String, Vec<String>)> = Vec::new();
    let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let (mut lines, mut skipped_lang, mut skipped_multi, mut no_pron) = (0u64, 0u64, 0u64, 0u64);

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        lines += 1;
        let Some(e) = parse_line(&line) else { continue };
        if e.lang_code != lang {
            skipped_lang += 1;
            continue;
        }
        if e.word.contains(' ') || e.word.contains('\t') {
            skipped_multi += 1;
            continue;
        }
        // Take only the first phonemic reading. kaikki lists the citation form
        // first and inflected or "incorrect"-tagged variants after -- "beyaz" is
        // [/beˈjɑz/, /bejɑzˈlaɾ/], the second being the plural. Emitting every
        // reading put those extra forms in the TSV under the wrong headword; the
        // first is the one that belongs to the word.
        let Some(r) = phonemic(&e.ipas).into_iter().next() else {
            no_pron += 1;
            continue;
        };
        let segs = segment(r);
        if segs.is_empty() {
            no_pron += 1;
            continue;
        }
        let joined = segs.join(" ");
        match index.get(&e.word) {
            Some(&i) => {
                let v: &mut Vec<String> = &mut words[i].1;
                if !v.contains(&joined) {
                    v.push(joined);
                }
            }
            None => {
                index.insert(e.word.clone(), words.len());
                words.push((e.word.clone(), vec![joined]));
            }
        }
    }

    if let Some(mut c) = child {
        let _ = c.wait();
    }

    let mut f = File::create(out_path).unwrap_or_else(|e| panic!("create {out_path}: {e}"));
    let mut pairs = 0u64;
    for (w, ipas) in &words {
        for ipa in ipas {
            writeln!(f, "{w}\t{ipa}").expect("write tsv");
            pairs += 1;
        }
    }

    println!("read {lines} lines: {skipped_lang} other-language, {skipped_multi} multi-word, {no_pron} no phonemic reading");
    println!("wrote {out_path}: {} words, {pairs} pairs", words.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_word_lang_and_sounds() {
        let l = r#"{"word":"accueil","lang_code":"fr","pos":"noun","sounds":[{"ipa":"\\a.kœj\\"},{"ipa":"[ɛ̃.n‿a.kœj]","audio":"x.ogg"}]}"#;
        let e = parse_line(l).unwrap();
        assert_eq!(e.word, "accueil");
        assert_eq!(e.lang_code, "fr");
        assert_eq!(e.ipas.len(), 2);
    }

    #[test]
    fn phonemic_prefers_broad_over_narrow() {
        // fr backslash + a narrow bracket: broad wins, bracket ignored.
        let fr = vec!["\\a.kœj\\".to_string(), "[ɛ̃.n‿a.kœj]".to_string()];
        assert_eq!(phonemic(&fr), vec!["a.kœj"]);
        // slash editions (de/es/it): broad slash form is taken.
        let it = vec!["/ˈkaza/".to_string(), "/ˈkasa/".to_string()];
        assert_eq!(phonemic(&it), vec!["ˈkaza", "ˈkasa"]);
    }

    #[test]
    fn phonemic_falls_back_to_bracket_when_no_broad() {
        // German writes only narrow brackets; without a fallback we would drop
        // the whole edition.
        let de = vec!["[haˈloː]".to_string()];
        assert_eq!(phonemic(&de), vec!["haˈloː"]);
    }

    #[test]
    fn phonemic_falls_back_to_bare_readings() {
        // Greek writes readings with no delimiters at all.
        let el = vec!["ˈle.ksi".to_string()];
        assert_eq!(phonemic(&el), vec!["ˈle.ksi"]);
        // bare junk is still rejected; empty/no-letter strings too
        assert!(phonemic(&["e b o 4".to_string()]).is_empty());
        assert!(phonemic(&["ˈ.".to_string()]).is_empty());
        // a delimited reading still wins over a bare one
        let mixed = vec!["ˈle.ksi".to_string(), "/ˈleksi/".to_string()];
        assert_eq!(phonemic(&mixed), vec!["ˈleksi"]);
    }

    #[test]
    fn phonemic_rejects_ascii_approximations() {
        // pt carries junk like /e."sEx.tu/ next to real IPA; keep only the real.
        let pt = vec!["/e.\"sEx.tu/".to_string(), "/esˈeʁtu/".to_string()];
        assert_eq!(phonemic(&pt), vec!["esˈeʁtu"]);
    }

    #[test]
    fn phonemic_rejects_xsampa_placeholders_and_merged_forms() {
        // X-SAMPA (digit tap, @ schwa, bare ~), a `…`/`*` placeholder, and a
        // two-form field with an inner `/` are all rejected.
        assert!(phonemic(&["/e b o 4 e ~ s i/".to_string()]).is_empty());
        assert!(phonemic(&["\\ptit@\\".to_string()]).is_empty());
        assert!(phonemic(&["[…]".to_string()]).is_empty());
        assert!(phonemic(&["/a t e/ , /a t/".to_string()]).is_empty());
        // a clean form alongside junk still comes through
        let mixed = vec!["/e b o 4/".to_string(), "[a ɡ o]".to_string()];
        assert_eq!(phonemic(&mixed), vec!["a ɡ o"]);
    }

    #[test]
    fn segment_normalises_ascii_length_and_stress() {
        // ASCII ':' folds to ː and attaches; ASCII apostrophe stress is dropped.
        assert_eq!(segment("m a : v i"), ["m", "aː", "v", "i"]);
        assert_eq!(segment("' s a l t u"), ["s", "a", "l", "t", "u"]);
    }

    #[test]
    fn segment_keeps_affricate_tie_as_one_token() {
        // U+0361 binds both sides: t͡s is one phoneme, not t͡ + s.
        assert_eq!(segment("t\u{0361}sar"), ["t\u{0361}s", "a", "r"]);
        assert_eq!(segment("d\u{0361}ʒun"), ["d\u{0361}ʒ", "u", "n"]);
        // below-bar U+035C too
        assert_eq!(segment("t\u{035C}s"), ["t\u{035C}s"]);
        // a tie bar plus a following length mark still binds the base pair
        assert_eq!(segment("k\u{0361}p"), ["k\u{0361}p"]);
    }

    #[test]
    fn segment_splits_and_keeps_combining_with_base() {
        assert_eq!(segment("a.kœj"), ["a", "k", "œ", "j"]);
        // ɔ + combining tilde is one phoneme
        assert_eq!(segment("bɔ\u{0303}ʒuʁ"), ["b", "ɔ\u{0303}", "ʒ", "u", "ʁ"]);
        // stress and syllable marks are not phonemes
        assert_eq!(segment("ˈli.ʁ"), ["l", "i", "ʁ"]);
        // length rides with its vowel
        assert_eq!(segment("aːb"), ["aː", "b"]);
    }

    #[test]
    fn segment_liaison_tie_stands_alone() {
        assert_eq!(segment("ptit‿"), ["p", "t", "i", "t", "‿"]);
    }

    #[test]
    fn skips_nested_braces_inside_strings() {
        let l = r#"{"senses":[{"gloss":"a { brace } in text","x":[1,2]}],"word":"x","lang_code":"fr","sounds":[{"ipa":"\\a\\"}]}"#;
        let e = parse_line(l).unwrap();
        assert_eq!(e.word, "x");
        assert_eq!(phonemic(&e.ipas), vec!["a"]);
    }

    #[test]
    fn handles_escapes_and_unicode_escapes() {
        let l = r#"{"word":"café","lang_code":"fr","sounds":[{"ipa":"\\ka.fe\\"}]}"#;
        let e = parse_line(l).unwrap();
        assert_eq!(e.word, "café");
    }

    #[test]
    fn no_word_is_none() {
        assert!(parse_line(r#"{"lang_code":"fr"}"#).is_none());
        assert!(parse_line("not json").is_none());
    }

    #[test]
    fn ipa_key_outside_sounds_is_ignored() {
        let l = r#"{"forms":[{"ipa":"\\WRONG\\"}],"word":"x","lang_code":"fr","sounds":[{"ipa":"\\ʁɛ\\"}]}"#;
        let e = parse_line(l).unwrap();
        assert_eq!(phonemic(&e.ipas), vec!["ʁɛ"]);
    }
}
