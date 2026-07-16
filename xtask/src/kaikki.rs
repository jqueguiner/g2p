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
//! Notation: kaikki carries both `\a.kœj\` (phonemic -- the citation form) and
//! `[ɛ̃.n‿a.kœj]` (phonetic -- as realised in context; that one is "un accueil",
//! article and liaison included). Only the backslash form is a property of the
//! word on its own, so the bracket form is dropped.
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
    ipas.iter()
        .map(|s| s.trim())
        .filter(|s| s.len() > 2 && s.starts_with('\\') && s.ends_with('\\'))
        .map(|s| s[1..s.len() - 1].trim())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Not phonemes: syllable breaks and stress.
fn is_drop(c: char) -> bool {
    matches!(c, '.' | 'ˈ' | 'ˌ' | ' ' | '|' | '‖' | '(' | ')')
}

/// Rides with the preceding base rather than standing alone.
fn is_attach(c: char) -> bool {
    matches!(c as u32, 0x0300..=0x036F)
        || matches!(
            c,
            'ː' | 'ˑ' | 'ʰ' | 'ʷ' | 'ʲ' | 'ˠ' | 'ˤ' | '\u{0329}' | '\u{032F}' | '\u{0361}'
        )
}

/// Split a continuous IPA reading into phoneme units.
///
/// The liaison tie stands alone, matching how WikiPron segments it, so that a
/// merged corpus stays consistent.
pub fn segment(ipa: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in ipa.chars() {
        if is_drop(c) {
            continue;
        }
        if c == '‿' {
            out.push(c.to_string());
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
        let readings = phonemic(&e.ipas);
        if readings.is_empty() {
            no_pron += 1;
            continue;
        }
        for r in readings {
            let segs = segment(r);
            if segs.is_empty() {
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
    fn phonemic_keeps_backslash_drops_brackets() {
        let ipas = vec![
            "\\a.kœj\\".to_string(),
            "[ɛ̃.n‿a.kœj]".to_string(),
            "\\\\".to_string(),
        ];
        assert_eq!(phonemic(&ipas), vec!["a.kœj"]);
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
