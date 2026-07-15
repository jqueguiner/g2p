//! Exact-match lexicon tier. Checked before the n-gram decoder for irregular
//! words, and used as the primary engine for logographic languages
//! (zh/ja/ko/...) where grapheme->phoneme n-grams do not apply.

use std::collections::HashMap;

#[derive(Default)]
pub struct Lexicon {
    map: HashMap<Box<str>, Box<str>>,
}

impl Lexicon {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, word: &str, ipa: &str) {
        self.map.insert(word.into(), ipa.into());
    }

    /// Exact lookup. Returns `None` on miss (caller falls through to n-gram).
    pub fn get(&self, word: &str) -> Option<&str> {
        self.map.get(word).map(|b| &**b)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_miss() {
        let mut lex = Lexicon::new();
        assert!(lex.is_empty());
        lex.insert("cat", "k a t");
        assert_eq!(lex.len(), 1);
        assert!(!lex.is_empty());
        assert_eq!(lex.get("cat"), Some("k a t"));
        assert_eq!(lex.get("dog"), None);
    }

    #[test]
    fn default_is_empty() {
        let lex = Lexicon::default();
        assert!(lex.is_empty());
        assert_eq!(lex.len(), 0);
    }
}
