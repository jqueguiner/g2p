//! Grapheme splitting + casing. Pure `std`, no Unicode-table crates.
//!
//! `graphemes` clusters a base char with its trailing combining marks so that
//! e.g. a base + combining tilde stays one unit. The combining-mark ranges
//! below cover Latin/Cyrillic/Greek/Arabic/Hebrew diacritics and common
//! symbol/Indic combining blocks. It is an approximation of full UAX #29
//! grapheme clustering — good enough for word-level G2P; extend the ranges if a
//! target language needs finer clustering.

/// Split a string into grapheme clusters (base char + trailing combining marks).
pub fn graphemes(s: &str) -> Vec<Box<str>> {
    let mut out: Vec<Box<str>> = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if is_combining(ch) && !cur.is_empty() {
            cur.push(ch); // attach to current base
        } else {
            if !cur.is_empty() {
                out.push(cur.as_str().into());
                cur.clear();
            }
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        out.push(cur.as_str().into());
    }
    out
}

/// Unicode-aware lowercasing (`std`). Note: Turkish dotless-i and a few
/// language-specific casings are not special-cased here.
pub fn lower(s: &str) -> String {
    s.to_lowercase()
}

#[inline]
fn is_combining(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F | // combining diacritical marks
        0x0483..=0x0489 | // Cyrillic
        0x0591..=0x05BD | 0x05BF | 0x05C1..=0x05C2 | 0x05C4..=0x05C5 | 0x05C7 | // Hebrew
        0x0610..=0x061A | 0x064B..=0x065F | 0x0670 | 0x06D6..=0x06DC | 0x06DF..=0x06E4 | // Arabic
        0x0900..=0x0903 | 0x093A..=0x094F | 0x0951..=0x0957 | // Devanagari (approx)
        0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_splits_per_char() {
        let g = graphemes("chat");
        assert_eq!(g.len(), 4);
    }

    #[test]
    fn combining_mark_attaches() {
        // "a" + U+0303 combining tilde -> one cluster
        let g = graphemes("a\u{0303}bc");
        assert_eq!(g.len(), 3);
        assert_eq!(&*g[0], "a\u{0303}");
    }

    #[test]
    fn empty_string_no_clusters() {
        assert!(graphemes("").is_empty());
    }

    #[test]
    fn leading_combining_stands_alone() {
        // combining mark with no preceding base starts its own cluster
        let g = graphemes("\u{0303}a");
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn lower_is_unicode_aware() {
        assert_eq!(lower("ÀÉ"), "àé");
    }
}
