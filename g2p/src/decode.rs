//! Beam-search decoder over the joint n-gram model.
//!
//! Walk the input grapheme by grapheme; at each position try every grapheme
//! chunk of length `1..=max_chunk`, expand each candidate joint token, score
//! with `logp`, keep the top-`BEAM` hypotheses. Close with an EOS transition.

use crate::model::{Model, EOS};
use crate::normalize::{graphemes, lower};

const BEAM: usize = 8;
/// Penalty for skipping an unknown grapheme (no chunk matched at that position).
const SKIP_PEN: f32 = -20.0;

struct Hyp {
    pos: usize,
    hist: Vec<u32>,
    out: Vec<u32>,
    score: f32,
}

impl Hyp {
    fn dup(&self) -> Hyp {
        Hyp {
            pos: self.pos,
            hist: self.hist.clone(),
            out: self.out.clone(),
            score: self.score,
        }
    }
}

/// Phonemize a single word.
///
/// Tiers: exact lexicon (word, then lowercased) -> n-gram beam ->
/// per-character lexicon fallback (logographic OOV).
pub fn phonemize(m: &Model, word: &str) -> String {
    // 1. exact lexicon (logographic words / irregulars)
    if let Some(ipa) = m.lexicon.get(word) {
        return ipa.to_string();
    }
    let lw = lower(word);
    if lw != word {
        if let Some(ipa) = m.lexicon.get(lw.as_str()) {
            return ipa.to_string();
        }
    }
    // Logographic: per-character lexicon before the (unreliable) n-gram.
    // Alphabetic: n-gram before the char fallback.
    if m.logo {
        let r = char_fallback(m, word);
        if !r.is_empty() {
            return r;
        }
        if !m.ngram.is_empty() {
            return beam(m, &lw);
        }
        String::new()
    } else {
        if !m.ngram.is_empty() {
            let r = beam(m, &lw);
            if !r.is_empty() {
                return r;
            }
        }
        char_fallback(m, word)
    }
}

/// Concatenate per-grapheme lexicon readings (logographic OOV). Empty if none hit.
fn char_fallback(m: &Model, word: &str) -> String {
    let mut out = String::new();
    let mut any = false;
    for ch in graphemes(word) {
        if let Some(ipa) = m.lexicon.get(&*ch) {
            out.push_str(ipa);
            any = true;
        }
    }
    if any {
        out
    } else {
        String::new()
    }
}

fn beam(m: &Model, lw: &str) -> String {
    let g = graphemes(lw);
    let n = g.len();
    if n == 0 {
        return String::new();
    }
    let ord = (m.order as usize).max(1);

    let mut beam = vec![Hyp {
        pos: 0,
        hist: Vec::new(),
        out: Vec::new(),
        score: 0.0,
    }];

    while !beam.iter().all(|h| h.pos == n) {
        let mut next: Vec<Hyp> = Vec::new();
        for h in &beam {
            if h.pos == n {
                next.push(h.dup());
                continue;
            }
            let maxk = m.max_chunk.min(n - h.pos);
            let mut matched = false;
            for k in 1..=maxk {
                let chunk: String = g[h.pos..h.pos + k].concat();
                if let Some(ids) = m.by_graph.get(chunk.as_str()) {
                    matched = true;
                    for &tok in ids {
                        let lp = m.logp(&h.hist, tok);
                        let mut hist = h.hist.clone();
                        hist.push(tok);
                        if ord > 1 && hist.len() > ord - 1 {
                            let d = hist.len() - (ord - 1);
                            hist.drain(0..d);
                        } else if ord == 1 {
                            hist.clear();
                        }
                        let mut out = h.out.clone();
                        out.push(tok);
                        next.push(Hyp {
                            pos: h.pos + k,
                            hist,
                            out,
                            score: h.score + lp,
                        });
                    }
                }
            }
            if !matched {
                // Unknown grapheme: skip one unit with a penalty, emit nothing.
                next.push(Hyp {
                    pos: h.pos + 1,
                    hist: h.hist.clone(),
                    out: h.out.clone(),
                    score: h.score + SKIP_PEN,
                });
            }
        }
        if next.is_empty() {
            break;
        }
        next.sort_by(|a, b| b.score.total_cmp(&a.score));
        next.truncate(BEAM);
        beam = next;
    }

    // Close each finished hypothesis with an EOS transition, pick the best.
    let best = beam
        .iter()
        .filter(|h| h.pos == n)
        .map(|h| (h.score + m.logp(&h.hist, EOS), h))
        .max_by(|a, b| a.0.total_cmp(&b.0));

    match best {
        Some((_, h)) => h.out.iter().map(|&t| &*m.tokens[t as usize].1).collect(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Toy model: graphemes a->ɑ, b->b, with a unigram n-gram so the beam runs.
    fn toy(logo: bool) -> Model {
        let mut m = Model {
            tokens: vec![
                (String::new().into(), String::new().into()), // EOS
                ("a".into(), "ɑ".into()),
                ("b".into(), "b".into()),
            ],
            order: 2,
            logo,
            max_chunk: 0,
            by_graph: HashMap::new(),
            ngram: HashMap::from([
                (Box::from(&[0u32][..]), -0.5f32),
                (Box::from(&[1u32][..]), -0.5f32),
                (Box::from(&[2u32][..]), -0.5f32),
            ]),
            backoff: HashMap::new(),
            unk: -5.0,
            lexicon: HashMap::new(),
        };
        m.index();
        m
    }

    #[test]
    fn empty_word_empty_output() {
        assert_eq!(phonemize(&toy(false), ""), "");
    }

    #[test]
    fn beam_decodes_known_graphemes() {
        assert_eq!(phonemize(&toy(false), "ab"), "ɑb");
    }

    #[test]
    fn uppercase_lowercased_before_decode() {
        assert_eq!(phonemize(&toy(false), "AB"), "ɑb");
    }

    #[test]
    fn unknown_grapheme_skipped_to_empty() {
        // 'z' matches nothing -> skipped -> no output
        assert_eq!(phonemize(&toy(false), "z"), "");
    }

    #[test]
    fn exact_lexicon_wins() {
        let mut m = toy(false);
        m.lexicon.insert("ab".into(), "EXACT".into());
        assert_eq!(phonemize(&m, "ab"), "EXACT");
    }

    #[test]
    fn lexicon_lowercased_lookup() {
        let mut m = toy(false);
        m.lexicon.insert("cat".into(), "kat".into());
        assert_eq!(phonemize(&m, "CAT"), "kat");
    }

    #[test]
    fn logographic_char_fallback() {
        let mut m = toy(true);
        m.lexicon.insert("中".into(), "tʂʊŋ".into());
        m.lexicon.insert("国".into(), "kwɔ".into());
        // whole word absent -> per-char fallback concatenates
        assert_eq!(phonemize(&m, "中国"), "tʂʊŋkwɔ");
    }

    #[test]
    fn logographic_falls_through_to_ngram() {
        // logo model, char fallback empty (a/b not in lexicon) -> n-gram beam
        assert_eq!(phonemize(&toy(true), "ab"), "ɑb");
    }

    #[test]
    fn no_ngram_no_lexicon_is_empty() {
        let mut m = toy(false);
        m.ngram.clear();
        assert_eq!(phonemize(&m, "ab"), "");
    }

    #[test]
    fn logo_all_miss_is_empty() {
        let mut m = toy(true);
        m.ngram.clear();
        assert_eq!(phonemize(&m, "z"), "");
    }

    #[test]
    fn multi_grapheme_token_and_dup() {
        // add a 2-grapheme token so one hypothesis finishes before another,
        // exercising the carry-forward (dup) path.
        let mut m = toy(false);
        m.tokens.push(("ab".into(), "AB".into()));
        m.ngram.insert(Box::from(&[3u32][..]), -0.4);
        m.index();
        // both "ɑb" (a+b) and "AB" (ab) are valid; decoder returns one of them
        let out = phonemize(&m, "ab");
        assert!(out == "AB" || out == "ɑb");
    }

    #[test]
    fn order_one_model_clears_history() {
        let mut m = toy(false);
        m.order = 1;
        assert_eq!(phonemize(&m, "ab"), "ɑb");
    }
}
