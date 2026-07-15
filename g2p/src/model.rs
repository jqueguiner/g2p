//! The n-gram G2P model + its binary `.g2p` format.
//!
//! One source of truth for serialization: `xtask` writes via [`Model::to_bytes`],
//! the runtime reads via [`Model::from_bytes`]. Same struct both sides -> the
//! blob format cannot drift.
//!
//! N-gram probabilities are stored **fully interpolated** (Witten-Bell) per seen
//! gram, so runtime lookup is a clean backoff recursion with no interpolation
//! math: exact gram hit returns the stored logprob (lower orders already folded
//! in); a miss multiplies by the history backoff weight and drops the oldest
//! token.

use std::collections::HashMap;

use crate::normalize::graphemes;

/// A joint alignment token: (grapheme side, phoneme side).
/// Token id 0 is reserved for [`EOS`] and has empty sides.
pub type Tok = (Box<str>, Box<str>);

/// End-of-sequence token id. Appended during training so the model learns
/// word-final phonotactics; the decoder must take it to close a word.
pub const EOS: u32 = 0;

const MAGIC: &[u8; 4] = b"G2P2";
/// Fixed-point scale for quantized ln-probabilities (i16). 0.001 resolution is
/// far finer than argmax decoding needs.
const QSCALE: f32 = 1000.0;

pub struct Model {
    /// id -> token. index 0 = EOS ("", "").
    pub tokens: Vec<Tok>,
    /// n-gram order (e.g. 6).
    pub order: u8,
    /// logographic language: prefer the exact/per-char lexicon over the n-gram,
    /// which is unreliable on ideographs.
    pub logo: bool,
    /// max grapheme-cluster length of any token's grapheme side (derived).
    pub max_chunk: usize,
    /// grapheme chunk -> candidate token ids (derived, not serialized).
    pub by_graph: HashMap<Box<str>, Vec<u32>>,
    /// exact gram (history ++ token) -> ln p, fully interpolated.
    pub ngram: HashMap<Box<[u32]>, f32>,
    /// history -> ln backoff weight (1 - lambda).
    pub backoff: HashMap<Box<[u32]>, f32>,
    /// ln floor for an unseen unigram.
    pub unk: f32,
    /// exact word -> IPA (continuous). Primary tier for logographic languages
    /// (zh/ja/yue) and irregulars; checked before the n-gram in `phonemize`.
    /// Single-character entries double as the per-character OOV fallback.
    pub lexicon: HashMap<Box<str>, Box<str>>,
}

impl Model {
    /// Rebuild the derived `by_graph` index and `max_chunk` from `tokens`.
    /// Empty-grapheme specials (EOS) are skipped — they are never decode
    /// candidates mid-word.
    pub fn index(&mut self) {
        self.by_graph.clear();
        self.max_chunk = 1;
        for (id, (g, _)) in self.tokens.iter().enumerate() {
            if g.is_empty() {
                continue;
            }
            self.max_chunk = self.max_chunk.max(graphemes(g).len());
            self.by_graph.entry(g.clone()).or_default().push(id as u32);
        }
    }

    /// `ln p(t | hist)`. `hist` is the last `order-1` token ids, most-recent
    /// last. Interpolation is baked into stored grams, so: exact gram hit
    /// returns it; otherwise back off by the history weight and drop the oldest
    /// token.
    pub fn logp(&self, hist: &[u32], t: u32) -> f32 {
        let mut key: Vec<u32> = Vec::with_capacity(hist.len() + 1);
        key.extend_from_slice(hist);
        key.push(t);
        if let Some(&lp) = self.ngram.get(key.as_slice()) {
            return lp;
        }
        if hist.is_empty() {
            return self.unk; // unseen unigram
        }
        let bow = self.backoff.get(hist).copied().unwrap_or(0.0); // ln(1) if history unseen
        bow + self.logp(&hist[1..], t)
    }

    // ---- binary format ----

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(MAGIC);
        o.push(self.order);
        o.push(self.logo as u8);
        o.extend_from_slice(&[0, 0]); // pad
        put_u32(&mut o, self.tokens.len() as u32);
        for (g, p) in &self.tokens {
            put_str(&mut o, g);
            put_str(&mut o, p);
        }
        put_map(&mut o, &self.ngram);
        put_map(&mut o, &self.backoff);
        o.extend_from_slice(&self.unk.to_le_bytes());
        put_u32(&mut o, self.lexicon.len() as u32);
        for (k, v) in &self.lexicon {
            put_str(&mut o, k);
            put_str(&mut o, v);
        }
        o
    }

    pub fn from_bytes(b: &[u8]) -> Model {
        let mut c = Cur { b, i: 0 };
        assert_eq!(c.take(4), MAGIC, "bad magic");
        let order = c.u8();
        let logo = c.u8() != 0;
        c.take(2); // pad
        let nt = c.u32() as usize;
        let tokens: Vec<Tok> = (0..nt).map(|_| (c.str(), c.str())).collect();
        let ngram = c.map();
        let backoff = c.map();
        let unk = c.f32();
        let nlex = c.u32() as usize;
        let mut lexicon = HashMap::with_capacity(nlex);
        for _ in 0..nlex {
            lexicon.insert(c.str(), c.str());
        }
        let mut m = Model {
            tokens,
            order,
            logo,
            max_chunk: 0,
            by_graph: HashMap::new(),
            ngram,
            backoff,
            unk,
            lexicon,
        };
        m.index();
        m
    }
}

// ---- little-endian helpers, no deps ----

fn put_u16(o: &mut Vec<u8>, v: u16) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn put_str(o: &mut Vec<u8>, s: &str) {
    put_u16(o, s.len() as u16);
    o.extend_from_slice(s.as_bytes());
}
/// LEB128 unsigned varint — token ids are small, so most take 1-2 bytes.
fn put_varint(o: &mut Vec<u8>, mut v: u32) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            o.push(b | 0x80);
        } else {
            o.push(b);
            break;
        }
    }
}
fn put_map(o: &mut Vec<u8>, m: &HashMap<Box<[u32]>, f32>) {
    put_u32(o, m.len() as u32);
    for (k, v) in m {
        o.push(k.len() as u8);
        for &id in k.iter() {
            put_varint(o, id);
        }
        let q = (v * QSCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        o.extend_from_slice(&q.to_le_bytes());
    }
}

struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}
impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        s
    }
    fn u8(&mut self) -> u8 {
        let v = self.b[self.i];
        self.i += 1;
        v
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take(2).try_into().unwrap())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn i16(&mut self) -> i16 {
        i16::from_le_bytes(self.take(2).try_into().unwrap())
    }
    fn varint(&mut self) -> u32 {
        let mut v = 0u32;
        let mut s = 0;
        loop {
            let b = self.u8();
            v |= ((b & 0x7f) as u32) << s;
            if b & 0x80 == 0 {
                break;
            }
            s += 7;
        }
        v
    }
    fn str(&mut self) -> Box<str> {
        let n = self.u16() as usize;
        String::from_utf8(self.take(n).to_vec()).unwrap().into()
    }
    fn map(&mut self) -> HashMap<Box<[u32]>, f32> {
        let n = self.u32() as usize;
        let mut m = HashMap::with_capacity(n);
        for _ in 0..n {
            let l = self.u8() as usize;
            let k: Box<[u32]> = (0..l).map(|_| self.varint()).collect();
            m.insert(k, self.i16() as f32 / QSCALE);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_preserves_lexicon_and_logo() {
        let mut m = Model {
            tokens: vec![
                (String::new().into(), String::new().into()),
                ("a".into(), "ɑ".into()),
            ],
            order: 6,
            logo: true,
            max_chunk: 0,
            by_graph: HashMap::new(),
            ngram: HashMap::from([(Box::from(&[1u32][..]), -1.5f32)]),
            backoff: HashMap::from([(Box::from(&[][..]), -0.3f32)]),
            unk: -9.0,
            lexicon: HashMap::from([("中".into(), "ʈʂʊŋ".into())]),
        };
        m.index();
        let back = Model::from_bytes(&m.to_bytes());
        assert!(back.logo);
        assert_eq!(back.order, 6);
        assert_eq!(back.lexicon.get("中").map(|s| &**s), Some("ʈʂʊŋ"));
        assert_eq!(back.ngram.get(&[1u32][..]).copied(), Some(-1.5));
        assert_eq!(back.by_graph.get("a").map(|s| s.len()), Some(1));
        // quantized logprob round-trips within i16 resolution
        assert!((back.unk - (-9.0)).abs() < 0.01);
    }

    #[test]
    fn logp_backoff_and_unk() {
        let m = Model {
            tokens: vec![
                (String::new().into(), String::new().into()),
                ("a".into(), "ɑ".into()),
            ],
            order: 3,
            logo: false,
            max_chunk: 0,
            by_graph: HashMap::new(),
            ngram: HashMap::from([(Box::from(&[1u32][..]), -1.5f32)]), // only unigram [1]
            backoff: HashMap::from([(Box::from(&[2u32][..]), -0.7f32)]),
            unk: -8.0,
            lexicon: HashMap::new(),
        };
        // exact unigram hit
        assert!((m.logp(&[], 1) - (-1.5)).abs() < 1e-6);
        // bigram [2,1] missing -> backoff(hist=[2]) + logp([],1) = -0.7 + -1.5
        assert!((m.logp(&[2], 1) - (-2.2)).abs() < 1e-6);
        // unseen unigram -> unk
        assert!((m.logp(&[], 99) - (-8.0)).abs() < 1e-6);
        // missing history backoff weight treated as ln(1)=0
        assert!((m.logp(&[7], 1) - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn varint_multibyte_roundtrip() {
        // token id > 127 forces a multi-byte LEB128 varint on both encode/decode
        let mut tokens: Vec<Tok> = (0..300)
            .map(|i| (i.to_string().into(), format!("p{i}").into()))
            .collect();
        tokens[0] = (String::new().into(), String::new().into());
        let m = Model {
            tokens,
            order: 2,
            logo: false,
            max_chunk: 0,
            by_graph: HashMap::new(),
            ngram: HashMap::from([(Box::from(&[257u32][..]), -2.0f32)]),
            backoff: HashMap::new(),
            unk: -7.0,
            lexicon: HashMap::new(),
        };
        let back = Model::from_bytes(&m.to_bytes());
        assert!((back.ngram.get(&[257u32][..]).copied().unwrap() - (-2.0)).abs() < 0.01);
    }
}
