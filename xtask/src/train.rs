//! Weighted Witten-Bell interpolated joint n-gram trainer.
//! Consumes Viterbi joint-token sequences (+ row weights), produces a `Model`
//! with fully-interpolated per-gram logprobs + per-history backoff weights.

use std::collections::HashMap;

use g2p::model::{Model, EOS};

type J = (String, String);

pub fn train_ngram(seqs: &[(Vec<J>, f64)], order: usize) -> Model {
    // 1. token vocab, id 0 reserved for EOS
    let mut vocab: HashMap<J, u32> = HashMap::new();
    let mut tokens: Vec<(Box<str>, Box<str>)> = vec![(String::new().into(), String::new().into())];
    vocab.insert((String::new(), String::new()), EOS);

    let intern =
        |t: &J, vocab: &mut HashMap<J, u32>, tokens: &mut Vec<(Box<str>, Box<str>)>| -> u32 {
            if let Some(&i) = vocab.get(t) {
                i
            } else {
                let i = tokens.len() as u32;
                tokens.push((t.0.as_str().into(), t.1.as_str().into()));
                vocab.insert(t.clone(), i);
                i
            }
        };

    // 2. id sequences, append EOS
    let idseqs: Vec<(Vec<u32>, f64)> = seqs
        .iter()
        .map(|(s, w)| {
            let mut v: Vec<u32> = s
                .iter()
                .map(|t| intern(t, &mut vocab, &mut tokens))
                .collect();
            v.push(EOS);
            (v, *w)
        })
        .collect();

    // 3. weighted counts, every order 1..=order
    let mut cnt: HashMap<Box<[u32]>, f64> = HashMap::new();
    for (v, w) in &idseqs {
        for i in 0..v.len() {
            for n in 1..=order.min(i + 1) {
                *cnt.entry(v[i + 1 - n..=i].into()).or_default() += *w;
            }
        }
    }

    // 4. history stats: h -> (c(h), N1+(h.))
    let mut hc: HashMap<Box<[u32]>, (f64, u32)> = HashMap::new();
    for (g, c) in &cnt {
        let e = hc.entry(g[..g.len() - 1].into()).or_insert((0.0, 0));
        e.0 += *c;
        e.1 += 1;
    }
    let vsize = tokens.len() as f64;
    let empty: Box<[u32]> = Box::from(&[][..]);
    let (c0, n10) = *hc.get(&empty).expect("empty history must exist");

    // 5. interpolated prob, memoized
    fn prob(
        seq: &[u32],
        cnt: &HashMap<Box<[u32]>, f64>,
        hc: &HashMap<Box<[u32]>, (f64, u32)>,
        vsize: f64,
        memo: &mut HashMap<Box<[u32]>, f64>,
    ) -> f64 {
        if let Some(&p) = memo.get(seq) {
            return p;
        }
        let (c, n1) = *hc.get(&seq[..seq.len() - 1]).unwrap();
        let lambda = c / (c + n1 as f64);
        let ml = cnt.get(seq).copied().unwrap_or(0.0) / c;
        let lower = if seq.len() == 1 {
            1.0 / vsize
        } else {
            prob(&seq[1..], cnt, hc, vsize, memo)
        };
        let p = lambda * ml + (1.0 - lambda) * lower;
        memo.insert(seq.into(), p);
        p
    }

    // 6. emit ln-probs + backoff weights
    let mut ngram = HashMap::new();
    let mut memo = HashMap::new();
    for g in cnt.keys() {
        ngram.insert(g.clone(), prob(g, &cnt, &hc, vsize, &mut memo).ln() as f32);
    }
    let mut backoff = HashMap::new();
    for (h, (c, n1)) in &hc {
        backoff.insert(h.clone(), (*n1 as f64 / (*c + *n1 as f64)).ln() as f32);
    }
    let unk = ((n10 as f64 / (c0 + n10 as f64)) * (1.0 / vsize)).ln() as f32;

    let mut m = Model {
        tokens,
        order: order as u8,
        logo: false,
        max_chunk: 0,
        by_graph: HashMap::new(),
        ngram,
        backoff,
        unk,
        lexicon: HashMap::new(),
    };
    m.index();
    m
}
