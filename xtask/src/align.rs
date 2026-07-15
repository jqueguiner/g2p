//! Many-to-many grapheme<->phoneme alignment via forward-backward EM,
//! then Viterbi to emit joint-token sequences for n-gram training.
//! Linear-space f64 (words are short). No external deps.

use std::collections::HashMap;

const GMAX: usize = 2; // grapheme chunk 1..=2
const PMAX: usize = 2; // phoneme chunk 0..=2 (0 = silent grapheme)
const FLOOR: f64 = 1e-9;

pub type Unit = Box<str>;
/// A joint alignment token: (graphemes, phonemes). "" phonemes = deletion.
pub type JTok = (String, String);

/// One training row.
pub struct Row {
    pub g: Vec<Unit>,
    pub p: Vec<Unit>,
    pub w: f64, // gold 1.0, silver 0.15..0.3
}

pub struct Aligner {
    prob: HashMap<JTok, f64>,
}

impl Aligner {
    /// Uniform init over every chunk pair that actually co-occurs.
    pub fn new(rows: &[Row]) -> Self {
        let mut seen: HashMap<JTok, f64> = HashMap::new();
        for r in rows {
            let (n, m) = (r.g.len(), r.p.len());
            for i in 0..n {
                for a in 1..=GMAX.min(n - i) {
                    let gc = r.g[i..i + a].concat();
                    for j in 0..=m {
                        for b in 0..=PMAX.min(m - j) {
                            let pc = if b == 0 {
                                String::new()
                            } else {
                                r.p[j..j + b].concat()
                            };
                            seen.entry((gc.clone(), pc)).or_insert(1.0);
                        }
                    }
                }
            }
        }
        let z: f64 = seen.values().sum();
        if z > 0.0 {
            for v in seen.values_mut() {
                *v /= z;
            }
        }
        Aligner { prob: seen }
    }

    #[inline]
    fn tok(&self, g: &[Unit], p: &[Unit], i: usize, a: usize, j: usize, b: usize) -> JTok {
        let gc = g[i..i + a].concat();
        let pc = if b == 0 {
            String::new()
        } else {
            p[j..j + b].concat()
        };
        (gc, pc)
    }

    #[inline]
    fn pr(&self, t: &JTok) -> f64 {
        *self.prob.get(t).unwrap_or(&FLOOR)
    }

    fn forward(&self, g: &[Unit], p: &[Unit]) -> Vec<Vec<f64>> {
        let (n, m) = (g.len(), p.len());
        let mut a = vec![vec![0.0; m + 1]; n + 1];
        a[0][0] = 1.0;
        for i in 0..=n {
            for j in 0..=m {
                if i == 0 && j == 0 {
                    continue;
                }
                let mut s = 0.0;
                for da in 1..=GMAX.min(i) {
                    for db in 0..=PMAX.min(j) {
                        let t = self.tok(g, p, i - da, da, j - db, db);
                        s += a[i - da][j - db] * self.pr(&t);
                    }
                }
                a[i][j] = s;
            }
        }
        a
    }

    fn backward(&self, g: &[Unit], p: &[Unit]) -> Vec<Vec<f64>> {
        let (n, m) = (g.len(), p.len());
        let mut b = vec![vec![0.0; m + 1]; n + 1];
        b[n][m] = 1.0;
        for i in (0..=n).rev() {
            for j in (0..=m).rev() {
                if i == n && j == m {
                    continue;
                }
                let mut s = 0.0;
                for da in 1..=GMAX {
                    if i + da > n {
                        break;
                    }
                    for db in 0..=PMAX {
                        if j + db > m {
                            break;
                        }
                        let t = self.tok(g, p, i, da, j, db);
                        s += self.pr(&t) * b[i + da][j + db];
                    }
                }
                b[i][j] = s;
            }
        }
        b
    }

    /// Expected joint-token counts over a slice of rows (one worker's share).
    fn e_step_chunk(&self, rows: &[Row]) -> HashMap<JTok, f64> {
        let mut cnt: HashMap<JTok, f64> = HashMap::new();
        for r in rows {
            let (n, m) = (r.g.len(), r.p.len());
            if n == 0 {
                continue;
            }
            let a = self.forward(&r.g, &r.p);
            let b = self.backward(&r.g, &r.p);
            let z = a[n][m];
            if z <= 0.0 {
                continue; // unreachable pair (e.g. m too long for GMAX/PMAX)
            }
            for i in 0..=n {
                for j in 0..=m {
                    for da in 1..=GMAX.min(i) {
                        for db in 0..=PMAX.min(j) {
                            let t = self.tok(&r.g, &r.p, i - da, da, j - db, db);
                            let post = a[i - da][j - db] * self.pr(&t) * b[i][j] / z;
                            *cnt.entry(t).or_default() += post * r.w;
                        }
                    }
                }
            }
        }
        cnt
    }

    /// Parallel E-step: split rows across worker threads, merge partial counts.
    fn e_step(&self, rows: &[Row]) -> HashMap<JTok, f64> {
        let nthreads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let chunk = (rows.len() / nthreads).max(1);
        let mut partials: Vec<HashMap<JTok, f64>> = Vec::new();
        std::thread::scope(|s| {
            let handles: Vec<_> = rows
                .chunks(chunk)
                .map(|ch| s.spawn(move || self.e_step_chunk(ch)))
                .collect();
            for h in handles {
                partials.push(h.join().unwrap());
            }
        });
        let mut cnt: HashMap<JTok, f64> = HashMap::new();
        for p in partials {
            for (k, v) in p {
                *cnt.entry(k).or_default() += v;
            }
        }
        cnt
    }

    /// Viterbi-align every row in parallel -> (joint-token seq, weight).
    pub fn viterbi_all(&self, rows: &[Row]) -> Vec<(Vec<JTok>, f64)> {
        let nthreads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let chunk = (rows.len() / nthreads).max(1);
        let mut out: Vec<(Vec<JTok>, f64)> = Vec::with_capacity(rows.len());
        std::thread::scope(|s| {
            let handles: Vec<_> = rows
                .chunks(chunk)
                .map(|ch| {
                    s.spawn(move || {
                        ch.iter()
                            .map(|r| (self.viterbi(&r.g, &r.p), r.w))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for h in handles {
                out.extend(h.join().unwrap());
            }
        });
        out
    }

    pub fn em(&mut self, rows: &[Row], iters: usize) {
        for _ in 0..iters {
            let cnt = self.e_step(rows);
            let total: f64 = cnt.values().sum();
            if total <= 0.0 {
                break;
            }
            self.prob = cnt.into_iter().map(|(k, v)| (k, v / total)).collect();
        }
    }

    /// Best joint-token sequence for one pair (Viterbi in log space).
    pub fn viterbi(&self, g: &[Unit], p: &[Unit]) -> Vec<JTok> {
        let (n, m) = (g.len(), p.len());
        let mut best = vec![vec![f64::NEG_INFINITY; m + 1]; n + 1];
        let mut bp: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; m + 1]; n + 1];
        best[0][0] = 0.0;
        for i in 0..=n {
            for j in 0..=m {
                if best[i][j] == f64::NEG_INFINITY {
                    continue;
                }
                for da in 1..=GMAX {
                    if i + da > n {
                        break;
                    }
                    for db in 0..=PMAX {
                        if j + db > m {
                            break;
                        }
                        let t = self.tok(g, p, i, da, j, db);
                        let s = best[i][j] + self.pr(&t).max(FLOOR).ln();
                        if s > best[i + da][j + db] {
                            best[i + da][j + db] = s;
                            bp[i + da][j + db] = Some((i, j));
                        }
                    }
                }
            }
        }
        let (mut i, mut j) = (n, m);
        let mut out = Vec::new();
        while let Some((pi, pj)) = bp[i][j] {
            out.push(self.tok(g, p, pi, i - pi, pj, j - pj));
            i = pi;
            j = pj;
        }
        out.reverse();
        out
    }
}
