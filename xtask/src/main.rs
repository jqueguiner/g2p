//! Build tool. For now: build a single language from a local WikiPron-style TSV
//! (word <TAB> space-separated IPA), and phonemize a word from a compiled blob.
//!
//!   cargo run -p xtask -- build fr xtask/data/fr.sample.tsv
//!   cargo run -p xtask -- say   fr bonjour

mod align;
mod bench;
mod fetch;
mod silver;
mod train;

use std::collections::HashMap;
use std::env;
use std::fs;

use align::{Aligner, Row};
use g2p::normalize::graphemes;
use g2p::{phonemize, Model};

fn segs(p: &str) -> Vec<Box<str>> {
    p.split_whitespace().map(|s| s.into()).collect()
}

/// OpenCC traditional->simplified per-character map (first value). Embedded.
fn load_t2s() -> std::collections::HashMap<char, char> {
    include_str!("../data/ts_chars.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('\t'))
        .filter_map(|(k, v)| {
            let t = k.trim().chars().next()?;
            let s = v.split_whitespace().next()?.chars().next()?;
            Some((t, s))
        })
        .collect()
}

fn load_tsv(path: &str) -> Vec<(String, String)> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    text.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter_map(|l| l.split_once('\t'))
        .map(|(w, p)| (w.trim().to_string(), p.trim().to_string()))
        .filter(|(w, p)| !w.is_empty() && !p.is_empty())
        .collect()
}

/// Cap on training pairs per language (strided downsample) to bound EM cost.
/// Override with the MAX_PAIRS env var.
fn max_pairs() -> usize {
    env::var("MAX_PAIRS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40_000)
}

/// Load a TSV into weighted rows (strided-capped). Missing file -> empty.
fn rows_from(path: &str, weight: f64, cap: usize) -> Vec<Row> {
    if !std::path::Path::new(path).exists() {
        return Vec::new();
    }
    let mut pairs = load_tsv(path);
    if pairs.len() > cap {
        let step = (pairs.len() / cap).max(1);
        pairs = pairs.into_iter().step_by(step).take(cap).collect();
    }
    pairs
        .iter()
        .map(|(w, p)| Row {
            g: graphemes(&w.to_lowercase()),
            p: segs(p),
            w: weight,
        })
        .filter(|r| !r.g.is_empty() && !r.p.is_empty())
        .collect()
}

fn build(lang: &str, gold_tsv: &str) {
    let cap = max_pairs();
    let mut rows = rows_from(gold_tsv, 1.0, cap); // gold: full weight
    let ng = rows.len();
    let sv = rows_from(&format!("data/silver/{lang}.tsv"), 0.2, cap); // silver: low weight
    let ns = sv.len();
    rows.extend(sv);
    if rows.is_empty() {
        eprintln!("{lang}: no data (gold+silver empty), skip");
        return;
    }
    println!("gold {ng} + silver {ns} = {} rows (cap {cap})", rows.len());

    let mut al = Aligner::new(&rows);
    al.em(&rows, 10);
    let seqs = al.viterbi_all(&rows);
    println!("EM + align done");

    let mut model = train::train_ngram(&seqs, 6);

    // Logographic languages (zh/ja/yue): the hani TSV is a word->IPA lexicon.
    // Load it as the exact-match tier; single-char entries serve OOV fallback.
    if g2p::lang::by_whisper(lang).is_some_and(|l| l.logo) {
        model.logo = true;
        for (w, p) in load_tsv(gold_tsv).iter().take(200_000) {
            let ipa: String = p.split_whitespace().collect(); // segments -> continuous
            model.lexicon.insert(w.as_str().into(), ipa.into());
        }
        // Chinese: cmn_hani is TRADITIONAL. Add simplified aliases (OpenCC t2s)
        // so simplified input also hits.
        if lang == "zh" {
            let t2s = load_t2s();
            let aliases: Vec<(Box<str>, Box<str>)> = model
                .lexicon
                .iter()
                .filter_map(|(w, ipa)| {
                    let simp: String = w
                        .chars()
                        .map(|c| t2s.get(&c).copied().unwrap_or(c))
                        .collect();
                    (simp != **w && !model.lexicon.contains_key(simp.as_str()))
                        .then(|| (simp.into(), ipa.clone()))
                })
                .collect();
            let added = aliases.len();
            model.lexicon.extend(aliases);
            println!("zh: +{added} simplified aliases");
        }
        // Optional hand/LLM-authored lexicon supplement (e.g. bare-kanji
        // readings for Japanese). Only fills keys the gold data lacks.
        let sup = format!("data/lex/{lang}.tsv");
        if std::path::Path::new(&sup).exists() {
            let mut added = 0;
            for (w, p) in load_tsv(&sup) {
                let ipa: String = p.split_whitespace().collect();
                model.lexicon.entry(w.into()).or_insert_with(|| {
                    added += 1;
                    ipa.into()
                });
            }
            println!("{lang}: +{added} supplement entries");
        }
        println!("{lang}: lexicon {} entries", model.lexicon.len());
    } else {
        // Alphabetic languages: EM is capped at MAX_PAIRS, so on a large gold set
        // the n-gram never sees most of it -- French keeps 40k of 97k pairs, and
        // the dropped words are decoded by rule. That is fine for regular
        // spellings and wrong for irregular ones: "saint" -> "zɛ̃", not "sɛ̃".
        //
        // The lexicon tier already runs before the n-gram (decode::phonemize) and
        // is already serialised, so fill it from the *full* gold set with only
        // the words the n-gram gets wrong. Regular words stay out, so the blob
        // grows by the irregulars alone.
        // Key on the surface form, not the lowercased one: case carries meaning.
        // WikiPron has both `Jean` (ʒɑ̃, the name) and `jean` (dʒin, the fabric),
        // and 160 more French pairs like it -- Aimé/aimé, Anglais/anglais. Folding
        // them together would let whichever sorts first define the other.
        // decode::phonemize tries the exact key before the lowercased one, so
        // storing both surface forms resolves each correctly.
        let mut variants: HashMap<String, Vec<String>> = HashMap::new();
        for (w, p) in load_tsv(gold_tsv) {
            let ipa: String = p.split_whitespace().collect();
            variants.entry(w).or_default().push(ipa);
        }
        let total = variants.len();
        let mut added = 0;
        for (w, ipas) in &variants {
            // WikiPron lists pronunciation variants; any of them is a correct
            // decode, so only words matching none need an entry.
            if !ipas.iter().any(|i| *i == phonemize(&model, w)) {
                model
                    .lexicon
                    .insert(w.as_str().into(), ipas[0].as_str().into());
                added += 1;
            }
        }
        let exact = total - added;
        println!(
            "{lang}: n-gram exact on {exact}/{total} ({:.1}%); +{added} residual lexicon entries",
            100.0 * exact as f64 / total.max(1) as f64
        );
    }

    let bytes = model.to_bytes();
    fs::create_dir_all("data").unwrap();
    let path = format!("data/{lang}.g2p");
    fs::write(&path, &bytes).unwrap();
    println!(
        "wrote {path}: {} tokens, {} grams, {} bytes",
        model.tokens.len(),
        model.ngram.len(),
        bytes.len()
    );
}

/// Build every language that has a downloaded TSV.
fn build_all() {
    let (mut ok, mut skip) = (0, 0);
    for lang in g2p::lang::LANGS {
        let tsv = format!("data/wikipron/{}.tsv", lang.whisper);
        let has_gold = std::path::Path::new(&tsv).exists();
        let has_silver =
            std::path::Path::new(&format!("data/silver/{}.tsv", lang.whisper)).exists();
        if has_gold || has_silver {
            println!("--- {} ---", lang.whisper);
            build(lang.whisper, &tsv); // build merges silver on its own
            ok += 1;
        } else {
            skip += 1;
        }
    }
    println!("\nbuilt {ok} models, skipped {skip} (no tsv)");
}

fn say(lang: &str, word: &str) {
    let path = format!("data/{lang}.g2p");
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e} (build first)"));
    let model = Model::from_bytes(&bytes);
    println!("{word}  ->  {}", phonemize(&model, word));
}

fn main() {
    let a: Vec<String> = env::args().skip(1).collect();
    match a.first().map(|s| s.as_str()) {
        Some("fetch") if a.len() >= 2 => {
            fetch::fetch_one(&a[1]);
        }
        Some("fetch-all") => fetch::fetch_all(),
        Some("silver") if a.len() >= 2 => silver::silver_one(&a[1]),
        Some("bench") => bench::run(a.get(1).map(|s| s.as_str()).unwrap_or("data/fr.g2p")),
        Some("build-all") => build_all(),
        // build from an explicit tsv, or default to the fetched data/wikipron/<lang>.tsv
        Some("build") if a.len() >= 2 => {
            let tsv = a
                .get(2)
                .cloned()
                .unwrap_or_else(|| format!("data/wikipron/{}.tsv", a[1]));
            build(&a[1], &tsv);
        }
        Some("say") if a.len() >= 3 => say(&a[1], &a[2]),
        _ => eprintln!(
            "usage:\n  xtask fetch <lang>\n  xtask fetch-all\n  \
             xtask build <lang> [tsv]\n  xtask say <lang> <word>"
        ),
    }
}
