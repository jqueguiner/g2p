//! Download WikiPron TSVs. The file index is embedded (avoids GitHub API rate
//! limits); only the selected TSV is fetched, by shelling out to `curl` (no
//! Rust HTTP crate — keeps the dependency surface minimal).

use std::fs;
use std::process::Command;

use g2p::lang::{by_whisper, Lang, LANGS};

const INDEX: &str = include_str!("../data/wikipron_index.txt");
const BASE: &str = "https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv";

/// Pick the best WikiPron filename for a language, or `None` if unavailable.
/// Ranking: prefer `broad` over narrow, `_filtered` (cleaned) over plain, then
/// shortest name.
pub fn select(lang: &Lang) -> Option<&'static str> {
    INDEX
        .lines()
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .filter(|f| f.starts_with(&format!("{}_", lang.iso)))
        .filter(|f| lang.pat.is_empty() || f.contains(lang.pat))
        .min_by_key(|f| {
            let broad = if f.contains("broad") { 0 } else { 1 };
            let filt = if f.contains("filtered") { 0 } else { 1 };
            (broad, filt, f.len())
        })
}

/// Download one language's TSV to `data/wikipron/{whisper}.tsv`. Returns the
/// local path on success.
pub fn fetch_one(code: &str) -> Option<String> {
    let lang = by_whisper(code)?;
    let file = match select(lang) {
        Some(f) => f,
        None => {
            eprintln!("{code}: no WikiPron file (needs silver)");
            return None;
        }
    };
    fs::create_dir_all("data/wikipron").unwrap();
    let out = format!("data/wikipron/{code}.tsv");
    let url = format!("{BASE}/{file}");
    let status = Command::new("curl")
        .args(["-sSL", "--fail", "-o", &out, &url])
        .status()
        .expect("curl not found");
    if status.success() {
        let n = fs::read_to_string(&out)
            .map(|s| s.lines().count())
            .unwrap_or(0);
        println!("{code:<4} <- {file}  ({n} pairs)");
        Some(out)
    } else {
        eprintln!("{code}: curl failed for {file}");
        None
    }
}

/// Fetch every language that has a WikiPron source.
pub fn fetch_all() {
    let (mut ok, mut skip) = (0, 0);
    for lang in LANGS {
        if fetch_one(lang.whisper).is_some() {
            ok += 1;
        } else {
            skip += 1;
        }
    }
    println!("\nfetched {ok}, skipped {skip} (need silver)");
}
