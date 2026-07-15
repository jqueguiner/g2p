//! Silver-data generation for languages WikiPron does not cover.
//!
//! Pipeline: fetch an epitran grapheme->IPA map (rule table), fetch a wordlist
//! from the language's Wikipedia page titles, apply the map by longest-match to
//! synthesize (word, IPA) pairs. Written to `data/silver/{lang}.tsv` and merged
//! into training at a low weight (or used alone for no-gold languages).
//!
//! Only languages with an epitran map are supported here; others (tat/lin/sun)
//! have neither WikiPron nor an epitran map and are left to the lexicon tier.

use std::collections::BTreeSet;
use std::fs;
use std::process::Command;

const EP_BASE: &str = "https://raw.githubusercontent.com/dmort27/epitran/master/epitran/data/map";

/// (whisper code, epitran map code, wikipedia iso2 subdomain)
const EP: &[(&str, &str, &str)] = &[
    ("sn", "sna-Latn", "sn"),
    ("so", "som-Latn", "so"),
    ("jw", "jav-Latn", "jv"),
];

fn spec(whisper: &str) -> Option<(&'static str, &'static str)> {
    EP.iter()
        .find(|(w, _, _)| *w == whisper)
        .map(|(_, ep, wiki)| (*ep, *wiki))
}

/// Wikimedia requires a descriptive User-Agent with contact info, else it
/// rate-limits aggressively.
const UA: &str = "G2P-silver/0.1 (https://github.com/; grapheme-to-phoneme dataset build)";

fn curl(url: &str) -> Option<String> {
    let out = Command::new("curl")
        .args(["-sSL", "--fail", "-A", UA, url])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// epitran map, sorted longest-grapheme-first for greedy longest match.
fn load_map(code: &str) -> Vec<(String, String)> {
    let csv = curl(&format!("{EP_BASE}/{code}.csv")).expect("fetch epitran map");
    let mut m: Vec<(String, String)> = csv
        .lines()
        .skip(1) // header: Orth,Phon
        .filter_map(|l| l.split_once(','))
        .map(|(o, p)| (o.trim().to_lowercase(), p.trim().to_string()))
        .filter(|(o, _)| !o.is_empty())
        .collect();
    m.sort_by(|a, b| b.0.chars().count().cmp(&a.0.chars().count()));
    m
}

/// Longest-match rewrite -> space-separated IPA segments (matches WikiPron form).
fn apply(map: &[(String, String)], word: &str) -> String {
    let ch: Vec<char> = word.chars().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut i = 0;
    'outer: while i < ch.len() {
        for (o, p) in map {
            let ol = o.chars().count();
            if ol > 0 && i + ol <= ch.len() && ch[i..i + ol].iter().collect::<String>() == *o {
                if !p.is_empty() {
                    out.push(p);
                }
                i += ol;
                continue 'outer;
            }
        }
        i += 1; // unmapped char: drop
    }
    out.join(" ")
}

/// Extract every `"title":"..."` value from an API JSON blob (naive, sufficient).
fn scan_titles(js: &str) -> Vec<String> {
    let mut out = Vec::new();
    let pat = "\"title\":\"";
    let bytes = js.as_bytes();
    let mut i = 0;
    while let Some(p) = js[i..].find(pat) {
        let start = i + p + pat.len();
        let mut j = start;
        while j < bytes.len() && bytes[j] != b'"' {
            if bytes[j] == b'\\' {
                j += 1;
            }
            j += 1;
        }
        out.push(js[start..j].to_string());
        i = j;
    }
    out
}

fn scan_continue(js: &str) -> Option<String> {
    let pat = "\"apcontinue\":\"";
    let p = js.find(pat)? + pat.len();
    let rest = &js[p..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Split a title into clean single-word tokens (all-alphabetic, len>=2, lower).
fn tokens(title: &str) -> Vec<String> {
    title
        .split(|c: char| !c.is_alphabetic())
        .filter(|t| t.chars().count() >= 2)
        .map(|t| t.to_lowercase())
        .collect()
}

/// Collect up to `target` distinct words from a Wikipedia's page titles.
fn wordlist(iso2: &str, target: usize) -> Vec<String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    let mut cont = String::new();
    for _ in 0..40 {
        let url = format!(
            "https://{iso2}.wikipedia.org/w/api.php?action=query&list=allpages\
             &aplimit=500&apnamespace=0&format=json{}",
            if cont.is_empty() {
                String::new()
            } else {
                format!("&apcontinue={}", cont.replace(' ', "%20"))
            }
        );
        let Some(js) = curl(&url) else { break };
        for t in scan_titles(&js) {
            for tok in tokens(&t) {
                set.insert(tok);
            }
        }
        match scan_continue(&js) {
            Some(c) => cont = c,
            None => break,
        }
        if set.len() >= target {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(400)); // be polite to the API
    }
    set.into_iter().take(target).collect()
}

/// Generate `data/silver/{lang}.tsv` for a language with an epitran map.
pub fn silver_one(whisper: &str) {
    let Some((ep, wiki)) = spec(whisper) else {
        eprintln!("{whisper}: no epitran map (lexicon-only)");
        return;
    };
    let map = load_map(ep);
    println!("{whisper}: epitran map {ep} ({} rules)", map.len());
    let words = wordlist(wiki, 15_000);
    println!("{whisper}: {} words from {wiki}.wikipedia", words.len());

    fs::create_dir_all("data/silver").unwrap();
    let path = format!("data/silver/{whisper}.tsv");
    let mut buf = String::new();
    let mut n = 0;
    for w in &words {
        let ipa = apply(&map, w);
        if !ipa.is_empty() {
            buf.push_str(w);
            buf.push('\t');
            buf.push_str(&ipa);
            buf.push('\n');
            n += 1;
        }
    }
    fs::write(&path, buf).unwrap();
    println!("{whisper}: wrote {path} ({n} silver pairs)");
}
