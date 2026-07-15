//! `g2p` — command-line grapheme-to-phoneme.
//!
//!   g2p <model.g2p> <word>...      phonemize each word
//!   echo "words" | g2p <model.g2p> read whitespace-separated words from stdin
//!
//! Models are the `.g2p` blobs produced by the build tool. Zero dependencies.

use std::io::Read;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(model_path) = args.first() else {
        eprintln!("usage: g2p <model.g2p> [word ...]   (words also read from stdin)");
        return ExitCode::from(2);
    };

    let bytes = match std::fs::read(model_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("g2p: cannot read {model_path}: {e}");
            return ExitCode::from(1);
        }
    };
    let model = g2p::Model::from_bytes(&bytes);

    let mut words: Vec<String> = args[1..].to_vec();
    if words.is_empty() {
        let mut buf = String::new();
        if std::io::stdin().read_to_string(&mut buf).is_ok() {
            words = buf.split_whitespace().map(String::from).collect();
        }
    }

    for w in words {
        println!("{}", g2p::phonemize(&model, &w));
    }
    ExitCode::SUCCESS
}
