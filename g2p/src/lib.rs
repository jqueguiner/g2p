//! Zero-dependency grapheme-to-phoneme (G2P) for the 101 Whisper languages.
//!
//! Runtime engine: pair joint n-gram + beam decode. Pure `std`, CPU.
//! Models are compiled offline by `xtask` into `.g2p` blobs and loaded via
//! [`Model::from_bytes`].

pub mod decode;
pub mod lang;
pub mod lexicon;
pub mod model;
pub mod normalize;

pub use decode::phonemize;
pub use model::Model;
