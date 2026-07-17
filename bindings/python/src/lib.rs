//! Python bindings for the `g2p` engine (PyO3 + maturin, abi3).
//!
//! ```python
//! from g2p2 import Model
//! m = Model.load("fr.g2p")
//! m.phonemize("bonjour")          # 'bɔ̃ʒuʁ'
//! m.phonemize_many(["a", "b"])    # ['...', '...']
//! ```

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// A loaded G2P model. Construct with `Model.load(path)` or `Model.from_bytes(b)`.
#[pyclass]
struct Model {
    inner: g2p::Model,
}

#[pymethods]
impl Model {
    /// Load a compiled `.g2p` blob from disk.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            inner: g2p::Model::from_bytes(&bytes),
        })
    }

    /// Load a model from raw `.g2p` bytes.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            inner: g2p::Model::from_bytes(bytes),
        }
    }

    /// Phonemize one word -> IPA string.
    fn phonemize(&self, word: &str) -> String {
        g2p::phonemize(&self.inner, word)
    }

    /// Phonemize many words -> list of IPA strings.
    fn phonemize_many(&self, words: Vec<String>) -> Vec<String> {
        words.iter().map(|w| g2p::phonemize(&self.inner, w)).collect()
    }

    /// Phonemize two words, then score their pronunciation similarity (0..1).
    #[pyo3(signature = (a, b, method = "weighted"))]
    fn word_similarity(&self, a: &str, b: &str, method: &str) -> f32 {
        let ia = g2p::phonemize(&self.inner, a);
        let ib = g2p::phonemize(&self.inner, b);
        g2p::similarity(&ia, &ib, method_of(method))
    }

    fn __repr__(&self) -> String {
        format!("<g2p2.Model {} tokens>", self.inner.tokens.len())
    }
}

fn method_of(s: &str) -> g2p::Method {
    match s {
        "fast" | "levenshtein" | "lev" => g2p::Method::Levenshtein,
        _ => g2p::Method::Weighted, // default: better
    }
}

/// Phonetic similarity between two IPA strings (0..1). method: "weighted" (default) | "fast".
#[pyfunction]
#[pyo3(signature = (a, b, method = "weighted"))]
fn similarity(a: &str, b: &str, method: &str) -> f32 {
    g2p::similarity(a, b, method_of(method))
}

/// Phonetic distance between two IPA strings (0..1). method: "weighted" (default) | "fast".
#[pyfunction]
#[pyo3(signature = (a, b, method = "weighted"))]
fn distance(a: &str, b: &str, method: &str) -> f32 {
    g2p::similarity::distance(a, b, method_of(method))
}

/// Spell digit tokens in `text` as words in `lang` before G2P
/// ("12 rue" -> "douze rue"). Non-numeric tokens pass through unchanged.
#[cfg(feature = "numbers")]
#[pyfunction]
fn expand_numbers(text: &str, lang: &str) -> String {
    g2p::expand_numbers(text, lang)
}

/// Spell one integer numeral as cardinal words in `lang` ("12","fr" ->
/// "douze"), or None if it isn't a bare integer / the language is unsupported.
#[cfg(feature = "numbers")]
#[pyfunction]
fn spell_cardinal(token: &str, lang: &str) -> Option<String> {
    g2p::spell_cardinal(token, lang)
}

/// Low-level native extension. The user-facing API lives in the `g2p2` Python
/// package (which adds language-based auto-loading on top of this).
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(similarity, m)?)?;
    m.add_function(wrap_pyfunction!(distance, m)?)?;
    #[cfg(feature = "numbers")]
    {
        m.add_function(wrap_pyfunction!(expand_numbers, m)?)?;
        m.add_function(wrap_pyfunction!(spell_cardinal, m)?)?;
    }
    Ok(())
}
