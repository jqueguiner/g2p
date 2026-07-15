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

    fn __repr__(&self) -> String {
        format!("<g2p2.Model {} tokens>", self.inner.tokens.len())
    }
}

#[pymodule]
fn g2p2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Model>()?;
    Ok(())
}
