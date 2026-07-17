"""Generate per-language `g2p2-lang-<code>` model packages + the `g2p2-models`
meta from a directory of .g2p files.

One wheel per language keeps every file small (largest model ~35 MB, xz ~15 MB,
well under PyPI's cap) and lets users install only what they need:
`pip install g2p2[fr]` pulls just French. `g2p2[all]` pulls `g2p2-models`, which
depends on every `g2p2-lang-<code>`.

    python scripts/pack_models.py <models_dir> <out_dir> <version>

Publish with a PyPI API token (data packages can't share one pending publisher).
"""

import lzma
import subprocess
import sys
from pathlib import Path

LANG_PYPROJECT = """\
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "g2p2-lang-{code}"
version = "{version}"
description = "g2p2 grapheme-to-phoneme model for {code}"
requires-python = ">=3.8"
license = {{ text = "MIT OR Apache-2.0" }}

[tool.setuptools]
packages = ["{module}"]

[tool.setuptools.package-data]
"{module}" = ["*.g2p.xz"]
"""

META_PYPROJECT = """\
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "g2p2-models"
version = "{version}"
description = "All g2p2 language models for offline use (install via g2p2[all])"
requires-python = ">=3.8"
license = {{ text = "MIT OR Apache-2.0" }}
dependencies = [{deps}]

[tool.setuptools]
packages = ["g2p2_models"]
"""


def wheel(pkg_dir, dist):
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dist), str(pkg_dir)],
        check=True, capture_output=True,
    )


def main(models_dir, out_dir, version):
    src, out = Path(models_dir), Path(out_dir)
    dist = out / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    codes = sorted(f.stem for f in src.glob("*.g2p"))
    print(f"{len(codes)} languages -> per-language packages")

    for code in codes:
        module = f"g2p2_lang_{code}"
        pkg = out / f"g2p2-lang-{code}"
        moddir = pkg / module
        moddir.mkdir(parents=True, exist_ok=True)
        (moddir / "__init__.py").write_text(f'"""g2p2 model data for {code}."""\n')
        (moddir / f"{code}.g2p.xz").write_bytes(lzma.compress((src / f"{code}.g2p").read_bytes(), preset=9))
        (pkg / "pyproject.toml").write_text(LANG_PYPROJECT.format(code=code, version=version, module=module))
        wheel(pkg, dist)

    meta = out / "g2p2-models"
    (meta / "g2p2_models").mkdir(parents=True, exist_ok=True)
    (meta / "g2p2_models" / "__init__.py").write_text('"""Meta: depends on every g2p2 language model."""\n')
    deps = ", ".join(f'"g2p2-lang-{c}=={version}"' for c in codes)
    (meta / "pyproject.toml").write_text(META_PYPROJECT.format(version=version, deps=deps))
    wheel(meta, dist)

    wheels = sorted(dist.glob("*.whl"))
    total = sum(w.stat().st_size for w in wheels)
    print(f"{len(wheels)} wheels, {total/1e6:.0f} MB total, largest {max(w.stat().st_size for w in wheels)/1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:4])
