"""Generate per-family `g2p2-group-<name>` model packages + the `g2p2-models`
meta from a directory of .g2p files and a groups.json ({group: [codes]}).

Grouping by language family keeps the package count low (~14, well under PyPI's
new-project rate limit) while still letting users install a slice:
`pip install g2p2[romance]` pulls the Romance models; `g2p2[all]` pulls the
g2p2-models meta which depends on every group.

    python scripts/pack_models.py <models_dir> <out_dir> <version> <groups.json>
"""

import json
import lzma
import subprocess
import sys
from pathlib import Path

GROUP_PYPROJECT = """\
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "g2p2-group-{name}"
version = "{version}"
description = "g2p2 grapheme-to-phoneme models: {name} languages"
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


def wheel(pkg, dist):
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dist), str(pkg)],
                   check=True, capture_output=True)


def main(models_dir, out_dir, version, groups_path):
    src, out = Path(models_dir), Path(out_dir)
    dist = out / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    groups = json.load(open(groups_path))
    print(f"{len(groups)} groups")

    for name, codes in groups.items():
        module = f"g2p2_group_{name}".replace("-", "_")
        pkg = out / f"g2p2-group-{name}"
        moddir = pkg / module
        moddir.mkdir(parents=True, exist_ok=True)
        (moddir / "__init__.py").write_text(f'"""g2p2 {name} models."""\n')
        for c in codes:
            f = src / f"{c}.g2p"
            if f.exists():
                (moddir / f"{c}.g2p.xz").write_bytes(lzma.compress(f.read_bytes(), preset=9))
        (pkg / "pyproject.toml").write_text(GROUP_PYPROJECT.format(name=name, version=version, module=module))
        wheel(pkg, dist)

    meta = out / "g2p2-models"
    (meta / "g2p2_models").mkdir(parents=True, exist_ok=True)
    (meta / "g2p2_models" / "__init__.py").write_text('"""Meta: depends on every g2p2 language group."""\n')
    deps = ", ".join(f'"g2p2-group-{n}=={version}"' for n in groups)
    (meta / "pyproject.toml").write_text(META_PYPROJECT.format(version=version, deps=deps))
    wheel(meta, dist)

    ws = sorted(dist.glob("*.whl"))
    print(f"{len(ws)} wheels, largest {max(w.stat().st_size for w in ws)/1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:5])
