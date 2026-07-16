"""Generate the `g2p2[all]` offline model packages from a directory of .g2p files.

PyPI caps files at 100 MB and the full-lexicon models total far more, so the
pack is split: N pure-data chunk packages (`g2p2-models-partN`, xz-compressed
blobs balanced to stay under the cap) plus one `g2p2-models` meta-package that
depends on the exact chunk versions. `g2p2.model_path` looks the chunks up at
runtime and decompresses into the user cache on first use.

    python scripts/pack_models.py <models_dir> <out_dir> <version>

Emits wheels into <out_dir>/dist, ready for `twine upload` / CI publish.
"""

import lzma
import shutil
import subprocess
import sys
from pathlib import Path

CHUNK_BUDGET = 85 * 1024 * 1024  # stay well under PyPI's 100 MB file cap

PYPROJECT = """\
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "{version}"
description = "{description}"
requires-python = ">=3.8"
license = {{ text = "MIT OR Apache-2.0" }}
{deps}
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


def main(models_dir: str, out_dir: str, version: str) -> None:
    src = Path(models_dir)
    out = Path(out_dir)
    dist = out / "dist"
    dist.mkdir(parents=True, exist_ok=True)

    # Compress everything up front so chunks balance on real sizes.
    xz_dir = out / "xz"
    xz_dir.mkdir(exist_ok=True)
    blobs = []
    for f in sorted(src.glob("*.g2p")):
        xz = xz_dir / (f.name + ".xz")
        if not xz.exists():
            xz.write_bytes(lzma.compress(f.read_bytes(), preset=9))
        blobs.append((xz, xz.stat().st_size))
    total = sum(s for _, s in blobs)
    print(f"{len(blobs)} models, {total / 1e6:.0f} MB compressed")

    # Greedy fill: biggest first, into the first chunk with room.
    blobs.sort(key=lambda b: -b[1])
    chunks: list[list[Path]] = []
    sizes: list[int] = []
    for xz, size in blobs:
        for i, used in enumerate(sizes):
            if used + size <= CHUNK_BUDGET:
                chunks[i].append(xz)
                sizes[i] += size
                break
        else:
            chunks.append([xz])
            sizes.append(size)

    part_names = []
    for i, files in enumerate(chunks, 1):
        name = f"g2p2-models-part{i}"
        module = f"g2p2_models_part{i}"
        part_names.append(name)
        pkg = out / name
        moddir = pkg / module
        moddir.mkdir(parents=True, exist_ok=True)
        (moddir / "__init__.py").write_text(
            f'"""g2p2 model data chunk {i}/{len(chunks)} — see g2p2[all]."""\n'
        )
        for xz in files:
            shutil.copy2(xz, moddir / xz.name)
        (pkg / "pyproject.toml").write_text(
            PYPROJECT.format(
                name=name,
                version=version,
                description=f"g2p2 language model data (chunk {i}/{len(chunks)})",
                module=module,
                deps="",
            )
        )
        langs = ", ".join(sorted(x.name.split(".")[0] for x in files))
        print(f"{name}: {sizes[i - 1] / 1e6:.0f} MB — {langs}")

    meta = out / "g2p2-models"
    (meta / "g2p2_models").mkdir(parents=True, exist_ok=True)
    (meta / "g2p2_models" / "__init__.py").write_text(
        '"""Meta-package: depending on it pulls every g2p2 model chunk."""\n'
    )
    deps = ", ".join(f'"{n}=={version}"' for n in part_names)
    (meta / "pyproject.toml").write_text(META_PYPROJECT.format(version=version, deps=deps))

    for pkg in [*(out / n for n in part_names), meta]:
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dist), str(pkg)],
            check=True,
            capture_output=True,
        )
    print(f"\nwheels in {dist}:")
    for w in sorted(dist.glob("*.whl")):
        print(f"  {w.name}  {w.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:4])
