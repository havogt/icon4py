# `gridtools_cpp` is not git-installable — why, and how gridtools could fix it

## Context

The gtfn-optimization work spans three branches (`havogt_gtfn_optimize` on each):

| component | how it's pinned in icon4py |
|---|---|
| **icon4py** | the branch itself (model_options / model_backends) |
| **gt4py** | clean git pin in `[tool.uv.sources]` — pure Python, git-installable |
| **gridtools_cpp** | **cannot be git-pinned** — see below |

`gt4py` requires `gridtools-cpp>=2.3.9,==2.*`, and the gtfn changes need *modified*
gridtools headers (scan-fusion in `fn/column_stage.hpp` + `fn/executor.hpp`, branchless
skip-reduce in `fn/unstructured.hpp`). The natural wish is to pin it the same way as gt4py:

```toml
gridtools_cpp = {git = "https://github.com/havogt/gridtools", branch = "havogt_gtfn_optimize", subdirectory = ".python_package"}
```

**This does not work.**

## The problem

`gridtools_cpp` is *not* the gridtools repo — it is a **wheel that bundles the C++ headers**,
produced from `gridtools/.python_package/`. Its build is **two-step** (see
`.python_package/noxfile.py::prepare`):

1. **CMake pre-step** — `cmake --install` configures gridtools and copies the headers + the
   `GridTools` CMake config into `src/gridtools_cpp/data/`; then `setup.cfg.in` is rendered to
   `setup.cfg` with the version read from `version.txt`.
2. **setuptools build** — `build-backend = "setuptools.build_meta"` packages
   `src/gridtools_cpp/data/` (the now-present headers) into the wheel.

When `pip`/`uv` build from a git source, they invoke **only the PEP 517 build backend
(step 2)**. Step 1 never runs. So at build time:

- `src/gridtools_cpp/data/` is **empty** → the wheel contains **0 headers**.
- `setup.cfg` does **not exist** (only the `.in` template) → no metadata, **version `0.0.0`**
  (which also fails gt4py's `gridtools-cpp>=2.3.9,==2.*`).

Empirically (bare `pip wheel ./.python_package`):
`gridtools_cpp-0.0.0-py3-none-any.whl`, **1.6 KB, 0 `.hpp` files** — a silent, broken install.

The packaging is designed for the **maintainer** to run (`nox -s build_wheel` → publish to the
gridtools PyPI index), not for a **consumer** to build from source.

## Current workaround (what icon4py does)

Build the real wheel from the branch and install it over the stock one:

```bash
cd gridtools/.python_package && nox -s build_wheel   # runs prepare (cmake-install) + build
uv pip install --force-reinstall <dist>/gridtools_cpp-2.3.9-py3-none-any.whl
```

This yields the correct wheel (≈1 MB, 521 headers, `py3-none-any`, version `2.3.9`, with the
modified `column_stage`/`executor`/`unstructured` headers). But it is a **manual overlay** that a
plain `uv sync` cannot reproduce — exactly the durability gap we want to remove.

## The fix (gridtools side): make `.python_package` self-building

Make the build backend run the CMake header-collection itself, so a plain
`pip install` (and therefore `git+…#subdirectory=.python_package`) does the whole thing.
For a `#subdirectory` install pip clones the *entire* repo, so the gridtools sources (`../include`)
are present at build time — the CMake step can run.

**Recommended: switch the backend to `scikit-build-core`** (the standard CMake-aware PEP 517
backend). Sketch:

```toml
# .python_package/pyproject.toml
[build-system]
requires = ["scikit-build-core>=0.10"]
build-backend = "scikit_build_core.build"

[project]
name = "gridtools_cpp"
dynamic = ["version"]      # from version.txt via a metadata plugin, or set explicitly

[tool.scikit-build]
cmake.source-dir = "."     # a thin CMakeLists that installs the gridtools headers from ../ into the wheel
wheel.packages = ["src/gridtools_cpp"]
```

with a small `.python_package/CMakeLists.txt` that does the current `prepare` install
(`-DBUILD_TESTING=OFF -DGT_INSTALL_EXAMPLES=OFF`, `install` the headers + CMake config into the
package data dir). scikit-build-core runs CMake during every build — including `pip install
git+…#subdirectory=.python_package` — so `data/` is populated and the version is set, with **no
out-of-band `nox` step**.

The wheel stays **header-only → `py3-none-any`** (one arch-independent wheel; no `cibuildwheel`
matrix needed), so this is purely a packaging-mechanism change, not a binary-distribution one.

**Alternative (no backend change):** keep `setuptools.build_meta` but add an **in-tree PEP 517
backend** (`backend-path = ["_build"]`, `build-backend = "_build.backend"`) that wraps
`setuptools.build_meta` and runs the CMake `prepare` inside `build_wheel` /
`get_requires_for_build_wheel`. This is hand-rolling what scikit-build-core does and is more
fragile; prefer the scikit-build-core route.

Either way, once `.python_package` builds from source under PEP 517, the icon4py pin collapses to
the clean one-liner above and `uv sync` reproduces the deployment with no manual wheel step.
