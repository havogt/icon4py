# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Write the CPU reference of `jw_tpu.ipynb`: the sampled prognostics after NUM_STEPS jitted JW
steps on R02B04, in single and in double precision, one process per precision.

    python scripts/colab/jw_reference.py <grid directory> [<out.npz>]
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile


NUM_STEPS = 10
HERE = pathlib.Path(__file__).parent


def child(grid_dir: pathlib.Path, out: pathlib.Path) -> None:
    import time  # noqa: PLC0415 [import-outside-top-level]

    import numpy as np  # noqa: PLC0415 [import-outside-top-level]

    sys.path.insert(0, str(HERE))
    import jw_run  # noqa: PLC0415 [import-outside-top-level]

    grid_file = jw_run.download_grid("R02B04", grid_dir)
    with tempfile.TemporaryDirectory() as output_path:
        static, state = jw_run.init(grid_file, pathlib.Path(output_path))
    report = jw_run.precision_report(static, state, jw_run.jax.jit(jw_run.jax_api.time_step))
    indices = jw_run.sample_indices(static, state)
    t0 = time.perf_counter()
    final = jw_run.prognostics(jw_run.run(static, state, NUM_STEPS))
    elapsed = time.perf_counter() - t0
    np.savez(
        out,
        **{f"idx_{k}": v for k, v in indices.items()},
        **{f"value_{k}": v[indices[k]] for k, v in final.items()},
        **{f"maxabs_{k}": np.abs(v).max() for k, v in final.items()},
        report=json.dumps(report),
        seconds=elapsed,
    )


def main(grid_dir: pathlib.Path, out: pathlib.Path) -> None:
    import numpy as np  # noqa: PLC0415 [import-outside-top-level]

    parts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for precision in ("single", "double"):
            part = pathlib.Path(tmp) / f"{precision}.npz"
            subprocess.run(
                [sys.executable, __file__, "--child", str(grid_dir), str(part)],
                env=os.environ | {"ICON4PY_FLOAT_PRECISION": precision},
                check=True,
            )
            with np.load(part) as data:
                parts[precision] = dict(data)

    sys.path.insert(0, str(HERE))
    import hashlib  # noqa: PLC0415 [import-outside-top-level]

    import gt4py  # noqa: PLC0415 [import-outside-top-level]
    import jax  # noqa: PLC0415 [import-outside-top-level]

    single, double = parts["single"], parts["double"]
    for key in (k for k in single if k.startswith("idx_")):
        np.testing.assert_array_equal(single[key], double[key], err_msg=key)
    icon4py_head = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            [
                "git",
                "-C",
                str(HERE),
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "../../model",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    meta = {
        "num_steps": NUM_STEPS,
        "grid": "R02B04",
        "icon4py_commit": icon4py_head,
        "icon4py_model_dirty": dirty,
        "gt4py_version": gt4py.__version__,
        "jax_version": jax.__version__,
        "device": "cpu",
        "config_sha256": hashlib.sha256((HERE / "jw_config.yml").read_bytes()).hexdigest(),
        "cpu_seconds_for_steps": {p: float(parts[p]["seconds"]) for p in parts},
        "precision_reports": {p: json.loads(str(parts[p]["report"])) for p in parts},
    }
    arrays = {k: v for k, v in single.items() if k.startswith("idx_")}
    for precision, data in parts.items():
        arrays |= {
            f"{precision}_{k.removeprefix('value_')}": v
            for k, v in data.items()
            if k.startswith("value_")
        }
        arrays |= {f"{precision}_{k}": v for k, v in data.items() if k.startswith("maxabs_")}
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    else:
        main(
            pathlib.Path(sys.argv[1]),
            pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "jw_reference_R02B04.npz",
        )
