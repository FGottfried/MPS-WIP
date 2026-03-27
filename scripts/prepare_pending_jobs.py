import argparse
import json
from pathlib import Path

DEFAULT_BOND_DIMS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
DEFAULT_N_QUBITS = [4, 8, 16, 32]


def _as_int_list(values):
    out = [int(v) for v in values]
    if len(out) == 0:
        raise ValueError("List must not be empty.")
    return out


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mode_settings(raw: dict, mode: str):
    full_bond_dims = _as_int_list(raw.get("bond_dims", DEFAULT_BOND_DIMS))
    full_n_qubits = _as_int_list(raw.get("n_qubits", DEFAULT_N_QUBITS))

    if mode == "sanity":
        sanity = raw.get("sanity_check", {})
        bond_dims = _as_int_list(sanity.get("bond_dims", full_bond_dims[:2]))
        n_qubits_list = _as_int_list(sanity.get("n_qubits", full_n_qubits[:2]))
        out_subdir = "sanity_check"
    elif mode == "full":
        bond_dims = full_bond_dims
        n_qubits_list = full_n_qubits
        out_subdir = "full_sweep"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return bond_dims, n_qubits_list, out_subdir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pending (n_qubits, bond_dim) jobs")
    parser.add_argument("--config", type=Path, default=Path("mps_embedding_config.json"))
    parser.add_argument("--mode", choices=["sanity", "full"], default="full")
    args = parser.parse_args()

    raw = load_config(args.config)
    bond_dims, n_qubits_list, out_subdir = mode_settings(raw, args.mode)
    base_out_dir = Path(raw.get("output", {}).get("out_dir", "."))
    out_dir = base_out_dir / out_subdir
    metrics_dir = out_dir / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pending_file = out_dir / "pending_jobs.txt"
    done = 0
    pending = 0

    with pending_file.open("w", encoding="utf-8") as f:
        for n_qubits in n_qubits_list:
            for bond_dim in bond_dims:
                run_name = f"nq{n_qubits}_bond{bond_dim}"
                marker = metrics_dir / f"{run_name}.json"
                if marker.exists():
                    done += 1
                else:
                    pending += 1
                    f.write(f"{n_qubits},{bond_dim}\n")

    print(f"Mode: {args.mode}")
    print(f"Completed jobs: {done}")
    print(f"Pending jobs: {pending}")
    print(f"Pending file: {pending_file}")


if __name__ == "__main__":
    main()
