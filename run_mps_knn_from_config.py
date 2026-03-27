import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import List, Sequence

from run_mps_knn_sweep import DEFAULT_BOND_DIMS, DEFAULT_N_QUBITS, SweepConfig, run_sweep


def _as_int_list(values: Sequence[int]) -> List[int]:
    out = [int(v) for v in values]
    if len(out) == 0:
        raise ValueError("List must not be empty.")
    return out


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sweep_config_from_json(raw: dict) -> SweepConfig:
    train = raw.get("train", {})
    eval_cfg = raw.get("evaluation", {})
    dataset = raw.get("dataset", {})
    output = raw.get("output", {})

    return SweepConfig(
        seed=int(raw.get("seed", 42)),
        data_root=str(dataset.get("data_root", "./data")),
        batch_size=int(train.get("batch_size", 64)),
        lr=float(train.get("lr", 1e-4)),
        num_epochs=int(train.get("num_epochs", 8)),
        margin=float(train.get("margin", 1.0)),
        knn_k=int(eval_cfg.get("knn_k", 5)),
        device=str(raw.get("device", "cpu")),
        max_train=int(dataset.get("max_train", 12000)),
        max_test=int(dataset.get("max_test", 2000)),
        out_dir=str(output.get("out_dir", ".")),
    )


def mode_settings(raw: dict, mode: str) -> tuple[List[int], List[int], str]:
    full_bond_dims = _as_int_list(raw.get("bond_dims", DEFAULT_BOND_DIMS))
    full_n_qubits = _as_int_list(raw.get("n_qubits", DEFAULT_N_QUBITS))

    if mode == "sanity":
        sanity = raw.get("sanity_check", {})
        bond_dims = _as_int_list(sanity.get("bond_dims", full_bond_dims[:2]))
        n_qubits = _as_int_list(sanity.get("n_qubits", full_n_qubits[:2]))
        out_subdir = "sanity_check"
    elif mode == "full":
        bond_dims = full_bond_dims
        n_qubits = full_n_qubits
        out_subdir = "full_sweep"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return bond_dims, n_qubits, out_subdir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MNIST MPS kNN sweep from JSON config")
    parser.add_argument("--config", type=Path, default=Path("mps_embedding_config.json"))
    parser.add_argument("--mode", choices=["sanity", "full"], default="sanity")
    args = parser.parse_args()

    raw = load_config(args.config)
    cfg = sweep_config_from_json(raw)
    bond_dims, n_qubits, out_subdir = mode_settings(raw, args.mode)
    base_out = Path(cfg.out_dir)
    cfg_mode = replace(cfg, out_dir=str(base_out / out_subdir))
    print(f"Running {args.mode} sweep with n_qubits={n_qubits}, bond_dims={bond_dims}")
    run_sweep(cfg_mode, bond_dims=bond_dims, n_qubits_list=n_qubits)


if __name__ == "__main__":
    main()
