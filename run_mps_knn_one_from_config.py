import argparse
from dataclasses import replace
from pathlib import Path

from run_mps_knn_from_config import load_config, mode_settings, sweep_config_from_json
from run_mps_knn_sweep import load_mnist_flat, run_single_combo, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one (n_qubits, bond_dim) experiment from JSON config")
    parser.add_argument("--config", type=Path, default=Path("mps_embedding_config.json"))
    parser.add_argument("--mode", choices=["sanity", "full"], default="full")
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--bond-dim", type=int, required=True)
    parser.add_argument("--skip-if-done", action="store_true")
    args = parser.parse_args()

    raw = load_config(args.config)
    cfg = sweep_config_from_json(raw)
    bond_dims, n_qubits_list, out_subdir = mode_settings(raw, args.mode)

    if args.bond_dim not in set(bond_dims):
        raise ValueError(f"bond_dim={args.bond_dim} is not part of mode '{args.mode}' list {bond_dims}")
    if args.n_qubits not in set(n_qubits_list):
        raise ValueError(f"n_qubits={args.n_qubits} is not part of mode '{args.mode}' list {n_qubits_list}")

    out_dir = Path(cfg.out_dir) / out_subdir
    run_name = f"nq{args.n_qubits}_bond{args.bond_dim}"
    done_marker = out_dir / "results" / "metrics" / f"{run_name}.json"

    if args.skip_if_done and done_marker.exists():
        print(f"Skipping {run_name}: already completed ({done_marker})", flush=True)
        return

    cfg_mode = replace(cfg, out_dir=str(out_dir))
    set_seed(cfg_mode.seed)
    print(f"Running single job: {run_name} in {args.mode} mode", flush=True)
    print("Loading MNIST dataset...", flush=True)
    x_train_flat, y_train, x_test_flat, y_test = load_mnist_flat(
        cfg_mode.data_root,
        cfg_mode.max_train,
        cfg_mode.max_test,
    )

    _ = run_single_combo(
        cfg=cfg_mode,
        x_train_flat=x_train_flat,
        y_train=y_train,
        x_test_flat=x_test_flat,
        y_test=y_test,
        n_qubits=args.n_qubits,
        bond_dim=args.bond_dim,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
