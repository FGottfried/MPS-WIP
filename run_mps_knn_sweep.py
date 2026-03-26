import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


DEFAULT_BOND_DIMS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
DEFAULT_N_QUBITS = [4, 8, 16, 32]


@dataclass(frozen=True)
class SweepConfig:
    seed: int
    data_root: str
    batch_size: int
    lr: float
    num_epochs: int
    margin: float
    knn_k: int
    device: str
    max_train: int
    max_test: int
    out_dir: str


class MPSMetricLearnerEmbedCenter(nn.Module):
    def __init__(self, length: int, bond_dim: int, output_dim: int, d_phys: int = 2):
        super().__init__()
        self.length = length
        self.bond_dim = bond_dim
        self.split_idx = length // 2

        def init_tensor(shape: Tuple[int, int, int]) -> nn.Parameter:
            t = torch.randn(*shape) * 0.01
            d_left, _, d_right = shape
            min_d = min(d_left, d_right)
            with torch.no_grad():
                for p in range(shape[1]):
                    for k in range(min_d):
                        t[k, p, k] += 1.0
            return nn.Parameter(t)

        self.left_tensors = nn.ParameterList()
        for i in range(self.split_idx):
            d_left = 1 if i == 0 else bond_dim
            self.left_tensors.append(init_tensor((d_left, d_phys, bond_dim)))

        self.center_tensor = nn.Parameter(torch.randn(bond_dim, bond_dim, output_dim) / max(output_dim, 1) ** 0.5)

        self.right_tensors = nn.ParameterList()
        num_right = length - self.split_idx
        for i in range(num_right):
            is_last = i == num_right - 1
            d_right = 1 if is_last else bond_dim
            self.right_tensors.append(init_tensor((bond_dim, d_phys, d_right)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        phi = get_feature_map(x)

        left_env = torch.zeros(bsz, 1, device=x.device)
        left_env[:, 0] = 1.0
        for i, a in enumerate(self.left_tensors):
            left_env = torch.einsum("bl,lpr,bp->br", left_env, a, phi[:, i])
            left_env = left_env / (left_env.norm(dim=1, keepdim=True) + 1e-8)

        right_env = torch.zeros(bsz, 1, device=x.device)
        right_env[:, 0] = 1.0
        for i in range(len(self.right_tensors) - 1, -1, -1):
            a = self.right_tensors[i]
            pixel_idx = self.split_idx + i
            right_env = torch.einsum("br,lpr,bp->bl", right_env, a, phi[:, pixel_idx])
            right_env = right_env / (right_env.norm(dim=1, keepdim=True) + 1e-8)

        out = torch.einsum("bl,br,lro->bo", left_env, right_env, self.center_tensor)
        return F.normalize(out, p=2, dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_feature_map(x: torch.Tensor) -> torch.Tensor:
    angle = (torch.pi / 2.0) * x
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def quantum_embedding_ops(x: torch.Tensor, n_qubits: int) -> None:
    wires = list(reversed(range(n_qubits)))
    for w in wires:
        qml.Hadamard(wires=w)
    for idx, w in enumerate(wires):
        qml.RZ(2.0 * x[..., idx], wires=w)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            a, b = wires[i], wires[j]
            theta = 2.0 * (math.pi - x[..., i]) * (math.pi - x[..., j])
            qml.CNOT(wires=[a, b])
            qml.PhaseShift(theta, wires=b)
            qml.CNOT(wires=[a, b])


def quantum_embedding_adjoint_ops(x: torch.Tensor, n_qubits: int) -> None:
    wires = list(reversed(range(n_qubits)))
    pairs = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            a, b = wires[i], wires[j]
            theta = 2.0 * (math.pi - x[..., i]) * (math.pi - x[..., j])
            pairs.append((a, b, theta))

    for a, b, theta in reversed(pairs):
        qml.CNOT(wires=[a, b])
        qml.PhaseShift(-theta, wires=b)
        qml.CNOT(wires=[a, b])

    for idx, w in reversed(list(enumerate(wires))):
        qml.RZ(-2.0 * x[..., idx], wires=w)

    for w in reversed(wires):
        qml.Hadamard(wires=w)


def build_fidelity_qlayer(n_qubits: int) -> nn.Module:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def fidelity_circuit(inputs: torch.Tensor):
        v1 = inputs[..., :n_qubits]
        v2 = inputs[..., n_qubits : 2 * n_qubits]
        quantum_embedding_ops(v1, n_qubits)
        quantum_embedding_adjoint_ops(v2, n_qubits)
        return qml.expval(qml.Projector([0] * n_qubits, wires=range(n_qubits)))

    return qml.qnn.TorchLayer(fidelity_circuit, weight_shapes={}).to("cpu")


class FidelityMPSModel(nn.Module):
    def __init__(self, input_dim: int, bond_dim: int, n_qubits: int, qlayer: nn.Module):
        super().__init__()
        self.encoder = MPSMetricLearnerEmbedCenter(
            length=input_dim,
            bond_dim=bond_dim,
            output_dim=n_qubits,
            d_phys=2,
        )
        self.qlayer = qlayer

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z = torch.cat([z1, z2], dim=1)
        y_cpu = self.qlayer(z.to("cpu"))
        return y_cpu.to(z.device)


def class_index_map(y: torch.Tensor) -> Dict[int, torch.Tensor]:
    return {int(c): torch.where(y == c)[0] for c in y.unique().tolist()}


def new_pair_batch_balanced(
    batch_size: int,
    x: torch.Tensor,
    y: torch.Tensor,
    class_to_idx: Dict[int, torch.Tensor],
    classes: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch_size % 2 != 0:
        raise ValueError("batch_size must be even for balanced pair sampling")

    half = batch_size // 2
    pos_1, pos_2 = [], []
    neg_1, neg_2 = [], []

    for _ in range(half):
        c = random.choice(classes)
        idxs = class_to_idx[c]
        picks = torch.randint(0, idxs.numel(), (2,))
        pos_1.append(x[idxs[picks[0]]])
        pos_2.append(x[idxs[picks[1]]])

    for _ in range(half):
        c1, c2 = random.sample(classes, 2)
        i1 = class_to_idx[c1][torch.randint(0, class_to_idx[c1].numel(), (1,)).item()]
        i2 = class_to_idx[c2][torch.randint(0, class_to_idx[c2].numel(), (1,)).item()]
        neg_1.append(x[i1])
        neg_2.append(x[i2])

    x1 = torch.stack(pos_1 + neg_1, dim=0)
    x2 = torch.stack(pos_2 + neg_2, dim=0)
    ypair = torch.cat([torch.ones(half), torch.zeros(half)]).to(torch.float32)
    return x1, x2, ypair


def contrastive_loss(scores: torch.Tensor, targets: torch.Tensor, margin: float) -> torch.Tensor:
    d = 1.0 - scores
    pos = targets * (d**2)
    neg = (1.0 - targets) * F.relu(margin - d) ** 2
    return (pos + neg).mean()


def train_pairwise_contrastive(
    model: nn.Module,
    x_train_flat: torch.Tensor,
    y_train: torch.Tensor,
    cfg: SweepConfig,
    run_name: str = "",
) -> List[float]:
    model.to(cfg.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    classes = [int(c) for c in y_train.unique().tolist()]
    class_to_idx = class_index_map(y_train)
    num_batches = x_train_flat.shape[0] // cfg.batch_size

    history = []
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        for _ in range(num_batches):
            x1, x2, ypair = new_pair_batch_balanced(
                cfg.batch_size,
                x_train_flat,
                y_train,
                class_to_idx,
                classes,
            )
            x1, x2, ypair = x1.to(cfg.device), x2.to(cfg.device), ypair.to(cfg.device)
            scores = model(x1, x2)
            loss = contrastive_loss(scores, ypair, margin=cfg.margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, num_batches)
        history.append(avg_loss)

        # Lightweight heartbeat logging for long remote jobs.
        should_print = (
            epoch == 0
            or epoch == cfg.num_epochs - 1
            or ((epoch + 1) % max(1, cfg.num_epochs // 4) == 0)
        )
        if should_print:
            prefix = f"[{run_name}] " if run_name else ""
            print(f"{prefix}epoch {epoch + 1}/{cfg.num_epochs} loss={avg_loss:.4f}", flush=True)

    return history


def compute_embeddings(
    encoder: nn.Module,
    x_flat: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    z_all = []
    y_all = []
    with torch.no_grad():
        n = x_flat.shape[0]
        for start in range(0, n, batch_size):
            xb = x_flat[start : start + batch_size].to(device)
            yb = y[start : start + batch_size]
            z = encoder(xb)
            z_all.append(F.normalize(z, p=2, dim=1).cpu())
            y_all.append(yb.cpu())
    return torch.cat(z_all, dim=0), torch.cat(y_all, dim=0)


def knn_accuracy(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_test: torch.Tensor,
    y_test: torch.Tensor,
    k: int,
) -> float:
    with torch.no_grad():
        sim = z_test @ z_train.T
        idx = sim.topk(k, dim=1, largest=True).indices
        preds = torch.mode(y_train[idx], dim=1).values
        return (preds == y_test).float().mean().item()


def load_mnist_flat(root: str, max_train: int, max_test: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    x_train = train_set.data.float().unsqueeze(1) / 255.0
    y_train = train_set.targets.long()
    x_test = test_set.data.float().unsqueeze(1) / 255.0
    y_test = test_set.targets.long()

    if max_train > 0:
        x_train = x_train[:max_train]
        y_train = y_train[:max_train]
    if max_test > 0:
        x_test = x_test[:max_test]
        y_test = y_test[:max_test]

    x_train_flat = x_train.flatten(1)
    x_test_flat = x_test.flatten(1)
    return x_train_flat, y_train, x_test_flat, y_test


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_sweep(
    cfg: SweepConfig,
    bond_dims: Sequence[int] = DEFAULT_BOND_DIMS,
    n_qubits_list: Sequence[int] = DEFAULT_N_QUBITS,
) -> None:
    out_dir = Path(cfg.out_dir)
    logs_dir = out_dir / "results" / "logs"
    metrics_dir = out_dir / "results" / "metrics"
    models_dir = out_dir / "results" / "models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)

    print("Loading MNIST dataset...", flush=True)
    x_train_flat, y_train, x_test_flat, y_test = load_mnist_flat(cfg.data_root, cfg.max_train, cfg.max_test)
    print(
        f"Dataset ready: train={x_train_flat.shape[0]} samples, test={x_test_flat.shape[0]} samples",
        flush=True,
    )

    summary_csv = metrics_dir / "knn_sweep_results.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "n_qubits",
                "bond_dim",
                "train_samples",
                "test_samples",
                "num_epochs",
                "batch_size",
                "knn_k",
                "knn_accuracy",
                "final_train_loss",
            ],
        )
        writer.writeheader()

        total_runs = len(n_qubits_list) * len(bond_dims)
        run_idx = 0
        for n_qubits in n_qubits_list:
            for bond_dim in bond_dims:
                run_idx += 1
                run_name = f"nq{n_qubits}_bond{bond_dim}"
                print(f"\\n=== Running {run_name} ({run_idx}/{total_runs}) ===", flush=True)

                qlayer = build_fidelity_qlayer(n_qubits)
                model = FidelityMPSModel(
                    input_dim=x_train_flat.shape[1],
                    bond_dim=bond_dim,
                    n_qubits=n_qubits,
                    qlayer=qlayer,
                )

                loss_hist = train_pairwise_contrastive(model, x_train_flat, y_train, cfg, run_name=run_name)

                z_train, y_train_emb = compute_embeddings(
                    model.encoder,
                    x_train_flat,
                    y_train,
                    batch_size=cfg.batch_size,
                    device=cfg.device,
                )
                z_test, y_test_emb = compute_embeddings(
                    model.encoder,
                    x_test_flat,
                    y_test,
                    batch_size=cfg.batch_size,
                    device=cfg.device,
                )
                acc = knn_accuracy(z_train, y_train_emb, z_test, y_test_emb, cfg.knn_k)

                row = {
                    "n_qubits": n_qubits,
                    "bond_dim": bond_dim,
                    "train_samples": int(x_train_flat.shape[0]),
                    "test_samples": int(x_test_flat.shape[0]),
                    "num_epochs": cfg.num_epochs,
                    "batch_size": cfg.batch_size,
                    "knn_k": cfg.knn_k,
                    "knn_accuracy": float(acc),
                    "final_train_loss": float(loss_hist[-1] if len(loss_hist) > 0 else float("nan")),
                }
                writer.writerow(row)
                f_csv.flush()

                run_log = {
                    "run_name": run_name,
                    "n_qubits": n_qubits,
                    "bond_dim": bond_dim,
                    "loss_history": loss_hist,
                    "knn_accuracy": float(acc),
                }
                save_json(logs_dir / f"{run_name}.json", run_log)

                torch.save(
                    {
                        "encoder_state_dict": model.encoder.state_dict(),
                        "n_qubits": n_qubits,
                        "bond_dim": bond_dim,
                        "knn_accuracy": float(acc),
                    },
                    models_dir / f"{run_name}.pt",
                )

                print(f"{run_name}: kNN accuracy={acc * 100:.2f}%", flush=True)

    print(f"\\nSweep complete. Results saved in: {out_dir.resolve()}", flush=True)


def parse_args() -> SweepConfig:
    parser = argparse.ArgumentParser(description="MNIST MPS kNN sweep over bond_dim and n_qubits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-train", type=int, default=12000, help="0 uses full training set")
    parser.add_argument("--max-test", type=int, default=2000, help="0 uses full test set")
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    return SweepConfig(
        seed=args.seed,
        data_root=args.data_root,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        margin=args.margin,
        knn_k=args.knn_k,
        device=args.device,
        max_train=args.max_train,
        max_test=args.max_test,
        out_dir=args.out_dir,
    )


def main() -> None:
    cfg = parse_args()
    run_sweep(cfg)


if __name__ == "__main__":
    main()
