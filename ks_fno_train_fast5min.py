import os
import time
import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class KSH5PreloadOneStep(Dataset):
    def __init__(self, h5_path: str, split: str, max_traj: int | None = None):
        with h5py.File(h5_path, "r") as f:
            grp = f[split]
            pde_key = [k for k in grp.keys() if k.startswith("pde_")][0]
            dset = grp[pde_key]
            if max_traj is None:
                u = dset[:]
            else:
                u = dset[:max_traj]
        # Keep dataset on disk as-is; cast to float32 for compute.
        if u.dtype != np.float32:
            u = u.astype(np.float32)
        self.u = u
        self.N, self.T, self.X = self.u.shape
        self.idxs = [(i, t) for i in range(self.N) for t in range(self.T - 1)]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i, t = self.idxs[idx]
        x = self.u[i, t][None, :]
        y = self.u[i, t + 1][None, :]
        return torch.from_numpy(x), torch.from_numpy(y)


class SpectralConv1d(nn.Module):
    """Spectral conv with real-valued parameters (stable on MPS).

    We keep Fourier weights as two real tensors (Wr, Wi) and do complex
    multiplication in real arithmetic.
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.Wr = nn.Parameter(scale * torch.randn(out_channels, in_channels, modes, dtype=torch.float32))
        self.Wi = nn.Parameter(scale * torch.randn(out_channels, in_channels, modes, dtype=torch.float32))

    def forward(self, x):
        # x: [B,C,X] real
        B, C, X = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # complex tensor
        out_ft = torch.zeros(B, self.Wr.shape[0], x_ft.shape[-1], device=x.device, dtype=x_ft.dtype)
        m = min(self.modes, x_ft.shape[-1])
        # real/imag parts
        xr = x_ft[:, :, :m].real
        xi = x_ft[:, :, :m].imag
        # (xr+i xi) * (Wr+i Wi)
        # real: xr*Wr - xi*Wi
        # imag: xr*Wi + xi*Wr
        or_ = torch.einsum('bim,oim->bom', xr, self.Wr[:, :, :m]) - torch.einsum('bim,oim->bom', xi, self.Wi[:, :, :m])
        oi_ = torch.einsum('bim,oim->bom', xr, self.Wi[:, :, :m]) + torch.einsum('bim,oim->bom', xi, self.Wr[:, :, :m])
        out_ft[:, :, :m] = torch.complex(or_, oi_)
        return torch.fft.irfft(out_ft, n=X, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64, depth=4):
        super().__init__()
        self.fc0 = nn.Conv1d(2, width, 1)
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.w = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Conv1d(width, 128, 1)
        self.fc2 = nn.Conv1d(128, 1, 1)

    def forward(self, u):
        B, _, X = u.shape
        grid = torch.linspace(0, 1, X, device=u.device, dtype=u.dtype)[None, None, :].repeat(B, 1, 1)
        x = torch.cat([u, grid], dim=1)
        x = self.fc0(x)
        for s, w in zip(self.spec, self.w):
            x = F.gelu(s(x) + w(x))
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


def radial_spectrum_1d(u: torch.Tensor):
    if u.ndim == 1:
        u = u[None, :]
    U = torch.fft.rfft(u, dim=-1)
    E = (U.real ** 2 + U.imag ** 2).mean(dim=0)
    k = torch.arange(E.shape[0], device=E.device)
    return k.cpu().numpy(), E.cpu().numpy()


@torch.no_grad()
def eval_spectrum_one_step(model, loader, device, outdir):
    model.eval()
    xs, ys, ps = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=False, dtype=torch.float32)
        yb = yb.to(device, non_blocking=False, dtype=torch.float32)
        pb = model(xb)
        xs.append(xb[:, 0, :].detach().cpu())
        ys.append(yb[:, 0, :].detach().cpu())
        ps.append(pb[:, 0, :].detach().cpu())
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    p = torch.cat(ps, dim=0)

    k, Ey = radial_spectrum_1d(y)
    _, Ep = radial_spectrum_1d(p)
    _, Ex = radial_spectrum_1d(x)

    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "spectrum.npz"), k=k, E_in=Ex, E_true=Ey, E_pred=Ep)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.loglog(k[1:], Ey[1:], label="true")
    plt.loglog(k[1:], Ep[1:], label="pred")
    plt.loglog(k[1:], Ex[1:], label="input")
    plt.xlabel("wavenumber k")
    plt.ylabel("power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spectrum.png"), dpi=200)
    plt.close()


@torch.no_grad()
def mean_mse(model, loader, device):
    model.eval()
    loss_sum = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=False, dtype=torch.float32)
        yb = yb.to(device, non_blocking=False, dtype=torch.float32)
        pb = model(xb)
        loss_sum += F.mse_loss(pb, yb, reduction="sum").item()
        n += yb.numel()
    return loss_sum / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--timeout_sec", type=float, default=300.0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_traj", type=int, default=0, help="0 means all trajectories; else load only first K")
    ap.add_argument("--outdir", default="outputs_fast5min")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="Training device")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    args = ap.parse_args()

    set_seed(args.seed)
    # Use all 8 CPU cores (user request)
    torch.set_num_threads(8)
    # Also respect common BLAS thread env vars if set externally.
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")

    t_start = time.time()

    train_path = os.path.join(args.data_dir, "KS_train_fixed_viscosity.h5")
    valid_path = os.path.join(args.data_dir, "KS_valid_fixed_viscosity.h5")
    test_path = os.path.join(args.data_dir, "KS_test_fixed_viscosity.h5")

    max_traj = None if args.max_train_traj == 0 else args.max_train_traj

    train_ds = KSH5PreloadOneStep(train_path, "train", max_traj=max_traj)
    valid_ds = KSH5PreloadOneStep(valid_path, "valid")
    test_ds = KSH5PreloadOneStep(test_path, "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("mps" if (args.device == "mps" and torch.backends.mps.is_available()) else "cpu")
    if args.device == "mps" and device.type != "mps":
        print("WARNING: --device mps requested but MPS is not available; falling back to CPU")

    model = FNO1d(modes=args.modes, width=args.width, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Optionally resume from a checkpoint created by this script
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"Resumed weights from {args.resume}")

    os.makedirs(args.outdir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        for it, (xb, yb) in enumerate(train_loader):
            if time.time() - t_start > args.timeout_sec:
                print(f"TIMEOUT after {args.timeout_sec}s during training (ep={ep}, it={it}). Exiting.")
                return
            xb = xb.to(device, non_blocking=False, dtype=torch.float32)
            yb = yb.to(device, non_blocking=False, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pb = model(xb)
            loss = F.mse_loss(pb, yb)
            loss.backward()
            opt.step()

        # Print immediately after finishing the epoch
        v = mean_mse(model, valid_loader, device)
        dt = time.time() - t_ep
        print(f"epoch {ep}/{args.epochs} ({dt:.2f}s) valid_mse={v:.6e}", flush=True)

    t = mean_mse(model, test_loader, device)
    print(f"DONE in {time.time()-t_start:.2f}s  test_mse={t:.6e}")

    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.outdir, "fno1d.pt"))
    eval_spectrum_one_step(model, test_loader, device, args.outdir)


if __name__ == "__main__":
    main()
