import os
import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: This script uses PyTorch `torch.load()` for checkpoints created by our
# own training scripts. If you don't trust the checkpoint file, don't load it.


class SpectralConv1d(nn.Module):
    """Spectral conv matching `ks_fno_train_fast5min.py` (real weights Wr/Wi)."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.Wr = nn.Parameter(scale * torch.randn(out_channels, in_channels, modes, dtype=torch.float32))
        self.Wi = nn.Parameter(scale * torch.randn(out_channels, in_channels, modes, dtype=torch.float32))

    def forward(self, x):
        B, C, X = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], device=x.device, dtype=x_ft.dtype)
        m = min(self.modes, x_ft.shape[-1])
        xr = x_ft[:, :, :m].real
        xi = x_ft[:, :, :m].imag
        or_ = torch.einsum('bim,oim->bom', xr, self.Wr[:, :, :m]) - torch.einsum('bim,oim->bom', xi, self.Wi[:, :, :m])
        oi_ = torch.einsum('bim,oim->bom', xr, self.Wi[:, :, :m]) + torch.einsum('bim,oim->bom', xi, self.Wr[:, :, :m])
        out_ft[:, :, :m] = torch.complex(or_, oi_)
        return torch.fft.irfft(out_ft, n=X, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.w(x))


class FNO1d(nn.Module):
    """FNO1d matching the checkpoint format from `ks_fno_train_fast5min.py`.

    (Uses `spec` and `w` ModuleLists instead of nested blocks.)
    """

    def __init__(self, modes: int = 16, width: int = 64, depth: int = 4):
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


def load_trajectory(h5_path: str, split: str, traj_idx: int = 0, time_step: int = 1, subsample_space: int = 1):
    with h5py.File(h5_path, "r") as f:
        grp = f[split]
        pde_key = [k for k in grp.keys() if k.startswith("pde_")][0]
        u = grp[pde_key][traj_idx]  # [T, X]
        xgrid = grp["x"][:]
        tgrid = grp["t"][:]

    if time_step > 1:
        u = u[::time_step]
        tgrid = tgrid[::time_step]
    if subsample_space > 1:
        u = u[:, ::subsample_space]
        xgrid = xgrid[::subsample_space]

    return u.astype(np.float32), xgrid.astype(np.float32), tgrid.astype(np.float32)


def power_spectrum_1d(u: np.ndarray):
    """u: [X]"""
    U = np.fft.rfft(u)
    E = (U.real ** 2 + U.imag ** 2)
    k = np.arange(E.shape[0])
    return k, E


@torch.no_grad()
def rollout_one_step(model, u0: np.ndarray, steps: int, device: torch.device):
    """Autoregressive rollout using the one-step model.

    u0: [X]
    returns pred trajectory: [steps+1, X] including initial
    """
    X = u0.shape[0]
    out = np.zeros((steps + 1, X), dtype=np.float32)
    out[0] = u0
    cur = torch.from_numpy(u0[None, None, :]).to(device)
    for s in range(steps):
        nxt = model(cur)
        out[s + 1] = nxt[0, 0].detach().cpu().numpy()
        cur = nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to fno1d.pt")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--variant", type=str, default="fixed", choices=["fixed", "conditional"])
    ap.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    ap.add_argument("--traj_idx", type=int, default=0)
    ap.add_argument("--time_step", type=int, default=1)
    ap.add_argument("--subsample_space", type=int, default=1)
    ap.add_argument("--rollout_steps", type=int, default=50, help="How many autoregressive steps")
    ap.add_argument("--plot_times", type=str, default="0,1,2,5,10,20,50", help="Comma-separated indices")
    ap.add_argument("--outdir", type=str, default="outputs_fno_plots")
    args = ap.parse_args()

    # Load checkpoint args to reconstruct model
    # Use weights_only=True to avoid arbitrary code execution during checkpoint loading.
    # This requires PyTorch >= 2.0.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    margs = ckpt.get("args", {})
    modes = int(margs.get("modes", 16))
    width = int(margs.get("width", 64))
    depth = int(margs.get("depth", 4))

    # CPU-only (user requested)
    device = torch.device("cpu")
    try:
        torch.set_num_threads(8)
    except Exception:
        pass
    model = FNO1d(modes=modes, width=width, depth=depth).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.variant == "fixed":
        fn = f"KS_{args.split}_fixed_viscosity.h5"
    else:
        fn = f"KS_{args.split}_conditional_viscosity.h5"

    h5_path = os.path.join(args.data_dir, fn)
    u_true, xgrid, tgrid = load_trajectory(
        h5_path, args.split, args.traj_idx, time_step=args.time_step, subsample_space=args.subsample_space
    )

    # Some files store x/t as [N, X] / [N, T]. If so, select the same trajectory.
    if xgrid.ndim == 2:
        xgrid = xgrid[args.traj_idx]
    if tgrid.ndim == 2:
        tgrid = tgrid[args.traj_idx]

    steps = min(args.rollout_steps, u_true.shape[0] - 1)
    u_pred = rollout_one_step(model, u_true[0], steps=steps, device=device)

    os.makedirs(args.outdir, exist_ok=True)

    # Save arrays
    np.savez(
        os.path.join(args.outdir, f"traj_{args.split}_idx{args.traj_idx}.npz"),
        x=xgrid,
        t=tgrid[: steps + 1],
        u_true=u_true[: steps + 1],
        u_pred=u_pred,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Profiles at selected times
    plot_times = [int(s) for s in args.plot_times.split(",") if s.strip()]
    plot_times = [tt for tt in plot_times if 0 <= tt <= steps]

    for tt in plot_times:
        plt.figure(figsize=(7, 3))
        plt.plot(xgrid, u_true[tt], label=f"true t_idx={tt}")
        plt.plot(xgrid, u_pred[tt], label=f"pred t_idx={tt}", linestyle="--")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"KS profile (traj {args.traj_idx}) at time index {tt}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"profile_t{tt:04d}.png"), dpi=200)
        plt.close()

    # Space-time heatmaps: true, pred, and error
    u_true_plot = u_true[: steps + 1]
    u_pred_plot = u_pred
    err = (u_pred_plot - u_true_plot)

    def save_heatmap(arr, fname, title, cmap="viridis", symmetric=False):
        plt.figure(figsize=(7, 4))
        if symmetric:
            vmax = float(np.max(np.abs(arr))) + 1e-12
            vmin = -vmax
        else:
            vmin = None
            vmax = None
        im = plt.imshow(arr, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im)
        plt.xlabel("space index")
        plt.ylabel("time index")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=200)
        plt.close()

    save_heatmap(u_true_plot, "true_heatmap.png", f"True trajectory (traj {args.traj_idx})", cmap="viridis")
    save_heatmap(u_pred_plot, "pred_heatmap.png", f"Predicted trajectory (traj {args.traj_idx})", cmap="viridis")
    save_heatmap(err, "error_heatmap.png", f"Autoregressive rollout error (pred-true), traj {args.traj_idx}", cmap="coolwarm", symmetric=True)

    # Spectra at selected times
    for tt in plot_times:
        k, Etrue = power_spectrum_1d(u_true[tt])
        _, Epred = power_spectrum_1d(u_pred[tt])
        plt.figure(figsize=(6, 4))
        plt.loglog(k[1:], Etrue[1:] + 1e-30, label="true")
        plt.loglog(k[1:], Epred[1:] + 1e-30, label="pred")
        plt.xlabel("wavenumber k")
        plt.ylabel("power")
        plt.title(f"Spectrum at time index {tt}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"spectrum_t{tt:04d}.png"), dpi=200)
        plt.close()

    # Aggregate spectrum over time
    Etrue_acc = None
    Epred_acc = None
    for tt in range(steps + 1):
        k, Etrue = power_spectrum_1d(u_true[tt])
        _, Epred = power_spectrum_1d(u_pred[tt])
        if Etrue_acc is None:
            Etrue_acc = Etrue
            Epred_acc = Epred
        else:
            Etrue_acc += Etrue
            Epred_acc += Epred
    Etrue_acc /= (steps + 1)
    Epred_acc /= (steps + 1)

    plt.figure(figsize=(6, 4))
    plt.loglog(k[1:], Etrue_acc[1:] + 1e-30, label="true (time-avg)")
    plt.loglog(k[1:], Epred_acc[1:] + 1e-30, label="pred (time-avg)")
    plt.xlabel("wavenumber k")
    plt.ylabel("power")
    plt.title("Time-averaged spectrum over rollout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "spectrum_timeavg.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
