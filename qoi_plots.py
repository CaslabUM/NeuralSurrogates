import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('outputs_mps')
traj_dirs = {
    0: base,
}
outdir = Path('./')
outdir.mkdir(exist_ok=True)

# Generate per-trajectory QoI plots (true vs pred) and relerr plots.

def qois(u):
    ux = (np.roll(u,-1,axis=1) - np.roll(u,1,axis=1))/2.0
    uxx = np.roll(u,-1,axis=1) - 2*u + np.roll(u,1,axis=1)
    return {
        'E': np.mean(u**2, axis=1),
        'G': np.mean(ux**2, axis=1),
        'D2': np.mean(uxx**2, axis=1),
        'mean': np.mean(u, axis=1),
        'var': np.var(u, axis=1),
    }

qoi_keys=[('E','mean(u^2)',True),('G','mean((du/dx)^2)',True),('D2','mean((d2u/dx2)^2)',True),('mean','mean(u)',False),('var','var(u)',True)]

for idx, d in traj_dirs.items():
    npz=np.load(d/f'traj_test_idx{idx}.npz')
    u_true=npz['u_true']; u_pred=npz['u_pred']
    qt=qois(u_true); qp=qois(u_pred)

    # combined QoI plot for this trajectory
    fig, axs = plt.subplots(2,3, figsize=(10,6))
    axs=axs.ravel()
    for j,(key,ylabel,logy) in enumerate(qoi_keys):
        ax=axs[j]
        ax.plot(qt[key], label='true', linewidth=2)
        ax.plot(qp[key], label='pred', linestyle='--')
        if logy:
            ax.set_yscale('log')
        ax.set_title(key)
        ax.set_xlabel('t')
        ax.set_ylabel(ylabel)
    axs[-1].axis('off')
    axs[0].legend(fontsize=8)
    fig.suptitle(f'QoIs (traj {idx}, 200-step rollout)')
    fig.tight_layout()
    fig.savefig(outdir/f'qoi_all_traj{idx}.png', dpi=200)
    plt.close(fig)

    # relative error plot for this trajectory
    fig, axs = plt.subplots(2,3, figsize=(10,6))
    axs=axs.ravel()
    for j,(key,ylabel,logy) in enumerate(qoi_keys):
        ax=axs[j]
        denom=np.maximum(np.abs(qt[key]), 1e-30)
        rel=np.abs(qp[key]-qt[key])/denom
        ax.plot(rel, color='tab:red')
        ax.set_yscale('log')
        ax.set_title(f'relerr {key}')
        ax.set_xlabel('t')
        ax.set_ylabel('relative error')
    axs[-1].axis('off')
    fig.suptitle(f'QoI relative errors (traj {idx}, 200-step rollout)')
    fig.tight_layout()
    fig.savefig(outdir/f'qoi_relerr_all_traj{idx}.png', dpi=200)
    plt.close(fig)

print('Wrote per-trajectory QoI figures to', outdir)

