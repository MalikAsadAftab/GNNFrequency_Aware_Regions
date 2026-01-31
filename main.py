#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat, savemat

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import SAGEConv, GATv2Conv, AttentionalAggregation


# -----------------------------
# MUSIC helper functions
# -----------------------------
def find_specified_peaks(phi, PS, L):
    phi_in = phi[(phi >= -np.pi / 2) & (phi <= np.pi / 2)]
    PS_in = PS[(phi >= -np.pi / 2) & (phi <= np.pi / 2)]
    peaks, _ = signal.find_peaks(np.abs(PS_in))
    if len(peaks) == 0:
        return np.array([phi_in[np.argmax(np.abs(PS_in))]])
    sorted_peaks = peaks[np.argsort(np.abs(PS_in)[peaks])]
    return phi_in[sorted_peaks[-L:]]


def music(a, R, phi, L):
    PS = np.zeros(a.shape[1], dtype=complex)
    D, Q = np.linalg.eigh(R)
    Qn = Q[:, 0: R.shape[0] - L]
    Rn = Qn @ np.conj(Qn.T)
    for t in range(len(phi)):
        a_tmp = np.reshape(a[:, t], (-1, 1))
        PS[t] = 1.0 / (np.conjugate(a_tmp.T) @ Rn @ a_tmp).item()
    return PS, find_specified_peaks(phi, PS, L)


# --------------------------------------
# Covariance utilities
# --------------------------------------
def cov_to_feature(R: np.ndarray, trace_norm: bool = True) -> np.ndarray:
    """Convert covariance matrix to [Re, Im] flattened feature."""
    if trace_norm:
        tr = np.trace(R)
        if np.abs(tr) > 1e-12:
            R = R / tr
    feat = np.concatenate([np.real(R).reshape(-1), np.imag(R).reshape(-1)], axis=0)
    return feat.astype(np.float32)


def cov_avg_over_subcarriers_list(CSI: np.ndarray, ant_start: int, M: int, sc_list,
                                  start_idx: int, n_samples: int) -> np.ndarray:
    """Average covariance across explicit subcarrier indices in sc_list."""
    R_sum = np.zeros((M, M), dtype=np.complex128)
    for sc in sc_list:
        x = np.squeeze(CSI[ant_start: ant_start + M, sc, start_idx: start_idx + n_samples])  # (M,n_samples)
        R_sum += (x @ np.conjugate(x.T)) / (x.shape[1] + 1e-9)
    return R_sum / max(1, len(sc_list))


# --------------------------------------
# Teacher MUSIC (list-groups, cov-avg)
# --------------------------------------
def music_teacher_grouped_covavg_listgroups(CSI, ant_start, M, groups_sc, start_idx, n_samples,
                                            phi, L, lam, d, agg="mean"):
    """
    For each group (list of subcarrier indices):
      - compute R_band = average_{sc in group} cov(sc)
      - compute PS_band(phi) with MUSIC
    Aggregate PS across groups, then argmax => label.
    """
    k_ant = np.arange(M).reshape((-1, 1))
    a = np.exp(-1j * 2 * np.pi / lam * d * np.outer(k_ant, np.sin(phi)))  # (M, |phi|)

    PS_list = []
    t0 = time.time()
    for sc_list in groups_sc:
        R_band = cov_avg_over_subcarriers_list(CSI, ant_start, M, sc_list, start_idx, n_samples)
        PS_band, _ = music(a, R_band, phi, L)
        PS_list.append(np.abs(PS_band))
    t1 = time.time()

    PS_stack = np.stack(PS_list, axis=0)
    PS_agg = np.mean(PS_stack, axis=0) if agg == "mean" else np.median(PS_stack, axis=0)

    mask = (phi >= -np.pi / 2) & (phi <= np.pi / 2)
    doa_rad = phi[mask][np.argmax(PS_agg[mask])]
    doa_deg = float(np.clip(np.degrees(doa_rad), -90.0, 90.0))
    return PS_agg, doa_deg, (t1 - t0)


# --------------------------------------
# Student graph builder (list-groups, cov-avg)
# --------------------------------------
def build_graph_slice_selected_covavg(CSI, ant_start, M, groups_sc, start_idx, n_samples,
                                      topk=0, trace_norm=True):
    """
    Nodes: groups_sc (each group is a list of subcarrier indices)
    Node features: cov-avg over subcarriers in the group (Re/Im)
    Edges: chain adjacency + optional top-k cosine similarity (computed from node features)
    """
    node_feats = []
    for sc_list in groups_sc:
        R = cov_avg_over_subcarriers_list(CSI, ant_start, M, sc_list, start_idx, n_samples)
        node_feats.append(cov_to_feature(R, trace_norm=trace_norm))

    x_nodes = np.stack(node_feats, axis=0)  # (N,F)
    N = x_nodes.shape[0]

    edges = []
    for i in range(N - 1):
        edges += [(i, i + 1), (i + 1, i)]

    if topk and N > 1:
        Xn = x_nodes / (np.linalg.norm(x_nodes, axis=1, keepdims=True) + 1e-9)
        sim = Xn @ Xn.T
        for i in range(N):
            sim[i, i] = -1.0
            nn_idx = np.argsort(sim[i])[-topk:]
            for j in nn_idx:
                edges += [(i, int(j)), (int(j), i)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.tensor(x_nodes, dtype=torch.float32), edge_index


# -----------------------
# GNN model (AoA sin/cos)
# -----------------------
class AoAGNN(nn.Module):
    def __init__(self, in_dim, hidden=128, use_gat=True):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if use_gat:
            self.conv1 = GATv2Conv(hidden, hidden, heads=2, concat=False)
            self.conv2 = GATv2Conv(hidden, hidden, heads=2, concat=False)
        else:
            self.conv1 = SAGEConv(hidden, hidden)
            self.conv2 = SAGEConv(hidden, hidden)

        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.enc(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = self.pool(x, batch)
        out = self.head(g)
        out = F.normalize(out, dim=1)
        return out


def doa_deg_to_vec(doa_deg: float):
    rad = np.deg2rad(doa_deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)


def vec_to_doa_deg(vec2: np.ndarray):
    rad = np.arctan2(vec2[..., 1], vec2[..., 0])
    deg = np.rad2deg(rad)
    return np.clip(deg, -90.0, 90.0)


def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    return 1.0 - torch.sum(pred * target, dim=1).mean()


# -----------------------
# Dataset loader
# -----------------------
def load_nokia_track_csi(dataset_dir: str, track: int, ant_max: int = 64, sc_max: int = 50):
    """
    Supports:
      1) dataset_dir/t{track}/*.mat
      2) dataset_dir/t{track}_*.mat
    Expects key 'Hd_all' in each .mat file.
    Returns CSI: complex np.ndarray [ant_max, sc_max, T_concat]
    """
    track_dir = os.path.join(dataset_dir, f"t{track}")
    if os.path.isdir(track_dir):
        mat_files = sorted(glob.glob(os.path.join(track_dir, "*.mat")))
    else:
        mat_files = sorted(glob.glob(os.path.join(dataset_dir, f"t{track}_*.mat")))

    if len(mat_files) == 0:
        raise FileNotFoundError(
            f"No .mat files found for track={track}.\n"
            f"Tried:\n  {track_dir}/*.mat\n  {dataset_dir}/t{track}_*.mat"
        )

    CSI_list = []
    for fp in mat_files:
        mat = loadmat(fp)
        if "Hd_all" not in mat:
            raise KeyError(f"'Hd_all' not found in {fp}. Keys={list(mat.keys())}")
        CSI_list.append(mat["Hd_all"][0:ant_max, 0:sc_max, :])

    CSI = np.concatenate(CSI_list, axis=2)
    return CSI, mat_files


# -----------------------
# Selection: correlation, fim, d-opt
# -----------------------
def parse_int_list(s: str):
    """Parse comma-separated int list like '10,20,30'."""
    if s is None or len(s.strip()) == 0:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def corr_similarity_matrix(X: np.ndarray, use_abs: bool = True) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    if use_abs:
        sim = np.abs(sim)
    np.fill_diagonal(sim, 0.0)
    return sim


def compute_subcarrier_signatures_trainonly(
    CSI: np.ndarray,
    ant_start: int,
    M: int,
    n_samples: int,
    interval: int,
    start_sample: int,
    max_train_slices: int = 200,
    trace_norm: bool = True
) -> np.ndarray:
    """
    Signature per subcarrier = average over training slices of cov_to_feature(R_k(slice)).
    Returns: (K,F) float32
    """
    K = CSI.shape[1]
    T = CSI.shape[2]
    total_slices = (T - n_samples) // interval
    if total_slices <= 0:
        raise RuntimeError(f"Not enough CSI length={T} for window_size={n_samples}, stride={interval}")

    train_g = [g for g in range(total_slices) if (g * interval) < start_sample]
    if len(train_g) == 0:
        raise RuntimeError(f"No training slices for selection (start_sample={start_sample}).")

    use_slices = train_g[:min(len(train_g), max_train_slices)]
    sig_sum = None
    count = 0

    for g in use_slices:
        start_idx = g * interval
        feats = []
        for k in range(K):
            x = np.squeeze(CSI[ant_start: ant_start + M, k, start_idx: start_idx + n_samples])
            R = (x @ np.conjugate(x.T)) / (x.shape[1] + 1e-9)
            feats.append(cov_to_feature(R, trace_norm=trace_norm))
        feats = np.stack(feats, axis=0)  # (K,F)
        sig_sum = feats.astype(np.float64) if sig_sum is None else (sig_sum + feats.astype(np.float64))
        count += 1

    return (sig_sum / max(1, count)).astype(np.float32)


def select_corr_centrality(sim: np.ndarray, n_select: int, top_p: int = 10):
    """Score(k)=mean(top_p(sim[k,*])) ; pick top n_select."""
    K = sim.shape[0]
    top_p = min(top_p, K - 1)
    scores = np.zeros(K, dtype=np.float32)
    for k in range(K):
        row = sim[k]
        scores[k] = float(np.mean(np.sort(row)[-top_p:])) if top_p > 0 else 0.0
    ranked = np.argsort(scores)[::-1]
    return ranked[:n_select].tolist(), scores


def greedy_group_by_similarity(selected, sim: np.ndarray, group_size: int = 5):
    """
    Group selected subcarriers into groups of size group_size using greedy similarity linkage.
    """
    selected = list(dict.fromkeys(selected))
    if len(selected) % group_size != 0:
        raise ValueError(f"len(selected)={len(selected)} must be divisible by group_size={group_size}")

    sel_list = sorted(selected)
    unassigned = set(sel_list)

    # anchor score within selected set (mean sim to others)
    anchor_score = {}
    for k in sel_list:
        others = [j for j in sel_list if j != k]
        anchor_score[k] = float(np.mean([sim[k, j] for j in others])) if others else 0.0

    groups = []
    while unassigned:
        anchor = max(unassigned, key=lambda kk: anchor_score.get(kk, 0.0))
        unassigned.remove(anchor)
        candidates = sorted(list(unassigned), key=lambda j: sim[anchor, j], reverse=True)
        take = candidates[:group_size - 1]
        for j in take:
            unassigned.remove(j)
        groups.append([anchor] + take)

    return groups


# -------- FIM / CRB-inspired ranking --------
def steering_derivative_norm_sq(phi_rad: float, M: int, lam: float, d: float):
    """
    For ULA a_m(phi)=exp(-j*2*pi/lam * d * m * sin(phi)), m=0..M-1
    ||da/dphi||^2 = ( (2*pi/lam*d)^2 * cos^2(phi) ) * sum(m^2)
    """
    const = (2.0 * np.pi / lam) * d
    cosv = np.cos(phi_rad)
    m = np.arange(M, dtype=np.float64)
    return (const ** 2) * (cosv ** 2) * float(np.sum(m ** 2))


def estimate_snr_proxy_from_cov(R: np.ndarray, L: int = 1, eps: float = 1e-12):
    """
    SNR proxy from eigenvalues:
      noise ~ mean of smallest (M-L) eigenvalues
      signal ~ largest eigenvalue - noise
    """
    vals = np.linalg.eigvalsh(R)
    vals = np.sort(np.real(vals))[::-1]  # descending
    M = len(vals)
    noise = float(np.mean(vals[L:])) if (M - L) > 0 else float(vals[-1])
    signal = float(max(vals[0] - noise, 0.0))
    return signal / (noise + eps)


def compute_fim_scores_trainonly(
    CSI: np.ndarray,
    ant_start: int,
    M: int,
    n_samples: int,
    interval: int,
    start_sample: int,
    max_train_slices: int,
    phi_grid: np.ndarray,
    lam: float,
    d: float,
    teacher_agg: str = "mean",
    L: int = 1,
    trace_norm: bool = True
):
    """
    Compute per-subcarrier expected FIM score on train slices:
      score(k) = mean_g [ snr_proxy(k,g) * ||da/dphi(theta_g)||^2 ]
    where theta_g is estimated by a quick teacher MUSIC computed from ALL subcarriers averaged (1 group).

    This is CRB/FIM motivated and strongly grounded (estimation theory).
    """
    K = CSI.shape[1]
    T = CSI.shape[2]
    total_slices = (T - n_samples) // interval
    train_g = [g for g in range(total_slices) if (g * interval) < start_sample]
    if len(train_g) == 0:
        raise RuntimeError(f"No training slices for FIM selection (start_sample={start_sample}).")

    use_slices = train_g[:min(len(train_g), max_train_slices)]
    scores = np.zeros(K, dtype=np.float64)

    # one-group teacher for theta_g (uses all K) – fast and stable
    all_group = [list(range(K))]

    # Precompute derivative factor per slice (depends only on theta)
    for g in use_slices:
        start_idx = g * interval

        # quick teacher theta_g using all subcarriers in one group
        PS, doa_deg, _ = music_teacher_grouped_covavg_listgroups(
            CSI=CSI, ant_start=ant_start, M=M,
            groups_sc=all_group,
            start_idx=start_idx, n_samples=n_samples,
            phi=phi_grid, L=L, lam=lam, d=d, agg=teacher_agg
        )
        theta = np.deg2rad(doa_deg)
        deriv_norm2 = steering_derivative_norm_sq(theta, M=M, lam=lam, d=d)

        # compute per-subcarrier SNR proxies and accumulate
        for k in range(K):
            x = np.squeeze(CSI[ant_start: ant_start + M, k, start_idx: start_idx + n_samples])
            Rk = (x @ np.conjugate(x.T)) / (x.shape[1] + 1e-9)
            if trace_norm:
                tr = np.trace(Rk)
                if np.abs(tr) > 1e-12:
                    Rk = Rk / tr
            snr_k = estimate_snr_proxy_from_cov(Rk, L=L)
            scores[k] += snr_k * deriv_norm2

    scores = scores / max(1, len(use_slices))
    return scores.astype(np.float32)


def select_topk_by_scores(scores: np.ndarray, n_select: int):
    ranked = np.argsort(scores)[::-1]
    return ranked[:n_select].tolist()


# -------- D-optimal design (log-det greedy on signature vectors) --------
def dopt_greedy_select_from_signatures(sigs: np.ndarray, n_select: int, proj_dim: int = 32, eps: float = 1e-6, seed: int = 7):
    """
    D-optimal selection to maximize log det( eps*I + sum_{k in S} v_k v_k^T ).
    Uses random projection of signatures to low dimension for stability/speed.

    This is classic optimal design and has strong theoretical backing (log-det submodular in many settings).
    """
    K, F = sigs.shape
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((F, proj_dim)).astype(np.float32) / np.sqrt(proj_dim)
    V = sigs @ W  # (K, proj_dim)

    # normalize each v_k
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    selected = []
    # information matrix A = eps*I + sum v v^T
    A = (eps * np.eye(proj_dim, dtype=np.float64))
    A_inv = np.linalg.inv(A)

    # Greedy: choose k maximizing logdet(A + vv^T) - logdet(A) = log(1 + v^T A_inv v)
    remaining = set(range(K))
    for _ in range(n_select):
        best_k = None
        best_gain = -1e18

        for k in remaining:
            v = V[k].astype(np.float64).reshape(-1, 1)
            gain = float(np.log1p((v.T @ A_inv @ v).item()))
            if gain > best_gain:
                best_gain = gain
                best_k = k

        # update A_inv using Sherman-Morrison
        v = V[best_k].astype(np.float64).reshape(-1, 1)
        denom = 1.0 + (v.T @ A_inv @ v).item()
        A_inv = A_inv - (A_inv @ v @ v.T @ A_inv) / denom

        selected.append(best_k)
        remaining.remove(best_k)

    return selected


# -----------------------
# Plotting: overall DoA comparison for all n_select under a method
# -----------------------
def plot_overall_doa_comparison(experiment_root: str, results_index: list):
    """
    Creates for each method:
      overall_doa_comparison_{method}.png
    with one row per n_select:
      left: MUSIC vs GNN DoA over slices
      right: abs error over slices
    """
    by_method = {}
    for r in results_index:
        by_method.setdefault(r["method"], []).append(r)

    for method, runs in by_method.items():
        runs = sorted(runs, key=lambda x: x["n_select"])
        series = []
        for r in runs:
            metrics_path = os.path.join(r["run_dir"], "metrics.csv")
            if not os.path.exists(metrics_path):
                print(f"[WARN] Missing metrics.csv: {metrics_path}")
                continue
            df = pd.read_csv(metrics_path).sort_values("slice").reset_index(drop=True)
            series.append({
                "n_select": r["n_select"],
                "slice": df["slice"].values,
                "music": df["music_doa"].values,
                "gnn": df["gnn_doa"].values,
                "abs_err": np.abs(df["gnn_doa"].values - df["music_doa"].values),
            })

        if len(series) == 0:
            continue

        nrows = len(series)
        plt.figure(figsize=(14, 3.2 * nrows))
        for i, s in enumerate(series):
            ax1 = plt.subplot(nrows, 2, 2*i + 1)
            ax1.plot(s["slice"], s["music"], label="MUSIC Teacher")
            ax1.plot(s["slice"], s["gnn"], label="GNN Prediction")
            ax1.set_title(f"method={method} | n_select={s['n_select']}  (DoA)")
            ax1.set_xlabel("Slice")
            ax1.set_ylabel("DoA (deg)")
            ax1.grid(True)
            ax1.legend()

            ax2 = plt.subplot(nrows, 2, 2*i + 2)
            ax2.plot(s["slice"], s["abs_err"], label="|GNN - MUSIC|")
            ax2.set_title(f"method={method} | n_select={s['n_select']}  (Abs Error)")
            ax2.set_xlabel("Slice")
            ax2.set_ylabel("Abs Error (deg)")
            ax2.grid(True)
            ax2.legend()

        plt.tight_layout()
        out_path = os.path.join(experiment_root, f"overall_doa_comparison_{method}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Saved overall DoA comparison: {out_path}")


# -----------------------
# Single run (one method + one n_select)
# -----------------------
def run_one_config(
    CSI: np.ndarray,
    args,
    experiment_root: str,
    selection_method: str,
    n_select: int,
    lam: float, d: float, phi: np.ndarray,
    M: int = 16,
    L: int = 1
):
    """
    Runs training + evaluation for a single selection_method and n_select.
    Returns: run_dir, summary dict.
    """
    n_samples = args.window_size
    interval = args.stride
    K = CSI.shape[1]
    T = CSI.shape[2]
    total_g = (T - n_samples) // interval
    if total_g <= 0:
        raise RuntimeError(f"Not enough CSI length={T} for window_size={n_samples} and stride={interval}.")

    if n_select > K:
        raise ValueError(f"n_select={n_select} cannot exceed total subcarriers K={K}.")
    if n_select % args.group_size != 0:
        raise ValueError(f"n_select={n_select} must be divisible by group_size={args.group_size}.")

    # antenna block
    ant_start = 0

    # Prepare directories
    run_dir = os.path.join(experiment_root, f"method_{selection_method}", f"sel{n_select}_gsize{args.group_size}")
    data_dir = os.path.join(run_dir, "data")
    gnn_figure_dir = os.path.join(run_dir, "gnn_figure")
    gnn_data_dir = os.path.join(run_dir, "gnn_data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(gnn_figure_dir, exist_ok=True)
    os.makedirs(gnn_data_dir, exist_ok=True)

    # ---- Build signatures + similarity (used by corr + grouping, and also for dopt base)
    sigs = compute_subcarrier_signatures_trainonly(
        CSI=CSI,
        ant_start=ant_start,
        M=M,
        n_samples=n_samples,
        interval=interval,
        start_sample=args.start_sample,
        max_train_slices=args.max_train_slices_for_selection,
        trace_norm=args.trace_norm
    )
    sim = corr_similarity_matrix(sigs, use_abs=True)

    # ---- Select subcarriers
    if selection_method == "corr":
        selected, scores = select_corr_centrality(sim, n_select=n_select, top_p=args.corr_top_p)

    elif selection_method == "fim":
        fim_scores = compute_fim_scores_trainonly(
            CSI=CSI, ant_start=ant_start, M=M,
            n_samples=n_samples, interval=interval, start_sample=args.start_sample,
            max_train_slices=args.max_train_slices_for_selection,
            phi_grid=phi, lam=lam, d=d,
            teacher_agg=args.teacher_agg,
            L=L,
            trace_norm=args.trace_norm
        )
        selected = select_topk_by_scores(fim_scores, n_select=n_select)
        scores = fim_scores

    elif selection_method == "dopt":
        selected = dopt_greedy_select_from_signatures(
            sigs=sigs, n_select=n_select,
            proj_dim=args.dopt_proj_dim, eps=args.dopt_eps, seed=args.dopt_seed
        )
        # optional "scores" (not centrality) just for logging
        scores = None

    else:
        raise ValueError(f"Unknown selection_method={selection_method}. Choose from corr|fim|dopt.")

    # ---- Group selected into groups of group_size using similarity linkage
    # (This makes your "strong 5 together" requirement explicit.)
    groups_sc = greedy_group_by_similarity(selected, sim, group_size=args.group_size)

    # Save selection metadata
    meta = {
        "selection_method": selection_method,
        "n_select": n_select,
        "group_size": args.group_size,
        "selected_subcarriers": selected,
        "groups_sc": groups_sc,
        "trace_norm": args.trace_norm,
        "topk_edges": args.topk,
        "teacher_agg": args.teacher_agg,
        "max_train_slices_for_selection": args.max_train_slices_for_selection,
        "corr_top_p": args.corr_top_p,
        "dopt_proj_dim": args.dopt_proj_dim,
        "dopt_eps": args.dopt_eps,
        "dopt_seed": args.dopt_seed,
    }
    with open(os.path.join(gnn_data_dir, "selection_groups.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Build datasets (train/test graphs) and save teacher spectra
    train_graphs, test_graphs, test_meta = [], [], []

    for g in range(total_g):
        start_idx = g * interval

        PS_teacher, doa_deg, music_time = music_teacher_grouped_covavg_listgroups(
            CSI=CSI, ant_start=ant_start, M=M,
            groups_sc=groups_sc,
            start_idx=start_idx, n_samples=n_samples,
            phi=phi, L=L, lam=lam, d=d,
            agg=args.teacher_agg
        )

        mask = (phi >= -np.pi / 2) & (phi <= np.pi / 2)
        phi_range = phi[mask]
        spec_music = PS_teacher[mask]

        savemat(
            os.path.join(data_dir, f"slice_{g}_music_grouped.mat"),
            {
                "phi": phi_range,
                "music_grouped": spec_music,
                "doa_music_deg": doa_deg,
                "selected_subcarriers": np.array(selected, dtype=np.int32),
                "groups_sc": np.array(groups_sc, dtype=np.int32),
            }
        )

        x_nodes, edge_index = build_graph_slice_selected_covavg(
            CSI=CSI, ant_start=ant_start, M=M,
            groups_sc=groups_sc,
            start_idx=start_idx, n_samples=n_samples,
            topk=args.topk,
            trace_norm=args.trace_norm
        )

        y_vec = torch.tensor(doa_deg_to_vec(doa_deg), dtype=torch.float32).view(1, 2)
        data = Data(x=x_nodes, edge_index=edge_index, y=y_vec)

        if start_idx < args.start_sample:
            train_graphs.append(data)
        else:
            test_graphs.append(data)
            test_meta.append((g, doa_deg, phi_range, spec_music, music_time))

    if len(train_graphs) == 0 or len(test_graphs) == 0:
        raise RuntimeError(
            f"Train or test set empty. train={len(train_graphs)}, test={len(test_graphs)}. "
            f"Try lowering --start_sample or increasing CSI length."
        )

    # ---- Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = GeoDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = GeoDataLoader(test_graphs, batch_size=max(64, args.batch_size), shuffle=False)

    in_dim = train_graphs[0].x.shape[1]
    model = AoAGNN(in_dim=in_dim, hidden=args.hidden, use_gat=args.use_gat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y.view(pred.shape[0], 2)
            loss = angular_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{selection_method} sel={n_select}] Epoch {epoch+1}/{args.epochs} | "
                  f"Loss={loss_sum / len(train_loader):.6f}")

    # Save model
    ckpt_path = os.path.join(gnn_data_dir, "gnn_model_state.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": in_dim,
            "hidden": args.hidden,
            "use_gat": args.use_gat,
            "selection_method": selection_method,
            "n_select": n_select,
            "group_size": args.group_size,
            "selected_subcarriers": selected,
            "groups_sc": groups_sc,
        },
        ckpt_path
    )

    # ---- Inference + timing
    model.eval()
    pred_vecs = []
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).cpu().numpy()
            pred_vecs.append(pred)
    t1 = time.time()

    pred_vecs = np.concatenate(pred_vecs, axis=0)
    gnn_preds_deg = vec_to_doa_deg(pred_vecs)
    gnn_time_per_slice = (t1 - t0) / max(1, len(gnn_preds_deg))

    np.save(os.path.join(gnn_data_dir, "gnn_predictions.npy"), gnn_preds_deg)

    # ---- Metrics + per-slice comparison plots
    metrics = []
    for idx, (g, music_doa, phi_range, spec_music, mus_time) in enumerate(test_meta):
        gnn_doa = float(gnn_preds_deg[idx])
        abs_err = abs(gnn_doa - music_doa)
        sq_err = (gnn_doa - music_doa) ** 2

        metrics.append({
            "slice": g,
            "gnn_doa": gnn_doa,
            "music_doa": music_doa,
            "abs_error": abs_err,
            "squared_error": sq_err,
            "gnn_infer_time": gnn_time_per_slice,
            "music_time": mus_time,
            "selection_method": selection_method,
            "n_select": n_select,
            "group_size": args.group_size,
            "num_groups": len(groups_sc),
        })

        # per-slice spectrum plot
        plt.figure()
        plt.semilogy(np.rad2deg(phi_range),
                     spec_music / (np.max(spec_music) + 1e-12),
                     label="MUSIC (teacher)")
        plt.axvline(gnn_doa, linestyle="--", label="GNN Prediction")
        plt.plot(gnn_doa, 1, "r*", markersize=10, label="GNN DoA Point")
        plt.xlabel("Angle (degree)")
        plt.ylabel("Spectrum (normalized)")
        plt.legend()
        plt.grid(True)
        plt.title(f"Slice {g} | method={selection_method} | sel={n_select}")
        plt.tight_layout()
        plt.savefig(os.path.join(gnn_figure_dir, f"slice_{g}_gnn_music_comparison.png"))
        plt.close()

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # time comparison plot
    plt.figure()
    plt.plot(df["slice"], df["music_time"], label="MUSIC Teacher Time / slice")
    plt.plot(df["slice"], df["gnn_infer_time"], label="GNN Inference Time / slice")
    plt.xlabel("Slice")
    plt.ylabel("Time (seconds)")
    plt.title(f"Time: MUSIC vs GNN | method={selection_method}, sel={n_select}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "time_comparison_plot.png"))
    plt.close()

    # DoA curves plot
    plt.figure()
    plt.plot(df["gnn_doa"].values, label="GNN Prediction")
    plt.plot(df["music_doa"].values, label="MUSIC Teacher")
    plt.title(f"GNN vs MUSIC DoA | method={selection_method}, sel={n_select}")
    plt.xlabel("Test Slice Index")
    plt.ylabel("DoA (degrees)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "DoA_GNN_vs_MUSIC_teacher_after_start_sample.png"))
    plt.close()

    mae = float(df["abs_error"].mean())
    rmse = float(np.sqrt(df["squared_error"].mean()))
    avg_music_time = float(df["music_time"].mean())
    print(f"[DONE] {selection_method} sel={n_select} | MAE={mae:.4f} | RMSE={rmse:.4f} | "
          f"GNN t/slice={gnn_time_per_slice:.6f}s | MUSIC t/slice={avg_music_time:.6f}s")
    return run_dir, {
        "selection_method": selection_method,
        "n_select": n_select,
        "group_size": args.group_size,
        "num_groups": len(groups_sc),
        "mae": mae,
        "rmse": rmse,
        "gnn_time_per_slice": gnn_time_per_slice,
        "music_time_per_slice": avg_music_time,
        "run_dir": run_dir,
    }


# -----------------------
# Main (sweep)
# -----------------------
def main():
    parser = argparse.ArgumentParser()

    # core
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--track", type=int, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)

    # slicing
    parser.add_argument("--start_sample", type=int, default=50000)
    parser.add_argument("--window_size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=100)

    # selection sweep
    parser.add_argument("--selection_method", type=str, default="corr",
                        choices=["corr", "fim", "dopt"],
                        help="Which method to run (corr|fim|dopt).")
    parser.add_argument("--run_all_methods", action="store_true",
                        help="If set, runs corr,fim,dopt for each n_select in n_select_list.")
    parser.add_argument("--n_select_list", type=str, default="20",
                        help="Comma list like 10,20,30,40 (must be divisible by group_size).")
    parser.add_argument("--group_size", type=int, default=5)

    # selection params
    parser.add_argument("--max_train_slices_for_selection", type=int, default=200)
    parser.add_argument("--corr_top_p", type=int, default=10)

    # dopt params
    parser.add_argument("--dopt_proj_dim", type=int, default=32)
    parser.add_argument("--dopt_eps", type=float, default=1e-6)
    parser.add_argument("--dopt_seed", type=int, default=7)

    # graph/features
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--trace_norm", action="store_true")
    parser.add_argument("--teacher_agg", type=str, default="mean", choices=["mean", "median"])

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_gat", action="store_true")
    parser.add_argument("--hidden", type=int, default=128)

    args = parser.parse_args()

    # constants
    f = 2.18e9
    c = 3e8
    lam = c / f
    d = lam / 2
    M = 16
    L = 1
    phi = np.arange(-np.pi, np.pi, np.pi / 360)

    # load CSI
    CSI, files = load_nokia_track_csi(args.dataset_dir, args.track, ant_max=64, sc_max=50)
    print(f"[OK] Loaded CSI shape={CSI.shape} from {len(files)} files.")
    print(f"[OK] Example file: {files[0]}")

    # experiment root
    experiment_root = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_root, exist_ok=True)

    # parse n_select list
    n_select_list = parse_int_list(args.n_select_list)
    if len(n_select_list) == 0:
        raise ValueError("n_select_list is empty. Provide like --n_select_list 10,20,30,40")

    # validate divisibility
    for nsel in n_select_list:
        if nsel % args.group_size != 0:
            raise ValueError(f"n_select={nsel} must be divisible by group_size={args.group_size}.")

    methods = ["corr", "fim", "dopt"] if args.run_all_methods else [args.selection_method]

    summary_rows = []
    results_index = []

    for method in methods:
        for nsel in n_select_list:
            print(f"\n[RUN] method={method} | n_select={nsel} | group_size={args.group_size}\n")
            run_dir, summ = run_one_config(
                CSI=CSI,
                args=args,
                experiment_root=experiment_root,
                selection_method=method,
                n_select=nsel,
                lam=lam, d=d, phi=phi,
                M=M, L=L
            )
            summary_rows.append(summ)
            results_index.append({"method": method, "n_select": nsel, "run_dir": run_dir})

    # write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(experiment_root, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Wrote summary: {summary_path}")

    # comparison plots (MAE/RMSE and time) across all runs
    # --- MAE / RMSE
    plt.figure()
    # plot each method as its own line per metric
    for method in methods:
        sdf = summary_df[summary_df["selection_method"] == method].sort_values("n_select")
        if len(sdf) == 0:
            continue
        plt.plot(sdf["n_select"], sdf["mae"], label=f"{method} MAE")
        plt.plot(sdf["n_select"], sdf["rmse"], label=f"{method} RMSE")
    plt.xlabel("n_select")
    plt.ylabel("Degrees")
    plt.title("MAE/RMSE vs n_select (all runs)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_root, "comparison_mae_rmse.png"))
    plt.close()

    # --- Time
    plt.figure()
    for method in methods:
        sdf = summary_df[summary_df["selection_method"] == method].sort_values("n_select")
        if len(sdf) == 0:
            continue
        plt.plot(sdf["n_select"], sdf["music_time_per_slice"], label=f"{method} MUSIC time/slice")
        plt.plot(sdf["n_select"], sdf["gnn_time_per_slice"], label=f"{method} GNN time/slice")
    plt.xlabel("n_select")
    plt.ylabel("Seconds")
    plt.title("Time per slice vs n_select (all runs)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_root, "comparison_time.png"))
    plt.close()

    # Overall DoA comparison across all n_select, per method
    plot_overall_doa_comparison(experiment_root, results_index)

    print(f"[DONE] All outputs under: {experiment_root}")


if __name__ == "__main__":
    main()
