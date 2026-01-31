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
# Covariance -> feature
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
    """
    Average covariance across explicit subcarrier indices in sc_list.
    CSI: [num_ant_total, num_subcarriers, T]
    Returns complex covariance (M,M)
    """
    R_sum = np.zeros((M, M), dtype=np.complex128)
    for sc in sc_list:
        x = np.squeeze(CSI[ant_start: ant_start + M, sc, start_idx: start_idx + n_samples])  # (M,n_samples)
        R = (x @ np.conjugate(x.T)) / (x.shape[1] + 1e-9)
        R_sum += R
    return R_sum / max(1, len(sc_list))


# --------------------------------------
# Correlation-based subcarrier selection + grouping
# --------------------------------------
def corr_similarity_matrix(X: np.ndarray, use_abs: bool = True) -> np.ndarray:
    """
    X: (K,F) signatures
    sim(i,j) = corr(X[i], X[j]) (implemented via normalized dot after mean-centering)
    """
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
    Build per-subcarrier signature vectors from TRAIN region only (start_idx < start_sample).
    Signature for subcarrier k: average over training slices of cov_to_feature(R_k(slice)).
    Returns sigs: (K,F) float32
    """
    K = CSI.shape[1]
    T = CSI.shape[2]
    total_slices = (T - n_samples) // interval
    if total_slices <= 0:
        raise RuntimeError(f"Not enough CSI length={T} for window_size={n_samples}, stride={interval}")

    # collect training slice indices (g) such that start_idx < start_sample
    train_g = []
    for g in range(total_slices):
        start_idx = g * interval
        if start_idx < start_sample:
            train_g.append(g)
    if len(train_g) == 0:
        raise RuntimeError(
            f"No training slices for selection (start_sample={start_sample}). "
            f"Try increasing start_sample or reducing window/stride."
        )

    use_slices = train_g[:min(len(train_g), max_train_slices)]

    sig_sum = None
    count = 0

    for g in use_slices:
        start_idx = g * interval
        feats = []
        for k in range(K):
            x = np.squeeze(CSI[ant_start: ant_start + M, k, start_idx: start_idx + n_samples])
            R = (x @ np.conjugate(x.T)) / (x.shape[1] + 1e-9)
            fk = cov_to_feature(R, trace_norm=trace_norm)
            feats.append(fk)
        feats = np.stack(feats, axis=0)  # (K,F)

        if sig_sum is None:
            sig_sum = feats.astype(np.float64)
        else:
            sig_sum += feats.astype(np.float64)
        count += 1

    sigs = (sig_sum / max(1, count)).astype(np.float32)
    return sigs


def select_strong_subcarriers(sim: np.ndarray, n_select: int = 20, top_p: int = 10):
    """
    Choose subcarriers with high "centrality": score(k) = mean of top_p similarities in row k.
    Returns selected list and scores array.
    """
    K = sim.shape[0]
    if n_select > K:
        raise ValueError(f"n_select={n_select} must be <= K={K}")
    top_p = min(top_p, K - 1)
    scores = np.zeros(K, dtype=np.float32)
    for k in range(K):
        row = sim[k]
        scores[k] = float(np.mean(np.sort(row)[-top_p:])) if top_p > 0 else 0.0
    ranked = np.argsort(scores)[::-1]
    selected = ranked[:n_select].tolist()
    return selected, scores


def greedy_group_by_similarity(selected, sim: np.ndarray, group_size: int = 5):
    """
    Given selected subcarriers, create groups of size group_size:
      - pick anchor with highest avg similarity to other selected
      - add top (group_size-1) most similar unassigned to anchor
      - repeat
    """
    selected = list(selected)
    if len(selected) % group_size != 0:
        raise ValueError(f"len(selected)={len(selected)} must be divisible by group_size={group_size}")

    selected_set = set(selected)
    sel_list = sorted(selected_set)

    # anchor score within selected set
    anchor_score = {}
    for k in sel_list:
        others = [j for j in sel_list if j != k]
        anchor_score[k] = float(np.mean([sim[k, j] for j in others])) if others else 0.0

    unassigned = set(sel_list)
    groups = []
    while unassigned:
        anchor = max(unassigned, key=lambda k: anchor_score.get(k, 0.0))
        unassigned.remove(anchor)

        candidates = sorted(list(unassigned), key=lambda j: sim[anchor, j], reverse=True)
        take = candidates[: (group_size - 1)]
        for j in take:
            unassigned.remove(j)
        groups.append([anchor] + take)

    return groups


# --------------------------------------
# Grouping-correct MUSIC teacher (list-groups, cov-avg)
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
    phi_in = phi[mask]
    PS_in = PS_agg[mask]

    doa_rad = phi_in[np.argmax(PS_in)]
    doa_deg = float(np.degrees(doa_rad))
    doa_deg = float(np.clip(doa_deg, -90.0, 90.0))

    music_time = t1 - t0
    return PS_agg, doa_deg, music_time


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

    # chain edges
    edges = []
    for i in range(N - 1):
        edges += [(i, i + 1), (i + 1, i)]

    # optional top-k similarity edges
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
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
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
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # [cos, sin]
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.enc(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = self.pool(x, batch)
        out = self.head(g)
        out = F.normalize(out, dim=1)  # keep on unit circle
        return out


def doa_deg_to_vec(doa_deg: float):
    rad = np.deg2rad(doa_deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)


def vec_to_doa_deg(vec2: np.ndarray):
    c = vec2[..., 0]
    s = vec2[..., 1]
    rad = np.arctan2(s, c)
    deg = np.rad2deg(rad)
    return np.clip(deg, -90.0, 90.0)


def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    return 1.0 - torch.sum(pred * target, dim=1).mean()


# -----------------------
# Dataset loader (robust)
# -----------------------
def load_nokia_track_csi(dataset_dir: str, track: int, ant_max: int = 64, sc_max: int = 50):
    """
    Supports both layouts:
      1) dataset_dir/t{track}/*.mat
      2) dataset_dir/*.mat (files contain t{track}_*.mat)
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
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--track", type=int, required=True)

    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to NOKIA dataset root (contains t1/, t2/.. or t{track}_*.mat files)")

    parser.add_argument("--start_sample", type=int, default=50000)
    parser.add_argument("--window_size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=100)

    # selection + grouping (NEW)
    parser.add_argument("--n_select", type=int, default=20,
                        help="Number of subcarriers to SELECT using correlation (default 20)")
    parser.add_argument("--group_size", type=int, default=5,
                        help="Group size (#subcarriers per node) after selection (default 5)")
    parser.add_argument("--corr_top_p", type=int, default=10,
                        help="For selection score: mean of top_p similarities per row")
    parser.add_argument("--max_train_slices_for_corr", type=int, default=200,
                        help="How many training slices to estimate correlation signatures (default 200)")

    # graph / features
    parser.add_argument("--topk", type=int, default=0, help="0 disables dynamic similarity edges")
    parser.add_argument("--trace_norm", action="store_true", help="trace normalize covariances")
    parser.add_argument("--teacher_agg", type=str, default="mean", choices=["mean", "median"])

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_gat", action="store_true", help="use GATv2Conv (else SAGEConv)")

    args = parser.parse_args()

    # --- constants ---
    f = 2.18e9
    c = 3e8
    lam = c / f
    d = lam / 2
    M = 16
    L = 1
    phi = np.arange(-np.pi, np.pi, np.pi / 360)

    n_samples, interval = args.window_size, args.stride

    # --- load CSI ---
    CSI, files = load_nokia_track_csi(args.dataset_dir, args.track, ant_max=64, sc_max=50)
    print(f"[OK] Loaded CSI shape={CSI.shape} from {len(files)} files.")
    print(f"[OK] Example file: {files[0]}")

    # --- pick 16-antenna block ---
    row_ant = 0
    ant_start = row_ant * M

    # --- total slices ---
    total_g = (CSI.shape[2] - n_samples) // interval
    if total_g <= 0:
        raise RuntimeError(f"Not enough CSI length={CSI.shape[2]} for window_size={n_samples} and stride={interval}.")

    # --- sanity for selection/grouping ---
    K = CSI.shape[1]
    if args.n_select > K:
        raise ValueError(f"--n_select {args.n_select} cannot exceed total subcarriers K={K}.")
    if args.n_select % args.group_size != 0:
        raise ValueError(f"--n_select {args.n_select} must be divisible by --group_size {args.group_size}.")

    # --- output dirs ---
    results_dir = os.path.join(
        args.output_dir,
        f"{args.experiment_name}_sel{args.n_select}_gsize{args.group_size}"
    )
    data_dir = os.path.join(results_dir, "data")
    gnn_figure_dir = os.path.join(results_dir, "gnn_figure")
    gnn_data_dir = os.path.join(results_dir, "gnn_data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(gnn_figure_dir, exist_ok=True)
    os.makedirs(gnn_data_dir, exist_ok=True)

    # --------------------------------------
    # CORRELATION-BASED SELECTION + GROUPING
    #   - compute signatures from TRAIN ONLY (start_idx < start_sample)
    #   - build abs-corr similarity matrix
    #   - select n_select strongest
    #   - greedy group into groups of group_size (e.g., 4 groups of 5)
    # --------------------------------------
    print("[INFO] Computing train-only subcarrier signatures for correlation selection...")
    sigs = compute_subcarrier_signatures_trainonly(
        CSI=CSI,
        ant_start=ant_start,
        M=M,
        n_samples=n_samples,
        interval=interval,
        start_sample=args.start_sample,
        max_train_slices=args.max_train_slices_for_corr,
        trace_norm=args.trace_norm
    )
    sim = corr_similarity_matrix(sigs, use_abs=True)

    selected, scores = select_strong_subcarriers(sim, n_select=args.n_select, top_p=args.corr_top_p)
    groups_sc = greedy_group_by_similarity(selected, sim, group_size=args.group_size)

    print(f"[INFO] Selected subcarriers (n={len(selected)}): {selected}")
    print(f"[INFO] Groups ({len(groups_sc)} groups x {args.group_size}): {groups_sc}")

    # save selection/groups for reproducibility
    with open(os.path.join(gnn_data_dir, "selection_groups.json"), "w") as fjs:
        json.dump(
            {
                "n_select": args.n_select,
                "group_size": args.group_size,
                "corr_top_p": args.corr_top_p,
                "max_train_slices_for_corr": args.max_train_slices_for_corr,
                "selected_subcarriers": selected,
                "groups_sc": groups_sc,
            },
            fjs,
            indent=2
        )

    train_graphs, test_graphs, test_meta = [], [], []

    # --- slicing loop ---
    for g in range(total_g):
        start_idx = g * interval

        # --- grouped MUSIC teacher (list-groups, cov-avg) ---
        PS_teacher, doa_deg, music_time = music_teacher_grouped_covavg_listgroups(
            CSI=CSI,
            ant_start=ant_start,
            M=M,
            groups_sc=groups_sc,
            start_idx=start_idx,
            n_samples=n_samples,
            phi=phi,
            L=L,
            lam=lam,
            d=d,
            agg=args.teacher_agg,
        )

        # crop to [-90, 90] for plotting and saving
        mask = (phi >= -np.pi / 2) & (phi <= np.pi / 2)
        phi_range = phi[mask]
        spec_music = PS_teacher[mask]

        # save teacher spectrum
        savemat(
            os.path.join(data_dir, f"slice_{g}_music_grouped.mat"),
            {
                "phi": phi_range,
                "music_grouped": spec_music,
                "doa_music_deg": doa_deg,
                "selected_subcarriers": np.array(selected, dtype=np.int32),
                "groups_sc": np.array(groups_sc, dtype=np.int32),
            },
        )

        # --- build frequency-only graph from SELECTED groups (cov-avg features) ---
        x_nodes, edge_index = build_graph_slice_selected_covavg(
            CSI=CSI,
            ant_start=ant_start,
            M=M,
            groups_sc=groups_sc,
            start_idx=start_idx,
            n_samples=n_samples,
            topk=args.topk,
            trace_norm=args.trace_norm,
        )

        y_vec = torch.tensor(doa_deg_to_vec(doa_deg), dtype=torch.float32).view(1, 2)
        data = Data(x=x_nodes, edge_index=edge_index, y=y_vec)

        if start_idx < args.start_sample:
            train_graphs.append(data)
        else:
            test_graphs.append(data)
            test_meta.append((g, doa_deg, phi_range, spec_music, music_time))

    print(f"[INFO] Train graphs: {len(train_graphs)} | Test graphs: {len(test_graphs)}")
    if len(train_graphs) == 0 or len(test_graphs) == 0:
        raise RuntimeError(
            f"Train or test set empty. train={len(train_graphs)}, test={len(test_graphs)}. "
            f"Try lowering --start_sample or increasing CSI length."
        )

    # --- train ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = GeoDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = GeoDataLoader(test_graphs, batch_size=max(64, args.batch_size), shuffle=False)

    in_dim = train_graphs[0].x.shape[1]
    model = AoAGNN(in_dim=in_dim, hidden=128, use_gat=args.use_gat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)  # (B,2) unit vectors
            target = batch.y.view(pred.shape[0], 2)
            loss = angular_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {loss_sum / len(train_loader):.6f}")

    # save model checkpoint
    ckpt_path = os.path.join(gnn_data_dir, "gnn_model_state.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": in_dim,
            "hidden": 128,
            "use_gat": args.use_gat,
            "topk": args.topk,
            "trace_norm": args.trace_norm,
            "teacher_agg": args.teacher_agg,
            "n_select": args.n_select,
            "group_size": args.group_size,
            "selected_subcarriers": selected,
            "groups_sc": groups_sc,
            "corr_top_p": args.corr_top_p,
            "max_train_slices_for_corr": args.max_train_slices_for_corr,
        },
        ckpt_path,
    )
    print(f"[INFO] Saved model: {ckpt_path}")

    # --- inference + timing ---
    model.eval()
    pred_vecs = []
    t_inf0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).cpu().numpy()
            pred_vecs.append(pred)
    t_inf1 = time.time()

    pred_vecs = np.concatenate(pred_vecs, axis=0)
    gnn_preds_deg = vec_to_doa_deg(pred_vecs)
    gnn_time_per_slice = (t_inf1 - t_inf0) / max(1, len(gnn_preds_deg))

    np.save(os.path.join(gnn_data_dir, "gnn_predictions.npy"), gnn_preds_deg)

    # --- metrics + plots ---
    metrics = []
    for idx, (g, music_doa, phi_range, spec_music, mus_time) in enumerate(test_meta):
        gnn_doa = float(gnn_preds_deg[idx])
        abs_err = abs(gnn_doa - music_doa)
        sq_err = (gnn_doa - music_doa) ** 2

        metrics.append(
            {
                "slice": g,
                "gnn_doa": gnn_doa,
                "music_doa": music_doa,
                "abs_error": abs_err,
                "squared_error": sq_err,
                "gnn_infer_time": gnn_time_per_slice,
                "music_time": mus_time,
                "n_select": args.n_select,
                "group_size": args.group_size,
                "num_groups": len(groups_sc),
            }
        )

        plt.figure()
        plt.semilogy(
            np.rad2deg(phi_range),
            spec_music / (np.max(spec_music) + 1e-12),
            label="MUSIC (grouped teacher)",
        )
        plt.axvline(gnn_doa, color="green", linestyle="--", label="GNN Prediction")
        plt.plot(gnn_doa, 1, "r*", markersize=10, label="GNN DoA Point")
        plt.xlabel("Angle (degree)")
        plt.ylabel("Spectrum (normalized)")
        plt.legend()
        plt.grid(True)
        plt.title(f"Slice {g} | sel={args.n_select} | groups={len(groups_sc)}")
        plt.tight_layout()
        plt.savefig(os.path.join(gnn_figure_dir, f"slice_{g}_gnn_music_comparison.png"))
        plt.close()

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

    plt.figure()
    plt.plot(df["slice"], df["music_time"], label="MUSIC Teacher Time per Slice")
    plt.plot(df["slice"], df["gnn_infer_time"], label="GNN Inference Time per Slice")
    plt.xlabel("Slice")
    plt.ylabel("Time (seconds)")
    plt.title(f"Time: MUSIC Teacher vs GNN (sel={args.n_select}, groups={len(groups_sc)})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "time_comparison_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(df["gnn_doa"].values, label="GNN Prediction")
    plt.plot(df["music_doa"].values, label="MUSIC Teacher")
    plt.title(f"GNN vs MUSIC Teacher DoA (sel={args.n_select}, groups={len(groups_sc)})")
    plt.xlabel("Test Slice Index")
    plt.ylabel("DoA (degrees)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "DoA_GNN_vs_MUSIC_teacher_after_start_sample.png"))
    plt.close()

    mae = float(df["abs_error"].mean())
    rmse = float(np.sqrt(df["squared_error"].mean()))
    print(f"[DONE] Saved under: {results_dir}")
    print(
        f"[SUMMARY] sel={args.n_select} | group_size={args.group_size} | num_groups={len(groups_sc)} "
        f"| MAE={mae:.4f} deg | RMSE={rmse:.4f} deg | GNN time/slice={gnn_time_per_slice:.6f}s"
    )


if __name__ == "__main__":
    main()
