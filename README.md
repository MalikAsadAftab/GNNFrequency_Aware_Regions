# Subcarrier-Selected Region-Aware GNN for AoA (MUSIC Teacher) — Sweep + Comparison

This repository implements a **teacher–student Angle-of-Arrival (AoA) estimation framework** based on:

* **Teacher:** MUSIC (Multiple Signal Classification)
* **Student:** Graph Neural Network (GNN) regressing AoA as a **unit-circle vector** `[cos(θ), sin(θ)]`
* **Key idea:** Instead of using *all* subcarriers, we **select a statistically meaningful subset** (e.g., 10 / 20 / 30 / 40) and **group** them into graph nodes (e.g., 5 subcarriers per node).

The framework supports **multiple subcarrier selection strategies**, **sweeps over different subset sizes**, and produces **comprehensive comparison plots**, including **overall DoA trajectories (GNN vs MUSIC)**.

---

## Table of Contents

* Motivation
* High-Level Pipeline
* Data Assumptions (Nokia CSI)
* Windowing and Train/Test Split
* Subcarrier Selection Methods

  * Correlation Centrality
  * Fisher Information / CRB-based
  * D-optimal Design
* Grouping Into Graph Nodes
* Teacher: Grouped MUSIC
* Student: Graph Construction and Features
* GNN Architecture and Loss
* Running the Code
* Output Structure
* Metrics and Plots
* Notes and Best Practices
* Troubleshooting

---

## Motivation

In practical CSI datasets, **not all subcarriers contribute equally to AoA estimation**.
Some subcarriers may be noisy, unstable, or redundant.

Instead of feeding all subcarriers to the GNN, this framework:

1. **Selects only the most informative subcarriers**
2. **Groups them into nodes** to form a compact graph
3. Trains a **student GNN** that is:

   * More efficient
   * More robust
   * Easier to analyze

It also allows **curriculum-style evaluation**, e.g.:

```
n_select = 10 → 20 → 30 → 40
```

to study accuracy–complexity tradeoffs.

---

## High-Level Pipeline

For each configuration `(selection_method, n_select)`:

1. Load CSI track from Nokia `.mat` files
2. Slice CSI into overlapping windows
3. Using **training data only**:

   * Select subcarriers
   * Group them into graph nodes
4. For each slice:

   * Compute MUSIC teacher DoA using grouped covariances
   * Build a graph with grouped node features
5. Train a GNN using angular regression
6. Evaluate on test slices
7. Aggregate results across `n_select` and methods

---

## Data Assumptions (Nokia CSI)

The dataset must contain MATLAB `.mat` files with:

* Key: `Hd_all`
* Shape:

  ```
  [num_antennas, num_subcarriers, time]
  ```

Supported layouts:

### Folder per track

```
dataset_dir/
  t12/
    *.mat
  t13/
    *.mat
```

### Flat layout

```
dataset_dir/
  t12_*.mat
  t13_*.mat
```

All CSI files are concatenated along the time axis.

---

## Windowing and Train/Test Split

CSI is sliced using:

* `window_size`: snapshots per slice
* `stride`: shift between slices

Slice start index:

```
start_idx = g × stride
```

Train/test split is **time-based**:

* Training: `start_idx < start_sample`
* Testing:  `start_idx ≥ start_sample`

This avoids temporal leakage and simulates future prediction.

---

## Subcarrier Selection Methods

Selection is performed **only on training slices**.

### 1. Correlation Centrality (`corr`)

**Idea:** Subcarriers that are strongly correlated with many others are more stable.

Steps:

1. Compute covariance-based signatures per subcarrier
2. Compute absolute Pearson correlation between signatures
3. Centrality score:

   ```
   score(k) = mean of top-p correlations
   ```
4. Select top `n_select`

**Pros:** Fast, simple
**Cons:** Can select redundant subcarriers

---

### 2. Fisher Information / CRB-based (`fim`)

**Estimation-theoretic approach**.

For a ULA steering vector:

```
a_m(θ) = exp(-j 2π/λ · d · m · sin(θ))
```

Fisher information contribution:

```
J_k(θ) ∝ SNR_k · || ∂a(θ)/∂θ ||²
```

Implementation:

* Estimate AoA per training slice via MUSIC
* Estimate SNR from covariance eigenvalues
* Average Fisher score across slices

**Pros:** Strong theoretical backing (CRB minimization)
**Cons:** Slightly higher computation

---

### 3. D-optimal Design (`dopt`)

Based on **optimal experiment design**.

Objective:

```
maximize log det( εI + Σ v_k v_kᵀ )
```

Where `v_k` are projected subcarrier signatures.

Uses greedy log-det gain with Sherman–Morrison updates.

**Pros:** Encourages diversity and information coverage
**Cons:** Projection dimension matters

---

## Grouping Into Graph Nodes

After selecting `n_select` subcarriers:

* `group_size` subcarriers per group
* Number of nodes:

  ```
  N = n_select / group_size
  ```

Grouping is similarity-based:

1. Pick an anchor subcarrier
2. Add the most similar remaining subcarriers
3. Repeat

This enforces **“strong 5 together”** behavior.

---

## Teacher: Grouped MUSIC

For each slice and group:

1. Compute covariance per subcarrier
2. Average covariances within group
3. Run MUSIC per group
4. Aggregate spectra across groups
5. Teacher label:

   ```
   θ = argmax P(φ), φ ∈ [-90°, 90°]
   ```

Teacher and student **share the same grouping**, ensuring consistency.

---

## Student: Graph Construction and Features

### Node features

For each group:

* Average covariance
* Feature vector:

  ```
  [Re(vec(R)), Im(vec(R))]
  ```
* Optional trace normalization

### Edges

* Chain edges: `(i ↔ i+1)`
* Optional top-k similarity edges (cosine similarity)

---

## GNN Architecture and Loss

### Output representation

Instead of regressing angles directly:

```
y = [cos(θ), sin(θ)]
```

### Angular loss

```
L = 1 − mean(ŷ · y)
```

Benefits:

* No angle wrapping issues
* Stable gradients
* Unit-circle constraint

---

## Running the Code

### Install dependencies

```
pip install numpy scipy matplotlib pandas torch torch-geometric
```

(Use official PyG wheels for CUDA setups.)

---

### Run a single method

```
python main_select_sweep.py \
  --dataset_dir /path/to/NOKIA \
  --experiment_name fim_run \
  --output_dir ./results \
  --track 12 \
  --selection_method fim \
  --n_select_list 20 \
  --group_size 5
```

---

### Sweep multiple n_select values

```
--n_select_list 10,20,30,40
```

---

### Run all methods

```
--run_all_methods
```

---

## Output Structure

```
results/experiment_name/
  method_corr/
    sel20_gsize5/
      data/
      gnn_data/
      gnn_figure/
      metrics.csv
  method_fim/
  method_dopt/
  summary_results.csv
  comparison_mae_rmse.png
  comparison_time.png
  overall_doa_comparison_corr.png
  overall_doa_comparison_fim.png
  overall_doa_comparison_dopt.png
```

---

## Metrics and Plots

### Per-run

* Slice-level errors and timing
* MUSIC vs GNN DoA plots
* Per-slice spectrum plots

### Experiment-level

* MAE / RMSE vs `n_select`
* Time vs `n_select`
* **Overall DoA trajectories for each method**

---

## Notes and Best Practices

* Always select subcarriers using **training data only**
* `fim` is recommended for theory-driven studies
* `dopt` is useful when diversity matters
* `group_size = 5` is a good default
* Reduce epochs / training slices for fast debugging

---

## Troubleshooting

**Train/test set empty**

* Adjust `start_sample`

**n_select not divisible by group_size**

* Example: `20 / 5` (Correct)
* Example: `30 / 8` (Wrong)

**Torch Geometric issues**

* Install matching wheels for Torch/CUDA

