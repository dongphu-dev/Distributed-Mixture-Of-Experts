# Optimal Transport Aggregation for Distributed Mixture-of-Experts

MATLAB implementation for the paper:

> **Optimal Transport Aggregation for Distributed Mixture-of-Experts**  
> _(Under review for AISTATS 2027)_

This repository contains code to reproduce the experiments in the paper, including one-shot optimal transport reduction (OT-RED), baseline methods, synthetic simulations, and real-world benchmark datasets.

---

## Overview

In distributed Mixture-of-Experts (MoE) regression, $M$ local workers train models independently on private data partitions. Aggregating these models at a central server is complicated by:

1. **Label switching**: Expert indexing is arbitrary on each local machine, so direct parameter averaging (WAVR) is ineffective.
2. **Covariate-dependent gating**: Gating probabilities $\pi_k(\mathbf{x})$ vary with inputs, preventing standard unconditional mixture reduction.
3. **Capacity heterogeneity**: Workers may observe different numbers of subpopulations, fitting local models with $K_m \neq K$.

The proposed method aggregates local models into a global K-expert MoE in a single one-way communication round by minimizing a covariate-dependent transportation divergence on a small supporting sample. The server alternates between:

- A Majorization-Minimization (MM) step to optimize expert hyperplanes without requiring gating information.
- An Iteratively Reweighted Least Squares (IRLS) step to estimate the global softmax gating parameters from optimal transport probabilities.

---

## Repository structure

```text
Distributed-Mixture-Of-Experts/
├── setup_paths.m              # Initializes MATLAB search paths
├── run_all_benchmarks.m       # Script to run all synthetic benchmarks
├── test_independence_suite.m  # Verification suite for path isolation and models
├── LICENSE
├── README.md
│
├── models/                    # MoE estimators
│   ├── Distributed_MixtureOfExperts_Gaussian.m  # DMoE (homogeneous K_m = K)
│   ├── Distributed_MixtureOfExperts_Hetero.m    # DMoE (heterogeneous K_m)
│   ├── Global_MixtureOfExperts.m                # Centralized oracle (GLB)
│   ├── Aligned_MixtureOfExperts.m               # Hungarian-aligned averaging (AAVR)
│   ├── Averaged_MixtureOfExperts.m              # Arithmetic parameter mean (WAVR)
│   ├── Greedy_MixtureOfExperts.m                # Greedy pairwise KL merging (GM)
│   ├── Median_MixtureOfExperts.m                # Coordinate-wise medoid (MED)
│   └── FedAvg_MixtureOfExperts.m                # Multi-round Federated Averaging (FED)
│
├── stattools/                 # Optimization and identifiability utilities (IRLS, sorting)
├── evaltools/                 # Evaluation metrics (transport divergence, RPE, ARI, MSE)
├── datatools/                 # Synthetic data generation utilities
├── data/                      # Benchmark synthetic datasets and ground-truth parameters
├── experiments/               # Experiment benchmark runner scripts
├── real_data/                 # Real-world datasets and scripts
│   ├── beijing_air_quality/   # Multi-site air quality network (M=12, d=19, K=4)
│   └── year_prediction_msd/   # YearPredictionMSD subset (M=32, d=90, K=4)
├── toy_examples/              # 2D illustrative toy experiment scripts
└── results/                   # Numerical experiment outputs (.mat)
```

---

## Requirements and Setup

- **MATLAB** (R2020b or newer recommended).
- **Parallel Computing Toolbox** (optional, used for local machine training and Monte Carlo runs).

To initialize paths, run in MATLAB:

```matlab
setup_paths;
```

To run a quick verification of all modules:

```matlab
test_independence_suite;
```

---

## Reproducing experiments

### Synthetic benchmarks

All synthetic experiments use $M = 16$ machines, $d = 20$ features, and $K = 5$ target experts on $N = 100{,}000$ samples.

1. **Experiment 1: Homogeneous benchmark** (Balanced, moderately imbalanced, and highly imbalanced profiles):

    ```matlab
    exp_official_homogeneous_benchmark;
    ```

    Evaluates GLB, DMoE, GM, A-AVR, W-AVR, MED, and FED.

2. **Experiment 2: Heterogeneous capacities** (K_m=3 to K=5):

    ```matlab
    exp_official_heterogeneous_Km_benchmark;
    ```

    Evaluates aggregation when local workers fit only K_m=3 experts, yielding L=48 components to aggregate into K=5.

3. **Experiment 3: Sensitivity to Support Sample Size $S$**:
    ```matlab
    exp_sensitivity_support_S;
    ```
    Varies $S \in \{30, 50, \dots, 5000\}$ across pooled, single-worker, and simulated covariate sources.

Alternatively, to run a fast 2-replicate smoke test of the benchmark suite:

```matlab
run_all_benchmarks('quick');
```

---

## Real-World datasets

1. **Beijing multi-site air quality** (M=12 stations, K=4, d=19):

    ```matlab
    cd real_data/beijing_air_quality
    run_beijing_benchmark;
    ```

    Predicts hourly $\text{PM}_{2.5}$ with out-of-time evaluation (3-year training, 1-year test).

2. **YearPredictionMSD** (M=32 nodes, K=4, d=90):
    ```matlab
    cd real_data/year_prediction_msd
    run_year_msd_benchmark;
    ```
    Predicts release year from timbre features on a subsampled partition.

---

## 2D Toy example

To visualize label switching and compare unaligned averaging with optimal transport reduction:

```matlab
cd toy_examples
main_toy_example;
```

---

## Evaluated estimators

| Estimator         | Description         | Aggregation Type                       | Handles Label Switching | Handles $K_m \neq K$ | Communication Rounds |
| :---------------- | :------------------ | :------------------------------------- | :---------------------: | :------------------: | :------------------: |
| **OT-RED** (Ours) | Distributed MoE     | One-shot optimal transport (MM + IRLS) |           Yes           |         Yes          |     1 (one-way)      |
| **GLB**           | Centralized oracle  | Pooled EM on $\bigcup \mathcal{D}_m$   |           N/A           |         N/A          |     Centralized      |
| **GM**            | Greedy merging      | Pairwise KL reduction (Runnalls, 2007) |           Yes           |         Yes          |     1 (one-way)      |
| **AAVR**          | Aligned averaging   | Hungarian matching + parameter mean    |           Yes           |          No          |     1 (one-way)      |
| **WAVR**          | Weighted averaging  | Unaligned parameter mean               |           No            |          No          |     1 (one-way)      |
| **MED**           | Coordinate medoid   | Parameter-wise median                  |           Yes           |          No          |     1 (one-way)      |
| **FED**           | Federated averaging | Multi-round optimization ($T$ rounds)  |        Sensitive        |          No          |         $2T$         |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
