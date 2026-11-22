#!/bin/bash
# ==============================================================================
# RUN_BENCHMARK_VPS.SH
# Production runner for Distributed-MoE experiments on Linux VPS.
#
# Usage:
#   ./experiments/run_benchmark_vps.sh [mode]
#
# Modes:
#   all        : Run full pipeline (Data -> Global -> Machines -> N -> S -> Plots & Tables) [Default]
#   machines   : Run Figure 5 machine scaling benchmark (M in {4, 16, 64, 128})
#   baseline   : Run Centralized Global MoE baseline
#   samplesize : Run sample size scaling benchmark (N in {100k, 300k, 1M})
#   support    : Run support sample size S sensitivity study
#   report     : Generate all publication figures (PNG/PDF) and LaTeX tables
#   dev        : Quick test run (macOS/Linux dev verification)
#
# Example (run detached in background on VPS):
#   nohup ./experiments/run_benchmark_vps.sh all > logs/vps_run.log 2>&1 &
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$WORK_DIR"

mkdir -p logs results/figures results/tables results/global_baseline \
         results/scaling_machines_M results/scaling_samplesize_N results/sensitivity_support_S

# Detect MATLAB
if command -v matlab >/dev/null 2>&1; then
    MATLAB_BIN="matlab"
elif [ -f "/Applications/MATLAB_R2026b.app/bin/matlab" ]; then
    MATLAB_BIN="/Applications/MATLAB_R2026b.app/bin/matlab"
elif [ -f "/Applications/MATLAB_R2024b.app/bin/matlab" ]; then
    MATLAB_BIN="/Applications/MATLAB_R2024b.app/bin/matlab"
elif [ -f "/usr/local/MATLAB/R2024b/bin/matlab" ]; then
    MATLAB_BIN="/usr/local/MATLAB/R2024b/bin/matlab"
elif [ -f "/usr/local/MATLAB/R2024a/bin/matlab" ]; then
    MATLAB_BIN="/usr/local/MATLAB/R2024a/bin/matlab"
else
    echo "ERROR: MATLAB executable not found in PATH or standard directories."
    exit 1
fi

MODE="${1:-all}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/benchmark_${MODE}_${TIMESTAMP}.log"

echo "========================================================================"
echo "  DISTRIBUTED MIXTURE-OF-EXPERTS BENCHMARK SUITE"
echo "  Mode      : $MODE"
echo "  Directory : $WORK_DIR"
echo "  MATLAB    : $MATLAB_BIN"
echo "  Log File  : $LOG_FILE"
echo "========================================================================"

run_matlab_cmd() {
    local cmd="$1"
    local desc="$2"
    echo ""
    echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] STARTING: $desc"
    echo "    Command: $cmd"
    $MATLAB_BIN -batch "addpath('data'); addpath('datatools'); addpath('models'); addpath('stattools'); addpath('evaltools'); addpath('experiments'); addpath('reporting'); $cmd" 2>&1 | tee -a "$LOG_FILE"
    echo ">>> [$(date +'%Y-%m-%d %H:%M:%S')] COMPLETED: $desc"
}

case "$MODE" in
    dev)
        echo "Running fast DEV verification suite..."
        run_matlab_cmd "data_generator_benchmark" "Benchmark Data Generation"
        run_matlab_cmd "exp_global_centralized_baseline" "Global Baseline"
        run_matlab_cmd "exp_scaling_machines_M" "Machine Scaling Benchmark"
        run_matlab_cmd "plot_boxplots_by_machines" "Figure 5 Boxplot Generation"
        run_matlab_cmd "export_latex_benchmark_tables" "LaTeX Table Generation"
        ;;
    machines)
        run_matlab_cmd "exp_scaling_machines_M" "Machine Scaling Benchmark (Figure 5)"
        run_matlab_cmd "plot_boxplots_by_machines" "Generate Figure 5 Boxplots"
        run_matlab_cmd "export_latex_benchmark_tables" "Export LaTeX Tables"
        ;;
    baseline)
        run_matlab_cmd "exp_global_centralized_baseline" "Centralized Global MoE Baseline"
        ;;
    samplesize)
        run_matlab_cmd "exp_scaling_samplesize_N" "Sample Size N Scaling Benchmark"
        run_matlab_cmd "plot_boxplots_by_samplesize" "Generate Sample Size Boxplots"
        ;;
    support)
        run_matlab_cmd "exp_sensitivity_support_S" "Support Sample Size S Sensitivity Study"
        run_matlab_cmd "plot_curves_by_support_S" "Generate Support Size S Curves"
        ;;
    report)
        run_matlab_cmd "plot_boxplots_by_machines" "Figure 5 Boxplots (vs M)"
        run_matlab_cmd "plot_boxplots_by_samplesize" "Sample Size Boxplots (vs N)"
        run_matlab_cmd "plot_curves_by_support_S" "Support Size Curves (vs S)"
        run_matlab_cmd "export_latex_benchmark_tables" "LaTeX Tables"
        ;;
    all)
        echo "Executing FULL benchmark pipeline..."
        run_matlab_cmd "data_generator_benchmark" "Step 1: Benchmark Data Generation"
        run_matlab_cmd "exp_global_centralized_baseline" "Step 2: Centralized Global Baseline"
        run_matlab_cmd "exp_scaling_machines_M" "Step 3: Machine Scaling Benchmark (Figure 5)"
        run_matlab_cmd "exp_scaling_samplesize_N" "Step 4: Sample Size N Scaling Benchmark"
        run_matlab_cmd "exp_sensitivity_support_S" "Step 5: Support Size S Sensitivity Study"
        run_matlab_cmd "plot_boxplots_by_machines" "Step 6: Figure 5 Boxplot Generation"
        run_matlab_cmd "plot_boxplots_by_samplesize" "Step 7: Sample Size Boxplot Generation"
        run_matlab_cmd "plot_curves_by_support_S" "Step 8: Support Size S Curve Generation"
        run_matlab_cmd "export_latex_benchmark_tables" "Step 9: LaTeX Table Generation"
        ;;
    master)
        echo "Executing MASTER 2x3 FACTORIAL BENCHMARK (IID & Non-IID across 3 Profiles)..."
        run_matlab_cmd "data_generator_profiles" "Step 1: Multi-Profile Data Generation (Balanced, Moderate, Severe)"
        run_matlab_cmd "exp_iid_benchmark" "Step 2: Master Experiment 1 (IID Benchmark - Table 1)"
        run_matlab_cmd "exp_non_iid_benchmark" "Step 3: Master Experiment 2 (Non-IID Benchmark - Table 2)"
        run_matlab_cmd "export_latex_two_tables" "Step 4: Publication LaTeX Tables Export"
        ;;
    iid)
        run_matlab_cmd "exp_iid_benchmark" "Master Experiment 1 (IID Benchmark - Table 1)"
        ;;
    noniid)
        run_matlab_cmd "exp_non_iid_benchmark" "Master Experiment 2 (Non-IID Benchmark - Table 2)"
        ;;
    official)
        echo "Executing OFFICIAL DME BENCHMARK SUITE (Homogeneous & Heterogeneous Km)..."
        run_matlab_cmd "data_generator_profiles(100, 100000)" "Step 1: Generate 100 Homogeneous Datasets (3 Profiles)"
        run_matlab_cmd "data_generator_official_heterogeneous(100, 100000)" "Step 2: Generate 100 Heterogeneous Datasets (Km=3)"
        run_matlab_cmd "run_official_experiments(100000)" "Step 3: Run Official Experiments 1 & 2"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: official, master, iid, noniid, all, machines, baseline, samplesize, support, report, dev"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "  BENCHMARK COMPLETED SUCCESSFULLY: $(date)"
echo "  Log available at: $LOG_FILE"
echo "  Figures in      : results/figures/"
echo "  Tables in       : results/tables/"
echo "========================================================================"
