#!/usr/bin/env bash
# Launch 16 cold-start experiments: 4 topologies × 4 benchmarks
# Each runs 100 iterations of blind OpenEvolve with unbuffered output

set -e

LOGDIR="/data/home/yuhan/optpilot/logs/cold_start_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

TOPOLOGIES=(linear linear_loop star star_loop)
BENCHMARKS=(math humaneval hotpotqa swebench)

echo "=== Launching 16 cold-start experiments ==="
echo "  Log directory: $LOGDIR"
echo ""

for topo in "${TOPOLOGIES[@]}"; do
    for bench in "${BENCHMARKS[@]}"; do
        logfile="$LOGDIR/${topo}_${bench}.log"
        echo "Starting: ${topo} × ${bench} → $logfile"
        nohup python -u -m experiments.run_openevolve \
            --topology "$topo" \
            --benchmark "$bench" \
            --iterations 100 \
            --train 100 \
            --test 100 \
            --eval-tasks 20 \
            > "$logfile" 2>&1 &
        echo "  PID: $!"
    done
done

echo ""
echo "=== All 16 experiments launched ==="
echo "  Monitor: tail -f $LOGDIR/*.log"
echo "  Check status: ps aux | grep run_openevolve | grep -v grep | wc -l"
