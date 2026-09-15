#!/usr/bin/env python3
"""
train.py — entrypoint alias for run_simulation.py.

Named `train.py` at the operator's original request, but there is no
training here: this runs run_simulation.py's CLI, which benchmarks the
connectome-driven LIF brain (VisionFlightBridge) standalone — no flygym, no
physics engine (removed in the vision-first pivot), no RL, no gradient
descent, no learned weights. Synaptic weights are real neuPrint synapse
counts; see run_simulation.py for the actual implementation.
"""

from run_simulation import main

if __name__ == "__main__":
    main()
