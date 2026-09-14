#!/usr/bin/env python3
"""
train.py — entrypoint alias for run_simulation.py.

Named `train.py` at the operator's request, but per the architecture lock
there is no training here: this simply runs the connectome-driven LIF brain
bridged to flygym's *native*, already-working CPG controller (see
run_simulation.py for the actual implementation). No RL, no gradient
descent, no learned weights — synaptic weights are real neuPrint synapse
counts, and the walking controller is flygym's own HybridTurningFly.
"""

from run_simulation import main

if __name__ == "__main__":
    main()
