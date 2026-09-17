# Metrology suite

Three standalone, **read-only** scripts that characterise what the
connectome-driven SNN in `run_simulation.py` actually does. None of them is
imported by the live pipeline (`telemetry_server.py`, `run_simulation.py`,
`physics_worker.py`, the dashboard), and none of them mutates saved state —
they load the graph, measure, print, and exit.

They exist because this project made, and then had to retract, a claim about
sensorimotor causality. They are kept in the repo as the record of how that
was caught.

| script | question it answers | runtime |
|---|---|---|
| `sham_causality_test.py` | Does olfactory stimulation move the motor neurons more than injecting the same current into an equally-sized set of unrelated neurons? | ~6 min |
| `lateralization_test.py` | Does *asymmetric* stimulation produce *asymmetric* motor output (i.e. steering)? | ~6 min |
| `ei_balance_trace.py` | Where along the ORN → premotor-DN pathway does the signal die — and is it cancelled by inhibition or diluted by fan-out? | ~2 min |

```bash
python3 sham_causality_test.py
python3 lateralization_test.py      # QUIET=True runs the silenced-network variant
python3 ei_balance_trace.py
```

## What they found

**1. Sham-controlled causality — null.**
An uncontrolled first pass appeared to show ORN stimulation raising motor
activation by ~14%. Under a paired, interleaved, size-matched sham control
that number collapsed to ~0.8% and failed to beat the sham
(`t(4) = −0.24 / −0.44`). The original 14% was network drift, not response.

The sham was *deliberately weak*: drawn from neurons with no synaptic path to
`vnc_motor` within 4 hops, 63% of which have zero out-degree, versus ORN's
mean out-degree of 42.6. ORN had a ~30× connectivity advantage and still did
not win.

**2. Lateralization — null, twice.**
Mirror-image design (left-odour vs right-odour in alternating blocks), so
drift and global excitability changes cancel in the difference-of-asymmetries
— the exact confound that invalidated the first test.

- Normal background: steering statistic `t(4) = +0.99`, sign inconsistent
  across cycles.
- **Fully silenced network** (`noise_std = 0`, `sensory_drive = 0`, verified
  baseline of **0 spikes/step**): `t(4) = +0.085`, effect ~40× *smaller* than
  with noise present.

Removing the noise floor did not reveal a masked signal, which refutes the
signal-to-noise-masking hypothesis. The informative detail: unilateral odour
does propagate (320–530 neurons activate) but produces a **bilaterally
symmetric** motor response — MN·L and MN·R differ by only ~4%. The pathway
carries the signal and loses the side.

**3. E/I balance trace — no inhibitory gate; dilution instead.**

| hop | net current | E/I | neurons reached | % of brain |
|---|---|---|---|---|
| 1 | +125.6 | ∞ | 627 | 0.4% |
| 2 | +1035.0 | 1.82 | 11,379 | 6.4% |
| 3 | +141.8 | 1.42 | 70,654 | 40.0% |
| 4 | +883.7 | 2.20 | 158,770 | **90.0%** |

Net current is **positive at every hop** — the signal is never cancelled. The
pathway is *less* inhibited than the graph average (global E/I = 1.64). The
premotor DNs receive net excitation throughout.

The mechanism is geometric: by hop 4 the stimulus has reached 90% of the
network, and only **~0.3%** of the delivered excitatory current lands on the
981 premotor DNs. No single inhibitory population accounts for more than
**1.2%** of inhibition onto them — a real gate would show one type at 20–40%.

## Conclusion

The connectome topology is correct and verified (neurotransmitter polarity,
synapse-count scaling, glomerular identities, premotor DN set). What is not
supported by measurement is *spatial* sensorimotor transformation: with
**homogeneous** LIF parameters applied to all 176,422 neurons, lateral
information does not survive the fan-out.

`male-cns:v1.0` supplies topology and neurotransmitter identity. It does not
supply per-cell-type thresholds, time constants, or gains — the parameters
real circuits use to compute. Introducing those would require
electrophysiological data that exists for only a small fraction of these cell
types.

This is why `physics_worker.set_forward_drive` locks the efferent mapping to
**symmetric forward drive** (overall network excitation → walking speed, which
works) and implements no yaw term. Adding one would amplify measurement noise
into fabricated steering.

## Caveats on the methods themselves

- `ei_balance_trace.py` is a **linear** propagation over the weight matrix. It
  ignores thresholds, refractoriness, synaptic delay and saturation, so it
  describes what the *wiring* delivers per hop, not what the spiking network
  does. It rectifies at each hop (`max(net, 0)`) because propagating signed
  activity lets a negative value through an inhibitory synapse produce
  *positive* current — an earlier version without rectification reported a
  spurious net-inhibitory hop.
- All statistics are `n = 5` cycles (`df = 4`). That is adequate for the
  observed nulls (t-values of 0.09–0.99 against a 2.776 threshold) but would
  be underpowered for resolving a small real effect. The premotor-DN response
  in the sham test (`+0.63 Hz`, `sem 0.87`) remains genuinely unresolved
  rather than proven absent.

---

# Phase 2 — dopaminergic plasticity (mushroom body)

Three-factor STDP on the real KC→MBON layer: 44,042 plastic synapses
(KC=4,064, MBON=97), eligibility trace × postsynaptic coincidence ×
dopamine. Findings, in the order they were forced by measurement.

**1. The sign was backwards.** The original rule potentiated on reward. In
*Drosophila*, DAN activity **depresses** KC→MBON synapses (Hige et al. 2015;
Cohn et al. 2015; Owald & Waddell 2015) — learning removes an MBON's vote
rather than strengthening it. Corrected to depression.

**2. A global D(t) is wrong.** PAM and PPL innervate different compartments,
so a signed PAM−PPL scalar lets them cancel and applies one valence's
dopamine to the other's synapses. Compartment membership is read from real
DAN→MBON wiring (2× dominance margin): **38 PAM-dominant, 57 PPL-dominant
MBONs** → 21,573 / 21,339 plastic synapses.

**3. Unsigned levels exposed a small-N artifact.** Resting levels, measured
with learning frozen: PAM **0.04632 ± 0.01033** (316 cells), PPL **0.10821 ±
0.02400** (24 cells) — a 2.34× ratio that is population-size bias, not
excitability. Without correction the PPL compartment depressed *continuously
at rest*. Fixed by subtracting each population's own baseline
(`max(0, level − baseline)`), the functional analogue of dopamine clearance.
Verified: **0 of 44,042 synapses move in an unstimulated network.**

**4. η had to be sized arithmetically.** 0.15 → 0.05 → 0.005 all saturated
against the −90% clip floor (19,283 of 21,573 synapses at 0.05). Cumulative
depression per episode ≈ `η · elig · d_dopa · sub_dt · 960`; with elig≈50 and
d_dopa≈0.22 that is ~26× the weight at 0.005. Set to **5e-5** (~25%
depression). Confirmed: 0 synapses at the floor.

**5. Valence specificity is structurally impossible here.** PAM↔PPL
connectivity is **192 + 453 = 645 synapses, all excitatory, zero
inhibitory**. Stimulating PAM alone raises PPL almost equally
(D_PAM/D_PPL = **1.09**). No individual DAN subtype is selectively
activatable either — PAM08/PAM01/PAM04/PAM06 all give PAM/PPL spike-fraction
ratios of 0.47–0.50, indistinguishable from each other, and PPL fires ~2×
PAM regardless of what is injected.

Consequence: with the full biological network, dopaminergic learning is
**real but generalised, not associative** — both compartments depress
roughly equally (−3.55% vs −3.78% under a PAM volley).

## The synthetic clamp (UI toggle, off by default)

Because no biological route to valence specificity exists, an explicitly
labelled synthetic override is provided. Two versions were tried:

- **Winner-take-all on measured EMAs — failed.** Crosstalk equalises them
  (d_PAM 0.2264 vs d_PPL 0.2284, a 0.9% gap), so the winner flipped randomly
  per sub-step; net specificity **+0.50 pp**, i.e. nothing.
- **By intention — works.** The clamp keys off which valence the *user*
  injected, latched for one dopamine time constant (750 steps). This is
  *more* synthetic: it ignores network state and obeys the button.

Validated bidirectionally, weights restored between episodes and dopamine
EMAs pre-warmed to steady state:

| condition | PAM comp | PPL comp | specificity |
|---|---|---|---|
| clamp OFF + fire PAM | −3.550% | −3.776% | −0.23 pp |
| clamp ON + fire PAM | −5.357% | **+0.000%** | +5.36 pp |
| clamp ON + fire PPL | **+0.000%** | −8.020% | −8.02 pp |

## Methodological note

Four favourable results in this phase turned out to be artifacts, each
caught only by a follow-up control: a +14% causality effect that was drift;
all-zero subtype selectivity that was EMA lag after `reset_state()`; a
+54.89 pp clamp specificity that was EMA warm-up between episodes; and
apparent PPL specificity that was a cold EMA never crossing its baseline.
Every *null* result survived its controls. Tests here therefore pre-warm the
EMAs, restore weights between conditions, and state falsifiable pass/fail
criteria before running.
