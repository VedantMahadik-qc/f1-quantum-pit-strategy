# Quantum Advantage for F1 Safety-Car Strategy

Solo research project investigating whether QAOA (quantum optimization) offers any genuine
advantage over classical methods for Formula 1 pit strategy decisions during Safety Car /
Virtual Safety Car periods.

**Origin note:** this idea started as ideation for the Qollab x IonQ hackathon, then got split
off into its own open-ended research project once it was clear the question deserved more room
than a 48-hour hackathon track (8–14 qubit cap, fixed deadline) allows. It's not tied to that
event, its rules, or its team — this is an independent project.

## The Question, Stated Honestly

"Does quantum computing help with F1 pit strategy?" is not really answerable at small scale,
and it's worth saying why up front: an 8–14 variable binary decision problem has at most
$2^{14} = 16{,}384$ possible states. A classical computer brute-forces that exactly in
microseconds. There is no speed advantage to chase at that size — none has ever been
demonstrated for QAOA on classical optimization problems at *any* size; this remains a
genuinely open research question in the field.

So "is there a quantum advantage" only becomes a real experiment if it's reframed as one of
these:

1. **Scaling study:** fix the SC pit-decision QUBO's structure (pit-lane throughput conflicts,
   tire thermal-cliff bias terms) but grow the decision window — more cars, multi-lap horizons,
   multi-team joint strategy — until exact brute force becomes impractical (roughly 25–30+
   variables). Then compare QAOA's *approximation ratio* and *time-to-good-solution* against
   classical heuristics (greedy, simulated annealing, OR-Tools with a matched time cutoff) — not
   against exact solvers, since those stop being tractable too. This is the closest thing to a
   fair "advantage" question.
2. **Structure study:** the pit-lane conflict graph arising from this problem is probably sparse
   / near-path-like (cars queue in a line), not a dense random QUBO. Does QAOA's performance at
   fixed depth $p$ degrade differently on this structured graph than on generic random instances
   of the same size? This is a legitimate, narrower research question independent of beating
   classical.
3. **Noise/shot-budget study:** how does solution quality degrade with shot count and circuit
   depth on the IonQ simulator (and optionally real IonQ hardware), and how does that compare to
   classical heuristics operating under an equivalent wall-clock budget? Useful empirical data,
   but be honest that current hardware/simulator latency is very unlikely to beat classical
   wall-clock time — frame this as a quality-per-resource comparison, not a speed claim.

Pick (1) if you want the most legitimate shot at an interesting result. Treat a negative result
("no advantage at these scales, here's the data showing why") as a fully valid, publishable
outcome — that's real research, not a failed demo.

## The Problem

During a Safety Car / VSC period, each car near the pit-entry decision point chooses to pit now
or stay out. Binary variable $x_i = 1$ if car $i$ pits this lap. Structure:

- **Linear bias $h_i$:** tire thermal-cliff degradation cost of staying out, fit from real
  lap-time decay curves.
- **Quadratic coupling $J_{ij}$:** pit-lane throughput penalty for cars likely to queue at pit
  entry on the same lap; track-position swap risk against rivals who stay out.

## Dataset

[FastF1](https://github.com/theOehrly/Fast-F1) (`pip install fastf1`) — official F1
timing/telemetry, lap times, tire compound/age, and per-lap `track_status` flags (filter
programmatically for SC/VSC codes rather than hard-coding a race).

## Findings So Far

**No quantum speed or quality advantage at these scales** — expected, per "The Question, Stated
Honestly" above. The actual results are structural/mechanistic, not a classical-beating one:

- **Scaling study (N=4–16, synthetic `path_like`/`random_sparse` instances):** plain QAOA tracks
  the exact optimum closely across the whole range. RQAOA (reps=1) degrades steadily as N grows —
  approx. ratio 1.00 → 0.84 (path_like) and 1.00 → 0.69 (random_sparse) by N=16.
- **Depth-rescue, and its limit:** a deeper inner QAOA circuit (reps=3) rescues RQAOA on
  `path_like` across the full N=4–16 range (ratio stays ≥0.92 throughout), but `random_sparse`
  keeps degrading regardless of depth — in fact *more* reps and/or more optimizer iterations made
  it measurably worse at N=16, ruling out "just not converged yet" as the explanation.
- **Mechanism:** using the true exact ground state as a referee at each RQAOA elimination step,
  `random_sparse` picks the wrong elimination sign ~42% of the time vs. ~9% for `path_like`, with
  errors spread through almost the entire recursion rather than confined to the endgame. This is
  RQAOA's known lack of a correctness guarantee (Bravyi et al. 2019) showing up concretely: a
  single-shot ⟨Z_i Z_j⟩ correlation estimate is just a much less reliable signal of the true
  ground-state relationship on denser/more frustrated graphs.
- **Real-race validation:** surveying SC/VSC periods across 8 races (2018–2023) confirms the
  pit-lane conflict graph really is sparse/path-like in practice (N≈15–19, density 0.02–0.15,
  approx. treewidth 1–3). Running the full exact/greedy/QAOA/RQAOA comparison on all 17 real
  periods found gives a clean, if uncomfortable, result: greedy (mean ratio 0.997) and plain QAOA
  (0.992) are reliably near-optimal on every single one, but RQAOA (mean 0.660, std 1.048) fails
  badly on 6 of 17 (35%) — including **two periods where it did worse than doing nothing**
  (negative/undefined approx. ratio vs. the trivial "nobody pits" baseline). An earlier look at a
  single hard instance suggested local hub nodes (high max-degree) predict RQAOA failure; that
  did *not* hold up across the full 17 — correlating RQAOA's ratio against max-degree, density,
  and treewidth all come out ≈0 (-0.14, -0.09, -0.06). Best read: those topology stats describe
  only which `J_ij` are nonzero, not their magnitudes or the `h_i` linear terms RQAOA's elimination
  choice actually depends on, so failures look instance/weight-specific rather than predictable
  from graph shape. Practical upshot: on real data, RQAOA isn't just occasionally suboptimal, it's
  unreliable enough (including anti-optimal outcomes) that plain QAOA or even greedy would be the
  safer choice, despite RQAOA being the literature's favored method for sparse graphs.
- **Weight-magnitude follow-up (also inconclusive):** since topology doesn't predict RQAOA's
  real-data failures, tested whether the actual magnitudes mattered instead — `h_i`/`J_ij` scale,
  coefficient of variation, the ratio between them, and the energy landscape's spectral gap
  (gap between the best and second-best distinct objective value) and near-degeneracy count.
  Excluding 7 degenerate periods where `h_i=0` everywhere (nothing is gained from pitting at all,
  not a fair test), none of these correlate meaningfully with RQAOA's ratio across the remaining
  10 real instances (strongest was the spectral gap at r≈0.30, not reliable at this sample size).
  Taken together with the topology result, this looks like a genuinely idiosyncratic failure mode:
  whether RQAOA succeeds depends on whether one specific QAOA-optimized state's correlation
  estimate happens to get a sign right, which isn't well-summarized by any single scalar feature of
  the instance tried so far. Practical implication holds regardless: there's no cheap pre-check for
  whether RQAOA is safe to trust on a given real instance — it needs validating against a classical
  baseline every time.
- **Shot-noise study:** every result above used exact analytic ⟨Z_i Z_j⟩ expectation values; this
  tested what happens under a realistic finite measurement budget instead (shots ∈ {100, 500,
  2000, 8000} vs. exact, N=14, 5 seeds/instance type). Plain QAOA behaves exactly as standard
  theory predicts — perfect on `path_like` at every shot level, and a clean monotonic improvement
  on `random_sparse` (ratio 0.967 → 1.000 as shots go from 100 to 8000), since it just takes the
  best of many sampled bitstrings. **RQAOA shows no such relationship.** Its elimination
  sign-error rate is flat/noisy on `path_like` (16–18% regardless of shot count) and genuinely
  non-monotonic on `random_sparse` (34% → 29% → 29% → 40% → 42% from 100 shots up to exact) —
  the *exact, noiseless* correlation estimate was the worst performer of all five levels tested.
  Consistent with every other result in this project: RQAOA's irrevocable per-step choices hinge
  on idiosyncratic optimizer-trajectory details that no single "more precision" knob (circuit
  depth, optimizer iterations, now measurement shots) reliably fixes. One reassuring note: on the
  realistic `path_like` structure, RQAOA's quality stays roughly stable (0.90–0.97) across the
  whole shot range — the shot budget doesn't add a new risk on top of the structural one already
  found.
- **A practical fix, partial but real:** every failure traced above points at the same root cause
  — RQAOA commits irrevocably to each elimination based on a *single* QAOA optimization's
  correlation estimate. If that unreliability is single-trajectory noise rather than a deep
  structural flaw, averaging independent trajectories should help. `solve_rqaoa` now supports
  `ensemble_k`: k independent QAOA restarts per elimination step, correlation matrices averaged
  before picking the pair/sign. At `ensemble_k=5` (5x the compute of one solve), the effect on the
  6 real periods that failed in the full 17-period sweep is striking: **3 fully fixed** (ratio
  →1.000, including the period that was previously anti-optimal), **1 nearly fixed** (0.841→0.948),
  **1 hugely improved but still poor** (-3.246→0.283), and **1 completely unchanged** (0.817→0.817,
  bit-for-bit identical). That last case is the interesting part: all 5 independent restarts
  agreed on the same wrong elimination at every step — not noise averaging out, but a *confidently
  wrong* answer reproduced consistently, which no amount of averaging at this configuration would
  fix. On synthetic `random_sparse` (N=14) the pattern is similar but weaker: sign-error rate
  dropped from 41.8%→29.1% and mean ratio from 0.716→0.774, still highly variable per seed
  (0.53–0.97). **Conclusion:** ensembling is a cheap, worthwhile practical mitigation — most
  real-data failures here really were single-trajectory noise — but not a guarantee; a genuinely
  distinct "confidently wrong" failure mode exists alongside the noisy one, so per-instance
  validation against a classical baseline is still necessary even with ensembling turned on.
  Follow-up at `ensemble_k=10` on the two still-weak real cases confirms this cleanly rather than
  closing the gap: the "confidently wrong" period (Germany lap 40) stayed at *exactly* 0.817 —
  bit-for-bit identical at k=1, 5, and 10, about as strong a confirmation of "deterministic, not
  noise" as this kind of test can give. The hugely-improved-but-still-poor period (Singapore lap
  19) got *worse* under more ensembling (0.283→0.113), fitting the pattern from every other
  "more resource" axis tried in this project (circuit depth, optimizer iterations, shot count):
  none of them give RQAOA a clean, monotonic path to reliability on its hard cases.
- **Cross-restart agreement: not the predictor either.** The obvious next question after the
  ensembling result above: can "noisy, ensembling will fix it" be told apart from "confidently
  wrong, it won't" *without* already knowing the classical answer — using only something visible
  at run time, like how much the `ensemble_k` independent restarts agree with each other?
  `solve_rqaoa` now exposes the pre-averaging per-restart correlation matrices (`restart_corrs_out`)
  to test this. Two agreement metrics per elimination step: `pair_agreement` (fraction of restarts
  whose own independent top-magnitude pair matches what the average elects) and `sign_agreement`
  (fraction of restarts agreeing on sign at that elected pair). Result: **not a clean signal.**
  `pair_agreement` is useless outright — every single one of the 6 real periods has at least one
  step where it hits exactly 0.0, regardless of outcome. `sign_agreement` shows a faint hint in the
  right direction on real data (confirmed-"confidently wrong" Germany lap 40 has the single highest
  mean sign-agreement, 0.900, of the 5 non-trivial real periods) but it's not usable: the
  still-largely-unfixed Singapore lap 19 (k5 ratio 0.283) ties *exactly* at 0.867 with the
  fully-fixed Australia lap 7 (k5 ratio 1.000) — identical agreement score, opposite outcomes. On
  synthetic N=14 data the direction isn't even consistent between instance types (`path_like`:
  agreement noticeably lower on wrong-sign steps, 0.800 vs 0.931 correct; `random_sparse`: no
  difference at all, 0.900 vs 0.897). **Conclusion:** restart disagreement, like every topology/
  weight/shot-count signal tried before it, does not reliably distinguish the two failure modes —
  the search for an advance predictor keeps coming up empty.
- **Performance note:** circuit simulation originally used Qiskit's pure-Python reference
  `StatevectorEstimator`; swapping to the `qiskit-aer` C++ backend (same exact noiseless
  statevector math) gave a ~7–10x wall-clock speedup, which is what makes N=16+ sweeps practical
  on a laptop.

See `scripts/` for the individual experiments and `results/` for the raw CSVs and plots behind
each claim above.

## What's Still Open

1. **Telling the two failure modes apart in advance:** ensembling cured most real failures but
   confirmed a genuinely distinct "confidently wrong" one (Germany lap 40 — identical at
   ensemble_k=1, 5, and 10). No instance-level predictor tried so far — topology, weight/landscape
   statistics, shot count, ensemble size, or (most recently) cross-restart agreement itself —
   distinguishes "noisy, ensembling will fix it" from "confidently wrong, no resource knob fixes
   it" ahead of time, short of just running RQAOA and checking against a classical baseline.
2. **Sweep infrastructure is single-threaded** despite every machine this has run on having many
   idle cores; parallelizing across seeds/configs would make larger/more-repeated sweeps practical.

## License

MIT — see [LICENSE](LICENSE).
