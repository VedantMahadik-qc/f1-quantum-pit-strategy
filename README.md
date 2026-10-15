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
  approx. ratio 1.00 → 0.91 (path_like) and 1.00 → 0.76 (random_sparse) by N=16. *(Re-verified after
  the seed-bug fix below; see that note for what changed and why.)*
- **Depth-rescue, and its limit:** a deeper inner QAOA circuit (reps=3) rescues RQAOA on
  `path_like` across the full N=4–16 range (ratio stays ≥0.97 throughout), but `random_sparse`
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
  sign-error rate is noisy and non-monotonic on both instance types — `path_like` ranges 3.6–14.5%
  with no shot-count trend (100→14.5%, 500→7.3%, 2000→3.6%, 8000→7.3%, exact→9.1%) and
  `random_sparse` similarly bounces 29–42% (34% → 29% → 29% → 40% → 42% from 100 shots up to
  exact) — the *exact, noiseless* correlation estimate was the worst performer of all five levels
  tested on `random_sparse`. Consistent with every other result in this project: RQAOA's
  irrevocable per-step choices hinge on idiosyncratic optimizer-trajectory details that no single
  "more precision" knob (circuit depth, optimizer iterations, now measurement shots) reliably
  fixes. One reassuring note: on the realistic `path_like` structure, RQAOA's quality stays
  roughly stable (0.93–0.99) across the whole shot range — the shot budget doesn't add a new
  risk on top of the structural one already
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
- **A per-step confidence signal that finally works — but only at the step level, not the run
  level.** `margin` (the top elected correlation minus the runner-up pair's, already logged per
  elimination step) was re-examined as a zero-cost check before writing any new code, and
  confirmed null again at 4x the data (440 steps, N=14, 20 seeds/instance_type: pooled correlation
  0.015, p=0.75, error rate flat 15–24% across quartiles — see
  `scripts/investigate_step_confidence_signals.py`). But a *different* signal from the exact same
  single `k=1` run — `abs_top_corr`, the raw magnitude of the correlation value RQAOA actually
  elects, not its margin over the runner-up — is genuinely predictive: pooled correlation **0.30**
  (p=7.7e-11), with per-step sign-error rate dropping monotonically from 31% in the lowest quartile
  to 3% in the highest (29% below-median vs 7% above-median). This is the first signal in the whole
  series with a real, statistically solid effect, and it costs nothing extra — RQAOA already
  computes it on the single run it does anyway. **But it doesn't answer item 1 below.** Run-level
  aggregates (min/mean `abs_top_corr` across all steps) of the same 6 known-failing real periods do
  *not* separate "confidently wrong" Germany lap 40 from the periods ensembling fixes — Germany lap
  40's min (≈0) is matched by fully-fixed Azerbaijan lap 30's min (≈0) too, and the means don't
  order by fixability either (Azerbaijan: fixed, mean 0.224; Australia lap 7: fixed, mean 0.600).
  So `abs_top_corr` is a real within-run tool — useful for flagging which *individual* elimination
  steps in a run are shaky — but, like everything before it, gives no advance read on whether a
  *whole run* will land in the noisy-and-fixable bucket or the confidently-wrong one.
- **Selective ensembling, built on that signal, cuts the cost of the practical fix — mostly.**
  `solve_rqaoa` now takes `confidence_threshold`: run one restart per elimination step, and only
  spend the remaining `ensemble_k - 1` restarts on that step if its own `abs_top_corr` is below the
  threshold (0.5, the below/above-median split point from the finding above), instead of ensembling
  every step uniformly. Reran all 6 known-failing real periods fresh — k=1, full `ensemble_k=5`, and
  selective `ensemble_k=5`, same seed, same pass — rather than trusting previously-recorded numbers
  (see `scripts/investigate_selective_ensembling.py`). On **4 of 6** periods, selective matches or
  slightly *beats* full ensembling's ratio (Azerbaijan lap 30 and Australia lap 7 both still hit
  1.000; Australia lap 53 actually edges full ensembling, 0.960 vs 0.948; Germany lap 2 matches
  1.000) while spending only **50–79%** of its restart budget (mean 70% across all 6 periods).
  Germany lap 40, the established "confidently wrong" case, stays at exactly 0.817 under selective
  too — expected and reassuring, since no amount of ensembling was ever going to move it. The one
  real wrinkle: Singapore lap 19 — already flagged above as getting *worse* under more ensembling
  at `ensemble_k=10` (0.283→0.113) — gets worse again under selective ensembling, and dramatically
  so (0.283→**-3.492**, actually below even its own `k=1` ratio of -3.246). This isn't a new failure
  mode selectivity introduces; it's the same already-erratic case reacting badly to yet another
  restart-count variation, consistent with (not contradicting) the pattern already on record.
  **Conclusion:** selective ensembling is a real, if partial, compute win on the periods where
  ensembling was already known to help or not matter — but it inherits, rather than fixes, that one
  case's instability, so it's not a safe blind substitute for full `ensemble_k` without per-instance
  validation either.
- **Why Singapore lap 19 specifically destabilizes: a case-study diagnosis (2 periods, not a
  generalized/validated rule yet).** First ruled out a boring explanation: its exact objective
  magnitude (|−0.261|) isn't unusually small — Azerbaijan lap 30's is even smaller (|−0.230|) and
  that period is rock-solid across every configuration tried. Both periods also tie *exactly* at
  3/15 sign errors at `k=1` against the classical referee — so sign-error *count* doesn't explain
  the gap either. The actual mechanism, traced step-by-step
  (`history_out`'s `top_corr`/`restarts_used`, comparing k=1/full-k5/selective-k5 side by side):
  **(1)** Some of Azerbaijan's wrong picks (4 of 8, in its full-ensemble run) land among variables
  with *exactly zero* correlation (`top_corr≈0.000`, i.e. genuinely disconnected in the conflict
  graph) — flipping a coin there costs nothing, since there's no edge weight for the sign to get
  wrong. But the other 4 are moderate-magnitude (`top_corr` -0.31 to 0.23), real, substantive wrong
  picks that *still* didn't stop that run from landing on the exact optimum (ratio 1.000 despite
  8/15 total sign errors — more errors than any Singapore run, in fact). Singapore, by contrast, has
  no zero-correlation freebies anywhere in any of its three runs — every single wrong pick, in
  every run, sits on a moderate-to-strong edge (`top_corr` 0.13 to 0.99). So Azerbaijan isn't just
  "lucky" about which variables happen to be disconnected — its back-substitution appears
  genuinely more forgiving of moderate sign errors in general, for reasons not yet isolated here;
  Singapore's isn't forgiving of any of them. **(2)** A confident single restart does *not* predict
  what a 5-restart average will do: at step n=16, selective's lone (threshold-passing) restart and
  full ensembling's first restart are bit-identical (same seed) and would have picked `RUS`/`NOR`
  alone — but averaging in the other 4 restarts flips full ensembling's actual choice to a
  completely different pair, `PIA`/`ALB`. This is the same fact already on record as
  `pair_agreement` saturating at 0.0 on every real period (see the cross-restart-agreement finding
  above), just caught in the act at one concrete fork. **(3)** Past that fork, selective's path
  happens to chain 4 *consecutive* wrong picks through a small set of recurring hub variables
  (`VER`, `PIA`, `ALB`), which compounds through back-substitution far worse than the same number
  of wrong picks scattered across unrelated variables, as happens in the other two runs. **Net:**
  this period's instability isn't a metric artifact or a simple error-rate problem — it's that
  Azerbaijan's structure tolerates sign errors (disconnected or not) in a way Singapore's doesn't,
  compounded by which restart happens to win a handful of pivotal, early-ish forks (not how
  confident any of them look) deciding whether the whole run lands in a good or catastrophic final
  basin. *Why* Azerbaijan's back-substitution is broadly error-tolerant while Singapore's isn't is
  itself still unexplained. Not yet checked against the other 4 periods or turned into a predictive
  rule (e.g. "fraction of zero-correlation tail," "back-substitution error tolerance," or "hub
  recurrence count") — a natural next step if this thread is picked back up.
- **A `path_like` seed bug, found and fixed, and every affected number re-verified.** While
  parallelizing the sweep infrastructure (next bullet), found that `f1qps.experiments.runner`'s
  (and several `investigate_*.py` scripts') generator dispatch called `path_like_instance(n, seed)`
  *positionally* — but that function's signature is `(n, h_scale, j_scale, seed=None)`, so the call
  silently bound "seed" to `h_scale` instead, leaving the real `seed` at its default, `None` (fully
  unseeded/random), every time. Confirmed via `results/scaling_study.csv`: `path_like` at `seed=0`
  gave `objective=0.0` for every method at every N — exactly what `h_scale=0` (an all-zero linear
  term) produces. Predates this session; fixed the same way `random_sparse`'s wrapper already was,
  then reran every affected study. Two of the four numeric claims above changed: the N=4–16
  scaling-study endpoint moved from 1.00→0.84 to 1.00→**0.91** (path_like, reps=1, N=16), and the
  depth-rescue floor moved from "stays ≥0.92" to "stays ≥**0.97**" — both *improvements*, since the
  old unseeded draws apparently included harder-than-typical instances (one `h_scale=0` case is
  outright degenerate). The shot-noise `path_like` sign-error-rate characterization also changed,
  from "flat 16–18%" to a lower, still-noisy 3.6–14.5% range (see that bullet). The other two held
  exactly: the elimination-correctness "~9% for path_like" claim re-ran to 9.09% — bit-identical to
  the old (buggy-seed) run, meaning that particular diagnostic is apparently insensitive to which
  random `path_like` draw is used. The cross-restart-agreement `path_like` numbers above were *not*
  rerun (lower priority; flagged in "What's Still Open"). One more thing surfaced while
  re-verifying: `random_sparse` at N=16 (reps=3, `scaling_study_reps3.csv`) also shifted on 2 of 5
  seeds (e.g. 0.628→0.253) despite never being touched by the seed bug. Traced this to environment/
  library drift, not a code bug — confirmed by running the exact historical `rqaoa.py`/`qaoa.py`
  (from the commit that generated the old CSV) against today's installed qiskit/numpy and getting
  the *same new* numbers, while a byte-diff of both files' relevant code paths shows zero change.
  COBYLA's optimization landscape is apparently chaotic enough that some seeds sit on a knife-edge
  where a different qiskit-aer/numpy version alone tips the result to a different local optimum —
  consistent with, and reinforcing, this project's central finding about RQAOA's fragility, just
  from an unexpected angle (library version, not algorithm choice).
- **Sweep infrastructure is now parallel.** `sweep_synthetic`'s new `max_workers` param distributes
  independent `(instance_type, N, seed)` cells across a `ProcessPoolExecutor` instead of running
  them one at a time — each cell is a self-contained QAOA/RQAOA simulation with no shared state,
  ideal for this. Needed one extra fix to actually work: each worker process's numpy/qiskit-aer
  defaults to spinning up its own full-width, multi-threaded BLAS/OpenMP pool, so `max_workers`
  workers simultaneously oversubscribes threads by `worker_count x core_count` — confirmed this
  crashes outright (`OpenBLAS: Memory allocation... giving up`, then `BrokenProcessPool`) at 8
  workers on a 16-core machine. Fix: cap each worker to a single BLAS/OpenMP thread via environment
  variables before spawning the pool, since the process-level parallelism is already the intended
  source of speedup here, not any inner library threading. With that fix, 8 workers on the same
  16-core machine gave a **4.31x wall-clock speedup** (63.4s → 14.7s) on a 48-instance QAOA/RQAOA
  benchmark sweep (N=8–12, 4 seeds, both instance types) — sublinear (process-spawn/import
  overhead, OS scheduling) but a solid real win. `run_scaling_study.py` now uses
  `os.cpu_count()` workers by default.
- **Performance note:** circuit simulation originally used Qiskit's pure-Python reference
  `StatevectorEstimator`; swapping to the `qiskit-aer` C++ backend (same exact noiseless
  statevector math) gave a ~7–10x wall-clock speedup, which is what makes N=16+ sweeps practical
  on a laptop.

See `scripts/` for the individual experiments and `results/` for the raw CSVs and plots behind
each claim above.

## What's Still Open

1. **Telling the two failure modes apart in advance, at the RUN level:** ensembling cured most real
   failures but confirmed a genuinely distinct "confidently wrong" one (Germany lap 40 — identical
   at ensemble_k=1, 5, and 10). No *run-level* predictor tried so far — topology, weight/landscape
   statistics, shot count, ensemble size, cross-restart agreement, or (most recently) run-level
   aggregates of per-step correlation magnitude — distinguishes "noisy, ensembling will fix it"
   from "confidently wrong, no resource knob fixes it" ahead of time, short of just running RQAOA
   and checking against a classical baseline.
2. **Selective ensembling (tested — real but partial win) leaves an open thread of its own:**
   matched or beat full `ensemble_k=5` on 4/6 real periods at 50–79% of its restart cost, but made
   Singapore lap 19 dramatically worse (0.283→-3.492) rather than just "not better." *Why* is now
   diagnosed as a case study (see the finding above: no harmless disconnected tail + hub variables +
   restart-level pair disagreement at a pivotal fork) — but only checked on this one period against
   Azerbaijan lap 30 as a contrast. **Still open:** does either structural signal (fraction of
   zero-correlation "free" variables late in the recursion, or how often the same few variables
   recur as elimination targets) generalize into something that predicts *which* periods are
   restart-count-sensitive, across all 6 (or more) real periods — not just explain this one in
   hindsight?
3. **The cross-restart-agreement `path_like` numbers weren't rerun** under the seed-bug fix (lower
   priority than the four main findings, which were all reran and are confirmed above) — if that
   investigation is revisited, redo its `path_like` half first.
4. **Several hand-rolled `investigate_*.py` scripts remain unparallelized**
   (`investigate_shot_noise.py`, `investigate_rqaoa_ensemble_fix.py`,
   `investigate_restart_disagreement.py`, `investigate_rqaoa_gap.py`) — only the scripts built on
   `sweep_synthetic` (`run_scaling_study.py`, `investigate_rqaoa_depth_full_range.py`,
   `investigate_random_sparse_rqaoa_failure.py`) got wired up to its new `max_workers`; the rest
   would need the same `ProcessPoolExecutor` + single-BLAS-thread treatment applied individually.

## License

MIT — see [LICENSE](LICENSE).
