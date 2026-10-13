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
  approx. treewidth 1–3). Running the full exact/greedy/QAOA/RQAOA comparison on two real periods
  refines the picture further: on 2019 German GP (lap 56, N=15, density 0.11) every method matched
  the true optimum exactly. But on 2023 Australian GP (lap 7, N=18, density 0.15 — the
  structurally "hardest" real period found), RQAOA still underperformed (ratio 0.83) despite low
  *global* density/treewidth, because that graph has a local hub node (degree 6 of 18). So the
  safe-for-RQAOA takeaway isn't "path-like/low-density graphs in aggregate" so much as "no
  high-degree hub nodes creating competing strong correlations" — a sharper, real-data-grounded
  version of the structural hypothesis.
- **Performance note:** circuit simulation originally used Qiskit's pure-Python reference
  `StatevectorEstimator`; swapping to the `qiskit-aer` C++ backend (same exact noiseless
  statevector math) gave a ~7–10x wall-clock speedup, which is what makes N=16+ sweeps practical
  on a laptop.

See `scripts/` for the individual experiments and `results/` for the raw CSVs and plots behind
each claim above.

## What's Still Open

1. **Shot-noise / resource-budget study** (the third framing above): everything so far uses exact
   statevector simulation. No result yet on how solution quality degrades with finite shot count,
   compared against classical heuristics at a matched wall-clock budget.
2. **Broader real-race RQAOA validation:** only 2 of the ~17 real SC/VSC periods surveyed have had
   the full solver comparison run on them. The hub-node hypothesis above is one data point, not a
   confirmed pattern — worth checking against more real periods, especially ones with a high max
   degree relative to N.
3. **Sweep infrastructure is single-threaded** despite every machine this has run on having many
   idle cores; parallelizing across seeds/configs would make larger/more-repeated sweeps practical.

## License

MIT — see [LICENSE](LICENSE).
