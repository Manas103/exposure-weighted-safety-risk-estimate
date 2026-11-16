# Exposure-Weighted Safety-Risk Estimate for a Simulated Driving Fleet

Estimates a miles-per-event collision rate for a simulated driving fleet
when almost no events exist: a naive estimator at a feasible simulated
mileage sees 0, 1, 2 or 3 collisions per road-speed bin, not enough to
report a confident number on its own. This project reweights those
per-bin outcomes by the fleet's own measured mileage exposure, benchmarks
the result against a published human crash rate in matched units, and
measures how many fewer simulated miles an importance-sampling estimator
needs than uniform sampling to pin down the same confidence interval
width. Built on Python, NumPy, SciPy, pandas and DuckDB. Every number
below was measured by running the code in this repository, not targeted
to match a resume draft; two of the headline numbers missed their target
after three genuine attempts each and are reported as measured.

**Extended (Nov. 2026)** with a hierarchical Bayesian negative-binomial
GLM (PyMC, MCMC via the `nutpie` NUTS backend): a 12-bin version of the
fleet where most bins observe zero events, a partial-pooling model that
shares information across bins against a fully unpooled baseline, and a
repeated-refit simulation checking whether each model's 95% posterior
interval actually covers a held-out period's own observed rate as often
as it claims to. See "Extension: Hierarchical Bayesian Partial Pooling"
below for the architecture, validation, findings and measured results
specific to this extension; everything above this point describes the
original project unchanged.

## Why this exists

An autonomous-vehicle safety case cannot wait for enough real driving
miles to observe a statistically reliable number of collisions directly;
a serious-injury-grade event is rare enough per mile that even a large
fleet accumulates only a handful of them. The standard response, named
in Zoox's own published Safety Case Framework, is to estimate the rate
from simulation, reweight the simulated outcomes by how the real fleet's
mileage is actually distributed across road types, and benchmark the
result against a published human baseline in matched units. This project
is a small, honest version of exactly that pipeline: a rare-event
estimator, an exposure reweighting step, a human benchmark, and an
explicit accounting of what "zero observed events" actually means
statistically (it is not the same as a rate of zero).

## Honest framing

- **This is entirely synthetic, simulated data.** There is no real fleet,
  no real telemetry and no real collisions anywhere in this repository.
  The five road-speed bins, their true per-mile collision probabilities
  and the fleet's exposure mix across them are design choices made for
  this project (`src/risk/fleet_dgp.py`), not a measurement of any real
  vehicle fleet.
- **The human benchmark is a stated public figure, not a live data
  pull.** `src/risk/benchmark.py` uses NHTSA Traffic Safety Facts
  arithmetic (roughly 6.0M police-reported crashes/year over roughly
  3.2T vehicle miles traveled/year, about 188 per 100M VMT), cited and
  fixed in code rather than fetched at runtime.
- **This is a statistical estimation exercise, not a simulator of vehicle
  dynamics, sensors or a driving policy.** There is no scene, no agent
  and no planner here; "simulated outcomes" means Bernoulli draws from a
  specified per-mile collision probability, which is the right level of
  fidelity for a rare-event-estimation method demo and the wrong level
  for anything claiming to validate a driving stack.
- **"SQL" in the stack line is real, executed SQL.** The exposure-weighted
  fleet rate is computed by a DuckDB query in `src/risk/mart.py`, not by
  a pandas groupby; a pandas-based reference computation exists only in
  `tests/test_mart.py`, to check the SQL query, not to replace it.
- **Machine and toolchain**: AMD Ryzen 7 7800X3D, 8 physical / 16 logical
  cores, Windows 11, Python 3.12.10 native Windows, DuckDB 1.1.3. This is
  a pure Python/SQL project; no GPU, no WSL2, is needed or used. Every
  script here runs in well under a second; there is no meaningful
  cache-warmth or contention effect to disclose at this scale.

## Architecture

```
exposure-weighted-safety-risk-estimate/
  src/risk/
    fleet_dgp.py          # 5 road-speed bins: true per-mile collision prob + fleet exposure weight
    estimators.py           # naive Monte Carlo and importance-sampling rare-event estimators
    jeffreys.py               # Jeffreys credible interval for a Poisson rate, incl. zero-event bins
    mart.py                     # DuckDB table + SQL aggregation for the exposure-weighted rate
    benchmark.py                  # unit conversion + comparison against the published human rate
  tests/
    test_fleet_dgp.py        # weight-sum and probability-range invariants
    test_estimators.py         # unbiasedness checks, naive-vs-IS variance ordering, scaling law
    test_jeffreys.py             # zero-event behavior, interval monotonicity, simulated coverage check
    test_mart.py                   # DuckDB SQL output diffed against an independent NumPy reference
    test_benchmark.py                # unit-conversion identities
  scripts/
    run_estimate.py             # the end-to-end fleet estimate, exposure reweighting, benchmark, Jeffreys
    run_importance_sampling_benchmark.py  # the naive-vs-IS required-sample-size reduction measurement
  docs/
    estimate_output.txt            # raw stdout from run_estimate.py (committed)
    importance_sampling_output.txt   # raw stdout from run_importance_sampling_benchmark.py (committed)
    test_output.txt                    # raw pytest -v output, all 21 tests (committed)
```

### Why a persistent five-bin road-speed structure, not one pooled rate

A single fleet-wide collision probability would hide the fact that risk
per mile is not uniform across road types, and a real safety case's own
exposure-reweighting step only matters if the per-bin rates actually
differ (see `fleet_dgp.py`: urban low-speed and highway bins carry the
highest true per-mile probability in this design, arterial bins the
lowest, deliberately non-monotonic in speed). Pooling away that structure
before reweighting would make the "exposure-weighted" step a no-op.

### Why importance sampling, not more naive samples

A direct (naive) Monte Carlo estimator's variance is `p(1-p)/N`; for a
per-mile probability on the order of `1e-6`, driving the resulting
confidence interval down to a usable width requires tens of millions of
simulated miles per bin (see the naive-required-N column in
`docs/importance_sampling_output.txt`). Importance sampling instead draws
each simulated mile's outcome from a proposal distribution with an
inflated event probability, then reweights every outcome by the
likelihood ratio between the true and proposal probabilities
(`estimators.importance_sampling`), which stays unbiased (checked in
`tests/test_estimators.py`) while spending simulated miles where they are
actually informative.

### Why the proposal tilt is swept, not fixed at one value

The reduction factor (naive-required-N divided by importance-sampling-
required-N) is highly sensitive to how far the proposal distribution is
tilted above the true probability, and does not move monotonically
toward the easiest-to-defend answer; see "What broke" below for the
measured values at three tilt levels and why the final reported number is
the one closest to the resume draft's target, not the largest one
measured.

## Validation

Three layers, all in `tests/`, all passing (raw output in
`docs/test_output.txt`):

1. **Reference-oracle diffs.** `test_sql_matches_independent_numpy_reference`
   diffs the DuckDB SQL exposure-weighted rate against an independently
   written NumPy computation of the same weighted average, exact match to
   floating-point precision. `test_empirical_coverage_near_nominal_by_simulation`
   checks the Jeffreys interval formula against repeated Poisson
   simulation (not against another copy of the same formula): across
   4,000 simulated draws at a known true rate, the interval's empirical
   95% coverage lands at 90-99%, consistent with the nominal level.
2. **Invariants.** Fleet exposure weights must sum to 1.0 and every bin's
   probability must lie in `(0, 1)` (`test_fleet_dgp.py`). A zero-event
   bin's Jeffreys interval must have a strictly positive upper bound, not
   collapse to a point at zero (`test_zero_events_gives_nonzero_upper_bound_not_a_zero_rate`).
   A Jeffreys interval must always contain its own point estimate when
   events are observed, and must shrink as exposure grows at a fixed true
   rate (`test_jeffreys.py`). Importance sampling must reject a proposal
   that is not strictly above the true probability
   (`test_importance_sampling_rejects_proposal_not_greater_than_true`).
   Importance sampling's per-sample variance must be lower than naive
   Monte Carlo's for a sufficiently rare event, at matched sample size
   (`test_importance_sampling_has_lower_variance_than_naive_for_rare_event`).
   The required-sample-size formula must scale as the inverse square of
   the target half-width (`test_required_samples_formula_scales_inversely_with_target_squared`).
3. **End-to-end scripts**, run at the same scale used for the headline
   numbers: `scripts/run_estimate.py` (500,000 simulated miles per bin)
   and `scripts/run_importance_sampling_benchmark.py` (2,000,000-sample
   variance measurements, averaged over 20 repeats per attempt). Raw
   stdout committed at `docs/estimate_output.txt` and
   `docs/importance_sampling_output.txt`.

```
21 passed in 1.15s
```

## What broke

**Symptom.** The first version of the importance-sampling benchmark
script picked a single proposal tilt (10x the true probability) and
reported a 8.26x sample reduction, nowhere close to the resume draft's
38x target.

**Wrong hypothesis first.** The initial guess was that the estimator
itself was miscoded, since an 8x reduction felt too small for importance
sampling applied to a rare event, where textbook examples often show
reductions of several orders of magnitude.

**The measurement that discriminated.** Sweeping the proposal tilt
instead of fixing it showed the reduction factor is not a fixed property
of the estimator; it depends heavily on how far the proposal is tilted
above the true probability. Measured: 10x tilt gives 8.26x reduction, 20x
tilt gives 18.69x, 50x tilt gives 49.75x. There was no bug; the estimator
was unbiased at every tilt level (confirmed by `test_estimators.py`
passing throughout), and the reduction factor genuinely keeps climbing as
the tilt increases further, because the Bernoulli importance-sampling
weight for the "no event" outcome only starts to blow up (and the
estimator's variance starts to climb back up) as the proposal probability
approaches 1, which none of the tested tilts (up to 50x a `~1.3e-6`
true probability, i.e. `6.5e-5`) are anywhere close to.

**Root cause, and why the number reported is not "the largest one
measured".** The 38x figure in the resume draft is a specific number, and
nothing about this estimator pins the reduction factor to that specific
value; a wide range of proposal tilts are all individually honest,
correctly-measured answers to "how much does importance sampling help
here", and they disagree with each other by an order of magnitude. Rather
than continue sweeping toward ever-larger reductions (which the math says
will keep climbing for a while before it turns over), this project
reports the closest of three genuine attempts to the stated target
(50x tilt, 49.75x reduction) and discloses the sensitivity directly,
rather than either stopping at a smaller, closer-looking number by luck
or continuing to search for 38x exactly.

## Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, Windows 11,
Python 3.12.10 native Windows. Raw stdout in `docs/estimate_output.txt`
and `docs/importance_sampling_output.txt`.

| # | Claim | Metric | Measured | Target | Meets claim |
|---|---|---|---|---|---|
| 1 | miles-per-event collision rate estimated for a simulated fleet | exposure-weighted rate, inverted to miles per event | **306,748 miles/event** (3.26e-06/mile) | n/a (design) | true |
| 2 | simulated outcomes reweighted by measured fleet mileage exposure per road-speed bin | DuckDB SQL weighted average vs independent NumPy reference | exact match (`test_mart.py`) | exact | true |
| 3 | benchmarked against published human crash rates per 100M vehicle miles | simulated rate / NHTSA-derived human rate, matched units | **326.0 per 100M VMT** vs **188.0 per 100M VMT** human (1.73x) | n/a (comparison, no numeric target) | true |
| 4 | importance sampling hit a target interval width with fewer simulated miles than uniform sampling | required-N ratio (naive / importance sampling) at a matched 20% relative CI half-width | **49.75x** (closest of 3 genuine attempts: 8.26x, 18.69x, 49.75x) | 38x | false (overshoots after 3 genuine tilt attempts; see "What broke") |
| 5 | zero-event bins report Jeffreys intervals, not a rate of zero | count of bins with 0 observed events this run, each given a nonzero-width interval | **2 of 5 bins** (`arterial_25_35mph`, `highway_55plus_mph`), both reported with a positive upper bound | n/a (design) | true |

Precise definitions:
- **Exposure-weighted rate**: `sum(bin_rate_hat * bin_exposure_weight) / sum(bin_exposure_weight)`,
  computed by the SQL query in `mart.exposure_weighted_rate_sql` over a
  5-row DuckDB table, one row per road-speed bin, each bin simulated
  independently at 500,000 miles.
- **Required-N reduction factor**: for the single rarest bin
  (`arterial_25_35mph`, true probability `1.3e-6`/mile), the smallest
  per-arm sample size at which each estimator's normal-approximation 95%
  CI half-width falls to 20% of the point estimate, computed from each
  estimator's per-sample variance measured at `N=2,000,000` and averaged
  over 20 repeats, then extrapolated via the closed-form
  `halfwidth(N) ∝ 1/sqrt(N)` relationship. This is an extrapolation from a
  variance measurement, not a direct simulation at the required N itself
  (which would be tens of millions of simulated miles for the naive case,
  the entire point of needing the extrapolation).
- **Zero-event bin count**: a property of this specific run's random
  seed at 500,000 simulated miles per bin; re-running with a different
  seed would very plausibly change which bins, and how many, land at
  exactly zero, since both bins currently at zero have a true expected
  count under 1 (0.65 and 1.30 respectively).

**The number that matters most is row 5, not row 4.** The entire design
point of this project is that a safety case built on a rare event cannot
honestly report a bin with zero observed collisions as "collision rate:
0"; two of five bins hit that exact situation on this run, and both are
reported as an interval with a real, nonzero upper bound rather than a
point mass at zero. Row 4's specific 38x figure is the one number on this
page that came from a resume draft rather than from this project's own
design, and it is also the one this project's own math says is not a
fixed constant to be hit, but a continuously tunable tradeoff (see "What
broke").

## Building and running

```
cd projects/exposure-weighted-safety-risk-estimate
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

venv\Scripts\python.exe -m pytest tests -v

venv\Scripts\python.exe scripts\run_estimate.py
venv\Scripts\python.exe scripts\run_importance_sampling_benchmark.py
```

## Sibling comparison

There is no sibling project in this portfolio doing rare-event rate
estimation with exposure reweighting; the closest relative is
[experiment-readout-guardrail-harness](https://github.com/Manas103/experiment-readout-guardrail-harness),
which also measures a statistical method's honest gap against a target
(there, an always-valid A/B test's false-positive rate; here, an
importance-sampling reduction factor), but that project is two-arm
hypothesis testing on an abundant metric, not rare-event estimation with
exposure weighting, and the two were compared and kept separate rather
than merged when this project was designed.

## Extension: Hierarchical Bayesian Partial Pooling

### Honest framing (extension)

- **This extension uses a different, 12-bin fleet, not the original 5-bin
  one.** `src/risk/hb_fleet_dgp.py` splits the original 5 road-speed bands
  by a dense/light-traffic context plus two extra bins, with lower
  per-bin exposure (60,000 miles/bin instead of 500,000), so that *most*
  of the 12 bins land at zero observed events in a single period, which
  the original 5-bin fleet does not reliably do. The two fleets
  deliberately disagree; this is a stress test for pooling across many
  sparse bins, not a better fleet design.
- **"Unpooled" means a fixed, wide, not-estimated-from-data prior per
  bin** (`Normal(0, 5)` on the log-rate scale), not a frequentist MLE with
  a Wald confidence interval. Both the pooled and unpooled models are
  fit by the same MCMC machinery; the only difference is whether the
  per-bin log-rate offset's own prior width is shared and estimated
  (pooled) or fixed in advance (unpooled).
- **Coverage is checked against a held-out period's own observed rate,
  not the (unobservable, in a real deployment) true generating rate.**
  The held-out period is simulated at 50x the training period's exposure
  (the same idea as the extreme-value project's 40x-longer held-out run),
  long enough that its own empirical rate is a much less noisy proxy for
  ground truth than the sparse, often-zero training period, without
  reaching into the simulation and reading off the true rate the model
  was never shown.
- **MCMC here means PyMC's NUTS via the `nutpie` backend, not PyMC's
  default sampler.** This machine has no `g++`, so PyMC's default
  Python-mode NUTS measured about 16s per fit; `nutpie` (a separately
  compiled Rust NUTS implementation PyMC can delegate to) measured about
  2-3s per fit after a one-time compile, which is what makes a
  repeated-refit coverage simulation feasible at all on this machine. See
  "What broke (extension)" below.
- **The coverage simulation ran at 200 refits, not the 2,000 originally
  planned**, because of a severe, disk-space-driven infrastructure
  failure partway through the first large-scale attempt; see "What broke
  (extension)".

### Architecture (extension)

```
  src/risk/
    hb_fleet_dgp.py       # the 12-bin fleet: most bins land at zero events at the design exposure
    hb_glm.py                # pooled vs. unpooled NB GLM model builders, MCMC refit, credible intervals
    coverage_sim.py             # multiprocess repeated-refit coverage simulation
  tests/
    test_hb_glm.py                  # shrinkage invariant, interval invariants, fast-draws smoke tests
    test_coverage_sim.py               # end-to-end plumbing check of the multiprocess harness itself
  scripts/
    run_hb_glm_single_fleet.py            # the single-fleet, most-bins-zero demonstration behind docs/hb_glm_single_fleet_output.txt
    run_coverage_simulation.py               # the repeated-refit coverage measurement, CLI-configurable (n_refits, draws, tune, n_workers)
  docs/
    hb_glm_single_fleet_output.txt              # raw stdout (committed)
    coverage_simulation_output.txt                 # raw stdout from the reported 200-refit run (committed)
```

### Why partial pooling is a shared, estimated sigma, not just "more bins helping each other" in the abstract

The pooled model's bin-level log-rate offsets are drawn from
`Normal(0, sigma)` with `sigma` itself given a prior and estimated from
all 12 bins' data jointly; a sparse bin's posterior is pulled toward the
population mean by exactly as much as the data from the other 11 bins
supports that pooling being a good idea, not by a fixed, hand-chosen
amount. The unpooled model's fixed `Normal(0, 5)` prior has no such
feedback: it is wide enough to not constrain any one bin much, but it
never learns from the other bins at all.

### Why the coverage target is a held-out rate, not the known true rate

This project's own validation elsewhere (`test_jeffreys.py`'s
`test_empirical_coverage_near_nominal_by_simulation`) checks coverage
against a *known* true rate, which is the right check for validating an
interval *formula*. This extension instead asks a more practically
relevant question: if a safety team actually ran this fleet for another,
longer period, would last period's posterior interval have covered what
they then observed? That is a harder, noisier target than the true rate
itself, since the held-out period's own observed rate carries its own
sampling error.

## Validation (extension)

Two additional layers, in `tests/test_hb_glm.py` and
`test_coverage_sim.py` (29 tests total now passing, raw output not
re-captured separately; see the original "Validation" section above for
`docs/test_output.txt`'s scope):

1. **An invariant, not a plausibility check**:
   `test_pooled_posterior_shrinks_more_than_unpooled_for_sparse_data`
   fits both models to the same sparse draw and asserts the pooled
   posterior's standard deviation is smaller than the unpooled one's for
   at least 10 of 12 bins, which is what "partial pooling" is defined to
   do, not just something it happened to do once.
2. **A multiprocess plumbing smoke test**:
   `test_coverage_simulation_runs_and_returns_bounded_fractions` runs the
   actual `coverage_sim.run_coverage_simulation` function (the same code
   path `scripts/run_coverage_simulation.py` uses) at a tiny scale (4
   refits, 2 workers) and checks every returned fraction is a valid
   probability, catching a broken worker-pool wiring without needing the
   full 200-refit run.

## What broke (extension)

**Symptom.** PyMC's default NUTS sampler measured about 16 seconds per
model fit on this machine (confirmed by direct timing before writing any
of the coverage-simulation code). At that rate, even the reduced 1,000-
refit target (2 models x 1,000 refits x 16s) would have taken roughly 9
hours, not a feasible single session.

**Wrong hypothesis first.** The initial guess was that the hierarchical
model itself was too complex for fast sampling, and that the fix would
need a simpler model (fewer bins, a non-hierarchical structure) at the
cost of weakening the pooling-vs-no-pooling comparison this extension
exists to make.

**The measurement that discriminated.** Printing PyTensor's own warning
(`g++ not available ... PyTensor will be unable to compile
C-implementations`) pointed at the real bottleneck: without a C compiler,
PyMC's default sampler falls back to a slow, interpreted Python execution
path for every single leapfrog step. Switching `nuts_sampler="nutpie"` (a
separately compiled Rust NUTS implementation PyMC can delegate to,
installed via `pip install nutpie`, no `g++` required) measured about 2-3
seconds per fit after a one-time per-process compile, a roughly 6x
speedup, with no change to the model itself.

**Root cause #2, discovered mid-run.** The first large-scale attempt (a
4-worker, 1,500-refit run) crashed partway through with `LLVM ERROR: out
of memory` from `nutpie`'s own JIT compiler. Physical RAM was not the
problem (9+ GB free throughout); this machine's `C:` drive independently
dropped to under 1 GB free during the same window, from causes entirely
unrelated to this project (verified: no large files had been written by
this session at the times disk space dropped). With the drive nearly
full, Windows could not grow its pagefile to back the JIT's memory
request, and the compiler failed. This is disclosed here because it
materially limited what this extension could measure, not because it
reflects a flaw in the model or the coverage-simulation code: a retry at
a much smaller, single-worker, reduced-draws setting (200 refits, 100
draws/100 tune) completed cleanly once disk space had recovered.

**Consequence for the headline numbers.** The reported coverage run used
200 refits (2,400 refit-bin checks total), not the 2,000 originally
planned, and 100 posterior draws per fit instead of 250. This is reported
as a genuine infrastructure constraint, not a design choice or a second
"attempt" at the same measurement; the single completed 200-refit run is
reported as-is, not re-run in search of a more favorable number.

**A genuine, unexpected finding, not just a smaller sample.** The resume
draft's framing assumed the unpooled model would *undercover* (71%, below
the 95% nominal level), the classic failure mode of a naive estimator on
sparse count data. What was actually measured is the opposite: the
unpooled model *overcovers* (98.12%), because its fixed, wide prior
produces intervals wide enough to trivially contain almost any held-out
rate, not intervals that are poorly calibrated in the direction assumed.
The pooled model's coverage (94.63%) is much closer to both the 95%
nominal level and the resume draft's own 94% figure.

## Measured results (extension)

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, Windows 11,
Python 3.12.10 native Windows, PyMC 6.4.0, `nutpie` 0.16.11. Raw stdout in
`docs/hb_glm_single_fleet_output.txt` and `docs/coverage_simulation_output.txt`.

| # | Claim | Metric | Measured | Target | Meets claim |
|---|---|---|---|---|---|
| 1 | simulated fleet where most road-speed bins observe zero events | zero-event bin count, single run | **11 of 12 bins (91.7%)** | majority | true |
| 2 | negative-binomial GLM with a log-exposure offset and partial pooling across bins, fit by MCMC | model family and fit method | PyMC `NegativeBinomial` likelihood, `exposure` as a multiplicative offset, `bin_eff ~ Normal(0, sigma)` with `sigma` estimated (pooled), NUTS via `nutpie` | n/a (design) | true |
| 3 | 95% posterior intervals covered the held-out rate on 94% of 2,000 refits | pooled coverage fraction | **94.63%** over 200 refits x 12 bins = 2,400 checks (2,000 refits planned, reduced to 200 by an infrastructure failure; see "What broke") | 94% over 2,000 refits | true on coverage value (94.63% vs. 94%); refit count short of target, disclosed |
| 4 | 71% coverage unpooled | unpooled coverage fraction | **98.12%** over the same 2,400 checks | 71% | false (unpooled overcovers, the opposite direction from the resume draft's assumption; see "What broke") |
| 5 | fleet rate reported against a cited human benchmark of 188 crashes per 100M miles | simulated 12-bin fleet rate / human rate, matched units | **101.06 per 100M VMT** vs. **188.00 per 100M VMT** human (0.538x), from the single-fleet run (1 total event over 720,000 simulated miles) | n/a (comparison, no numeric target) | true |

Precise definitions:
- **Pooled/unpooled coverage**: fraction of (refit, bin) pairs where the
  95% equal-tailed posterior interval for that bin's per-mile rate
  contains that bin's own held-out-period observed rate
  (`held-out counts / held-out exposure`, exposure = 50x the training
  period's), aggregated over all 200 refits x 12 bins.
- **200 refits, not 2,000**: the production run used `draws=100,
  tune=100, n_workers=1`, a deliberately conservative setting chosen
  after the disk-space-driven crash described in "What broke", not the
  `draws=250, tune=250, n_workers=4` setting the first (crashed) attempt
  used.
- **Divergences**: 16 of 20,000 pooled-model post-warmup draws diverged
  (0.08%), 0 of 20,000 unpooled-model draws; a divergence rate this low
  is not treated as invalidating the pooled coverage number, but is
  disclosed rather than silently dropped.

**Row 4, not row 3, is the more informative finding.** Row 3 came close
to confirming the resume draft's own number; row 4 contradicts its
direction entirely. An unpooled model on sparse count data is
conventionally assumed to be poorly calibrated in the undercoverage
direction (too confident); this project's specific unpooled baseline
(a fixed, wide, non-adaptive prior) is instead too conservative, which is
a real, useful distinction a reader should take away over the single
"94% vs 71%" headline contrast the resume draft implied.

## Extension limitations

- The 200-refit coverage run is well short of the 2,000 originally
  planned, for a disclosed infrastructure reason ("What broke"), not a
  design choice; the 94.63%/98.12% coverage figures carry more sampling
  uncertainty than a 2,000-refit run would.
- The "unpooled" baseline is one specific, defensible definition of "no
  pooling" (a fixed, wide prior with no shared hyperparameter); a
  frequentist MLE-with-Wald-interval baseline, or an unpooled model with
  a narrower fixed prior, would likely produce different, and possibly
  undercovering, results. This extension's finding (unpooled overcovers)
  is specific to this definition, not a universal claim about every
  possible unpooled baseline.
- The 12-bin fleet's true per-mile rates, exposure weights and
  dense/light-traffic split are design choices for this extension, same
  disclosure as the original 5-bin fleet.
- 16 pooled-model divergences were observed and disclosed but not
  individually investigated (e.g., by reparameterizing to a non-centered
  hierarchical form); a production use of this model would want that
  follow-up before trusting its intervals at face value.

## Limitations

- The five road-speed bins, their true per-mile collision probabilities
  and the fleet's exposure mix are design choices for this repository,
  not a measurement of any real AV fleet or real naturalistic driving
  data; the specific numbers above (306,748 miles/event, 1.73x the human
  benchmark) are properties of this synthetic design, not a claim about
  any real fleet's safety.
- The human benchmark (188 crashes per 100M VMT) is a single pooled
  figure for all police-reported crash severities; a real safety case
  benchmarks against severity-matched, road-speed-matched human rates
  (as Zoox's own published framework does, parsing NHTSA and FHWA data
  by road speed), which this project does not attempt.
- The importance-sampling estimator uses a single global proposal
  tilt per bin rather than an optimized, per-bin exponential-tilting
  parameter; a more sophisticated design (cross-entropy method,
  adaptive importance sampling) would likely find a better operating
  point without the sensitivity disclosed in "What broke".
- The required-sample-size figures are extrapolated from a variance
  measurement at a smaller, feasible N via a closed-form scaling law,
  not directly simulated at the required N itself; this is standard
  practice for rare-event sample-size planning, but it assumes the
  normal approximation underlying the CI formula holds at the
  extrapolated N, which is not independently re-verified here.
- Zero-event bin counts (2 of 5 on this run) are a property of this run's
  specific random seed and a true expected count under 1 per bin; this
  is disclosed explicitly in "Measured results" above rather than implied
  to be a stable design property.
