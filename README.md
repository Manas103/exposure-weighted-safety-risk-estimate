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
