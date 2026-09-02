# Crowd-level prediction for the Mumbai Suburban Railway (Harbour Line)

**Predicting how full a single coach will be, under a loss function that knows
calling a dangerous coach "safe" is not the same kind of mistake as the
reverse.**

Machine learning capstone project. Two course outcomes are implemented end to
end against one problem:

| | Outcome | What is built |
|---|---|---|
| **CO2** | Supervised regression | Coach-level crowd density predicted from time, calendar, weather and operating features, trained under **asymmetric loss** and turned into an operational alert by an explicit **cost-optimal decision layer** — point-forecast thresholds, a quantile ensemble, and a direct band classifier, all scored against one cost matrix. |
| **CO5** | Unsupervised clustering | The 35 Harbour-line stations grouped by their 24-hour boarding/alighting signatures into interpretable operational **roles**, then fed back into CO2 to test whether the structure is useful as well as pretty. |

---

## 1. The problem, and why the loss function is the whole point

The Mumbai suburban network carries around 7.5 million passenger journeys a
day on infrastructure sized for a fraction of that. The Harbour Line runs
CSMT–Panvel and CSMT–Goregaon, and in the morning peak a second-class coach
routinely carries 300–350 people in a space with 100 seats and roughly 25 m²
of standing floor. Indian Railways has a name for the top of that range:
**super-dense crush load**, 14–16 standees per square metre. It is also the
regime in which people ride footboards, and in which the network's roughly
2,000 annual deaths — largely falls from moving trains and platform-gap
incidents — actually happen.

Now consider a model that forecasts coach density and gets it wrong in the two
possible directions:

- **It over-predicts.** It says CRUSH; the coach arrives merely BUSY. A relief
  rake was held, staff were moved to a foot-over-bridge, a display told people
  to wait. The cost is money and a little credibility. It is bounded and it is
  recoverable.
- **It under-predicts.** It says COMFORTABLE; the coach arrives at super-dense
  crush. Nobody was sent. Nothing was held. More people were waved onto a
  platform that was already the problem. The cost is not bounded and it is not
  recoverable.

Squared error says those two errors are identical. **They are not, and the
central technical claim of this project is that the objective function is
where that fact belongs** — not in a post-hoc fudge factor added to the
model's output, and not in a threshold quietly nudged after the fact.

So the estimator being sought is not

$$\arg\min_{\hat y}\ \mathbb{E}\left[(y-\hat y)^2\right]$$

but

$$\arg\min_{\hat y}\ \mathbb{E}\left[C(y,\hat y)\right]$$

for a cost $C$ that is steeper below the truth than above it. Everything in
`src/mumbai_crowd/losses.py` is a way of writing that $C$ down so a
gradient-boosted ensemble can optimise it directly.

### The bridge from "safety matters more" to a training objective

The link is exact rather than hand-wavy. If being one standee/m² short costs
$c_\text{under}$ and being one over costs $c_\text{over}$, then the cost is
piecewise linear, and the forecast that minimises expected cost is the
$\tau$-quantile of the predictive distribution with

$$\tau = \frac{c_\text{under}}{c_\text{under}+c_\text{over}}$$

With this project's cost ratio of 6:1 that is $\tau = 0.857$. **The
"safety margin" is not a number I chose. It is a number the cost ratio
implies**, and `PinballLoss.from_costs()` computes it. Changing the operator's
view of the cost ratio changes the model, monotonically and predictably — see
the sensitivity analysis in §6.6.

---

## 2. Quickstart

```bash
git clone <this repo> && cd Local-train-crowd-level-prediction
pip install -r requirements.txt

python scripts/run_all.py            # everything, ~15 min
python scripts/run_all.py --quick    # 60 days instead of 180, ~5 min
```

Or step by step:

```bash
python scripts/01_generate_data.py --days 180 --monitored 0.08   # simulate
python scripts/02_train_regression.py                            # CO2
python scripts/03_cluster_stations.py --k 5                      # CO5
python -m pytest                                                 # 155 tests, ~3 min
```

Figures land in `reports/figures/`, tables in `reports/tables/`.

---

## 3. The data — read this before believing any number

**There is no public coach-level occupancy dataset for the Mumbai suburban
network.** UTS/ATVM ticketing data is not published at the granularity this
problem needs, and nothing counts passengers *per coach per station*. So the
dataset here is **simulated**, and this section is deliberately placed before
the results rather than in a footnote.

It is not a random draw dressed up as data. `src/mumbai_crowd/simulate.py`
runs a mesoscopic queueing model of the actual line:

1. **Trip generation.** Each station produces trips for three purposes —
   `to_work`, `to_home`, `other` — in proportion to its residential and
   employment catchment, shaped by purpose-specific time-of-day profiles.
   Arrivals at a platform are a Poisson process at the prevailing hourly rate,
   accrued over the gap since the previous train actually left.
2. **Destination choice.** A gravity model with a 30 km decay scale (Mumbai
   commutes are long: Panvel–CSMT is 49 km and entirely routine), attracted to
   employment mass in the morning and residential mass in the evening, with
   extra pull towards interchanges.
3. **Platform queueing.** Passengers accumulate at the platform between
   trains. Each arriving service picks up as many as its remaining capacity
   allows; the rest are **left behind** and compound into the next headway.
4. **Coach allocation.** Train load is split across coaches by demand pool
   (general / ladies / first / ladies-first — each a separate queue with its
   own capacity) and by **proximity to the station's foot-over-bridge**, which
   is why coaches under the bridge run 30–50% denser than coaches at the far
   end of the same train.
5. **Observation.** Only ~8% of rakes are treated as carrying load-cell
   instrumentation, and only ~30% of those coaches, with multiplicative sensor
   noise. Ground truth is partial, as it would be in reality.

The default run is **180 operating days** (1 June – 27 November 2024, chosen to
straddle the monsoon), producing **403,459 coach-arrival observations**,
259,940 station-hour flow records and 106,410 train-level records. It takes
about 140 seconds and writes ~20 MB of gzipped CSV, so `data/` is gitignored
and rebuilt rather than committed.

Why go to this trouble instead of sampling a marginal distribution? Because
the thing being predicted is a *downstream consequence of a queue*. Only a
mechanism produces the structure a model must actually learn: the tidal AM/PM
reversal, load peaking mid-route rather than at the terminus, left-behind
passengers compounding when rain stretches the headway, and a long right tail
where the danger lives. Sampling a marginal would give a dataset whose "hard"
cases are pure noise, and any claimed skill on it would be an artefact.

### What is grounded in published fact vs. assumed

| Grounded | Assumed (and documented in code) |
|---|---|
| Station list, order, interchanges, approximate chainage (Central Railway timetable) | `population_index` / `employment_index` per station — ordinal 0–1 catchment scores |
| Rake composition and coach classes | Exact coach-by-coach layout of a 12/15-car rake |
| IR load taxonomy: normal crush ~6/m², dense ~10/m², **super-dense 14–16/m²** | Cost matrix values (indicative rupees) |
| Mumbai monthly rainfall normals (simulated year totals ~2,450 mm, ~95% in Jun–Sep) | Rain → demand and rain → headway elasticities |
| Ganeshotsav dates; Anant Chaturdashi as the peak crowd night | Festival demand multiplier shape |
| Harbour Line ridership order of magnitude (~1.2–1.5 M journeys/day) | Peak headway split between the two service patterns |

**Conclusions about model behaviour under asymmetric loss transfer to real
data. Specific numbers do not.** Every parameter lives in
`network.py`, `demand.py` or `config.SimConfig`, and `features.py` is written
so that swapping in real AFC or CCTV counts means replacing one loader, not
rewriting the pipeline.

### Sanity checks the simulator has to pass

`tests/test_simulate.py` asserts the physics rather than just "it ran":

- passengers are conserved along every run (cumulative boardings − alightings
  equals onboard load at every stop, to within 1 passenger);
- every run ends empty;
- no coach exceeds its physical capacity;
- the AM peak flows towards the CBD and the PM peak away from it;
- load peaks **mid-route**, not at the origin terminus;
- rain increases peak crowding;
- the DANGEROUS band is rare but present.

---

## 4. Problem formulation

**Target.** `density_depart` — standees per m² of usable standing floor in one
coach as it leaves one station. Seated passengers do not contribute, because
crush risk is a standing-space phenomenon.

**Bands.** Cut-points follow IR's own suburban load taxonomy, not the
Fruin/TCQSM scale used for Western metros (which tops out long before a Mumbai
local does):

| Band | Density | ≈ people in a second-class coach | Operator action |
|---|---|---|---|
| COMFORTABLE | < 4 | < 200 | none |
| BUSY | 4–8 | 200–300 | advisory on PIS displays |
| CRUSH | 8–12 | 300–400 | platform marshalling, hold doors, RPF on the FOB |
| **DANGEROUS** | **≥ 12** | **400+** | inject relief service, gate-control station entry |

**Cost matrix** (`config.COST_MATRIX`, indicative ₹ per coach-arrival; rows =
truth, columns = prediction):

| true ↓ / predicted → | COMFORTABLE | BUSY | CRUSH | DANGEROUS |
|---|---|---|---|---|
| COMFORTABLE | 0 | 60 | 400 | 1,500 |
| BUSY | 350 | 0 | 180 | 900 |
| CRUSH | 4,200 | 1,800 | 0 | 500 |
| **DANGEROUS** | **22,000** | **12,000** | 3,000 | 0 |

The bottom-left cell — telling a station master that a super-dense-crush coach
is comfortable — is priced roughly 15× the top-right cell, which is the
symmetric mistake in the opposite corner. Every model in the project is scored
against this matrix, and the matrix lives in one file so a reviewer can
disagree with it in one place.

---

## 5. CO2 — method

### 5.1 The loss library

All in `src/mumbai_crowd/losses.py`, all exposing `elementwise`, `grad_hess`
and `optimal_constant`, all pluggable into LightGBM as a custom objective.

| Loss | Shape | Minimiser | When to use it |
|---|---|---|---|
| `SquaredError` | symmetric parabola | conditional **mean** | the baseline everything is measured against |
| `AsymmetricSquaredError(w_u, w_o)` | parabola with one steep wall | a **weighted mean** above the mean | smooth, easy to explain, easy to tune |
| `PinballLoss(τ)` | piecewise linear, slopes τ and 1−τ | the **τ-quantile** | the principled choice: τ = c_u/(c_u+c_o) makes it *exactly* cost-optimal |
| `LinexLoss(a)` | linear one side, exponential the other | no closed form | when the bad tail is catastrophic rather than merely expensive |
| `AsymmetricHuber(δ, w_u, w_o)` | quadratic core, linear tails, unequal slopes | robust weighted centre | when the target has genuine sensor outliers |

`tests/test_losses.py` checks every one against a central-difference
derivative, checks that the minimiser is what the theory says (`SquaredError`
→ mean, `PinballLoss(τ)` → the τ-quantile), and checks that a harsher weight
ratio always raises the forecast.

### 5.2 Two numerical traps, both hit and both fixed

These are in the write-up because they cost real debugging time and because
"use a custom objective" is usually presented as if it just works.

**LINEX diverged to −4,000.** The LINEX hessian is `a² e^{a(y−ŷ)}`, which on
the *over*-prediction side decays towards zero. A Newton step `−g/h` with a
vanishing `h` is unbounded: boosting took one leaf step of order 10⁵ and never
came back. The fix is a hessian floor, which bounds the step at `a/floor`. It
costs nothing statistically, because the region it touches is the one this
loss barely cares about. The same trap, milder, applies to the Huber tails,
where the standard surrogate `δ/|r|` is the right answer.
`test_newton_step_is_bounded` is the regression test.

**Pinball looked hopeless until it was given enough rounds.** Its gradient is
*bounded* — exactly `−τ` or `1−τ`, however wrong the prediction — so with a
unit pseudo-hessian each tree can move a prediction by at most the learning
rate. Travelling ten standees/m² at `lr = 0.05` needs 200 trees just to
climb. At 100 rounds pinball is the worst model in the zoo; at 500 it is the
best:

| boosting rounds | L2 expected cost (₹) | pinball expected cost (₹) |
|---|---|---|
| 80 | 240 | 288 |
| 200 | 236 | 187 |
| 500 | 236 | 166 |
| 1200 | 236 | 162 |

Shrinking the pseudo-hessian (`hessian_scale=0.25`) enlarges the step
proportionally and reaches the same solution in a quarter of the rounds. It is
the same device as raising the learning rate for that loss alone, but it keeps
one learning rate across the whole comparison — so the loss stays the only
thing that varies between models, which is the entire experimental design.

**A model trained with a custom objective needs its init score added back.**
LightGBM's `Booster.predict` returns only the boosted part; the
`init_score` is not included. Forgetting it shifts every prediction by a
constant, silently. `BoostedRegressor` sets `init_score` to
`loss.optimal_constant(y_train)` — which also removes the handicap that would
otherwise make the more asymmetric losses look bad, since they start further
from zero — and adds it back on predict. `test_init_score_is_added_back_on_predict`
pins this down.

### 5.3 Leakage discipline

The easiest way to produce a beautiful, worthless R² here is to feed the model
the boardings and alightings recorded *at the same stop of the same train*
whose density is the target. Those are measured when the doors close — the
same instant as the target — so a model using them is not forecasting, it is
reading the answer.

`features.LEAKY_COLUMNS` names them, `assert_no_leakage()` is called inside
`build_design()`, and `tests/test_features.py` asserts that the guard actually
fires. Two feature sets are defined and never mixed:

- **`schedule`** (53 features) — what a control room knows the evening before:
  timetable, calendar, station, coach, weather forecast, and historical
  averages. Supports *planning*: where to position tomorrow evening's relief
  rakes.
- **`realtime`** (58 features) — the above plus what the network reports
  minutes before the event: this service's delay, the actual gap since the
  previous train, and **the density of this same coach one stop back**.
  Supports *intervention*: telling the station master at Kurla what is about
  to pull in.

Historical averages are target encoding, which is the second-easiest way to
leak. `HistoricalProfileEncoder` is fitted **on the training slice only**,
with an empirical-Bayes shrink towards the global mean so thin cells
(Manasarovar, 05:00, Sunday) contribute signal rather than noise.
`test_historical_encoder_is_fitted_on_train_only` checks that its global mean
equals the *train* mean and not the full-data mean.

**Splits are temporal, never random.** A random split would put 09:14 and
09:19 of the same Tuesday on opposite sides of the fence, and consecutive
services on the same track share almost all of their state. Train is the first
138 days, validation the next 18, test the final 24 — and the test period is
post-monsoon while much of training is monsoon, so the evaluation includes a
genuine distribution shift rather than an idealised one.

### 5.4 The decision layer

A regression returns a number; a station master needs an action. Comparing the
point forecast to the physical band edges throws away most of the remaining
safety, and for a reason that is a fact about point forecasts rather than a
bug: a point forecast summarises `p(y|x)`, and thresholding it assumes the
whole distribution sits on one side. A coach forecast at 9.5 with a wide right
tail is *point*-CRUSH but may carry a 30% probability of super-dense crush —
and 30% of a ₹22,000 outcome dominates the ₹500 cost of over-reacting.

Four policies are implemented and compared (`src/mumbai_crowd/decision.py`):

1. **`NaivePolicy`** — compare the point forecast to the band edges. The thing
   everyone writes first, kept as the comparison point.
2. **`ThresholdPolicy`** — three alert cut-points fitted by coordinate descent
   on validation against the cost matrix. The objective is a step function of
   the thresholds, so gradients are useless and a direct grid sweep is both
   simpler and exact. Deployable: it is three numbers a control room can act on.
3. **`ProbabilityPolicy`** — the Bayes action taken straight from a model that
   outputs band probabilities (`src/mumbai_crowd/classification.py`). No
   thresholds to tune at all: once the probabilities exist the cost matrix
   decides.
4. **`DistributionalPolicy`** — a quantile ensemble gives `p(y|x)`, which is
   converted to band probabilities and then to the same Bayes action
   `argmin_a Σ_b P(band=b|x) C[b,a]`. Textbook-correct, and needs no threshold
   tuning at all.

---

## 6. CO2 — results

Test period: 2024-11-04 to 2024-11-27, **56,763 coach-arrivals**, never seen in
training or threshold tuning. Feature set: `schedule` (the harder of the two).
DANGEROUS base rate in the test window: **1.13%** (640 coach-arrivals out of 56,763).

### 6.1 The model zoo, scored on the physical band edges

| model | RMSE ↓ | R² ↑ | bias | cost ₹/arrival ↓ | said-safe-when-dangerous ↓ | danger recall ↑ | false alarm ↓ |
|---|---|---|---|---|---|---|---|
| `lgbm_l2` (symmetric) | **1.554** | **0.652** | +0.10 | 143.0 | 43.8% | 6.3% | **1.4%** |
| `lgbm_linex` | 1.754 | 0.557 | +0.54 | 106.6 | 26.1% | 18.8% | 2.9% |
| `lgbm_asym_l2` | 1.897 | 0.482 | +0.85 | 97.1 | 23.1% | 22.7% | 3.3% |
| `lgbm_asym_huber` | 1.861 | 0.501 | +0.80 | 94.7 | 22.0% | 28.8% | 3.6% |
| `lgbm_pinball` (τ=0.857) | 2.107 | 0.361 | +1.04 | **84.9** | 17.3% | 42.2% | 4.9% |
| `lgbm_l2_margin` (strawman) | 3.463 | −0.727 | +3.10 | 84.2 | **13.3%** | 40.5% | 6.3% |
| `linear_asym` | 2.064 | 0.387 | +0.52 | 138.5 | 50.9% | 2.0% | 3.1% |
| `linear_l2` | 1.819 | 0.524 | −0.30 | 230.3 | 95.6% | 0% | 0.05% |
| `historical_profile` | 2.085 | 0.374 | +0.13 | 309.5 | 100% | 0% | 0% |
| `mean_l2` | 2.637 | −0.001 | +0.10 | 387.6 | 100% | 0% | 0% |

**The trade-off is exactly as advertised and it is not free.** Moving from L2
to pinball costs 0.55 standees/m² of RMSE and 0.29 of R² — and buys a **41%
cut in expected operator cost**, cuts said-safe-when-dangerous from 43.8% to
17.3%, and raises danger recall from 6.3% to 42.2%, for a false-alarm rate
that rises from 1.4% to 4.9%.

A model selected on RMSE would pick `lgbm_l2`, which is by the operator's own
cost function the **second-worst** of the seven fitted models — beating only a
linear model on the same symmetric loss, and losing to every asymmetric one.
**That is the entire argument of the project in one row.**

### 6.2 The result I did not want: the strawman almost wins

`lgbm_l2_margin` is an L2 model plus a flat +3.10 safety margin tuned on
validation. On the operator cost metric it comes **first** (₹84.2 vs ₹84.9) —
a difference well inside noise. It would have been easy to leave this baseline
out. It is in, because it is the honest comparison and because *what it costs
to win that way* is the interesting part:

- Its R² is **−0.727**. It is worse than predicting a constant. It is no
  longer a density forecast at all — it cannot be reused for capacity
  planning, passenger information, or anything else, because at 03:00 on an
  empty train it confidently predicts 3.1 standees/m².
- It buys its safety by inflating *every* prediction uniformly, so it pays
  for it in false alarms (6.3%, the worst in the table) rather than by being
  selective about where danger actually is.
- Its margin is a number fitted to one validation period under one cost
  matrix. τ is computed from the cost ratio in closed form.

So the honest conclusion is narrower and more useful than "asymmetric loss
wins": **a flat margin can match an asymmetric loss on the one metric it was
tuned for, and only an asymmetric loss gets there while remaining a usable
forecast.** If all you will ever do is raise alarms, the strawman is fine. If
the same model has to answer any other question, it is not.

### 6.3 The decision layer

Alert cut-points fitted on validation against the cost matrix, applied to
test:

| model | BUSY cut | CRUSH cut | DANGEROUS cut |
|---|---|---|---|
| physical band edge | 4.00 | 8.00 | 12.00 |
| `lgbm_l2` | 2.02 (−1.98) | 4.22 (−3.78) | 7.47 (−4.53) |
| `lgbm_asym_l2` | 3.20 (−0.80) | 5.25 (−2.75) | 9.45 (−2.55) |
| `lgbm_pinball` | 3.22 (−0.78) | 5.67 (−2.33) | 10.67 (−1.33) |

**Every fitted cut-point sits below the physical edge**, and by how much is a
quantitative statement rather than a hunch: under a 6:1 cost ratio you have to
start worrying about a coach roughly 1.3 standees/m² before your point
forecast says it is dangerous. Note also that **the more asymmetric the loss,
the smaller the shift the decision layer still has to apply** — L2 needs a
−4.53 correction, pinball only −1.33, because pinball has already absorbed
most of the asymmetry into the model itself. The two mechanisms are
substitutes, and that is visible in the numbers.

Policy comparison on test:

| policy | cost ₹ ↓ | said-safe-when-dangerous ↓ | danger recall ↑ | false alarm ↓ | relief rakes |
|---|---|---|---|---|---|
| `lgbm_l2` + naive edges | 143.0 | 43.8% | 6.3% | 1.4% | 0.2% |
| `lgbm_asym_l2` + naive edges | 97.1 | 23.1% | 22.7% | 3.3% | 0.8% |
| `lgbm_pinball` + naive edges | 84.9 | 17.3% | 42.2% | 4.9% | 1.8% |
| `lgbm_asym_l2` + cost-optimal thresholds | 86.1 | 8.8% | 60.9% | 10.7% | 3.4% |
| `lgbm_pinball` + cost-optimal thresholds | 87.5 | **8.1%** | 59.4% | 11.7% | 3.3% |
| quantile ensemble + Bayes action | 116.0 | 10.9% | 30.0% | 7.4% | 1.1% |
| *(band classifier + Bayes action — see §6.5)* | ***80.97*** | *9.1%* | *31.1%* | *8.6%* | *1.5%* |

Two things worth reading carefully:

- The cheapest policies and the *safest* policies are **not the same
  policies**, and they differ by a few rupees per arrival. `pinball +
  thresholds` more than halves said-safe-when-dangerous (17.3% → 8.1%) for a
  cost increase the matrix says is not worth paying. Whether it *is* worth
  paying is a question about how much the matrix under-prices a death, which
  is a policy question and is flagged as one rather than resolved here.
- **The theoretically optimal policy came last but one.** That is §6.4, and
  the fix it points to is §6.5 — which ends up beating everything in this
  table.

### 6.4 Why the Bayes-optimal policy lost

`DistributionalPolicy` computes `argmin_a Σ_b P(band=b|x) C[b,a]`, which is
provably optimal *given correct probabilities*. It cost ₹116.0 against ₹87.5
for a directly tuned threshold. Since the Bayes rule minimises expected cost
pointwise, losing means the probabilities are wrong — so the useful thing to
do is find out how, not to quietly drop the method.

**The marginal calibration is fine.** Empirical coverage tracks nominal τ to
within 0.035 at every level:

| τ | 0.10 | 0.30 | 0.50 | 0.70 | 0.80 | 0.90 | 0.95 | 0.98 | 0.995 |
|---|---|---|---|---|---|---|---|---|---|
| empirical | 0.135 | 0.316 | 0.520 | 0.720 | 0.787 | 0.894 | 0.940 | 0.991 | 0.998 |

**The conditional calibration is not**, and that is what the decision rule
actually consumes. Binning test arrivals by predicted `P(DANGEROUS | x)`:

| predicted | observed | n |
|---|---|---|
| 0.010 | 0.000 | 22,324 |
| 0.012 | 0.004 | 31,222 |
| 0.033 | **0.085** | 1,056 |
| 0.076 | **0.123** | 634 |
| 0.147 | 0.172 | 1,034 |
| 0.265 | 0.327 | 407 |
| 0.381 | 0.412 | 85 |

Every bin above the floor sits **above** the diagonal: where it matters, the
model under-states the probability of super-dense crush, by up to 2.6× in the
0.02–0.05 bin. And 94% of all arrivals are crushed into the two bottom bins,
where the ensemble has no resolution left at all.

Now the arithmetic bites. With this cost matrix, dispatching a relief rake
only beats platform marshalling once `P(DANGEROUS | x) ≳ 0.25`. Only **314 of
56,763** arrivals (0.55%) clear that bar — against **640** arrivals that were
genuinely dangerous (1.13%). The optimal rule cannot fire on events it is
never told are likely, so it declines to act, and a cruder threshold on the
point forecast beats it.

**This is a real limitation of the approach at this event rate, not a bug**,
and the fix is not more quantiles: a nine-point grid already spends four of
its points above τ = 0.9. The fix is a model that targets the bands directly,
rather than one that infers a 1% tail from a quantile function fitted to the
whole distribution. That model is §6.5, and it turns out to win.

### 6.5 Closing the loop: modelling the bands directly

A LightGBM multiclass model over the four bands, with the same features and
the same Bayes rule, produces **the cheapest policy in the project**:

| policy | cost ₹ ↓ | said-safe-when-dangerous | danger recall | false alarm |
|---|---|---|---|---|
| **classifier + Bayes action** | **80.97** | 9.1% | 31.1% | 8.6% |
| `lgbm_l2_margin` + naive edges | 84.18 | 13.3% | 40.5% | 6.3% |
| `lgbm_pinball` + naive edges | 84.88 | 17.3% | 42.2% | 4.9% |
| `lgbm_pinball` + cost-optimal thresholds | 87.51 | 8.1% | 59.4% | 11.7% |
| quantile ensemble + Bayes action | 115.98 | 10.9% | 30.0% | 7.4% |
| `lgbm_l2` + naive edges (baseline) | 143.00 | 43.8% | 6.3% | 1.4% |

**₹143.0 → ₹81.0, a 43.4% reduction**, and the diagnosis in §6.4 is
vindicated: the Bayes rule was never the problem. The same rule, fed
probabilities from a model that targets the bands, goes from worst-but-one to
first.

#### It is resolution, not calibration

The tempting summary — "the classifier is better calibrated" — is wrong, and
the reliability diagram says so. At the top of the range the classifier is
*worse* calibrated than the quantile ensemble: it sits **below** the diagonal
(over-confident) where the ensemble sits above it. What the classifier has is
**resolution** — it separates the population instead of huddling it near the
base rate:

| | rows in bins with mean P ≥ 0.25 | highest bin probability |
|---|---|---|
| quantile ensemble | 492 | 0.381 |
| band classifier | 803 | 0.785 |

The Bayes rule needs both properties, and the quantile route was short of the
second one. A model can be beautifully calibrated and useless if it never says
anything is likely.

#### Recalibration helps a broken model and hurts a sound one

Two levers are usually applied together — up-weight the rare class, then
recalibrate — so the pipeline crosses them and reports all six cells:

| training weighting | no recalibration | temperature (1 parameter) | isotonic (non-parametric) |
|---|---|---|---|
| **none** | **₹80.97** | ₹84.03 | ₹90.62 |
| balanced | ₹212.32 | ₹133.78 | ₹86.43 |

Read along each row. For the **balanced** model — whose probabilities are
broken by construction, since it was fitted to a re-weighted distribution —
recalibration helps enormously and the more flexible the calibrator the better
(212 → 134 → 86). For the **unweighted** model, trained on plain multiclass
log-loss, recalibration *hurts*, monotonically in flexibility (81 → 84 → 91).

The reason is measurable rather than mysterious, and it is a property of this
dataset that a real deployment would share:

| period | P(DANGEROUS) |
|---|---|
| train (Jun–Oct, mostly monsoon) | 1.61% |
| validation half used for calibration (late Oct) | 1.52% |
| **test (Nov, post-monsoon)** | **1.13%** |

On the calibration window the raw model *under*-states danger, so isotonic
learns to push probabilities up — mapping a raw 0.10 to 0.22 and a raw 0.40 to
0.55. On the test window, with 26% less danger about, that correction is
exactly backwards, and the calibrated model ends up predicting 0.61 where the
truth is 0.24. The calibrator faithfully transferred a base rate that had
expired. Temperature scaling, with one degree of freedom, has less to transfer
wrongly and lands in between — which is the point of running both.

**The practical conclusion is not "calibrate your classifier".** It is: train
on a proper scoring rule, do not break the probability scale with class
weights, and then you have nothing to repair. Post-hoc calibration is a
*repair*, and repairs fitted on one regime do not survive a change of regime.

#### The row an operator might actually want

`classifier[balanced] + no recalibration` costs ₹212 — nearly three times the
best policy — and misses **0.16%** of dangerous coaches, against 9.1% for the
cheapest. It is the wrong answer under this cost matrix, and it would be the
right answer under a matrix that priced a death an order of magnitude higher.
It is left in the table for exactly that reason: the ranking is a consequence
of the cost assumptions in §4, and the reader should be able to see which
model the other assumption would have selected.

### 6.6 The cost ratio is a dial, and it behaves

Retraining pinball at τ = r/(r+1) for a range of assumed cost ratios r:

| ratio c_u : c_o | τ | learned bias | said-safe-when-dangerous | false alarm |
|---|---|---|---|---|
| 1:1 | 0.500 | −0.21 | 68.9% | 0.6% |
| 2:1 | 0.667 | +0.21 | 39.4% | 1.8% |
| 4:1 | 0.800 | +0.74 | 23.3% | 3.7% |
| **6:1** | **0.857** | **+1.04** | **17.3%** | **4.9%** |
| 10:1 | 0.909 | +1.47 | 11.3% | 6.6% |
| 20:1 | 0.952 | +2.74 | 7.0% | 11.2% |
| 40:1 | 0.976 | +6.66 | 0.6% | 29.9% |

Monotone in both directions, with no tuning and no thresholds — the safety
margin is *derived* from the cost ratio rather than chosen. This table is the
deliverable an operator would actually want: not "here is the model" but
"tell us what a missed crush event costs relative to an unnecessary relief
rake, and this is the system you get."

### 6.7 What real-time telemetry is worth

| feature set | features | RMSE | R² | cost ₹ | said-safe-when-dangerous | danger recall |
|---|---|---|---|---|---|---|
| `schedule` (night before) | 53 | 2.107 | 0.361 | 87.5 | 8.1% | 59.4% |
| `realtime` (+ this coach one stop back) | 58 | **0.750** | **0.919** | **16.2** | **0.0%** | **93.9%** |

Five features change everything, and the reason is not subtle: the density of
*this same coach at the previous station*, three minutes ago, is close to a
sufficient statistic for its density now. The operational reading is that
these are two different products:

- **Schedule-only** supports *planning* — where to position tomorrow evening's
  relief rakes — and it is genuinely hard, because a specific coach on a
  specific service two days out is mostly irreducible noise. R² = 0.36 is a
  real ceiling, not a modelling failure.
- **Real-time** supports *intervention* — telling Kurla what is about to pull
  in — and it is nearly solved, but only for operators who have already
  installed the sensors.

That distinction is worth more than either number on its own: it says exactly
what the instrumentation investment buys.

---

## 7. CO5 — station-profile clustering

### 7.1 What is clustered, and the one design decision that matters

Each of the 35 stations becomes one row combining:

- **shape** — normalised hourly boarding and alighting curves (20 hours each),
  saying *when* a station is a source and *when* it is a sink;
- **scale and asymmetry** — log volume, AM/PM peak shares, net-source indices
  for each peak, directional imbalance, weekend ratio, measured rain
  elasticity, and left-behind rate.

**Normalising the curves before clustering is the decision that makes this
interesting.** Raw hourly counts would cluster stations by *size* — Panvel and
CSMT together because both are big — which is a fact already visible in the
timetable. Normalising makes the algorithm answer the useful question: which
stations *behave* alike?

### 7.2 Choosing k honestly

Six criteria are computed and they disagree, which is the normal outcome on 35
points and is reported rather than hidden. The elbow rule is implemented as
distance to the chord, so it is a computation rather than a squint at a chart.

The pipeline deliberately runs the **bootstrap stability scan before stating a
choice of k**, so the choice can be argued from the numbers instead of being
decorated with them afterwards. That ordering is not cosmetic: my first pass
picked k = 5 on interpretability, and the bootstrap then said a five-way split
had an ARI of 0.68 — i.e. it would not survive a different sample of days.
The choice moved to k = 4. See §7.5.

### 7.3 Naming clusters without lying about them

Cluster names are generated **from the centroids by rule**, not written by
hand, so re-running with a different k or seed cannot leave the labels
describing a partition that no longer exists. Two details of the rule are
worth stating because both were wrong first:

- A **median split** on volume forces every cluster into "high" or "low", and
  produced the label *low-volume interchange churn* for a cluster containing
  Andheri, Bandra and Kurla. The rule now attaches a volume word only when the
  cluster is genuinely at one end of the range (|z| > 0.8 across clusters) and
  otherwise says nothing — a fifteen-station cluster spanning Andheri to
  Chunabhatti does not have a meaningful single volume anyway.
- Collisions are disambiguated, so two clusters can never silently share a
  name.

### 7.4 Validation without labels

There is no ground truth, so three independent checks are used instead:

1. **Method agreement.** k-means, Ward linkage and a Gaussian mixture are run
   independently and compared by adjusted Rand index. If three quite different
   algorithms recover the same partition, the structure is in the data rather
   than in the algorithm.
2. **Bootstrap stability.** *Days* are resampled with replacement, the whole
   profile-building pipeline is re-run, and the partition is compared to the
   full-data one by ARI. The 35 stations are the population, but the days are a
   sample — so this is the only honest uncertainty statement available here.
3. **DBSCAN sweep.** Included because it is the one method that can say "these
   stations belong to no cluster at all", and on a network with genuine
   oddities that answer deserves a hearing.

Cluster names are generated **from the centroids by rule**, not written by
hand, so re-running with a different k or seed cannot leave the labels lying
about what the clusters contain.

### 7.5 CO5 results

Five criteria were computed over k = 2…10 and they pick five different
answers. That is reported rather than smoothed over:

| criterion | would pick |
|---|---|
| elbow (max distance to chord) | **k = 4** |
| silhouette (max) | k = 2 |
| Calinski-Harabasz (max) | k = 3 |
| Davies-Bouldin (min) | k = 10 |
| GMM BIC (min) | k = 10 |
| **bootstrap stability (max ARI)** | k = 3 |

The internal indices split two ways and neither failure mode is informative:
some favour the coarsest partition available, which on 35 points is what they
usually do, and some run to the top of the k range, which is what fitting 53
dimensions with 35 observations looks like. The two criteria that do carry
information are the elbow and the bootstrap, and reading them together settles
it:

| k | bootstrap ARI (25 resamples of days) |
|---|---|
| 3 | 0.996 ± 0.021 |
| **4** | **0.984 ± 0.036** |
| 5 | 0.681 ± 0.196 |
| 6 | 0.728 ± 0.193 |

Stability is flat and near-perfect at k = 3 and k = 4 and then **collapses at
k = 5**: a five-way split is not reproducible from a different sample of days.
So k = 4 is the finest partition that is still reproducible, and it is also
exactly where the elbow sits. Chosen on that basis, not on a single index.

**The four roles** (silhouette 0.263; no station has a negative silhouette,
i.e. none is sitting in the wrong cluster):

| role | n | AM net source | PM net source | interchange | stations |
|---|---|---|---|---|---|
| **high-volume employment sink** | 2 | −0.56 | +0.53 | 50% | CSMT, Masjid |
| **interchange churn** | 15 | +0.10 | −0.08 | 47% | Kurla, Bandra, Andheri, Vadala Road, Nerul, Chembur, Mahim Jn, Vile Parle, Santacruz, Khar Road, King's Circle, Seawoods-Darave, Sanpada, Chunabhatti, GTB Nagar |
| **balanced mixed-use** | 7 | −0.02 | +0.21 | 29% | Sandhurst Road, Dockyard Road, Reay Road, Cotton Green, Sewri, Belapur CBD, Vashi |
| **low-volume morning-source dormitory** | 11 | +0.46 | −0.13 | 18% | Panvel, Kharghar, Khandeshwar, Manasarovar, Govandi, Mankhurd, Juinagar, Tilak Nagar, Goregaon, Jogeshwari, Ram Mandir |

`reports/figures/23_cluster_profiles.png` is the figure that makes this
concrete. The employment sink's alightings spike at 09:00 and its boardings at
19:00; the dormitory cluster is the exact mirror image; the two middle
clusters have boardings and alightings tracking each other all day, which is
what "people pass through here" looks like in a curve.

Nothing about the geography, the station names or the branch structure was
given to the algorithm — only 40 normalised hourly flow shares and 13 summary
statistics. It recovered the dock belt, the CBD, the interchange spine and the
Navi Mumbai dormitory belt anyway, and `reports/figures/24_line_map.png` shows
those roles laid out along the line.

**External validation.** Three algorithms were run independently on the same
profiles:

| | k-means | Ward | GMM |
|---|---|---|---|
| k-means | 1.000 | 0.968 | 0.902 |
| Ward | 0.968 | 1.000 | 0.870 |
| GMM | 0.902 | 0.870 | 1.000 |

Adjusted Rand indices of 0.87–0.97 between three quite different algorithms
means the structure is in the data, not in the choice of k-means. (At k = 5 the
same table read 0.42–0.81 — another sign that the five-way split was an
artefact.)

DBSCAN, given the same space, finds 2–4 clusters but leaves 22–29 of the 35
stations as noise at every eps worth reporting. That is a fair verdict on a
35-point, 53-dimensional space: density-based clustering has nothing to work
with here, and it says so instead of inventing structure.

---

## 8. CO5 → CO2: is the structure actually useful?

Clusters that look interpretable are cheap. The test is whether they carry
information, so the cluster label was fed back into the CO2 regression:

| variant | RMSE | R² | cost ₹ | said-safe-when-dangerous | danger recall |
|---|---|---|---|---|---|
| station identity only (baseline) | 2.107 | 0.361 | **87.5** | **8.1%** | 59.4% |
| station identity **+** cluster label | 2.136 | 0.343 | 88.7 | 8.4% | 62.5% |
| cluster label **instead of** station identity | 2.115 | 0.356 | 89.4 | 8.1% | **69.7%** |

**Adding the cluster label to a model that already has `station_code` does not
help** — it is very slightly worse, which is what a redundant feature looks
like after regularisation. That is the expected answer and it is reported as
such: a 4-way coarsening cannot tell a boosted tree anything that a 35-level
categorical did not already contain.

The third row is the one worth keeping. Replacing station identity entirely
with the 4-way role costs **2.2% of expected cost** (₹87.5 → ₹89.4) while
holding the dangerous-miss rate flat and actually *raising* danger recall.
So the roles are a **9× compression of station identity that costs about 2%**,
and that is a genuinely useful property: it is how you would score a station
with no operating history — a newly opened stop, or a station on a line you
have not instrumented — by assigning it a role from its catchment instead of
waiting a year for its own history to accumulate.

---

## 9. Figures

All regenerated by `python scripts/run_all.py` into `reports/figures/`.

| file | what it shows |
|---|---|
| `01_loss_shapes.png` | the five losses, normalised so the asymmetry is the visible thing |
| `02_target_distribution.png` | the target is zero-inflated with a rare, long right tail |
| `03_demand_surface.png` | density by station × hour, both directions — the tidal reversal |
| `04_rain_effect.png` | rain raises the mean, and raises the dangerous tail faster |
| `05_model_comparison.png` | RMSE vs cost vs safety vs false alarms, same features throughout |
| `06_prediction_scatter.png` | predicted vs actual; points below the diagonal are the dangerous errors |
| `07_residual_asymmetry.png` | asymmetric losses move the whole residual distribution right |
| `08_confusion_grid.png` | band confusion, with the two catastrophic cells outlined |
| `09_threshold_policy.png` | cost vs each alert cut-point, with the fitted minimum marked |
| `10_cost_sensitivity.png` | the cost ratio as a policy dial, behaving monotonically |
| `11_feature_importance.png` | what the schedule-only model actually uses |
| `12_quantile_calibration.png` | nominal vs empirical coverage of the quantile ensemble |
| `13_danger_reliability.png` | reliability *and resolution* of two competing probability models |
| `14_calibration_grid.png` | class weighting × recalibration, and how they interact |
| `20_k_selection.png` | six criteria for k, including the bootstrap-stability panel |
| `21_dendrogram.png` | Ward linkage over station profiles, with the k = 4 cut |
| `22_cluster_pca.png` | the four roles in PCA space, stations labelled |
| `23_cluster_profiles.png` | each role's boarding/alighting signature over the day |
| `24_line_map.png` | the Harbour Line as a branching schematic, coloured by role |
| `25_silhouette.png` | per-station silhouette; no station is in the wrong cluster |
| `26_cluster_feature_value.png` | does the CO5 structure help the CO2 model? |

---

## 10. Repository layout

```
src/mumbai_crowd/
  network.py       35 Harbour-line stations, chainage, rake and coach layout,
                   FOB-position bias.  All static reference data in one place.
  config.py        Crowd bands, the cost matrix, and every simulation /
                   model / clustering hyper-parameter as a dataclass.
  weather.py       Mumbai climatology + a two-state Markov rain process.
  calendar_in.py   Public holidays and the Ganeshotsav window.
  demand.py        Trip generation by purpose and a gravity destination model.
  simulate.py      The mesoscopic queueing simulation. Produces the 3 tables.
  features.py      Derived columns, leakage guard, temporal split, target
                   encoder fitted on train only.
  losses.py        ** The asymmetric losses. The core of the project. **
  metrics.py       Statistical and decision metrics, always reported together.
  decision.py      Naive / threshold / Bayes policies over model output.
  classification.py  CO2 decision layer: a multiclass band model plus
                   identity / temperature / isotonic recalibration.
  regression.py    CO2: baselines, linear-on-asymmetric-loss, LightGBM with a
                   pluggable objective, quantile ensemble.
  clustering.py    CO5: profiles, k selection, fitting, naming, validation.
  plots.py         Every figure in the report.

scripts/
  01_generate_data.py     simulate and write data/
  02_train_regression.py  CO2 end to end
  03_cluster_stations.py  CO5 end to end + the CO5→CO2 bridge
  run_all.py              all three, with a --quick mode

tests/                    155 tests: loss gradients, simulator physics
                          (conservation, capacity, tidal direction), leakage
                          guards, decision policies and the Bayes rule,
                          calibrators, clustering, estimators.
```

Run `make help` for the shortcuts.

---

## 11. Limitations, stated plainly

1. **The data is simulated.** This is the big one, and §3 is where it is
   argued rather than hidden. The *relative* behaviour of losses and policies
   is a property of the loss geometry and the cost matrix, so it transfers.
   The *absolute* numbers — 43% cost reduction, a 0.36 schedule-only R² — are
   properties of this simulator's parameters and should not be quoted as
   findings about the real Harbour Line.
2. **The cost matrix is asserted, not measured.** Every headline number is
   conditional on it. That is why §6.6's sensitivity analysis exists: the honest
   claim is not "the cost falls 43%" but "here is how the model changes as you
   change the cost ratio, and the relationship is monotone and interpretable."
   Getting real numbers would mean pricing a relief rake, staff time, and
   — unavoidably — a statistical life, which is a policy question and not a
   modelling one.
3. **No fatality or incident data is used.** The DANGEROUS band is defined by
   density, not by observed harm. A real deployment would want to validate the
   12 standees/m² cut-point against actual incident records rather than
   inheriting it from the IR load taxonomy.
4. **Only two service patterns are modelled.** Fast/semi-fast services,
   short-turns, the Trans-Harbour line and the Uran branch are out of scope,
   and each would change the capacity available at a given station-hour.
5. **The station-role clustering has n = 35.** No test set exists and none can.
   The bootstrap resamples days, which is real, but it cannot tell you whether
   a 36th station would fit the taxonomy.
6. **Coach loads are allocated, not simulated per passenger.** Boarding is
   split across coaches by FOB proximity and alighting proportionally to
   current load. Real passengers walk along platforms and change compartments;
   this approximation is why the coach-level tail is probably smoother than
   reality's.

## 12. What I would do first with real data

- Replace `simulate.py` with a loader for AFC/UTS gate counts and whatever
  coach-level signal exists (CCTV head-count, load cells, Wi-Fi/BLE probe
  counts). `features.py` and everything downstream would not change.
- Re-estimate the cost matrix with the operator, and re-run §6.6's sensitivity
  sweep as the *deliverable* rather than as a robustness check.
- Validate the band cut-points against incident records.
- Deploy the `realtime` feature set for intervention and the `schedule` set for
  overnight planning, since they answer different questions and only one of
  them needs telemetry.
- Re-fit the band classifier's probabilities on a rolling window rather than a
  fixed one. §6.5 shows post-hoc calibration transferring an expired base rate
  across a single seasonal boundary; in production that boundary arrives every
  year, and the answer is to keep the training window moving rather than to
  add a calibrator on top.
- Monitor the under-prediction rate on DANGEROUS coaches as the production SLO.
  RMSE is the wrong thing to alert on.

---

## 13. References

- Indian Railways suburban load taxonomy (normal / dense / super-dense crush
  load); Central Railway Harbour Line timetable and station list.
- Varian, H. R. (1975). *A Bayesian approach to real estate assessment* —
  the LINEX loss.
- Koenker, R. and Bassett, G. (1978). *Regression quantiles* — the pinball
  loss and the quantile-as-minimiser result the project's τ is derived from.
- Elkan, C. (2001). *The foundations of cost-sensitive learning* — the
  expected-cost decision rule used in `DistributionalPolicy`.
- Gneiting, T. (2011). *Making and evaluating point forecasts* — why the
  choice of loss determines which functional of `p(y|x)` you are estimating.
- Fruin, J. J. (1971). *Pedestrian Planning and Design* — the level-of-service
  density bands the IR taxonomy extends.
- Ke, G. et al. (2017). *LightGBM: A highly efficient gradient boosting
  decision tree*.

---

*Course capstone. Data is synthetic and generated by the simulator in this
repository; see §3.*
