# GARCH-DDPM Financial Risk Assessment

A regime-conditional denoising diffusion probabilistic model for generating
return scenario distributions and estimating Value-at-Risk on equity indices.
The model combines GARCH(1,1) volatility forecasting with a tabular DDPM
to produce statistically validated VaR estimates that pass the Kupiec
Proportion of Failures backtest at both the 95% and 99% confidence levels.

---

## Motivation

Standard parametric VaR (Gaussian or Student-t) cannot capture the
regime-dependent, fat-tailed nature of equity return distributions. Historical
simulation VaR is distribution-free but ignores current market conditions.
This project asks: can a conditional generative model learn the full
conditional return distribution — including tail shape — from market
state features, and produce VaR estimates that are statistically valid
on out-of-sample data?

The answer from the backtest: yes, when the generative model is combined
with GARCH volatility scaling and empirical quantile mapping.

---

## Architecture Overview

The system has three components that work together at inference time.

**GARCH(1,1) — volatility envelope**

A GARCH(1,1) model is fit on training returns to produce a conditional
daily volatility forecast for each day. This captures volatility clustering —
the well-known empirical fact that large moves cluster together. The GARCH
component handles the scale of the return distribution; the DDPM handles
its shape.

**Tabular DDPM — shock distribution**

The DDPM does not learn raw returns. It learns GARCH-standardized residuals:

```
garch_resid = target_return / garch_vol
```

These residuals have unit variance by construction and capture the
non-Gaussian shape of the shock distribution — skewness, excess kurtosis,
and regime-dependent tail behavior — after the volatility envelope has been
removed. Training a generative model on residuals rather than raw returns
makes the learning problem substantially easier and eliminates the
train/test volatility mismatch that caused early versions of this model
to generate distributions that were systematically too wide.

**Empirical quantile mapping — distribution calibration**

The DDPM's raw output has incorrect absolute scale due to sampling
instability in the reverse diffusion chain. Rather than fixing this by
constraining the output (which collapsed the tail), the system uses a
post-hoc quantile map built from training residuals. Each generated sample
is converted to its rank-based quantile, which is then mapped to the
actual training residual value at that quantile. This preserves the rank
ordering learned by the model while replacing the broken absolute values
with historically-grounded ones. The result is a generated distribution
whose 5th and 1st percentiles match the empirical training distribution
exactly, regardless of the model's absolute output scale.

---

## Model Architecture

```
Input: [xt (noisy residual), regime_label, timestep_embedding]
          │
          ├── market_encoder(24 scaled features) → 64-dim
          │     [realized_vol, VIX, RSI, skew, kurtosis,
          │      vol_spike, RSI_velocity, garch_vol,
          │      6 lagged returns, 4 cumulative returns,
          │      4 realized vol windows, 2 vol ratios,
          │      regime_score]
          │
          ├── time_encoder(8 sin/cos features) → 32-dim
          │     [dow, month, day-of-month, quarter — all sinusoidal]
          │
          └── cond_fuse(64 + 32 → 64-dim conditioning embedding)
                    │
                    ▼
         Backbone: fc1 → FiLM → fc2 → FiLM → fc3 → FiLM → fc_out
                   (128)         (128)         (128)         (1)
                    │
                    ▼
         Output: predicted noise ε̂
```

**FiLM conditioning** (Feature-wise Linear Modulation) applies
per-layer scale and shift to the backbone activations using the
conditioning embedding. Initialized as identity (scale=1, shift=0)
so conditioning starts neutral and is learned from data.

**Classifier-Free Guidance** trains the model to handle both conditional
and unconditional generation. During training, 15% of samples have their
market features zeroed and regime label set to null — training the model
to generate without conditioning. At inference, two forward passes are
run per step (unconditional and conditional) and combined:

```
ε_guided = ε_uncond + guidance_scale × (ε_cond - ε_uncond)
```

This sharpens the conditioning signal — the model is more responsive
to today's market state than it would be from a single conditional pass.

---

## Training

**Target variable:** GARCH(1,1) standardized residuals, clipped to ±10σ.

**Noise schedule:** Cosine beta schedule with T=100 diffusion steps.
Cosine scheduling produces smoother alpha_bar decay than linear scheduling,
which is important at low T where each step carries more weight.

**Loss function:** SNR-weighted Smooth L1 loss.

```python
snr     = alphas_bar[t] / (1.0 - alphas_bar[t])
weights = (snr / snr.mean()).clamp(0.1, 10.0)
loss    = (weights * F.smooth_l1_loss(eps_pred, eps, reduction="none")).mean()
```

SNR weighting upweights low-noise timesteps (small t) where the denoising
prediction is hardest and most important for tail shape. Smooth L1 is
used instead of MSE because it is less sensitive to occasional large
residuals from extreme return observations.

**Regularization:** Weight decay 1e-4, dropout 0.1 in the market encoder
and between backbone layers, gradient clipping at norm 1.0.

**Schedule:** Linear warmup for 10 epochs, then cosine decay to zero over
the remaining epochs. Early stopping with patience 25 on validation loss.

**EMA:** Exponential moving average of model weights with decay 0.9999.
Sampling uses EMA weights rather than live training weights, which
produces cleaner distributions by averaging out SGD noise.

**Data split:** Strict chronological 80/10/10 — no shuffling across splits.
Scalers fit only on training data and applied to validation and test sets.
GARCH fit only on training returns; parameters fixed when computing
conditional volatility for the full dataset.

---

## Inference Pipeline

For each day in the test set, given a conditioning row with market features
and a GARCH volatility forecast:

**1. Initialize noise**

```python
xt = torch.zeros((n_paths, 1))
torch.nn.init.trunc_normal_(xt, mean=0, std=1.0, a=-3.0, b=3.0)
```

Truncated normal bounded to ±3 prevents extreme starting values from
causing trajectory explosion in the early denoising steps.

**2. Reverse diffusion (100 steps)**

At each step t from 99 down to 0, run two model forward passes and
combine with CFG. Apply eps clamp at ±3 to prevent per-step explosion.
Apply adaptive xt clamp during intermediate steps with width
`3 / sqrt(alpha_bar_t) + 1` — wide at high t (noisy), tight at low t
(nearly clean). No clamp at t=0 so that extreme tail samples retain
distinct values.

**3. Inverse-transform**

Apply the StandardScaler inverse transform to convert from scaled
residual space back to raw residual space.

**4. Quantile mapping**

```python
ranks  = (rankdata(raw_resids) - 0.5) / n_paths
mapped = quantile_map(ranks)
```

The quantile map was built once from training residuals:

```python
q_levels = np.linspace(0.0001, 0.9999, 10000)
q_values = np.quantile(train_resids, q_levels)
quantile_map = interp1d(q_levels, q_values, ...)
```

Each generated sample is replaced by the training residual at its
corresponding quantile. This maps the generated distribution's shape
onto the empirical historical distribution while preserving rank ordering.

**5. GARCH rescaling**

```python
returns = mapped_residuals * row["garch_vol"]
```

Multiplying by the GARCH daily volatility forecast scales the unit-variance
shock distribution to current market conditions. The distribution is
automatically wider on high-volatility days and narrower on low-volatility
days without any model changes.

**6. VaR and ES extraction**

```python
var_95 = np.percentile(paths, 5,  axis=1)
var_99 = np.percentile(paths, 1,  axis=1)
es_95  = paths[paths < var_95].mean()
es_99  = paths[paths < var_99].mean()
```

---

## Feature Set (33 total)

| Group | Features | Count |
|---|---|---|
| Lagged returns | 1, 2, 3, 5, 10, 21 day | 6 |
| Cumulative returns | 5, 10, 21, 63 day | 4 |
| Realized volatility | 5, 10, 21, 63 day annualized | 4 |
| Volatility ratios | 5d/21d, 21d/63d | 2 |
| Market indicators | Realized vol, VIX, RSI, RSI velocity | 4 |
| Statistical moments | Rolling skewness, kurtosis (60d) | 2 |
| Volume | Vol spike (relative to 20d mean) | 1 |
| GARCH | Daily conditional volatility forecast | 1 |
| Regime | Continuous score [0,1], binary label | 2 |
| Time encodings | Day-of-week, month, day, quarter (sin/cos) | 8 |

All features use only information available as of yesterday's close.
No lookahead. VIX and GARCH vol are shifted by one day before merging.
Rolling statistics use `.shift(1)` throughout.

---

## Regime Score

The regime score is a soft stress indicator bounded in [0, 1] combining
four lagged signals with fixed weights:

```
regime_score = 0.30 × price_stress
             + 0.30 × vol_percentile
             + 0.20 × skew_stress
             + 0.20 × vix_percentile
```

Where price_stress measures distance below SMA-200, vol_percentile is the
252-day rolling percentile rank of realized volatility, skew_stress captures
negative skewness, and vix_percentile is VIX's 252-day rolling rank.

The binary regime label is derived as `regime = (regime_score > 0.5)`.
Both the continuous score and binary label are used as conditioning inputs —
the score enters the market encoder as a raw feature, the binary label
is the CFG class variable.

---

## Backtest Results

Evaluated on the out-of-sample test set using the Kupiec Proportion of
Failures test. The null hypothesis is that the model's breach rate equals
the target alpha. A p-value above 0.05 means we cannot reject correct
calibration.

| Metric | Target | Actual | Breaches | p-value | Status |
|---|---|---|---|---|---|
| VaR 95% | 5.000% | 5.913% | 23 / 389 | 0.4138 | PASS |
| VaR 99% | 1.000% | 1.799% | 7 / 389 | 0.1185 | PASS |

VaR99/VaR95 ratio: 1.78 (training residual ratio: 1.78 — exact match,
confirming the quantile map is correctly calibrated).

The PIT histogram is approximately uniform across [0, 1], confirming
the model is well-calibrated across the full return distribution, not
just at the tail quantiles tested by Kupiec.

---

## Dependencies

```
torch >= 1.10
yfinance
pandas
numpy
scikit-learn
scipy
arch
tqdm
matplotlib
```

---

## Repository Structure

```
├──  ddpm_backtest
        ├── __init__.py
        ├── _modidx.py
        ├── core.py
        ├── data_loaders.py
        ├── diffusion_utils.py
        ├── models.py
        ├── noising_time.py
        ├── sampling_utils.py
├──  nbs
    ├── _quarto.yml
    ├── 00_core.ipynb
    ├── best_model.pt
    ├── cond_for_testing.ipynb
    ├── data_loaders.ipynb
    ├── diffusion_model.ipynb
    ├── generating_data.ipynb
    ├── index.ipynb
    ├── model.ipynb
    ├── nbdev.yml
    ├── noising_time.ipynb
├── plots
├── setup.py
├── README.md               — this file
└── garch_ddpm_checkpoint.pkl  — saved model weights, scalers, quantile map
```

---

## Limitations and Future Work

The quantile mapping step anchors the generated distribution to the
empirical training distribution. This is correct when future residuals
come from the same distribution as past residuals, but will underestimate
tail risk during structural breaks or regimes not present in training data.

The DDPM's raw output is numerically unstable — the absolute values are
meaningless and only rank ordering is preserved. A more stable training
procedure (DDIM sampling, flow matching, or a different noise schedule)
would eliminate the need for post-hoc quantile mapping and allow the
model's conditioning to influence the distribution shape more directly.

The current architecture generates one scalar (tomorrow's return) given
today's market state. Extending to multi-step path generation would
enable scenario simulation over horizons longer than one day.

Replacing the normal GARCH distribution assumption with a Student-t or
GJR-GARCH specification would better capture the asymmetric leverage
effect and improve the accuracy of the GARCH volatility forecast itself,
which feeds directly into VaR quality.
