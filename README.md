# Diffusion-Based Financial Risk Modeling (NIFTY 50)

A **Tabular DDPM** with **Classifier-Free Guidance (CFG)** and **FiLM conditioning** for generating financial return scenarios and estimating tail risk metrics (VaR / CVaR) on NIFTY 50 data.

> **Key result:** 99% VaR estimated within ~16% error of realized losses using diffusion-generated scenarios — outperforming parametric Gaussian baselines on tail coverage.

---

## Why This Exists

Standard risk models (parametric VaR, Historical Simulation) fail on empirical return distributions because they assume Gaussian innovations or are constrained by historical sample size. Real returns have fat tails, volatility clustering, skewness, and regime-dependent behavior.

This project uses a **diffusion model to learn the full empirical return distribution** — no parametric assumptions — and generates synthetic scenarios conditioned on current market state for risk estimation.

---

## Results

| Metric | Confidence Level | generated_data | actual_data | Error |
|--------|-----------------|------------|----------------------|-----------------|
| VaR    | 95%             | -0.010617  | -0.010422            | 1.7%            |
| VaR    | 99%             | -0.013996  | -0.014863            | 5.8%            |
| CVaR   | 95%             | -0.014970  | -0.015068            | 0.7%            |
| CVaR   | 99%             | -0.020772  | -0.023314            | 6.9%            |


---

## Architecture

```
Market Data (NIFTY 50, VIX, Volume)
        ↓
  Feature Engineering
  [realized vol, RSI, skewness, kurtosis, regime label,regime_score]
        ↓
  Forward Diffusion (cosine schedule)
  x_t = √αₜ · x₀ + √(1−αₜ) · ε
        ↓
  FiLMNet (conditioning on market state)
  γ(z) · h + β(z)   ← dynamic modulation per timestep
        ↓
  Reverse Diffusion + CFG
  [classifier-free guidance for regime-conditional generation]
        ↓
  Synthetic Return Scenarios
        ↓
  VaR / CVaR Estimation
```

**FiLM conditioning** allows the network to modulate its activations based on market context — realized vol, VIX, RSI, skewness, kurtosis, regime — rather than naively concatenating conditioning variables to input.

**CFG** enables regime-conditional generation with guidance scale control, interpolating between conditional and unconditional distributions at inference.

---

## Market Regime Conditioning

The model is conditioned on a discrete **market regime label** (low-vol / high-vol / crisis) in addition to continuous market state variables. This allows scenario generation targeted at specific market environments — useful for stress testing.

---

## Dataset

- **Source:** `yfinance` (NIFTY 50 index, VIX, volume)
- **Features:** Realized volatility, VIX, RSI, RSI velocity, return skewness, return kurtosis, volatility spike indicator, market regime,regime_score

---

## Project Structure

```
ddpm_risk/
├── ddpm_lib/
      ├── data_loaders.py      # Raw and processed datasets
      ├── models.py             # FiLM and f_net architecture
      ├── diffusion_utils.py          # FiLMnet architecture
      ├── noisify_time_embeddings.py # cosine nosify and timestep emb
      ├── sampling.py # sampling architecture
      ├── VaR_and_Es.py #value at risk and expected shortfall metric evaluation
├── nbs/          # Training experiments and analysis
      ├── load_data.ipynb
      ├── noising_time.ipynb
      ├── f_net_FiLM.ipynb
      ├── FiLMnet_arc.ipynb
      ├── generating_data.ipynb
      ├── Var.ipynb
      ├── abalation_test.ipynb
├── plots/              # Distribution and risk visualizations

└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/dhanwariagarvit21/ddpm_risk.git
cd ddpm_risk
pip install -r requirements.txt

# Run main notebook
jupyter VaR_ES.ipynb
```

---

## Backtesting


*(Stress-testing validation under active development.)*

---

## Roadmap

- [x] Tabular DDPM with cosine noise schedule
- [x] FiLM conditioning on market state
- [x] Classifier-Free Guidance (CFG)
- [x] VaR / CVaR estimation from generated scenarios
- [] Kupiec / Christoffersen backtesting
- [ ] GARCH pre-processing for autocorrelation removal in inputs
- [ ] DDPM vs GARCH empirical comparison
- [ ] Multi-asset portfolio extension
- [ ] Stress testing under simulated regimes

---

## References

- Ho et al. *Denoising Diffusion Probabilistic Models* (NeurIPS 2020)
- Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations* (ICLR 2021)
- Hull, J. *Options, Futures and Other Derivatives*
- FastAI *Practical Deep Learning for Coders* — Stable Diffusion lectures

---
## Limitations of v1

The initial version evaluated generated samples by comparing
their marginal distributions to real data.

This ignored temporal dependencies (autocorrelation, clustering),
leading to misleadingly strong results.


*Research and educational purposes only. Not financial advice.*
