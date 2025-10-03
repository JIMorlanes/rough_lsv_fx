
# Calibration Pipeline

This document outlines the calibration steps used to construct a realistic FX volatility surface from market quotes, and prepare it for pricing under Local-Stochastic Volatility (SLV) models.

---

## Step 1: Market Quote Preparation (`06_eSSVI_surface_prep.ipynb`)

We use quotes extracted from an article (ATM, 25Δ Put/Call, domestic and foreign interest rates, and spot). The goal is to produce structured targets of the form:

$$
(T, k, \sigma), \quad \text{with} \quad k = \log(K/F), \quad w = \sigma^2 T
$$

### Key Conversions

- Forward:
  $$
  F(T) = S_0 \cdot \exp\left((r_d - r_f) T\right)
  $$

- Strikes from delta (Call and Put, premium-included forward delta convention):
  $$
  K_C(\delta) = F \cdot \exp\left( -\sigma \sqrt{T} \left( N^{-1}(\delta) + \frac{1}{2} \sigma \sqrt{T} \right) \right)
  $$
  $$
  K_P(\delta) = F \cdot \exp\left( -\sigma \sqrt{T} \left( N^{-1}(1 - \delta) + \frac{1}{2} \sigma \sqrt{T} \right) \right)
  $$

- Log-moneyness and total variance:
  $$
  k = \log(K/F), \quad w = \sigma^2 T
  $$

- Calendar monotonicity of ATM variance:
  $$
  \theta(T) = \sigma_{\text{ATM}}^2 T
  $$
  Checked via finite differences and optionally smoothed using isotonic regression.

---

## Step 2: eSSVI Calibration (`07_fit_eSSVI_from_targets.ipynb`)

We calibrate the eSSVI model to the $(T, k, w)$ targets using the following parameterization:

$$
w(T,k) = \frac{\theta(T)}{2} \left(1 + \rho \phi(T) k + \sqrt{(\phi(T) k + \rho)^2 + 1 - \rho^2} \right)
$$

Where:
- $\theta(T)$: total ATM variance (monotone)
- $\phi(T) = \eta$: curvature parameter
- $\rho$: correlation parameter (typically $\rho < 0$ for FX skew)

### Fit Details

- Optimized variables: $\theta(T_i)$ (monotone), global $\rho \in (-1, 1)$, and $\eta > 0$
- Objective: weighted least squares (soft L1)
- Weights: inverse squared bid-ask spreads, ATM boosted
- Constraints:
  - $w$ is non-decreasing in $T$
  - $w$ is convex in $k$

### Outputs

- Implied volatility surface $\sigma_{\text{impl}}(T, k)$
- Heatmaps, smile plots, residual diagnostics
- Saved interpolators or surfaces for Dupire step

---

## Step 3: Dupire Local Volatility (`08_dupire_and_leverage.ipynb`)

From the fitted implied surface $\sigma_{\text{impl}}(T, k)$, compute local vol via the Dupire formula:

$$
\sigma_{\text{loc}}^2(T, K) =
\frac{ \partial_T C(T, K) + (r_d - r_f) K \, \partial_K C(T, K) }
{ \frac{1}{2} K^2 \, \partial_{KK} C(T, K) }
$$

Where $C(T, K)$ is the undiscounted call price from the surface.

---

## Step 4: SLV Leverage Extraction

Using the SLV condition:

$$
\sigma_{\text{SLV}}(t, S, V) = L(t, S) \cdot \sqrt{V_t}, \quad \text{with} \quad L^2(t, S) = \frac{\sigma_{\text{loc}}^2(t, S)}{\mathbb{E}[V_t \mid S_t = S]}
$$

We:
- Choose a stochastic volatility model (e.g., Heston or rough Heston)
- Run MC simulation
- Estimate conditional expectation $\mathbb{E}[V_t \mid S_t = S$ via binning or regression
- Compute $L(t, S)$ on a grid

---

More detail will be added as the calibration proceeds.
