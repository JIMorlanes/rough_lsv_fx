
# Calibration Pipeline

This document outlines the calibration steps used to construct a realistic FX volatility surface from market quotes, and prepare it for pricing under Local-Stochastic Volatility (SLV) models.

---

## Step 1: Market Quote Preparation (`06_eSSVI_surface_prep.ipynb`)

We use quotes extracted from an article (ATM, 25Δ Put/Call, domestic and foreign interest rates, and spot). The goal is to produce structured targets of the form:

$(T, k, \sigma), \quad \text{with} \quad k = \log(K/F), \quad w = \sigma^2 T$

### Key Conversions

- Forward:

$F(T) = S_0 \cdot \exp\left((r_d - r_f) T\right)$

- Strikes from delta (Call and Put, premium-included forward delta convention):

$$
K_C(\delta) = F \cdot \exp\left( -\sigma \sqrt{T} \left( N^{-1}(\delta) + \frac{1}{2} \sigma \sqrt{T} \right) \right)
$$

$$
K_P(\delta) = F \cdot \exp\left( -\sigma \sqrt{T} \left( N^{-1}(1 - \delta) + \frac{1}{2} \sigma \sqrt{T} \right) \right)
$$

- Log-moneyness and total variance:

$k = \log(K/F), \quad w = \sigma^2 T$

- Calendar monotonicity of ATM variance:

$\theta(T) = \sigma_{\text{ATM}}^2 T$

Checked via finite differences and optionally smoothed using isotonic regression.

## Isotonic regression: why and how we use it here

**Goal.** Enforce **calendar monotonicity** of per-tenor total variance series (e.g., $w(T)=\sigma^2 T$ for 25ΔP / ATM / 25ΔC), i.e. we want
$$
w(T_1)\le w(T_2)\le \cdots \le w(T_n)\quad \text{for } T_1<\cdots<T_n,
$$
because total variance must be **non-decreasing in $T$** (no calendar arbitrage).

**Problem.** Raw quotes are noisy; you may see tiny dips (e.g., at some $T$). We want the **closest** non-decreasing sequence to the observed data, changing it as little as possible.

**Isotonic regression (non-decreasing).** Given data $y_1,\dots,y_n$ at increasing tenors, solve
$$
\min_{x_1\le x_2\le \cdots \le x_n}\; \sum_{i=1}^n (x_i - y_i)^2.
$$
- This is the **orthogonal projection** (least-squares sense) onto the convex cone of non-decreasing sequences.
- The solution is computed by the **Pool-Adjacent-Violators Algorithm (PAVA)**:
  1. Start with each point as its own “block” with value $y_i$.
  2. Scan left→right. If a block’s value exceeds the next block’s value (a violation), **pool** the two blocks: replace both by a **weighted average** (weights = block sizes).
  3. Keep pooling backward until monotonicity is restored locally.
  4. Continue the scan; the result is piecewise constant and **non-decreasing**.

**Why here?** It gives an **objective, minimal** cleanup of $w(T)$ (or $\theta(T)$) so eSSVI/Dupire are fed **no-arb** inputs, without ad-hoc nudges. Small dips get “flattened” into short plateaus; rising regions remain untouched.

**What to apply it to.** Apply to just the **ATM** series $w_{\text{ATM}}(T)$, or to **all three** buckets $w_{25P}(T)$, $w_{\text{ATM}}(T)$, $w_{25C}(T)$ (calendar must hold at each fixed $k$ as well).

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
- Estimate conditional expectation $\mathbb{E}[V_t \mid S_t=S]$ via binning or regression
- Compute $L(t, S)$ on a grid

---

More detail will be added as the calibration proceeds.
