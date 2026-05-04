# Project — Likelihood-free inference for sums of log-normal variates

## Setup

We observe $n$ data points $Y_1, \dots, Y_n$, where each $Y_i$ is a sum of $L$ log-normal variates:

$$
Y_i = \sum_{\ell=1}^{L} \exp(X_{i,\ell}), \qquad X_{i,\ell} \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2).
$$

The goal is to estimate $\theta = (\mu, \sigma^2)$. The density of $Y_i$ has no closed form, which rules out standard likelihood-based methods. Throughout this project we use synthetic data generated from the true parameters $L = 10$, $\mu_0 = 0$, $\sigma_0 = 0.3$ (i.e. $\sigma_0^2 = 0.09$), so we always know the actual truth.

---

## Question 1 — Reject-ABC

*The code for this section and more detailed insights about the results can be found in `Reject_ABC.ipynb`.*

### Why ABC

Because the density of $Y_i$ is intractable, we cannot evaluate the likelihood directly. What we can do is simulate data for any proposed $\theta$. Reject-ABC exploits exactly this: propose a parameter, simulate fake data, and keep the proposal only if the fake data looks close enough to what we observed.

### Distance

Following Bernton et al. (2017), we use a **MEWE-style averaged score**: for each proposed $\theta'$, we simulate $K = 3$ independent fake datasets and average their 1-Wasserstein distances to the observed sample,

$$
\bar{W}_{1,K}(\theta') = \frac{1}{K}\sum_{k=1}^{K} W_1\!\left(Y^{(k)}(\theta'),\,y^\star\right),
$$

where each 1-Wasserstein term is just the mean absolute gap between sorted samples,

$$
W_1\!\left(Y^{(k)}(\theta'),\, y^\star\right) = \frac{1}{n}\sum_{i=1}^{n}\left|Y^{(k)}_{(i)}(\theta') - y^\star_{(i)}\right|.
$$

Averaging over $K$ replications reduces the Monte Carlo noise that comes from judging a proposal with a single fake sample. It compares the full empirical distributions rather than a few hand-picked summary statistics, which matters here because no single statistic clearly captures all the information.

### Prior

Following the project statement we use:

$$
\mu \sim \mathcal{N}(0, s^2), \qquad \log(\sigma^2) \sim \mathcal{N}(0, t^2).
$$

The log-normal reparametrisation keeps $\sigma^2$ positive automatically.

![Observed data distribution](Reject_ABC_plots/observed_data_dist.png)
*Figure 1: Observed data distribution.*

### The algorithm

```
while number of accepted draws < N:
    1. draw θ' = (μ', σ²') from the prior
    2. for k = 1, ..., K: simulate Y^(k)(θ') = (Y^(k)_1(θ'), ..., Y^(k)_n(θ')) under θ'
    3. compute W̄_{1,K}(θ') = (1/K) Σ_k W1(Y^(k)(θ'), y*)
    4. if W̄_{1,K}(θ') ≤ ε: keep θ'
```

The accepted draws are i.i.d. samples from the ABC posterior.

### Calibrating ε

Before running, we calibrated $\varepsilon$ by looking at the prior predictive distribution of distances: draw many $\theta$ values from the prior, simulate data under each, compute $W_1$, and use a low quantile of those distances as $\varepsilon$.

We chose the **1% quantile** ($\varepsilon \approx 0.98$), which corresponds to an acceptance rate of about 1% — roughly 100 proposals per accepted draw. Tighter tolerances improve the approximation but cost much more.


| Quantile | $\varepsilon$ | Acceptance rate | Proposals per accept |
| -------- | ------------- | --------------- | -------------------- |
| 20%      | ~4.2          | ~20%            | ~5                   |
| 10%      | ~2.8          | ~10%            | ~10                  |
| 1%       | ~0.98         | ~1%             | ~100                 |
| 0.5%     | ~0.73         | ~0.5%           | ~200                 |

### Results

![Epsilon calibration](Reject_ABC_plots/eps_calibration.png)
*Figure 2: Epsilon calibration.*

With $\varepsilon$ at the 1% quantile, the posterior for $\mu$ is reasonably centered near the truth. The 95% credible interval covers $\mu_0 = 0$. The joint posterior also shows the expected negative correlation between $\mu$ and $\sigma^2$: both parameters affect the scale of $Y_i$, so they partially compensate each other.

The harder case is $\sigma^2$. Its posterior sits above the true value $0.09$. It seems to be the effect of two things: a finite positive $\varepsilon$ (which always leaves some ABC approximation bias), and a prior on $\log\sigma^2$ centered at $\sigma^2 = 1$, which pulls the estimates upward.

**Effect of $\varepsilon$:**

As $\varepsilon$ decreases, the posterior gets more concentrated around a specific value which is not exactly the true value of the parameter (bias). The improvement from 10% to 1% is large. Going further (to 0.5%) helps a bit more but costs twice as many proposals. Indeed, the acceptance rate increases with epsilon, which is an expecte result as

![Posterior as a function of epsilon](Reject_ABC_plots/posterior_eps.png)
*Figure 3: The posterior distribution for some values of epsilon*

![Acceptance rate as a function of epsilon](Reject_ABC_plots/acc_rate_eps.png)
*Figure 4: As epsilon increases, the acceptance rate gets higher.*

**Effect of $s$ and $t$:**

![Sensitivity to the prior](Reject_ABC_plots/prior_sensitivity.png)
*Figure 5: Sensitivity of the posterior to a change of s and t.*

Varying $s$ barely changes the posterior. A wider prior just lowers the acceptance rate by sending more proposals to extreme $\mu$ values, but since the prior is already centered at the true $\mu_0 = 0$, the posterior summaries stay almost the same.


| Value of s | Acceptance rate |
| :--------: | :-------------: |
|    0.3    |      3.34%      |
|    1.0    |      1.02%      |
|    3.0    |      0.35%      |

Varying $t$ matters a lot more. When $t = 0.3$, the prior is so tight around $\sigma^2 = 1$ that almost no mass reaches the true region near $0.09$. Only 11 draws were accepted, which is close to a failure. When $t = 3.0$, the prior finally covers the truth, the acceptance rate goes up, and the posterior moves much closer to $\sigma_0^2 = 0.09$.

Eventually, $t$ matters much more than $s$ in this problem.

**Numerical error:**

In order to identify separately the standard deviation of our simulation with the standard deviation due to the ABC method itself, we repeated the algorithm 20 times with different seeds, keeping $\varepsilon$ fixed.

![Numerical error](Reject_ABC_plots/num_error.png)
*Figure 6: Assessment of the numerical error*

The Monte Carlo standard deviations across runs are tiny compared with the gap between estimates and true values. So the main issue is actually the approximation itself due to a finite $\varepsilon$ and prior misspecification.

**Posterior predictive check:**

![Posterior check](Reject_ABC_plots/posterior_check.png)
*Figure 7: Checking the posterior distribution*

Samples from the ABC posterior, when used to simulate new datasets, produce data that is centered roughly right but visibly more spread out than $y^\star$. The observed standard deviation is about 1.01 while the mean replicated one is about 1.65. This overdispersion confirms that the posterior is still placing too much mass on large $\sigma^2$ values.

---

## Question 2 — MCMC-ABC

*The code for this section and more detailed insights about the results can be found in `MCMC_ABC_optimizedv3.ipynb` (performance assessment) and `MCMC_ABCv2.py` (base functions).*

### Why MCMC-ABC

Reject-ABC is conceptually simple, but it is wasteful: every proposal is drawn independently from the prior, so most simulated datasets are far from the observed one and are immediately discarded. MCMC-ABC, introduced by Marjoram et al. (2003), keeps the same ABC idea but turns it into a Markov chain. Instead of restarting from the prior each time, it proposes a new parameter close to the current one, simulates pseudo-data, and accepts the move only if the simulated data is close enough to the observations and the Metropolis-Hastings ratio is favorable. In practice, this means that once the chain has reached a plausible region, it keeps exploring locally instead of wasting most proposals in irrelevant parts of the prior.

### Distance

We keep the same MEWE distance as in Q1 as it is more robust. $K\_MEWE$, the number of W1 computed to derive the MEWE is set to 20.

### Prior

As in Q1, we use

$$
\mu \sim \mathcal{N}(0, s^2), \qquad \log(\sigma^2) \sim \mathcal{N}(0, t^2).
$$

The log-parametrisation keeps $\sigma^2 > 0$ automatically. The hyperparameters $s$ and $t$ control how diffuse the prior is on $\mu$ and $\log(\sigma^2)$ respectively.

### The algorithm

The MCMC-ABC chain targets the augmented distribution

$$
\pi(\theta, y^{\mathrm{aug}}) \propto \pi(\theta)\;\mathbf{1}\!\left[W_1(y^{\mathrm{aug}}, y^\star) \leq \varepsilon\right],
$$

whose marginal in $\theta$ is the ABC posterior. At each step we use a random-walk proposal on $(\mu, \log \sigma)$ with step size $\delta$:

```
Given current state (θ, y^aug):
  1. Propose θ' = θ + δ · ξ,   ξ ~ N(0, I_2)
  2. Simulate K_MEWE datasets Y^(k)(θ'), k = 1, ..., K_MEWE under θ'
  3. If MEWE(θ', y*) > ε: reject immediately
  4. Otherwise accept θ' with probability min(1, π(θ') / π(θ))
```

So $\varepsilon$ controls the ABC approximation itself, while $\delta$ controls how far each proposal jumps from the current state. The first parameter mainly affects bias, the second mainly affects mixing.

The assessment of MCMC-ABC is presented hereafter and contains:
- **A baseline run:** The algorithm is run using a fixed set of parameters with N_CHAINS completing N_BURN + N_ITER steps over each observed dataset (N_DATASETS in total). It yields some performance metrics as well as three types of standard deviation.
- **Scans:** For some parameters of interest, the algorithm is run multiple times over a grid of value for one varying parameter. It provides an idea of how some metrics are affected by the specified parameter.

### Baseline run and recurring diagnostics

The fixed set of parameters used in this baseline is $\varepsilon = 1$, $\delta = 0.25$, $N_{\mathrm{burn}} = 30\,000$, $N_{\mathrm{iter}} = 15\,000$, $N_{\mathrm{chains}} = 4$, $N_{\mathrm{datasets}} = 3$, $s = t = 1$. The key diagnostics are:

| Metric               | Definition                         | Value (avg. datasets) | Inter-dataset std (C) |
|---------------------|-----------------------------------------------|-----------------------|-----------------------|
| Bias μ              | posterior_mean − 0.0                        | -0.08002              | 0.00251               |
| Bias σ              | posterior_mean − 0.3                        | 0.18157               | 0.00354               |
| Posterior std μ (A) | std (pooled samples)                     | 0.06561               | 0.00123               |
| Posterior std σ (A) | std (pooled samples)                     | 0.09717               | 0.00325               |
| MC std μ (B)        | std of chain's mean                        | 0.00295               | 0.00128               |
| MC std σ (B)        | std of chain's mean                        | 0.00725               | 0.00263               |
| 95% CI width μ      | q97.5 − q2.5 (pooled)                         | 0.23842               | 0.00391               |
| 95% CI width σ      | q97.5 − q2.5 (pooled)                         | 0.35978               | 0.01194               |
| R-hat μ             | Gelman-Rubin stat                 | 1.00066               | 0.00036               |
| R-hat σ             | Gelman-Rubin stat                 | 1.00219               | 0.00116               |
| ESS ratio μ         | ESS / (N_CHAINS × N_ITER)                     | 0.01079               | 0.00020               |
| ESS ratio σ         | ESS / (N_CHAINS × N_ITER)                     | 0.00582               | 0.00093               |
| Acceptance rate     | average over the chains                         | 0.15615               | 0.00124               |
Note that three types of standard deviations are reported are: (A); (B) and (C)

- (A) refers to the std of the posterior distribution itself;
- (B) refers to the std of the results among the $N_{\mathrm{chains}}$ chains run simultaneously over each dataset;
- (C) refers to the std of the results among the $N_{\mathrm{datasets}}$ datasets.

The metrics above are clear: std of type (B) and (C) are almost negligeable. For this reason, the scan over $s$ and $t$ will be performed only on one chain and one dataset (as the computations are alrerady quite expensive)

Also, the Gelman Rubin stat tests if multiple chains converge towards the same distribution. It should be close to 1, and values above about 1.1 indicate that chains have not stabilized to the same distribution.

With this set of parameters on the baseline, the posterior for $\mu$ is reasonably close to the truth: the mean bias is about $-0.008$. The harder parameter is $\sigma$, whose bias is about $0.18$. This is the same qualitative issue as in Q1: with a finite $\varepsilon$, the ABC posterior is only an approximation, and the prior on $\log(\sigma^2)$ is centered around $\sigma^2 = 1$, which tends to pull the inference upward. The chain is usable but not efficient, with ESS ratios around 1.0% for $\mu$ and 0.5% for $\sigma$.

### Scan on $\varepsilon$

Smaller $\varepsilon$ reduces the ABC approximation bias but also reduces the acceptance rate, which slows chain mixing and raises Monte Carlo uncertainty. The table below ($\delta = 0.25$ fixed, averaged over 3 datasets) illustrates the trade-off:

| Values of $\varepsilon$| 0.15  | 0.46  | 0.66  | 0.97  | 1.47  | 2.49  | 3.00  |
|---------------------|-------|-------|-------|-------|-------|-------|-------|
| Acceptance rate     | 1.2%  | 6.5%  | 9.8%  | 14.9% | 23.6% | 39.4% | 45.3% |
| Bias μ              | -0.006| -0.025| -0.044| -0.076| -0.128| -0.231| -0.277|
| Bias σ              | +0.018| +0.071| +0.114| +0.174| +0.256| +0.378| +0.423|
| ESS ratio μ         | 0.13% | 0.46% | 0.69% | 0.92% | 1.36% | 1.56% | 1.53% |

![Epsilon scan — 9 diagnostics](MCMC-ABC_plots/scan_epsilon.png)
*Figure 2: Sensitivity of all diagnostics to $\varepsilon$ (5 datasets, 5 chains each). Each curve is the mean over datasets; the shaded band is ±1 inter-dataset std.*

Some elements from these curves:

**Bias increases with $\varepsilon$.** Both $|\mathrm{bias}_\mu|$ and $|\mathrm{bias}_\sigma|$ grow roughly linearly with $\varepsilon$. At $\varepsilon = 3.0$, the bias on $\mu$ reaches $-0.277$ and the bias on $\sigma$ reaches $+0.423$.

**The acceptance rate increases with $\varepsilon$.** A looser constraint accepts more proposals. To reach the 20–40% target zone (which is good compromise between acceptance rate and bias) one needs $\varepsilon \gtrsim 1.5$, but that comes at the cost of increased bias.

**The confidence intervals widen with $\varepsilon$.** The 95% CI width grows linearly: a larger $\varepsilon$ accepts a broader range of parameters, inflating the posterior.

**Behavior at very small $\varepsilon$.** At $\varepsilon = 0.05$, almost no proposals are accepted (acceptance rate $\approx 0.003\%$). The chain barely moves away from its initialization point. As a consequence, $\hat{R}$ jumps to $\approx 1.23$. The chains are stuck near their respective starting positions and never reach a common distribution. With such small values of epsilon, the results are dominated by where the chain started, not by the actual posterior. It explains the behavior of all teh curves between 0 and 0.5.

The key takeaway here is the same as in Q1: there is no free lunch. Reducing $\varepsilon$ improves accuracy but costs efficiency.

### Effect of $\delta$

$\delta$ is the random-walk step size on $(\mu, \log \sigma)$. It controls mixing efficiency but not the ABC approximation quality: across all tested values, $\mathrm{bias}_\mu \approx -0.13$ and $\mathrm{bias}_\sigma \approx +0.25$ remain essentially flat ($\varepsilon = 1.5$ fixed throughout).

| $\delta$ | Acceptance rate |
| --- | --- |
| 0.01 | 89.9% |
| 0.09 | 47.2% |
| 0.17 | 25.0% |
| **0.20** | **20.7%** |
| 0.26 | 14.9% |
| 0.50 | 5.2% |
| 0.80 | 2.2% |

![Delta scan](MCMC-ABC_plots/scan_delta.png)
*Figure 4: ESS ratio and acceptance rate as a function of $\delta$ ($\varepsilon = 1.5$ fixed, 5 datasets).

Some elements from these curves:

**Very small $\delta$:** acceptance rate is high but consecutive draws are nearly identical, which yields a low ESS. When $\delta$ becomes larger, proposals jump far and rarely pass the ABC tests, which yields a low acceptance rate and a low ESS. The ESS ratio peaks around $\delta \approx 0.20$ with an acceptance rate near 21%. Note that in every case, the ESS ratio remains very low.

### Effect of $s$ and $t$

We scan a $10 \times 10$ grid of $(s, t)$ values. For computational time reasons, the scan presented relies on a single chain, which has been run on a single dataset.

![Prior sensitivity heatmaps](MCMC-ABC_plots/scan_prior_heatmaps.png)
*Figure 5: Heatmaps over the full $(s,t)$ grid. The vertical structure (columns nearly uniform) shows that $s$ barely matters; the horizontal structure (rows vary strongly) shows that $t$ dominates.*

![Prior sensitivity 1D cuts](MCMC-ABC_plots/scan_prior_coupes1D.png)
*Figure 6: 1D cross-sections of the scan on (s,t)*

**Effect of $s$ (width of prior on $\mu$).** 

The bias and ESS ratio are nearly flat across all tested $s$ values. The posterior summaries barely change because the data already concentrates the posterior near the truth. This is exactly the same finding as in Q1.

**Effect of $t$ (width of prior on $\log \sigma^2$).** 

$t$ has a much stronger effect:

At $t = 0.1$ the prior on $\log \sigma^2$ is so tight around $\sigma^2 = 1$ that it barely covers the true $\sigma_0^2 = 0.09$. The acceptance rate collapses to $0.023\%$, meaning that the chain nearly fails. For greater values of $t$, the bias on both parameters decreases: the chain is able to explore more the space of parameters and ends up converging closer to their actual values. For similar reasons, the acceptance rate also increases with $t$.

The conclusion is the same as Q1: $s$ **barely matters** whereas $t$ **matters a lot**.

### Computational scaling and conclusion

![Computation time](MCMC-ABC_plots/scaling_time.png)
*Figure 7: Computation time as a function of $N_{\mathrm{iter}}$ (left) and $N_{\mathrm{chains}}$ under `vmap` parallelization (right).*

With the number of chains fixed, runtime grows roughly linearly. With `vmap` (jax), increasing the number of chains from 1 to 16 raises runtime roughly linearly too. This result is suprising as we would expect to see an almost constant by part curve when $N_{\mathrm{chains}}$ varies. However, the `vmap` relies on the CPU, which may not be optimal (it should be faster on a GPU). It is likely that the CPU is already overwhelmed with a single chain, meaning that the parallelization is quite limited.

Overall, MCMC-ABC has qualitatively a similar behavior to Reject-ABC. But the main lesson from the scans is that the roles of the new tuning parameters (introduce by the chains) are different. $\varepsilon$ is the main accuracy-efficiency trade-off: larger values improve acceptance but increase bias and posterior width. $\delta$ mainly controls mixing and should be tuned near the region where acceptance stays around 20–25%. Finally, prior sensitivity is driven much more by $t$ than by $s$. In this problem, the method works reasonably well for $\mu$, but inference on $\sigma$ remains the hardest part and stays sensitive to both the ABC tolerance and the prior on $\log(\sigma^2)$.


### Direct comparison with Reject-ABC

After studying the behavior of MCMC-ABC, it is useful to compare it directly with Reject-ABC on the same tolerance grid:

$$
\varepsilon \in \{0.5,\ 0.7,\ 1.0,\ 1.5,\ 2.0,\ 3.0\}.
$$

Both algorithms are run on the same observed datasets, so the comparison is fair. The goal here is not only to know which method is faster, but also to understand what we gain or lose when replacing the simple rejection scheme by a Markov chain.

![Reject-ABC vs MCMC-ABC benchmark](benchmark_results/benchmark2_plots.png)

*Figure: Benchmark comparison between Reject-ABC and MCMC-ABC across several values of $\varepsilon$.*

Reject-ABC is much more sensitive to $\varepsilon$. At $\varepsilon = 0.5$, it takes about $66.5$ seconds, while MCMC-ABC takes about $16.0$ seconds. This happens because Reject-ABC proposes directly from the prior, so when the ABC tolerance is strict, most proposals are rejected. When $\varepsilon$ increases, Reject-ABC becomes much faster and reaches only about $2.2$ seconds at $\varepsilon = 3.0$, simply because the acceptance condition is easier.

MCMC-ABC has a more stable runtime, around $13$ to $17$ seconds. This is because the number of burn-in and sampling iterations is fixed. Changing $\varepsilon$ changes the acceptance rate, but not the total number of iterations.

The acceptance rates confirm this difference. Reject-ABC accepts almost nothing for small $\varepsilon$, and only reaches about $12\%$ at $\varepsilon = 3.0$. MCMC-ABC accepts much more often: about $13\%$ at $\varepsilon = 0.5$, $26\%$ at $\varepsilon = 1.0$, and more than $60\%$ at $\varepsilon = 3.0$. This is because MCMC-ABC proposes locally around the current parameter, while Reject-ABC restarts from the prior each time.

However, higher acceptance is not necessarily better. With large $\varepsilon$, the ABC condition is looser, so the algorithm accepts more simulations that are not very close to the observed data. This improves speed, but worsens the approximation.

This is visible in the posterior means. For a fixed $\varepsilon$, both algorithms give very similar estimates, so they seem to target the same ABC posterior. But as $\varepsilon$ increases, the posterior means move away from the truth: $\mu$ goes from about $-0.03$ to about $-0.27$, while the true value is $0$, and $\sigma^2$ goes from about $0.14$ to more than $0.55$, while the true value is $0.09$.

So the main issue is not the choice between Reject-ABC and MCMC-ABC. The visible bias mainly comes from the finite ABC tolerance. This is also consistent with the prior sensitivity results above, where the prior on $\log(\sigma^2)$ had an important effect on the variance estimate.

Finally, ESS per second gives another view of the comparison. For large $\varepsilon$, Reject-ABC becomes very efficient because its accepted draws are independent and it finishes quickly. MCMC-ABC is limited by the correlation between consecutive samples. Still, MCMC-ABC is more useful for small $\varepsilon$, where Reject-ABC is too slow.

Overall, MCMC-ABC improves computation when the tolerance is strict, but it does not remove the ABC bias caused by $\varepsilon$. In this benchmark, $\varepsilon \approx 1.0$ seems to be the best compromise: both methods have similar runtime, the acceptance rate is reasonable, and the bias is still much smaller than for $\varepsilon = 2.0$ or $\varepsilon = 3.0$.

---
 
## Question 3 — Exact Gibbs Sampler via Data Augmentation

*The code for this section can be found in `exact_sampler.ipynb`.*


### Mathematical Justification

The density of $Y_i$ is intractable, so a direct posterior on $\theta = (\mu, \sigma^2)$ cannot be evaluated. The key idea is data augmentation: we treat the latent variables $X = \{X_{i,\ell}\}$ as part of the target, giving the new posterior:

$$
p(\theta, X \mid Y) \;\propto\; \pi(\theta)\; p(X \mid \theta) \prod_{i=1}^n \mathbf{1}\!\left(Y_i = \sum_{\ell=1}^L e^{X_{i,\ell}}\right)
$$

Because the observation $Y_i$ is a deterministic function of $X_i$, the likelihood collapses to an indicator and the augmented target is tractable. We then run a Gibbs sampler cycling between two blocks.

**Block 1 - $\theta \mid X$ (exact draw).**

Given $X$, the indicator provides no additional information on $\theta$ as $Y$ is perfectly known once we know $X$, so $p(\theta \mid X, Y) = p(\theta \mid X)$. Using a Normal-Inverse-Gamma (NIG) conjugate prior on $(\mu, \sigma^2)$:

$$
\sigma^2 \sim \mathrm{IG}(a_0, b_0), \qquad \mu \mid \sigma^2 \sim \mathcal{N}(m_0,\, \sigma^2/\kappa_0)
$$

the posterior after observing $N = nL$ latent values is again NIG with the parameters:

$$
\kappa_n = \kappa_0 + N, \quad m_n = \frac{\kappa_0 m_0 + N\bar X}{\kappa_n}, \quad a_n = a_0 + \frac{N}{2}, \quad b_n = b_0 + \frac{S^2}{2} + \frac{N\kappa_0}{2\kappa_n}(\bar X - m_0)^2
$$

We draw $\sigma^2 \sim \mathrm{IG}(a_n, b_n)$ then $\mu \mid \sigma^2 \sim \mathcal{N}(m_n, \sigma^2/\kappa_n)$. So for this bloc, once we know the distribution to sample from, is not an issue anymore.

**Block 2 - $X \mid \theta, Y$ (MH-within-Gibbs).**

Because the indicator is making sure that $\sum_\ell e^{X_{i,\ell}} = Y_i$, we cannot update $X_{i,\ell}$ coordinates independently as the equality would not be true anymore. For each observation $i$, we randomly pick a pair $(\ell, \ell')$ and propose an update that preserves the constraint exactly. Defining the invariant sum $s = e^{X_{i,\ell}} + e^{X_{i,\ell'}}$ and the unconstrained difference $\delta = X_{i,\ell} - X_{i,\ell'}$, we propose:

$$\delta^{\mathrm{new}} \sim \mathcal{N}(\delta^{\mathrm{curr}},\, \tau^2)$$

and recover the constrained coordinates using the softplus function:

$$X_\ell^{\mathrm{new}} = \log s - \mathrm{softplus}(-\delta^{\mathrm{new}}), \qquad X_{\ell'}^{\mathrm{new}} = \log s - \mathrm{softplus}(\delta^{\mathrm{new}})$$

The MH acceptance ratio simplifies and eventually, the Gaussian log-prior ratio is:

$$\log \alpha = -\frac{(X_\ell^{\mathrm{new}} - \mu)^2 + (X_{\ell'}^{\mathrm{new}} - \mu)^2}{2\sigma^2} + \frac{(X_\ell - \mu)^2 + (X_{\ell'} - \mu)^2}{2\sigma^2}$$

---

### Implementation

**Initialization.**

The chain must start on the constraint space $\{\sum_\ell e^{X_{i,\ell}} = Y_i\}$. We initialize with $X_{i,\ell}^{(0)} = \log(Y_i / L)$, which satisfies the constraint exactly by construction.

**Hyperparameter calibration.**

Rather than fixing arbitrary hyperparameters, we calibrate them empirically from the data: $m_0 = \overline{\log(Y/L)}$, $\kappa_0 = 1$, $a_0 = 2$, $b_0 = \mathrm{Var}(\log(Y/L))$. This places the prior near the data without being informative.

**Some details about our implementation**

- Exponentiating large $X$ values risks overflow. The invariant sum is therefore computed as $\log s = \mathrm{logsumexp}(X_\ell, X_{\ell'})$, and the back-projection uses the softplus function throughout, so that all arithmetic stays in the log domain.
- The $n$ observations are conditionally independent given $\theta$. We use `jax.vmap` to run the pairwise MH kernel over all $n$ observations simultaneously, replacing a Python loop with a single batch operation.
- To reduce autocorrelation of the $X$ chain between $\theta$ updates, we run $K = 10$ pairwise MH sub-steps per Gibbs iteration. The step-size $\tau = 0.5$ was calibrated manually to obtain acceptance rates in the 30–70% range.
- The first 20% of iterates are discarded. All credible intervals are computed from the remaining post-burn-in draws.

---

### Results

With $n = 10\,000$, $L = 10$, $\mu_0 = 0$, $\sigma_0^2 = 0.09$, 3 000 iterations and $K = 10$ sub-steps, the sampler recovers:


| Parameter  | Truth | Posterior median | 95% confidence Interval |
| ---------- | ----- | ---------------- | ----------------------- |
| $\mu$      | 0.00  | ≈ 0.000         | tight, covers truth     |
| $\sigma^2$ | 0.09  | ≈ 0.090         | tight, covers truth     |

Unlike ABC, there is no approximation $\varepsilon$: the sampler targets the exact posterior, without any bias. The confidence intervals center directly on the true parameter values.

![Trace plots](exact_sampler_plots/traces.png)
*Figure 1: Trace Plots.*

Trace plots show rapid mixing for $\mu$ and good mixing for $\sigma^2$ once the chain leaves the initialization. The post-burn-in traces are stationary with no visible trend.

![Posterior Hist](exact_sampler_plots/posterior_hist.png)
*Figure 2: Posterior of the distribution.*

Posterior histograms are concentrated around the truth (much tighter than the ABC posteriors at any value of $\varepsilon$), and without the bias on $\sigma^2$ that ABC was suffering of due to prior misspecification and finite tolerance.

**Computational cost.**

Each Gibbs iteration runs in fully vectorized way over all $n$ observations. The dominant cost is the $nK$ pairwise MH proposals per iteration. However, in practice, this is fast enough for $n = 10\,000$ in a few minutes on CPU.

**Sensitivity to $\tau$.**

The step-size $\tau$ controls the mixing of Block 2. If $\tau$ is too small, it gives high acceptance but the exploration is slower. Also, a too large $\tau$ gives low acceptance and poor mixing. The value $\tau = 0.5$ was chosen to balance these effects and can be tuned using pilot runs if needed.

---

### Verifying that the Gibbs Sampler is unbiased

*The code for this sub-section can be found in `bias_test.py`.*

In this sub part we take a look at the results of multiple runs of the Gibbs Sampler across different Datasets.
The methodology is as follows:

```
For 100 iterations:
  Simulate a Dataset Y
  Run the Gibbs Sampler
    Save the posteriors on μ and σ²
 
Compile the medians for each observation
Observe the distribution of medians thus made
```

![median_distrib](mcmc_outputs/mcmc_100runs_sigma2=0.09.png)

*Figure 3: Distribution of the best estimates across N=100 Gibbs Sampler runs across different datasets θ = (0, 0.09)*

This check over 100 Datasets seems to confirm that the Gibbs Sampler is unbiased with an error on the scale of 0.0001 for both components of θ.
Ideally we would like to run this process for a larger amount of datasets to see the average of the medians converging but it entices large computational requirements.

We ran the same process with different values for the parameters (**μ**,**σ**) and saw similar results, wit.
Here is the result for higher variance (here **σ**² = 0.3 > 0.09)

![gibbs_bias](mcmc_outputs/mcmc_100runs_diagnostics.png)

*Figure 4: Distribution of the best estimates across N=100 Gibbs Sampler runs using high-variance data*

---

## Question 4 — Comparing the methods

Until now, we have always known the "true value" of the estimators, it is hence natural to use them to define the Bias introduiced by the ABC methods. Yet, this project aims at conducting inference on the parameters that generated the dataset Y. Which means that our ABC and exact implementations should be used to infere unknown parameters.
This is the reason why, we will proceed as if the true parameters were unknown to us.

*The code for this section can be found in `estimating_epsilon_bias.py` and `plots_epsilon_bias.py`.*

This part aims at estimating the bias introduced by **ε** in the ABC methods. In order to do so we will consider the exact sampler from Question 3 as the ground truth or "best estimator" and we will focus on the empirical average of the posteriors to asses the bias induced by **ε**. Note that we will compare the Gibbs sampler to the MCMC-ABC algorithm derived in question 2 as the best representant of our ABC class.

The methodology is as follows:

```
<estimating_espilon_bias.py>
Generate 10 datasets Y
For each Y:
  Run the Gibbs sampler and save the posterior on θ
  For values of ε in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2, 5]:
    Run the MCMC-ABC algorithm and save the posterior on θ

Save all posteriors and aggreagated metrics to csv files

<plots_epsilon_bias.py>
Load the saved data and generate informative plots on the run
```

Here are the results we obtained from this study

![Posterior_shifting](epsilon_comparison/figures/fig2_posterior_shift.png)

*Figure 1: Posterior shift for different **ε** values on one dataset*

The first result we notice is, similarly to part 1, that an increase in epsilon induces a flatter posterior (for both **μ** and **σ²**) and a directional shift of the center of the distribution away from the ground truth. Note that we observed that the posterior shifts to the left (negatives) for **μ** and to the right for **σ**² regardless of the dataset.
Here is another example, showing a similar shift:

![Posterior_shifting_other example](epsilon_comparison/figures/fig2_posterior_shift_ds_9.png)

*Figure 2: Another example of posterior shift for different **ε** values on another dataset*

The reason behind the fact that the direction of the shift is always the same is quite intuitive:
First of all, **μ** and **σ** are coupled because  $ \mathbb{E}[Y\_i] = L \exp(\mu + \frac{\sigma^2}{2})$ and the sampler needs to fit data.
Second, the posterior we get depends on the prior we chose. Since we chose **$\log(\sigma^2) \sim \mathcal{N}(0, t^2)$** with t close to 1 and we ran simulations with **σ**²= 0.09 << 1, the posterior on **σ** is dragged upwards (and **μ** downwards, to compensate).

Thus the sign of the Bias is dependant on the prior we chose with respect to the "real" parameters.

Now let's focus on the scale of the Bias introduced by **ε**:

![Posterior_shifting_other example](epsilon_comparison/figures/fig1_bias_vs_epsilon.png)
![Posterior_shifting_other example](epsilon_comparison/figures/fig4_bias_acceptance_tradeoff.png)
*Figure 3&4: Growth of  $ \text{Bias} = \left| \mathbb{E}_{\text{ABC}}[\theta] - \mathbb{E}_{\text{Gibbs}}[\theta] \right| $ with respect to **ε***

The results highlight a direct and continuous relationship between the tolerance **$\varepsilon$** and the estimation error. When **$\varepsilon$** is very small, the bias is almost zero, reflecting strong fidelity to the target distribution. However, as the tolerance widens, the mean absolute bias increases systematically. This degradation in precision does not affect the parameters equally: the error on the variance (**$\sigma^2$**) grows much more steeply and severely than that on the mean (**$\mu$**). In summary, relaxing the acceptance criterion **$\varepsilon$** inevitably deteriorates the inference, with a particularly marked penalty on the estimation of the data's dispersion.

Furthermore, the growth of the bias is not strictly linear, but rather sub-linear (showing a logarithmic-like concavity). Interestingly, the trajectory of the bias curves mirrors the growth curve of the acceptance rate. This similarity highlights that the bias is linked to the acceptance probability: as the algorithm accepts a progressively larger fraction of the prior space, the estimation error grows following the same decelerating rate.

![Posterior_shifting_other example](epsilon_comparison/figures/fig3_bias_per_dataset.png)
*Figure 5: Individual Bias for all Datasets*

Last but not least, the dispersion of the bias across the datasets Y appears to be very small with respect to the value of the bias. This result indicates that our estimation of the bias is reliable and that there is no need to verify with more data for this set
