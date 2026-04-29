# Project — Likelihood-free inference for sums of log-normal variates
 
## Setup
 
We observe $n$ data points $Y_1, \dots, Y_n$, where each $Y_i$ is a sum of $L$ log-normal variates:
 
$$Y_i = \sum_{\ell=1}^{L} \exp(X_{i,\ell}), \qquad X_{i,\ell} \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2).$$
 
The goal is to estimate $\theta = (\mu, \sigma^2)$. The density of $Y_i$ has no closed form, which rules out standard likelihood-based methods. Throughout this project we use synthetic data generated from the true parameters $L = 10$, $\mu_0 = 0$, $\sigma_0 = 0.3$ (i.e. $\sigma_0^2 = 0.09$), so we always know the actual truth.
 
---
 
## Question 1 — Reject-ABC
 
*The code for this section and more detailed insights about the results can be found in `Reject_ABC.ipynb`.*
 
### Why ABC
 
Because the density of $Y_i$ is intractable, we cannot evaluate the likelihood directly. What we can do is simulate data for any proposed $\theta$. Reject-ABC exploits exactly this: propose a parameter, simulate fake data, and keep the proposal only if the fake data looks close enough to what we observed.
 
### Distance
 
We use the 1-Wasserstein distance $W_1$, which in one dimension simply means sorting both samples and taking the mean absolute gap:
 
$$W_1(y^{\text{sim}}, y^\star) = \frac{1}{n}\sum_{i=1}^{n}\left|y^{\text{sim}}_{(i)} - y^\star_{(i)}\right|.$$
 
This is the distance used in Bernton et al. (2017), the paper the project is based on. It compares the full empirical distributions rather than a few hand-picked summary statistics, which matters here because no single statistic clearly captures all the information.
 
### Prior
 
Following the project statement we use:
 
$$\mu \sim \mathcal{N}(0, s^2), \qquad \log(\sigma^2) \sim \mathcal{N}(0, t^2).$$
 
The log-normal reparametrisation keeps $\sigma^2$ positive automatically.
 
![Observed data distribution](Reject_ABC_plots/observed_data_dist.png)
*Figure 1: Observed data distribution.*

### The algorithm
 
```
while number of accepted draws < N:
    1. draw θ* = (μ*, σ²*) from the prior
    2. simulate Y*_1, ..., Y*_n under θ*
    3. compute W1(Y*, y_obs)
    4. if W1 ≤ ε: keep θ*
```
 
The accepted draws are i.i.d. samples from the ABC posterior.
 
### Calibrating ε
 
Before running, we calibrated $\varepsilon$ by looking at the prior predictive distribution of distances: draw many $\theta$ values from the prior, simulate data under each, compute $W_1$, and use a low quantile of those distances as $\varepsilon$.
 
We chose the **1% quantile** ($\varepsilon \approx 0.98$), which corresponds to an acceptance rate of about 1% — roughly 100 proposals per accepted draw. Tighter tolerances improve the approximation but cost much more.
 
| Quantile | $\varepsilon$ | Acceptance rate | Proposals per accept |
|----------|--------------|-----------------|----------------------|
| 20%      | ~4.4         | ~20%            | ~5                   |
| 10%      | ~2.8         | ~10%            | ~10                  |
| 1%       | ~0.98        | ~1%             | ~100                 |
| 0.5%     | ~0.72        | ~0.5%           | ~200                 |
 
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
|:----------:|:---------------:|
| 0.3        | 3,39%           |
| 1.0        | 1,00%           |
| 3.0        | 0.35%           | 


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

Samples from the ABC posterior, when used to simulate new datasets, produce data that is centered roughly right but visibly more spread out than $y^\star$. The observed standard deviation is about 1.01 while the mean replicated one is about 1.59. This overdispersion confirms that the posterior is still placing too much mass on large $\sigma^2$ values.


---
 
## Question 2 — MCMC-ABC

SENGHAK

---
 
## Question 3 — Introducing latent variables

THEO

---
 
## Question 4 — Comparing the methods

GABRIEL