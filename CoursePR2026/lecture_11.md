# Pattern Recognition and Machine Learning
## Chapter 11: Sampling Methods and Monte Carlo Inference

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 11 Sampling Methods (§11.1-§11.6)  
> Teaching emphasis: Monte Carlo estimation, importance sampling, Markov chains, stationary distributions, Metropolis-Hastings, Gibbs sampling, burn-in, correlated samples, and mixing

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Monte Carlo Estimation: Replace a Hard Integral by a Sample Average](#1-monte-carlo-estimation-replace-a-hard-integral-by-a-sample-average)
3. [§2 Basic Sampling Intuition and Rejection Sampling](#2-basic-sampling-intuition-and-rejection-sampling)
4. [§3 Importance Sampling](#3-importance-sampling)
5. [§4 Markov Chains and Stationary Distributions](#4-markov-chains-and-stationary-distributions)
6. [§5 Metropolis-Hastings](#5-metropolis-hastings)
7. [§6 Gibbs Sampling](#6-gibbs-sampling)
8. [§7 Burn-in, Correlated Samples, Effective Sample Size, and Mixing](#7-burn-in-correlated-samples-effective-sample-size-and-mixing)
9. [§8 Slice Sampling and Hamiltonian Monte Carlo: Brief Modern Perspective](#8-slice-sampling-and-hamiltonian-monte-carlo-brief-modern-perspective)
10. [§9 Worked Textbook Exercises and Classroom Examples](#9-worked-textbook-exercises-and-classroom-examples)
11. [§10 Chapter Summary, Figure Checklist, and Teaching Flow](#10-chapter-summary-figure-checklist-and-teaching-flow)

---

## Notation and Variable Definitions

Chapter 10 introduced approximate inference methods that replace an intractable distribution by a simpler analytical approximation. Chapter 11 studies a different strategy:

> Instead of forcing the posterior into a simple formula, generate representative samples and use those samples to approximate the quantities we need.

The most important notation is summarized below.

| Symbol | Definition |
|--------|------------|
| $\mathbf{z}$ | A random vector whose distribution is of interest. It may contain latent variables, model parameters, or both. |
| $z^{(l)}$ | The $l$th sample. The superscript labels a sample and is not a power. |
| $L$ | Number of generated samples. |
| $p(\mathbf{z})$ | Target distribution from which we would ideally like to draw samples. |
| $\widetilde p(\mathbf{z})$ | Unnormalized target density, with $p(\mathbf{z})=\widetilde p(\mathbf{z})/Z_p$. |
| $Z_p$ | Normalizing constant or partition function. It may be unknown. |
| $f(\mathbf{z})$ | A quantity whose expectation under $p$ is required. |
| $\mathbb{E}_p[f]$ | Expectation of $f(\mathbf{z})$ when $\mathbf{z}\sim p(\mathbf{z})$. |
| $\widehat{\mathbb{E}}_p[f]$ | Monte Carlo estimate of the expectation. |
| $q(\mathbf{z})$ | Proposal, sampling, or importance distribution that is easier to sample from. |
| $r_l$ | Unnormalized importance weight for sample $l$. |
| $w_l$ | Normalized importance weight, satisfying $\sum_l w_l=1$. |
| $\mathbf{z}^{(\tau)}$ | State of a Markov chain at iteration $\tau$. |
| $T(\mathbf{z},\mathbf{z}')$ | Markov transition kernel from state $\mathbf{z}$ to state $\mathbf{z}'$. |
| $p_*(\mathbf{z})$ | Stationary or invariant distribution of a Markov chain. |
| $A(\mathbf{z}^*,\mathbf{z})$ | Metropolis-Hastings acceptance probability for candidate $\mathbf{z}^*$ from current state $\mathbf{z}$. |
| $B$ | Number of initial iterations discarded as burn-in. |
| $\rho_k$ | Autocorrelation between samples separated by lag $k$. |
| $L_{\mathrm{eff}}$ | Effective sample size: the number of independent samples carrying roughly the same information. |

> **Teaching focus.** Students do not need to memorize every sampling algorithm in the chapter. They should understand one central pipeline:
>
> $$
> \text{hard integral}
> \longrightarrow
> \text{samples}
> \longrightarrow
> \text{sample average}
> \longrightarrow
> \text{diagnose whether the samples are trustworthy}.
> $$

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch. 11 opening; §11.1-§11.6

## 0.1 Why Sampling Appears in Machine Learning

Many probabilistic machine-learning tasks require an expectation such as

$$
\mathbb{E}_{p(\mathbf{z})}[f(\mathbf{z})]
=
\int f(\mathbf{z})p(\mathbf{z})\,d\mathbf{z}.
$$

Examples include:

- a posterior mean of a parameter;
- the predictive probability of a class;
- the expected reconstruction of a latent-variable model;
- the expected loss under posterior uncertainty;
- an average over latent variables in a Bayesian model.

The difficulty is that the integral may be impossible to evaluate exactly. The dimension of $\mathbf{z}$ may be large, the posterior may be non-Gaussian, and the normalizing constant may be unknown.

Sampling gives a simple alternative:

1. Generate plausible values of $\mathbf{z}$.
2. Evaluate $f(\mathbf{z})$ at those values.
3. Average the results.

This chapter is therefore less about obtaining one “best” latent state and more about representing an entire uncertain distribution through samples.

## 0.2 The Three Levels of Difficulty

It is helpful to separate three situations.

| Situation | What We Can Do | Typical Method |
|-----------|----------------|----------------|
| We can sample directly from $p(\mathbf{z})$. | Draw independent samples and average. | Basic Monte Carlo |
| We cannot sample from $p$, but can sample from a related $q$. | Correct the mismatch using weights. | Importance sampling |
| We cannot obtain independent samples from $p$. | Construct a chain whose long-run distribution is $p$. | MCMC: Metropolis-Hastings or Gibbs |

The progression is:

$$
\text{independent samples}
\quad\rightarrow\quad
\text{weighted samples}
\quad\rightarrow\quad
\text{correlated Markov-chain samples}.
$$

## 0.3 What We Will Not Emphasize

The full textbook chapter contains several technically important topics. For this EE-oriented course, the following are reduced or omitted:

- detailed construction of adaptive rejection samplers;
- full slice-sampling implementation details;
- Hamiltonian dynamics and leapfrog derivations;
- partition-function estimation;
- formal proofs of detailed balance.

These topics are valuable in advanced Bayesian computation, but they are not necessary for understanding the sampling methods most relevant to modern machine learning.

## 0.4 Learning Objectives

After this chapter, students should be able to:

1. explain why a sample average can approximate a difficult expectation;
2. distinguish independent sampling, importance sampling, and MCMC;
3. compute simple importance weights;
4. explain the Markov property and stationary distribution;
5. execute a few iterations of Metropolis-Hastings by hand;
6. explain why a rejected Metropolis-Hastings proposal still produces a repeated sample;
7. execute coordinate-wise Gibbs updates from full conditional distributions;
8. explain burn-in, autocorrelation, effective sample size, and mixing;
9. identify common failure modes such as weight degeneracy and a chain trapped in one mode;
10. recognize HMC as a modern method for efficient Bayesian inference without deriving its dynamics.

---

# §1 Monte Carlo Estimation: Replace a Hard Integral by a Sample Average

> 📖 Textbook Ch. 11 opening, pp. 523-525; Textbook Fig. 11.1; Exercise 11.1

## 1.1 The Expectation We Want

Suppose the target distribution is $p(\mathbf{z})$ and we want the expectation of a function $f$:

$$
\mathbb{E}_p[f]
=
\int f(\mathbf{z})p(\mathbf{z})\,d\mathbf{z}.
$$

For a discrete variable, the integral becomes a sum:

$$
\mathbb{E}_p[f]
=
\sum_{\mathbf{z}} f(\mathbf{z})p(\mathbf{z}).
$$

> ![Figure 11.1](./CoursePR2026/Fig/Chapter_11/lecture_fig_11_1__textbook_fig_11_1__p524.png)
>
> *Figure 11.1 (Textbook Fig. 11.1, p. 524): The expectation depends on both the probability density $p(z)$ and the function $f(z)$. Regions contribute strongly when the product $p(z)f(z)$ is large.*

The figure makes an important point. Sampling only needs to spend substantial effort in regions that matter to the integral. A region where $p(z)$ is almost zero contributes very little, even if $f(z)$ is large.

## 1.2 Monte Carlo Estimator

Assume we can generate independent samples

$$
\mathbf{z}^{(1)},\mathbf{z}^{(2)},\ldots,\mathbf{z}^{(L)}
\sim p(\mathbf{z}).
$$

The Monte Carlo estimator is

$$
\widehat{\mathbb{E}}_p[f]
=
\frac{1}{L}\sum_{l=1}^{L}f\left(\mathbf{z}^{(l)}\right).
$$

This formula should be read in three steps:

1. sample $\mathbf{z}^{(l)}$ from the target distribution;
2. compute $f(\mathbf{z}^{(l)})$;
3. average the computed values.

> **Main intuition.** A probability distribution tells us how frequently different regions should appear. Therefore, the ordinary average over correctly distributed samples already performs the probability weighting for us.

## 1.3 A Very Simple Example

Let $z$ be a Bernoulli variable with

$$
p(z=1)=0.7,
\qquad
p(z=0)=0.3.
$$

Suppose we want

$$
\mathbb{E}[z].
$$

Analytically,

$$
\mathbb{E}[z]
=1\cdot 0.7+0\cdot 0.3
=0.7.
$$

Now suppose ten samples are

$$
1,1,0,1,1,1,0,1,0,1.
$$

The Monte Carlo estimate is

$$
\widehat{\mathbb{E}}[z]
=
\frac{1+1+0+1+1+1+0+1+0+1}{10}
=
0.7.
$$

With another set of ten samples we might obtain $0.6$ or $0.8$. The estimator is random, but as the sample size grows, it tends to stabilize near the true expectation.

## 1.4 Why the Estimator Has the Correct Mean

Let

$$
\widehat f
=
\frac{1}{L}\sum_{l=1}^{L}f(\mathbf{z}^{(l)}).
$$

Taking the expectation of the estimator gives

$$
\mathbb{E}[\widehat f]
=
\mathbb{E}\left[
\frac{1}{L}\sum_{l=1}^{L}f(\mathbf{z}^{(l)})
\right].
$$

Expectation is linear, so

$$
\mathbb{E}[\widehat f]
=
\frac{1}{L}\sum_{l=1}^{L}
\mathbb{E}\left[f(\mathbf{z}^{(l)})\right].
$$

Every sample has the same distribution $p$, so every term equals $\mathbb{E}_p[f]$:

$$
\mathbb{E}[\widehat f]
=
\frac{1}{L}\cdot L\mathbb{E}_p[f]
=
\mathbb{E}_p[f].
$$

Therefore, the basic Monte Carlo estimator is **unbiased** when the samples are drawn from the correct distribution.

## 1.5 Why More Samples Help

If the samples are independent, then

$$
\operatorname{var}[\widehat f]
=
\frac{1}{L}\operatorname{var}_p[f].
$$

The standard deviation of the estimator is therefore

$$
\operatorname{sd}[\widehat f]
=
\frac{\operatorname{sd}_p[f]}{\sqrt{L}}.
$$

This gives the famous Monte Carlo rate:

$$
\text{typical error}
\propto
\frac{1}{\sqrt{L}}.
$$

Doubling the number of samples does not halve the error. To reduce the typical error by a factor of two, we need about four times as many independent samples.

| Desired Error Reduction | Approximate Increase in Samples |
|-------------------------|---------------------------------|
| Error divided by $2$ | $4\times$ samples |
| Error divided by $3$ | $9\times$ samples |
| Error divided by $10$ | $100\times$ samples |

## 1.6 Why Monte Carlo Is Attractive in High Dimensions

A regular grid becomes exponentially expensive with dimension. If each coordinate uses only ten grid points, then a $D$-dimensional grid contains

$$
10^D
$$

points. For $D=20$, this is already $10^{20}$ points.

Monte Carlo estimation does not require constructing such a grid. Its idealized $1/\sqrt{L}$ convergence rate does not explicitly become worse simply because $D$ increases.

This does **not** mean high-dimensional sampling is easy. The difficulty moves into another question:

> Can we generate representative samples from the high-dimensional target distribution?

Importance sampling and MCMC address precisely this question.

## 1.7 Independent Samples versus Correlated Samples

The variance formula above assumes independent samples. MCMC samples are usually correlated. If a chain moves only a tiny distance at each step, then many consecutive samples may contain almost the same information.

For example, the following sequence contains ten numbers:

$$
1.00,1.01,1.02,1.01,1.03,1.04,1.03,1.05,1.04,1.06.
$$

Numerically there are ten samples, but they behave more like a few independent observations because each value is strongly tied to the previous one.

This motivates three later ideas:

- autocorrelation;
- effective sample size;
- mixing.

---

# §2 Basic Sampling Intuition and Rejection Sampling

> 📖 Textbook §11.1.1-§11.1.3; Textbook Fig. 11.4

## 2.1 Direct Sampling from Standard Distributions

Some distributions are easy to sample from because standard software provides reliable generators. Common examples include:

- uniform distributions;
- Gaussian distributions;
- Bernoulli and categorical distributions;
- gamma distributions;
- beta distributions.

More complicated distributions may sometimes be generated by transforming samples from a simple distribution. For example, a multivariate Gaussian can be constructed from independent standard Gaussian variables using a matrix factorization.

For this course, the important point is not the transformation algebra. It is the distinction:

- **evaluating** a density at a point;
- **drawing** a random sample from that density.

A model may make the first operation easy while the second operation remains difficult.

## 2.2 Rejection Sampling: Draw, Test, Keep or Reject

Suppose the target density is known only up to a constant:

$$
p(z)
=
\frac{\widetilde p(z)}{Z_p}.
$$

We choose a proposal density $q(z)$ that is easy to sample from and a constant $k$ such that

$$
\widetilde p(z)
\leq
kq(z)
$$

for every $z$.

> ![Figure 11.2](./CoursePR2026/Fig/Chapter_11/lecture_fig_11_2__textbook_fig_11_4__p529.png)
>
> *Figure 11.2 (Textbook Fig. 11.4, p. 529): Rejection sampling draws from the easy proposal $q(z)$ and retains a point with probability proportional to the target-to-envelope ratio.*

A single rejection-sampling step is:

1. Draw $z^*\sim q(z)$.
2. Draw $u\sim\operatorname{Uniform}(0,1)$.
3. Accept $z^*$ if

$$
u
\leq
\frac{\widetilde p(z^*)}{kq(z^*)}.
$$

Otherwise reject it and try again.

## 2.3 Why the Normalizing Constant Is Not Needed

The test uses $\widetilde p(z)$ rather than the normalized density $p(z)$. This is useful because Bayesian posteriors often have the form

$$
p(\mathbf{z}\mid \mathcal{D})
\propto
p(\mathcal{D}\mid \mathbf{z})p(\mathbf{z}),
$$

where the proportional expression is easy to evaluate but the evidence is not.

## 2.4 Efficiency Depends on the Envelope

The proposal envelope should fit the target closely.

- If $kq(z)$ lies far above $\widetilde p(z)$, most proposals are rejected.
- If $q(z)$ has lighter tails than the target, a finite valid $k$ may not exist.
- In high dimensions, even a small mismatch can cause an extremely low acceptance rate.

Therefore rejection sampling is useful mainly for simple low-dimensional subproblems. It is not the central high-dimensional method of this chapter.

## 2.5 Rejection Sampling versus Metropolis-Hastings

Students often confuse these two algorithms.

| Rejection Sampling | Metropolis-Hastings |
|--------------------|---------------------|
| Each proposal is independent of the previous accepted sample. | Proposal usually depends on the current state. |
| A rejected proposal produces no output sample. | A rejected proposal repeats the current state as the next chain sample. |
| Requires a global envelope $kq(z)\geq \widetilde p(z)$. | Does not require a global envelope. |
| Accepted samples are independent under the standard construction. | Consecutive samples are correlated. |

---

# §3 Importance Sampling

> 📖 Textbook §11.1.4-§11.1.5, pp. 532-536; Textbook Fig. 11.8

## 3.1 The Main Problem

Suppose we cannot easily sample from $p(\mathbf{z})$, but we can:

- evaluate $p(\mathbf{z})$ or an unnormalized version of it;
- sample from another distribution $q(\mathbf{z})$.

Importance sampling rewrites the expectation so that samples from $q$ can be used.

> ![Figure 11.3](./CoursePR2026/Fig/Chapter_11/lecture_fig_11_3__textbook_fig_11_8__p532.png)
>
> *Figure 11.3 (Textbook Fig. 11.8, p. 532): Samples are drawn from $q(z)$ rather than $p(z)$. The ratio $p(z)/q(z)$ corrects the mismatch.*

## 3.2 Step-by-Step Derivation

Start with

$$
\mathbb{E}_p[f]
=
\int f(\mathbf{z})p(\mathbf{z})\,d\mathbf{z}.
$$

Multiply and divide by $q(\mathbf{z})$:

$$
\mathbb{E}_p[f]
=
\int
f(\mathbf{z})
\frac{p(\mathbf{z})}{q(\mathbf{z})}
q(\mathbf{z})
\,d\mathbf{z}.
$$

This is now an expectation under $q$:

$$
\mathbb{E}_p[f]
=
\mathbb{E}_{q}
\left[
\frac{p(\mathbf{z})}{q(\mathbf{z})}f(\mathbf{z})
\right].
$$

If

$$
\mathbf{z}^{(l)}\sim q(\mathbf{z}),
$$

then

$$
\mathbb{E}_p[f]
\approx
\frac{1}{L}
\sum_{l=1}^{L}
\frac{p(\mathbf{z}^{(l)})}{q(\mathbf{z}^{(l)})}
 f(\mathbf{z}^{(l)}).
$$

Define the importance ratio

$$
r_l
=
\frac{p(\mathbf{z}^{(l)})}{q(\mathbf{z}^{(l)})}.
$$

Then

$$
\widehat{\mathbb{E}}_p[f]
=
\frac{1}{L}\sum_{l=1}^{L}r_l f(\mathbf{z}^{(l)}).
$$

## 3.3 Intuition for the Weight

The weight answers:

> Did the proposal generate this point too often or too rarely compared with the target?

- If $p(\mathbf{z})>q(\mathbf{z})$, then the point is underrepresented by $q$, so its weight is larger than one.
- If $p(\mathbf{z})<q(\mathbf{z})$, then the point is overrepresented by $q$, so its weight is smaller than one.

The proposal decides **where** samples are generated. The weights decide **how much influence** each sample has.

## 3.4 Self-Normalized Importance Sampling

Often we know only

$$
p(\mathbf{z})
=
\frac{\widetilde p(\mathbf{z})}{Z_p},
$$

where $Z_p$ is unknown. Define unnormalized weights

$$
\widetilde r_l
=
\frac{\widetilde p(\mathbf{z}^{(l)})}{q(\mathbf{z}^{(l)})}.
$$

Normalize them:

$$
w_l
=
\frac{\widetilde r_l}{\sum_{m=1}^{L}\widetilde r_m}.
$$

Then

$$
\sum_{l=1}^{L}w_l=1,
$$

and the self-normalized estimator is

$$
\widehat{\mathbb{E}}_p[f]
=
\sum_{l=1}^{L}w_l f(\mathbf{z}^{(l)}).
$$

This is one of the most useful forms in practice because the unknown normalizing constant cancels.

## 3.5 A Small Discrete Example

Let the target distribution over $z\in\{1,2,3\}$ be

$$
p(z)
=
(0.1,0.2,0.7),
$$

while the easy proposal is

$$
q(z)
=
(0.3,0.3,0.4).
$$

Suppose the proposal produces five samples:

$$
3,1,3,2,3.
$$

The importance ratios are

$$
\frac{p(1)}{q(1)}=\frac{0.1}{0.3}=0.333,
$$

$$
\frac{p(2)}{q(2)}=\frac{0.2}{0.3}=0.667,
$$

$$
\frac{p(3)}{q(3)}=\frac{0.7}{0.4}=1.75.
$$

For the five sampled states, the unnormalized weights are therefore

$$
1.75,\ 0.333,\ 1.75,\ 0.667,\ 1.75.
$$

Their sum is approximately

$$
6.25.
$$

The normalized weights are approximately

$$
0.280,\ 0.053,\ 0.280,\ 0.107,\ 0.280.
$$

To estimate $\mathbb{E}_p[z]$, compute

$$
\widehat{\mathbb{E}}_p[z]
=
0.280(3)+0.053(1)+0.280(3)+0.107(2)+0.280(3).
$$

Thus

$$
\widehat{\mathbb{E}}_p[z]
\approx
2.79.
$$

The exact value is

$$
\mathbb{E}_p[z]
=1(0.1)+2(0.2)+3(0.7)
=2.6.
$$

Five samples are too few for a highly accurate result, but the example shows the mechanics of weighting.

## 3.6 Weight Degeneracy

Importance sampling can fail when a small number of samples receive almost all the weight.

For example, suppose normalized weights are

$$
0.001,0.001,0.002,0.006,0.990.
$$

Although there are five stored samples, the estimate is controlled almost entirely by one sample. This is called **weight degeneracy**.

A useful diagnostic is the importance-sampling effective sample size

$$
L_{\mathrm{eff}}
\approx
\frac{1}{\sum_{l=1}^{L}w_l^2}.
$$

Two extremes are useful:

- equal weights $w_l=1/L$ give $L_{\mathrm{eff}}=L$;
- one weight equal to one gives $L_{\mathrm{eff}}=1$.

## 3.7 The Most Dangerous Failure Mode

The proposal must cover every important region of the target:

$$
q(\mathbf{z})>0
\quad\text{whenever}\quad
p(\mathbf{z})f(\mathbf{z})\neq 0.
$$

If $q$ assigns almost no probability to an important target region, no sample may reach that region. The resulting estimate may look numerically stable while being badly wrong.

This is more dangerous than an obviously noisy estimate because the failure may not be visible from the observed weights.

## 3.8 Sampling-Importance-Resampling

Sampling-importance-resampling uses the weighted sample set to produce an approximately unweighted sample set:

1. Draw $L$ proposals from $q$.
2. Compute normalized weights $w_l$.
3. Resample from the stored points with probabilities $w_l$.

High-weight samples may appear multiple times; low-weight samples may disappear.

This basic idea later appears in particle filters and sequential Monte Carlo.

## 3.9 When Importance Sampling Is Appropriate

Importance sampling is most attractive when:

- a good proposal is available;
- the target and proposal overlap strongly;
- the effective dimension is not too large;
- we need expectations rather than a long chain of dependent states.

It becomes fragile when the target is high-dimensional, multimodal, or much sharper than the proposal.

---

# §4 Markov Chains and Stationary Distributions

> 📖 Textbook §11.2-§11.2.1, pp. 537-541

## 4.1 Why We Need a Markov Chain

Importance sampling still assumes that we can find one global proposal distribution $q(\mathbf{z})$ that covers the target well. This becomes difficult in high dimensions.

MCMC takes a different approach:

> Do not try to generate a perfect independent sample in one step. Start somewhere and make a sequence of local moves whose long-run visiting frequency matches the target distribution.

The generated sequence is

$$
\mathbf{z}^{(0)},
\mathbf{z}^{(1)},
\mathbf{z}^{(2)},
\ldots
$$

Successive states are usually dependent.

## 4.2 The Markov Property

A first-order Markov chain satisfies

$$
p\left(
\mathbf{z}^{(\tau+1)}
\mid
\mathbf{z}^{(0)},\ldots,\mathbf{z}^{(\tau)}
\right)
=
p\left(
\mathbf{z}^{(\tau+1)}
\mid
\mathbf{z}^{(\tau)}
\right).
$$

In words:

> Once the current state is known, the next state does not need the complete earlier history.

This does not mean the past is irrelevant in an absolute sense. The past influenced the current state. The Markov property says that the current state is a sufficient summary for predicting the next transition.

## 4.3 Transition Kernel

The transition rule can be written as

$$
T(\mathbf{z},\mathbf{z}')
=
p\left(
\mathbf{z}^{(\tau+1)}=\mathbf{z}'
\mid
\mathbf{z}^{(\tau)}=\mathbf{z}
\right).
$$

For discrete states,

$$
\sum_{\mathbf{z}'}T(\mathbf{z},\mathbf{z}')=1.
$$

For continuous states, the sum becomes an integral.

The transition kernel tells us how the chain moves. It is not necessarily equal to the target distribution.

## 4.4 Stationary Distribution

A distribution $p_*(\mathbf{z})$ is stationary or invariant for the chain if applying one transition leaves it unchanged:

$$
p_*(\mathbf{z}')
=
\int
p_*(\mathbf{z})
T(\mathbf{z},\mathbf{z}')
\,d\mathbf{z}.
$$

For discrete states,

$$
p_*(\mathbf{z}')
=
\sum_{\mathbf{z}}
p_*(\mathbf{z})T(\mathbf{z},\mathbf{z}').
$$

The intuitive meaning is:

> If the current state is already distributed according to $p_*$, one more Markov transition does not change that distribution.

In MCMC, we design $T$ so that the desired target $p$ is stationary.

## 4.5 A Two-State Example

Consider two states, $A$ and $B$, with transition matrix

$$
\mathbf{T}
=
\begin{bmatrix}
0.8 & 0.2\\
0.3 & 0.7
\end{bmatrix}.
$$

Rows represent the current state and columns represent the next state.

- From $A$, the chain remains in $A$ with probability $0.8$ and moves to $B$ with probability $0.2$.
- From $B$, the chain moves to $A$ with probability $0.3$ and remains in $B$ with probability $0.7$.

Let the stationary distribution be

$$
\boldsymbol{\pi}
=(\pi_A,\pi_B).
$$

Stationarity requires

$$
\boldsymbol{\pi}
=
\boldsymbol{\pi}\mathbf{T}.
$$

For state $A$,

$$
\pi_A
=
0.8\pi_A+0.3\pi_B.
$$

Because

$$
\pi_B=1-\pi_A,
$$

we obtain

$$
\pi_A
=
0.8\pi_A+0.3(1-\pi_A).
$$

Expanding,

$$
\pi_A
=
0.8\pi_A+0.3-0.3\pi_A
=
0.5\pi_A+0.3.
$$

Therefore,

$$
0.5\pi_A=0.3,
$$

so

$$
\pi_A=0.6,
\qquad
\pi_B=0.4.
$$

If the chain runs for a long time under suitable conditions, approximately $60\%$ of its states will be $A$ and $40\%$ will be $B$.

## 4.6 Stationarity Is Not Enough

A transition kernel can have the target as a stationary distribution and still fail to explore it correctly from an arbitrary starting point.

A simple failure is the identity transition:

$$
T(\mathbf{z},\mathbf{z}')
=
\begin{cases}
1,&\mathbf{z}'=\mathbf{z},\\
0,&\text{otherwise}.
\end{cases}
$$

Every distribution is stationary under this transition, but the chain never moves.

Therefore we also need the chain to be able to explore the relevant state space.

## 4.7 Ergodicity: The Practical Intuition

For this course, an ergodic chain can be understood as one that:

1. can eventually reach every relevant region of the target space;
2. does not get trapped in a fixed deterministic cycle;
3. has a unique long-run equilibrium distribution.

Under appropriate conditions, the effect of the initial state disappears and

$$
p\left(\mathbf{z}^{(\tau)}\right)
\longrightarrow
p_*(\mathbf{z})
$$

as $\tau\to\infty$.

## 4.8 Detailed Balance: State the Idea, Skip the Proof

A commonly used sufficient condition for stationarity is detailed balance:

$$
p_*(\mathbf{z})T(\mathbf{z},\mathbf{z}')
=
p_*(\mathbf{z}')T(\mathbf{z}',\mathbf{z}).
$$

This compares probability flow in the two directions between two states.

> At equilibrium, the flow from $\mathbf{z}$ to $\mathbf{z}'$ is balanced by the reverse flow.

Detailed balance is sufficient but not necessary. We will use it as design intuition for Metropolis-Hastings, but we will not prove it in detail.

## 4.9 MCMC Samples Are Not Independent

The chain remembers its current state. Therefore, even after reaching stationarity,

$$
\mathbf{z}^{(\tau)}
\quad\text{and}\quad
\mathbf{z}^{(\tau+1)}
$$

are usually correlated.

Nevertheless, under appropriate conditions, the sample average

$$
\frac{1}{L}
\sum_{l=1}^{L}
f\left(\mathbf{z}^{(B+l)}\right)
$$

can still converge to the desired expectation. The price is that correlated samples are less informative than the same number of independent samples.

---

# §5 Metropolis-Hastings

> 📖 Textbook §11.2 and §11.2.2, pp. 537-542; Textbook Figs. 11.9-11.10

## 5.1 Core Idea

Metropolis-Hastings constructs a Markov chain by repeatedly:

1. proposing a candidate near or related to the current state;
2. deciding whether to accept it;
3. moving to the candidate if accepted;
4. staying at the current state if rejected.

The acceptance rule corrects for both the target density and the proposal mechanism.

## 5.2 Proposal Distribution

At iteration $\tau$, the current state is $\mathbf{z}^{(\tau)}$. Draw a candidate

$$
\mathbf{z}^*
\sim
q\left(\mathbf{z}^*\mid\mathbf{z}^{(\tau)}\right).
$$

A common random-walk proposal is

$$
\mathbf{z}^*
=
\mathbf{z}^{(\tau)}+\boldsymbol{\epsilon},
$$

where

$$
\boldsymbol{\epsilon}
\sim
\mathcal{N}(\mathbf{0},\rho^2\mathbf{I}).
$$

The parameter $\rho$ controls the proposal step size.

## 5.3 Acceptance Probability

The Metropolis-Hastings acceptance probability is

$$
A(\mathbf{z}^*,\mathbf{z})
=
\min\left(
1,
\frac{
\widetilde p(\mathbf{z}^*)
q(\mathbf{z}\mid\mathbf{z}^*)
}{
\widetilde p(\mathbf{z})
q(\mathbf{z}^*\mid\mathbf{z})
}
\right).
$$

Here $\mathbf{z}$ denotes the current state.

The ratio contains two corrections:

| Factor | Role |
|--------|------|
| $\widetilde p(\mathbf{z}^*)/\widetilde p(\mathbf{z})$ | Favors candidates with higher target density. |
| $q(\mathbf{z}\mid\mathbf{z}^*)/q(\mathbf{z}^*\mid\mathbf{z})$ | Corrects asymmetry in the proposal. |

## 5.4 Why the Unknown Normalizing Constant Cancels

Suppose

$$
p(\mathbf{z})
=
\frac{\widetilde p(\mathbf{z})}{Z_p}.
$$

Then the target ratio is

$$
\frac{p(\mathbf{z}^*)}{p(\mathbf{z})}
=
\frac{\widetilde p(\mathbf{z}^*)/Z_p}{\widetilde p(\mathbf{z})/Z_p}
=
\frac{\widetilde p(\mathbf{z}^*)}{\widetilde p(\mathbf{z})}.
$$

This cancellation is a major reason MCMC is useful for Bayesian posteriors.

## 5.5 Symmetric Proposal: The Metropolis Algorithm

If

$$
q(\mathbf{z}^*\mid\mathbf{z})
=
q(\mathbf{z}\mid\mathbf{z}^*),
$$

then the proposal terms cancel and

$$
A(\mathbf{z}^*,\mathbf{z})
=
\min\left(
1,
\frac{\widetilde p(\mathbf{z}^*)}{\widetilde p(\mathbf{z})}
\right).
$$

This gives a simple interpretation:

- moving to a higher-density point is always accepted;
- moving to a lower-density point may still be accepted;
- accepting some downhill moves prevents the chain from becoming a greedy optimizer.

> **Sampling is not optimization.** An optimizer tries to climb to a mode and stay there. A sampler must spend time across the whole distribution in proportion to probability mass.

## 5.6 Accept or Stay

Draw

$$
u\sim\operatorname{Uniform}(0,1).
$$

If

$$
u<A(\mathbf{z}^*,\mathbf{z}^{(\tau)}),
$$

accept:

$$
\mathbf{z}^{(\tau+1)}=\mathbf{z}^*.
$$

Otherwise reject:

$$
\mathbf{z}^{(\tau+1)}=\mathbf{z}^{(\tau)}.
$$

The repeated state after rejection is part of the Markov chain and must be included in the sample sequence.

## 5.7 Metropolis-Hastings Pseudocode

```text
Input: unnormalized target density p_tilde(z), proposal q(z_new | z_old)
Choose initial state z(0)

for tau = 0, 1, ..., T-1:
    draw candidate z_star ~ q(z_star | z(tau))

    compute
        R = [p_tilde(z_star) q(z(tau) | z_star)]
            / [p_tilde(z(tau)) q(z_star | z(tau))]

    A = min(1, R)
    draw u ~ Uniform(0, 1)

    if u < A:
        z(tau+1) = z_star
    else:
        z(tau+1) = z(tau)
```

## 5.8 Hand Calculation with a Gaussian Target

Let the unnormalized target be

$$
\widetilde p(z)
=
\exp\left(-\frac{z^2}{2}\right).
$$

Use a symmetric proposal, so the acceptance ratio is only the target ratio.

### Candidate Moving Toward the Center

Current state:

$$
z=1.
$$

Candidate:

$$
z^*=0.5.
$$

The ratio is

$$
\frac{\widetilde p(0.5)}{\widetilde p(1)}
=
\frac{\exp(-0.5^2/2)}{\exp(-1^2/2)}.
$$

Combine the exponents:

$$
=
\exp\left(-\frac{0.25}{2}+\frac{1}{2}\right)
=
\exp(0.375)
>1.
$$

Therefore,

$$
A=1.
$$

The candidate is always accepted.

### Candidate Moving into the Tail

Current state:

$$
z=1.
$$

Candidate:

$$
z^*=2.
$$

Then

$$
\frac{\widetilde p(2)}{\widetilde p(1)}
=
\exp\left(-\frac{4}{2}+\frac{1}{2}\right)
=
\exp(-1.5)
\approx
0.223.
$$

Thus

$$
A\approx0.223.
$$

The chain accepts this lower-density move with probability about $22.3\%$.

## 5.9 A Geometric View of Accepted and Rejected Proposals

> ![Figure 11.4](./CoursePR2026/Fig/Chapter_11/lecture_fig_11_4__textbook_fig_11_9__p539.png)
>
> *Figure 11.4 (Textbook Fig. 11.9, p. 539): The chain uses local Gaussian proposals to explore a correlated Gaussian target. Accepted moves are shown in green; rejected proposals are shown in red.*

The figure illustrates several points:

- the chain explores by local moves rather than independent global draws;
- some candidates point outside high-density regions and are rejected;
- the path is jagged because the proposal is random;
- consecutive accepted states remain correlated.

## 5.10 Proposal Scale and the Acceptance-Exploration Trade-off

The proposal scale $\rho$ is crucial.

### Proposal Too Small

- high acceptance rate;
- tiny movement per iteration;
- strong correlation between nearby samples;
- slow exploration.

### Proposal Too Large

- candidates often land in very low-density regions;
- low acceptance rate;
- many repeated states;
- slow exploration.

### A Useful Intermediate Scale

- proposals move a meaningful distance;
- enough candidates remain plausible;
- the chain explores without excessive rejection.

> ![Figure 11.5](./CoursePR2026/Fig/Chapter_11/lecture_fig_11_5__textbook_fig_11_10__p542.png)
>
> *Figure 11.5 (Textbook Fig. 11.10, p. 542): An isotropic proposal must use a small step size to fit the narrow direction of an elongated target. It then moves very slowly along the long direction.*

## 5.11 Correlated and Anisotropic Targets

A spherical proposal may be poorly matched to an elongated posterior. In Figure 11.5, the target has:

- a narrow direction with scale $\sigma_{\min}$;
- a long direction with scale $\sigma_{\max}$.

To avoid rejection in the narrow direction, the proposal scale must be small. But that same small scale makes progress along the long direction very slow.

This is a geometric explanation of poor mixing.

Modern approaches try to adapt to posterior geometry through:

- preconditioning;
- covariance-aware proposals;
- reparameterization;
- block updates;
- Hamiltonian Monte Carlo.

## 5.12 Log-Density Computation

In practice, probabilities can be extremely small. It is safer to compute the log acceptance ratio:

$$
\log R
=
\log\widetilde p(\mathbf{z}^*)
-
\log\widetilde p(\mathbf{z})
+
\log q(\mathbf{z}\mid\mathbf{z}^*)
-
\log q(\mathbf{z}^*\mid\mathbf{z}).
$$

Then accept when

$$
\log u
<
\min(0,\log R).
$$

This avoids numerical underflow and is the standard implementation pattern.

## 5.13 What the Acceptance Rate Does and Does Not Tell Us

Acceptance rate is useful but incomplete.

- A rate near $100\%$ may mean the steps are far too small.
- A rate near $0\%$ means the chain barely moves.
- A moderate rate does not guarantee movement between separated modes.

Therefore acceptance rate must be interpreted together with trace plots, autocorrelation, effective sample size, and multiple chains.

---

# §6 Gibbs Sampling

> 📖 Textbook §11.3, pp. 542-546; Textbook Figs. 11.11-11.12

## 6.1 Main Idea

Suppose

$$
\mathbf{z}
=(z_1,z_2,\ldots,z_M)
$$

has joint distribution

$$
p(z_1,z_2,\ldots,z_M).
$$

Direct sampling from the joint distribution may be difficult, but sampling from each full conditional may be easy:

$$
p(z_i\mid\mathbf{z}_{\setminus i}).
$$

Here $\mathbf{z}_{\setminus i}$ denotes all components except $z_i$.

Gibbs sampling repeatedly updates one variable at a time from its conditional distribution given the current values of all other variables.

## 6.2 Three-Variable Update

Suppose the current state is

$$
(z_1^{(\tau)},z_2^{(\tau)},z_3^{(\tau)}).
$$

A full sweep performs:

$$
z_1^{(\tau+1)}
\sim
p\left(z_1\mid z_2^{(\tau)},z_3^{(\tau)}\right),
$$

then immediately uses the new value of $z_1$:

$$
z_2^{(\tau+1)}
\sim
p\left(z_2\mid z_1^{(\tau+1)},z_3^{(\tau)}\right),
$$

and finally

$$
z_3^{(\tau+1)}
\sim
p\left(z_3\mid z_1^{(\tau+1)},z_2^{(\tau+1)}\right).
$$

The updated value is used as soon as it becomes available.

## 6.3 Gibbs Sampling Pseudocode

```text
Choose initial values z1(0), ..., zM(0)

for tau = 0, 1, ..., T-1:
    sample z1(tau+1) from p(z1 | z2(tau), ..., zM(tau))
    sample z2(tau+1) from p(z2 | z1(tau+1), z3(tau), ..., zM(tau))
    ...
    sample zM(tau+1) from p(zM | z1(tau+1), ..., zM-1(tau+1))
```

One complete update of all variables is called a **sweep**.

## 6.4 Gibbs as a Special Case of Metropolis-Hastings

For a coordinate $z_i$, Gibbs chooses the proposal

$$
q(z_i^*\mid\mathbf{z})
=
p(z_i^*\mid\mathbf{z}_{\setminus i}).
$$

With this choice, the Metropolis-Hastings ratio simplifies to one. Therefore every Gibbs proposal is accepted.

> **Important clarification.** “Always accepted” does not mean “always efficient.” Gibbs can still move very slowly when variables are strongly correlated.

## 6.5 A Small Binary Example

Consider two binary variables $x,y\in\{0,1\}$ with joint distribution

| $p(x,y)$ | $y=0$ | $y=1$ |
|----------|-------|-------|
| $x=0$ | $0.4$ | $0.1$ |
| $x=1$ | $0.2$ | $0.3$ |

The probabilities sum to one.

Suppose the current state is

$$
(x,y)=(0,0).
$$

### Update $x$ Given $y=0$

The conditional probability of $x=1$ is

$$
p(x=1\mid y=0)
=
\frac{p(x=1,y=0)}{p(y=0)}.
$$

The denominator is

$$
p(y=0)
=
0.4+0.2
=
0.6.
$$

Therefore,

$$
p(x=1\mid y=0)
=
\frac{0.2}{0.6}
=
\frac{1}{3}.
$$

Similarly,

$$
p(x=0\mid y=0)
=
\frac{2}{3}.
$$

Suppose the sampled update gives $x=1$.

### Update $y$ Given the New $x=1$

Now

$$
p(y=1\mid x=1)
=
\frac{p(x=1,y=1)}{p(x=1)}.
$$

The denominator is

$$
p(x=1)
=
0.2+0.3
=
0.5.
$$

Thus

$$
p(y=1\mid x=1)
=
\frac{0.3}{0.5}
=
0.6.
$$

The updated pair is sampled using two easy one-dimensional conditional distributions rather than the joint distribution directly.

## 6.6 Correlation Can Cause Slow Movement

> ![Figure 11.6](./CoursePR2026/Fig/Chapter_11/lecture_fig_11_6__textbook_fig_11_11__p545.png)
>
> *Figure 11.6 (Textbook Fig. 11.11, p. 545): Coordinate-wise Gibbs updates move horizontally and vertically. For a strongly correlated elongated target, the chain advances only slowly along the long axis.*

When two variables are strongly correlated, their full conditional distributions may be narrow. Updating one coordinate while holding the other fixed gives only a small move.

The chain then follows a staircase-like path and needs many iterations to move across the posterior.

This is the Gibbs version of the same geometric problem seen in random-walk Metropolis.

## 6.7 Blocking

Instead of updating one scalar variable at a time, blocked Gibbs sampling updates a group:

$$
\mathbf{z}_{B}
\sim
p(\mathbf{z}_{B}\mid\mathbf{z}_{\setminus B}).
$$

Blocking can reduce correlation when strongly related variables are updated together. The challenge is that the block conditional may be harder to sample from.

## 6.8 Gibbs Sampling in Graphical Models

In a graphical model, the full conditional for one node depends only on its Markov blanket.

> ![Figure 11.7](./CoursePR2026/Fig/Chapter_11/lecture_fig_11_7__textbook_fig_11_12__p546.png)
>
> *Figure 11.7 (Textbook Fig. 11.12, p. 546): For an undirected graph, a node's Markov blanket is its neighbours. For a directed graph, it consists of the node's parents, children, and co-parents.*

This is computationally useful. Although the complete model may contain many variables, a single Gibbs update may require only local information.

For a directed graphical model,

$$
p(z_i\mid\mathbf{z}_{\setminus i})
\propto
p(z_i\mid\operatorname{pa}(z_i))
\prod_{c\in\operatorname{ch}(z_i)}
p(c\mid\operatorname{pa}(c)).
$$

Only factors involving $z_i$ need to be recomputed.

## 6.9 When Gibbs Sampling Is Attractive

Gibbs sampling is especially useful when:

- every full conditional has a standard form;
- conjugate priors produce easy conditionals;
- the graphical model gives sparse local dependencies;
- variables can be grouped into useful blocks.

It is less attractive when:

- the full conditionals are themselves difficult;
- variables are strongly correlated;
- the posterior has separated modes;
- coordinate updates cannot move between disconnected regions.

---

# §7 Burn-in, Correlated Samples, Effective Sample Size, and Mixing

> 📖 Textbook §11.2-§11.3; modern diagnostic interpretation

## 7.1 Why the Initial State Matters

An MCMC chain must start somewhere:

$$
\mathbf{z}^{(0)}.
$$

This initial value is usually not drawn from the target distribution. Early states therefore reflect both:

- the target distribution;
- the arbitrary initial condition.

As the chain runs, the influence of the initial state should decrease.

## 7.2 Burn-in

The first $B$ iterations are often discarded:

$$
\mathbf{z}^{(0)},\ldots,\mathbf{z}^{(B-1)}.
$$

The retained sequence is

$$
\mathbf{z}^{(B)},\mathbf{z}^{(B+1)},\ldots.
$$

This discarded prefix is called **burn-in** or **warm-up**.

> Burn-in is not a magical fixed number. It should be long enough for the chain to reach the typical region of the target from its starting point.

A chain can still be wrong after burn-in if it is trapped in one mode or mixes very slowly.

## 7.3 Trace Plots

A trace plot displays a sampled quantity against iteration number.

A reasonably mixed one-dimensional trace often looks like a noisy horizontal band:

- it revisits the same range repeatedly;
- it has no persistent upward or downward trend;
- it does not stay frozen for very long periods;
- different chains overlap after warm-up.

Warning signs include:

| Trace Pattern | Possible Problem |
|---------------|------------------|
| Slow drift | Burn-in may be insufficient. |
| Long flat sections | Very low acceptance or numerical failure. |
| Long periods in separate levels | Multimodality or poor movement between modes. |
| Different chains occupy different regions | Chains have not mixed to a common distribution. |
| Very smooth local movement | Strong autocorrelation and small effective sample size. |

## 7.4 Autocorrelation

For a scalar chain quantity $g^{(\tau)}=g(\mathbf{z}^{(\tau)})$, lag-$k$ autocorrelation measures similarity between samples $k$ iterations apart:

$$
\rho_k
=
\operatorname{corr}
\left(g^{(\tau)},g^{(\tau+k)}\right).
$$

Typical behavior is:

- $\rho_0=1$;
- $\rho_k$ decreases as $k$ increases;
- slow decay means the chain forgets its past slowly;
- fast decay means samples become nearly independent after fewer steps.

## 7.5 Effective Sample Size for MCMC

For independent samples, $L$ samples carry information roughly proportional to $L$. For correlated samples, the effective number is smaller.

A common approximation is

$$
L_{\mathrm{eff}}
\approx
\frac{L}{1+2\sum_{k=1}^{\infty}\rho_k}.
$$

The denominator is called the integrated autocorrelation time.

### Interpretation

Suppose a chain stores

$$
L=10,000
$$

samples, but the autocorrelation structure gives

$$
L_{\mathrm{eff}}=500.
$$

The Monte Carlo accuracy is closer to what we would expect from 500 independent samples, not 10,000.

## 7.6 Monte Carlo Standard Error

If the posterior standard deviation of $f$ is approximately $s_f$, then a rough Monte Carlo standard error is

$$
\operatorname{MCSE}
\approx
\frac{s_f}{\sqrt{L_{\mathrm{eff}}}}.
$$

This separates two kinds of uncertainty:

- **posterior uncertainty:** genuine uncertainty represented by the target distribution;
- **Monte Carlo uncertainty:** numerical error because only finitely many correlated samples were generated.

Running the chain longer reduces Monte Carlo uncertainty, but it does not remove genuine posterior uncertainty.

## 7.7 Mixing

Mixing describes how quickly a chain explores the target and loses memory of its starting point.

A well-mixing chain:

- moves across the typical set efficiently;
- visits important modes in appropriate proportions;
- has quickly decaying autocorrelation;
- gives a large effective sample size per unit computation.

A poorly mixing chain:

- moves through the state space like a slow random walk;
- remains trapped in a local region;
- produces highly correlated samples;
- may give misleading estimates even after many iterations.

## 7.8 Random-Walk Behavior

In a simple random walk, the typical distance travelled after $\tau$ steps grows like

$$
\sqrt{\tau},
$$

not like $\tau$.

This is why local random-walk proposals can be inefficient. To move ten times farther, the chain may require roughly one hundred times as many steps.

The problem is visible in:

- random-walk Metropolis with a small proposal scale;
- Gibbs sampling for highly correlated variables;
- elongated posteriors with poorly chosen coordinates.

## 7.9 Multiple Chains

A practical strategy is to run several chains from dispersed initial values.

Suppose four chains begin in different regions. After warm-up, we hope that:

- their trace plots overlap;
- their means and variances are similar;
- all chains visit the same modes;
- between-chain and within-chain variation are comparable.

A modern convergence summary often called $\widehat R$ compares between-chain and within-chain variation. Values close to one are desirable, but no single diagnostic proves convergence.

## 7.10 Thinning

Thinning keeps every $M$th sample and discards the rest.

For example, with $M=10$:

$$
\mathbf{z}^{(B)},
\mathbf{z}^{(B+10)},
\mathbf{z}^{(B+20)},
\ldots
$$

Thinning can reduce storage and make plots easier to inspect. However, it usually does not create more information from a fixed computational budget. The discarded intermediate samples may still contribute useful information to an expectation estimate.

> Modern default: retain all post-warm-up samples unless storage or downstream computation is a genuine limitation.

## 7.11 Mixing Is Quantity-Specific

A chain may mix well for one function and poorly for another.

For example:

- a posterior mean may stabilize quickly;
- a tail probability may need far more samples;
- a mode indicator may mix badly if mode switching is rare.

Therefore diagnostics should be examined for the actual quantities used in the final scientific conclusion.

## 7.12 Practical Checklist Before Trusting MCMC Output

Before reporting an MCMC estimate, check:

1. Did several dispersed chains reach the same region?
2. Are trace plots stable after warm-up?
3. Does autocorrelation decay reasonably fast?
4. Is the effective sample size adequate for the reported quantity?
5. Are acceptance rates and step sizes sensible?
6. Did every important posterior mode receive visits?
7. Is the result stable when the run length is increased?
8. Is the Monte Carlo standard error small enough?

---

# §8 Slice Sampling and Hamiltonian Monte Carlo: Brief Modern Perspective

> 📖 Textbook §11.4-§11.5; introduction only

## 8.1 Slice Sampling: The One-Sentence Idea

Slice sampling introduces an auxiliary height variable and alternates between:

1. selecting a vertical level under the unnormalized density curve;
2. sampling a new state from the horizontal slice above that level.

The method can adapt its effective step size to the local width of the distribution. This reduces the need to choose one fixed random-walk proposal scale.

For this course, students only need to recognize the basic idea. We will not implement the bracketing and shrinkage details.

## 8.2 Why HMC Was Developed

Random-walk Metropolis proposes blind local perturbations. In high dimensions, most large blind moves land in low-density regions and are rejected.

Hamiltonian Monte Carlo, also called hybrid Monte Carlo in the textbook, uses gradient information to propose distant states that remain in high-probability regions.

It augments the state $\mathbf{z}$ with an auxiliary momentum $\mathbf{r}$ and defines an energy

$$
H(\mathbf{z},\mathbf{r})
=
U(\mathbf{z})+K(\mathbf{r}),
$$

where

$$
U(\mathbf{z})
=-\log\widetilde p(\mathbf{z})
$$

is potential energy and $K(\mathbf{r})$ is kinetic energy.

## 8.3 HMC Intuition without Dynamics

A useful analogy is a ball moving across a smooth landscape:

- the negative log density defines the landscape;
- gradients indicate the slope;
- momentum carries the state through the space;
- numerical simulation proposes a distant point;
- a Metropolis correction removes discretization bias.

The important contrast is:

| Random-Walk Metropolis | HMC |
|------------------------|-----|
| Proposes a blind local step. | Uses gradients to follow the geometry. |
| Often diffuses slowly. | Can make long, directed moves. |
| Simple but sensitive to scale. | More computation per iteration, often much better mixing. |

## 8.4 Modern Bayesian Inference Context

HMC and its adaptive variants are widely used for continuous Bayesian models. They are especially useful when:

- gradients of the log posterior are available;
- the parameter space is continuous;
- the posterior is moderately smooth;
- random-walk methods mix too slowly.

Students are not required to derive Hamilton's equations or the leapfrog integrator in this course. The learning goal is to understand why gradient-guided proposals can outperform random walks.

## 8.5 Methods Deliberately Omitted

The following topics are outside the required material:

- adaptive rejection sampling construction;
- detailed slice-sampling bracket updates;
- Hamiltonian dynamics proofs;
- volume-preservation proofs;
- partition-function estimation;
- annealed chains for normalization constants.

---

# §9 Worked Textbook Exercises and Classroom Examples

> 📖 Textbook Exercises 11.1, 11.10, and 11.12, pp. 556-557

## 9.1 Textbook Exercise 11.1: Mean and Variance of the Monte Carlo Estimator

Let

$$
\widehat f
=
\frac{1}{L}
\sum_{l=1}^{L}
f\left(z^{(l)}\right),
$$

where the samples are independent and distributed according to $p(z)$.

### Part A: Show the Mean Is Correct

Using linearity of expectation,

$$
\mathbb{E}[\widehat f]
=
\frac{1}{L}
\sum_{l=1}^{L}
\mathbb{E}\left[f(z^{(l)})\right].
$$

Each term is the same:

$$
\mathbb{E}\left[f(z^{(l)})\right]
=
\mathbb{E}_p[f].
$$

Therefore,

$$
\mathbb{E}[\widehat f]
=
\frac{1}{L}
L\mathbb{E}_p[f]
=
\mathbb{E}_p[f].
$$

### Part B: Show the Variance Decreases as $1/L$

Because the samples are independent,

$$
\operatorname{var}\left[
\sum_{l=1}^{L}f(z^{(l)})
\right]
=
\sum_{l=1}^{L}
\operatorname{var}\left[f(z^{(l)})\right].
$$

Thus

$$
\operatorname{var}[\widehat f]
=
\operatorname{var}\left[
\frac{1}{L}
\sum_{l=1}^{L}f(z^{(l)})
\right].
$$

Pulling out the constant gives

$$
\operatorname{var}[\widehat f]
=
\frac{1}{L^2}
\sum_{l=1}^{L}
\operatorname{var}_p[f].
$$

There are $L$ identical terms, so

$$
\operatorname{var}[\widehat f]
=
\frac{1}{L^2}
L\operatorname{var}_p[f]
=
\frac{1}{L}
\operatorname{var}_p[f].
$$

> **Classroom message.** The standard error decreases as $1/\sqrt{L}$, not $1/L$.

## 9.2 Textbook Exercise 11.10: Why a Random Walk Moves Slowly

Consider increments $\Delta^{(\tau)}$ with

$$
\Delta^{(\tau)}
=
\begin{cases}
+1,&\text{probability }0.25,\\
0,&\text{probability }0.50,\\
-1,&\text{probability }0.25.
\end{cases}
$$

The walk evolves as

$$
z^{(\tau)}
=
z^{(\tau-1)}+\Delta^{(\tau)}.
$$

The increment has mean

$$
\mathbb{E}[\Delta]
=1(0.25)+0(0.50)-1(0.25)
=0.
$$

Its squared value has expectation

$$
\mathbb{E}[\Delta^2]
=1^2(0.25)+0^2(0.50)+(-1)^2(0.25)
=0.5.
$$

Now square the update:

$$
(z^{(\tau)})^2
=
(z^{(\tau-1)}+\Delta^{(\tau)})^2.
$$

Expanding,

$$
(z^{(\tau)})^2
=
(z^{(\tau-1)})^2
+2z^{(\tau-1)}\Delta^{(\tau)}
+(\Delta^{(\tau)})^2.
$$

Taking expectations, the middle term is zero because the new increment has zero mean and is independent of the previous position. Therefore,

$$
\mathbb{E}[(z^{(\tau)})^2]
=
\mathbb{E}[(z^{(\tau-1)})^2]
+0.5.
$$

Starting from $z^{(0)}=0$ gives

$$
\mathbb{E}[(z^{(\tau)})^2]
=
\frac{\tau}{2}.
$$

The root-mean-square distance is

$$
\sqrt{\mathbb{E}[(z^{(\tau)})^2]}
=
\sqrt{\frac{\tau}{2}}.
$$

Thus the typical distance grows only with $\sqrt{\tau}$.

## 9.3 Textbook Exercise 11.12: A Gibbs Chain That Cannot Cross a Gap

> ![Figure 11.8](./CoursePR2026/Fig/Chapter_11/lecture_fig_11_8__textbook_fig_11_15__p557.png)
>
> *Figure 11.8 (Textbook Fig. 11.15, p. 557): The target is uniform on two disconnected regions and zero elsewhere.*

Suppose the chain starts inside the lower-left region.

Standard two-variable Gibbs sampling alternates:

$$
z_1\sim p(z_1\mid z_2),
$$

and

$$
z_2\sim p(z_2\mid z_1).
$$

A coordinate update changes only one variable at a time. To reach the upper-right region, the chain would need to pass through intermediate horizontal or vertical positions where the joint probability is zero.

The conditional distributions assign zero probability to those moves. Therefore the chain remains trapped in the region where it started.

The target distribution is stationary, but the chain is not ergodic over the full support. It does not sample the correct relative probability of the two regions.

> **Main lesson.** Correct conditional updates are not enough. The resulting chain must also be able to move between all relevant parts of the target support.

## 9.4 Classroom Example: One Metropolis-Hastings Decision

Let the current state be $z=0$, and suppose an asymmetric proposal gives:

$$
q(z^*=1\mid z=0)=0.8,
$$

$$
q(z=0\mid z^*=1)=0.4.
$$

Let the unnormalized target values be

$$
\widetilde p(0)=0.2,
\qquad
\widetilde p(1)=0.6.
$$

The MH ratio is

$$
R
=
\frac{0.6\times0.4}{0.2\times0.8}
=
\frac{0.24}{0.16}
=1.5.
$$

Therefore,

$$
A=\min(1,1.5)=1.
$$

The proposal is always accepted.

Without the proposal correction, the target ratio alone would be $3$. The correction is necessary because the forward proposal is easier than the reverse proposal.

## 9.5 Classroom Example: Effective Sample Size of Importance Weights

Suppose four normalized importance weights are

$$
0.25,0.25,0.25,0.25.
$$

Then

$$
L_{\mathrm{eff}}
=
\frac{1}{4(0.25)^2}
=
4.
$$

Now suppose the weights are

$$
0.97,0.01,0.01,0.01.
$$

Then

$$
L_{\mathrm{eff}}
=
\frac{1}{0.97^2+3(0.01)^2}.
$$

Numerically,

$$
L_{\mathrm{eff}}
\approx
1.06.
$$

Although four samples are stored, they carry approximately the information of only one equally weighted sample.

## 9.6 Concept Check Questions

1. Why is the Monte Carlo estimator an ordinary average rather than a probability-weighted average when samples come directly from $p$?
2. Why can an importance sampler have many samples but an effective sample size near one?
3. Why must a rejected MH proposal repeat the current state?
4. Why is a $100\%$ MH acceptance rate not automatically good?
5. What is the difference between stationarity and good mixing?
6. Why can Gibbs sampling be slow even though every proposal is accepted?
7. What problem does burn-in address, and what problems does it not address?
8. Why does thinning usually not increase information per unit computation?

---

# §10 Chapter Summary, Figure Checklist, and Teaching Flow

## 10.1 The Central Story

The chapter can be summarized in one sequence:

$$
\boxed{
\text{Expectation}
\rightarrow
\text{Samples}
\rightarrow
\text{Sample average}
\rightarrow
\text{diagnose sample quality}
}
$$

When direct sampling is easy,

$$
\mathbb{E}_p[f]
\approx
\frac{1}{L}
\sum_{l=1}^{L}f(z^{(l)}).
$$

When samples come from another distribution,

$$
\mathbb{E}_p[f]
\approx
\sum_{l=1}^{L}w_l f(z^{(l)}).
$$

When independent sampling is impossible, construct a Markov chain with target stationary distribution $p$.

## 10.2 Formula Sheet

### Monte Carlo

$$
\widehat{\mathbb{E}}_p[f]
=
\frac{1}{L}
\sum_{l=1}^{L}f(z^{(l)}).
$$

$$
\operatorname{var}[\widehat f]
=
\frac{\operatorname{var}_p[f]}{L}
\quad\text{for independent samples}.
$$

### Importance Sampling

$$
r_l
=
\frac{p(z^{(l)})}{q(z^{(l)})}.
$$

$$
w_l
=
\frac{\widetilde p(z^{(l)})/q(z^{(l)})}
{\sum_m\widetilde p(z^{(m)})/q(z^{(m)})}.
$$

$$
\widehat{\mathbb{E}}_p[f]
=
\sum_{l=1}^{L}w_l f(z^{(l)}).
$$

$$
L_{\mathrm{eff}}
\approx
\frac{1}{\sum_l w_l^2}.
$$

### Markov Chain

$$
p(z^{(\tau+1)}\mid z^{(0)},\ldots,z^{(\tau)})
=
p(z^{(\tau+1)}\mid z^{(\tau)}).
$$

$$
p_*(z')
=
\int p_*(z)T(z,z')\,dz.
$$

### Metropolis-Hastings

$$
A(z^*,z)
=
\min\left(
1,
\frac{
\widetilde p(z^*)q(z\mid z^*)
}{
\widetilde p(z)q(z^*\mid z)
}
\right).
$$

### Gibbs Sampling

$$
z_i^{\mathrm{new}}
\sim
p(z_i\mid\mathbf{z}_{\setminus i}).
$$

### MCMC Effective Sample Size

$$
L_{\mathrm{eff}}
\approx
\frac{L}{1+2\sum_{k\geq1}\rho_k}.
$$

## 10.3 Comparison of Main Methods

| Method | Samples Independent? | Needs Normalized Target? | Main Strength | Main Weakness |
|--------|----------------------|--------------------------|---------------|---------------|
| Direct Monte Carlo | Yes | No, if a direct sampler exists | Simple and reliable | Direct sampler may not exist |
| Rejection sampling | Yes | No | Exact accepted samples | Poor acceptance in high dimensions |
| Importance sampling | Yes from $q$, weighted for $p$ | No | Direct expectation estimates | Weight degeneracy |
| Metropolis-Hastings | No | No | General MCMC framework | Proposal tuning and slow mixing |
| Gibbs sampling | No | No | Easy when full conditionals are standard | Slow under strong correlation |
| HMC | No | No | Efficient long moves in continuous spaces | Needs gradients and more machinery |

## 10.4 Common Student Confusions

| Confusion | Clarification |
|-----------|---------------|
| “Sampling is the same as optimization.” | No. Optimization finds high-density points; sampling must represent the whole probability mass. |
| “A rejected MH proposal should be deleted.” | No. The current state is repeated and counts as the next sample. |
| “High acceptance means a good chain.” | Not necessarily. Tiny proposals can give high acceptance and terrible mixing. |
| “Stationary means the chain has converged.” | Stationarity is a property of the transition rule and distribution; a finite chain may not yet have reached it. |
| “Burn-in fixes all MCMC problems.” | Burn-in only reduces initialization bias. It does not fix trapping or poor mixing. |
| “Ten thousand MCMC samples equal ten thousand independent samples.” | Correlation can make the effective sample size much smaller. |
| “Gibbs is efficient because acceptance is always one.” | Acceptance one does not prevent slow coordinate-wise movement. |
| “Thinning creates independent samples.” | It may reduce correlation between stored points, but usually discards useful information. |
| “Importance weights correct any proposal.” | Only if the proposal covers all important target regions and the weight variance is manageable. |

## 10.5 Figure Checklist

| Lecture Figure | Textbook Source | File |
|----------------|-----------------|------|
| Figure 11.1 | Textbook Fig. 11.1, p. 524 | `lecture_fig_11_1__textbook_fig_11_1__p524.png` |
| Figure 11.2 | Textbook Fig. 11.4, p. 529 | `lecture_fig_11_2__textbook_fig_11_4__p529.png` |
| Figure 11.3 | Textbook Fig. 11.8, p. 532 | `lecture_fig_11_3__textbook_fig_11_8__p532.png` |
| Figure 11.4 | Textbook Fig. 11.9, p. 539 | `lecture_fig_11_4__textbook_fig_11_9__p539.png` |
| Figure 11.5 | Textbook Fig. 11.10, p. 542 | `lecture_fig_11_5__textbook_fig_11_10__p542.png` |
| Figure 11.6 | Textbook Fig. 11.11, p. 545 | `lecture_fig_11_6__textbook_fig_11_11__p545.png` |
| Figure 11.7 | Textbook Fig. 11.12, p. 546 | `lecture_fig_11_7__textbook_fig_11_12__p546.png` |
| Figure 11.8 | Textbook Fig. 11.15, p. 557 | `lecture_fig_11_8__textbook_fig_11_15__p557.png` |

All displayed figures are cropped directly from the textbook PDF.

## 10.6 Suggested Teaching Flow

A practical delivery plan is two 75-minute sessions.

### Session A: From Integrals to MCMC

| Time | Topic |
|------|-------|
| 0-15 min | Why expectations are hard; Monte Carlo sample average |
| 15-28 min | Unbiasedness, variance, $1/\sqrt{L}$ convergence |
| 28-40 min | Rejection sampling intuition |
| 40-60 min | Importance sampling and normalized weights |
| 60-75 min | Weight degeneracy and effective sample size |

### Session B: Markov-Chain Sampling

| Time | Topic |
|------|-------|
| 0-15 min | Markov property and stationary distribution |
| 15-40 min | Metropolis-Hastings algorithm and hand calculation |
| 40-52 min | Proposal scale and mixing |
| 52-65 min | Gibbs sampling and graphical-model connection |
| 65-75 min | Burn-in, autocorrelation, ESS, and HMC overview |

If only one shorter lecture is available, prioritize:

1. Monte Carlo estimation;
2. importance sampling;
3. stationary distribution intuition;
4. Metropolis-Hastings;
5. Gibbs sampling;
6. burn-in and mixing.

## 10.7 What Students Should Remember One Week Later

Students do not need to remember all derivations. They should remember these six statements:

1. A hard expectation can be approximated by averaging a function over samples.
2. Importance sampling corrects samples from the wrong distribution using weights.
3. MCMC produces correlated samples through a Markov chain.
4. Metropolis-Hastings accepts or rejects proposals so that the target is stationary.
5. Gibbs sampling repeatedly samples from full conditional distributions.
6. A sample set is useful only when burn-in, correlation, effective sample size, and mixing have been checked.

## 10.8 Bridge to Chapter 12

Chapter 11 provides computational tools for distributions that are difficult to integrate analytically. Chapter 12 turns to models with continuous latent variables, including principal component analysis and its probabilistic extensions.

The connection is important:

> Continuous latent-variable models often require expectations or posterior inference. When exact calculations are unavailable, sampling methods provide a general computational fallback.
