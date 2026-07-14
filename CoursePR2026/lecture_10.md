# Pattern Recognition and Machine Learning
## Chapter 10: Approximate Inference — Variational Methods and Expectation Propagation（近似推断）

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 10 Approximate Inference (§10.1–§10.7)

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Why Approximate Inference Is Needed](#1-why-approximate-inference-is-needed)
3. [§2 Variational Inference and the Evidence Lower Bound](#2-variational-inference-and-the-evidence-lower-bound)
4. [§3 Mean-Field Variational Inference](#3-mean-field-variational-inference)
5. [§4 What Mean-Field Approximation Gets Right and Wrong](#4-what-mean-field-approximation-gets-right-and-wrong)
6. [§5 A Worked Variational Example: Unknown Gaussian Mean and Precision](#5-a-worked-variational-example-unknown-gaussian-mean-and-precision)
7. [§6 Variational Gaussian Mixtures](#6-variational-gaussian-mixtures)
8. [§7 Variational Linear Regression and Variational Message Passing](#7-variational-linear-regression-and-variational-message-passing)
9. [§8 Local Variational Bounds and Logistic Regression](#8-local-variational-bounds-and-logistic-regression)
10. [§9 Expectation Propagation](#9-expectation-propagation)
11. [§10 Modern Machine-Learning Connections](#10-modern-machine-learning-connections)
12. [§11 Chapter Summary and Bridge to Sampling Methods](#11-chapter-summary-and-bridge-to-sampling-methods)

---

## Notation and Variable Definitions

Approximate inference contains many distributions with similar-looking symbols. Before studying the algorithms, it is useful to separate three objects:

1. the **joint model** $p(\mathbf{X},\mathbf{Z})$;
2. the **true posterior** $p(\mathbf{Z}\mid\mathbf{X})$;
3. the **approximate posterior** $q(\mathbf{Z})$.

> **Teaching focus.** Keep the following pipeline visible throughout the lecture:
>
> $$
> p(\mathbf{X},\mathbf{Z})
> \longrightarrow
> p(\mathbf{Z}\mid\mathbf{X})\ \text{is intractable}
> \longrightarrow
> q(\mathbf{Z})
> \longrightarrow
> \text{optimize an approximation objective}.
> $$
>
> The central question is not “Can we write Bayes' rule?” Bayes' rule is easy to write. The central question is “Can we actually compute and use the posterior?”

### General Inference Notation

| Symbol | Definition |
|--------|------------|
| $\mathbf{X}$ | Observed data. It may denote one observation, a full data set, or all observed variables in a graphical model. |
| $\mathbf{Z}$ | All unobserved quantities: latent variables, parameters, or both. |
| $p(\mathbf{X},\mathbf{Z})$ | Joint probability model. This is usually easier to evaluate than the posterior. |
| $p(\mathbf{Z}\mid\mathbf{X})$ | Exact posterior distribution after observing $\mathbf{X}$. |
| $p(\mathbf{X})$ | Marginal likelihood or evidence: $p(\mathbf{X})=\int p(\mathbf{X},\mathbf{Z})\,d\mathbf{Z}$. |
| $q(\mathbf{Z})$ | A tractable distribution used to approximate the exact posterior. |
| $\mathcal{Q}$ | The chosen family of allowable approximating distributions. |
| $\mathbb{E}_{q}[f(\mathbf{Z})]$ | Expectation of $f(\mathbf{Z})$ under $q(\mathbf{Z})$. |
| $H[q]$ | Entropy of $q$: $H[q]=-\mathbb{E}_q[\ln q(\mathbf{Z})]$. |

### Variational-Inference Notation

| Symbol | Definition |
|--------|------------|
| $\mathcal{L}(q)$ | Evidence lower bound, usually abbreviated **ELBO**. |
| $\mathrm{KL}(q\Vert p)$ | Kullback–Leibler divergence from $q$ to $p$. |
| $q_i(\mathbf{Z}_i)$ | One factor in a mean-field approximation. |
| $\mathbf{Z}_{-j}$ | All groups of latent variables except $\mathbf{Z}_j$. |
| $\mathbb{E}_{-j}[\cdot]$ | Expectation with respect to all variational factors except $q_j(\mathbf{Z}_j)$. |
| $q_j^*(\mathbf{Z}_j)$ | Optimal update for the $j$th factor while all other factors are fixed. |
| CAVI | Coordinate-Ascent Variational Inference: update one variational factor at a time. |

### Mixture-Model Notation

| Symbol | Definition |
|--------|------------|
| $K$ | Number of candidate mixture components. |
| $z_{nk}$ | Binary indicator that data point $n$ is assigned to component $k$. |
| $r_{nk}=\mathbb{E}[z_{nk}]$ | Variational responsibility of component $k$ for point $n$. |
| $N_k=\sum_n r_{nk}$ | Effective number of data points assigned to component $k$. |
| $\boldsymbol{\pi}$ | Mixture coefficients. |
| $\boldsymbol{\mu}_k$ | Mean of component $k$. |
| $\boldsymbol{\Lambda}_k$ | Precision matrix of component $k$, equal to the inverse covariance. |
| $\boldsymbol{\alpha}$ | Dirichlet parameters controlling the mixture coefficients. |

### Local Bounds and Expectation Propagation

| Symbol | Definition |
|--------|------------|
| $\xi$ or $\xi_n$ | Local variational parameter used to bound a difficult nonlinear term. |
| $\sigma(a)$ | Logistic sigmoid: $\sigma(a)=1/(1+e^{-a})$. |
| $f_i(\boldsymbol{\theta})$ | One exact factor in a target distribution. |
| $\widetilde f_i(\boldsymbol{\theta})$ | A tractable site approximation used by expectation propagation. |
| $q^{\setminus i}(\boldsymbol{\theta})$ | EP cavity distribution obtained by removing site $i$. |
| $\widehat p_i(\boldsymbol{\theta})$ | EP tilted distribution obtained by restoring the exact factor $f_i$. |

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch.10 opening; §10.1–§10.7

## 0.1 What This Chapter Is Really About

Earlier chapters often gave us exact posterior distributions because the model was conjugate or Gaussian. For example, Gaussian likelihood plus Gaussian prior gives a Gaussian posterior. Unfortunately, most useful machine-learning models are not so cooperative.

The posterior is formally

$$
p(\mathbf{Z}\mid\mathbf{X})
=
\frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{X})},
$$

where

$$
p(\mathbf{X})
=
\int p(\mathbf{X},\mathbf{Z})\,d\mathbf{Z}.
$$

The formula is simple, but the integral may be impossible to evaluate exactly. Even if the posterior can be written up to a proportionality constant, the expectations needed for prediction may still be unavailable.

This chapter studies **deterministic approximate inference**. The basic strategy is:

1. choose a tractable family $\mathcal{Q}$;
2. find a distribution $q(\mathbf{Z})\in\mathcal{Q}$ that resembles the exact posterior;
3. use $q$ to approximate posterior expectations and predictions.

The chapter contains two major families of ideas:

| Method | Main Objective | Main Computational Idea |
|--------|----------------|-------------------------|
| **Variational inference（变分推断）** | Minimize $\mathrm{KL}(q\Vert p)$, equivalently maximize the ELBO | Turn inference into optimization over distributions |
| **Expectation propagation（期望传播）** | Locally minimize a reverse-direction KL and match moments | Replace difficult factors one at a time by tractable site approximations |

## 0.2 Why This Topic Matters for Modern Machine Learning

Variational inference is one of the conceptual foundations of modern latent-variable learning. The exact implementations have changed since the textbook was written, but the central principle remains the same:

$$
\text{intractable posterior}
\quad\Longrightarrow\quad
\text{tractable approximate posterior learned by optimization}.
$$

This idea appears in:

- variational autoencoders;
- Bayesian neural networks;
- probabilistic topic models;
- large latent-variable models;
- approximate uncertainty estimation;
- probabilistic matrix factorization;
- scalable Bayesian learning.

For EE students, the same viewpoint is useful in signal reconstruction, channel estimation, source separation, tracking, inverse problems, and probabilistic sensor fusion.

## 0.3 Scope Decisions for This Lecture

Chapter 10 is mathematically dense. To keep the lecture useful and accessible, we will make the following choices.

### We will study carefully

- why the posterior and evidence become intractable;
- the ELBO decomposition and its meaning;
- the mean-field update rule;
- the effect of KL direction;
- a simple Gaussian mean/precision example;
- the main structure of variational Gaussian mixtures;
- the idea of local variational bounds;
- a readable version of variational logistic regression;
- the basic EP cycle and moment matching.

### We will simplify

- Gaussian–Wishart normalizing constants;
- long digamma-function expressions in variational mixtures;
- complete lower-bound bookkeeping;
- the full generic conjugate-exponential derivation;
- EP energy functions and detailed convergence analysis.

### We will treat only as optional enrichment

- the alpha family of divergences;
- the full clutter-model derivation;
- detailed comparisons of floating-point cost;
- every exercise at the end of the textbook chapter.

The goal is to understand **the reusable algorithmic pattern**, not to memorize every special-function term.

## 0.4 Chapter Roadmap

A useful mental map is:

$$
\boxed{\text{Bayesian model}}
\rightarrow
\boxed{\text{intractable posterior}}
\rightarrow
\boxed{\text{ELBO}}
\rightarrow
\boxed{\text{mean-field updates}}
\rightarrow
\boxed{\text{model-specific algorithm}}.
$$

Then we will compare this with EP:

$$
\boxed{\text{factorized target}}
\rightarrow
\boxed{\text{remove one site}}
\rightarrow
\boxed{\text{restore exact factor}}
\rightarrow
\boxed{\text{match moments}}.
$$

---

# §1 Why Approximate Inference Is Needed

> 📖 Textbook Ch.10 opening, pp. 461–462

## 1.1 The Two Integrals That Cause Trouble

Suppose our model contains observed data $\mathbf{X}$ and latent variables $\mathbf{Z}$. Bayesian inference often asks for two quantities.

### Quantity 1: the evidence

$$
p(\mathbf{X})
=
\int p(\mathbf{X},\mathbf{Z})\,d\mathbf{Z}.
$$

The evidence is needed to normalize the posterior and to compare models.

### Quantity 2: a posterior expectation

For a function $f(\mathbf{Z})$,

$$
\mathbb{E}_{p(\mathbf{Z}\mid\mathbf{X})}[f(\mathbf{Z})]
=
\int f(\mathbf{Z})p(\mathbf{Z}\mid\mathbf{X})\,d\mathbf{Z}.
$$

Predictions, posterior means, posterior variances, and expected losses are all examples of posterior expectations.

When $\mathbf{Z}$ is high-dimensional, these integrals can be difficult for several reasons:

1. the posterior may have no standard functional form;
2. variables may be strongly coupled;
3. the integral dimension may be large;
4. discrete latent configurations may grow exponentially;
5. nonlinear likelihoods may destroy conjugacy.

## 1.2 “Known Up to a Constant” Is Not Always Enough

Frequently we can evaluate

$$
p(\mathbf{Z}\mid\mathbf{X})
\propto
p(\mathbf{X},\mathbf{Z}),
$$

but we do not know the normalizing constant $p(\mathbf{X})$.

For optimization, this may be sufficient: the MAP estimate can sometimes be found from the unnormalized posterior. But for uncertainty-aware prediction, we generally need more than one mode. We need integrals over the posterior.

This distinction is important:

| Task | Is an unnormalized posterior often sufficient? |
|------|-----------------------------------------------|
| MAP estimation | Often yes |
| Comparing posterior density at two points | Often yes |
| Computing posterior probabilities | Usually no |
| Computing posterior mean or variance | Usually no |
| Bayesian prediction | Usually no |
| Model evidence | No |

## 1.3 Exact, Deterministic Approximate, and Sampling-Based Inference

There are three broad possibilities.

| Category | Representative Methods | Strength | Limitation |
|----------|------------------------|----------|------------|
| Exact inference | Conjugate updates, Gaussian algebra, tree message passing | No approximation error under model assumptions | Often unavailable or exponentially expensive |
| Deterministic approximation | Laplace, variational inference, EP | Usually faster and easier to monitor | Introduces systematic approximation bias |
| Stochastic approximation | Monte Carlo, MCMC | Can approach exact expectations with enough samples | Sampling may mix slowly and be computationally expensive |

Chapter 10 focuses on deterministic methods. Chapter 11 will study sampling methods.

## 1.4 The Basic Variational Idea

Instead of working directly with the exact posterior, choose a tractable $q(\mathbf{Z})$ and make it as close as possible to $p(\mathbf{Z}\mid\mathbf{X})$.

> ![Figure 10.1](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_1__textbook_fig_10_1__p464.png)
>
> *Figure 10.1 (Textbook Fig. 10.1, p. 464): A simple one-dimensional variational approximation. The approximation $q(z)$ is chosen from a restricted family and adjusted to resemble the target distribution. The right panel shows the objective as a function of the approximation parameter.*

This converts inference into optimization:

$$
q^*
=
\arg\min_{q\in\mathcal{Q}}
\mathrm{KL}\bigl(q(\mathbf{Z})\Vert p(\mathbf{Z}\mid\mathbf{X})\bigr).
$$

The family $\mathcal{Q}$ must balance two goals:

- **expressiveness:** rich enough to resemble the true posterior;
- **tractability:** simple enough for expectations and updates to be computed.

A very simple family is fast but biased. A very flexible family can be accurate but expensive. Approximate inference is therefore also a problem of choosing a useful representation for uncertainty.

---

# §2 Variational Inference and the Evidence Lower Bound

> 📖 Textbook §10.1, especially Eqs. (10.1)–(10.4)

## 2.1 The Central Decomposition

The most important identity in the chapter is

$$
\boxed{
\ln p(\mathbf{X})
=
\mathcal{L}(q)
+
\mathrm{KL}\left(q(\mathbf{Z})\Vert p(\mathbf{Z}\mid\mathbf{X})\right)
}
$$

where

$$
\mathcal{L}(q)
=
\int q(\mathbf{Z})
\ln\frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}
\,d\mathbf{Z}.
$$

Because KL divergence is non-negative,

$$
\mathrm{KL}(q\Vert p)\geq 0,
$$

we immediately have

$$
\boxed{\mathcal{L}(q)\leq \ln p(\mathbf{X}).}
$$

That is why $\mathcal{L}(q)$ is called the **evidence lower bound**, or ELBO.

## 2.2 Step-by-Step Derivation of the ELBO Identity

This derivation should not be memorized as a trick. Each step has a clear purpose.

### Step 1: start from the log evidence

Since $q(\mathbf{Z})$ is normalized,

$$
\int q(\mathbf{Z})\,d\mathbf{Z}=1.
$$

Therefore,

$$
\ln p(\mathbf{X})
=
\int q(\mathbf{Z})\ln p(\mathbf{X})\,d\mathbf{Z}.
$$

### Step 2: use Bayes' theorem

Bayes' theorem gives

$$
p(\mathbf{Z}\mid\mathbf{X})
=
\frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{X})}.
$$

Rearranging,

$$
p(\mathbf{X})
=
\frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{Z}\mid\mathbf{X})}.
$$

Substitute this into the logarithm:

$$
\ln p(\mathbf{X})
=
\int q(\mathbf{Z})
\ln\frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{Z}\mid\mathbf{X})}
\,d\mathbf{Z}.
$$

### Step 3: multiply and divide by $q(\mathbf{Z})$

Inside the logarithm,

$$
\frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{Z}\mid\mathbf{X})}
=
\frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}
\cdot
\frac{q(\mathbf{Z})}{p(\mathbf{Z}\mid\mathbf{X})}.
$$

Taking the logarithm turns the product into a sum:

$$
\ln p(\mathbf{X})
=
\int q(\mathbf{Z})
\ln\frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}
\,d\mathbf{Z}
+
\int q(\mathbf{Z})
\ln\frac{q(\mathbf{Z})}{p(\mathbf{Z}\mid\mathbf{X})}
\,d\mathbf{Z}.
$$

### Step 4: recognize the two terms

The first term is the ELBO:

$$
\mathcal{L}(q)
=
\int q(\mathbf{Z})
\ln\frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}
\,d\mathbf{Z}.
$$

The second term is the KL divergence:

$$
\mathrm{KL}(q\Vert p)
=
\int q(\mathbf{Z})
\ln\frac{q(\mathbf{Z})}{p(\mathbf{Z}\mid\mathbf{X})}
\,d\mathbf{Z}.
$$

Hence,

$$
\ln p(\mathbf{X})
=
\mathcal{L}(q)
+
\mathrm{KL}(q\Vert p).
$$

## 2.3 Why Maximizing the ELBO Approximates the Posterior

For a fixed model and fixed data, $\ln p(\mathbf{X})$ is a constant with respect to $q$. Therefore,

$$
\arg\max_q \mathcal{L}(q)
=
\arg\min_q \mathrm{KL}\left(q(\mathbf{Z})\Vert p(\mathbf{Z}\mid\mathbf{X})\right).
$$

The gap between the log evidence and the ELBO is exactly the KL divergence:

$$
\ln p(\mathbf{X})-\mathcal{L}(q)
=
\mathrm{KL}(q\Vert p).
$$

Thus:

- a larger ELBO means a smaller KL gap;
- if $q$ equals the exact posterior, the KL is zero;
- in that ideal case, the bound becomes tight:

$$
\mathcal{L}(q)=\ln p(\mathbf{X}).
$$

## 2.4 The ELBO as “Fit Plus Entropy”

Expand the ELBO:

$$
\mathcal{L}(q)
=
\mathbb{E}_q[\ln p(\mathbf{X},\mathbf{Z})]
-
\mathbb{E}_q[\ln q(\mathbf{Z})].
$$

Since

$$
H[q]
=-\mathbb{E}_q[\ln q(\mathbf{Z})],
$$

we can write

$$
\boxed{
\mathcal{L}(q)
=
\mathbb{E}_q[\ln p(\mathbf{X},\mathbf{Z})]
+
H[q].
}
$$

This gives a useful interpretation.

| Term | Effect |
|------|--------|
| $\mathbb{E}_q[\ln p(\mathbf{X},\mathbf{Z})]$ | Encourages $q$ to place mass where the model assigns high joint probability. |
| $H[q]$ | Discourages $q$ from collapsing too aggressively; rewards uncertainty or spread. |

The ELBO balances **fitting the model** and **maintaining uncertainty**.

## 2.5 A Second Derivation Using Jensen's Inequality

The evidence can also be written as

$$
p(\mathbf{X})
=
\int q(\mathbf{Z})
\frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}
\,d\mathbf{Z}.
$$

Taking logs,

$$
\ln p(\mathbf{X})
=
\ln\mathbb{E}_{q}
\left[
\frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}
\right].
$$

Because $\ln(\cdot)$ is concave, Jensen's inequality gives

$$
\ln\mathbb{E}_q[Y]
\geq
\mathbb{E}_q[\ln Y].
$$

Set

$$
Y=\frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}.
$$

Then

$$
\ln p(\mathbf{X})
\geq
\mathbb{E}_q
\left[
\ln\frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}
\right]
=
\mathcal{L}(q).
$$

The KL decomposition is more informative because it tells us the exact size of the gap. The Jensen derivation is useful because it shows why lower bounds arise naturally.

## 2.6 Guided Textbook Exercise 10.1: Verify the Decomposition

> ![Textbook Exercise 10.1](./CoursePR2026/Fig/Chapter_10/lecture_ex_10_1__textbook_ex_10_1__p517.png)
>
> *Textbook Exercise 10.1 (p. 517): Verify that the log evidence decomposes into the ELBO plus a KL divergence.*

A clean solution is:

$$
\begin{aligned}
\mathcal{L}(q)
+
\mathrm{KL}(q\Vert p)
&=
\int q(\mathbf{Z})
\ln\frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}
\,d\mathbf{Z}\\
&\quad+
\int q(\mathbf{Z})
\ln\frac{q(\mathbf{Z})}{p(\mathbf{Z}\mid\mathbf{X})}
\,d\mathbf{Z}.
\end{aligned}
$$

Combine the logarithms:

$$
\begin{aligned}
\mathcal{L}(q)+\mathrm{KL}(q\Vert p)
&=
\int q(\mathbf{Z})
\ln
\left[
\frac{p(\mathbf{X},\mathbf{Z})}{q(\mathbf{Z})}
\frac{q(\mathbf{Z})}{p(\mathbf{Z}\mid\mathbf{X})}
\right]
\,d\mathbf{Z}\\
&=
\int q(\mathbf{Z})
\ln
\frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{Z}\mid\mathbf{X})}
\,d\mathbf{Z}.
\end{aligned}
$$

By Bayes' theorem,

$$
\frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{Z}\mid\mathbf{X})}
=p(\mathbf{X}).
$$

This quantity does not depend on $\mathbf{Z}$, so

$$
\begin{aligned}
\mathcal{L}(q)+\mathrm{KL}(q\Vert p)
&=
\int q(\mathbf{Z})\ln p(\mathbf{X})\,d\mathbf{Z}\\
&=
\ln p(\mathbf{X})
\int q(\mathbf{Z})\,d\mathbf{Z}\\
&=
\ln p(\mathbf{X}).
\end{aligned}
$$

The last step uses the normalization of $q$.

## 2.7 What the ELBO Does Not Guarantee

A monotonic increase in the ELBO is useful, but it does not guarantee that the final approximation is globally optimal.

Three limitations should be kept separate:

1. **family limitation:** the exact posterior may not belong to $\mathcal{Q}$;
2. **optimization limitation:** the ELBO may have multiple local optima;
3. **model limitation:** even exact inference cannot fix a poor probabilistic model.

A high ELBO is meaningful only relative to a specified model, data set, and variational family.

---

# §3 Mean-Field Variational Inference

> 📖 Textbook §10.1.1 Factorized Distributions

## 3.1 The Mean-Field Assumption

Suppose the hidden variables are divided into $M$ groups:

$$
\mathbf{Z}
=
(\mathbf{Z}_1,\mathbf{Z}_2,\ldots,\mathbf{Z}_M).
$$

Mean-field variational inference assumes

$$
\boxed{
q(\mathbf{Z})
=
\prod_{i=1}^{M}q_i(\mathbf{Z}_i).
}
$$

This is an **assumption about the approximation**, not necessarily an assumption about the true posterior. The exact posterior may contain strong dependence between the groups even though the approximation does not.

Why make this assumption? Because it often turns one difficult high-dimensional optimization into a sequence of simpler updates.

## 3.2 Optimizing One Factor at a Time

Fix every factor except $q_j(\mathbf{Z}_j)$. We want to find the best $q_j$.

Start from the ELBO:

$$
\mathcal{L}(q)
=
\int q(\mathbf{Z})
\ln p(\mathbf{X},\mathbf{Z})
\,d\mathbf{Z}
-
\int q(\mathbf{Z})
\ln q(\mathbf{Z})
\,d\mathbf{Z}.
$$

Using the factorization

$$
q(\mathbf{Z})=q_j(\mathbf{Z}_j)q_{-j}(\mathbf{Z}_{-j}),
$$

where

$$
q_{-j}(\mathbf{Z}_{-j})
=
\prod_{i\neq j}q_i(\mathbf{Z}_i),
$$

we can collect all terms that depend on $q_j$.

## 3.3 Step-by-Step Derivation of the Mean-Field Update

### Step 1: average the log joint over all other factors

Define

$$
\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]
=
\int q_{-j}(\mathbf{Z}_{-j})
\ln p(\mathbf{X},\mathbf{Z})
\,d\mathbf{Z}_{-j}.
$$

This is now a function of $\mathbf{Z}_j$ only.

### Step 2: isolate the terms involving $q_j$

Up to constants independent of $q_j$,

$$
\mathcal{L}(q)
=
\int q_j(\mathbf{Z}_j)
\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]
\,d\mathbf{Z}_j
-
\int q_j(\mathbf{Z}_j)\ln q_j(\mathbf{Z}_j)
\,d\mathbf{Z}_j
+
\text{const}.
$$

### Step 3: define an unnormalized target for this factor

Let

$$
\ln \widetilde p_j(\mathbf{Z}_j)
=
\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]
+
\text{const}.
$$

Equivalently,

$$
\widetilde p_j(\mathbf{Z}_j)
\propto
\exp\left
\{
\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]
\right
\}.
$$

### Step 4: recognize a negative KL divergence

The part of the ELBO depending on $q_j$ can be written as

$$
-\mathrm{KL}
\left(
q_j(\mathbf{Z}_j)
\Vert
\widetilde p_j(\mathbf{Z}_j)
\right)
+
\text{const}.
$$

The KL is minimized when the two distributions match. Therefore,

$$
\boxed{
\ln q_j^*(\mathbf{Z}_j)
=
\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]
+
\text{const}.
}
$$

Or, in exponential form,

$$
\boxed{
q_j^*(\mathbf{Z}_j)
\propto
\exp\left
\{
\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]
\right
\}.
}
$$

This is the most reusable equation in mean-field variational inference.

## 3.4 How to Read the Update Rule

The update rule says:

> To update one group of variables, take the log of the complete joint model, average over all other variable groups using their current approximations, then exponentiate and normalize.

A practical memory aid is:

$$
\boxed{
\text{new factor}
\propto
\exp(\text{expected log joint}).
}
$$

The expectation removes the variables we are not updating. The exponential turns the expected log density back into a density shape.

## 3.5 Coordinate-Ascent Variational Inference

The resulting algorithm is coordinate ascent.

### CAVI algorithm

1. Initialize $q_1,\ldots,q_M$.
2. For $j=1,\ldots,M$:
   1. compute $\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]$;
   2. set
      $$
      q_j(\mathbf{Z}_j)
      \propto
      \exp\{\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]\};
      $$
   3. normalize the factor.
3. Evaluate or monitor the ELBO.
4. Repeat until the ELBO or variational parameters change very little.

Each exact coordinate update cannot decrease the ELBO, because it maximizes the bound with respect to one factor while holding the others fixed.

## 3.6 Why Updates Are Often Closed Form

The update rule becomes especially useful for conjugate-exponential models. If the complete conditional for $\mathbf{Z}_j$ belongs to a familiar exponential family, the variational factor usually belongs to the same family.

Examples:

- Gaussian conditional $\Rightarrow$ Gaussian variational factor;
- categorical conditional $\Rightarrow$ categorical variational factor;
- gamma conditional $\Rightarrow$ gamma variational factor;
- Dirichlet conditional $\Rightarrow$ Dirichlet variational factor.

This is why the method often looks like Bayesian updating with uncertain sufficient statistics replaced by their expectations.

## 3.7 Relationship to EM

EM and mean-field VI are closely related but not identical.

| EM | Variational Inference |
|----|-----------------------|
| Usually keeps a point estimate of parameters $\boldsymbol{\theta}$ | Can keep distributions over parameters and latent variables |
| E-step computes the exact posterior over latent variables given current parameters | Each variational step computes an approximate factor |
| M-step maximizes an expected complete-data log likelihood | Coordinate updates maximize the ELBO |
| Objective is usually the data log likelihood | Objective is the ELBO |

A useful special case is obtained if the variational factor over parameters is forced to be a delta distribution:

$$
q(\boldsymbol{\theta})
=
\delta(\boldsymbol{\theta}-\boldsymbol{\theta}_0).
$$

Then variational optimization reduces to an EM-like alternating procedure.

## 3.8 Practical Convergence Checks

A robust implementation should monitor more than one quantity.

| Check | What It Detects |
|-------|-----------------|
| ELBO should not decrease substantially | Algebra or implementation errors |
| Variational parameters should stabilize | Numerical convergence |
| Responsibilities should sum to one | Normalization errors |
| Covariance/precision matrices should remain valid | Numerical instability |
| Multiple initializations should be compared | Local-optimum sensitivity |

Small numerical decreases can occur from floating-point error, but systematic decreases usually indicate a bug.

---

# §4 What Mean-Field Approximation Gets Right and Wrong

> 📖 Textbook §10.1.2 Properties of Factorized Approximations

## 4.1 A Correlated Gaussian Example

Consider a two-dimensional Gaussian target distribution

$$
p(\mathbf{z})
=
\mathcal{N}(\mathbf{z}\mid\boldsymbol{\mu},\boldsymbol{\Sigma}),
$$

with precision matrix

$$
\boldsymbol{\Lambda}
=
\boldsymbol{\Sigma}^{-1}
=
\begin{bmatrix}
\Lambda_{11} & \Lambda_{12}\\
\Lambda_{21} & \Lambda_{22}
\end{bmatrix}.
$$

We approximate it by a factorized Gaussian:

$$
q(z_1,z_2)=q_1(z_1)q_2(z_2).
$$

The exact target may have tilted elliptical contours, indicating correlation. The factorized approximation cannot represent that tilt.

> ![Figure 10.2](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_2__textbook_fig_10_2__p468.png)
>
> *Figure 10.2 (Textbook Fig. 10.2, p. 468): The two KL directions lead to different Gaussian approximations. Minimizing $\mathrm{KL}(q\Vert p)$ tends to fit inside the target contours, whereas minimizing $\mathrm{KL}(p\Vert q)$ tends to cover the target mass.*

## 4.2 Deriving the Factor Updates

The log density of the target, ignoring constants, is

$$
\ln p(\mathbf{z})
=
-\frac{1}{2}
(\mathbf{z}-\boldsymbol{\mu})^T
\boldsymbol{\Lambda}
(\mathbf{z}-\boldsymbol{\mu})
+
\text{const}.
$$

To update $q_1(z_1)$, take expectation over $q_2(z_2)$:

$$
\ln q_1^*(z_1)
=
\mathbb{E}_{q_2}[\ln p(z_1,z_2)]
+
\text{const}.
$$

Keeping only terms involving $z_1$ gives a quadratic function:

$$
\ln q_1^*(z_1)
=
-\frac{1}{2}\Lambda_{11}(z_1-m_1)^2
+
\text{const},
$$

so

$$
q_1(z_1)
=
\mathcal{N}
\left(
 z_1\mid m_1,\Lambda_{11}^{-1}
\right).
$$

Its mean is

$$
m_1
=
\mu_1
-
\Lambda_{11}^{-1}\Lambda_{12}
\left(\mathbb{E}[z_2]-\mu_2\right).
$$

Similarly,

$$
q_2(z_2)
=
\mathcal{N}
\left(
 z_2\mid m_2,\Lambda_{22}^{-1}
\right),
$$

with

$$
m_2
=
\mu_2
-
\Lambda_{22}^{-1}\Lambda_{21}
\left(\mathbb{E}[z_1]-\mu_1\right).
$$

Because $\mathbb{E}[z_1]=m_1$ and $\mathbb{E}[z_2]=m_2$, the means are coupled and must be iterated.

## 4.3 Guided Textbook Exercise 10.2: Solve the Coupled Means

> ![Textbook Exercise 10.2](./CoursePR2026/Fig/Chapter_10/lecture_ex_10_2__textbook_ex_10_2__p517.png)
>
> *Textbook Exercise 10.2 (p. 517): Show that the fixed-point means of the factorized Gaussian approximation equal the true Gaussian means, provided the original distribution is nonsingular.*

Define deviations from the true means:

$$
\delta_1=m_1-\mu_1,
\qquad
\delta_2=m_2-\mu_2.
$$

The update equations become

$$
\delta_1
=-a\delta_2,
\qquad
 a=\Lambda_{11}^{-1}\Lambda_{12},
$$

and

$$
\delta_2
=-b\delta_1,
\qquad
 b=\Lambda_{22}^{-1}\Lambda_{21}.
$$

Substitute the second equation into the first:

$$
\delta_1
=-a(-b\delta_1)
=ab\delta_1.
$$

Therefore,

$$
(1-ab)\delta_1=0.
$$

For a nonsingular Gaussian precision matrix, the degenerate case $ab=1$ does not occur. Hence,

$$
\delta_1=0.
$$

Then

$$
\delta_2=-b\delta_1=0.
$$

Thus,

$$
\boxed{m_1=\mu_1,\qquad m_2=\mu_2.}
$$

The approximation recovers the correct means in this special Gaussian example.

## 4.4 Why the Variances Are Too Small

Although the means are correct, the factor variances are

$$
\operatorname{var}_{q_1}[z_1]=\Lambda_{11}^{-1},
$$

and

$$
\operatorname{var}_{q_2}[z_2]=\Lambda_{22}^{-1}.
$$

These are conditional-type variances, not the full marginal variances of the correlated Gaussian. When $z_1$ and $z_2$ are correlated, the full marginal uncertainty is larger.

The mean-field approximation therefore often becomes **overconfident**:

> It cannot represent posterior correlation, so it may compensate by shrinking each independent factor around a plausible region.

This is an important practical warning. Variational posterior means can be useful even when uncertainty intervals are too narrow.

## 4.5 The Direction of KL Divergence

KL divergence is asymmetric:

$$
\mathrm{KL}(q\Vert p)
\neq
\mathrm{KL}(p\Vert q).
$$

This asymmetry has a visible effect when the target is multimodal.

> ![Figure 10.3](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_3__textbook_fig_10_3__p469.png)
>
> *Figure 10.3 (Textbook Fig. 10.3, p. 469): For a bimodal target, minimizing $\mathrm{KL}(q\Vert p)$ may select one mode, while minimizing $\mathrm{KL}(p\Vert q)$ produces a broader approximation that covers both modes.*

### Forward variational direction: $\mathrm{KL}(q\Vert p)$

$$
\mathrm{KL}(q\Vert p)
=
\int q(z)\ln\frac{q(z)}{p(z)}\,dz.
$$

The expectation is taken under $q$. If $q$ puts mass where $p$ is nearly zero, the ratio $q/p$ becomes very large. This is strongly penalized.

Therefore $q$ tends to avoid low-density gaps in $p$. For a multimodal target, it may choose a single mode rather than spreading across the gap. This behavior is sometimes called:

- **zero forcing**;
- **mode seeking**.

### Reverse direction: $\mathrm{KL}(p\Vert q)$

$$
\mathrm{KL}(p\Vert q)
=
\int p(z)\ln\frac{p(z)}{q(z)}\,dz.
$$

Now the expectation is taken under $p$. If $p$ has mass where $q$ is nearly zero, the divergence becomes large.

Therefore $q$ is encouraged to cover all important regions of $p$. This behavior is sometimes called:

- **zero avoiding**;
- **mass covering**.

## 4.6 A Simple Two-Mode Thought Experiment

Suppose the target consists of two narrow peaks far apart.

A one-Gaussian approximation has two natural choices:

1. place itself on one peak;
2. place itself between the peaks and become wide enough to cover both.

Under $\mathrm{KL}(q\Vert p)$, the second option puts substantial $q$ mass in the low-density gap. This is expensive, so one peak may be selected.

Under $\mathrm{KL}(p\Vert q)$, ignoring either peak is expensive because $p$ has mass there. The broad covering solution is preferred.

Neither behavior is universally better. The suitable approximation depends on the downstream task.

## 4.7 Common Misinterpretation

It is incorrect to say:

> “Variational inference always underestimates variance.”

A more accurate statement is:

> Mean-field variational inference with $\mathrm{KL}(q\Vert p)$ often underestimates uncertainty when the true posterior has strong dependencies or multiple modes.

The result depends on the variational family, the divergence, and the model.

---

# §5 A Worked Variational Example: Unknown Gaussian Mean and Precision

> 📖 Textbook §10.1.3 Example: The Univariate Gaussian

## 5.1 Problem Setup

Suppose we observe scalar data

$$
\mathcal{D}=\{x_1,x_2,\ldots,x_N\}
$$

from a Gaussian distribution with unknown mean $\mu$ and unknown precision $\tau$:

$$
p(x_n\mid\mu,\tau)
=
\mathcal{N}(x_n\mid\mu,\tau^{-1}).
$$

The likelihood for all data is

$$
p(\mathcal{D}\mid\mu,\tau)
=
\prod_{n=1}^{N}
\mathcal{N}(x_n\mid\mu,\tau^{-1}).
$$

We use a conjugate Gaussian–Gamma prior:

$$
p(\mu,\tau)
=
p(\mu\mid\tau)p(\tau),
$$

with

$$
p(\mu\mid\tau)
=
\mathcal{N}
\left(
\mu\mid\mu_0,(\lambda_0\tau)^{-1}
\right),
$$

and

$$
p(\tau)
=
\operatorname{Gam}(\tau\mid a_0,b_0).
$$

The exact posterior couples $\mu$ and $\tau$. We deliberately approximate it by

$$
q(\mu,\tau)
=
q_\mu(\mu)q_\tau(\tau).
$$

This is a small example in which the complete mean-field procedure can be seen clearly.

## 5.2 Updating the Mean Factor $q_\mu(\mu)$

Use the general rule:

$$
\ln q_\mu^*(\mu)
=
\mathbb{E}_{q_\tau}[\ln p(\mathcal{D},\mu,\tau)]
+
\text{const}.
$$

Keep only terms involving $\mu$. The likelihood contributes

$$
-\frac{\tau}{2}
\sum_{n=1}^{N}(x_n-\mu)^2.
$$

The prior on $\mu$ contributes

$$
-\frac{\lambda_0\tau}{2}
(\mu-\mu_0)^2.
$$

Take expectation over $q_\tau$ by replacing $\tau$ with $\mathbb{E}[\tau]$:

$$
\ln q_\mu^*(\mu)
=
-\frac{\mathbb{E}[\tau]}{2}
\left[
\sum_{n=1}^{N}(x_n-\mu)^2
+
\lambda_0(\mu-\mu_0)^2
\right]
+
\text{const}.
$$

This is quadratic in $\mu$, so $q_\mu$ is Gaussian:

$$
\boxed{
q_\mu(\mu)
=
\mathcal{N}(\mu\mid\mu_N,\lambda_N^{-1}).
}
$$

The mean is

$$
\boxed{
\mu_N
=
\frac{\lambda_0\mu_0+N\bar{x}}
{\lambda_0+N},
}
$$

where

$$
\bar{x}
=
\frac{1}{N}\sum_{n=1}^{N}x_n.
$$

The precision is

$$
\boxed{
\lambda_N
=
(\lambda_0+N)\mathbb{E}[\tau].
}
$$

### Interpretation of the mean update

The posterior mean $\mu_N$ is a weighted average of:

- prior mean $\mu_0$, with weight $\lambda_0$;
- sample mean $\bar{x}$, with weight $N$.

As $N$ grows, the data dominate the prior.

## 5.3 Updating the Precision Factor $q_\tau(\tau)$

Now use

$$
\ln q_\tau^*(\tau)
=
\mathbb{E}_{q_\mu}[\ln p(\mathcal{D},\mu,\tau)]
+
\text{const}.
$$

Collecting the terms involving $\tau$ gives a gamma distribution:

$$
\boxed{
q_\tau(\tau)
=
\operatorname{Gam}(\tau\mid a_N,b_N).
}
$$

The updated shape is

$$
\boxed{
a_N=a_0+\frac{N}{2}.
}
$$

The updated rate is

$$
\boxed{
b_N
=
b_0
+
\frac{1}{2}
\mathbb{E}_{q_\mu}
\left[
\sum_{n=1}^{N}(x_n-\mu)^2
+
\lambda_0(\mu-\mu_0)^2
\right].
}
$$

For a gamma distribution in shape–rate form,

$$
\mathbb{E}[\tau]
=
\frac{a_N}{b_N}.
$$

This expectation is fed back into the update for $q_\mu$.

## 5.4 Evaluating the Required Squared-Error Expectation

The term

$$
\mathbb{E}[(x_n-\mu)^2]
$$

can be expanded without skipping steps.

Write

$$
x_n-\mu
=
(x_n-\mu_N)+(\mu_N-\mu).
$$

Squaring,

$$
(x_n-\mu)^2
=
(x_n-\mu_N)^2
+2(x_n-\mu_N)(\mu_N-\mu)
+(\mu_N-\mu)^2.
$$

Take expectation under $q_\mu$.

Because

$$
\mathbb{E}[\mu_N-\mu]
=
\mu_N-\mathbb{E}[\mu]
=0,
$$

the cross term vanishes. Therefore,

$$
\boxed{
\mathbb{E}[(x_n-\mu)^2]
=
(x_n-\mu_N)^2
+
\operatorname{var}(\mu).
}
$$

Since

$$
\operatorname{var}(\mu)=\lambda_N^{-1},
$$

we have

$$
\mathbb{E}[(x_n-\mu)^2]
=
(x_n-\mu_N)^2
+
\lambda_N^{-1}.
$$

This formula shows an important Bayesian idea: squared error includes both the error at the posterior mean and uncertainty about that mean.

## 5.5 The Coordinate-Ascent Cycle

The two factors depend on each other:

- $q_\mu$ needs $\mathbb{E}[\tau]$;
- $q_\tau$ needs expectations under $q_\mu$.

Therefore, iterate:

1. initialize $\mathbb{E}[\tau]$;
2. update $q_\mu(\mu)$;
3. compute $\mathbb{E}[\mu]$ and $\mathbb{E}[\mu^2]$;
4. update $q_\tau(\tau)$;
5. compute $\mathbb{E}[\tau]=a_N/b_N$;
6. repeat until convergence.

> ![Figure 10.4](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_4__textbook_fig_10_4__p472.png)
>
> *Figure 10.4 (Textbook Fig. 10.4, p. 472): Coordinate-ascent updates for the factorized approximation $q_\mu(\mu)q_\tau(\tau)$. Each update improves the approximation in one coordinate while holding the other factor fixed.*

## 5.6 What This Example Teaches

This small example contains the general pattern of variational Bayes:

1. choose a factorization;
2. write the log joint;
3. keep terms involving the current variable group;
4. take expectations over the other groups;
5. recognize the resulting distribution family;
6. update moments and repeat.

The difficult-looking derivations are usually repeated applications of this same pattern.

## 5.7 A Small Numerical Illustration

Suppose:

$$
N=4,
\qquad
x=(1,2,2,3),
$$

so

$$
\bar{x}=2.
$$

Let the prior parameters be

$$
\mu_0=0,
\qquad
\lambda_0=1.
$$

Then the updated mean is

$$
\mu_N
=
\frac{1\cdot 0+4\cdot 2}{1+4}
=
\frac{8}{5}
=1.6.
$$

The prior pulls the estimate from the sample mean $2$ toward the prior mean $0$. If $N$ were much larger, this pull would become relatively weak.

Suppose the current expected precision is

$$
\mathbb{E}[\tau]=2.
$$

Then

$$
\lambda_N
=(1+4)\cdot 2
=10,
$$

so

$$
\operatorname{var}_{q_\mu}(\mu)
=\lambda_N^{-1}
=0.1.
$$

This is the uncertainty of the variational factor for the mean at the current iteration.

---

# §6 Variational Gaussian Mixtures

> 📖 Textbook §10.2 Illustration: Variational Mixture of Gaussians

## 6.1 Why Revisit Gaussian Mixtures?

In Chapter 9, maximum-likelihood EM estimated one value for each mixture parameter:

$$
\{\boldsymbol{\pi},\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k\}.
$$

That approach has two well-known problems:

1. the likelihood can become singular if a covariance collapses around one data point;
2. the number of components must usually be selected externally.

A Bayesian treatment places distributions over the parameters. Variational inference makes this Bayesian treatment computationally manageable.

## 6.2 The Bayesian Mixture Model

For each data point $\mathbf{x}_n$:

1. choose a component assignment $\mathbf{z}_n$ from a categorical distribution;
2. generate $\mathbf{x}_n$ from the corresponding Gaussian.

The likelihood is

$$
p(\mathbf{X}\mid\mathbf{Z},\boldsymbol{\mu},\boldsymbol{\Lambda})
=
\prod_{n=1}^{N}
\prod_{k=1}^{K}
\mathcal{N}
(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Lambda}_k^{-1})^{z_{nk}}.
$$

The assignment prior is

$$
p(\mathbf{Z}\mid\boldsymbol{\pi})
=
\prod_{n=1}^{N}
\prod_{k=1}^{K}
\pi_k^{z_{nk}}.
$$

We place a Dirichlet prior on the mixing coefficients:

$$
p(\boldsymbol{\pi})
=
\operatorname{Dir}(\boldsymbol{\pi}\mid\boldsymbol{\alpha}_0).
$$

Each Gaussian component receives a conjugate Gaussian–Wishart prior over its mean and precision.

> ![Figure 10.5](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_5__textbook_fig_10_5__p475.png)
>
> *Figure 10.5 (Textbook Fig. 10.5, p. 475): Directed graphical model for the Bayesian Gaussian mixture. The latent assignments, mixture coefficients, component means, and precision matrices are all treated probabilistically.*

## 6.3 The Variational Factorization

A standard factorization is

$$
q(\mathbf{Z},\boldsymbol{\pi},\boldsymbol{\mu},\boldsymbol{\Lambda})
=
q(\mathbf{Z})
q(\boldsymbol{\pi},\boldsymbol{\mu},\boldsymbol{\Lambda}).
$$

The conjugate structure induces a further factorization:

$$
q(\boldsymbol{\pi},\boldsymbol{\mu},\boldsymbol{\Lambda})
=
q(\boldsymbol{\pi})
\prod_{k=1}^{K}
q(\boldsymbol{\mu}_k,\boldsymbol{\Lambda}_k).
$$

Thus the algorithm alternates between:

- assignment uncertainty $q(\mathbf{Z})$;
- mixture-weight uncertainty $q(\boldsymbol{\pi})$;
- component-parameter uncertainty $q(\boldsymbol{\mu}_k,\boldsymbol{\Lambda}_k)$.

## 6.4 Updating the Responsibilities

The assignment factor has the form

$$
q(\mathbf{Z})
=
\prod_{n=1}^{N}
\prod_{k=1}^{K}
r_{nk}^{z_{nk}},
$$

where

$$
\sum_{k=1}^{K}r_{nk}=1.
$$

The unnormalized responsibility is

$$
\ln \rho_{nk}
=
\mathbb{E}[\ln \pi_k]
+
\frac{1}{2}\mathbb{E}[\ln|\boldsymbol{\Lambda}_k|]
-
\frac{D}{2}\ln(2\pi)
-
\frac{1}{2}
\mathbb{E}
\left[
(\mathbf{x}_n-\boldsymbol{\mu}_k)^T
\boldsymbol{\Lambda}_k
(\mathbf{x}_n-\boldsymbol{\mu}_k)
\right].
$$

Normalize using

$$
\boxed{
r_{nk}
=
\frac{\rho_{nk}}
{\sum_{j=1}^{K}\rho_{nj}}.
}
$$

### How this differs from ordinary EM

In maximum-likelihood EM, a responsibility uses point estimates of $\pi_k$, $\boldsymbol{\mu}_k$, and $\boldsymbol{\Sigma}_k$.

In variational Bayes, it uses **expected log parameters** and **expected quadratic distances**. Parameter uncertainty directly affects assignment confidence.

## 6.5 Effective Counts and Weighted Statistics

Once responsibilities are available, compute

$$
N_k
=
\sum_{n=1}^{N}r_{nk},
$$

$$
\overline{\mathbf{x}}_k
=
\frac{1}{N_k}
\sum_{n=1}^{N}r_{nk}\mathbf{x}_n,
$$

and

$$
\mathbf{S}_k
=
\frac{1}{N_k}
\sum_{n=1}^{N}
r_{nk}
(\mathbf{x}_n-\overline{\mathbf{x}}_k)
(\mathbf{x}_n-\overline{\mathbf{x}}_k)^T.
$$

These look like the EM sufficient statistics, but they are used to update posterior distributions rather than point estimates.

## 6.6 Updating the Mixture Coefficients

The variational distribution over $\boldsymbol{\pi}$ is Dirichlet:

$$
q(\boldsymbol{\pi})
=
\operatorname{Dir}(\boldsymbol{\pi}\mid\boldsymbol{\alpha}),
$$

with

$$
\boxed{
\alpha_k
=
\alpha_{0k}+N_k.
}
$$

This update is easy to understand:

> posterior pseudo-count = prior pseudo-count + soft data count.

The expected mixing coefficient is

$$
\mathbb{E}[\pi_k]
=
\frac{\alpha_k}{\sum_j\alpha_j}.
$$

The responsibility update needs $\mathbb{E}[\ln\pi_k]$, which involves the digamma function. For this lecture, the conceptual point matters more than memorizing that special-function formula.

## 6.7 Updating Component Means and Precisions

Each factor remains Gaussian–Wishart:

$$
q(\boldsymbol{\mu}_k,\boldsymbol{\Lambda}_k)
=
q(\boldsymbol{\mu}_k\mid\boldsymbol{\Lambda}_k)
q(\boldsymbol{\Lambda}_k).
$$

Important parameter updates include

$$
\beta_k
=
\beta_0+N_k,
$$

$$
\boxed{
\mathbf{m}_k
=
\frac{\beta_0\mathbf{m}_0+N_k\overline{\mathbf{x}}_k}
{\beta_0+N_k},
}
$$

and

$$
\nu_k
=
\nu_0+N_k.
$$

The mean update again has the form of a weighted average:

- prior location $\mathbf{m}_0$;
- responsibility-weighted sample mean $\overline{\mathbf{x}}_k$.

The full scale-matrix update combines within-component scatter and the difference between the prior mean and sample mean. It is useful when implementing the algorithm, but it is not necessary to memorize for conceptual understanding.

## 6.8 The Variational Mixture Algorithm

A readable algorithm is:

1. Choose a reasonably large candidate number of components $K$.
2. Initialize responsibilities $r_{nk}$.
3. Compute $N_k$, $\overline{\mathbf{x}}_k$, and $\mathbf{S}_k$.
4. Update $q(\boldsymbol{\pi})$.
5. Update each $q(\boldsymbol{\mu}_k,\boldsymbol{\Lambda}_k)$.
6. Recompute responsibilities using expected log parameters.
7. Evaluate the ELBO.
8. Repeat until convergence.

The pattern is similar to EM, but every “parameter” update is a distributional update.

## 6.9 Automatic Complexity Control

One of the most attractive properties is that unnecessary components can become inactive.

If component $k$ receives almost no responsibility, then

$$
N_k\approx 0.
$$

Its posterior parameters remain close to the prior, and its expected mixing coefficient becomes small. The component effectively switches off.

> ![Figure 10.6](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_6__textbook_fig_10_6__p480.png)
>
> *Figure 10.6 (Textbook Fig. 10.6, p. 480): A variational Bayesian Gaussian mixture begins with more components than necessary. During optimization, unsupported components lose responsibility and become inactive.*

This is not magic deletion. It is a consequence of Bayesian complexity control: a component must explain enough data to overcome the prior and the cost of allocating probability mass to it.

## 6.10 Why the Singularity Problem Is Reduced

In maximum-likelihood GMMs, one component can collapse its covariance onto a data point, causing the likelihood to diverge.

In the Bayesian treatment, priors and parameter uncertainty prevent a component from obtaining an unlimited reward from an infinitesimal covariance. The variational objective includes prior and entropy terms, not only data fit.

This does not remove every numerical issue, but it eliminates the classic maximum-likelihood singularity under proper priors.

## 6.11 Selecting the Number of Components

The variational lower bound approximates the log evidence. Therefore, it can be used to compare candidate values of $K$.

> ![Figure 10.7](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_7__textbook_fig_10_7__p484.png)
>
> *Figure 10.7 (Textbook Fig. 10.7, p. 484): Variational lower bound as a function of the number of mixture components. The evidence-like objective balances fit against complexity instead of rewarding an arbitrarily large model.*

Two related strategies are possible:

1. fit separate candidate values of $K$ and compare lower bounds;
2. start with a generous $K$ and let weak components become inactive.

Because the objective can have local optima, multiple initializations are still recommended.

## 6.12 Predictive Distribution

After integrating over component parameters, the predictive density becomes a mixture of Student's $t$ distributions rather than a mixture of plug-in Gaussians.

The heavier tails of the Student's $t$ distribution represent parameter uncertainty. When a component has little data, its predictive distribution is wider. When it has abundant data, the predictive distribution approaches a Gaussian.

## 6.13 What Students Should Remember

Do not memorize the full Gaussian–Wishart algebra. Remember the structure:

$$
\text{responsibilities}
\leftrightarrow
\text{posterior component parameters}.
$$

The algorithm alternates between asking:

1. Which component probably generated each point?
2. Given those soft assignments, what should we believe about each component?

This is the Bayesian analogue of EM.

---

# §7 Variational Linear Regression and Variational Message Passing

> 📖 Textbook §10.3–§10.4

## 7.1 Variational Linear Regression: Why It Is Included

Bayesian linear regression is exactly tractable when the noise precision and weight precision are known. If a hyperparameter such as the weight precision $\alpha$ is also uncertain, the posterior can couple $\mathbf{w}$ and $\alpha$.

A common mean-field approximation is

$$
q(\mathbf{w},\alpha)
=
q(\mathbf{w})q(\alpha).
$$

The two factors are updated alternately:

- $q(\mathbf{w})$ is Gaussian;
- $q(\alpha)$ is gamma.

## 7.2 Model Setup

Use the linear model

$$
t_n
=
\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)
+\epsilon_n,
$$

with

$$
\epsilon_n\sim\mathcal{N}(0,\beta^{-1}).
$$

The likelihood is

$$
p(\mathbf{t}\mid\mathbf{w},\beta)
=
\mathcal{N}(\mathbf{t}\mid\boldsymbol{\Phi}\mathbf{w},\beta^{-1}\mathbf{I}).
$$

Place a Gaussian prior on the weights:

$$
p(\mathbf{w}\mid\alpha)
=
\mathcal{N}(\mathbf{w}\mid\mathbf{0},\alpha^{-1}\mathbf{I}),
$$

and a gamma prior on $\alpha$.

## 7.3 Updating $q(\mathbf{w})$

The mean-field rule gives a Gaussian:

$$
q(\mathbf{w})
=
\mathcal{N}(\mathbf{w}\mid\mathbf{m}_N,\mathbf{S}_N),
$$

with precision

$$
\boxed{
\mathbf{S}_N^{-1}
=
\mathbb{E}[\alpha]\mathbf{I}
+
\beta\boldsymbol{\Phi}^T\boldsymbol{\Phi}.
}
$$

The mean is

$$
\boxed{
\mathbf{m}_N
=
\beta\mathbf{S}_N\boldsymbol{\Phi}^T\mathbf{t}.
}
$$

Compare this with ordinary Bayesian linear regression: the fixed regularization strength $\alpha$ is replaced by its current posterior expectation $\mathbb{E}[\alpha]$.

## 7.4 Updating $q(\alpha)$

The gamma factor depends on the expected squared weight magnitude:

$$
\mathbb{E}[\mathbf{w}^T\mathbf{w}]
=
\mathbf{m}_N^T\mathbf{m}_N
+
\operatorname{Tr}(\mathbf{S}_N).
$$

This identity is important. It says:

$$
\text{expected squared magnitude}
=
\text{squared posterior mean}
+
\text{posterior uncertainty}.
$$

The update for $q(\alpha)$ increases precision when weights are believed to be small and decreases precision when the weights need larger magnitude.

## 7.5 Prediction

The predictive mean is

$$
\mathbb{E}[t\mid\mathbf{x}]
=
\mathbf{m}_N^T\boldsymbol{\phi}(\mathbf{x}).
$$

The predictive variance contains:

1. observation noise;
2. uncertainty in the weight vector.

Thus,

$$
\operatorname{var}[t\mid\mathbf{x}]
\approx
\beta^{-1}
+
\boldsymbol{\phi}(\mathbf{x})^T
\mathbf{S}_N
\boldsymbol{\phi}(\mathbf{x}).
$$

The second term grows in input regions where the weights are uncertain.

## 7.6 Why This Section Is Useful

This example shows that variational inference can learn hyperparameters jointly with model parameters, instead of selecting one regularization value by a grid search.

It also illustrates the repeated structure:

$$
q(\mathbf{w})
\longleftrightarrow
q(\alpha).
$$

Each update uses moments of the other factor.

## 7.7 Exponential-Family View

An exponential-family distribution can be written as

$$
p(\mathbf{z}\mid\boldsymbol{\eta})
=
h(\mathbf{z})g(\boldsymbol{\eta})
\exp\left
\{
\boldsymbol{\eta}^T\mathbf{u}(\mathbf{z})
\right
\},
$$

where:

- $\boldsymbol{\eta}$ are natural parameters;
- $\mathbf{u}(\mathbf{z})$ are sufficient statistics.

In conjugate graphical models, the expected log joint is linear in sufficient statistics. Therefore the mean-field update can often be performed by adding expected natural-parameter contributions from neighboring factors.

## 7.8 Variational Message Passing

Variational message passing organizes mean-field updates locally on a factor graph.

Instead of deriving one large global expression, each factor sends information to neighboring variables. The messages contain expected natural parameters or sufficient statistics.

A variable node combines incoming messages to form its updated variational distribution.

The conceptual rule is:

$$
\boxed{
\text{updated natural parameter}
=
\text{sum of expected contributions from neighboring factors}.
}
$$

This is valuable because it makes inference modular:

- change one local model factor;
- update only the corresponding message rule;
- reuse the rest of the inference engine.

## 7.9 What We Skip Here

The textbook gives general conjugate-exponential update equations. They are powerful for software automation but can be abstract on a first reading. For this course, the main lesson is:

> Conjugacy plus exponential-family structure turns global variational inference into local moment and message updates.

The detailed symbolic templates are optional reference material rather than required memorization.

---

# §8 Local Variational Bounds and Logistic Regression

> 📖 Textbook §10.5–§10.6

## 8.1 Why Mean-Field Factorization Is Sometimes Not Enough

Mean-field updates require expectations of the log joint. Sometimes a nonlinear term prevents those expectations from being computed in closed form.

Logistic regression is a classic example. Its likelihood contains

$$
\sigma(a)
=
\frac{1}{1+e^{-a}},
$$

and the Gaussian expectation of $\ln\sigma(a)$ is not available in a simple closed form.

A **local variational method** replaces the difficult function by a tractable bound containing an adjustable parameter $\xi$.

## 8.2 Tangent Bounds from Convexity and Concavity

For a convex function $f(x)$, the tangent line is a global lower bound:

$$
\boxed{
f(x)
\geq
f(\xi)+f'(\xi)(x-\xi).
}
$$

For a concave function, the tangent line is a global upper bound.

> ![Figure 10.10](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_10__textbook_fig_10_10__p493.png)
>
> *Figure 10.10 (Textbook Fig. 10.10, p. 493): Local variational bounds constructed from tangents. An adjustable parameter chooses where the bound touches the original function.*

The variational parameter $\xi$ is not a model parameter. It exists only to make the bound as tight as possible.

## 8.3 The Logic of a Local Bound

Suppose a difficult likelihood factor satisfies

$$
f(a)\geq g(a,\xi)
$$

for every $a$ and every allowed $\xi$.

Then replacing $f(a)$ by $g(a,\xi)$ gives a lower bound on the likelihood or log likelihood. We optimize both:

- the approximate posterior $q$;
- the local parameters $\xi$.

The algorithm alternates:

1. fix $\xi$, update $q$;
2. fix $q$, tighten the bound by updating $\xi$;
3. repeat.

This resembles EM: each block update improves a common lower bound.

## 8.4 Quadratic Lower Bound for the Logistic Sigmoid

A particularly useful bound is

$$
\boxed{
\sigma(a)
\geq
\sigma(\xi)
\exp
\left
\{
\frac{a-\xi}{2}
-
\lambda(\xi)(a^2-\xi^2)
\right
\},
}
$$

where

$$
\lambda(\xi)
=
\frac{\tanh(\xi/2)}{4\xi}.
$$

The bound is tight at

$$
a=\pm\xi.
$$

> ![Figure 10.12](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_12__textbook_fig_10_12__p496.png)
>
> *Figure 10.12 (Textbook Fig. 10.12, p. 496): A quadratic-exponential lower bound on the logistic sigmoid. Adjusting $\xi$ changes where the bound touches the sigmoid.*

## 8.5 Why a Quadratic Bound Helps

Let

$$
a_n
=(2t_n-1)\mathbf{w}^T\boldsymbol{\phi}_n,
$$

where $t_n\in\{0,1\}$. The logistic likelihood for one point is

$$
p(t_n\mid\mathbf{w})
=
\sigma(a_n).
$$

After applying the bound, the logarithm contains terms that are:

- linear in $a_n$;
- quadratic in $a_n$.

Because $a_n$ is linear in $\mathbf{w}$, the bound becomes quadratic in $\mathbf{w}$. A Gaussian prior multiplied by an exponential quadratic produces a Gaussian posterior approximation.

This is the entire computational reason for the bound:

$$
\text{nonconjugate logistic likelihood}
\quad\Longrightarrow\quad
\text{quadratic lower bound}
\quad\Longrightarrow\quad
\text{Gaussian update}.
$$

## 8.6 Variational Posterior for Logistic Regression

Assume a Gaussian prior

$$
p(\mathbf{w})
=
\mathcal{N}(\mathbf{w}\mid\mathbf{m}_0,\mathbf{S}_0).
$$

After applying one local bound per data point, the approximate posterior is

$$
q(\mathbf{w})
=
\mathcal{N}(\mathbf{w}\mid\mathbf{m}_N,\mathbf{S}_N).
$$

The precision update is

$$
\boxed{
\mathbf{S}_N^{-1}
=
\mathbf{S}_0^{-1}
+
2\sum_{n=1}^{N}
\lambda(\xi_n)
\boldsymbol{\phi}_n\boldsymbol{\phi}_n^T.
}
$$

The mean update is

$$
\boxed{
\mathbf{m}_N
=
\mathbf{S}_N
\left[
\mathbf{S}_0^{-1}\mathbf{m}_0
+
\sum_{n=1}^{N}
\left(t_n-\frac{1}{2}\right)
\boldsymbol{\phi}_n
\right].
}
$$

The structure resembles Gaussian linear regression, but each data point receives a curvature contribution controlled by $\lambda(\xi_n)$.

## 8.7 Updating the Local Parameters

The optimal local parameter satisfies

$$
\boxed{
\xi_n^2
=
\boldsymbol{\phi}_n^T
\left(
\mathbf{S}_N
+
\mathbf{m}_N\mathbf{m}_N^T
\right)
\boldsymbol{\phi}_n.
}
$$

Why does the second moment appear? Because

$$
\mathbb{E}
\left[
(\mathbf{w}^T\boldsymbol{\phi}_n)^2
\right]
=
\boldsymbol{\phi}_n^T
\mathbb{E}[\mathbf{w}\mathbf{w}^T]
\boldsymbol{\phi}_n,
$$

and

$$
\mathbb{E}[\mathbf{w}\mathbf{w}^T]
=
\mathbf{S}_N
+
\mathbf{m}_N\mathbf{m}_N^T.
$$

Thus $\xi_n$ adapts to the current posterior uncertainty in the classifier score.

## 8.8 The Alternating Algorithm

1. Initialize $\xi_1,\ldots,\xi_N$.
2. Compute $\lambda(\xi_n)$ for each point.
3. Update $\mathbf{S}_N$ and $\mathbf{m}_N$.
4. Update each $\xi_n$ using the posterior second moment.
5. Evaluate the variational bound.
6. Repeat until convergence.

Each complete block update tightens or improves the same lower bound.

## 8.9 Geometric Interpretation

> ![Figure 10.13](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_13__textbook_fig_10_13__p502.png)
>
> *Figure 10.13 (Textbook Fig. 10.13, p. 502): Bayesian logistic regression under the variational approximation. Multiple sampled weight vectors give different decision boundaries, visualizing posterior uncertainty rather than only one separating line.*

A point estimate gives one decision boundary. A posterior over $\mathbf{w}$ gives a distribution over decision boundaries.

Near well-supported regions, sampled boundaries are similar. In uncertain regions, they vary more. This is the practical value of Bayesian classification: the model can express uncertainty about the boundary itself.

## 8.10 What Not to Overemphasize

The Jaakkola–Jordan bound is historically important and still conceptually useful, but modern implementations may instead use:

- stochastic gradients;
- automatic differentiation;
- numerical quadrature;
- augmentation methods;
- black-box variational inference.

The lasting lesson is broader than this particular sigmoid bound:

> When a nonlinear factor breaks conjugacy, construct or learn a tractable surrogate objective that preserves a valid bound.

---

# §9 Expectation Propagation

> 📖 Textbook §10.7; introduction-focused treatment

## 9.1 Motivation

Variational inference usually minimizes

$$
\mathrm{KL}(q\Vert p).
$$

Expectation propagation, or EP, uses local projections associated with the reverse direction

$$
\mathrm{KL}(p\Vert q).
$$

This produces different approximation behavior, especially for multimodal or skewed distributions.

## 9.2 Moment Matching from Reverse KL

Suppose $q(\mathbf{z})$ belongs to an exponential family:

$$
q(\mathbf{z})
=
h(\mathbf{z})g(\boldsymbol{\eta})
\exp\left
\{
\boldsymbol{\eta}^T\mathbf{u}(\mathbf{z})
\right
\}.
$$

Minimizing

$$
\mathrm{KL}(p\Vert q)
$$

with respect to the natural parameters $\boldsymbol{\eta}$ gives

$$
\boxed{
\mathbb{E}_{q}[\mathbf{u}(\mathbf{z})]
=
\mathbb{E}_{p}[\mathbf{u}(\mathbf{z})].
}
$$

Thus the optimal exponential-family approximation matches the expected sufficient statistics.

For a Gaussian approximation, the sufficient statistics include

$$
\mathbf{z}
\quad\text{and}\quad
\mathbf{z}\mathbf{z}^T.
$$

Therefore moment matching means matching:

- the mean;
- the covariance.

## 9.3 A One-Dimensional Illustration

> ![Figure 10.14](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_14__textbook_fig_10_14__p508.png)
>
> *Figure 10.14 (Textbook Fig. 10.14, p. 508): Comparison of variational and expectation-propagation-style Gaussian approximations. The reverse-KL/moment-matching approximation is broader and covers more of the target mass.*

The figure reinforces the earlier KL-direction discussion:

- variational $\mathrm{KL}(q\Vert p)$ tends to avoid placing mass in low-density regions;
- EP's local reverse-KL projection tends to preserve moments and cover mass.

## 9.4 Factorizing the Target into Sites

Suppose the target distribution can be written up to normalization as

$$
p(\boldsymbol{\theta})
\propto
\prod_{i=1}^{M}
f_i(\boldsymbol{\theta}).
$$

EP approximates each difficult factor by a tractable site:

$$
q(\boldsymbol{\theta})
\propto
\prod_{i=1}^{M}
\widetilde f_i(\boldsymbol{\theta}).
$$

The product is chosen to belong to a tractable exponential family, often Gaussian.

## 9.5 The Four Core EP Steps

To update site $j$:

### Step 1: remove the old site

Define the cavity distribution

$$
\boxed{
q^{\setminus j}(\boldsymbol{\theta})
\propto
\frac{q(\boldsymbol{\theta})}
{\widetilde f_j(\boldsymbol{\theta})}.
}
$$

The cavity asks:

> What does the current approximation believe without factor $j$?

### Step 2: restore the exact factor

Form the tilted distribution

$$
\boxed{
\widehat p_j(\boldsymbol{\theta})
\propto
f_j(\boldsymbol{\theta})
q^{\setminus j}(\boldsymbol{\theta}).
}
$$

This distribution contains the exact local factor and approximate information from all other factors.

### Step 3: project back to the tractable family

Find a new tractable approximation by minimizing

$$
\mathrm{KL}
\left(
\widehat p_j
\Vert
q_{\mathrm{new}}
\right).
$$

For an exponential family, this is done by matching moments.

### Step 4: update the site

Set

$$
\boxed{
\widetilde f_j^{\mathrm{new}}(\boldsymbol{\theta})
\propto
\frac{q_{\mathrm{new}}(\boldsymbol{\theta})}
{q^{\setminus j}(\boldsymbol{\theta})}.
}
$$

Then move to another site and repeat.

## 9.6 EP in Plain Language

A useful analogy is group discussion.

- The global approximation $q$ summarizes all current opinions.
- Remove one participant's old contribution: cavity distribution.
- Let that participant provide the exact local information: tilted distribution.
- Compress the result back into the agreed summary format: moment matching.
- Replace the participant's old contribution with the new one.

EP repeatedly refines local approximations while keeping a tractable global form.

## 9.7 EP Compared with Mean-Field VI

| Property | Mean-Field VI | EP |
|----------|---------------|----|
| Main KL direction | $\mathrm{KL}(q\Vert p)$ | Local $\mathrm{KL}(p\Vert q)$ projections |
| Main operation | Expected log joint | Moment matching |
| Approximation structure | Often factorizes variables | Often factorizes likelihood/site terms |
| Typical uncertainty | Can be narrow/mode-seeking | Often broader/mass-covering |
| Convergence | Coordinate ELBO updates are monotonic | Convergence is not guaranteed in general |
| Implementation | Often simple for conjugate models | Can require difficult tilted moments |

## 9.8 Damping and Stability

EP updates can oscillate or diverge. A common stabilization is damping:

$$
\boldsymbol{\eta}_{\mathrm{site}}^{\mathrm{new}}
\leftarrow
(1-\rho)
\boldsymbol{\eta}_{\mathrm{site}}^{\mathrm{old}}
+
\rho
\boldsymbol{\eta}_{\mathrm{site}}^{\mathrm{proposed}},
$$

where

$$
0<\rho\leq 1.
$$

A smaller $\rho$ makes each update more conservative.

## 9.9 Relationship to Belief Propagation

On suitable graphical models, EP can be understood as a generalization of message passing. Loopy belief propagation appears as a special case under particular choices of approximating families and projections.

The shared idea is local computation:

- factors exchange compressed information;
- global inference emerges from repeated local updates.

## 9.10 A Textbook Performance Comparison

> ![Figure 10.17](./CoursePR2026/Fig/Chapter_10/lecture_fig_10_17__textbook_fig_10_17__p514.png)
>
> *Figure 10.17 (Textbook Fig. 10.17, p. 514): One problem-specific comparison of EP, variational inference, and Laplace approximation. The figure illustrates that accuracy and computational cost depend strongly on the target problem; it should not be interpreted as a universal ranking.*

This figure is useful as a caution:

> There is no single approximate-inference method that is always best.

Method choice depends on:

- posterior geometry;
- required accuracy;
- available computation;
- whether uncertainty calibration matters;
- whether reliable convergence diagnostics exist.

---

# §10 Modern Machine-Learning Connections

## 10.1 From Classical Variational Bayes to Variational Autoencoders

Classical mean-field VI often introduces one variational factor for each latent variable in a fixed data set:

$$
q(\mathbf{Z})
=
\prod_n q_n(\mathbf{z}_n).
$$

For a large data set, storing and optimizing separate parameters for every $q_n$ can be expensive.

A variational autoencoder uses **amortized inference**:

$$
q_\boldsymbol{\phi}(\mathbf{z}\mid\mathbf{x}),
$$

where a neural network with shared parameters $\boldsymbol{\phi}$ predicts the approximate posterior parameters for each input.

The ELBO for one observation becomes

$$
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x})
=
\mathbb{E}_{q_\boldsymbol{\phi}(\mathbf{z}\mid\mathbf{x})}
[\ln p_\boldsymbol{\theta}(\mathbf{x}\mid\mathbf{z})]
-
\mathrm{KL}
\left(
q_\boldsymbol{\phi}(\mathbf{z}\mid\mathbf{x})
\Vert
p(\mathbf{z})
\right).
$$

This is the same variational principle:

- reconstruction term encourages the latent code to explain the observation;
- KL term regularizes the approximate posterior toward the prior.

The main difference is computational: neural networks and stochastic gradients replace hand-derived coordinate updates.

## 10.2 Coordinate Ascent versus Gradient-Based VI

| Classical CAVI | Modern Gradient-Based VI |
|----------------|--------------------------|
| Closed-form factor updates | Gradient updates of variational parameters |
| Often conjugate exponential-family models | Handles nonconjugate and neural models |
| Usually full-batch | Often minibatch stochastic optimization |
| Easy monotonicity when updates are exact | No strict monotonicity per minibatch step |
| Model-specific algebra | More reusable automatic differentiation |

Both maximize an ELBO. They differ mainly in how the objective is optimized.

## 10.3 Reparameterization Idea

Suppose

$$
q_\boldsymbol{\phi}(\mathbf{z}\mid\mathbf{x})
=
\mathcal{N}
\left(
\mathbf{z}
\mid
\boldsymbol{\mu}_\boldsymbol{\phi}(\mathbf{x}),
\operatorname{diag}(\boldsymbol{\sigma}^2_\boldsymbol{\phi}(\mathbf{x}))
\right).
$$

Instead of sampling $\mathbf{z}$ directly, write

$$
\mathbf{z}
=
\boldsymbol{\mu}_\boldsymbol{\phi}(\mathbf{x})
+
\boldsymbol{\sigma}_\boldsymbol{\phi}(\mathbf{x})
\odot
\boldsymbol{\epsilon},
$$

where

$$
\boldsymbol{\epsilon}
\sim
\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

Now randomness is isolated in $\boldsymbol{\epsilon}$, allowing gradients to pass through $\boldsymbol{\mu}_\boldsymbol{\phi}$ and $\boldsymbol{\sigma}_\boldsymbol{\phi}$.

This technique is not the focus of PRML Chapter 10, but it is a direct modern continuation of the same variational objective.

## 10.4 Mean-Field Limitations in Deep Models

A diagonal Gaussian approximate posterior is a modern version of a mean-field assumption. It is scalable, but it cannot represent arbitrary correlations or multiple modes.

More expressive families include:

- full-covariance Gaussians;
- low-rank plus diagonal covariance;
- normalizing flows;
- mixture variational distributions;
- hierarchical variational models;
- implicit distributions.

Increasing expressiveness can reduce approximation bias, but it increases optimization and computation cost.

## 10.5 Posterior Collapse as an ELBO Phenomenon

In powerful latent-variable decoders, the model may learn to ignore $\mathbf{z}$, giving

$$
q_\boldsymbol{\phi}(\mathbf{z}\mid\mathbf{x})
\approx
p(\mathbf{z}).
$$

Then the KL term becomes small, but the latent representation carries little information about $\mathbf{x}$. This is called posterior collapse.

The phenomenon illustrates an important lesson:

> Maximizing an ELBO does not automatically guarantee that the learned latent variables have the semantic structure we desire.

The objective, model capacity, optimization path, and data all matter.

## 10.6 Signal-Processing Connections

For EE applications, variational inference can be interpreted as iterative soft estimation.

Examples include:

- estimating a channel while also estimating symbol uncertainty;
- estimating source signals while also estimating noise precision;
- separating multiple sources with uncertain assignments;
- tracking hidden states with uncertain dynamics;
- learning sparse coefficients with uncertain scale parameters.

The common pattern is:

$$
\text{estimate one hidden block}
\quad\text{using expectations of the others}.
$$

This is closely related in spirit to soft interference cancellation, iterative decoding, and message-passing algorithms.

## 10.7 Choosing an Approximate-Inference Method

A practical decision table is:

| Situation | Reasonable Starting Point |
|-----------|---------------------------|
| Conjugate model with moderate size | Mean-field CAVI |
| Large data set, local latent variables | Stochastic/amortized VI |
| Posterior near one smooth mode | Laplace approximation |
| Accurate local moments are available and mass coverage matters | EP |
| Strong multimodality and enough computation | Sampling or richer variational families |
| Need only a point estimate | MAP/optimization may be sufficient |

The first question should always be:

> Which posterior information is actually needed by the downstream task?

A precise mean may be enough for one task. Another task may require calibrated tails, multiple modes, or model evidence.

---

# §11 Chapter Summary and Bridge to Sampling Methods

## 11.1 Conceptual Summary

This chapter can be summarized in one sentence:

> Approximate inference replaces an intractable posterior computation by a tractable optimization or projection problem over probability distributions.

The key ideas are:

| Topic | Main Lesson |
|-------|-------------|
| Intractable inference | Bayes' rule may be easy to write but hard to compute. |
| Variational family | Choose a tractable family $\mathcal{Q}$ to represent approximate uncertainty. |
| ELBO | $\ln p(\mathbf{X})=\mathcal{L}(q)+\mathrm{KL}(q\Vert p)$. |
| Mean-field approximation | Factorize $q$ and optimize one factor at a time. |
| Mean-field update | $\ln q_j^*=\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]+\text{const}$. |
| KL direction | $\mathrm{KL}(q\Vert p)$ tends to be mode-seeking; $\mathrm{KL}(p\Vert q)$ tends to cover mass. |
| Variational GMM | Replaces point estimates by distributions and can deactivate unsupported components. |
| Local bounds | Replace a difficult nonlinear factor by a tractable bound with optimized local parameters. |
| Variational logistic regression | A quadratic sigmoid bound produces a Gaussian posterior approximation. |
| EP | Remove one site, form a tilted distribution, match moments, and update the site. |
| Modern connection | VAEs and black-box VI optimize the same ELBO with neural networks and stochastic gradients. |

## 11.2 Mathematical Map

### Exact Posterior and Evidence

$$
p(\mathbf{Z}\mid\mathbf{X})
=
\frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{X})}
$$

$$
p(\mathbf{X})
=
\int p(\mathbf{X},\mathbf{Z})\,d\mathbf{Z}
$$

### ELBO Decomposition

$$
\ln p(\mathbf{X})
=
\mathcal{L}(q)
+
\mathrm{KL}
\left(
q(\mathbf{Z})
\Vert
p(\mathbf{Z}\mid\mathbf{X})
\right)
$$

$$
\mathcal{L}(q)
=
\mathbb{E}_q[\ln p(\mathbf{X},\mathbf{Z})]
+
H[q]
$$

### Mean-Field Approximation

$$
q(\mathbf{Z})
=
\prod_{i=1}^{M}q_i(\mathbf{Z}_i)
$$

$$
\ln q_j^*(\mathbf{Z}_j)
=
\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]
+
\text{const}
$$

### Variational Mixture Responsibilities

$$
r_{nk}
=
\frac{\rho_{nk}}{\sum_j\rho_{nj}}
$$

$$
N_k
=
\sum_n r_{nk}
$$

### Local Logistic Bound

$$
\sigma(a)
\geq
\sigma(\xi)
\exp
\left
\{
\frac{a-\xi}{2}
-
\lambda(\xi)(a^2-\xi^2)
\right
\}
$$

### EP Cycle

$$
q^{\setminus j}
\propto
\frac{q}{\widetilde f_j}
$$

$$
\widehat p_j
\propto
f_jq^{\setminus j}
$$

$$
\widetilde f_j^{\mathrm{new}}
\propto
\frac{q_{\mathrm{new}}}{q^{\setminus j}}
$$

## 11.3 Common Student Confusions

| Confusion | Clarification |
|-----------|---------------|
| “The ELBO is the posterior.” | No. The ELBO is a scalar objective; $q(\mathbf{Z})$ is the approximate posterior. |
| “The ELBO equals the evidence.” | Only when $q$ equals the exact posterior. Otherwise it is a lower bound. |
| “Mean-field says the real latent variables are independent.” | No. It imposes independence only in the approximating distribution. |
| “Coordinate ascent finds the global optimum.” | It improves the ELBO but can converge to a local optimum. |
| “A larger ELBO always means a better model.” | Only for comparable models, data, and variational treatments; a loose bound can complicate model comparison. |
| “Variational inference is just EM.” | EM is a special limiting case; variational inference can retain distributions over parameters. |
| “KL divergence is symmetric.” | No. Its direction changes approximation behavior. |
| “VI always underestimates variance.” | Mean-field $\mathrm{KL}(q\Vert p)$ often does, but this is not a universal theorem for all variational methods. |
| “EP always converges.” | No. Damping and careful scheduling may be needed. |
| “Approximate inference corrects a wrong model.” | No. It approximates inference under the chosen model. |

## 11.4 Minimal Board Derivation Plan

A concise blackboard sequence can be:

### Step 1: Start from Bayes' theorem

$$
p(\mathbf{Z}\mid\mathbf{X})
=
\frac{p(\mathbf{X},\mathbf{Z})}{p(\mathbf{X})}.
$$

### Step 2: Derive the ELBO decomposition

$$
\ln p(\mathbf{X})
=
\mathcal{L}(q)
+
\mathrm{KL}(q\Vert p).
$$

### Step 3: Introduce mean-field factorization

$$
q(\mathbf{Z})
=
\prod_i q_i(\mathbf{Z}_i).
$$

### Step 4: Derive one coordinate update

$$
\ln q_j^*(\mathbf{Z}_j)
=
\mathbb{E}_{-j}[\ln p(\mathbf{X},\mathbf{Z})]
+
\text{const}.
$$

### Step 5: Apply it to a simple Gaussian model

$$
q_\mu(\mu)
\leftrightarrow
q_\tau(\tau).
$$

### Step 6: Contrast KL directions

$$
\mathrm{KL}(q\Vert p)
\quad\text{versus}\quad
\mathrm{KL}(p\Vert q).
$$

### Step 7: Introduce EP as local moment matching

$$
\text{cavity}
\rightarrow
\text{tilted distribution}
\rightarrow
\text{moment matching}.
$$

## 11.5 Recommended Teaching Emphasis

For an EE undergraduate/graduate audience, a balanced lecture can allocate time as follows:

| Topic | Suggested Emphasis |
|-------|--------------------|
| Motivation and intractability | High |
| ELBO derivation | Very high |
| Mean-field update rule | Very high |
| Gaussian mean/precision example | High |
| Variational GMM | High, but simplify special functions |
| Variational linear regression | Moderate |
| Generic message-passing algebra | Low to moderate |
| Local sigmoid bound | Moderate |
| Full logistic-regression equations | Moderate, optional derivation details |
| EP | Conceptual introduction only |
| Alpha divergences and EP energy | Skip unless extra time |

## 11.6 Bridge to Chapter 11

Variational methods approximate a posterior with a tractable analytical distribution. Chapter 11 takes a different route:

> Instead of replacing the posterior by a simpler distribution, draw samples and approximate expectations by sample averages.

The Monte Carlo principle is

$$
\mathbb{E}_{p(\mathbf{z})}[f(\mathbf{z})]
\approx
\frac{1}{L}
\sum_{l=1}^{L}
f(\mathbf{z}^{(l)}),
$$

where

$$
\mathbf{z}^{(l)}\sim p(\mathbf{z}).
$$

The trade-off is:

| Variational Methods | Sampling Methods |
|---------------------|------------------|
| Fast deterministic optimization | Stochastic estimation |
| Systematic approximation bias | Sampling error that can shrink with more samples |
| Often easy convergence objective | Requires mixing and effective-sample diagnostics |
| Can be highly scalable | Can be computationally demanding |

The next chapter will study how to generate useful samples when direct sampling from the target distribution is impossible.

---

## End-of-Lecture Checklist

After this lecture, students should be able to:

1. explain why a formally defined posterior can still be computationally intractable;
2. derive the ELBO decomposition without skipping the normalization step;
3. explain why maximizing the ELBO minimizes $\mathrm{KL}(q\Vert p)$;
4. derive and interpret the mean-field coordinate update;
5. explain why factorized VI can underestimate uncertainty;
6. describe a variational Gaussian-mixture algorithm at the responsibility/update level;
7. explain how a local quadratic bound restores tractability in logistic regression;
8. describe the cavity, tilted, projection, and site-update stages of EP;
9. connect classical variational Bayes to modern amortized variational inference;
10. distinguish deterministic variational approximation from Monte Carlo sampling.
