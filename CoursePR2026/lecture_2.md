# Pattern Recognition and Machine Learning
## Chapter 2: Probability Distributions and Density Estimation

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 2 Probability Distributions (§2.1-§2.5)

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Binary Variables](#1-binary-variables)
3. [§2 Multinomial Variables](#2-multinomial-variables)
4. [§3 The Gaussian Distribution](#3-the-gaussian-distribution)
5. [§4 The Exponential Family](#4-the-exponential-family)
6. [§5 Nonparametric Methods](#5-nonparametric-methods)
7. [§6 Chapter Summary, Figure Checklist, and Teaching Flow](#6-chapter-summary-figure-checklist-and-teaching-flow)

---

## Notation and Variable Definitions

Chapter 1 introduced probability as the language of uncertainty. Chapter 2 asks a more concrete question:

> Once we agree to model uncertainty probabilistically, **which probability distributions should we use, how do we estimate their parameters, and how do priors change the answer when data are limited?**

This chapter is therefore the bridge between abstract probability rules and the probabilistic models used throughout the rest of the course.

### Generic Data and Distribution Notation

> **Teaching focus.** Do not treat this section as a dictionary of symbols. For Chapter 2, highlight the notation that tells students how a distributional model is built from data:
>
> $$
> \mathcal{D}=\{x_n\}_{n=1}^N
> \quad\longrightarrow\quad
> p(x\mid \boldsymbol{\theta})
> \quad\longrightarrow\quad
> \ln p(\mathcal{D}\mid \boldsymbol{\theta})
> \quad\longrightarrow\quad
> \mathbb{E}[x],\operatorname{var}[x],\operatorname{cov}[\mathbf{x}].
> $$
>
> The key message is: **data are summarized by counts or moments, distributions encode assumptions, and likelihood connects the two.** In class, put most emphasis on $\mathcal{D}$, $N$, $D$, $K$, $p(x\mid\boldsymbol{\theta})$, log likelihood, and the three moment quantities.

| Symbol | Definition |
|--------|------------|
| $x$ | A random variable. In §2.1 it is binary; in §2.3 it may be continuous. |
| $\mathbf{x}$ | A $D$-dimensional random vector. |
| $\mathcal{D}$ | Observed data set. Often $\mathcal{D}=\{x_1,\ldots,x_N\}$. |
| $N$ | Number of independent observations. |
| $D$ | Dimensionality of a continuous input vector. |
| $K$ | Number of states/classes/categories. |
| $p(x\mid \boldsymbol{\theta})$ | Probability distribution with parameter vector $\boldsymbol{\theta}$. |
| $\ln p(\mathcal{D}\mid \boldsymbol{\theta})$ | Log likelihood of parameters given data. |
| $\mathbb{E}[x]$ | Expectation of $x$. |
| $\operatorname{var}[x]$ | Variance of $x$. |
| $\operatorname{cov}[\mathbf{x}]$ | Covariance matrix of a vector random variable. |

### Binary, Multinomial, and Conjugate Priors

| Symbol | Definition |
|--------|------------|
| $x\in\{0,1\}$ | Binary random variable. |
| $\mu$ | Bernoulli probability of $x=1$. |
| $m$ | Number of observations equal to 1, for example number of heads in coin flips. |
| $\operatorname{Bern}(x\mid\mu)$ | Bernoulli distribution. |
| $\operatorname{Bin}(m\mid N,\mu)$ | Binomial distribution over the count $m$. |
| $\operatorname{Beta}(\mu\mid a,b)$ | Beta distribution, a prior/posterior distribution over $\mu$. |
| $\mathbf{x}$ | 1-of-$K$ coded vector for a categorical outcome. Exactly one entry equals 1. |
| $\boldsymbol{\mu}=(\mu_1,\ldots,\mu_K)^T$ | Vector of category probabilities. Each $\mu_k$ is nonnegative, and all probabilities sum to one: $\sum_k\mu_k=1$. |
| $m_k$ | Number of observations assigned to category $k$. |
| $\operatorname{Dir}(\boldsymbol{\mu}\mid\boldsymbol{\alpha})$ | Dirichlet distribution, a prior/posterior distribution over a multinomial probability vector. |
| $\alpha_k$ | Dirichlet concentration parameter for class $k$. |

### Gaussian and Related Distributions

| Symbol | Definition |
|--------|------------|
| $\mathcal{N}(x\mid\mu,\sigma^2)$ | Univariate Gaussian with mean $\mu$ and variance $\sigma^2$. |
| $\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})$ | Multivariate Gaussian with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$. |
| $\boldsymbol{\Sigma}$ | Covariance matrix. Its eigenvectors determine ellipsoid directions; eigenvalues determine squared axis lengths. |
| $\boldsymbol{\Lambda}=\boldsymbol{\Sigma}^{-1}$ | Precision matrix. |
| $\lambda$ | Scalar precision, typically $\lambda=1/\sigma^2$. |
| $\operatorname{St}(x\mid\mu,\lambda,\nu)$ | Student's $t$-distribution; useful as a heavy-tailed alternative to a Gaussian. |
| $\pi_k$ | Mixing coefficient for the $k$th mixture component. |
| $\gamma_k(\mathbf{x})$ | Responsibility of mixture component $k$ for data point $\mathbf{x}$. |

### Exponential Family and Nonparametric Density Estimation

| Symbol | Definition |
|--------|------------|
| $\boldsymbol{\eta}$ | Natural parameter of an exponential-family distribution. |
| $\mathbf{u}(x)$ | Sufficient-statistic vector. |
| $h(x)$ | Base measure in exponential-family form. |
| $g(\boldsymbol{\eta})$ | Normalization term in Bishop's exponential-family notation. |
| $\Delta$ | Histogram bin width. |
| $h$ | Kernel bandwidth / smoothing parameter. Do not confuse with the base measure $h(x)$. |
| $K$ | Number of nearest neighbours in K-NN methods. Context distinguishes it from number of classes. |
| $V$ | Volume of a small region around the point where density is estimated. |

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch.2 opening; §2.1-§2.5

## 0.1 What This Chapter Is Really About

Chapter 1 introduced probability rules, likelihood, priors, posteriors, and Bayesian prediction. Chapter 2 makes those ideas operational by studying probability distributions that appear again and again in pattern recognition.

The chapter can be read as answering five linked questions.

| Topic | Core Question | Long-Term Role in the Course |
|-------|---------------|-------------------------------|
| **Binary variables** | How do we model yes/no observations? | Classification labels, coin flips, Bernoulli likelihoods, logistic regression. |
| **Multinomial variables** | How do we model one-of-$K$ categorical outcomes? | Multi-class classification, word counts, mixture assignments. |
| **Gaussian distributions** | How do we model continuous real-valued vectors? | Linear regression, Gaussian processes, factor analysis, Kalman filters, graphical models. |
| **Exponential family** | Why do many different distributions have the same algebraic structure? | Sufficient statistics, conjugate priors, generalized linear models, variational inference. |
| **Nonparametric methods** | What can we do when we do not want to commit to a fixed parametric form? | Histograms, kernel density estimation, K-NN classification, bias-variance trade-offs. |

The recurring pattern is:

$$
\text{choose a distribution} \longrightarrow
\text{write the likelihood} \longrightarrow
\text{estimate parameters or place priors} \longrightarrow
\text{make predictions}.
$$

In formulas, this usually looks like:

$$
\begin{aligned}
\text{choose a distribution:}\quad
&p(x\mid\boldsymbol{\theta}) \\
\text{write the likelihood:}\quad
&p(\mathcal{D}\mid\boldsymbol{\theta})
=\prod_{n=1}^N p(x_n\mid\boldsymbol{\theta}) \\
\text{estimate parameters:}\quad
&\hat{\boldsymbol{\theta}}
=\arg\max_{\boldsymbol{\theta}}\ln p(\mathcal{D}\mid\boldsymbol{\theta}) \\
\text{make predictions:}\quad
&p(x_{\mathrm{new}}\mid\hat{\boldsymbol{\theta}}).
\end{aligned}
$$

For a Bayesian treatment, the estimate step is replaced by a posterior distribution $p(\boldsymbol{\theta}\mid\mathcal{D})$, and prediction averages over parameter uncertainty.

This chapter is not a catalogue of formulas. It teaches a way of thinking: probability distributions are **modeling assumptions**, and each assumption determines what kind of structure, uncertainty, and learning behavior the model can express.

## 0.2 Parametric versus Nonparametric Modeling

A **parametric model** assumes a fixed functional form with a finite set of parameters. For example, a Bernoulli distribution is determined by one parameter $\mu$, and a $D$-dimensional Gaussian is determined by a mean vector and covariance matrix.

A **nonparametric model** does not mean “no parameters”. It means that the effective complexity of the model can grow with the amount of data. Histograms, kernel density estimators, and nearest-neighbour methods keep the training data in a direct way and use local neighborhoods to estimate density or class membership.

The trade-off is important:

| Modeling Choice | Strength | Weakness |
|-----------------|----------|----------|
| Parametric | Compact, interpretable, often efficient; can generalize from limited data if the model is appropriate. | Can be badly biased if the chosen family is too restrictive. |
| Nonparametric | Flexible; can approximate complex distributions with enough data. | Data-hungry, computationally expensive, and vulnerable to the curse of dimensionality. |

## 0.3 The Central Theme: Counting Becomes Probability

A large part of this chapter can be understood through the idea of **counting**.

For a binary variable, the sufficient statistic is simply

$$
\sum_{n=1}^N x_n.
$$

For a categorical variable, the sufficient statistics are the class counts

$$
\mathbf{m}=(m_1,\ldots,m_K)^T.
$$

For a Gaussian, the sufficient statistics become the empirical mean and covariance-like quantities:

$$
\sum_{n=1}^N \mathbf{x}_n,
\qquad
\sum_{n=1}^N \mathbf{x}_n\mathbf{x}_n^T.
$$

The deeper idea is that many probability models do not need to remember every training example once certain summary statistics have been computed. This idea will reappear as **sufficient statistics** in the exponential family.

---

# §1 Binary Variables

> 📖 Textbook §2.1 Binary Variables; §2.1.1 The beta distribution

## 1.1 Bernoulli Distribution

A binary random variable takes one of two values:

$$
x\in\{0,1\}.
$$

We write $x=1$ for a “success” event and $x=0$ for a “failure” event. If the probability of success is $\mu$, then

$$
p(x=1\mid\mu)=\mu,
\qquad
p(x=0\mid\mu)=1-\mu.
$$

The Bernoulli distribution writes both cases in one compact expression:

$$
\operatorname{Bern}(x\mid\mu)=\mu^x(1-\mu)^{1-x},
\qquad 0\leq \mu\leq 1.
$$

This notation works because:

- if $x=1$, then $\mu^x(1-\mu)^{1-x}=\mu$;
- if $x=0$, then $\mu^x(1-\mu)^{1-x}=1-\mu$.

The mean and variance are

$$
\mathbb{E}[x]=\mu,
\qquad
\operatorname{var}[x]=\mu(1-\mu).
$$

The variance is largest when $\mu=0.5$, because then the outcome is most uncertain. The variance becomes zero when $\mu=0$ or $\mu=1$, because the outcome is deterministic.

## Textbook Exercise 2.1: Bernoulli Mean, Variance, and Entropy

> ![Textbook Exercise 2.1](./CoursePR2026/Fig/Chapter_2/lecture_ex_2_1__textbook_ex_2_1_p127.png)
>
> *Textbook Exercise 2.1 (p. 127): Verify normalization, mean, variance, and entropy for a Bernoulli random variable.*

For a Bernoulli variable, there are only two possible outcomes. So every
calculation is just a two-row table.

| $x$ | Probability |
|-----|-------------|
| $0$ | $1-\mu$ |
| $1$ | $\mu$ |

Normalization means the two probabilities add to one:

$$
p(0\mid\mu)+p(1\mid\mu)=(1-\mu)+\mu=1.
$$

The expectation is the probability-weighted average:

$$
\mathbb{E}[x]=0\cdot(1-\mu)+1\cdot\mu=\mu.
$$

Because $x^2=x$ when $x\in\{0,1\}$,

$$
\mathbb{E}[x^2]=\mu.
$$

Therefore

$$
\operatorname{var}[x]
=\mathbb{E}[x^2]-\mathbb{E}[x]^2
=\mu-\mu^2
=\mu(1-\mu).
$$

For a concrete value, suppose $\mu=0.7$. Then

$$
\mathbb{E}[x]=0.7,
\qquad
\operatorname{var}[x]=0.7\cdot0.3=0.21.
$$

The entropy is

$$
H[x]=-\mu\ln\mu-(1-\mu)\ln(1-\mu).
$$

For $\mu=0.7$,

$$
H[x]
=-0.7\ln0.7-0.3\ln0.3
\approx 0.611.
$$

The plain-English message is: when the coin is almost deterministic, entropy is
small; when $\mu$ is near $0.5$, entropy is large because the outcome is hard to
predict.

## 1.2 Maximum Likelihood for Bernoulli Data

Suppose we observe

$$
\mathcal{D}=\{x_1,\ldots,x_N\},
\qquad x_n\in\{0,1\},
$$

and assume the observations are independent and identically distributed. The likelihood is

$$
p(\mathcal{D}\mid\mu)
=\prod_{n=1}^N p(x_n\mid\mu)
=\prod_{n=1}^N \mu^{x_n}(1-\mu)^{1-x_n}.
$$

The log likelihood is

$$
\ln p(\mathcal{D}\mid\mu)
=\sum_{n=1}^N \{x_n\ln\mu+(1-x_n)\ln(1-\mu)\}.
$$

Let

$$
m=\sum_{n=1}^N x_n.
$$

Then $m$ is the number of observations for which $x=1$. The maximum likelihood estimate is

$$
\mu_{\mathrm{ML}}=\frac{1}{N}\sum_{n=1}^N x_n=\frac{m}{N}.
$$

This result is intuitive: the best estimate of the probability of heads is the observed fraction of heads.

However, this intuition can be dangerous for small data sets. If we flip a coin 3 times and observe 3 heads, then

$$
\mu_{\mathrm{ML}}=1.
$$

The model would assign probability zero to tails. This is an extreme form of over-fitting: the estimate exactly matches the small data set but gives an unreasonable prediction for future data.

## 1.3 Binomial Distribution

The Bernoulli distribution describes one binary trial. The binomial distribution describes the probability that exactly $m$ successes occur in $N$ independent trials:

$$
\operatorname{Bin}(m\mid N,\mu)
=\binom{N}{m}\mu^m(1-\mu)^{N-m},
$$

where

$$
\binom{N}{m}=\frac{N!}{(N-m)!m!}.
$$

The combinatorial term counts how many different sequences contain exactly $m$ successes.

For example, if $N=3$ and $m=2$, the possible success/failure sequences are

$$
110,
\quad 101,
\quad 011.
$$

There are $\binom{3}{2}=3$ such sequences.

The binomial mean and variance are

$$
\mathbb{E}[m]=N\mu,
\qquad
\operatorname{var}[m]=N\mu(1-\mu).
$$

> ![Figure 2.1](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_1__textbook_fig_2_1_p70.png)
>
> *Figure 2.1 (Textbook Fig. 2.1, p. 70): Binomial distribution for $N=10$ and $\mu=0.25$. The distribution is over the count $m$, not over a single binary variable. The most probable counts are near $N\mu=2.5$.*

The key teaching point is that a probability model can be placed either on an individual observation or on a statistic computed from many observations. Bernoulli models one trial; binomial models the count across many trials.

## 1.4 Beta Distribution: A Distribution over Probabilities

The Bernoulli and binomial distributions treat $\mu$ as a parameter. In Bayesian inference, we treat $\mu$ as uncertain and place a prior distribution over it.

Because $\mu$ must lie in $[0,1]$, a natural prior is the beta distribution:

$$
\operatorname{Beta}(\mu\mid a,b)
=\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1},
\qquad 0\leq\mu\leq 1.
$$

The gamma function generalizes the factorial:

$$
\Gamma(n)=(n-1)!
\quad\text{for positive integers } n.
$$

The beta distribution has mean and variance

$$
\mathbb{E}[\mu]=\frac{a}{a+b},
$$

$$
\operatorname{var}[\mu]
=\frac{ab}{(a+b)^2(a+b+1)}.
$$

The parameters $a$ and $b$ can be interpreted as **effective prior counts**:

- $a$ behaves like a prior count of successes plus one;
- $b$ behaves like a prior count of failures plus one;
- $a+b$ controls the strength or concentration of the prior.

> ![Figure 2.2](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_2__textbook_fig_2_2_p72.png)
>
> *Figure 2.2 (Textbook Fig. 2.2, p. 72): Beta distributions with different hyperparameters. Small values below 1 can concentrate mass near the boundaries; $a=b=1$ gives a uniform prior; larger values produce more concentrated beliefs about $\mu$.*

The beta distribution is flexible enough to express many qualitatively different beliefs:

| Hyperparameters | Shape | Interpretation |
|-----------------|-------|----------------|
| $a=b=1$ | Uniform | Before seeing data, every value of $\mu$ is equally plausible. |
| $a=b>1$ | Concentrated near $0.5$ | We believe the two outcomes are roughly balanced. |
| $a>b$ | Shifted toward 1 | We expect more successes than failures. |
| $a<b$ | Shifted toward 0 | We expect more failures than successes. |
| $a,b<1$ | U-shaped | We believe $\mu$ is likely near 0 or near 1, but not near the middle. |

## Textbook Exercise 2.6: Mean, Variance, and Mode of a Beta Distribution

> ![Textbook Exercise 2.6](./CoursePR2026/Fig/Chapter_2/lecture_ex_2_6__textbook_ex_2_6_p128.png)
>
> *Textbook Exercise 2.6 (p. 128): Use the beta normalization result to obtain the mean, variance, and mode.*

For teaching, it is useful to read a beta distribution as a soft belief about an
unknown probability $\mu$. The three most useful summary numbers are:

$$
\mathbb{E}[\mu]=\frac{a}{a+b},
$$

$$
\operatorname{var}[\mu]
=\frac{ab}{(a+b)^2(a+b+1)},
$$

and, when $a>1$ and $b>1$,

$$
\operatorname{mode}[\mu]=\frac{a-1}{a+b-2}.
$$

Now plug in a simple example. Let $a=3$ and $b=2$. Then

$$
\mathbb{E}[\mu]=\frac{3}{3+2}=0.6,
$$

$$
\operatorname{var}[\mu]
=\frac{3\cdot2}{(3+2)^2(3+2+1)}
=\frac{6}{150}
=0.04,
$$

and

$$
\operatorname{mode}[\mu]
=\frac{3-1}{3+2-2}
=\frac{2}{3}
\approx 0.667.
$$

So this prior says: values above $0.5$ are more plausible, the average belief is
$0.6$, and the most likely single value is about $0.667$. The mean and mode do
not have to be identical because the beta distribution can be asymmetric.

## 1.5 Conjugacy: Beta Prior plus Binomial Likelihood

The beta distribution is conjugate to the binomial likelihood. This means that if the prior is beta and the likelihood is binomial/Bernoulli, then the posterior is also beta.

Suppose the prior is

$$
p(\mu)=\operatorname{Beta}(\mu\mid a,b).
$$

If we observe $m$ successes in $N$ trials, then the likelihood is proportional to

$$
p(\mathcal{D}\mid\mu)\propto \mu^m(1-\mu)^{N-m}.
$$

Multiplying prior and likelihood gives

$$
p(\mu\mid\mathcal{D})
\propto
\mu^{m+a-1}(1-\mu)^{N-m+b-1}.
$$

Therefore

$$
p(\mu\mid\mathcal{D})
=
\operatorname{Beta}(\mu\mid a+m,b+N-m).
$$

This is one of the most important patterns in Bayesian learning:

$$
\text{posterior parameter}=\text{prior parameter}+\text{observed count}.
$$

The posterior mean is

$$
\mathbb{E}[\mu\mid\mathcal{D}]
=\frac{a+m}{a+b+N}.
$$

This lies between the prior mean $a/(a+b)$ and the maximum likelihood estimate $m/N$. When $N$ is small, the prior matters. As $N$ grows, the data dominate.

## 1.6 Sequential Bayesian Updating

Bayesian updating can be done one observation at a time. If we observe a success, we increase $a$ by 1. If we observe a failure, we increase $b$ by 1.

This means the posterior after one observation becomes the prior for the next observation:

$$
\operatorname{Beta}(a,b)
\xrightarrow{x=1}
\operatorname{Beta}(a+1,b),
$$

$$
\operatorname{Beta}(a,b)
\xrightarrow{x=0}
\operatorname{Beta}(a,b+1).
$$

> ![Figure 2.3](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_3__textbook_fig_2_3_p73.png)
>
> *Figure 2.3 (Textbook Fig. 2.3, p. 73): One step of sequential Bayesian inference. A beta prior is multiplied by a Bernoulli likelihood, producing a beta posterior. The shape changes in the direction supported by the new observation.*

This is the simplest example of online learning. We do not need to store all past observations; we only need the current values of $a$ and $b$.

The important conceptual distinction is:

| Method | Estimate / Belief after 3 heads |
|--------|----------------------------------|
| Maximum likelihood | $\mu_{\mathrm{ML}}=1$, so tails receive probability zero. |
| Bayesian with beta prior | $\mathbb{E}[\mu\mid\mathcal{D}]<1$ unless the prior itself is degenerate. |

Bayesian inference prevents extreme certainty from tiny data sets by preserving uncertainty over the parameter.

---

# §2 Multinomial Variables

> 📖 Textbook §2.2 Multinomial Variables; §2.2.1 The Dirichlet distribution

## 2.1 1-of-K Coding

A binary variable can represent two states. To represent one of $K$ states, we use a 1-of-$K$ coding scheme.

For example, if $K=4$, the four possible outcomes are

$$
(1,0,0,0)^T,
\quad
(0,1,0,0)^T,
\quad
(0,0,1,0)^T,
\quad
(0,0,0,1)^T.
$$

In general, $\mathbf{x}$ has elements $x_k\in\{0,1\}$ and satisfies

$$
\sum_{k=1}^K x_k=1.
$$

If category $k$ occurs with probability $\mu_k$, then

$$
\mu_k\geq 0,
\qquad
\sum_{k=1}^K\mu_k=1.
$$

The categorical distribution can be written compactly as

$$
p(\mathbf{x}\mid\boldsymbol{\mu})
=
\prod_{k=1}^K \mu_k^{x_k}.
$$

Only one factor is active because only one $x_k$ equals 1.

## 2.2 Maximum Likelihood for Categorical Data

Suppose we observe $N$ independent 1-of-$K$ vectors:

$$
\mathcal{D}=\{\mathbf{x}_1,\ldots,\mathbf{x}_N\}.
$$

The likelihood is

$$
p(\mathcal{D}\mid\boldsymbol{\mu})
=
\prod_{n=1}^N\prod_{k=1}^K \mu_k^{x_{nk}}.
$$

Define the count for category $k$ as

$$
m_k=\sum_{n=1}^N x_{nk}.
$$

Then the log likelihood becomes

$$
\ln p(\mathcal{D}\mid\boldsymbol{\mu})
=
\sum_{k=1}^K m_k\ln\mu_k.
$$

We maximize this subject to the simplex constraint $\sum_k\mu_k=1$. Using a Lagrange multiplier gives

$$
\mu_{k,\mathrm{ML}}=\frac{m_k}{N}.
$$

Again, maximum likelihood gives the observed frequency.

## 2.3 Multinomial Distribution over Counts

The multinomial distribution gives the probability of observing counts

$$
\mathbf{m}=(m_1,\ldots,m_K)^T,
\qquad
\sum_{k=1}^K m_k=N,
$$

after $N$ independent trials:

$$
\operatorname{Mult}(\mathbf{m}\mid N,\boldsymbol{\mu})
=
\frac{N!}{m_1!m_2!\cdots m_K!}
\prod_{k=1}^K\mu_k^{m_k}.
$$

This is the $K$-class analogue of the binomial distribution.

| Binary Case | Multiclass Case |
|-------------|-----------------|
| $x\in\{0,1\}$ | $\mathbf{x}$ is 1-of-$K$ coded. |
| Parameter $\mu$ | Parameter vector $\boldsymbol{\mu}$. |
| Count $m$ | Count vector $\mathbf{m}$. |
| Bernoulli / binomial | Categorical / multinomial. |
| Beta prior | Dirichlet prior. |

## 2.4 The Probability Simplex

The vector $\boldsymbol{\mu}$ cannot be arbitrary. It must satisfy nonnegativity and sum-to-one constraints. For $K=3$, the allowed values form a triangle called a **simplex**.

> ![Figure 2.4](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_4__textbook_fig_2_4_p77.png)
>
> *Figure 2.4 (Textbook Fig. 2.4, p. 77): For three probabilities $\mu_1,\mu_2,\mu_3$, the constraints $\mu_k\geq0$ and $\sum_k\mu_k=1$ restrict the parameter vector to a triangular simplex.*

A point near a vertex means one category has high probability. A point near the center means the categories have similar probabilities.

The simplex is important because many machine-learning outputs live on it. For example, the softmax layer of a neural classifier outputs a probability vector on a simplex.

## 2.5 Dirichlet Distribution

The Dirichlet distribution is a distribution over probability vectors:

$$
\operatorname{Dir}(\boldsymbol{\mu}\mid\boldsymbol{\alpha})
=
\frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_K)}
\prod_{k=1}^K \mu_k^{\alpha_k-1},
$$

where

$$
\alpha_0=\sum_{k=1}^K\alpha_k.
$$

The parameters $\alpha_k$ are concentration parameters. They act like pseudo-counts.

The mean of the Dirichlet distribution is

$$
\mathbb{E}[\mu_k]=\frac{\alpha_k}{\alpha_0}.
$$

When all $\alpha_k$ are equal, the distribution is symmetric. When they are small, probability mass tends to concentrate near the boundaries or vertices. When they are large, the distribution concentrates near its mean.

> ![Figure 2.5](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_5__textbook_fig_2_5_p78.png)
>
> *Figure 2.5 (Textbook Fig. 2.5, p. 78): Dirichlet distributions over the 3-class simplex for different concentration settings. The distribution can prefer corners, spread uniformly, or concentrate around the center.*

## 2.6 Dirichlet-Multinomial Conjugacy

The Dirichlet distribution is conjugate to the multinomial distribution.

If the prior is

$$
p(\boldsymbol{\mu})
=
\operatorname{Dir}(\boldsymbol{\mu}\mid\boldsymbol{\alpha}),
$$

and the observed class-count vector is

$$
\mathbf{m}=(m_1,
\ldots,m_K)^T,
$$

then the posterior is

$$
p(\boldsymbol{\mu}\mid\mathcal{D})
=
\operatorname{Dir}(\boldsymbol{\mu}\mid\boldsymbol{\alpha}+\mathbf{m}).
$$

That is,

$$
\alpha_k^{\mathrm{posterior}}=\alpha_k^{\mathrm{prior}}+m_k.
$$

This exactly generalizes the beta-Bernoulli update.

| Model | Prior | Data Contribution | Posterior |
|-------|-------|-------------------|-----------|
| Bernoulli / binomial | $\operatorname{Beta}(a,b)$ | successes and failures | $\operatorname{Beta}(a+m,b+N-m)$ |
| Multinomial | $\operatorname{Dir}(\boldsymbol{\alpha})$ | class counts $\mathbf{m}$ | $\operatorname{Dir}(\boldsymbol{\alpha}+\mathbf{m})$ |

The predictive probability of category $k$ under the posterior is

$$
\mathbb{E}[\mu_k\mid\mathcal{D}]
=
\frac{\alpha_k+m_k}{\alpha_0+N}.
$$

This is a smoothed version of the empirical frequency $m_k/N$.

---

# §3 The Gaussian Distribution

> 📖 Textbook §2.3 The Gaussian Distribution (§2.3.1-§2.3.9)

## 3.1 Why the Gaussian Is Central

The Gaussian distribution is one of the most important probability distributions in machine learning.

There are several reasons:

1. Many real-valued measurement errors are approximately Gaussian.
2. The central limit theorem explains why sums or averages of many weakly dependent random effects often become approximately Gaussian.
3. Gaussian distributions have convenient algebraic properties: marginal and conditional distributions remain Gaussian.
4. Linear-Gaussian models can often be solved exactly.
5. Gaussian distributions provide a local quadratic approximation to many log densities.

> ![Figure 2.6](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_6__textbook_fig_2_6_p79.png)
>
> *Figure 2.6 (Textbook Fig. 2.6, p. 79): Histograms of averages of uniformly distributed random variables. As the number of averaged variables increases, the distribution becomes increasingly Gaussian-like.*

The univariate Gaussian is

$$
\mathcal{N}(x\mid\mu,\sigma^2)
=
\frac{1}{(2\pi\sigma^2)^{1/2}}
\exp\left\{-\frac{1}{2\sigma^2}(x-\mu)^2\right\}.
$$

The $D$-dimensional multivariate Gaussian is

$$
\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})
=
\frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}}
\exp\left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T
\boldsymbol{\Sigma}^{-1}
(\mathbf{x}-\boldsymbol{\mu})\right\}.
$$

The exponent contains the squared Mahalanobis distance:

$$
\Delta^2
=(\mathbf{x}-\boldsymbol{\mu})^T
\boldsymbol{\Sigma}^{-1}
(\mathbf{x}-\boldsymbol{\mu}).
$$

Unlike Euclidean distance, Mahalanobis distance accounts for scale and correlation.

## 3.2 Geometry of the Multivariate Gaussian

The covariance matrix can be eigen-decomposed as

$$
\boldsymbol{\Sigma}=\sum_{i=1}^D \lambda_i\mathbf{u}_i\mathbf{u}_i^T,
$$

where $\lambda_i$ are eigenvalues and $\mathbf{u}_i$ are orthonormal eigenvectors.

The contours of constant density are ellipsoids. Their axes are aligned with the eigenvectors of $\boldsymbol{\Sigma}$, and the axis lengths are proportional to $\sqrt{\lambda_i}$.

> ![Figure 2.7](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_7__textbook_fig_2_7_p81.png)
>
> *Figure 2.7 (Textbook Fig. 2.7, p. 81): The covariance eigenvectors define the principal axes of the Gaussian ellipsoid, and the eigenvalues determine the squared lengths along those axes.*

This figure is one of the most important geometric pictures in the course. It says:

- the mean $\boldsymbol{\mu}$ locates the center;
- eigenvectors determine directions of variation;
- eigenvalues determine the amount of variation;
- correlations appear as rotated ellipses.

## 3.3 Covariance Restrictions and Model Complexity

A full covariance matrix in $D$ dimensions has

$$
\frac{D(D+1)}{2}
$$

free parameters because it is symmetric.

This can be too many when $D$ is large. Common restrictions include:

| Covariance Form | Formula | Number of Parameters | Shape |
|-----------------|---------|----------------------|-------|
| Full | $\boldsymbol{\Sigma}$ | $D(D+1)/2$ | Rotated ellipsoid. |
| Diagonal | $\boldsymbol{\Sigma}=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_D^2)$ | $D$ | Axis-aligned ellipsoid. |
| Isotropic | $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$ | $1$ | Sphere / circle. |

## Textbook Exercise 2.21: Count the Parameters in a Symmetric Matrix

> ![Textbook Exercise 2.21](./CoursePR2026/Fig/Chapter_2/lecture_ex_2_21__textbook_ex_2_21_p131.png)
>
> *Textbook Exercise 2.21 (p. 131): Show that a real symmetric $D\times D$ matrix has $D(D+1)/2$ independent parameters.*

A covariance matrix is symmetric:

$$
\Sigma_{ij}=\Sigma_{ji}.
$$

This means the upper-right half and lower-left half contain duplicate
information. We only need to count:

- $D$ diagonal entries;
- $D(D-1)/2$ off-diagonal pairs.

Therefore the number of free parameters is

$$
D+\frac{D(D-1)}{2}
=\frac{D(D+1)}{2}.
$$

Concrete counts:

| Dimension $D$ | Full Covariance Parameters |
|---------------|----------------------------|
| 2 | $2\cdot3/2=3$ |
| 10 | $10\cdot11/2=55$ |
| 100 | $100\cdot101/2=5{,}050$ |

This is why covariance modeling quickly becomes a data problem. A diagonal
covariance in $D=100$ has only 100 parameters, but a full covariance has 5,050.
The full model can represent correlations, but it needs much more data to estimate
reliably.

> ![Figure 2.8](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_8__textbook_fig_2_8_p84.png)
>
> *Figure 2.8 (Textbook Fig. 2.8, p. 84): Full, diagonal, and isotropic covariance structures. Restrictions reduce parameters but also reduce the shapes the model can express.*

This is the first appearance of a major modeling trade-off:

$$
\text{flexibility} \quad \text{versus} \quad \text{statistical reliability}.
$$

A full covariance model is expressive but data-hungry. A diagonal or isotropic covariance model is less expressive but easier to estimate.

## 3.4 Conditional Gaussian Distributions

Now partition the vector into two parts:

$$
\mathbf{x}=\begin{pmatrix}\mathbf{x}_a\\ \mathbf{x}_b\end{pmatrix},
\qquad
\boldsymbol{\mu}=\begin{pmatrix}\boldsymbol{\mu}_a\\ \boldsymbol{\mu}_b\end{pmatrix}.
$$

Partition the covariance and precision matrices as

$$
\boldsymbol{\Sigma}
=
\begin{pmatrix}
\boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab}\\
\boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb}
\end{pmatrix},
\qquad
\boldsymbol{\Lambda}=\boldsymbol{\Sigma}^{-1}
=
\begin{pmatrix}
\boldsymbol{\Lambda}_{aa} & \boldsymbol{\Lambda}_{ab}\\
\boldsymbol{\Lambda}_{ba} & \boldsymbol{\Lambda}_{bb}
\end{pmatrix}.
$$

A remarkable property of the Gaussian is that the conditional distribution is also Gaussian:

$$
p(\mathbf{x}_a\mid\mathbf{x}_b)
=
\mathcal{N}(\mathbf{x}_a\mid\boldsymbol{\mu}_{a\mid b},\boldsymbol{\Sigma}_{a\mid b}).
$$

Using the precision matrix, the conditional mean and covariance are

$$
\boldsymbol{\mu}_{a\mid b}
=\boldsymbol{\mu}_a
-\boldsymbol{\Lambda}_{aa}^{-1}\boldsymbol{\Lambda}_{ab}(\mathbf{x}_b-\boldsymbol{\mu}_b),
$$

$$
\boldsymbol{\Sigma}_{a\mid b}
=\boldsymbol{\Lambda}_{aa}^{-1}.
$$

The conditional mean is a linear function of the observed variable $\mathbf{x}_b$. This is the algebraic basis of many linear-Gaussian prediction models.

## 3.5 Marginal Gaussian Distributions

The marginal distribution is also Gaussian:

$$
p(\mathbf{x}_a)=\mathcal{N}(\mathbf{x}_a\mid\boldsymbol{\mu}_a,\boldsymbol{\Sigma}_{aa}).
$$

The Gaussian is therefore closed under both conditioning and marginalization.

> ![Figure 2.9](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_9__textbook_fig_2_9_p90.png)
>
> *Figure 2.9 (Textbook Fig. 2.9, p. 90): A joint Gaussian over two variables. Conditioning on a fixed value of one variable gives a narrower conditional distribution, while marginalization integrates over the other variable.*

The difference between conditioning and marginalization is fundamental.

| Operation | Question Answered | Result |
|-----------|-------------------|--------|
| Marginalization | What is the distribution of $\mathbf{x}_a$ if we ignore $\mathbf{x}_b$? | $p(\mathbf{x}_a)$ |
| Conditioning | What is the distribution of $\mathbf{x}_a$ after observing $\mathbf{x}_b$? | $p(\mathbf{x}_a\mid\mathbf{x}_b)$ |

Conditioning usually reduces uncertainty; marginalization does not use the observed value and therefore does not gain that information.

## 3.6 Bayes' Theorem for Gaussian Variables

Consider a linear-Gaussian model:

$$
p(\mathbf{x})=\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Lambda}^{-1}),
$$

$$
p(\mathbf{y}\mid\mathbf{x})
=
\mathcal{N}(\mathbf{y}\mid\mathbf{A}\mathbf{x}+\mathbf{b},\mathbf{L}^{-1}).
$$

Then both the marginal distribution $p(\mathbf{y})$ and posterior distribution $p(\mathbf{x}\mid\mathbf{y})$ are Gaussian. This identity is reused throughout the book, including Bayesian linear regression, probabilistic PCA, factor analysis, and Kalman filtering.

The conceptual form is:

$$
\text{Gaussian prior} + \text{linear Gaussian observation model}
\Longrightarrow
\text{Gaussian posterior}.
$$

This is the continuous-variable analogue of beta-Bernoulli and Dirichlet-multinomial conjugacy.

## 3.7 Maximum Likelihood for the Gaussian

Given data $\mathcal{D}=\{\mathbf{x}_1,
\ldots,\mathbf{x}_N\}$ sampled independently from a Gaussian, the log likelihood is

$$
\ln p(\mathcal{D}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})
= -\frac{ND}{2}\ln(2\pi)
-\frac{N}{2}\ln|\boldsymbol{\Sigma}|
-\frac{1}{2}\sum_{n=1}^N
(\mathbf{x}_n-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}_n-\boldsymbol{\mu}).
$$

The maximum likelihood estimates are

$$
\boldsymbol{\mu}_{\mathrm{ML}}
=\frac{1}{N}\sum_{n=1}^N\mathbf{x}_n,
$$

$$
\boldsymbol{\Sigma}_{\mathrm{ML}}
=\frac{1}{N}\sum_{n=1}^N
(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})^T.
$$

The ML covariance estimate is biased. Its expectation is

$$
\mathbb{E}[\boldsymbol{\Sigma}_{\mathrm{ML}}]
=\frac{N-1}{N}\boldsymbol{\Sigma}.
$$

The usual unbiased estimate replaces $1/N$ by $1/(N-1)$:

$$
\widetilde{\boldsymbol{\Sigma}}
=\frac{1}{N-1}\sum_{n=1}^N
(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})^T.
$$

This is another example of a finite-sample effect: maximum likelihood can be systematically overconfident when estimating variance.

## Textbook Exercise 2.27: Add Independent Random Vectors

> ![Textbook Exercise 2.27](./CoursePR2026/Fig/Chapter_2/lecture_ex_2_27__textbook_ex_2_27_p132.png)
>
> *Textbook Exercise 2.27 (p. 132): For independent random vectors, the mean and covariance of the sum are the sums of the means and covariances.*

Let

$$
\mathbf{y}=\mathbf{x}+\mathbf{z},
$$

and assume $\mathbf{x}$ and $\mathbf{z}$ are independent. The mean is easy:

$$
\mathbb{E}[\mathbf{y}]
=\mathbb{E}[\mathbf{x}+\mathbf{z}]
=\mathbb{E}[\mathbf{x}]+\mathbb{E}[\mathbf{z}].
$$

Independence also makes the covariance matrices add:

$$
\operatorname{cov}[\mathbf{y}]
=\operatorname{cov}[\mathbf{x}]+\operatorname{cov}[\mathbf{z}].
$$

For a concrete calculation, suppose

$$
\mathbb{E}[\mathbf{x}]
=
\begin{bmatrix}
1\\
2
\end{bmatrix},
\qquad
\operatorname{cov}[\mathbf{x}]
=
\begin{bmatrix}
4 & 1\\
1 & 9
\end{bmatrix},
$$

and

$$
\mathbb{E}[\mathbf{z}]
=
\begin{bmatrix}
-1\\
3
\end{bmatrix},
\qquad
\operatorname{cov}[\mathbf{z}]
=
\begin{bmatrix}
1 & 0\\
0 & 4
\end{bmatrix}.
$$

Then

$$
\mathbb{E}[\mathbf{y}]
=
\begin{bmatrix}
1\\
2
\end{bmatrix}
+
\begin{bmatrix}
-1\\
3
\end{bmatrix}
=
\begin{bmatrix}
0\\
5
\end{bmatrix},
$$

and

$$
\operatorname{cov}[\mathbf{y}]
=
\begin{bmatrix}
4 & 1\\
1 & 9
\end{bmatrix}
+
\begin{bmatrix}
1 & 0\\
0 & 4
\end{bmatrix}
=
\begin{bmatrix}
5 & 1\\
1 & 13
\end{bmatrix}.
$$

The practical message is the same as in Exercise 1.10: independent uncertainty
adds in variance/covariance, not in standard deviation.

## 3.8 Sequential Mean Update and SGD Intuition

The sample mean can be updated sequentially. After seeing $N-1$ data points, suppose we have

$$
\boldsymbol{\mu}_{\mathrm{ML}}^{(N-1)}.
$$

After observing $\mathbf{x}_N$, the updated mean is

$$
\boldsymbol{\mu}_{\mathrm{ML}}^{(N)}
=
\boldsymbol{\mu}_{\mathrm{ML}}^{(N-1)}
+\frac{1}{N}\left(\mathbf{x}_N-\boldsymbol{\mu}_{\mathrm{ML}}^{(N-1)}\right).
$$

This formula has a simple interpretation:

> Move the old estimate a small step toward the new observation.

The step size $1/N$ decreases over time, so later observations have smaller individual influence.

For this course, the important modern-ML connection is stochastic optimization.
Many deep-learning algorithms also update parameters by repeatedly moving a
current estimate a small step using one mini-batch:

$$
\text{new estimate}
=
\text{old estimate}
+
\text{step size}\times\text{new information}.
$$

The useful classroom message is simply: online learning updates an estimate
incrementally instead of recomputing everything from scratch.

## 3.9 Bayesian Inference for the Gaussian Mean

First consider the case where the variance $\sigma^2$ is known but the mean $\mu$ is unknown.

It is convenient to write precision

$$
\lambda=\frac{1}{\sigma^2}.
$$

Assume a Gaussian prior over $\mu$:

$$
p(\mu)=\mathcal{N}(\mu\mid\mu_0,\lambda_0^{-1}).
$$

Given data $\mathcal{D}$, the posterior is also Gaussian:

$$
p(\mu\mid\mathcal{D})
=\mathcal{N}(\mu\mid\mu_N,\lambda_N^{-1}),
$$

with

$$
\lambda_N=\lambda_0+N\lambda,
$$

$$
\mu_N
=\frac{\lambda_0\mu_0+N\lambda\mu_{\mathrm{ML}}}{\lambda_0+N\lambda}.
$$

The posterior mean is a precision-weighted average of the prior mean and the sample mean.

> ![Figure 2.12](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_12__textbook_fig_2_12_p99.png)
>
> *Figure 2.12 (Textbook Fig. 2.12, p. 99): Bayesian inference for the mean of a Gaussian. As more data are observed, the posterior becomes sharper and moves toward the true mean.*

This figure illustrates two Bayesian effects:

1. The posterior mean shifts toward the value supported by the data.
2. The posterior variance shrinks as data accumulate.

## 3.10 Student's t-Distribution

The Student's $t$-distribution is a heavy-tailed alternative to the Gaussian.
Intuitively, it behaves like a Gaussian near the center but is more tolerant of
large deviations.

The exact density is not important for this lecture. What matters is the shape:
Student's $t$ has heavier tails than a Gaussian. The parameter $\nu$ controls tail
heaviness; smaller $\nu$ means heavier tails, and as $\nu\to\infty$ the distribution
approaches a Gaussian.

> ![Figure 2.15](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_15__textbook_fig_2_15_p103.png)
>
> *Figure 2.15 (Textbook Fig. 2.15, p. 103): Student's $t$-distributions for different values of $\nu$. Smaller $\nu$ gives heavier tails; as $\nu\to\infty$, the distribution approaches a Gaussian.*

The most important practical property of the Student's $t$ is **heavy tails**. Heavy-tailed distributions assign more probability to extreme observations than Gaussians do.

## 3.11 Robustness to Outliers

Because a Gaussian assigns very low probability to outliers, maximum likelihood fitting with a Gaussian can be strongly distorted by a few extreme values. The Student's $t$ is more robust because it expects occasional large deviations.

> ![Figure 2.16](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_16__textbook_fig_2_16_p104.png)
>
> *Figure 2.16 (Textbook Fig. 2.16, p. 104): Comparison of Gaussian and Student's $t$ fitting. Outliers can pull the Gaussian fit strongly, while the Student's $t$ fit is less affected because of its heavier tails.*

The teaching message is broad:

> The choice of likelihood determines what the model regards as noise, and therefore determines how it reacts to unusual data.

A Gaussian likelihood treats squared error as natural. A heavy-tailed likelihood is more forgiving toward occasional large errors.

## 3.12 Mixtures of Gaussians

A single Gaussian is unimodal. It can describe one elliptical cluster, but it cannot describe multiple separated clusters well.

The Old Faithful data set is a classic example: eruption duration and waiting time form two clusters.

> ![Figure 2.21](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_21__textbook_fig_2_21_p110.png)
>
> *Figure 2.21 (Textbook Fig. 2.21, p. 110): A single Gaussian fails to capture the two-cluster structure of the Old Faithful data, whereas a mixture of two Gaussians gives a better representation.*

A mixture of Gaussians has the form

$$
p(\mathbf{x})
=
\sum_{k=1}^K \pi_k\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k),
$$

where

$$
0\leq\pi_k\leq 1,
\qquad
\sum_{k=1}^K\pi_k=1.
$$

Each component is a Gaussian, and $\pi_k$ is the prior probability of choosing component $k$.

> ![Figure 2.22](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_22__textbook_fig_2_22_p111.png)
>
> *Figure 2.22 (Textbook Fig. 2.22, p. 111): A one-dimensional Gaussian mixture. Individual Gaussian components are combined linearly to form a more complex density.*

> ![Figure 2.23](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_23__textbook_fig_2_23_p112.png)
>
> *Figure 2.23 (Textbook Fig. 2.23, p. 112): A two-dimensional mixture of three Gaussians. Mixtures can model multimodal densities and curved-looking density contours by combining simple components.*

The posterior probability that component $k$ generated a point $\mathbf{x}$ is called the responsibility:

$$
\gamma_k(\mathbf{x})
=
\frac{\pi_k\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}
{\sum_j \pi_j\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}.
$$

Mixtures introduce latent variables: the component identity is not observed. This is why maximum likelihood for Gaussian mixtures does not have the same simple closed-form solution as maximum likelihood for a single Gaussian. The EM algorithm in Chapter 9 will solve this problem.

---

# §4 The Exponential Family

> 📖 Textbook §2.4 The Exponential Family (§2.4.1-§2.4.3)

## 4.1 General Form

Many distributions in this chapter can be written in the exponential-family form

$$
p(x\mid\boldsymbol{\eta})
=h(x)g(\boldsymbol{\eta})
\exp\{\boldsymbol{\eta}^T\mathbf{u}(x)\}.
$$

Here:

| Term | Meaning |
|------|---------|
| $\boldsymbol{\eta}$ | Natural parameter. |
| $\mathbf{u}(x)$ | Sufficient-statistic vector. |
| $h(x)$ | Base measure. |
| $g(\boldsymbol{\eta})$ | Normalization factor ensuring the density integrates or sums to 1. |

The exponential family is important because it explains why different distributions have similar learning rules.

## 4.2 Bernoulli as an Exponential-Family Distribution

The Bernoulli distribution is

$$
p(x\mid\mu)=\mu^x(1-\mu)^{1-x}.
$$

Rewrite it as

$$
p(x\mid\mu)
=(1-\mu)\exp\left\{x\ln\frac{\mu}{1-\mu}\right\}.
$$

The natural parameter is

$$
\eta=\ln\frac{\mu}{1-\mu}.
$$

Solving for $\mu$ gives the logistic sigmoid:

$$
\mu=\sigma(\eta)=\frac{1}{1+\exp(-\eta)}.
$$

This is why the sigmoid function appears naturally in logistic regression: it maps an unconstrained real-valued natural parameter to a probability in $[0,1]$.

## 4.3 Multinomial and Softmax

For the multinomial/categorical distribution, the exponential-family form leads to the softmax function:

$$
\mu_k=\frac{\exp(\eta_k)}{\sum_j\exp(\eta_j)}.
$$

Again, the reason is constraint handling. The natural parameters $\eta_k$ can be arbitrary real numbers, while the resulting $\mu_k$ must be nonnegative and sum to one.

Thus:

| Output Constraint | Natural Mapping |
|------------------|-----------------|
| Binary probability $0\leq\mu\leq1$ | Sigmoid |
| Probability vector $\sum_k\mu_k=1$ | Softmax |

This will become central in Chapter 4 when we study probabilistic discriminative classifiers.

## 4.4 Gaussian as an Exponential-Family Distribution

The Gaussian distribution is also an exponential-family distribution. In the univariate case, we can write

$$
\mathcal{N}(x\mid\mu,\sigma^2)
=\frac{1}{(2\pi\sigma^2)^{1/2}}
\exp\left\{-\frac{x^2}{2\sigma^2}+\frac{\mu x}{\sigma^2}-\frac{\mu^2}{2\sigma^2}\right\}.
$$

The sufficient statistics are

$$
\mathbf{u}(x)=\begin{pmatrix}x\\x^2\end{pmatrix}.
$$

This explains why maximum likelihood estimation of a Gaussian depends on the data through sums of $x_n$ and $x_n^2$.

## 4.5 Maximum Likelihood and Sufficient Statistics

For an exponential-family distribution, the log likelihood for independent data is

$$
\ln p(\mathcal{D}\mid\boldsymbol{\eta})
=
\sum_{n=1}^N \ln h(x_n)
+N\ln g(\boldsymbol{\eta})
+\boldsymbol{\eta}^T\sum_{n=1}^N\mathbf{u}(x_n).
$$

The data enter the parameter-dependent part only through

$$
\sum_{n=1}^N\mathbf{u}(x_n).
$$

This is why $\mathbf{u}(x)$ is called the sufficient statistic.

A statistic is sufficient if, after computing it, the remaining details of the data do not provide additional information about the parameter within the assumed model family.

Examples:

| Distribution | Sufficient Statistics |
|-------------|----------------------|
| Bernoulli | $\sum_n x_n$ |
| Multinomial | counts $m_k=\sum_n x_{nk}$ |
| Gaussian with unknown mean and variance | $\sum_n x_n$, $\sum_n x_n^2$ |
| Multivariate Gaussian | $\sum_n \mathbf{x}_n$, $\sum_n \mathbf{x}_n\mathbf{x}_n^T$ |

---

# §5 Nonparametric Methods

> 📖 Textbook §2.5 Nonparametric Methods (§2.5.1-§2.5.2)

## 5.1 Why Nonparametric Methods?

Parametric models assume a fixed distributional form. A Gaussian assumes one elliptical cluster; a mixture of Gaussians assumes a chosen number of components; a Bernoulli assumes binary outcomes.

Nonparametric density estimation asks a different question:

> Can we estimate a probability density more directly from local data counts, without committing to one fixed finite-dimensional distribution family?

The basic idea is local counting.

If a small region $\mathcal{R}$ around $\mathbf{x}$ has volume $V$, and $K$ out of $N$ observations fall inside it, then

$$
p(\mathbf{x})\approx \frac{K}{NV}.
$$

Two major strategies follow:

| Strategy | Fixed Quantity | Adaptive Quantity | Method |
|----------|----------------|-------------------|--------|
| Fix region volume $V$ | Window size / bandwidth | Number of points $K$ inside | Histogram, kernel density estimation |
| Fix number of points $K$ | Number of neighbours | Region volume $V$ expands/shrinks | K-nearest-neighbour density estimation |

## 5.2 Histogram Density Estimation

In one dimension, divide the input space into bins of width $\Delta$. If $n_i$ points fall in bin $i$, the density estimate in that bin is

$$
p_i=\frac{n_i}{N\Delta}.
$$

The bin width $\Delta$ controls the smoothness.

> ![Figure 2.24](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_24__textbook_fig_2_24_p121.png)
>
> *Figure 2.24 (Textbook Fig. 2.24, p. 121): Histogram density estimation with different bin widths. Small bins give noisy estimates; large bins oversmooth the structure.*

Histograms are simple and intuitive, but they have important limitations:

1. The estimate is discontinuous at bin boundaries.
2. The result depends on the bin origin and bin width.
3. In high dimensions, the number of bins grows exponentially.
4. The density estimate is constant inside each bin.

This is a direct example of the bias-variance trade-off.

| Bin Width | Bias | Variance | Behavior |
|-----------|------|----------|----------|
| Very small | Low | High | Spiky and noisy. |
| Very large | High | Low | Smooth but may miss structure. |
| Intermediate | Balanced | Balanced | Often best empirically. |

## 5.3 Kernel Density Estimation

Kernel density estimation replaces hard histogram bins with smooth kernels centered at each data point.

Using a kernel function $k(u)$ and bandwidth $h$, the density estimate is

$$
p(\mathbf{x})
=
\frac{1}{N}
\sum_{n=1}^N
\frac{1}{h^D}
k\left(\frac{\mathbf{x}-\mathbf{x}_n}{h}\right).
$$

For a Gaussian kernel,

$$
k(\mathbf{u})
=\frac{1}{(2\pi)^{D/2}}
\exp\left(-\frac{1}{2}\mathbf{u}^T\mathbf{u}\right).
$$

The bandwidth $h$ controls smoothing.

> ![Figure 2.25](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_25__textbook_fig_2_25_p124.png)
>
> *Figure 2.25 (Textbook Fig. 2.25, p. 124): Kernel density estimation with different bandwidths. A bandwidth that is too small creates a noisy density; a bandwidth that is too large washes out multimodal structure.*

The analogy to histograms is:

| Histogram | Kernel Density Estimator |
|-----------|--------------------------|
| Hard bin membership | Smooth contribution from each point |
| Bin width $\Delta$ | Bandwidth $h$ |
| Piecewise constant density | Smooth density |
| Sensitive to bin boundaries | Less sensitive, but still bandwidth-dependent |

Kernel density estimation is flexible but computationally expensive because evaluating the density at a new point requires summing over training examples.

## 5.4 Nearest-Neighbour Density Estimation

Kernel methods fix the volume and count how many points fall inside. K-nearest-neighbour density estimation fixes the count $K$ and expands the region until it contains $K$ points.

If $V(\mathbf{x})$ is the volume of the region centered at $\mathbf{x}$ that contains the $K$ nearest neighbours, then

$$
p(\mathbf{x})\approx \frac{K}{N V(\mathbf{x})}.
$$

In dense regions, the required volume is small, so the density estimate is high. In sparse regions, the required volume is large, so the density estimate is low.

> ![Figure 2.26](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_26__textbook_fig_2_26_p125.png)
>
> *Figure 2.26 (Textbook Fig. 2.26, p. 125): K-nearest-neighbour density estimates. Small $K$ gives a noisy estimate; large $K$ gives a smoother estimate.*

K-nearest-neighbour density estimation is adaptive because the local volume changes with the density of data.

## 5.5 K-Nearest-Neighbour Classification

K-nearest-neighbour classification uses local class counts.

Suppose a test point $\mathbf{x}$ has a neighbourhood containing $K$ training points. Let $K_k$ be the number of those neighbours whose class is $C_k$. Then the posterior class probability is estimated as

$$
p(C_k\mid\mathbf{x})\approx\frac{K_k}{K}.
$$

The classifier chooses the class with the largest local count.

> ![Figure 2.27](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_27__textbook_fig_2_27_p126.png)
>
> *Figure 2.27 (Textbook Fig. 2.27, p. 126): In nearest-neighbour classification, the class of a new point is determined by the class labels of nearby training points.*

For $K=1$, the rule is especially simple: assign the test point to the class of the nearest training point.

> ![Figure 2.28](./CoursePR2026/Fig/Chapter_2/lecture_fig_2_28__textbook_fig_2_28_p126.png)
>
> *Figure 2.28 (Textbook Fig. 2.28, p. 126): K-NN decision regions for different values of $K$. Small $K$ produces highly fragmented decision regions; large $K$ produces smoother boundaries.*

## 5.6 Strengths and Weaknesses of Nonparametric Methods

Nonparametric methods are conceptually simple and often useful as baselines. However, they have clear limitations.

| Method | Strength | Weakness |
|--------|----------|----------|
| Histogram | Simple; fast after binning. | Discontinuous; sensitive to bin choices; poor in high dimensions. |
| Kernel density estimation | Smooth; flexible; easy to understand. | Bandwidth selection is difficult; evaluation can be expensive. |
| K-NN density | Adaptive local volume. | Density estimate may not integrate properly; sensitive to distance metric. |
| K-NN classification | Strong simple baseline; no training phase. | Stores all data; prediction can be expensive; suffers in high dimensions. |

The curse of dimensionality is the central obstacle. In high-dimensional spaces, local neighborhoods often contain few meaningful nearby points unless the data set is enormous or the data lie near a lower-dimensional manifold.

---

# §6 Chapter Summary, Figure Checklist, and Teaching Flow

> 📖 Textbook Ch.2 summary perspective

## 6.1 Conceptual Summary

Chapter 2 builds the distributional toolkit for the rest of the course.

The major ideas are:

1. **Maximum likelihood often equals observed frequency or sample statistics.**
   - Bernoulli: $\mu_{\mathrm{ML}}=m/N$.
   - Multinomial: $\mu_{k,\mathrm{ML}}=m_k/N$.
   - Gaussian: sample mean and sample covariance.

2. **Maximum likelihood can overfit in small samples.**
   - Three heads in three flips gives $\mu_{\mathrm{ML}}=1$.
   - Gaussian ML covariance is biased downward.

3. **Conjugate priors give analytically simple Bayesian updates.**
   - Beta prior plus Bernoulli/binomial likelihood gives beta posterior.
   - Dirichlet prior plus multinomial likelihood gives Dirichlet posterior.
   - Gaussian priors for Gaussian means illustrate the same "prior plus data" idea.

4. **Gaussian distributions are algebraically special.**
   - Marginals are Gaussian.
   - Conditionals are Gaussian.
   - Linear-Gaussian Bayesian updates are Gaussian.

5. **The exponential family unifies many distributions.**
   - It explains sufficient statistics.
   - It explains why sigmoid and softmax appear naturally.

6. **Nonparametric methods trade assumptions for data.**
   - They are flexible but data-hungry.
   - Smoothing parameters control the bias-variance trade-off.
   - High-dimensional spaces make local density estimation difficult.

## 6.2 Important Equivalences and Analogies

| Binary | Multiclass | Continuous Gaussian |
|--------|------------|--------------------|
| Bernoulli likelihood | Categorical likelihood | Gaussian likelihood |
| Count successes | Count classes | Compute sample mean/covariance |
| Beta prior | Dirichlet prior | Gaussian prior over a mean |
| Posterior adds counts | Posterior adds counts | Posterior adds sufficient statistics |
| Predict by posterior mean | Predict by posterior mean | Predict by posterior distribution |

A useful teaching slogan is:

$$
\text{Bayesian updating} = \text{prior pseudo-data} + \text{observed data}.
$$

This slogan is not literally true for every model, but it captures the intuition behind conjugate exponential-family models.

## 6.3 Figure Checklist

All displayed figures in this lecture are screenshots from the uploaded textbook PDF and are stored under:

```text
./CoursePR2026/Fig/Chapter_2/
```

| Lecture Figure | Textbook Figure | File |
|----------------|-----------------|------|
| Figure 2.1 | Textbook Fig. 2.1, p. 70 | `lecture_fig_2_1__textbook_fig_2_1_p70.png` |
| Figure 2.2 | Textbook Fig. 2.2, p. 72 | `lecture_fig_2_2__textbook_fig_2_2_p72.png` |
| Figure 2.3 | Textbook Fig. 2.3, p. 73 | `lecture_fig_2_3__textbook_fig_2_3_p73.png` |
| Figure 2.4 | Textbook Fig. 2.4, p. 77 | `lecture_fig_2_4__textbook_fig_2_4_p77.png` |
| Figure 2.5 | Textbook Fig. 2.5, p. 78 | `lecture_fig_2_5__textbook_fig_2_5_p78.png` |
| Figure 2.6 | Textbook Fig. 2.6, p. 79 | `lecture_fig_2_6__textbook_fig_2_6_p79.png` |
| Figure 2.7 | Textbook Fig. 2.7, p. 81 | `lecture_fig_2_7__textbook_fig_2_7_p81.png` |
| Figure 2.8 | Textbook Fig. 2.8, p. 84 | `lecture_fig_2_8__textbook_fig_2_8_p84.png` |
| Figure 2.9 | Textbook Fig. 2.9, p. 90 | `lecture_fig_2_9__textbook_fig_2_9_p90.png` |
| Figure 2.12 | Textbook Fig. 2.12, p. 99 | `lecture_fig_2_12__textbook_fig_2_12_p99.png` |
| Figure 2.15 | Textbook Fig. 2.15, p. 103 | `lecture_fig_2_15__textbook_fig_2_15_p103.png` |
| Figure 2.16 | Textbook Fig. 2.16, p. 104 | `lecture_fig_2_16__textbook_fig_2_16_p104.png` |
| Figure 2.21 | Textbook Fig. 2.21, p. 110 | `lecture_fig_2_21__textbook_fig_2_21_p110.png` |
| Figure 2.22 | Textbook Fig. 2.22, p. 111 | `lecture_fig_2_22__textbook_fig_2_22_p111.png` |
| Figure 2.23 | Textbook Fig. 2.23, p. 112 | `lecture_fig_2_23__textbook_fig_2_23_p112.png` |
| Figure 2.24 | Textbook Fig. 2.24, p. 121 | `lecture_fig_2_24__textbook_fig_2_24_p121.png` |
| Figure 2.25 | Textbook Fig. 2.25, p. 124 | `lecture_fig_2_25__textbook_fig_2_25_p124.png` |
| Figure 2.26 | Textbook Fig. 2.26, p. 125 | `lecture_fig_2_26__textbook_fig_2_26_p125.png` |
| Figure 2.27 | Textbook Fig. 2.27, p. 126 | `lecture_fig_2_27__textbook_fig_2_27_p126.png` |
| Figure 2.28 | Textbook Fig. 2.28, p. 126 | `lecture_fig_2_28__textbook_fig_2_28_p126.png` |

Added textbook exercise screenshots:

| Lecture Exercise | Textbook Exercise | File |
|------------------|-------------------|------|
| Exercise 2.1 | Textbook Ex. 2.1, p. 127 | `lecture_ex_2_1__textbook_ex_2_1_p127.png` |
| Exercise 2.6 | Textbook Ex. 2.6, p. 128 | `lecture_ex_2_6__textbook_ex_2_6_p128.png` |
| Exercise 2.21 | Textbook Ex. 2.21, p. 131 | `lecture_ex_2_21__textbook_ex_2_21_p131.png` |
| Exercise 2.27 | Textbook Ex. 2.27, p. 132 | `lecture_ex_2_27__textbook_ex_2_27_p132.png` |

## 6.4 Suggested Teaching Flow

A clean lecture sequence is:

1. Start with Bernoulli maximum likelihood and show why $3/3$ heads should not imply absolute certainty.
2. Introduce the beta prior as a distribution over $\mu$, then emphasize conjugacy and sequential updating.
3. Generalize binary outcomes to $K$-class outcomes using 1-of-$K$ coding, the multinomial, and the Dirichlet.
4. Move to continuous variables with the Gaussian. Spend time on covariance geometry, because this intuition is reused everywhere.
5. Teach conditional and marginal Gaussians as the algebraic foundation for future latent-variable and Bayesian regression models.
6. Present Gaussian Bayesian inference only through the unknown-mean case, emphasizing weighted averaging between prior and data.
7. Introduce Student's $t$ briefly as a heavy-tailed robust alternative to the Gaussian.
8. Introduce Gaussian mixtures as the first major example of latent-variable density modeling.
9. Use the exponential family to explain sufficient statistics, sigmoid, and softmax.
10. Close with nonparametric density estimation and K-NN to contrast flexible local methods against compact parametric models.

## 6.5 What Students Should Be Able to Do After This Lecture

After studying this chapter, students should be able to:

1. derive the ML estimate for a Bernoulli parameter;
2. explain why ML can overfit for tiny data sets;
3. update a beta prior after observing binary data;
4. derive the ML estimate for a multinomial probability vector under the simplex constraint;
5. update a Dirichlet prior using class counts;
6. interpret the covariance matrix of a multivariate Gaussian geometrically;
7. distinguish Gaussian marginalization from Gaussian conditioning;
8. state why Gaussian distributions are closed under linear-Gaussian Bayesian inference;
9. explain why Student's $t$ distributions are more robust to outliers than Gaussians;
10. identify sigmoid and softmax as natural mappings from exponential-family natural parameters to constrained probabilities;
11. explain the bias-variance role of histogram bin width, KDE bandwidth, and K-NN neighbourhood size.

## 6.6 Connection to Later Chapters

Chapter 2 is foundational for the rest of the course.

| Later Chapter | Dependency on Chapter 2 |
|---------------|-------------------------|
| Ch.3 Linear Models for Regression | Gaussian likelihood, Gaussian priors, Bayesian linear regression. |
| Ch.4 Linear Models for Classification | Bernoulli/multinomial distributions, sigmoid, softmax, exponential-family likelihoods. |
| Ch.5 Neural Networks | Output distributions and loss functions; softmax cross-entropy. |
| Ch.6 Kernel Methods | Gaussian processes and kernel density intuition. |
| Ch.8 Graphical Models | Conditional and marginal distributions, conjugacy, exponential-family structure. |
| Ch.9 Mixture Models and EM | Gaussian mixtures, responsibilities, latent component indicators. |
| Ch.10 Approximate Inference | Exponential-family form, variational factors, sufficient statistics. |
| Ch.11 Sampling Methods | Sampling from distributions and approximating expectations. |
| Ch.12 Continuous Latent Variables | Linear-Gaussian identities, Gaussian marginalization and conditioning. |
| Ch.13 Sequential Data | Sequential estimation, Gaussian updates, Kalman filtering. |

The main lesson is that probability distributions are not isolated formulas. They are reusable building blocks for models, inference algorithms, and learning objectives.
