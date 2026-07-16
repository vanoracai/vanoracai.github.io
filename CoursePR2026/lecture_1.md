---
layout: course
title: PRML Lecture 1
---

# Pattern Recognition and Machine Learning
## Chapter 1: Introduction: Probability, Decision Theory, and Information Theory

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 1 Introduction (§1.1-§1.6)

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Pattern Recognition and Machine Learning](#1-pattern-recognition-and-machine-learning)
3. [§2 Polynomial Curve Fitting（多项式曲线拟合）](#2-polynomial-curve-fitting)
4. [§3 Probability Theory Essentials（概率论基础）](#3-probability-theory-essentials)
5. [§4 Probabilistic Curve Fitting](#4-probabilistic-curve-fitting)
6. [§5 Model Selection and the Curse of Dimensionality](#5-model-selection-and-the-curse-of-dimensionality)
7. [§6 Decision Theory](#6-decision-theory)
8. [§7 Information Theory](#7-information-theory)
9. [§8 Chapter Summary and Bridge to Chapter 2](#8-chapter-summary-and-bridge-to-chapter-2)

---

## Notation and Variable Definitions

This first chapter introduces the notation that will be used throughout the course. The important shift is from thinking of machine learning as “fitting a curve” to thinking of it as **reasoning under uncertainty**.

> **Teaching focus.** Do not treat this section as a dictionary to memorize. In class, highlight the symbols that form the course pipeline:
>
> $$D_{\mathrm{train}} \rightarrow \mathbf{x},t \rightarrow y(\mathbf{x}) \rightarrow p(C_k\mid \mathbf{x}) \rightarrow R_k,\ L_{kj}.$$
>
> The key message is: **data define the learning problem, probability represents uncertainty, and decision theory turns uncertainty into action.**

### Data, Inputs, Targets, and Models

| Symbol | Definition |
|--------|------------|
| **$\mathbf{x}$** | **Input vector / feature vector.** In digit recognition, it can be a vector of pixel intensities. |
| $x$ | A scalar input variable, used in the polynomial curve-fitting example. |
| **$t$** | **Target variable.** In regression it is continuous; in classification it represents a class label. |
| **$D_{\mathrm{train}}$** | **Training data set**, usually $D_{\mathrm{train}}=\{(\mathbf{x}_n,t_n)\}_{n=1}^N$. |
| **$N$** | **Number of training examples.** Use this later when discussing generalization and overfitting. |
| $D$ | Dimensionality of the input vector $\mathbf{x}$. |
| $K$ | Number of classes in a classification problem. |
| **$y(\mathbf{x})$** | **Model output / prediction** as a function of the input. |
| **$\mathbf{w}$** | **Parameter vector of a model.** Learning usually means estimating or constraining $\mathbf{w}$. |
| $M$ | Polynomial order in §1.1. A polynomial of order $M$ has $M+1$ coefficients. |

### Probability and Statistics

| Symbol | Definition |
|--------|------------|
| $p(X)$ | Probability that discrete random variable $X$ takes a particular value. |
| $p(x)$ | Probability density of a continuous variable $x$. |
| $p(X,Y)$ | Joint probability of $X$ and $Y$. |
| **$p(X \mid Y)$** | **Conditional probability** of $X$ given $Y$. This is the basic language for updating uncertainty. |
| **$p(\mathbf{x}\mid C_k)$** | **Class-conditional density** for class $C_k$: what inputs from class $k$ tend to look like. |
| **$p(C_k\mid \mathbf{x})$** | **Posterior probability** of class $C_k$ after observing $\mathbf{x}$: what we believe after seeing the data. |
| **$\mathbb{E}[f]$** | **Expectation or average value** of a function $f$. This becomes the language of risk and loss. |
| $\operatorname{var}[x]$ | Variance of random variable $x$. |
| $\operatorname{cov}[\mathbf{x}]$ | Covariance matrix of random vector $\mathbf{x}$. |
| **$\mathrm{N}(x\mid \mu,\sigma^2)$** | **Univariate Gaussian density** with mean $\mu$ and variance $\sigma^2$. This connects noise assumptions to least squares. |
| $\mathrm{N}(\mathbf{x}\mid \boldsymbol{\mu},\boldsymbol{\Sigma})$ | Multivariate Gaussian density with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$. |
| $\beta$ | Precision parameter, equal to inverse variance: $\beta=1/\sigma^2$. |

### Decision Theory and Information Theory

| Symbol | Definition |
|--------|------------|
| **$C_k$** | **Class $k$.** |
| **$R_k$** | **Decision region** assigned to class $C_k$. This is where inference becomes a concrete decision rule. |
| **$L_{kj}$** | **Loss** incurred when the true class is $C_k$ but the decision is $C_j$. This explains why the most probable class is not always the best action. |
| **$H[x]$** | **Entropy** of random variable $x$: a measure of uncertainty. |
| **$\mathrm{KL}(p\Vert q)$** | **Kullback-Leibler divergence** from distribution $q$ to distribution $p$: a measure of distribution mismatch. |
| $I[x,y]$ | Mutual information between $x$ and $y$. |

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch.1 opening; §1.1-§1.6

## 0.1 What This Chapter Is Really About

This chapter is called “Introduction”, but it is not merely a motivational chapter. It establishes the three conceptual pillars of the whole course:

| Pillar | Main Question | Why It Matters |
|--------|---------------|----------------|
| **Probability theory（概率论）** | How do we represent uncertainty? | Real data contain noise, ambiguity, missing information, and finite-sample uncertainty. |
| **Decision theory** | How do we make an action or prediction once probabilities are known? | A posterior probability is not yet a decision. Different applications have different costs. |
| **Information theory** | How do we quantify uncertainty, surprise, and distributional mismatch? | Entropy, KL divergence, and mutual information appear repeatedly in density estimation, latent-variable models, variational inference, and neural networks. |

A useful way to read the chapter is the following:

1. The digit example tells us what pattern recognition is.
2. Polynomial curve fitting（多项式曲线拟合） gives a concrete miniature version of supervised learning.
3. Probability theory（概率论） explains why least squares, regularization, and Bayesian prediction are not arbitrary tricks.
4. Model selection and high-dimensional geometry explain why “more flexible” does not automatically mean “better”.
5. Decision theory separates estimating uncertainty from acting under uncertainty.
6. Information theory gives a language for uncertainty and distribution comparison.

## 0.2 The Big Picture: From Data to Decision

A pattern-recognition system typically follows this conceptual pipeline:

$$
\text{raw input} \longrightarrow \text{features} \longrightarrow \text{probabilistic model} \longrightarrow \text{decision rule} \longrightarrow \text{action}.
$$

For example, in handwritten digit recognition:

1. The raw input is an image of a handwritten digit.
2. The image is represented numerically as a vector of pixel values.
3. A model estimates class probabilities such as $p(C_3\mid \mathbf{x})$ or $p(C_8\mid \mathbf{x})$.
4. A decision rule chooses the most appropriate output class.
5. The final action could be storing a postal code, rejecting the example, or asking a human to check it.

The key lesson is that **learning** and **decision-making** are not the same thing. Learning gives us a model of uncertainty; decision theory tells us what to do with that uncertainty.

---

# §1 Pattern Recognition and Machine Learning

> 📖 Textbook Ch.1 opening, pp. 1-4; digit-recognition running example

## 1.1 What Is Pattern Recognition?

Pattern recognition is the automatic discovery of regularities in data, and the use of those regularities to make predictions, classify new examples, or uncover structure.

A simple but important example is handwritten digit recognition.

> ![Figure 1.1](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_1__textbook_fig_1_1__p2.png)
>
> *Figure 1.1 (Textbook Fig. 1.1, p. 2): Handwritten digit examples. The same digit class can appear in many different visual forms, so a fixed hand-written rule system is fragile. A learning system instead infers regularities from many examples.*

In the digit example, each image can be treated as a vector:

$$
\mathbf{x}=(x_1,x_2,\ldots,x_D)^T.
$$

If the image is $28\times 28$ pixels, then $D=784$. The target could be one of ten labels:

$$
t\in\{0,1,2,\ldots,9\}.
$$

The goal is not to memorize the training digits. The goal is to correctly classify new images that were not seen during training. This is called **generalization**.

## 1.2 Training Set, Test Set, and Generalization

A training set is a collection of examples:

$$
D_{\mathrm{train}}=\{(\mathbf{x}_1,t_1),(\mathbf{x}_2,t_2),\ldots,(\mathbf{x}_N,t_N)\}.
$$

A model uses the training set to choose its parameters. After training, it is evaluated on a test set that was not used for fitting. The test set estimates how well the model generalizes.

The central problem is this:

> We want the model to capture the underlying regularity in the data, not the accidental noise of the training sample.

This distinction will appear repeatedly in the polynomial example. A model can fit the training data perfectly and still perform badly on new data.

## 1.3 Supervised, Unsupervised, and Reinforcement Learning

The textbook distinguishes several broad learning settings.

| Learning Setting | Data Format | Goal | Examples |
|------------------|-------------|------|----------|
| **Supervised learning** | Inputs plus targets $(\mathbf{x},t)$ | Learn a mapping from input to target | Classification, regression |
| **Unsupervised learning** | Inputs only $\mathbf{x}$ | Discover structure in the input distribution | Clustering, density estimation, visualization |
| **Reinforcement learning** | Actions, states, rewards | Learn actions that maximize long-term reward | Control, game playing, robotics |

This course focuses mostly on supervised and unsupervised probabilistic models. Reinforcement learning is mentioned for context but is not the main topic.

## 1.4 Classification versus Regression

Supervised learning divides into two main cases.

### Classification

The target is discrete. For example:

$$
t\in\{0,1,2,\ldots,9\}.
$$

The model assigns an input to a class. A probabilistic classifier estimates:

$$
p(C_k\mid \mathbf{x}).
$$

### Regression

The target is continuous. For example, in curve fitting, the input $x$ is a scalar and the target $t$ is a noisy real-valued observation.

A regression model predicts a real value:

$$
y(x)\approx t.
$$

The polynomial curve-fitting example is a regression problem. It is intentionally simple, but it contains nearly all of the important ideas: model complexity, overfitting, regularization, likelihood, prior, posterior, and predictive uncertainty.

---

# §2 Polynomial Curve Fitting（多项式曲线拟合）

> 📖 Textbook §1.1 Example: Polynomial Curve Fitting（多项式曲线拟合）

## 2.1 The Curve-Fitting Setup

We start with a synthetic data set. The input $x$ lies in $[0,1]$, and the true underlying function is

$$
\sin(2\pi x).
$$

The observed target $t$ is noisy. This means the data are not exactly on the true curve.

> ![Figure 1.2](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_2__textbook_fig_1_2__p4.png)
>
> *Figure 1.2 (Textbook Fig. 1.2, p. 4): A small training data set generated from a sinusoidal function with added noise. The learning task is to infer a useful predictive rule from a limited number of noisy observations.*

The training data are

$$
\mathbf{x}=(x_1,\ldots,x_N)^T, \qquad \mathbf{t}=(t_1,\ldots,t_N)^T.
$$

We fit a polynomial of order $M$:

$$
y(x,\mathbf{w})=w_0+w_1x+w_2x^2+\cdots+w_Mx^M
=\sum_{j=0}^{M}w_jx^j.
$$

The word “order” means the highest power of $x$. A polynomial of order $M$ has $M+1$ coefficients.

| Polynomial Order | Model Form | Number of Coefficients | Flexibility |
|------------------|------------|------------------------|-------------|
| $M=0$ | $w_0$ | 1 | Constant only |
| $M=1$ | $w_0+w_1x$ | 2 | Straight line |
| $M=3$ | $w_0+w_1x+w_2x^2+w_3x^3$ | 4 | Smooth curve |
| $M=9$ | $\sum_{j=0}^{9}w_jx^j$ | 10 | Very flexible for $N=10$ points |

The model is nonlinear in $x$, but it is **linear in the parameters** $w_j$. This is important: many models in machine learning are nonlinear functions of inputs while still being linear in their trainable parameters.

## 2.2 Sum-of-Squares Error

To choose $\mathbf{w}$, we need a criterion. A common choice is the sum-of-squares error:

$$
E(\mathbf{w})=\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\mathbf{w})-t_n\}^2.
$$

The factor $1/2$ is included only to simplify derivatives. It does not change the minimizer.

> ![Figure 1.3](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_3__textbook_fig_1_3__p6.png)
>
> *Figure 1.3 (Textbook Fig. 1.3, p. 6): Sum-of-squares error measures the vertical discrepancy between the model prediction and each target value. Squaring penalizes large deviations strongly and gives a smooth differentiable objective.*

For the polynomial model, $E(\mathbf{w})$ is a quadratic function (二次函数) of the parameters. The reason is simple: the prediction is linear in $\mathbf{w}$, and the squared-error loss squares this linear expression. Therefore the error contains terms like $w_iw_j$, and the minimizing coefficients can be obtained by solving linear equations.

However, a small training error does not automatically mean good prediction on future data.

## 2.3 Model Complexity: Underfitting and Overfitting

The order $M$ controls the flexibility of the polynomial.

> ![Figure 1.4](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_4__textbook_fig_1_4__p7.png)
>
> *Figure 1.4 (Textbook Fig. 1.4, p. 7): Polynomial fits for several values of $M$. Low-order models are too rigid; an intermediate model captures the trend; a very high-order model can fit every training point while oscillating (震荡) badly between them.*

This figure gives the first major lesson of the course.

| Case | What Happens | Name |
|------|--------------|------|
| $M=0,1$ | The model is too simple to represent the sinusoidal pattern. | Underfitting |
| $M=3$ | The model captures the main trend without chasing every noisy point. | Good generalization |
| $M=9$ | The model interpolates the training data but oscillates (震荡) wildly. | Overfitting |

Overfitting is not merely “the model is complicated.” A complicated model is harmful when it uses its flexibility to explain noise rather than stable structure.

## 2.4 Training Error versus Test Error

To measure prediction performance, Bishop uses the root-mean-square error:

$$
E_{\mathrm{RMS}}=\sqrt{\frac{2E(\mathbf{w}^{\star})}{N}}.
$$

Here $\mathbf{w}^{\star}$ denotes the fitted parameter vector. The square root puts the error on the same scale as the target variable $t$.

> ![Figure 1.5](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_5__textbook_fig_1_5__p8.png)
>
> *Figure 1.5 (Textbook Fig. 1.5, p. 8): Training error decreases as model complexity increases, but test error can rise sharply when the model overfits. Generalization must be evaluated on data not used for fitting.*

The important pattern is:

- Training error usually decreases as model complexity increases.
- Test error often decreases at first and then increases.
- Therefore the model with the smallest training error is not necessarily the model with the best test performance.

This is a central theme in machine learning: **we care about expected future performance, not just empirical training performance**.

## 2.5 Data Size and Model Complexity

Overfitting depends not only on the model class but also on the amount of data.

> ![Figure 1.6](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_6__textbook_fig_1_6__p9.png)
>
> *Figure 1.6 (Textbook Fig. 1.6, p. 9): The same high-order polynomial behaves differently as the number of training points increases. More data can constrain a flexible model and reduce overfitting.*

A useful heuristic is:

> The more flexible the model, the more data are needed to constrain it.

This heuristic is not a theorem by itself, but it is extremely useful. A high-capacity model can generalize well when trained on enough representative data. With too little data, it may fit accidental noise.

## 2.6 Regularization: Penalizing Overly Large Coefficients

Another way to control overfitting is regularization. Instead of minimizing only training error, we minimize a penalized objective:

$$
\widetilde{E}(\mathbf{w})=\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\mathbf{w})-t_n\}^2+
\frac{\lambda}{2}\|\mathbf{w}\|^2.
$$

Here

$$
\|\mathbf{w}\|^2=\sum_{j=0}^{M}w_j^2.
$$

The parameter $\lambda\geq 0$ controls the strength of the penalty.

| $\lambda$ | Effect |
|----------|--------|
| Very small | Similar to unregularized least squares; may overfit. |
| Moderate | Suppresses extreme coefficients; often improves generalization. |
| Very large | Forces coefficients close to zero; may underfit. |

> ![Figure 1.7](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_7__textbook_fig_1_7__p10.png)
>
> *Figure 1.7 (Textbook Fig. 1.7, p. 10): Regularization can suppress the extreme oscillations (震荡) of a high-order polynomial. If the penalty is too strong, however, the model becomes overly flat and underfits.*

Regularization can be understood in three equivalent ways:

1. **Optimization view:** add a penalty for large coefficients.
2. **Complexity-control view:** reduce the effective flexibility of the model.
3. **Bayesian view:** impose a prior belief that very large coefficients are unlikely.

The Bayesian view will be derived in §4.4.

## 2.7 Choosing the Regularization Strength

The regularization parameter cannot be chosen by training error alone. If $\lambda$ is reduced, training error generally improves, but generalization may get worse.

> ![Figure 1.8](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_8__textbook_fig_1_8__p11.png)
>
> *Figure 1.8 (Textbook Fig. 1.8, p. 11): The regularization parameter controls the effective complexity of the fitted polynomial. A validation set or cross-validation procedure is needed to choose it in practice.*

In practice, model-complexity parameters such as $M$ and $\lambda$ are called **hyperparameters**. They are not usually learned by direct training-error minimization. They are selected using validation data, cross-validation, or Bayesian model comparison.

## 2.8 Key Lessons from the Polynomial Example

The polynomial example is simple, but it already contains the major themes of the course.

| Concept | How It Appears in Curve Fitting | General Machine-Learning Meaning |
|--------|----------------------------------|----------------------------------|
| Model | Polynomial $y(x,\mathbf{w})$ | A parametric function family |
| Parameters | Coefficients $\mathbf{w}$ | Quantities learned from training data |
| Error function | Sum-of-squares error | Training objective |
| Complexity | Polynomial order $M$ | Capacity/flexibility of model class |
| Generalization | Test error | Performance on unseen data |
| Overfitting | $M=9$ oscillates (震荡) | Fits noise rather than stable structure |
| Regularization | $\lambda\|\mathbf{w}\|^2/2$ | Bias toward simpler/smoother solutions |
| Hyperparameter | $M,\lambda$ | Chosen by validation or model selection |

---

# §3 Probability Theory Essentials（概率论基础）

> 📖 Textbook §1.2 Probability Theory（概率论） (§1.2.1-§1.2.4)

## 3.1 Why Probability Is Needed

Machine learning must deal with uncertainty. Uncertainty appears because:

1. measurements are noisy,
2. training data are finite,
3. the target may be inherently ambiguous,
4. model assumptions are imperfect,
5. future test points are not known during training.

Probability theory gives a consistent language for representing such uncertainty.

## 3.2 A Simple Fruit-Box Example

Bishop introduces probability using boxes of fruit.

> ![Figure 1.9](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_9__textbook_fig_1_9__p12.png)
>
> *Figure 1.9 (Textbook Fig. 1.9, p. 12): A simple fruit-box example for introducing priors, likelihoods, and posteriors. After seeing the fruit, our belief about the selected box changes.*

Let $B$ denote the selected box and $F$ denote the selected fruit.

Before observing the fruit, we have a prior distribution over boxes:

$$
p(B=b).
$$

The composition of each box gives a likelihood:

$$
p(F=f\mid B=b).
$$

After observing the fruit, we update our belief using Bayes' theorem:

$$
p(B=b\mid F=f)=\frac{p(F=f\mid B=b)p(B=b)}{p(F=f)}.
$$

This example is small, but it already contains the general Bayesian logic used throughout the course.

## Textbook Exercise 1.3: Bayes' Rule with Fruit Boxes

> ![Textbook Exercise 1.3](./CoursePR2026/Fig/Chapter_1/lecture_ex_1_3__textbook_ex_1_3__p58.png)
>
> *Textbook Exercise 1.3 (p. 58): Compute a marginal fruit probability and a posterior box probability.*

First compute the probability of selecting an apple. This is a total-probability
calculation: average over the possible boxes.

$$
p(\text{apple})
=p(r)p(\text{apple}\mid r)
+p(b)p(\text{apple}\mid b)
+p(g)p(\text{apple}\mid g).
$$

From the question:

$$
p(\text{apple})=0.2\cdot\frac{3}{10}
+0.2\cdot\frac{1}{2}
+0.6\cdot\frac{3}{10}
=0.34.
$$

Now suppose the observed fruit is an orange. We want the probability that the
box was green:

$$
p(g\mid \text{orange})
=\frac{p(\text{orange}\mid g)p(g)}{p(\text{orange})}.
$$

The denominator again averages over all boxes:

$$
p(\text{orange})
=0.2\cdot\frac{4}{10}
+0.2\cdot\frac{1}{2}
+0.6\cdot\frac{3}{10}
=0.36.
$$

So

$$
p(g\mid \text{orange})
=\frac{0.6\cdot 3/10}{0.36}
=\frac{0.18}{0.36}
=0.5.
$$

The important idea is not the arithmetic. The important idea is the direction:
before seeing the fruit we average over boxes; after seeing the fruit we update
which box is plausible.

## 3.3 Why Probability Matters in Pattern Recognition

At a high level, probability is the language we use when the input is incomplete, noisy, or ambiguous. A model usually cannot say "this is certainly class A" or "this curve is certainly correct." Instead, it should say how plausible different explanations are after seeing the data.

This viewpoint will appear repeatedly later in the course:

| Idea | Why it matters later |
|------|----------------------|
| Likelihood (似然) | Used for learning model parameters: after the data are fixed, compare which parameter values make those data more probable. Least squares will become maximum likelihood under Gaussian noise. |
| Posterior (后验) | Used for inference after seeing data: classify an input, update beliefs about parameters, and represent uncertainty. |
| Evidence (证据/归一化常数) | Used to normalize posterior probabilities, and later to compare models with different complexity. |
| Marginalization (边缘化) | Used to average over unknown variables or uncertain parameters instead of pretending they have one fixed value. |

So the purpose of this probability section is not to memorize rules in isolation. The purpose is to build the machinery for three recurring questions:

1. **Learning:** Which parameters make the data likely?
2. **Inference:** After observing data, what should we believe?
3. **Decision:** Given uncertainty, what action minimizes expected loss?

The sum rule (求和法则) and product rule (乘法法则) are the basic algebra behind these operations.

The **sum rule** removes a variable we do not care about by adding over its possible values:

$$
p(X)=\sum_Y p(X,Y).
$$

The **product rule** decomposes a joint probability into a marginal probability and a conditional probability:

$$
p(X,Y)=p(Y\mid X)p(X).
$$

Together, these rules lead directly to Bayes' theorem:

$$
p(Y\mid X)=\frac{p(X\mid Y)p(Y)}{p(X)}.
$$

## 3.4 Bayes' Theorem: Posterior Is Proportional to Likelihood Times Prior

Bayes' theorem is often remembered as:

$$
\text{posterior} = \frac{\text{likelihood}\times\text{prior}}{\text{evidence}}.
$$

More explicitly:

$$
p(Y\mid X)=\frac{p(X\mid Y)p(Y)}{p(X)}.
$$

| Term | Formula | Meaning |
|------|---------|---------|
| Prior (先验) | $p(Y)$ | Belief about $Y$ before observing $X$ |
| Likelihood (似然) | $p(X\mid Y)$ | How probable the observation would be if $Y$ were true |
| Evidence (证据/归一化常数) | $p(X)$ | Normalizing probability of the observation |
| Posterior (后验) | $p(Y\mid X)$ | Updated belief after observing $X$ |

The evidence $p(X)$ is called a normalizing probability because it makes the posterior probabilities over all possible $Y$ values sum to one. It is the total probability of observing $X$, considering every possible explanation $Y$.

The denominator can be computed by the sum rule:

$$
p(X)=\sum_Y p(X\mid Y)p(Y).
$$

For continuous variables, the sum is replaced by an integral (积分).

Use the **likelihood** when you want to evaluate or learn a model: given a possible hypothesis or parameter value $Y$, how likely is the observed data $X$? Use the **posterior** when you want to make inference or decisions after seeing data: given the observation $X$, which hypothesis $Y$ should we believe, and with what probability?

## 3.5 Marginal and Conditional Distributions

> ![Figure 1.11](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_11__textbook_fig_1_11__p16.png)
>
> *Figure 1.11 (Textbook Fig. 1.11, p. 16): A joint distribution contains enough information to compute marginal distributions and conditional distributions. Marginalization discards one variable; conditioning focuses on a subset of cases.*

The difference between marginalization and conditioning is essential.

### Marginalization

Marginalization answers:

> What is the distribution of $X$ if we ignore $Y$?

For discrete variables:

$$
p(X)=\sum_Y p(X,Y).
$$

For continuous variables, use an integral (积分):

$$
p(x)=\int p(x,y)\,dy.
$$

### Conditioning

Conditioning answers:

> What is the distribution of $X$ after we know $Y=y$?

For discrete variables:

$$
p(X\mid Y)=\frac{p(X,Y)}{p(Y)}.
$$

For continuous variables:

$$
p(x\mid y)=\frac{p(x,y)}{p(y)}.
$$

## 3.6 Independence

Two variables $X$ and $Y$ are independent if knowing one does not change the probability of the other:

$$
p(X\mid Y)=p(X).
$$

Equivalently,

$$
p(X,Y)=p(X)p(Y).
$$

Conditional independence is similar but depends on a third variable:

$$
p(X,Y\mid Z)=p(X\mid Z)p(Y\mid Z).
$$

Conditional independence will become extremely important in graphical models.

## 3.7 Probability Densities

For a continuous variable, the probability of an exact value is usually zero. Instead, we use a probability density $p(x)$.

> ![Figure 1.12](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_12__textbook_fig_1_12__p18.png)
>
> *Figure 1.12 (Textbook Fig. 1.12, p. 18): A probability density assigns probability mass to intervals. The probability of a small interval is approximately density times width.*

For a small interval $[x,x+\delta x]$,

$$
p(x\leq X\leq x+\delta x)\simeq p(x)\delta x.
$$

A valid density must satisfy:

$$
p(x)\geq 0,
$$

and

$$
\int_{-\infty}^{\infty}p(x)\,dx=1.
$$

The probability that $x$ lies in an interval $(a,b)$ is

$$
p(x\in(a,b))=\int_a^b p(x)\,dx.
$$

## 3.8 Change of Variables for Densities

Densities transform differently from ordinary functions. Suppose

$$
x=g(y).
$$

Then probabilities must be preserved:

$$
p_y(y)\,dy=p_x(x)\,dx.
$$

In words, the probability of a small interval in $y$-space should equal the probability of the corresponding small interval in $x$-space.

Therefore,

$$
p_y(y)=p_x(g(y))\left|\frac{dg(y)}{dy}\right|.
$$

For example, if

$$
x=2y,
$$

then a small interval of length $dy$ in $y$-space corresponds to an interval of length

$$
dx=2dy
$$

in $x$-space. Therefore the density is multiplied by $2$. This does not mean that probability has increased; it means that the same interval in $y$ covers a longer interval in $x$.

The absolute derivative is the one-dimensional Jacobian. In multiple dimensions, the corresponding term is the absolute determinant of the Jacobian matrix.

Here $p_x(x)$ means a probability density, not the probability of the exact value $x$. For a continuous variable, probability is assigned to an interval:

$$
P(x\leq X\leq x+\Delta x)\simeq p_x(x)\Delta x.
$$

So density tells us how much probability mass there is per unit length, area, or volume near a point.

This is a common source of mistakes. **A density is not itself a probability; probability is density times volume.**

## 3.9 Expectations, Variances, and Covariances

The expectation of a function $f(x)$ is its probability-weighted average.

Intuitively, expectation is the **average location** of a random quantity.
If $x$ represents exam scores, then $\mathbb{E}[x]$ is the average score.
If $f(x)$ is some quantity computed from $x$, then $\mathbb{E}[f(x)]$ is the
average value of that computed quantity.

For a discrete variable:

$$
\mathbb{E}[f]=\sum_x p(x)f(x).
$$

For a continuous variable:

$$
\mathbb{E}[f]=\int p(x)f(x)\,dx.
$$

In practice, if we have samples $x_1,\ldots,x_N$, we often approximate the expectation by the sample average:

$$
\mathbb{E}[f]\simeq \frac{1}{N}\sum_{n=1}^{N}f(x_n).
$$

The variance measures spread around the mean. A useful way to read the formula is:

1. Compute the distance from the mean: $x-\mathbb{E}[x]$.
2. Square this distance so that positive and negative deviations do not cancel.
3. Take the expectation, which gives the average squared deviation.

So variance answers the question:

> How far is $x$ from its mean, on average?

$$
\operatorname{var}[x]=\mathbb{E}\left[(x-\mathbb{E}[x])^2\right]
=\mathbb{E}[x^2]-\mathbb{E}[x]^2.
$$

The second expression,
$\mathbb{E}[x^2]-\mathbb{E}[x]^2$, is mainly a convenient computational form.
Conceptually, the first expression is more important: variance is the average
squared distance from the mean.

For two variables, covariance measures linear co-variation. It tells us whether
two variables tend to move together.

$$
\operatorname{cov}[x,y]=\mathbb{E}\left[(x-\mathbb{E}[x])(y-\mathbb{E}[y])\right].
$$

For example, suppose

- $x$ is study time;
- $y$ is exam score.

The term $x-\mathbb{E}[x]$ tells us whether study time is above or below average.
The term $y-\mathbb{E}[y]$ tells us whether the exam score is above or below
average.

If study time and exam score are often above average together, or below average
together, then the two deviations usually have the same sign. In the covariance
formula, the two deviations are multiplied:

$$
(x-\mathbb{E}[x])(y-\mathbb{E}[y]).
$$

When both deviations have the same sign, their product is positive. Therefore:

- $\operatorname{cov}[x,y]>0$: $x$ and $y$ tend to increase or decrease together.
- $\operatorname{cov}[x,y]<0$: when one is above average, the other tends to be below average.
- $\operatorname{cov}[x,y]\simeq 0$: there is little linear co-variation.

For a vector $\mathbf{x}$, the covariance matrix is

$$
\operatorname{cov}[\mathbf{x}]=\mathbb{E}\left[(\mathbf{x}-\mathbb{E}[\mathbf{x}])(\mathbf{x}-\mathbb{E}[\mathbf{x}])^T\right].
$$

If

$$
\mathbf{x}=
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix},
$$

then the covariance matrix collects all variances and covariances in one table:

$$
\operatorname{cov}[\mathbf{x}]
=
\begin{bmatrix}
\operatorname{var}[x_1] & \operatorname{cov}[x_1,x_2] & \operatorname{cov}[x_1,x_3]\\
\operatorname{cov}[x_2,x_1] & \operatorname{var}[x_2] & \operatorname{cov}[x_2,x_3]\\
\operatorname{cov}[x_3,x_1] & \operatorname{cov}[x_3,x_2] & \operatorname{var}[x_3]
\end{bmatrix}.
$$

The diagonal entries are variances: they describe how much each variable varies
by itself. The off-diagonal entries are covariances: they describe how pairs of
variables vary together.

A short summary is:

> Expectation is the average location. Variance is the spread around that
> location. Covariance is whether two variables move together. A covariance
> matrix puts all of these pairwise relationships into one matrix.

## Textbook Exercise 1.10: Add Independent Uncertainties

> ![Textbook Exercise 1.10](./CoursePR2026/Fig/Chapter_1/lecture_ex_1_10__textbook_ex_1_10__p59.png)
>
> *Textbook Exercise 1.10 (p. 59): Compute the mean and variance of a sum of independent variables.*

Use this as a quick calculation rule:

| Quantity | $x$ | $z$ |
|----------|-----|-----|
| Mean | $\mathbb{E}[x]=10$ | $\mathbb{E}[z]=3$ |
| Variance | $\operatorname{var}[x]=4$ | $\operatorname{var}[z]=9$ |

If $x$ and $z$ are independent, then

$$
\mathbb{E}[x+z]=10+3=13.
$$

The variances also add:

$$
\operatorname{var}[x+z]=4+9=13.
$$

So the standard deviation of the sum is

$$
\sqrt{13}\approx 3.61.
$$

The useful classroom message is: independent noise sources add in variance, not
in standard deviation. Two small uncertainties can combine into a noticeably
larger uncertainty.

## 3.10 Frequentist and Bayesian Views of Probability

Probability can be interpreted in two related but different ways.

| View | Probability Means | Parameters |
|------|-------------------|------------|
| Frequentist | Long-run frequency of repeatable events | Fixed but unknown quantities |
| Bayesian | Degree of uncertainty or belief | Random/uncertain quantities represented by distributions |

For example, suppose $\mathbf{w}$ is a model parameter.

In the frequentist view, $\mathbf{w}$ is fixed but unknown. We estimate it from data.

In the Bayesian view, uncertainty about $\mathbf{w}$ is represented by a distribution:

$$
p(\mathbf{w}\mid D_{\mathrm{train}}).
$$

This posterior distribution can be used to make predictions by averaging over parameter uncertainty.

## 3.11 The Gaussian Distribution

The Gaussian distribution is central because it is mathematically convenient, widely occurring, and closely connected to least squares.

The univariate Gaussian density is a Gaussian probability density for **one**
continuous variable. "Univariate" means one variable. For example, $x$ might be
one student's exam score, one measurement error, or one input feature.

We write it as $\mathrm{N}(x\mid \mu,\sigma^2)$. This means:

- the variable is $x$;
- the mean is $\mu$, which sets the center of the bell curve;
- the variance is $\sigma^2$, which controls how wide or narrow the bell curve is.

$$
\mathrm{N}(x\mid \mu,\sigma^2)
=\frac{1}{(2\pi\sigma^2)^{1/2}}
\exp\left\{-\frac{1}{2\sigma^2}(x-\mu)^2\right\}.
$$

> ![Figure 1.13](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_13__textbook_fig_1_13__p25.png)
>
> *Figure 1.13 (Textbook Fig. 1.13, p. 25): A univariate Gaussian is controlled by a mean $\mu$ and a standard deviation $\sigma$. The mean sets the center; the variance sets the spread.*

For a Gaussian,

$$
\mathbb{E}[x]=\mu,
$$

and

$$
\operatorname{var}[x]=\sigma^2.
$$

The multivariate Gaussian is the same idea, but for a **vector** instead of one
number. "Multivariate" means multiple variables.

For example, one data point might contain several features:

$$
\mathbf{x}=
\begin{bmatrix}
\text{height}\\
\text{weight}\\
\text{age}
\end{bmatrix}.
$$

Then the mean is no longer a single number. It becomes a mean vector
$\boldsymbol{\mu}$. The variance is also no longer a single number. It becomes a
covariance matrix $\boldsymbol{\Sigma}$, which describes both the spread of each
feature and how different features vary together.

$$
\mathrm{N}(\mathbf{x}\mid \boldsymbol{\mu},\boldsymbol{\Sigma})
=\frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}}
\exp\left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\}.
$$

The quadratic term

$$
(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})
$$

is the squared Mahalanobis distance.

Intuitively, this is a distance from $\mathbf{x}$ to the mean
$\boldsymbol{\mu}$, but it is not ordinary Euclidean distance. It measures
distance after accounting for the covariance structure of the data.

This matters because different directions may have different natural spreads.
Moving 10 units in a direction where the data usually varies a lot may not be
surprising. Moving 10 units in a direction where the data is usually very tight
may be very surprising. The Mahalanobis distance adjusts for this.

So the multivariate Gaussian assigns high density to points that are close to
the mean in this covariance-aware sense, and low density to points that are far
away in this covariance-aware sense.

This is different from ordinary Euclidean distance. Euclidean distance only asks
how far two points are geometrically:

$$
\|\mathbf{x}-\boldsymbol{\mu}\|^2
=(\mathbf{x}-\boldsymbol{\mu})^T(\mathbf{x}-\boldsymbol{\mu}).
$$

Mahalanobis distance instead asks how unusual the point is under the data
distribution:

$$
(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}).
$$

The difference is the matrix $\boldsymbol{\Sigma}^{-1}$. It rescales distance
according to the covariance. Directions with large variance are penalized less,
because variation in those directions is common. Directions with small variance
are penalized more, because variation in those directions is more surprising.

In short:

> Euclidean distance asks: how far is this point geometrically?
>
> Mahalanobis distance asks: how unusual is this point relative to the data
> distribution?

## 3.12 Likelihood for Gaussian Parameters

In the previous section, we introduced the Gaussian distribution. But in a real
problem, we usually do not know the correct mean $\mu$ or variance $\sigma^2$ in
advance.

So the question in this section is:

> Given observed data, which Gaussian distribution should we choose?

In other words, we want to estimate the Gaussian parameters from data. The mean
$\mu$ controls where the Gaussian is centered, and the variance $\sigma^2$
controls how wide it is. Maximum likelihood gives one standard way to choose
these parameters.

The basic idea is simple:

> Choose the parameters that make the observed data look most likely under the
> model.

Suppose we observe independent data $x_1,\ldots,x_N$ from a Gaussian
distribution. Here the data values are already observed, and we want to estimate
the unknown parameters $\mu$ and $\sigma^2$.

The likelihood is a function of the parameters:

$$
p(\mathbf{x}\mid \mu,\sigma^2)=\prod_{n=1}^{N}\mathrm{N}(x_n\mid \mu,\sigma^2).
$$

This formula says:

- $\mathbf{x}=(x_1,\ldots,x_N)$ is the whole observed data set.
- $\mathrm{N}(x_n\mid \mu,\sigma^2)$ is the Gaussian density value assigned to
  one data point $x_n$.
- $\prod_{n=1}^{N}$ means multiply these density values for all data points.

For example, if we have three observations, the product expands to

$$
p(\mathbf{x}\mid \mu,\sigma^2)
=\mathrm{N}(x_1\mid \mu,\sigma^2)
\times \mathrm{N}(x_2\mid \mu,\sigma^2)
\times \mathrm{N}(x_3\mid \mu,\sigma^2).
$$

The reason we multiply is the independence assumption. If data points are
independently generated from the same Gaussian distribution, then the density of
the whole data set is the product of the individual density values.

The goal of maximum likelihood is to choose the parameter values that make the
observed data look most plausible under the Gaussian model:

$$
(\mu_{\mathrm{ML}},\sigma^2_{\mathrm{ML}})
=\arg\max_{\mu,\sigma^2} p(\mathbf{x}\mid \mu,\sigma^2).
$$

In words, we ask:

> Which Gaussian curve gives high density to the data points we actually
> observed?

> ![Figure 1.14](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_14__textbook_fig_1_14__p26.png)
>
> *Figure 1.14 (Textbook Fig. 1.14, p. 26): The Gaussian likelihood is the product of the density values assigned to the observed data points. Maximum likelihood chooses parameters that make the observed data most probable under the model.*

Products of many small density values can be inconvenient to work with.
Therefore, it is easier to maximize the log likelihood:

$$
\ln p(\mathbf{x}\mid \mu,\sigma^2)
= -\frac{N}{2}\ln(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2.
$$

To find the maximum-likelihood estimates, we differentiate the log likelihood
with respect to the unknown parameters and set the derivatives to zero.

First, differentiate with respect to $\mu$. Only the squared-error term depends
on $\mu$:

$$
\frac{\partial}{\partial \mu}
\ln p(\mathbf{x}\mid \mu,\sigma^2)
=\frac{1}{\sigma^2}\sum_{n=1}^{N}(x_n-\mu).
$$

At the maximum, this derivative should be zero:

$$
\frac{1}{\sigma^2}\sum_{n=1}^{N}(x_n-\mu)=0.
$$

Since $\sigma^2>0$, this is equivalent to

$$
\sum_{n=1}^{N}(x_n-\mu)=0.
$$

Expanding the sum gives

$$
\sum_{n=1}^{N}x_n-N\mu=0,
$$

so the maximum-likelihood estimate of the mean is

$$
\mu_{\mathrm{ML}}=\frac{1}{N}\sum_{n=1}^{N}x_n,
$$

which is simply the sample average.

Next, differentiate with respect to the variance. To make the notation simpler,
let

$$
s=\sigma^2.
$$

The part of the log likelihood that depends on $s$ can be written as

$$
\ell(s)=-\frac{N}{2}\ln s-\frac{1}{2s}\sum_{n=1}^{N}(x_n-\mu)^2+\text{constant}.
$$

Let

$$
A=\sum_{n=1}^{N}(x_n-\mu)^2.
$$

Then

$$
\ell(s)=-\frac{N}{2}\ln s-\frac{A}{2s}+\text{constant}.
$$

Differentiating with respect to $s$ gives

$$
\frac{d\ell}{ds}
=-\frac{N}{2s}+\frac{A}{2s^2}.
$$

Set this derivative to zero:

$$
-\frac{N}{2s}+\frac{A}{2s^2}=0.
$$

Multiplying both sides by $2s^2$ gives

$$
-Ns+A=0,
$$

so

$$
s=\frac{A}{N}.
$$

Substituting back $s=\sigma^2$ and using $\mu_{\mathrm{ML}}$ gives

$$
\sigma^2_{\mathrm{ML}}=\frac{1}{N}\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2.
$$

So maximum likelihood chooses the sample mean as the Gaussian center and the
average squared distance from that center as the Gaussian variance.

## 3.13 Bias of the Maximum-Likelihood Variance Estimate

Core message:

> The maximum-likelihood variance estimate divides by $N$, but this tends to
> underestimate the true variance. The unbiased sample variance divides by
> $N-1$ instead.

In formulas,

$$
\sigma^2_{\mathrm{ML}}
=\frac{1}{N}\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2,
$$

whereas

$$
\widehat{\sigma}^2
=\frac{1}{N-1}\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2.
$$

The difference between $N$ and $N-1$ is the main point of this section.

The ML formula looks natural:

$$
\sigma^2_{\mathrm{ML}}
=\frac{1}{N}\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2.
$$

It computes the squared distances from the fitted mean and takes their average.
But there is a subtle issue. The mean $\mu_{\mathrm{ML}}$ is estimated from the
same data, so it is pulled toward the observed points.

Because the fitted mean is chosen to be close to the data, the distances
$x_n-\mu_{\mathrm{ML}}$ are slightly smaller than the distances to the true mean
$x_n-\mu$. Therefore the ML variance tends to underestimate the true variance.

The key calculation is based on the identity

$$
\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2
=
\sum_{n=1}^{N}(x_n-\mu)^2
-N(\mu_{\mathrm{ML}}-\mu)^2.
$$

This identity says:

> The squared distances to the fitted mean are smaller than the squared
> distances to the true mean.

Before taking expectations, let us clarify the symbols in the next few lines:

| Symbol | Meaning |
|--------|---------|
| $\mu$ | The true mean of the data-generating Gaussian. We do not know it in practice. |
| $\mu_{\mathrm{ML}}$ | The sample mean, used as the ML estimate of $\mu$. It changes from sample to sample, so it is a random variable. |
| $\sigma^2$ | The true variance of the data-generating Gaussian. |
| $N$ | Number of observed data points. |
| $\mathbb{E}[\cdot]$ | Expectation: average value over many possible repeated data sets. |
| $\operatorname{var}[\cdot]$ | Variance: how much a random quantity varies around its average. |

Important viewpoint:

> After we observe one particular data set, $\mu_{\mathrm{ML}}$ is just one
> fixed number. But when we analyze an estimator, we imagine repeating the
> sampling process many times. Each repeated data set gives a different
> $\mu_{\mathrm{ML}}$. In this repeated-sampling sense, $\mu_{\mathrm{ML}}$ is a
> random variable, so it can have an expectation and a variance.

Now take expectations on both sides.

First,

$$
\mathbb{E}\left[\sum_{n=1}^{N}(x_n-\mu)^2\right]=N\sigma^2,
$$

because each data point has variance $\sigma^2$.

Second, the sample mean $\mu_{\mathrm{ML}}$ also varies from data set to data
set. This is easy to miss. Although $\mu_{\mathrm{ML}}$ is one number after we
compute it from our current data, it would be a different number if we collected
a different data set.

Since

$$
\mu_{\mathrm{ML}}=\frac{1}{N}\sum_{n=1}^{N}x_n,
$$

it is an average of $N$ independent observations. Averages are more stable than
individual observations. If each $x_n$ has variance $\sigma^2$, then the average
has variance $\sigma^2/N$.

Here is the calculation:

$$
\operatorname{var}[\mu_{\mathrm{ML}}]
=\operatorname{var}\left[\frac{1}{N}\sum_{n=1}^{N}x_n\right].
$$

Constants come out of variance as squares:

$$
\operatorname{var}\left[\frac{1}{N}\sum_{n=1}^{N}x_n\right]
=\frac{1}{N^2}\operatorname{var}\left[\sum_{n=1}^{N}x_n\right].
$$

For independent data points, variances add:

$$
\operatorname{var}\left[\sum_{n=1}^{N}x_n\right]
=\sum_{n=1}^{N}\operatorname{var}[x_n]
=N\sigma^2.
$$

Therefore,

$$
\operatorname{var}[\mu_{\mathrm{ML}}]=\frac{\sigma^2}{N}.
$$

This tells us how much the estimated mean wiggles around the true mean. Since
the sample mean is centered around the true mean,

$$
\mathbb{E}[\mu_{\mathrm{ML}}]=\mu.
$$

So $\mu_{\mathrm{ML}}-\mu$ means "the error in the estimated mean". Its average
is zero, because sometimes the sample mean is above the true mean and sometimes
it is below.

The average squared size of this error is

$$
\mathbb{E}\left[(\mu_{\mathrm{ML}}-\mu)^2\right]
=\operatorname{var}[\mu_{\mathrm{ML}}]
=\frac{\sigma^2}{N}.
$$

In words:

> The estimated mean is not exactly the true mean. Its typical squared error is
> $\sigma^2/N$.

In the earlier identity, this error term appears multiplied by $N$:

$$
\mathbb{E}\left[N(\mu_{\mathrm{ML}}-\mu)^2\right]
=N\cdot\frac{\sigma^2}{N}
=\sigma^2.
$$

So estimating the mean "uses up" an amount of squared variation equal to one
$\sigma^2$.

Now return to the identity:

$$
\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2
=
\sum_{n=1}^{N}(x_n-\mu)^2
-N(\mu_{\mathrm{ML}}-\mu)^2.
$$

Taking expectations gives:

- The first term has expected value $N\sigma^2$.
- The second term has expected value $\sigma^2$.

Therefore,

$$
\mathbb{E}\left[\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2\right]
=N\sigma^2-\sigma^2
=(N-1)\sigma^2.
$$

This is the key point: after we fit the mean from the data, the expected total
squared deviation is not $N\sigma^2$ anymore. It is only $(N-1)\sigma^2$.

But the ML variance divides this sum by $N$, so

$$
\mathbb{E}[\sigma^2_{\mathrm{ML}}]
=\frac{1}{N}(N-1)\sigma^2
=\frac{N-1}{N}\sigma^2.
$$

This is smaller than the true variance $\sigma^2$. That is why we say the ML
variance estimate is biased downward.

To correct this bias, we divide by $N-1$ instead of $N$:

$$
\widehat{\sigma}^2
=\frac{1}{N-1}\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2.
$$

> ![Figure 1.15](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_15__textbook_fig_1_15__p28.png)
>
> *Figure 1.15 (Textbook Fig. 1.15, p. 28): Estimating the mean from the same small data set pulls the fitted Gaussian toward the data, causing the maximum-likelihood variance estimate to be too small on average.*

How general is this result?

The formula

$$
\operatorname{var}[\mu_{\mathrm{ML}}]=\frac{\sigma^2}{N}
$$

is not a Gaussian approximation. It follows from independence and finite
variance. The data points do not have to be exactly Gaussian for this variance
of the sample mean to hold.

The Gaussian assumption is used in this section to write down the Gaussian
likelihood and derive the maximum-likelihood estimates. But the main lesson of
this section is broader:

> If we estimate the mean from the same data, the deviations from that fitted
> mean are slightly too small on average. That is why the unbiased sample
> variance divides by $N-1$ rather than $N$.

So this section is still useful even when real data are not perfectly Gaussian.
It teaches a general warning: parameter estimates computed from data also have
their own uncertainty, and this uncertainty can affect later estimates.

The intuition is important:

> The fitted mean is chosen from the data, so it sits closer to the observed
> data points than the true mean would on average. Dividing by $N-1$ corrects
> this downward bias.

---

# §4 Probabilistic Curve Fitting

> 📖 Textbook §1.2.5 Curve fitting re-visited; §1.2.6 Bayesian curve fitting

Before this section, we fitted curves by minimizing an error function such as
sum-of-squares error. That gives us a best-fitting curve, but it does not yet
explain where the error function comes from or how to represent uncertainty.

The main question in this section is:

> Can we reinterpret curve fitting as a probabilistic model for how data are
> generated?

The key idea is to treat each observed target value as

$$
\text{observed target}=\text{curve prediction}+\text{random noise}.
$$

In symbols,

$$
t=y(x,\mathbf{w})+\epsilon.
$$

This says that the data point does not have to lie exactly on the curve. Instead,
the curve gives the average prediction, and the noise explains random deviations
around the curve.

This probabilistic view is useful for three reasons:

1. It explains why least squares appears naturally.
2. It lets us describe prediction uncertainty using probability distributions.
3. It prepares us for Bayesian curve fitting, where uncertainty about the
   parameters $\mathbf{w}$ is also represented probabilistically.

In short:

> Probabilistic curve fitting turns "find the best curve" into "build a
> probability model for how the data could have been generated."

## 4.1 From Least Squares to a Probabilistic Model

We now make this probabilistic model precise for polynomial curve fitting.

Assume that each target is generated by

$$
t=y(x,\mathbf{w})+\epsilon,
$$

where the noise is Gaussian:

$$
\epsilon\sim \mathrm{N}(0,\beta^{-1}).
$$

Equivalently,

$$
p(t\mid x,\mathbf{w},\beta)=\mathrm{N}(t\mid y(x,\mathbf{w}),\beta^{-1}).
$$

> ![Figure 1.16](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_16__textbook_fig_1_16__p29.png)
>
> *Figure 1.16 (Textbook Fig. 1.16, p. 29): Probabilistic regression predicts a distribution around the curve, not just a single curve value. For each input $x$, the model places a Gaussian distribution over possible target values $t$. The mean is the polynomial prediction; the variance describes noise around the curve.*

The key takeaway from the figure is:

> The curve gives the average prediction, while the Gaussian spread around the
> curve describes how much the observed targets may vary.

This model says:

- $y(x,\mathbf{w})$ is the mean prediction,
- $\beta^{-1}$ is the noise variance,
- targets near the curve are more probable than targets far away.

## 4.2 Maximum Likelihood Gives Least Squares

This section explains the most important connection in probabilistic curve
fitting:

> If the observation noise is Gaussian, then maximum likelihood gives the same
> solution as least squares.

The unknown quantity we want to choose is $\mathbf{w}$, the parameter vector
that controls the shape of the curve. For example, in a polynomial model,

$$
y(x,\mathbf{w})=w_0+w_1x+w_2x^2+\cdots+w_Mx^M.
$$

Different values of $\mathbf{w}$ give different curves.

For one data point, the model says:

> If $y(x_n,\mathbf{w})$ is close to the observed target $t_n$, then this data
> point is likely. If it is far away, then this data point is unlikely.

For all data points together, maximum likelihood chooses the curve parameters
that make the observed targets most likely:

$$
\mathbf{w}_{\mathrm{ML}}
=\arg\max_{\mathbf{w}}p(\mathbf{t}\mid \mathbf{x},\mathbf{w},\beta).
$$

The full likelihood multiplies the Gaussian density values for all data points:

$$
p(\mathbf{t}\mid \mathbf{x},\mathbf{w},\beta)
=
\prod_{n=1}^{N}\mathrm{N}(t_n\mid y(x_n,\mathbf{w}),\beta^{-1}).
$$

Why does this become least squares?

For one data point, the Gaussian density has the form

$$
\mathrm{N}(t_n\mid y(x_n,\mathbf{w}),\beta^{-1})
\propto
\exp\left\{
-\frac{\beta}{2}(t_n-y(x_n,\mathbf{w}))^2
\right\}.
$$

The important part is the squared error:

$$
(t_n-y(x_n,\mathbf{w}))^2.
$$

If the prediction $y(x_n,\mathbf{w})$ is far from the observed target $t_n$, this
squared error is large, and the Gaussian density becomes small. If the
prediction is close to $t_n$, the squared error is small, and the density is
large.

For all data points, multiplying the Gaussian densities gives a likelihood. When
we take the log, the product becomes a sum:

$$
\ln p(\mathbf{t}\mid \mathbf{x},\mathbf{w},\beta)
=\text{constant}
-\frac{\beta}{2}\sum_{n=1}^{N}(t_n-y(x_n,\mathbf{w}))^2.
$$

The constant does not depend on $\mathbf{w}$. Also, $\beta>0$, so maximizing the
log likelihood with respect to $\mathbf{w}$ is the same as minimizing the sum of
squared errors:

$$
E(\mathbf{w})
=\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\mathbf{w})-t_n\}^2.
$$

This is exactly the objective from ordinary least squares. The factor $1/2$ does
not change the minimizer; it is included because it makes later derivatives
cleaner.

So the main takeaway is:

> Least squares is maximum likelihood under a Gaussian noise assumption.

This means least squares is not just a convenient algebraic trick. It corresponds
to a clear probabilistic assumption: the data are generated by a curve plus
Gaussian noise.

## 4.3 Estimating the Noise Precision

After we choose the best curve parameters $\mathbf{w}_{\mathrm{ML}}$, we can also
estimate how noisy the data are.

Let the noise variance be $\sigma_{\mathrm{noise}}^2$. This number describes how
widely the observed targets scatter around the fitted curve.

Instead of writing the Gaussian noise using variance, PRML often uses
**precision**, denoted by $\beta$. Precision is the inverse of variance:

$$
\beta=\frac{1}{\sigma_{\mathrm{noise}}^2}.
$$

This is just a different way to describe the same noise level:

- if $\sigma_{\mathrm{noise}}^2$ is large, the noise is large, so $\beta$ is small;
- if $\sigma_{\mathrm{noise}}^2$ is small, the noise is small, so $\beta$ is large.

That is why a large $\beta$ means a narrow Gaussian around the curve, and a small
$\beta$ means a wide Gaussian around the curve.

Maximum likelihood estimates the noise variance by the average squared residual:

$$
\frac{1}{\beta_{\mathrm{ML}}}=\frac{1}{N}\sum_{n=1}^{N}\{y(x_n,\mathbf{w}_{\mathrm{ML}})-t_n\}^2.
$$

Here a residual is the prediction error

$$
y(x_n,\mathbf{w}_{\mathrm{ML}})-t_n.
$$

So this formula says:

> Estimate the noise level by looking at how far the observed targets are from
> the fitted curve.

For this section, the main point is simply: after fitting the curve, the
remaining residuals tell us how noisy the observations are.

## 4.4 MAP Estimation and Regularized Least Squares

Maximum likelihood only asks one question:

> Which curve fits the observed data best?

This can be dangerous when the model is flexible. The curve may twist too much
just to chase the training points. This is overfitting.

MAP estimation adds a second question:

> Among curves that fit the data, can we prefer a simpler one?

In polynomial curve fitting, the parameter vector $\mathbf{w}$ controls the
shape of the curve. Very large weights often allow the curve to change sharply
and fit noise in the training data.

So MAP uses this idea:

> Fit the data, but also discourage very large weights.

This leads to the regularized objective

$$
\underbrace{\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\mathbf{w})-t_n\}^2}_{\text{fit the data}}
+
\underbrace{\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}}_{\text{penalize large weights}}.
$$

The first term is the usual least-squares error. The second term is the
regularization penalty. It becomes large when the weights become large.
The parameter $\lambda$ controls how strong this penalty is:

- small $\lambda$: weak regularization, the curve can fit the data more freely;
- large $\lambda$: strong regularization, large weights are discouraged more.

In probability language, this penalty comes from a prior belief:

> Before seeing the data, smaller weights are more plausible than very large
> weights.

This prior can be written as a Gaussian distribution over $\mathbf{w}$:

$$
p(\mathbf{w}\mid \alpha)=\mathrm{N}(\mathbf{w}\mid \mathbf{0},\alpha^{-1}\mathbf{I}).
$$

Here $\alpha$ controls how strongly the prior prefers small weights. A larger
$\alpha$ means the prior is more concentrated near zero, so it more strongly
discourages large weights.

You do not need to focus on the full prior formula at this stage. The important
meaning is simple: it gives a probabilistic reason for penalizing large weights.

Main takeaway:

> Maximum likelihood says: fit the data.
>
> MAP says: fit the data, but prefer smaller and simpler weights.

So regularized least squares is not just an optimization trick. It can be viewed
as MAP estimation with a prior preference for smaller weights.

## 4.5 Full Bayesian Curve Fitting

Maximum likelihood and MAP both return one chosen parameter vector:

- maximum likelihood returns $\mathbf{w}_{\mathrm{ML}}$;
- MAP returns $\mathbf{w}_{\mathrm{MAP}}$.

Full Bayesian curve fitting does something different. It does not commit to only
one curve. Instead, it keeps a distribution over plausible parameter values.

The idea is:

> Several curves may fit the data reasonably well. Bayesian prediction averages
> over them instead of choosing just one.

The Bayesian predictive distribution is written as

$$
p(t\mid x,\mathbf{x},\mathbf{t})=
\int p(t\mid x,\mathbf{w})p(\mathbf{w}\mid \mathbf{x},\mathbf{t})\,d\mathbf{w}.
$$

This formula looks intimidating, but do not focus on the integral symbol first.
It is just a weighted average over many possible curves.

Read the pieces as follows:

| Term | Meaning |
|------|---------|
| $\mathbf{w}$ | One possible set of curve parameters, hence one possible curve. |
| $p(\mathbf{w}\mid \mathbf{x},\mathbf{t})$ | How plausible that curve is after seeing the training data. |
| $p(t\mid x,\mathbf{w})$ | The prediction made by that particular curve. |
| $\int \cdots d\mathbf{w}$ | Average over all possible curves. |

So the meaning is:

> Predict using many possible curves, weighted by how plausible each curve is
> after seeing the data.

> ![Figure 1.17](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_17__textbook_fig_1_17__p32.png)
>
> *Figure 1.17 (Textbook Fig. 1.17, p. 32): Bayesian curve fitting does not just draw one fitted curve. It gives a predictive mean curve and an uncertainty band around it. Where there are many nearby data points, the band is narrow. Where data are sparse or we extrapolate beyond the data, the band becomes wider.*

The key message of Figure 1.17 is:

> Bayesian prediction knows when it is uncertain. It is more confident near
> observed data and less confident far away from observed data.

The Bayesian view gives two advantages:

1. It naturally expresses predictive uncertainty.
2. It avoids being too confident when data are limited.

The main takeaway is:

> ML and MAP choose one curve. Full Bayesian prediction averages over many
> plausible curves.

In later chapters, this idea will reappear in Bayesian linear regression,
Gaussian processes, Bayesian neural networks, variational inference, and
graphical models.

---

# §5 Model Selection and the Curse of Dimensionality

> 📖 Textbook §1.3 Model Selection; §1.4 The Curse of Dimensionality

## 5.1 Model Selection

Model selection means choosing between model structures or hyperparameter settings.

Examples include:

- choosing polynomial order $M$,
- choosing regularization strength $\lambda$,
- choosing the number of mixture components,
- choosing the number of hidden units in a neural network,
- choosing a kernel width in a kernel method.

A common approach is to split data into three parts.

| Data Split | Purpose |
|------------|---------|
| Training set | Fit model parameters. |
| Validation set | Choose hyperparameters and model class. |
| Test set | Estimate final generalization performance. |

The test set should not be repeatedly used for model selection, because then it becomes part of the training procedure indirectly.

## 5.2 Cross-Validation

When data are limited, holding out a large validation set can waste valuable training data. Cross-validation reduces this problem.

> ![Figure 1.18](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_18__textbook_fig_1_18__p33.png)
>
> *Figure 1.18 (Textbook Fig. 1.18, p. 33): In $S$-fold cross-validation, the data are split into $S$ groups. Each group is used once as validation data while the remaining groups are used for training.*

In $S$-fold cross-validation:

1. Split the data into $S$ subsets.
2. For each fold, train on $S-1$ subsets and validate on the remaining subset.
3. Average the validation performance across the $S$ folds.
4. Choose the model or hyperparameter setting with the best average validation performance.

Special case: if $S=N$, then each validation set has one example. This is called leave-one-out cross-validation.

Cross-validation is useful but computationally expensive. If there are many candidate hyperparameters, the total cost can become large.

## 5.3 The Curse of Dimensionality: Cell Counting

High-dimensional spaces behave very differently from low-dimensional spaces. A naive approach to classification is to divide the input space into cells and classify a test point by the majority class in its cell.

> ![Figure 1.19](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_19__textbook_fig_1_19__p34.png)
>
> *Figure 1.19 (Textbook Fig. 1.19, p. 34): A two-dimensional projection of a classification problem. Even in two dimensions, local neighborhoods can contain mixed class labels.*

> ![Figure 1.20](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_20__textbook_fig_1_20__p35.png)
>
> *Figure 1.20 (Textbook Fig. 1.20, p. 35): A simple grid-based classifier assigns a test point according to the majority class in the same cell. This becomes impractical in high dimensions.*

These figures show a simple local classification idea: divide the input space into small regions and classify a new point by the training examples nearby. This works only when the space is low-dimensional and well populated. In high dimensions, the number of regions grows exponentially, so most regions contain little or no data.

If each dimension is divided into $L$ intervals, then the total number of grid cells is

$$
L^D.
$$

Read $L^D$ as "$L$ to the power of $D$."

This grows exponentially with dimension $D$.

> ![Figure 1.21](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_21__textbook_fig_1_21__p35.png)
>
> *Figure 1.21 (Textbook Fig. 1.21, p. 35): The number of grid regions grows exponentially as dimension increases. This is one form of the curse of dimensionality.*

For example, if $L=10$:

| Dimension $D$ | Number of Cells $10^D$ |
|--------------|-------------------------|
| 1 | 10 |
| 2 | 100 |
| 3 | 1,000 |
| 10 | 10,000,000,000 |
| 100 | $10^{100}$ |

A finite data set becomes sparse in high dimensions. Therefore methods that rely on local counts in small cells quickly become impractical.

## 5.4 Polynomial Coefficient Growth

The curse of dimensionality also appears in polynomial models.

Suppose the input has $D$ dimensions, and we use all polynomial terms up to order $M$. The number of possible terms grows rapidly with both $D$ and $M$.

For example, a second-order polynomial in $D$ variables includes:

- one constant term,
- $D$ linear terms,
- many quadratic and cross terms.

The exact count depends on whether terms are repeated and how the polynomial is represented, but the key point is that complexity grows quickly as dimension increases.

## Textbook Exercise 1.16: Count Cubic Polynomial Parameters

> ![Textbook Exercise 1.16](./CoursePR2026/Fig/Chapter_1/lecture_ex_1_16__textbook_ex_1_16__p61.png)
>
> *Textbook Exercise 1.16 (p. 61): Evaluate how many independent parameters a cubic polynomial has in $D$ dimensions.*

For all polynomial terms up to order $M$, the textbook gives

$$
N(D,M)=\frac{(D+M)!}{D!M!}.
$$

For a cubic polynomial, $M=3$, so

$$
N(D,3)=\frac{(D+3)!}{D!3!}
=\frac{(D+1)(D+2)(D+3)}{6}.
$$

Now plug in the two values from the exercise.

For $D=10$:

$$
N(10,3)=\frac{11\cdot 12\cdot 13}{6}=286.
$$

For $D=100$:

$$
N(100,3)=\frac{101\cdot 102\cdot 103}{6}=176{,}851.
$$

This is the practical meaning of the curse of dimensionality: even a cubic model
goes from a few hundred parameters to almost two hundred thousand parameters
when the input dimension grows from 10 to 100.

This motivates models that exploit structure, such as:

- sparse representations,
- kernels,
- neural networks with shared parameters,
- latent-variable models,
- manifold learning,
- regularization and priors.

## 5.5 Geometry in High Dimensions: Volume Near the Boundary

High-dimensional geometry is often unintuitive. A useful example is a unit ball in $D$ dimensions: all points whose distance from the center is at most 1.

In one dimension, the outer 10% of the interval is only 10% of the length. In two dimensions, the outer 10% of the radius already contains

$$
1 - 0.9^2 = 0.19
$$

of the area. In 20 dimensions, the same outer 10% of the radius contains

$$
1 - 0.9^{20} \approx 0.88
$$

of the volume.

So a "thin shell" near the boundary can contain most of the space. This is the main message: in high dimensions, a randomly chosen point in a ball is usually not near the center. It is much more likely to be close to the boundary.

This means that our low-dimensional intuition can be misleading. In high dimensions, "typical" points may not be where we expect them to be.

## 5.6 Gaussian Probability Mass in High Dimensions

A similar effect occurs for high-dimensional Gaussian distributions.

For a Gaussian distribution, the density is highest at the mean. For example, in a standard Gaussian centered at 0, the point 0 has the largest density.

But this does not mean that most samples lie very close to 0. In high dimensions, the region near the mean has very little volume. A larger-radius shell has lower density at each individual point, but it contains many more possible points.

A useful analogy is a city: downtown may have the highest population density per block, but most people may live in the much larger surrounding residential area. High density at one location is not the same as large total population in a whole region.

For a $D$-dimensional standard Gaussian, a typical sample is usually at distance about

$$
\sqrt{D}
$$

from the mean, not near distance 0.

The main message is: in high dimensions, the highest-density point is not necessarily where most probability mass is located.

## 5.7 Why Machine Learning Is Still Possible

The curse of dimensionality is serious, but it does not make learning impossible. Real data often have structure.

Examples:

- Images are not arbitrary pixel vectors; nearby pixels are correlated.
- Natural language sequences have grammar and semantics.
- Human motion lies on constrained physical manifolds.
- Biological measurements often depend on a smaller number of latent factors.

Machine learning works by exploiting such structure through assumptions, architectures, priors, smoothness, invariance, and compositionality.

---

# §6 Decision Theory

> 📖 Textbook §1.5 Decision Theory (§1.5.1-§1.5.5)

This section explains how to turn model uncertainty into practical action.

A posterior probability such as $p(C_k\mid \mathbf{x})$ tells us what the model believes, but not what we should do. If all mistakes have the same cost, we choose the most probable class. If different mistakes have different costs, we minimize expected loss. If the model is too uncertain, we may reject the example instead of forcing a decision.

The same idea appears in regression: the loss function determines what prediction is optimal. Squared loss leads to the conditional mean, while absolute loss leads to the conditional median.

> Decision theory is the bridge from probabilities to decisions.

## 6.1 Inference versus Decision

Probability theory can tell us quantities such as

$$
p(C_k\mid \mathbf{x}).
$$

But a posterior probability is not yet a final decision. Decision theory asks:

> Given uncertainty and possible costs, what action should we take?

This distinction is crucial.

| Stage | Output | Example |
|-------|--------|---------|
| Inference | Posterior probabilities | $p(\text{cancer}\mid \mathbf{x})=0.03$ |
| Decision | Action | Treat, monitor, reject, request more tests |

The best action depends on the loss associated with different mistakes.

## 6.2 Minimizing Misclassification Rate

Suppose there are classes $C_1,\ldots,C_K$. A classifier divides input space into decision regions $R_1,\ldots,R_K$.

If $\mathbf{x}\in R_k$, the classifier assigns class $C_k$.

To understand the rule, first focus on one input location $\mathbf{x}$. If $p(C_1\mid \mathbf{x})$ is larger than $p(C_2\mid \mathbf{x})$, then choosing $C_1$ gives a smaller chance of being wrong. If $p(C_2\mid \mathbf{x})$ is larger, we choose $C_2$ instead.

> ![Figure 1.24](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_24__textbook_fig_1_24__p40.png)
>
> *Figure 1.24 (Textbook Fig. 1.24, p. 40): Decision regions for a two-class classification problem.*

Figure 1.24 applies this same idea across the whole input space. Each point is assigned to the class with the larger posterior probability. The decision boundary appears where the two posterior probabilities are equal.

If all mistakes have equal cost, the optimal decision rule is:

$$
\text{choose } C_k \text{ such that } p(C_k\mid \mathbf{x}) \text{ is largest.}
$$

Equivalently,

$$
\hat{k}=\arg\max_k p(C_k\mid \mathbf{x}).
$$

This is the standard maximum-posterior classification rule.

## 6.3 Minimizing Expected Loss

In many applications, different errors have different consequences. For example, consider a medical diagnosis system with two possible decisions:

- decide "healthy",
- decide "needs further testing".

If a healthy patient is sent for extra testing, the cost is usually limited: time, money, and inconvenience. But if a seriously ill patient is classified as healthy, the cost can be much larger. This means that the most probable class is not always the best decision.

> ![Figure 1.25](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_25__textbook_fig_1_25__p41.png)
>
> *Figure 1.25 (Textbook Fig. 1.25, p. 41): A loss matrix for a medical diagnosis example. Missing a serious condition can be much more costly than a false alarm.*

Let $L_{kj}$ be the loss incurred when the true class is $C_k$ but we decide $C_j$.

If we observe $\mathbf{x}$ and choose class $C_j$, we do not know the true class for sure. We only have posterior probabilities $p(C_k\mid \mathbf{x})$. Therefore we compute the average possible loss, weighted by these probabilities:

$$
\mathbb{E}[L\mid \mathbf{x},\text{choose }C_j]
=\sum_k L_{kj}p(C_k\mid \mathbf{x}).
$$

The optimal decision is the one with the smallest expected loss:

$$
\hat{j}=\arg\min_j \sum_k L_{kj}p(C_k\mid \mathbf{x}).
$$

The key message is: decision making should consider both probability and consequence. Maximum-posterior classification is the special case where all wrong decisions have the same cost.

## 6.4 The Reject Option

Sometimes the best action is not to classify. If the model is uncertain, we may reject the example and ask for human review.

> ![Figure 1.26](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_26__textbook_fig_1_26__p42.png)
>
> *Figure 1.26 (Textbook Fig. 1.26, p. 42): With a reject option, inputs with insufficient posterior confidence are not assigned to a class.*

A simple reject rule is:

$$
\max_k p(C_k\mid \mathbf{x}) < \theta \quad \Longrightarrow \quad \text{reject}.
$$

In words: if the model is not confident enough in any class, we reject the input instead of forcing a classification.

The threshold $\theta$ controls the tradeoff:

| Threshold | Effect |
|-----------|--------|
| Low $\theta$ | Few rejections, more forced decisions |
| High $\theta$ | More rejections, fewer risky classifications |

Reject options are common in safety-critical settings, medical diagnosis, fraud detection, and human-in-the-loop systems.

## 6.5 Three Approaches: Generative, Discriminative, and Discriminant Models

Bishop distinguishes three modeling approaches for classification. The difference is what the model tries to learn and what kind of output it gives us.

Suppose we want to classify an email as spam or not spam. There are three natural ways to think about the problem.

### Approach A: Generative Modeling

A generative model asks:

> What does each class usually look like?

For spam detection, it tries to learn what spam emails look like and what normal emails look like. Then a new email is assigned to the class it looks most similar to.

The simple formula is:

$$
p(\mathbf{x}\mid C_k)
$$

This means: if the class is $C_k$, how likely is it to see this input $\mathbf{x}$?

After learning this, we can convert it into a posterior probability $p(C_k\mid \mathbf{x})$ using Bayes' rule. The main idea is:

$$
p(\mathbf{x}\mid C_k) \Rightarrow p(C_k\mid \mathbf{x}).
$$

Generative models are useful when we care about the data distribution itself, such as missing data, outlier detection, or data simulation. The downside is that modeling everything about the input can be difficult in high dimensions.

### Approach B: Discriminative Probabilistic Modeling

A discriminative probabilistic model asks:

> Given this input, how likely is each class?

For spam detection, it directly estimates probabilities such as "this email is 92% likely to be spam."

The simple formula is:

$$
p(C_k\mid \mathbf{x}).
$$

This is often enough for classification, and it directly supports probability-based decisions such as expected loss and reject options.

### Approach C: Discriminant Functions

A discriminant function asks:

> Which class gets the highest score?

It may not output a probability. It only gives each class a score and chooses the largest one:

$$
f_k(\mathbf{x})
$$

and

$$
\hat{k}=\arg\max_k f_k(\mathbf{x}).
$$

This can be simple and efficient. The downside is that scores are not necessarily calibrated probabilities, so it is harder to handle uncertainty, rejection, asymmetric loss, or model combination.

> ![Figure 1.27](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_27__textbook_fig_1_27__p44.png)
>
> *Figure 1.27 (Textbook Fig. 1.27, p. 44): Generative modeling captures class-conditional densities, while discriminative modeling focuses on posterior probabilities and decision boundaries. Some density structure may be irrelevant for classification.*

In plain language, the left plot asks: what does each class look like as a distribution over $x$? The right plot asks: after seeing a particular $x$, which class is more likely?

The decision boundary is determined by the right plot, not by every detail of the left plot. For example, the extra bump in the blue class density on the left does not really affect the final boundary. This is why a generative model may spend effort modeling details of the data distribution that are not needed for classification.

The comparison is:

| Approach | Simple question | Learns | Output style |
|----------|-----------------|--------|--------------|
| Generative | What does each class look like? | $p(\mathbf{x}\mid C_k)$ | Can be converted to probabilities |
| Discriminative probabilistic | How likely is each class for this input? | $p(C_k\mid \mathbf{x})$ | Posterior probabilities |
| Discriminant function | Which class gets the highest score? | $f_k(\mathbf{x})$ | Scores or direct labels |

## 6.6 Loss Functions for Regression

Decision theory also applies to regression. In classification, the decision is which class to choose. In regression, the decision is which number to predict.

Suppose we predict $y(\mathbf{x})$ and the true target is $t$. A common choice is squared loss:

$$
L(t,y)=\{y(\mathbf{x})-t\}^2.
$$

Squared loss penalizes large errors strongly. For example, an error of $2$ gives loss $4$, while an error of $5$ gives loss $25$.

If the same input $\mathbf{x}$ can lead to different possible target values $t$, then the best prediction should not chase one particular outcome. Under squared loss, the best prediction is the average target value for that input:

$$
y(\mathbf{x})=\mathbb{E}[t\mid \mathbf{x}].
$$

In words, predict the conditional mean (条件均值).

The full expected squared loss is

$$
\mathbb{E}[L]=\int\int \{y(\mathbf{x})-t\}^2p(\mathbf{x},t)\,d\mathbf{x}\,dt.
$$

This formula averages the squared error over all possible inputs and targets.

> ![Figure 1.28](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_28__textbook_fig_1_28__p47.png)
>
> *Figure 1.28 (Textbook Fig. 1.28, p. 47): Under squared loss, the optimal regression function is the mean of the conditional distribution $p(t\mid x)$.*

This tells us why regression is often formulated as estimating a conditional average.

## 6.7 Expected Squared Loss Decomposition

This section explains a practical point:

> Not all prediction error comes from a bad model.

Suppose two houses have almost the same features: size, location, age, and number of rooms. Their final selling prices may still differ because of negotiation, timing, market noise, or unobserved factors. Even the best model cannot predict all of this variation perfectly.

So prediction error has two sources:

| Source | Meaning | Can we reduce it by improving the model? |
|--------|---------|------------------------------------------|
| Model error | Our prediction is not equal to the best conditional average. | Yes |
| Irreducible noise | The target itself varies even for the same input. | No |

For squared loss, the compact mathematical version is:

$$
\mathbb{E}[L]
=\int \{y(\mathbf{x})-\mathbb{E}[t\mid \mathbf{x}]\}^2p(\mathbf{x})\,d\mathbf{x}
+\int \operatorname{var}[t\mid \mathbf{x}]p(\mathbf{x})\,d\mathbf{x}.
$$

The first term is the reducible part: it becomes zero if we choose the conditional mean.

The second term is the irreducible part: it remains even for the optimal predictor because $t$ itself may be random given $\mathbf{x}$.

The key message is simple: a better model can reduce model error, but it cannot remove noise that is intrinsic to the data-generating process.

## 6.8 Minkowski Loss Family

The previous sections used squared loss, but squared loss is not the only possible choice. This matters because the loss function defines what kind of mistake we care about.

For example, suppose we predict house prices for a neighborhood. Most similar houses sell around 1 million dollars, but a few unusual houses sell for much more. If we use squared loss, those large errors get amplified, so the prediction may be pulled upward by the expensive outliers. If we use absolute loss, the prediction is less sensitive to extreme values.

Bishop summarizes this using the Minkowski loss family:

$$
L_q=|y-t|^q.
$$

The parameter $q$ controls how strongly large errors are punished. Larger $q$ means large errors become much more expensive.

> ![Figure 1.29](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_29__textbook_fig_1_29__p49.png)
>
> *Figure 1.29 (Textbook Fig. 1.29, p. 49): The shape of the loss function changes with $q$. Different choices of loss lead to different optimal predictions.*

Important cases:

| $q$ | Loss | Optimal Prediction |
|-----|------|-------------------|
| $q=2$ | Squared loss | Conditional mean |
| $q=1$ | Absolute loss | Conditional median |
| $q\to 0$ | Strong preference for high-density target values | Conditional mode |

This table says that "the best prediction" is not a universal idea. Under squared loss, the best prediction is the average. Under absolute loss, it is the median. If we care mainly about the most likely target value, the prediction moves toward the mode.

The key message is: choosing a loss function is also choosing what kind of summary of $p(t\mid \mathbf{x})$ we want the model to report.

---

# §7 Information Theory

> 📖 Textbook §1.6 Information Theory (§1.6.1)

This section introduces a language for measuring uncertainty and distribution mismatch.

Probability tells us how likely events are. Information theory asks related questions: how surprising is an event, how uncertain is a distribution, how different are two distributions, and how much does one variable tell us about another?

These ideas will reappear later in maximum likelihood, cross-entropy loss, KL divergence, variational inference, and representation learning.

Maximum likelihood tells us how to fit a probabilistic model. Information theory explains what this fitting means: reducing surprise, minimizing coding cost, and making the model distribution close to the data distribution.

## 7.1 Information Content

Information theory begins with a simple intuition:

> The more surprising an event is, the more information we get when it happens.

For example, if a fair coin lands heads, this is not very surprising. If a very rare event happens, we learn much more from seeing it.

If event $x$ has probability $p(x)$, its information content is

$$
h(x)=-\log_2 p(x).
$$

This formula has the behavior we want:

- Rare event: small $p(x)$, large information.
- Certain event: $p(x)=1$, so $h(x)=0$. We learn nothing from an event that was guaranteed to happen.
- Independent events: information adds up. This is why the logarithm appears: it turns products of probabilities into sums.

The base of the logarithm determines the unit. Base 2 gives bits; natural logarithms give nats. In this course, the unit is less important than the intuition: information measures surprise.

## 7.2 Entropy

Entropy is the average amount of surprise.

If a random variable is easy to predict, its entropy is low. If many outcomes are possible and hard to guess, its entropy is high.

For example:

- A coin that is almost always heads has low entropy.
- A fair coin has higher entropy because the outcome is less predictable.
- A uniform choice among many options has even higher entropy.

Mathematically, entropy is the expected information content:

$$
H[x]=-\sum_x p(x)\log_2 p(x).
$$

For a continuous variable, the analogous quantity is differential entropy:

$$
H[x]=-\int p(x)\ln p(x)\,dx.
$$

So entropy measures the average uncertainty of a random variable.

> ![Figure 1.30](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_30__textbook_fig_1_30__p52.png)
>
> *Figure 1.30 (Textbook Fig. 1.30, p. 52): A broader distribution has higher entropy because observations are less predictable on average.*

A discrete distribution over $M$ states has maximum entropy when it is uniform:

$$
p(x_i)=\frac{1}{M}.
$$

The maximum entropy value is

$$
H[x]=\log M.
$$

Intuitively, the uniform distribution is the most uncertain case because no outcome is more likely than any other outcome.

This result can be derived using a Lagrange multiplier to enforce the normalization constraint $\sum_i p(x_i)=1$, but the main idea is simple: maximum entropy means maximum uncertainty under the given constraints.

## 7.3 Maximum Entropy and the Gaussian

A recurring idea in probabilistic modeling is maximum entropy. The question is:

> If we only know a few facts, what distribution should we choose?

The answer is: choose the distribution that adds the least extra assumptions.

For example, suppose there are $M$ possible outcomes and we know nothing else. Then we should not prefer one outcome over another. The most neutral choice is the uniform distribution.

For a continuous variable, suppose we only know its mean and variance. We know where the distribution is centered and roughly how spread out it is, but we do not know any other shape details. In that case, the most neutral choice is the Gaussian.

Important results:

| Known Constraints | Maximum-Entropy Distribution |
|------------------|------------------------------|
| Discrete variable with $M$ states and no other constraints | Uniform distribution |
| Continuous variable with fixed mean and variance | Gaussian distribution |

This helps explain why the Gaussian distribution is so central. It is not just mathematically convenient. It is also the distribution that says: "I know the mean and variance, but I will not invent extra structure beyond that."

## 7.4 Conditional Entropy and the Chain Rule

Conditional entropy asks:

> After we know $x$, how much uncertainty about $y$ is still left?

If $x$ tells us a lot about $y$, then $H[y\mid x]$ is small. If $x$ tells us almost nothing about $y$, then $H[y\mid x]$ remains large.

For example, knowing the weather may reduce uncertainty about traffic. Knowing a random coin flip probably tells us nothing about traffic.

The conditional entropy of $y$ given $x$ is

$$
H[y\mid x]=-
\int\int p(y,x)\ln p(y\mid x)\,dy\,dx.
$$

The entropy chain rule says that the total uncertainty in two variables can be decomposed into two parts: uncertainty in one variable, plus the remaining uncertainty in the other after the first is known.

$$
H[x,y]=H[y\mid x]+H[x].
$$

This means: to describe both $x$ and $y$, we can first describe $x$, and then describe the remaining uncertainty in $y$ after $x$ is known.

Equivalently, we can choose the opposite order:

$$
H[x,y]=H[x\mid y]+H[y].
$$

This means: first describe $y$, and then describe the remaining uncertainty in $x$ after $y$ is known.

Both formulas describe the same total joint uncertainty $H[x,y]$. They differ only in the order in which we decompose the uncertainty.

This mirrors the probability product rule:

$$
p(x,y)=p(y\mid x)p(x).
$$

We could also write the same joint probability in the opposite order:

$$
p(x,y)=p(x\mid y)p(y).
$$

The analogy is useful: probability factorization becomes entropy decomposition.

## 7.5 Convexity and Jensen's Inequality

This section introduces one mathematical tool that will be useful soon. The next section will compare two probability distributions, and Jensen's inequality is one of the standard tools for proving that comparison behaves sensibly.

The main idea is simple: for a convex function, averaging before applying the function is different from applying the function first and then averaging.

> ![Figure 1.31](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_31__textbook_fig_1_31__p56.png)
>
> *Figure 1.31 (Textbook Fig. 1.31, p. 56): A convex function lies below its chords. This geometric property underlies Jensen's inequality, which is used to prove non-negativity of KL divergence.*

Intuitively, a convex function is bowl-shaped. The line segment between two points on the curve lies above the curve.

For example, $f(x)=x^2$ is convex. If $x$ is equally likely to be $1$ or $3$, then

$$
\mathbb{E}[x]=2.
$$

If we average first and then square, we get

$$
f(\mathbb{E}[x])=2^2=4.
$$

If we square first and then average, we get

$$
\mathbb{E}[f(x)]=\frac{1^2+3^2}{2}=5.
$$

So in this example,

$$
f(\mathbb{E}[x])\leq \mathbb{E}[f(x)].
$$

A function $f$ is convex if

$$
f(\lambda a+(1-\lambda)b)\leq \lambda f(a)+(1-\lambda)f(b)
$$

for $0\leq\lambda\leq 1$.

Jensen's inequality generalizes this to expectations:

$$
f(\mathbb{E}[x])\leq \mathbb{E}[f(x)]
$$

for convex $f$.

In words: for a convex function, "function of the average" is no larger than "average of the function." This inequality is one of the key tools behind distribution comparison and variational inference in Chapter 10.

## 7.6 KL Divergence

KL divergence is a way to compare two probability distributions.

In machine learning, a useful way to read it is:

> $p$ is the distribution we want to match, and $q$ is our model's distribution.

Then $\mathrm{KL}(p\Vert q)$ asks:

> If the real data follow $p$, how bad is it to describe them using $q$?

If $q$ is close to $p$, the KL divergence is small. If $q$ gives low probability to events that often happen under $p$, the KL divergence becomes large. In other words, KL strongly penalizes a model for missing important parts of the real data distribution.

For example, suppose real data often contain digit 8, but the model says digit 8 is almost impossible. Then the model is badly mismatched to the data, so the KL divergence should be large.

The best possible value is 0, which means the two distributions match. KL divergence cannot be negative.

The Kullback-Leibler divergence from $q$ to $p$ is

$$
\mathrm{KL}(p\Vert q)
= -\int p(x)\ln\left\{\frac{q(x)}{p(x)}\right\}\,dx
=\int p(x)\ln\left\{\frac{p(x)}{q(x)}\right\}\,dx.
$$

The formula is an average under $p$: it checks the mismatch at values that actually matter according to the target distribution $p$.

Important properties:

- **Non-negative:** KL divergence measures the extra average cost of using $q$ to describe data that actually follow $p$. The best possible case is no extra cost, so the value cannot go below zero.

$$
\mathrm{KL}(p\Vert q)\geq 0.
$$

- **Zero means perfect match:** KL divergence is zero only when the two distributions are the same.

$$
\mathrm{KL}(p\Vert q)=0
\quad \text{only when} \quad
p(x)=q(x).
$$

- **Not symmetric:** comparing $p$ to $q$ is not the same as comparing $q$ to $p$.

$$
\mathrm{KL}(p\Vert q)\neq \mathrm{KL}(q\Vert p)
$$

Because of this asymmetry, KL divergence is not a distance metric in the strict mathematical sense.

## 7.7 Maximum Likelihood as KL Minimization

This section connects maximum likelihood back to information theory.

There are two distributions to keep in mind:

- the empirical data distribution: what the training data actually look like;
- the model distribution: what the model thinks the data should look like.

Learning means adjusting the model parameters so that the model distribution becomes close to the data distribution.

Suppose the empirical distribution of the data is

$$
\widehat{p}_{\mathrm{data}}(x)=\frac{1}{N}\sum_{n=1}^{N}\delta(x-x_n).
$$

This is a formal way of saying: put probability mass on the observed training examples.

Maximum likelihood chooses model parameters $\theta$ to maximize

$$
\sum_{n=1}^{N}\ln p(x_n\mid \theta).
$$

In words, it chooses parameters that give high probability to the observed data points.

This is equivalent to minimizing

$$
\mathrm{KL}(\widehat{p}_{\mathrm{data}}\Vert p(\cdot\mid\theta))
$$

up to terms that do not depend on $\theta$.

Thus maximum likelihood can be interpreted as choosing the model distribution closest to the empirical data distribution in the KL sense.

The key message is: maximum likelihood is not just a parameter-fitting trick. It is a way to make the model distribution match the data distribution as closely as possible.

This does not mean every loss function must be KL divergence. KL is natural when the learning goal is to match probability distributions. But many tasks define success through a particular decision or error measure rather than through a full distribution.

For example, squared loss in regression encourages the model to predict the conditional mean. Absolute loss encourages the conditional median. Cross-entropy is natural for probabilistic classification because it compares predicted probabilities with target labels. Hinge loss focuses on classification margins. Ranking losses focus on ordering examples correctly.

So the choice of loss function encodes what we care about: distribution matching, numerical prediction error, robustness to outliers, classification margin, ranking quality, or task-specific cost.

## 7.8 Mutual Information

Mutual information measures how much two variables tell us about each other.

The basic question is:

> If we know $y$, does that help us predict $x$?

If the answer is no, then $x$ and $y$ are independent and the mutual information is zero. If knowing one variable makes the other much easier to predict, then the mutual information is large.

For example, an image and its class label should have high mutual information. A random coin flip and the class label should have almost no mutual information.

One definition is

$$
I[x,y]
=\mathrm{KL}(p(x,y)\Vert p(x)p(y)).
$$

This compares the true joint distribution $p(x,y)$ with the distribution we would get if $x$ and $y$ were independent, $p(x)p(y)$. If these two distributions are the same, then knowing one variable tells us nothing about the other.

Equivalently,

$$
I[x,y]=H[x]-H[x\mid y]=H[y]-H[y\mid x].
$$

This version says the same thing in terms of uncertainty:

> mutual information = original uncertainty - remaining uncertainty after observing the other variable.

So mutual information is large when observing one variable removes a lot of uncertainty about the other.

Mutual information appears in feature selection, representation learning, clustering, and information-theoretic views of Bayesian learning.

---

# §8 Chapter Summary and Bridge to Chapter 2

## 8.1 Conceptual Summary

This chapter can be summarized in one sentence:

> Machine learning is about building models that generalize under uncertainty, and probability provides the language for both learning and decision-making.

The key ideas are:

| Topic | Main Lesson |
|-------|-------------|
| Pattern recognition | Learn regularities from data rather than writing brittle manual rules. |
| Curve fitting | Training error and generalization error are different. |
| Overfitting | A flexible model can fit noise and fail on new data. |
| Regularization | Penalizing complexity can improve generalization. |
| Probability | Sum/product rules and Bayes' theorem provide consistent uncertainty calculus. |
| Gaussian noise | Least squares corresponds to maximum likelihood under Gaussian noise. |
| Bayesian learning | Priors and posteriors represent uncertainty over parameters. |
| Model selection | Hyperparameters require validation, cross-validation, or Bayesian comparison. |
| Curse of dimensionality | High-dimensional spaces are sparse and geometrically unintuitive. |
| Decision theory | Optimal actions depend on posterior probabilities and losses. |
| Information theory | Entropy, KL divergence, and mutual information quantify uncertainty and distribution mismatch. |

## 8.2 Mathematical Map

The following equations are the essential mathematical backbone of the chapter.

### Polynomial Curve Fitting

$$
y(x,\mathbf{w})=\sum_{j=0}^{M}w_jx^j
$$

$$
E(\mathbf{w})=\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\mathbf{w})-t_n\}^2
$$

$$
\widetilde{E}(\mathbf{w})=E(\mathbf{w})+\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}
$$

### Probability

$$
p(X)=\sum_Y p(X,Y)
$$

$$
p(X,Y)=p(Y\mid X)p(X)
$$

$$
p(Y\mid X)=\frac{p(X\mid Y)p(Y)}{p(X)}
$$

### Gaussian

$$
\mathrm{N}(x\mid \mu,\sigma^2)=\frac{1}{(2\pi\sigma^2)^{1/2}}
\exp\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}
$$

$$
\mu_{\mathrm{ML}}=\frac{1}{N}\sum_{n=1}^{N}x_n
$$

$$
\sigma^2_{\mathrm{ML}}=\frac{1}{N}\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2
$$

### Decision Theory

$$
\hat{k}=\arg\max_k p(C_k\mid \mathbf{x})
$$

$$
\hat{j}=\arg\min_j\sum_k L_{kj}p(C_k\mid \mathbf{x})
$$

$$
y(\mathbf{x})=\mathbb{E}[t\mid \mathbf{x}]
$$

### Information Theory

$$
H[x]=-\sum_x p(x)\log p(x)
$$

$$
\mathrm{KL}(p\Vert q)=\int p(x)\ln\frac{p(x)}{q(x)}\,dx
$$

$$
I[x,y]=\mathrm{KL}(p(x,y)\Vert p(x)p(y))
$$

## 8.3 Common Student Confusions

| Confusion | Clarification |
|-----------|---------------|
| “A lower training error means a better model.” | Not necessarily. We care about unseen data. Training error can be misleading under overfitting. |
| “A high-order model is always bad.” | No. It is bad only when data and regularization are insufficient to constrain it. |
| “Probability density is probability.” | No. For continuous variables, probability is the integral of density over a region. |
| “Bayes' theorem is only for subjective beliefs.” | In this course, it is a formal rule for updating uncertainty and computing posterior distributions. |
| “Least squares is just a numerical trick.” | Least squares is maximum likelihood under Gaussian noise. |
| “The best classifier always chooses the most probable class.” | Only when all errors have equal cost. With asymmetric loss, minimize expected loss instead. |
| “KL divergence is a distance.” | It measures distribution mismatch but is not symmetric and is not a metric. |

## 8.4 Bridge to Chapter 2

Chapter 1 introduces probability as a language. Chapter 2 develops the main probability distributions used in pattern recognition and machine learning.

The next chapter will study:

- Bernoulli and binomial distributions,
- beta priors,
- multinomial and Dirichlet distributions,
- Gaussian distributions in more depth,
- exponential-family distributions,
- nonparametric density estimation.

The most important transition is:

> Chapter 1 explains why probabilistic modeling is needed. Chapter 2 gives the basic distributional building blocks from which probabilistic models are constructed.
