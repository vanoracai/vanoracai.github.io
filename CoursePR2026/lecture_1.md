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
9. [§8 Chapter Summary, Figure Checklist, and Teaching Flow](#8-chapter-summary-figure-checklist-and-teaching-flow)

---

## Notation and Variable Definitions

This first chapter introduces the notation that will be used throughout the course. The important shift is from thinking of machine learning as “fitting a curve” to thinking of it as **reasoning under uncertainty**.

> **Teaching focus.** Do not treat this section as a dictionary to memorize. In class, highlight the symbols that form the course pipeline:
>
> $$\mathcal{D} \rightarrow \mathbf{x},t \rightarrow y(\mathbf{x}) \rightarrow p(C_k\mid \mathbf{x}) \rightarrow \mathcal{R}_k,\ L_{kj}.$$
>
> The key message is: **data define the learning problem, probability represents uncertainty, and decision theory turns uncertainty into action.**

### Data, Inputs, Targets, and Models

| Symbol | Definition |
|--------|------------|
| **$\mathbf{x}$** | **Input vector / feature vector.** In digit recognition, it can be a vector of pixel intensities. |
| $x$ | A scalar input variable, used in the polynomial curve-fitting example. |
| **$t$** | **Target variable.** In regression it is continuous; in classification it represents a class label. |
| **$\mathcal{D}$** | **Training data set**, usually $\mathcal{D}=\{(\mathbf{x}_n,t_n)\}_{n=1}^N$. |
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
| **$\mathcal{N}(x\mid \mu,\sigma^2)$** | **Univariate Gaussian density** with mean $\mu$ and variance $\sigma^2$. This connects noise assumptions to least squares. |
| $\mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu},\boldsymbol{\Sigma})$ | Multivariate Gaussian density with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$. |
| $\beta$ | Precision parameter, equal to inverse variance: $\beta=1/\sigma^2$. |

### Decision Theory and Information Theory

| Symbol | Definition |
|--------|------------|
| **$C_k$** | **Class $k$.** |
| **$\mathcal{R}_k$** | **Decision region** assigned to class $C_k$. This is where inference becomes a concrete decision rule. |
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
\mathcal{D}=\{(\mathbf{x}_1,t_1),(\mathbf{x}_2,t_2),\ldots,(\mathbf{x}_N,t_N)\}.
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

The variance measures spread:

$$
\operatorname{var}[x]=\mathbb{E}\left[(x-\mathbb{E}[x])^2\right]
=\mathbb{E}[x^2]-\mathbb{E}[x]^2.
$$

For two variables, covariance measures linear co-variation:

$$
\operatorname{cov}[x,y]=\mathbb{E}\left[(x-\mathbb{E}[x])(y-\mathbb{E}[y])\right].
$$

For a vector $\mathbf{x}$, the covariance matrix is

$$
\operatorname{cov}[\mathbf{x}]=\mathbb{E}\left[(\mathbf{x}-\mathbb{E}[\mathbf{x}])(\mathbf{x}-\mathbb{E}[\mathbf{x}])^T\right].
$$

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
p(\mathbf{w}\mid \mathcal{D}).
$$

This posterior distribution can be used to make predictions by averaging over parameter uncertainty.

## 3.11 The Gaussian Distribution

The Gaussian distribution is central because it is mathematically convenient, widely occurring, and closely connected to least squares.

The univariate Gaussian density is

$$
\mathcal{N}(x\mid \mu,\sigma^2)
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

The multivariate Gaussian is

$$
\mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu},\boldsymbol{\Sigma})
=\frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}}
\exp\left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\}.
$$

The quadratic term

$$
(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})
$$

is the squared Mahalanobis distance.

## 3.12 Likelihood for Gaussian Parameters

Suppose we observe independent data $x_1,\ldots,x_N$ from a Gaussian distribution. The likelihood of the parameters is

$$
p(\mathbf{x}\mid \mu,\sigma^2)=\prod_{n=1}^{N}\mathcal{N}(x_n\mid \mu,\sigma^2).
$$

> ![Figure 1.14](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_14__textbook_fig_1_14__p26.png)
>
> *Figure 1.14 (Textbook Fig. 1.14, p. 26): The Gaussian likelihood is the product of the density values assigned to the observed data points. Maximum likelihood chooses parameters that make the observed data most probable under the model.*

It is easier to maximize the log likelihood:

$$
\ln p(\mathbf{x}\mid \mu,\sigma^2)
= -\frac{N}{2}\ln(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2.
$$

Taking derivatives gives the maximum likelihood estimates:

$$
\mu_{\mathrm{ML}}=\frac{1}{N}\sum_{n=1}^{N}x_n,
$$

and

$$
\sigma^2_{\mathrm{ML}}=\frac{1}{N}\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2.
$$

## 3.13 Bias of the Maximum-Likelihood Variance Estimate

The maximum-likelihood estimate of variance is biased. Its expectation is

$$
\mathbb{E}[\sigma^2_{\mathrm{ML}}]=\frac{N-1}{N}\sigma^2.
$$

The unbiased estimate is

$$
\widehat{\sigma}^2=\frac{1}{N-1}\sum_{n=1}^{N}(x_n-\mu_{\mathrm{ML}})^2.
$$

> ![Figure 1.15](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_15__textbook_fig_1_15__p28.png)
>
> *Figure 1.15 (Textbook Fig. 1.15, p. 28): Estimating the mean from the same small data set pulls the fitted Gaussian toward the data, causing the maximum-likelihood variance estimate to be too small on average.*

The intuition is important. When the mean is estimated from the data, the fitted mean is closer to the observed points than the true mean would be on average. This reduces the measured squared deviations and therefore underestimates variance.

---

# §4 Probabilistic Curve Fitting

> 📖 Textbook §1.2.5 Curve fitting re-visited; §1.2.6 Bayesian curve fitting

## 4.1 From Least Squares to a Probabilistic Model

We now revisit polynomial curve fitting using probability.

Assume that each target is generated by

$$
t=y(x,\mathbf{w})+\epsilon,
$$

where the noise is Gaussian:

$$
\epsilon\sim \mathcal{N}(0,\beta^{-1}).
$$

Equivalently,

$$
p(t\mid x,\mathbf{w},\beta)=\mathcal{N}(t\mid y(x,\mathbf{w}),\beta^{-1}).
$$

> ![Figure 1.16](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_16__textbook_fig_1_16__p29.png)
>
> *Figure 1.16 (Textbook Fig. 1.16, p. 29): A probabilistic regression model places a Gaussian distribution over $t$ for each input $x$. The mean is the polynomial prediction; the variance describes noise around the curve.*

This model says:

- $y(x,\mathbf{w})$ is the mean prediction,
- $\beta^{-1}$ is the noise variance,
- targets near the curve are more probable than targets far away.

## 4.2 Maximum Likelihood Gives Least Squares

Assuming independent observations, the likelihood is

$$
p(\mathbf{t}\mid \mathbf{x},\mathbf{w},\beta)=
\prod_{n=1}^{N}\mathcal{N}(t_n\mid y(x_n,\mathbf{w}),\beta^{-1}).
$$

Taking logs gives

$$
\ln p(\mathbf{t}\mid \mathbf{x},\mathbf{w},\beta)
=\frac{N}{2}\ln\beta-\frac{N}{2}\ln(2\pi)
-\beta E(\mathbf{w}),
$$

where

$$
E(\mathbf{w})=\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\mathbf{w})-t_n\}^2.
$$

For fixed $\beta$, maximizing the log likelihood with respect to $\mathbf{w}$ is exactly the same as minimizing the sum-of-squares error.

This equivalence is one of the most important points in the chapter:

> Least squares is not just an algebraic convenience. It is maximum likelihood under a Gaussian noise assumption.

## 4.3 Estimating the Noise Precision

After finding $\mathbf{w}_{\mathrm{ML}}$, we can estimate $\beta$ by maximum likelihood:

$$
\frac{1}{\beta_{\mathrm{ML}}}=\frac{1}{N}\sum_{n=1}^{N}\{y(x_n,\mathbf{w}_{\mathrm{ML}})-t_n\}^2.
$$

This is the residual variance under the fitted model.

The predictive distribution under the maximum-likelihood parameters is

$$
p(t\mid x,\mathbf{w}_{\mathrm{ML}},\beta_{\mathrm{ML}})=
\mathcal{N}(t\mid y(x,\mathbf{w}_{\mathrm{ML}}),\beta_{\mathrm{ML}}^{-1}).
$$

This predictive distribution includes observation noise but does not include uncertainty about the fitted parameters $\mathbf{w}$.

## 4.4 MAP Estimation and Regularized Least Squares

Now place a Gaussian prior on the weights:

$$
p(\mathbf{w}\mid \alpha)=\mathcal{N}(\mathbf{w}\mid \mathbf{0},\alpha^{-1}\mathbf{I}).
$$

By Bayes' theorem,

$$
p(\mathbf{w}\mid \mathbf{x},\mathbf{t},\alpha,\beta)
\propto p(\mathbf{t}\mid \mathbf{x},\mathbf{w},\beta)p(\mathbf{w}\mid \alpha).
$$

The maximum a posteriori estimate is

$$
\mathbf{w}_{\mathrm{MAP}}=\arg\max_{\mathbf{w}}p(\mathbf{w}\mid \mathbf{x},\mathbf{t},\alpha,\beta).
$$

Taking the negative log posterior gives an objective proportional to

$$
\frac{\beta}{2}\sum_{n=1}^{N}\{y(x_n,\mathbf{w})-t_n\}^2+\frac{\alpha}{2}\mathbf{w}^T\mathbf{w}.
$$

Dividing by $\beta$ gives

$$
\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\mathbf{w})-t_n\}^2+\frac{\lambda}{2}\mathbf{w}^T\mathbf{w},
$$

where

$$
\lambda=\frac{\alpha}{\beta}.
$$

Therefore:

> Regularized least squares is MAP estimation under a Gaussian prior over the weights.

This connects optimization, probability, and regularization.

## 4.5 Full Bayesian Curve Fitting

MAP estimation still returns a single parameter vector. Full Bayesian prediction keeps the entire posterior distribution over parameters.

The Bayesian predictive distribution is

$$
p(t\mid x,\mathbf{x},\mathbf{t})=
\int p(t\mid x,\mathbf{w})p(\mathbf{w}\mid \mathbf{x},\mathbf{t})\,d\mathbf{w}.
$$

This integral averages predictions over all plausible parameter values, weighted by their posterior probability.

> ![Figure 1.17](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_17__textbook_fig_1_17__p32.png)
>
> *Figure 1.17 (Textbook Fig. 1.17, p. 32): Bayesian curve fitting produces both a predictive mean and a predictive uncertainty band. Uncertainty is larger where the data provide less constraint.*

The Bayesian view gives two advantages:

1. It naturally expresses predictive uncertainty.
2. It reduces overconfident extrapolation when data are limited.

In later chapters, this idea will reappear in Bayesian linear regression, Gaussian processes, Bayesian neural networks, variational inference, and graphical models.

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

If each dimension is divided into $L$ intervals, then the total number of grid cells is

$$
L^D.
$$

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

This motivates models that exploit structure, such as:

- sparse representations,
- kernels,
- neural networks with shared parameters,
- latent-variable models,
- manifold learning,
- regularization and priors.

## 5.5 Geometry in High Dimensions: Volume Near the Boundary

High-dimensional geometry is often unintuitive. Consider a unit sphere in $D$ dimensions. The fraction of volume lying near the surface becomes large as $D$ increases.

> ![Figure 1.22](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_22__textbook_fig_1_22__p37.png)
>
> *Figure 1.22 (Textbook Fig. 1.22, p. 37): In high dimensions, most of the volume of a sphere lies in a thin shell near the surface.*

This means that our low-dimensional intuition can be misleading. In high dimensions, “typical” points may not be near the center.

## 5.6 Gaussian Probability Mass in High Dimensions

A similar effect occurs for high-dimensional Gaussian distributions.

> ![Figure 1.23](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_23__textbook_fig_1_23__p37.png)
>
> *Figure 1.23 (Textbook Fig. 1.23, p. 37): For a high-dimensional Gaussian, most probability mass lies in a shell at a nonzero radius rather than near the mean.*

The density is highest at the mean, but the amount of volume near the mean is small. The product of density and volume determines probability mass.

This distinction matters in machine learning. High density and high probability mass are not always the same thing in high dimensions.

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

Suppose there are classes $C_1,\ldots,C_K$. A classifier divides input space into decision regions $\mathcal{R}_1,\ldots,\mathcal{R}_K$.

If $\mathbf{x}\in\mathcal{R}_k$, the classifier assigns class $C_k$.

> ![Figure 1.24](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_24__textbook_fig_1_24__p40.png)
>
> *Figure 1.24 (Textbook Fig. 1.24, p. 40): For two classes, errors occur in regions where the chosen decision region does not match the true class distribution. The optimal boundary for minimizing error assigns each $x$ to the most probable class.*

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

In many applications, different errors have different consequences.

> ![Figure 1.25](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_25__textbook_fig_1_25__p41.png)
>
> *Figure 1.25 (Textbook Fig. 1.25, p. 41): A loss matrix for a medical diagnosis example. Missing a serious condition can be much more costly than a false alarm.*

Let $L_{kj}$ be the loss incurred when the true class is $C_k$ but we decide $C_j$.

If we observe $\mathbf{x}$ and choose class $C_j$, the expected loss is

$$
\mathbb{E}[L\mid \mathbf{x},\text{choose }C_j]
=\sum_k L_{kj}p(C_k\mid \mathbf{x}).
$$

The optimal decision is

$$
\hat{j}=\arg\min_j \sum_k L_{kj}p(C_k\mid \mathbf{x}).
$$

This is more general than maximum-posterior classification. Maximum-posterior classification is the special case where all wrong decisions have the same cost.

## 6.4 The Reject Option

Sometimes the best action is not to classify. If the model is uncertain, we may reject the example and ask for human review.

> ![Figure 1.26](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_26__textbook_fig_1_26__p42.png)
>
> *Figure 1.26 (Textbook Fig. 1.26, p. 42): With a reject option, inputs with insufficient posterior confidence are not assigned to a class.*

A simple reject rule is:

$$
\max_k p(C_k\mid \mathbf{x}) < \theta \quad \Longrightarrow \quad \text{reject}.
$$

The threshold $\theta$ controls the tradeoff:

| Threshold | Effect |
|-----------|--------|
| Low $\theta$ | Few rejections, more forced decisions |
| High $\theta$ | More rejections, fewer risky classifications |

Reject options are common in safety-critical settings, medical diagnosis, fraud detection, and human-in-the-loop systems.

## 6.5 Three Approaches: Generative, Discriminative, and Discriminant Models

Bishop distinguishes three modeling approaches for classification.

### Approach A: Generative Modeling

Model the class-conditional density and class prior:

$$
p(\mathbf{x}\mid C_k),\qquad p(C_k).
$$

Then compute the posterior using Bayes' theorem:

$$
p(C_k\mid \mathbf{x})=\frac{p(\mathbf{x}\mid C_k)p(C_k)}{p(\mathbf{x})}.
$$

Generative models can generate or evaluate input data. They are useful for missing data, outlier detection, and data simulation. But modeling the full input distribution can be difficult in high dimensions.

### Approach B: Discriminative Probabilistic Modeling

Model the posterior directly:

$$
p(C_k\mid \mathbf{x}).
$$

This often requires fewer assumptions about the input distribution and can be more accurate for classification when the only goal is predicting labels.

### Approach C: Discriminant Functions

Learn a direct mapping from input to class label:

$$
f(\mathbf{x})\rightarrow C_k.
$$

This can be computationally simple, but it does not provide calibrated posterior probabilities. Without posterior probabilities, it is harder to handle rejection, asymmetric loss, class-prior changes, and model combination.

> ![Figure 1.27](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_27__textbook_fig_1_27__p44.png)
>
> *Figure 1.27 (Textbook Fig. 1.27, p. 44): Generative modeling captures class-conditional densities, while discriminative modeling focuses on posterior probabilities and decision boundaries. Some density structure may be irrelevant for classification.*

The comparison is:

| Approach | Learns | Advantages | Limitations |
|----------|--------|------------|-------------|
| Generative | $p(\mathbf{x}\mid C_k)$ and $p(C_k)$ | Can model data distribution; useful for outliers and missing data | May waste effort modeling irrelevant input variation |
| Discriminative probabilistic | $p(C_k\mid \mathbf{x})$ | Directly supports probabilistic decisions | Does not model full input distribution |
| Discriminant function | Direct class label | Simple and often efficient | Loses uncertainty information |

## 6.6 Loss Functions for Regression

Decision theory also applies to regression. Suppose we predict $y(\mathbf{x})$ and the true target is $t$.

For squared loss,

$$
L(t,y)=\{y(\mathbf{x})-t\}^2.
$$

The expected loss is

$$
\mathbb{E}[L]=\int\int \{y(\mathbf{x})-t\}^2p(\mathbf{x},t)\,d\mathbf{x}\,dt.
$$

The function that minimizes expected squared loss is the conditional mean:

$$
y(\mathbf{x})=\mathbb{E}[t\mid \mathbf{x}].
$$

> ![Figure 1.28](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_28__textbook_fig_1_28__p47.png)
>
> *Figure 1.28 (Textbook Fig. 1.28, p. 47): Under squared loss, the optimal regression function is the mean of the conditional distribution $p(t\mid x)$.*

This tells us why regression is often formulated as estimating a conditional average.

## 6.7 Expected Squared Loss Decomposition

For squared loss, the expected loss can be decomposed into two parts:

$$
\mathbb{E}[L]
=\int \{y(\mathbf{x})-\mathbb{E}[t\mid \mathbf{x}]\}^2p(\mathbf{x})\,d\mathbf{x}
+\int \operatorname{var}[t\mid \mathbf{x}]p(\mathbf{x})\,d\mathbf{x}.
$$

The first term depends on our chosen predictor. It is zero if we choose the conditional mean.

The second term is irreducible noise. It remains even for the optimal predictor because $t$ itself may be random given $\mathbf{x}$.

This decomposition is important: not all prediction error is due to a bad model. Some error may be intrinsic to the data-generating process.

## 6.8 Minkowski Loss Family

Bishop also considers the loss

$$
L_q=|y-t|^q.
$$

> ![Figure 1.29](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_29__textbook_fig_1_29__p49.png)
>
> *Figure 1.29 (Textbook Fig. 1.29, p. 49): The shape of the loss function changes with $q$. Different choices of loss lead to different optimal predictions.*

Important cases:

| $q$ | Loss | Optimal Prediction |
|-----|------|-------------------|
| $q=2$ | Squared loss | Conditional mean |
| $q=1$ | Absolute loss | Conditional median |
| $q\to 0$ | Strong preference for high-density target values | Conditional mode |

This explains why the choice of loss function is not a minor technical detail. It determines what kind of summary of $p(t\mid \mathbf{x})$ the model is encouraged to predict.

---

# §7 Information Theory

> 📖 Textbook §1.6 Information Theory (§1.6.1)

## 7.1 Information Content

Information theory begins with the idea that unlikely events carry more information than likely events.

If event $x$ has probability $p(x)$, its information content is

$$
h(x)=-\log_2 p(x).
$$

Properties:

1. If $p(x)$ is small, then $h(x)$ is large.
2. If $p(x)=1$, then $h(x)=0$.
3. Independent events have additive information because logarithms turn products into sums.

The base of the logarithm determines the unit. Base 2 gives bits; natural logarithms give nats.

## 7.2 Entropy

Entropy is the expected information content:

$$
H[x]=-\sum_x p(x)\log_2 p(x).
$$

For a continuous variable, the analogous quantity is differential entropy:

$$
H[x]=-\int p(x)\ln p(x)\,dx.
$$

Entropy measures the average uncertainty of a random variable.

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

This result can be derived using a Lagrange multiplier to enforce the normalization constraint $\sum_i p(x_i)=1$.

## 7.3 Maximum Entropy and the Gaussian

A recurring idea in probabilistic modeling is maximum entropy.

If we only know limited facts about a distribution, the maximum-entropy distribution is the least committed distribution consistent with those facts.

Important results:

| Known Constraints | Maximum-Entropy Distribution |
|------------------|------------------------------|
| Discrete variable with $M$ states and no other constraints | Uniform distribution |
| Continuous variable with fixed mean and variance | Gaussian distribution |

This helps explain why the Gaussian distribution is so central. It is the maximum-entropy continuous distribution when only mean and variance are specified.

## 7.4 Conditional Entropy and the Chain Rule

The conditional entropy of $y$ given $x$ is

$$
H[y\mid x]=-
\int\int p(y,x)\ln p(y\mid x)\,dy\,dx.
$$

The entropy chain rule says

$$
H[x,y]=H[y\mid x]+H[x].
$$

Equivalently,

$$
H[x,y]=H[x\mid y]+H[y].
$$

This mirrors the probability product rule:

$$
p(x,y)=p(y\mid x)p(x).
$$

The analogy is useful: probability factorization becomes entropy decomposition.

## 7.5 Convexity and Jensen's Inequality

KL divergence and many later variational methods rely on convexity.

> ![Figure 1.31](./CoursePR2026/Fig/Chapter_1/lecture_fig_1_31__textbook_fig_1_31__p56.png)
>
> *Figure 1.31 (Textbook Fig. 1.31, p. 56): A convex function lies below its chords. This geometric property underlies Jensen's inequality, which is used to prove non-negativity of KL divergence.*

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

This inequality is one of the key tools behind variational inference in Chapter 10.

## 7.6 KL Divergence

The Kullback-Leibler divergence from $q$ to $p$ is

$$
\mathrm{KL}(p\Vert q)
= -\int p(x)\ln\left\{\frac{q(x)}{p(x)}\right\}\,dx
=\int p(x)\ln\left\{\frac{p(x)}{q(x)}\right\}\,dx.
$$

It measures the mismatch between two distributions.

Important properties:

1. $\mathrm{KL}(p\Vert q)\geq 0$.
2. $\mathrm{KL}(p\Vert q)=0$ only when $p(x)=q(x)$ almost everywhere.
3. KL divergence is not symmetric:

$$
\mathrm{KL}(p\Vert q)\neq \mathrm{KL}(q\Vert p)
$$

in general.

Because it is not symmetric, KL divergence is not a distance metric in the strict mathematical sense.

## 7.7 Maximum Likelihood as KL Minimization

Suppose the empirical distribution of the data is

$$
\widehat{p}_{\mathrm{data}}(x)=\frac{1}{N}\sum_{n=1}^{N}\delta(x-x_n).
$$

Maximum likelihood chooses model parameters $\theta$ to maximize

$$
\sum_{n=1}^{N}\ln p(x_n\mid \theta).
$$

This is equivalent to minimizing

$$
\mathrm{KL}(\widehat{p}_{\mathrm{data}}\Vert p(\cdot\mid\theta))
$$

up to terms that do not depend on $\theta$.

Thus maximum likelihood can be interpreted as choosing the model distribution closest to the empirical data distribution in the KL sense.

## 7.8 Mutual Information

Mutual information measures how much knowing one variable reduces uncertainty about another.

It is defined as

$$
I[x,y]
=\mathrm{KL}(p(x,y)\Vert p(x)p(y)).
$$

Equivalently,

$$
I[x,y]=H[x]-H[x\mid y]=H[y]-H[y\mid x].
$$

Interpretation:

- If $x$ and $y$ are independent, then $p(x,y)=p(x)p(y)$ and $I[x,y]=0$.
- If knowing $y$ strongly reduces uncertainty about $x$, then $I[x,y]$ is large.

Mutual information appears in feature selection, representation learning, clustering, and information-theoretic views of Bayesian learning.

---

# §8 Chapter Summary, Figure Checklist, and Teaching Flow

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
\mathcal{N}(x\mid \mu,\sigma^2)=\frac{1}{(2\pi\sigma^2)^{1/2}}
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

## 8.3 Figure Checklist

Filename convention: `lecture_fig_<lecture-number>__textbook_fig_<original-textbook-number>__p<textbook-page>.png`.


| Lecture Figure | Textbook Figure | File |
|----------------|-----------------|------|
| Figure 1.1 | PRML Fig. 1.1 | `lecture_fig_1_1__textbook_fig_1_1__p2.png` |
| Figure 1.2 | PRML Fig. 1.2 | `lecture_fig_1_2__textbook_fig_1_2__p4.png` |
| Figure 1.3 | PRML Fig. 1.3 | `lecture_fig_1_3__textbook_fig_1_3__p6.png` |
| Figure 1.4 | PRML Fig. 1.4 | `lecture_fig_1_4__textbook_fig_1_4__p7.png` |
| Figure 1.5 | PRML Fig. 1.5 | `lecture_fig_1_5__textbook_fig_1_5__p8.png` |
| Figure 1.6 | PRML Fig. 1.6 | `lecture_fig_1_6__textbook_fig_1_6__p9.png` |
| Figure 1.7 | PRML Fig. 1.7 | `lecture_fig_1_7__textbook_fig_1_7__p10.png` |
| Figure 1.8 | PRML Fig. 1.8 | `lecture_fig_1_8__textbook_fig_1_8__p11.png` |
| Figure 1.9 | PRML Fig. 1.9 | `lecture_fig_1_9__textbook_fig_1_9__p12.png` |
| Figure 1.11 | PRML Fig. 1.11 | `lecture_fig_1_11__textbook_fig_1_11__p16.png` |
| Figure 1.12 | PRML Fig. 1.12 | `lecture_fig_1_12__textbook_fig_1_12__p18.png` |
| Figure 1.13 | PRML Fig. 1.13 | `lecture_fig_1_13__textbook_fig_1_13__p25.png` |
| Figure 1.14 | PRML Fig. 1.14 | `lecture_fig_1_14__textbook_fig_1_14__p26.png` |
| Figure 1.15 | PRML Fig. 1.15 | `lecture_fig_1_15__textbook_fig_1_15__p28.png` |
| Figure 1.16 | PRML Fig. 1.16 | `lecture_fig_1_16__textbook_fig_1_16__p29.png` |
| Figure 1.17 | PRML Fig. 1.17 | `lecture_fig_1_17__textbook_fig_1_17__p32.png` |
| Figure 1.18 | PRML Fig. 1.18 | `lecture_fig_1_18__textbook_fig_1_18__p33.png` |
| Figure 1.19 | PRML Fig. 1.19 | `lecture_fig_1_19__textbook_fig_1_19__p34.png` |
| Figure 1.20 | PRML Fig. 1.20 | `lecture_fig_1_20__textbook_fig_1_20__p35.png` |
| Figure 1.21 | PRML Fig. 1.21 | `lecture_fig_1_21__textbook_fig_1_21__p35.png` |
| Figure 1.22 | PRML Fig. 1.22 | `lecture_fig_1_22__textbook_fig_1_22__p37.png` |
| Figure 1.23 | PRML Fig. 1.23 | `lecture_fig_1_23__textbook_fig_1_23__p37.png` |
| Figure 1.24 | PRML Fig. 1.24 | `lecture_fig_1_24__textbook_fig_1_24__p40.png` |
| Figure 1.25 | PRML Fig. 1.25 | `lecture_fig_1_25__textbook_fig_1_25__p41.png` |
| Figure 1.26 | PRML Fig. 1.26 | `lecture_fig_1_26__textbook_fig_1_26__p42.png` |
| Figure 1.27 | PRML Fig. 1.27 | `lecture_fig_1_27__textbook_fig_1_27__p44.png` |
| Figure 1.28 | PRML Fig. 1.28 | `lecture_fig_1_28__textbook_fig_1_28__p47.png` |
| Figure 1.29 | PRML Fig. 1.29 | `lecture_fig_1_29__textbook_fig_1_29__p49.png` |
| Figure 1.30 | PRML Fig. 1.30 | `lecture_fig_1_30__textbook_fig_1_30__p52.png` |
| Figure 1.31 | PRML Fig. 1.31 | `lecture_fig_1_31__textbook_fig_1_31__p56.png` |

All figures used in this lecture were rendered from the supplied textbook PDF and saved under:

```text
./CoursePR2026/Fig/Chapter_1/
```

## 8.4 Suggested Teaching Flow

A practical teaching sequence for this chapter is:

1. Start with handwritten digit recognition to motivate vector inputs and classification.
2. Use polynomial curve fitting as the main running example.
3. Emphasize the difference between training error and test error using Figures 1.4-1.5.
4. Introduce regularization as a practical solution, then reinterpret it probabilistically.
5. Teach probability rules slowly: sum rule, product rule, Bayes' theorem.
6. Connect Gaussian likelihood directly to least squares.
7. Explain model selection and cross-validation before moving to high-dimensional geometry.
8. Separate inference from decision using medical loss and reject-option examples.
9. End with entropy, KL divergence, and mutual information as tools that will reappear later.

## 8.5 Common Student Confusions

| Confusion | Clarification |
|-----------|---------------|
| “A lower training error means a better model.” | Not necessarily. We care about unseen data. Training error can be misleading under overfitting. |
| “A high-order model is always bad.” | No. It is bad only when data and regularization are insufficient to constrain it. |
| “Probability density is probability.” | No. For continuous variables, probability is the integral of density over a region. |
| “Bayes' theorem is only for subjective beliefs.” | In this course, it is a formal rule for updating uncertainty and computing posterior distributions. |
| “Least squares is just a numerical trick.” | Least squares is maximum likelihood under Gaussian noise. |
| “The best classifier always chooses the most probable class.” | Only when all errors have equal cost. With asymmetric loss, minimize expected loss instead. |
| “KL divergence is a distance.” | It measures distribution mismatch but is not symmetric and is not a metric. |

## 8.6 Bridge to Chapter 2

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
