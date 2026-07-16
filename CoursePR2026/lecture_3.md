---
layout: course
title: PRML Lecture 3
---

# Pattern Recognition and Machine Learning
## Chapter 3: Linear Models for Regression

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 3 Linear Models for Regression (§3.1-§3.6)

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Linear Basis Function Models](#1-linear-basis-function-models)
3. [§2 The Bias-Variance Decomposition](#2-the-bias-variance-decomposition)
4. [§3 Bayesian Linear Regression](#3-bayesian-linear-regression)
5. [§4 Bayesian Model Comparison](#4-bayesian-model-comparison)
6. [§5 The Evidence Approximation](#5-the-evidence-approximation)
7. [§6 Limitations of Fixed Basis Functions](#6-limitations-of-fixed-basis-functions)
8. [§7 Chapter Summary, Figure Checklist, Exercises, and Teaching Flow](#7-chapter-summary-figure-checklist-exercises-and-teaching-flow)

---

## Notation and Variable Definitions

Chapter 1 introduced regression through polynomial curve fitting. Chapter 2 introduced probability distributions and parameter estimation. Chapter 3 now brings these ideas together and studies a central supervised-learning model:

> Given an input vector $\mathbf{x}$ and a continuous target $t$, construct a function or predictive distribution that can estimate $t$ for new inputs.

The key phrase in this chapter is **linear models**. This does **not** mean the prediction must be a linear function of the original input $\mathbf{x}$. It means the prediction is linear in the adjustable parameters $\mathbf{w}$. By transforming inputs through basis functions $\boldsymbol{\phi}(\mathbf{x})$, we can obtain nonlinear regression curves while keeping parameter estimation mathematically tractable.

> **Teaching focus.** This chapter should feel like a careful extension of the polynomial curve-fitting example in Chapter 1. The core classroom goal is not to make students memorize every matrix identity. The core goal is:
>
> $$
> \text{features} \rightarrow \text{least squares} \rightarrow \text{regularization} \rightarrow \text{uncertainty}.
> $$
>
> For EE undergraduate and graduate students, the most useful parts are linear basis models, ridge/lasso intuition, bias-variance, and the idea that Bayesian regression gives a predictive distribution. The evidence approximation and equivalent-kernel derivations are useful conceptually, but can be treated as optional if time is limited.

### Generic Regression Notation

| Symbol | Definition |
|--------|------------|
| $\mathbf{x}$ | Input vector. In many illustrative figures, $x$ is one-dimensional. |
| $t$ | Continuous target variable. |
| $\mathcal{D}$ | Training data set, usually $\mathcal{D}=\{(\mathbf{x}_n,t_n)\}_{n=1}^N$. |
| $N$ | Number of training observations. |
| $D$ | Dimensionality of the original input vector $\mathbf{x}$. |
| $M$ | Number of basis functions / number of adaptive weights. |
| $y(\mathbf{x},\mathbf{w})$ | Regression function or predictive mean. |
| $\mathbf{w}$ | Weight vector of model parameters. |
| $w_0$ | Bias/intercept parameter. |
| $\boldsymbol{\phi}(\mathbf{x})$ | Feature vector / basis-function vector. |
| $\phi_j(\mathbf{x})$ | The $j$th basis function evaluated at input $\mathbf{x}$. |
| $\mathbf{t}$ | Target vector $(t_1,\ldots,t_N)^T$. |
| $\mathbf{y}$ | Vector of model outputs $(y(\mathbf{x}_1),\ldots,y(\mathbf{x}_N))^T$. |

### Matrix Notation for Linear Basis Models

| Symbol | Definition |
|--------|------------|
| $\boldsymbol{\Phi}$ | Design matrix; row $n$ is $\boldsymbol{\phi}(\mathbf{x}_n)^T$. |
| $\Phi_{nj}$ | Entry of the design matrix: $\Phi_{nj}=\phi_j(\mathbf{x}_n)$. |
| $\boldsymbol{\Phi}^\dagger$ | Moore-Penrose pseudo-inverse of $\boldsymbol{\Phi}$. |
| $\mathbf{W}$ | Weight matrix for multiple-output regression. |
| $\mathbf{T}$ | Target matrix for multiple-output regression. |
| $\mathbf{I}$ | Identity matrix. |

### Probabilistic Regression Notation

| Symbol | Definition |
|--------|------------|
| $\epsilon$ | Additive noise variable. |
| $\beta$ | Noise precision; variance is $\beta^{-1}$. |
| $p(t\mid \mathbf{x},\mathbf{w},\beta)$ | Conditional likelihood of target $t$. |
| $E_D(\mathbf{w})$ | Data error term, typically a sum-of-squares error. |
| $E_W(\mathbf{w})$ | Weight penalty / regularization term. |
| $\lambda$ | Regularization coefficient in frequentist regularized least squares. |
| $\alpha$ | Prior precision over weights in Bayesian linear regression. |

### Bayesian Linear Regression Notation

| Symbol | Definition |
|--------|------------|
| $p(\mathbf{w})$ | Prior distribution over weights. |
| $p(\mathbf{w}\mid \mathbf{t})$ | Posterior distribution over weights after observing targets. |
| $\mathbf{m}_0,\mathbf{S}_0$ | Prior mean and prior covariance of $\mathbf{w}$. |
| $\mathbf{m}_N,\mathbf{S}_N$ | Posterior mean and posterior covariance after $N$ data points. |
| $\mathbf{w}_{\mathrm{ML}}$ | Maximum-likelihood weight estimate. |
| $\mathbf{w}_{\mathrm{MAP}}$ | Maximum-a-posteriori weight estimate. |
| $\sigma_N^2(\mathbf{x})$ | Predictive variance at input $\mathbf{x}$. |
| $k(\mathbf{x},\mathbf{x}')$ | Equivalent kernel / smoothing weight between two inputs. |

### Model Comparison and Evidence Notation

| Symbol | Definition |
|--------|------------|
| $\mathcal{M}_i$ | Candidate model, such as a polynomial model with order $i$. |
| $p(\mathcal{D}\mid \mathcal{M}_i)$ | Model evidence / marginal likelihood. |
| $p(\mathcal{M}_i\mid\mathcal{D})$ | Posterior probability of model $\mathcal{M}_i$. |
| $\gamma$ | Effective number of parameters in the evidence approximation. |
| $\lambda_i$ | Eigenvalue of the Hessian-related matrix $\beta\boldsymbol{\Phi}^T\boldsymbol{\Phi}$. |
| $\mathbf{A}$ | Posterior precision matrix for weights. |

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch.3 opening; §3.1-§3.6

## 0.1 What This Chapter Is Really About

Regression is the supervised-learning problem in which the target is continuous. Examples include predicting house price from location and size, predicting temperature from time and sensor readings, or predicting a real-valued physical quantity from an image or signal feature vector.

The chapter begins with a deceptively simple model:

$$
y(\mathbf{x},\mathbf{w})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}).
$$

The function is linear in $\mathbf{w}$, but it can be nonlinear in $\mathbf{x}$ because $\boldsymbol{\phi}(\mathbf{x})$ can contain nonlinear transformations of the input. This distinction is the main reason linear models are so important: they combine **nonlinear representation** with **linear parameter estimation**.

This chapter can be read as answering six linked questions.

| Topic | Core Question | Long-Term Role in the Course |
|-------|---------------|-------------------------------|
| **Linear basis functions** | How can a model be nonlinear in the input but linear in the weights? | Foundation for regression, classification, neural networks, kernels. |
| **Least squares and ML** | Why does minimizing squared error correspond to maximum likelihood under Gaussian noise? | Connects optimization objectives to probabilistic assumptions. |
| **Regularization** | How do we control overly flexible models? | Introduces weight decay, lasso, MAP estimation, and model complexity control. |
| **Bias-variance** | Why does a model fail either by being too rigid or too sensitive to data noise? | Frequentist view of generalization. |
| **Bayesian regression** | How do we represent uncertainty over parameters and predictions? | Foundation for Bayesian neural networks and Gaussian processes. |
| **Evidence** | Can a Bayesian method choose model complexity or hyperparameters automatically? | Foundation for empirical Bayes, model comparison, and Occam factors. |

The recurring pattern of the chapter is:

$$
\text{choose basis functions}
\longrightarrow
\text{fit weights}
\longrightarrow
\text{control complexity}
\longrightarrow
\text{quantify predictive uncertainty}.
$$

## 0.2 Why Linear Models Still Matter

At first sight, linear models may look too simple for modern machine learning. However, they are essential for three reasons.

First, many complex models contain linear models as building blocks. Neural-network last layers, kernel ridge regression, Gaussian-process prediction, and many probabilistic graphical models all reuse the same ideas: basis representations, quadratic objectives, Gaussian posteriors, and linear algebra.

Second, linear models are mathematically transparent. We can derive closed-form solutions, understand the geometry of least squares, and see exactly how regularization changes the answer.

Third, linear models give a clean setting for learning the difference between point estimation and Bayesian prediction. Maximum likelihood returns one weight vector. Bayesian regression returns a full posterior distribution over weight vectors and therefore a predictive distribution over targets.

## 0.3 The Key Conceptual Transition from Chapter 1

In Chapter 1, polynomial curve fitting appeared as an introductory example. There, we saw that increasing polynomial order can reduce training error but may produce severe over-fitting. We also saw that regularization can suppress large coefficients and improve generalization.

Chapter 3 generalizes that example in three directions.

| Chapter 1 View | Chapter 3 Generalization |
|---------------|--------------------------|
| Polynomial basis functions only | General basis functions: polynomial, Gaussian, sigmoidal, Fourier, wavelet. |
| Sum-of-squares error as an optimization criterion | Sum-of-squares derived from a Gaussian likelihood. |
| Regularization as a heuristic penalty | Regularization as MAP estimation under a prior. |
| One fitted curve | A predictive distribution with uncertainty. |
| Validation for model complexity | Bayesian evidence and effective number of parameters. |

This chapter should therefore be taught as a bridge: it starts from familiar least squares and ends with Bayesian model comparison.

---

# §1 Linear Basis Function Models

> 📖 Textbook §3.1 Linear Basis Function Models; §3.1.1-§3.1.5

## 1.1 Model Form: Linear in Parameters, Nonlinear in Inputs

The basic linear regression model is

$$
y(\mathbf{x},\mathbf{w})=w_0+\sum_{j=1}^{M-1}w_j\phi_j(\mathbf{x}).
$$

It is often convenient to absorb the bias term into the basis-function vector by defining

$$
\phi_0(\mathbf{x})=1.
$$

Then the model becomes

$$
y(\mathbf{x},\mathbf{w})=\sum_{j=0}^{M-1}w_j\phi_j(\mathbf{x})
=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}).
$$

This compact notation is extremely important. It says that prediction is an inner product between a weight vector and a feature vector. The feature vector can be highly nonlinear, but once the features are fixed, the model is linear in the weights.

Common choices of basis functions include the following.

| Basis Type | Example | Intuition |
|-----------|---------|-----------|
| Polynomial | $\phi_j(x)=x^j$ | Global functions; changing one coefficient affects the curve broadly. |
| Gaussian | $\phi_j(x)=\exp\{-(x-\mu_j)^2/(2s^2)\}$ | Local bumps centered at $\mu_j$. |
| Sigmoidal | $\phi_j(x)=\sigma((x-\mu_j)/s)$ | Smooth step-like transitions. |
| Fourier | $\sin(\omega_jx),\cos(\omega_jx)$ | Periodic decomposition into frequencies. |
| Wavelet | Localized oscillatory basis | Localized in both space and frequency. |

The main modeling question is not simply “what are the weights?” but also “what representation $\boldsymbol{\phi}(\mathbf{x})$ makes the regression problem easy?” This idea foreshadows kernels and neural networks. Kernel methods implicitly choose very high-dimensional basis functions; neural networks learn the basis functions from data.

> ![Figure 3.1](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_1__textbook_fig_3_1_p140_basis_functions.png)
>
> *Figure 3.1 (Textbook Fig. 3.1, p. 140): The left panel shows polynomial basis functions. They are global: a polynomial basis function has nonlocal influence over the full input range. The middle panel shows Gaussian basis functions. They are local: each basis function is active mainly around its center. The right panel shows sigmoidal basis functions, which create smooth threshold-like transitions. These three panels are a useful way to explain that “linear model” does not mean “straight line in input space.” It means that the final function is formed by linearly combining fixed basis functions.*

## 1.2 Probabilistic Formulation: Gaussian Noise Model

To connect least squares with probability, assume the target is generated as

$$
t=y(\mathbf{x},\mathbf{w})+\epsilon,
$$

where $\epsilon$ is zero-mean Gaussian noise with precision $\beta$:

$$
\epsilon\sim\mathcal{N}(0,\beta^{-1}).
$$

Equivalently,

$$
p(t\mid\mathbf{x},\mathbf{w},\beta)
=\mathcal{N}\bigl(t\mid y(\mathbf{x},\mathbf{w}),\beta^{-1}\bigr).
$$

This equation carries an important modeling assumption. It says that for a fixed input $\mathbf{x}$, the target $t$ fluctuates around the deterministic model output $y(\mathbf{x},\mathbf{w})$, and the fluctuations are Gaussian with the same variance for every input.

The conditional mean is

$$
\mathbb{E}[t\mid\mathbf{x}]=y(\mathbf{x},\mathbf{w}),
$$

and the conditional variance is

$$
\operatorname{var}[t\mid\mathbf{x}]=\beta^{-1}.
$$

Thus the model separates two ideas:

1. $y(\mathbf{x},\mathbf{w})$ describes the systematic trend;
2. $\beta^{-1}$ describes the irreducible target noise around that trend.

## 1.3 Maximum Likelihood and Least Squares

Given training data

$$
\mathcal{D}=\{(\mathbf{x}_n,t_n)\}_{n=1}^N,
$$

and assuming independent observations, the likelihood is

$$
p(\mathbf{t}\mid\mathbf{X},\mathbf{w},\beta)
=\prod_{n=1}^N\mathcal{N}\bigl(t_n\mid \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n),\beta^{-1}\bigr).
$$

The log likelihood is

$$
\ln p(\mathbf{t}\mid\mathbf{X},\mathbf{w},\beta)
=\frac{N}{2}\ln\beta-\frac{N}{2}\ln(2\pi)
-\frac{\beta}{2}\sum_{n=1}^N\{t_n-\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2.
$$

For fixed $\beta$, maximizing this log likelihood with respect to $\mathbf{w}$ is equivalent to minimizing the sum-of-squares error

$$
E_D(\mathbf{w})
=\frac{1}{2}\sum_{n=1}^N\{t_n-\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2.
$$

This is one of the most important equivalences in the course:

$$
\boxed{\text{Gaussian noise likelihood} \quad \Longleftrightarrow \quad \text{least-squares error}.}
$$

It means that squared error is not just a convenient algebraic choice. It corresponds to a specific assumption about the distribution of target noise.

## 1.4 Design Matrix and Normal Equations

Define the design matrix $\boldsymbol{\Phi}$ by

$$
\Phi_{nj}=\phi_j(\mathbf{x}_n),
$$

so that

$$
\boldsymbol{\Phi}
=\begin{pmatrix}
\phi_0(\mathbf{x}_1) & \phi_1(\mathbf{x}_1) & \cdots & \phi_{M-1}(\mathbf{x}_1)\\
\phi_0(\mathbf{x}_2) & \phi_1(\mathbf{x}_2) & \cdots & \phi_{M-1}(\mathbf{x}_2)\\
\vdots & \vdots & \ddots & \vdots\\
\phi_0(\mathbf{x}_N) & \phi_1(\mathbf{x}_N) & \cdots & \phi_{M-1}(\mathbf{x}_N)
\end{pmatrix}.
$$

The vector of predictions on the training set is

$$
\mathbf{y}=\boldsymbol{\Phi}\mathbf{w}.
$$

The sum-of-squares error can be written as

$$
E_D(\mathbf{w})=\frac{1}{2}\|\mathbf{t}-\boldsymbol{\Phi}\mathbf{w}\|^2.
$$

Taking the gradient with respect to $\mathbf{w}$ gives

$$
\nabla_{\mathbf{w}}E_D(\mathbf{w})
=\boldsymbol{\Phi}^T\boldsymbol{\Phi}\mathbf{w}-\boldsymbol{\Phi}^T\mathbf{t}.
$$

Setting this gradient to zero gives the normal equations:

$$
\boldsymbol{\Phi}^T\boldsymbol{\Phi}\mathbf{w}_{\mathrm{ML}}
=\boldsymbol{\Phi}^T\mathbf{t}.
$$

If $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ is invertible, then

$$
\mathbf{w}_{\mathrm{ML}}
=(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{t}.
$$

This is often written using the Moore-Penrose pseudo-inverse:

$$
\mathbf{w}_{\mathrm{ML}}=\boldsymbol{\Phi}^{\dagger}\mathbf{t},
\qquad
\boldsymbol{\Phi}^{\dagger}=(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T.
$$

The pseudo-inverse notation is useful because it emphasizes that least squares is a linear operation on the target vector. However, in numerical computation, directly forming $(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}$ can be unstable when the design matrix is ill-conditioned. Practical implementations often use QR decomposition, singular value decomposition, or regularized solvers.

## Textbook Exercise 3.3: Weighted Least Squares

> ![Textbook Exercise 3.3](./CoursePR2026/Fig/Chapter_3/lecture_ex_3_3__textbook_ex_3_3_p174.png)
>
> *Textbook Exercise 3.3 (p. 174): Derive the weighted least-squares solution and interpret the weights.*

This is a good exercise because it changes only one thing from ordinary least squares: every data point now has a weight $r_n>0$.

Write the error in matrix form. Let

$$
\mathbf{R}=\operatorname{diag}(r_1,\ldots,r_N).
$$

Then

$$
E_D(\mathbf{w})
=\frac{1}{2}(\mathbf{t}-\boldsymbol{\Phi}\mathbf{w})^T
\mathbf{R}
(\mathbf{t}-\boldsymbol{\Phi}\mathbf{w}).
$$

Differentiate with respect to $\mathbf{w}$:

$$
\nabla_{\mathbf{w}}E_D
=-\boldsymbol{\Phi}^T\mathbf{R}(\mathbf{t}-\boldsymbol{\Phi}\mathbf{w}).
$$

Set the gradient to zero:

$$
\boldsymbol{\Phi}^T\mathbf{R}\boldsymbol{\Phi}\mathbf{w}^{\star}
=\boldsymbol{\Phi}^T\mathbf{R}\mathbf{t}.
$$

So the solution is

$$
\boxed{
\mathbf{w}^{\star}
=
(\boldsymbol{\Phi}^T\mathbf{R}\boldsymbol{\Phi})^{-1}
\boldsymbol{\Phi}^T\mathbf{R}\mathbf{t}.
}
$$

There are two simple ways to understand $r_n$.

| Interpretation | Meaning |
|----------------|---------|
| Data-dependent noise | A large $r_n$ means point $n$ is trusted more. In a Gaussian noise model, this corresponds to smaller noise variance. |
| Replicated data | If $r_n$ is an integer, it behaves like repeating point $n$ that many times in the data set. |

The main idea is practical: not every measurement has to be trusted equally. This appears in sensor fusion, robust regression, and later in IRLS for logistic regression.

## 1.5 Estimating the Noise Precision

After finding $\mathbf{w}_{\mathrm{ML}}$, we can maximize the likelihood with respect to $\beta$. The result is

$$
\frac{1}{\beta_{\mathrm{ML}}}
=\frac{1}{N}\sum_{n=1}^N\{t_n-\mathbf{w}_{\mathrm{ML}}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2.
$$

This says that the estimated noise variance is the average squared residual under the fitted model. If the model fits the training data extremely closely, the residuals are small and $\beta_{\mathrm{ML}}$ becomes large. This can be misleading when the model is over-flexible, because small training residuals may reflect fitting noise rather than discovering the true function.

## 1.6 Geometry of Least Squares

Least squares has a simple geometric interpretation. The target vector $\mathbf{t}$ lives in an $N$-dimensional space, because it contains one target value for each training example. The columns of $\boldsymbol{\Phi}$ also live in this same $N$-dimensional space. The model can only produce prediction vectors of the form

$$
\mathbf{y}=\boldsymbol{\Phi}\mathbf{w},
$$

so all possible predictions lie in the subspace spanned by the columns of $\boldsymbol{\Phi}$.

Least squares chooses the prediction vector $\mathbf{y}$ in that subspace that is closest to $\mathbf{t}$ in Euclidean distance. Therefore, the residual vector

$$
\mathbf{r}=\mathbf{t}-\mathbf{y}
$$

is orthogonal to the model subspace. Algebraically, this orthogonality condition is

$$
\boldsymbol{\Phi}^T(\mathbf{t}-\boldsymbol{\Phi}\mathbf{w}_{\mathrm{ML}})=\mathbf{0},
$$

which is exactly the normal equation.

> ![Figure 3.2](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_2__textbook_fig_3_2_p143_least_squares_geometry.png)
>
> *Figure 3.2 (Textbook Fig. 3.2, p. 143): The vector $\mathbf{t}$ is the observed target vector. The green/blue basis-function vectors span a subspace. Least squares projects $\mathbf{t}$ orthogonally onto that subspace, producing the fitted vector $\mathbf{y}$. The residual vector is perpendicular to every basis direction. This figure is worth emphasizing because it turns the normal equations from a symbolic formula into a geometric fact.*

## 1.7 Sequential Learning and the LMS Algorithm

The closed-form solution uses all training data at once. In many settings, however, data arrive sequentially. We may want to update the model after each new example instead of recomputing the full solution.

The sum-of-squares objective can be written as a sum of per-example errors:

$$
E_D(\mathbf{w})=\sum_{n=1}^N E_n(\mathbf{w}),
$$

where

$$
E_n(\mathbf{w})=\frac{1}{2}\{t_n-\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2.
$$

The gradient of the single-example error is

$$
\nabla E_n(\mathbf{w})
=-\{t_n-\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}\boldsymbol{\phi}(\mathbf{x}_n).
$$

A stochastic gradient update is therefore

$$
\mathbf{w}^{(\tau+1)}
=\mathbf{w}^{(\tau)}+
\eta\{t_n-\mathbf{w}^{(\tau)T}\boldsymbol{\phi}(\mathbf{x}_n)\}\boldsymbol{\phi}(\mathbf{x}_n),
$$

where $\eta$ is the learning rate. This is known as the **least-mean-squares** or **LMS** algorithm.

The update has a simple interpretation:

- compute the prediction error on the current example;
- move the weights in the direction of the feature vector;
- scale the movement by the error and the learning rate.

The learning rate controls the stability-speed trade-off. If $\eta$ is too small, learning is slow. If $\eta$ is too large, the weights may oscillate or diverge. This issue will reappear throughout the course in neural-network optimization.

## 1.8 Regularized Least Squares

Least squares can overfit when the basis expansion is flexible or the data set is small. Regularization controls model complexity by adding a penalty on the weights:

$$
E(\mathbf{w})=E_D(\mathbf{w})+\lambda E_W(\mathbf{w}).
$$

The most common choice is quadratic regularization:

$$
E_W(\mathbf{w})=\frac{1}{2}\mathbf{w}^T\mathbf{w}.
$$

Then the objective becomes

$$
E(\mathbf{w})
=\frac{1}{2}\sum_{n=1}^N\{t_n-\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2
+\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}.
$$

The solution is

$$
\mathbf{w}
=(\lambda\mathbf{I}+\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{t}.
$$

This is known as ridge regression or weight decay. The term $\lambda\mathbf{I}$ improves numerical conditioning and discourages large weights.

There is a subtle but important point about the bias parameter $w_0$. In many applications, we do not regularize $w_0$, because the intercept merely shifts the function up or down and should not be penalized in the same way as shape-controlling weights. Bishop notes the common convention of omitting $w_0$ from the regularizer.

## 1.9 General $q$-Norm Regularization and Sparsity

A more general regularizer is

$$
E_W(\mathbf{w})=\frac{1}{2}\sum_{j=1}^{M}|w_j|^q.
$$

Different values of $q$ produce different geometries.

| $q$ | Geometry | Effect |
|-----|----------|--------|
| $q=2$ | Smooth circular/elliptical contours | Shrinks weights smoothly but rarely makes them exactly zero. |
| $q=1$ | Diamond-shaped contours | Can drive weights exactly to zero; gives sparse solutions. |
| $q<1$ | Nonconvex contours with sharper corners | Strong sparsity but harder optimization. |
| $q>2$ | Squarer contours | Penalizes large weights strongly but does not favor sparsity. |

> ![Figure 3.3](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_3__textbook_fig_3_3_p145_regularization_q_norm_contours.png)
>
> *Figure 3.3 (Textbook Fig. 3.3, p. 145): The shape of the regularization contours determines what kinds of solutions are encouraged. The $q=2$ contour is smooth, so the optimum usually has many small nonzero weights. The $q=1$ contour has corners on the coordinate axes. When the error contours touch a corner, one coordinate becomes exactly zero. This is the geometric reason lasso can produce sparse models.*

The lasso corresponds to $q=1$. Its objective is not differentiable at zero, but it is convex and often yields models in which only a subset of basis functions are active. This is useful when we want both prediction and feature selection.

## 1.10 Constrained View of Regularization

Regularized least squares can also be written as a constrained optimization problem:

$$
\min_{\mathbf{w}} E_D(\mathbf{w})
\quad \text{subject to} \quad
E_W(\mathbf{w})\leq \eta.
$$

The regularization coefficient $\lambda$ and the constraint size $\eta$ are two ways of controlling the same trade-off. A large $\lambda$ corresponds to a small allowed weight region. A small $\lambda$ corresponds to a loose constraint.

> ![Figure 3.4](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_4__textbook_fig_3_4_p146_lasso_sparsity_constraint_geometry.png)
>
> *Figure 3.4 (Textbook Fig. 3.4, p. 146): The blue contours represent the unregularized error. The feasible region represents the regularization constraint. With $q=2$, the circular boundary usually touches an error contour away from the axes, so both weights remain nonzero. With $q=1$, the diamond boundary has sharp corners on the axes, and the optimum can occur at a corner. That is why lasso often sets some coefficients exactly to zero.*

## Textbook Exercise 3.5: Regularization as a Constraint

> ![Textbook Exercise 3.5](./CoursePR2026/Fig/Chapter_3/lecture_ex_3_5__textbook_ex_3_5_p174.png)
>
> *Textbook Exercise 3.5 (p. 174): Show that penalized least squares is equivalent to constrained least squares.*

This exercise is mainly about interpretation. There are two ways to say the same preference.

The penalized version is

$$
\min_{\mathbf{w}}
\left[
E_D(\mathbf{w})+\lambda E_W(\mathbf{w})
\right].
$$

The constrained version is

$$
\min_{\mathbf{w}} E_D(\mathbf{w})
\quad \text{subject to} \quad
E_W(\mathbf{w})\leq \eta.
$$

Using a Lagrange multiplier, the constrained problem becomes

$$
\mathcal{L}(\mathbf{w},\lambda)
=E_D(\mathbf{w})+\lambda(E_W(\mathbf{w})-\eta).
$$

For a fixed active constraint, the term $-\lambda\eta$ is constant with respect to $\mathbf{w}$, so minimizing $\mathcal{L}$ over $\mathbf{w}$ is equivalent to minimizing

$$
E_D(\mathbf{w})+\lambda E_W(\mathbf{w}).
$$

The relationship between $\lambda$ and $\eta$ is inverse in spirit:

| Parameter | Intuition |
|-----------|-----------|
| Large $\lambda$ | Strong penalty, so the allowed effective weight size is small. |
| Small $\lambda$ | Weak penalty, so the allowed effective weight size is larger. |
| Small $\eta$ | Tight constraint, similar to strong regularization. |
| Large $\eta$ | Loose constraint, similar to weak regularization. |

In class, this is worth explaining geometrically: regularization either adds a cost for leaving the origin, or says the solution must stay inside a ball/diamond. The final fitted point is where the lowest possible error contour first touches that allowed region.

## 1.11 Multiple Outputs

So far the target $t$ has been scalar. If each input has a vector target

$$
\mathbf{t}_n=(t_{n1},\ldots,t_{nK})^T,
$$

we can use a matrix of weights $\mathbf{W}$ and write

$$
\mathbf{y}(\mathbf{x},\mathbf{W})=\mathbf{W}^T\boldsymbol{\phi}(\mathbf{x}).
$$

Let $\mathbf{T}$ be the $N\times K$ matrix of targets. The least-squares solution is

$$
\mathbf{W}_{\mathrm{ML}}
=(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{T}
=\boldsymbol{\Phi}^{\dagger}\mathbf{T}.
$$

This result shows that multiple-output regression decomposes into independent least-squares problems for each output dimension when the noise model is independent and isotropic across outputs. The basis representation is shared, but each target dimension gets its own weight vector.

---

# §2 The Bias-Variance Decomposition

> 📖 Textbook §3.2 The Bias-Variance Decomposition

## 2.1 Why Training Error Is Not Enough

A flexible model can fit the training data very well and still generalize poorly. This happens when the model fits random noise in the training set. Conversely, a very simple model may generalize poorly because it cannot represent the underlying function at all.

The bias-variance decomposition gives a frequentist way to describe these two failure modes.

- **Bias** measures systematic error caused by an overly restrictive model.
- **Variance** measures sensitivity to the particular training set.
- **Noise** measures irreducible randomness in the target variable.

The decomposition is not primarily a practical algorithm; it is a conceptual tool for understanding model complexity.

## 2.2 Expected Squared Loss

Let $h(\mathbf{x})$ be the true regression function:

$$
h(\mathbf{x})=\mathbb{E}[t\mid\mathbf{x}].
$$

Suppose we train a model on a data set $\mathcal{D}$ and obtain the prediction function

$$
y(\mathbf{x};\mathcal{D}).
$$

Because the training set is random, $y(\mathbf{x};\mathcal{D})$ is also random. We consider the expected squared error averaged over possible training sets:

$$
\mathbb{E}_{\mathcal{D}}\left[(y(\mathbf{x};\mathcal{D})-h(\mathbf{x}))^2\right].
$$

Add and subtract the average prediction $\mathbb{E}_{\mathcal{D}}[y(\mathbf{x};\mathcal{D})]$:

$$
y(\mathbf{x};\mathcal{D})-h(\mathbf{x})
=
\{y(\mathbf{x};\mathcal{D})-\mathbb{E}_{\mathcal{D}}[y(\mathbf{x};\mathcal{D})]\}
+
\{\mathbb{E}_{\mathcal{D}}[y(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\}.
$$

Squaring and averaging over data sets gives

$$
\mathbb{E}_{\mathcal{D}}\left[(y(\mathbf{x};\mathcal{D})-h(\mathbf{x}))^2\right]
=
\underbrace{\left(\mathbb{E}_{\mathcal{D}}[y(\mathbf{x};\mathcal{D})]-h(\mathbf{x})\right)^2}_{\text{bias}^2}
+
\underbrace{\mathbb{E}_{\mathcal{D}}\left[(y(\mathbf{x};\mathcal{D})-\mathbb{E}_{\mathcal{D}}[y(\mathbf{x};\mathcal{D})])^2\right]}_{\text{variance}}.
$$

If we include the randomness of the target $t$ around $h(\mathbf{x})$, the full expected squared loss becomes

$$
\mathbb{E}_{\mathcal{D}}\mathbb{E}_{t\mid\mathbf{x}}
\left[(y(\mathbf{x};\mathcal{D})-t)^2\right]
=\text{bias}^2+\text{variance}+\text{noise}.
$$

## 2.3 Intuitive Meaning of the Three Terms

The decomposition is easiest to understand by imagining that we repeatedly collect new training sets from the same data-generating process and train a new model each time.

| Term | What It Measures | Typical Cause | What It Looks Like |
|------|------------------|---------------|--------------------|
| Bias | Difference between average fitted function and true function | Model too simple, too much regularization | All fitted curves miss the same structure. |
| Variance | Variability among fitted functions across data sets | Model too flexible, too little regularization | Different fitted curves wiggle in different ways. |
| Noise | Randomness in $t$ even when $\mathbf{x}$ is known | Inherent measurement noise or unobserved variables | Cannot be eliminated by better modeling. |

A useful teaching phrase is:

> Bias is being consistently wrong. Variance is being inconsistently sensitive.

## 2.4 Regularization Controls the Trade-Off

In the sinusoidal example, Bishop uses a model with many Gaussian basis functions and varies the regularization coefficient $\lambda$.

When $\lambda$ is large, the weights are strongly constrained. The model is smooth and stable across data sets, so variance is low. But it may be too rigid to match the true sinusoidal function, so bias is high.

When $\lambda$ is small, the model can adapt strongly to each data set. Bias may be low because the model class can represent the true function, but variance may be high because each fitted curve follows different noise patterns.

> ![Figure 3.5](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_5__textbook_fig_3_5_p150_bias_variance_lambda_fits.png)
>
> *Figure 3.5 (Textbook Fig. 3.5, p. 150): Each row corresponds to a different regularization strength. The left column shows many fitted curves from many data sets, and the right column shows their average. Strong regularization gives similar curves but a biased average. Weak regularization gives flexible curves with high variability. The middle setting balances the two effects.*

## 2.5 Bias-Variance Curves

The relationship can be summarized by plotting squared bias and variance as functions of $\ln\lambda$.

> ![Figure 3.6](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_6__textbook_fig_3_6_p151_bias_variance_curves.png)
>
> *Figure 3.6 (Textbook Fig. 3.6, p. 151): As regularization decreases, variance tends to increase and bias tends to decrease. As regularization increases, bias tends to increase and variance tends to decrease. Good generalization usually occurs near the minimum of their sum, not at the minimum of either term alone.*

## 2.6 Why the Decomposition Is Limited in Practice

Although the bias-variance decomposition is insightful, it has an important limitation. It requires an expectation over many independently sampled training sets. In real applications, we usually have only one training set.

Therefore, the decomposition is mainly a conceptual explanation of generalization, not a direct model-selection tool. It motivates the next part of the chapter: Bayesian methods, where uncertainty is represented through distributions over parameters rather than through hypothetical repeated data sets.

---

# §3 Bayesian Linear Regression

> 📖 Textbook §3.3 Bayesian Linear Regression; §3.3.1-§3.3.3
>
> **Teaching choice.** Keep the main story simple: Bayesian regression replaces one best-fit weight vector by a distribution over plausible weight vectors. Students should understand the predictive mean and predictive variance. The exact completion-of-square derivation can be assigned as reading or skipped.

## 3.1 From Point Estimates to Parameter Distributions

Maximum likelihood gives one fitted weight vector:

$$
\mathbf{w}_{\mathrm{ML}}.
$$

Bayesian linear regression instead treats $\mathbf{w}$ as an unknown random variable. Before observing data, we express uncertainty through a prior distribution. After observing data, we update that uncertainty using Bayes' theorem:

$$
p(\mathbf{w}\mid\mathbf{t})
=\frac{p(\mathbf{t}\mid\mathbf{w})p(\mathbf{w})}{p(\mathbf{t})}.
$$

The denominator

$$
p(\mathbf{t})=\int p(\mathbf{t}\mid\mathbf{w})p(\mathbf{w})\,d\mathbf{w}
$$

normalizes the posterior and will later become the model evidence.

The Bayesian view changes the learning question:

| Frequentist Least Squares | Bayesian Linear Regression |
|--------------------------|----------------------------|
| Find the best $\mathbf{w}$. | Infer a distribution over plausible $\mathbf{w}$. |
| Prediction is a point estimate. | Prediction is a distribution. |
| Regularization is a penalty. | Regularization comes from a prior. |
| Uncertainty often requires resampling or asymptotics. | Uncertainty follows from posterior variance. |

## 3.2 Gaussian Prior and Gaussian Posterior

A common prior is a zero-mean isotropic Gaussian:

$$
p(\mathbf{w}\mid\alpha)=\mathcal{N}(\mathbf{w}\mid\mathbf{0},\alpha^{-1}\mathbf{I}).
$$

Here $\alpha$ is the prior precision. A large $\alpha$ means a narrow prior and strong preference for small weights. A small $\alpha$ means a broad prior and weak regularization.

More generally, we can use

$$
p(\mathbf{w})=\mathcal{N}(\mathbf{w}\mid\mathbf{m}_0,\mathbf{S}_0).
$$

Because the likelihood is Gaussian in $\mathbf{w}$ and the prior is Gaussian, the posterior is also Gaussian:

$$
p(\mathbf{w}\mid\mathbf{t})=\mathcal{N}(\mathbf{w}\mid\mathbf{m}_N,\mathbf{S}_N).
$$

For the zero-mean isotropic prior, the posterior covariance and mean are

$$
\mathbf{S}_N^{-1}=\alpha\mathbf{I}+\beta\boldsymbol{\Phi}^T\boldsymbol{\Phi},
$$

$$
\mathbf{m}_N=\beta\mathbf{S}_N\boldsymbol{\Phi}^T\mathbf{t}.
$$

These equations show two forces.

- The prior contributes $\alpha\mathbf{I}$ to the posterior precision.
- The data contribute $\beta\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ to the posterior precision.

As more data arrive, the data term grows, and posterior uncertainty usually shrinks in directions informed by the data.

## 3.3 MAP Estimation and Ridge Regression

The posterior mode is the maximum-a-posteriori estimate:

$$
\mathbf{w}_{\mathrm{MAP}}=\arg\max_{\mathbf{w}}p(\mathbf{w}\mid\mathbf{t}).
$$

Because

$$
p(\mathbf{w}\mid\mathbf{t})\propto p(\mathbf{t}\mid\mathbf{w})p(\mathbf{w}),
$$

maximizing the posterior is equivalent to minimizing the negative log likelihood plus the negative log prior.

For a Gaussian likelihood and zero-mean Gaussian prior, the objective is

$$
\frac{\beta}{2}\sum_{n=1}^N\{t_n-\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2
+\frac{\alpha}{2}\mathbf{w}^T\mathbf{w}.
$$

Dividing by $\beta$ gives

$$
\frac{1}{2}\sum_{n=1}^N\{t_n-\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2
+\frac{\alpha}{2\beta}\mathbf{w}^T\mathbf{w}.
$$

Thus ridge regression corresponds to MAP estimation with

$$
\lambda=\frac{\alpha}{\beta}.
$$

This is the probabilistic interpretation of weight decay: it expresses a Gaussian prior belief that large weights are unlikely.

## 3.4 Sequential Bayesian Learning

Bayesian learning can be performed sequentially. Suppose after observing some data, the posterior is

$$
p(\mathbf{w}\mid\mathcal{D}_{\text{old}}).
$$

When a new data point arrives, we use the old posterior as the new prior:

$$
p(\mathbf{w}\mid\mathcal{D}_{\text{old}},t_{\text{new}})
\propto
p(t_{\text{new}}\mid\mathbf{w})p(\mathbf{w}\mid\mathcal{D}_{\text{old}}).
$$

This makes Bayesian learning naturally online. Unlike stochastic gradient descent, which updates a point estimate, sequential Bayesian learning updates a full distribution.

> ![Figure 3.7](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_7__textbook_fig_3_7_p155_sequential_bayesian_linear_regression.png)
>
> *Figure 3.7 (Textbook Fig. 3.7, p. 155): This figure is one of the most important visual explanations in the chapter. The first column shows probability mass in weight space $(w_0,w_1)$. The second column shows sampled lines from that weight distribution. The third column shows data space. As observations accumulate, the posterior over weights becomes concentrated near the true parameter values, and sampled regression lines become more consistent. This demonstrates that data reduce uncertainty, but they reduce it anisotropically: uncertainty shrinks faster in directions that the data constrain strongly.*

## 3.5 Predictive Distribution

Prediction should account for uncertainty in the weights. Instead of plugging in one weight vector, Bayesian regression integrates over the posterior:

$$
p(t\mid\mathbf{x},\mathbf{t},\alpha,\beta)
=\int p(t\mid\mathbf{x},\mathbf{w},\beta)p(\mathbf{w}\mid\mathbf{t},\alpha,\beta)\,d\mathbf{w}.
$$

Because both terms are Gaussian and the model is linear in $\mathbf{w}$, the predictive distribution is Gaussian:

$$
p(t\mid\mathbf{x},\mathbf{t},\alpha,\beta)
=\mathcal{N}\bigl(t\mid \mathbf{m}_N^T\boldsymbol{\phi}(\mathbf{x}),\sigma_N^2(\mathbf{x})\bigr),
$$

where

$$
\sigma_N^2(\mathbf{x})
=\frac{1}{\beta}+\boldsymbol{\phi}(\mathbf{x})^T\mathbf{S}_N\boldsymbol{\phi}(\mathbf{x}).
$$

This variance has two terms.

| Term | Meaning |
|------|---------|
| $\beta^{-1}$ | Noise in the target even if the true function is known. |
| $\boldsymbol{\phi}(\mathbf{x})^T\mathbf{S}_N\boldsymbol{\phi}(\mathbf{x})$ | Uncertainty due to not knowing the weights exactly. |

As the amount of data increases, the second term usually shrinks near regions supported by data. The first term does not vanish because it represents intrinsic observation noise.

> ![Figure 3.8](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_8__textbook_fig_3_8_p157_predictive_distribution_examples.png)
>
> *Figure 3.8 (Textbook Fig. 3.8, p. 157): The red curve is the predictive mean, and the shaded region represents predictive uncertainty. With few observations, uncertainty is large. As more data are observed, the predictive mean approaches the underlying sinusoidal trend and the uncertainty band shrinks near the training data. This figure should be connected directly to the formula for $\sigma_N^2(x)$.*

## 3.6 Samples from the Posterior over Functions

The predictive variance at each input gives pointwise uncertainty. But Bayesian linear regression also defines correlations between predictions at different input values. To visualize this joint uncertainty, we can sample weights from the posterior and plot the resulting functions.

> ![Figure 3.9](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_9__textbook_fig_3_9_p158_posterior_function_samples.png)
>
> *Figure 3.9 (Textbook Fig. 3.9, p. 158): Each red curve is a function obtained by sampling one weight vector from the posterior. Early in learning, sampled functions vary widely. Later, the sampled functions are more concentrated around the data-supported trend. This shows something that pointwise error bars alone cannot show: uncertainty is correlated across input locations.*

## 3.7 Equivalent Kernel

The posterior predictive mean can be written as

$$
y(\mathbf{x},\mathbf{m}_N)=\mathbf{m}_N^T\boldsymbol{\phi}(\mathbf{x}).
$$

Using

$$
\mathbf{m}_N=\beta\mathbf{S}_N\boldsymbol{\Phi}^T\mathbf{t},
$$

we get

$$
y(\mathbf{x},\mathbf{m}_N)
=\beta\boldsymbol{\phi}(\mathbf{x})^T\mathbf{S}_N\boldsymbol{\Phi}^T\mathbf{t}.
$$

This can be written as a weighted sum of training targets:

$$
y(\mathbf{x},\mathbf{m}_N)=\sum_{n=1}^N k(\mathbf{x},\mathbf{x}_n)t_n,
$$

where

$$
k(\mathbf{x},\mathbf{x}')
=\beta\boldsymbol{\phi}(\mathbf{x})^T\mathbf{S}_N\boldsymbol{\phi}(\mathbf{x}').
$$

This function is called the **equivalent kernel**. It tells us how strongly the target value at training input $\mathbf{x}'$ influences the prediction at test input $\mathbf{x}$.

> ![Figure 3.10](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_10__textbook_fig_3_10_p159_equivalent_kernel_gaussian_basis.png)
>
> *Figure 3.10 (Textbook Fig. 3.10, p. 159): Even though the model is expressed in terms of basis functions and weights, the predictive mean can be interpreted as a kernel smoother over training targets. For Gaussian basis functions, the equivalent kernel is localized: training points near the test input usually have stronger influence.*

> ![Figure 3.11](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_11__textbook_fig_3_11_p160_equivalent_kernel_polynomial_sigmoidal.png)
>
> *Figure 3.11 (Textbook Fig. 3.11, p. 160): The equivalent kernel can be localized even when the original basis functions themselves are global, as in polynomial or sigmoidal bases. This is an important conceptual bridge to kernel methods: we can sometimes think directly in terms of similarity functions rather than explicitly constructing basis functions.*

## 3.8 Why This Leads Toward Gaussian Processes

The equivalent-kernel view suggests a deeper idea. If prediction can be written using a kernel function, then perhaps we can define learning directly through kernels rather than through a finite set of basis functions.

This is the direction taken by Gaussian processes in Chapter 6. Bayesian linear regression with basis functions gives a finite-dimensional Gaussian distribution over weights. Gaussian processes instead define a distribution directly over functions. The covariance between function values is specified by a kernel.

Thus Chapter 3 prepares two future paths:

1. **Neural networks**: learn adaptive nonlinear basis functions.
2. **Kernel methods / Gaussian processes**: use kernels to define similarity and function uncertainty directly.

---

# §4 Bayesian Model Comparison

> 📖 Textbook §3.4 Bayesian Model Comparison
>
> **Teaching choice.** Treat this section as a conceptual bridge rather than a derivation-heavy topic. The main idea is that evidence rewards good fit but penalizes models that need very finely tuned parameters. Detailed Bayes-factor calculations are optional for this course.

## 4.1 The Model Selection Problem

So far we have assumed that the basis functions and hyperparameters are given. But in practice we must choose among different models. Examples include:

- polynomial order $M$;
- number and width of Gaussian basis functions;
- regularization strength;
- prior precision;
- different feature representations.

A frequentist approach often uses validation or cross-validation. A Bayesian approach uses posterior probabilities over models.

For candidate models $\mathcal{M}_1,\ldots,\mathcal{M}_K$, Bayes' theorem gives

$$
p(\mathcal{M}_i\mid\mathcal{D})
=\frac{p(\mathcal{D}\mid\mathcal{M}_i)p(\mathcal{M}_i)}{p(\mathcal{D})}.
$$

If the model priors $p(\mathcal{M}_i)$ are equal, model comparison is driven by

$$
p(\mathcal{D}\mid\mathcal{M}_i),
$$

called the **model evidence** or **marginal likelihood**.

## 4.2 Evidence as Parameter Marginalization

For a model with parameters $\mathbf{w}$, the evidence is

$$
p(\mathcal{D}\mid\mathcal{M})
=\int p(\mathcal{D}\mid\mathbf{w},\mathcal{M})p(\mathbf{w}\mid\mathcal{M})\,d\mathbf{w}.
$$

This integral averages the likelihood over the prior distribution of parameters. A model receives high evidence if it assigns high probability to the observed data after accounting for all possible parameter values under its prior.

The evidence is not simply the best possible likelihood. It rewards fit, but it also penalizes models whose good fit requires fine-tuned parameter values.

## 4.3 Occam Factor

Suppose the posterior over parameters is sharply peaked around $\mathbf{w}_{\mathrm{MAP}}$. Then the evidence can be roughly approximated as

$$
p(\mathcal{D})
\simeq
p(\mathcal{D}\mid\mathbf{w}_{\mathrm{MAP}})
\times
\frac{\Delta w_{\mathrm{posterior}}}{\Delta w_{\mathrm{prior}}}.
$$

Taking logs gives

$$
\ln p(\mathcal{D})
\simeq
\ln p(\mathcal{D}\mid\mathbf{w}_{\mathrm{MAP}})
+
\ln\left(\frac{\Delta w_{\mathrm{posterior}}}{\Delta w_{\mathrm{prior}}}\right).
$$

The first term rewards data fit. The second term is the Occam factor. Since the posterior width is usually smaller than the prior width, the ratio is less than one, so the log term is negative.

> ![Figure 3.12](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_12__textbook_fig_3_12_p163_evidence_occam_factor.png)
>
> *Figure 3.12 (Textbook Fig. 3.12, p. 163): A complex model may fit the data well for some parameter settings, but if those settings occupy only a tiny fraction of the prior parameter space, the model pays a complexity penalty. Evidence automatically balances goodness of fit against the volume of parameter space that supports that fit.*

## 4.4 Automatic Occam's Razor

The evidence implements an automatic Occam's razor. It does not always prefer the simplest model, and it does not always prefer the most flexible model.

- A too-simple model gives low likelihood because it cannot fit the data.
- A too-complex model spreads probability mass over too many possible data sets, so it may assign lower probability density to the observed data.
- An intermediate model can achieve the highest evidence.

> ![Figure 3.13](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_13__textbook_fig_3_13_p164_evidence_model_complexity.png)
>
> *Figure 3.13 (Textbook Fig. 3.13, p. 164): The horizontal axis represents possible data sets. Simple models concentrate probability on a limited range of data sets. Complex models spread probability over a broader range. For the observed data set $\mathcal{D}_0$, the intermediate model may assign the largest probability. This is a conceptual explanation of why Bayesian model comparison can prefer an intermediate complexity.*

## 4.5 Bayes Factors

The ratio of evidences for two models is called the Bayes factor:

$$
\frac{p(\mathcal{D}\mid\mathcal{M}_i)}{p(\mathcal{D}\mid\mathcal{M}_j)}.
$$

If model priors are equal, the Bayes factor is also the posterior odds ratio. A Bayes factor larger than one supports $\mathcal{M}_i$ over $\mathcal{M}_j$; a Bayes factor smaller than one supports $\mathcal{M}_j$ over $\mathcal{M}_i$.

Bayesian model comparison is powerful, but it depends on the prior. The evidence is not defined for improper priors, because the normalization of the prior affects the marginal likelihood. In practical applications, it is often wise to evaluate final predictive performance on independent test data as well.

---

# §5 The Evidence Approximation

> 📖 Textbook §3.5 The Evidence Approximation; §3.5.1-§3.5.3
>
> **Teaching choice.** This is the most mathematically dense part of Chapter 3. For a modern ML course aimed at EE students, it is enough to explain empirical Bayes, effective number of parameters, and why evidence can tune regularization. The re-estimation equations for $\alpha$ and $\beta$ are useful reference material but do not need to be derived line by line in lecture.

## 5.1 Hyperparameters in Bayesian Linear Regression

Bayesian linear regression with a zero-mean isotropic Gaussian prior has two important hyperparameters:

$$
\alpha \quad \text{and} \quad \beta.
$$

Here

- $\alpha$ controls the prior precision over weights;
- $\beta$ controls the noise precision of the targets.

A fully Bayesian approach would place priors over $\alpha$ and $\beta$ and integrate them out. The evidence approximation, also called **empirical Bayes** or **type-2 maximum likelihood**, instead chooses $\alpha$ and $\beta$ by maximizing the marginal likelihood:

$$
p(\mathbf{t}\mid\alpha,\beta)
=\int p(\mathbf{t}\mid\mathbf{w},\beta)p(\mathbf{w}\mid\alpha)\,d\mathbf{w}.
$$

This is an intermediate approach:

| Approach | Treatment of $\mathbf{w}$ | Treatment of $\alpha,\beta$ |
|----------|----------------------------|------------------------------|
| ML | Point estimate | Usually point estimate |
| MAP | Point estimate with prior | Usually fixed |
| Evidence approximation | Integrated out | Optimized by marginal likelihood |
| Full Bayes | Integrated out | Integrated out |

## 5.2 Evaluating the Evidence Function

For Bayesian linear regression,

$$
p(\mathbf{t}\mid\alpha,\beta)
=\int p(\mathbf{t}\mid\mathbf{w},\beta)p(\mathbf{w}\mid\alpha)\,d\mathbf{w}
$$

can be evaluated in closed form because both the likelihood and prior are Gaussian.

Define the regularized error

$$
E(\mathbf{w})=\beta E_D(\mathbf{w})+\alpha E_W(\mathbf{w}),
$$

where

$$
E_D(\mathbf{w})=\frac{1}{2}\|\mathbf{t}-\boldsymbol{\Phi}\mathbf{w}\|^2,
\qquad
E_W(\mathbf{w})=\frac{1}{2}\mathbf{w}^T\mathbf{w}.
$$

The posterior precision matrix is

$$
\mathbf{A}=\alpha\mathbf{I}+\beta\boldsymbol{\Phi}^T\boldsymbol{\Phi}.
$$

The log evidence has the form

$$
\ln p(\mathbf{t}\mid\alpha,\beta)
=
\frac{M}{2}\ln\alpha+\frac{N}{2}\ln\beta
-E(\mathbf{m}_N)
-\frac{1}{2}\ln|\mathbf{A}|
-\frac{N}{2}\ln(2\pi).
$$

Each term has an interpretation.

| Term | Interpretation |
|------|----------------|
| $\frac{M}{2}\ln\alpha$ | Prior normalization contribution. |
| $\frac{N}{2}\ln\beta$ | Noise likelihood normalization contribution. |
| $-E(\mathbf{m}_N)$ | Penalized data fit at posterior mean. |
| $-\frac{1}{2}\ln|\mathbf{A}|$ | Complexity/volume correction. |
| $-\frac{N}{2}\ln(2\pi)$ | Gaussian normalization constant. |

> ![Figure 3.14](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_14__textbook_fig_3_14_p168_model_evidence_polynomial_order.png)
>
> *Figure 3.14 (Textbook Fig. 3.14, p. 168): The evidence curve favors a polynomial order around $M=3$ for the synthetic curve-fitting problem. Low-order models underfit and have poor data fit. Very high-order models are penalized by the evidence because their extra flexibility is not sufficiently supported by the data. This is the evidence version of model selection.*

## 5.3 Maximizing the Evidence with Respect to $\alpha$

To maximize the evidence, Bishop derives re-estimation equations. Let $\lambda_i$ be eigenvalues of

$$
\beta\boldsymbol{\Phi}^T\boldsymbol{\Phi}.
$$

Define

$$
\gamma=\sum_i\frac{\lambda_i}{\alpha+\lambda_i}.
$$

Then the evidence update for $\alpha$ can be written as

$$
\alpha=\frac{\gamma}{\mathbf{m}_N^T\mathbf{m}_N}.
$$

This equation is implicit because $\gamma$ and $\mathbf{m}_N$ both depend on $\alpha$. In practice, one iterates:

1. initialize $\alpha$ and $\beta$;
2. compute $\mathbf{m}_N$ and $\mathbf{A}$;
3. compute $\gamma$;
4. update $\alpha$ and $\beta$;
5. repeat until convergence.

## 5.4 Effective Number of Parameters

The quantity

$$
\gamma=\sum_i\frac{\lambda_i}{\alpha+\lambda_i}
$$

is called the **effective number of parameters**.

Each term

$$
\frac{\lambda_i}{\alpha+\lambda_i}
$$

lies between 0 and 1.

- If $\lambda_i\gg\alpha$, then the term is close to 1. The data strongly determine that parameter direction.
- If $\lambda_i\ll\alpha$, then the term is close to 0. The prior dominates that direction.

Therefore, $\gamma$ counts how many parameter directions are effectively determined by the data.

This is more nuanced than simply counting $M$ parameters. A model may have many weights, but if strong regularization or weak data support prevents many directions from being used, the effective number of parameters may be much smaller than $M$.

> ![Figure 3.15](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_15__textbook_fig_3_15_p170_effective_parameters_likelihood_prior.png)
>
> *Figure 3.15 (Textbook Fig. 3.15, p. 170): The likelihood contours are elongated in directions that the data constrain weakly and narrow in directions they constrain strongly. The prior pulls the posterior mode toward the origin. Directions with high likelihood curvature are well determined by data; directions with low curvature are dominated by the prior. This is the geometric meaning of effective parameters.*

## 5.5 Maximizing the Evidence with Respect to $\beta$

The corresponding update for the noise precision is

$$
\frac{1}{\beta}
=\frac{1}{N-\gamma}\sum_{n=1}^N\{t_n-\mathbf{m}_N^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2.
$$

This resembles the maximum-likelihood noise variance estimate, but with $N$ replaced by $N-\gamma$. This is analogous to degrees-of-freedom correction: if the model has effectively used $\gamma$ parameters to fit the data, the residual variance should be normalized by the remaining degrees of freedom.

## 5.6 Evidence Curves for $\alpha$

> ![Figure 3.16](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_16__textbook_fig_3_16_p172_evidence_alpha_gamma_curves.png)
>
> *Figure 3.16 (Textbook Fig. 3.16, p. 172): The left panel shows the intersection condition that defines the evidence optimum for $\alpha$. The right panel shows the log evidence as a function of $\ln\alpha$. This visualization helps students see that evidence optimization is not arbitrary tuning; it chooses the regularization strength that best balances parameter shrinkage and data fit under the Bayesian model.*

## 5.7 Weight Paths and Effective Parameters

> ![Figure 3.17](./CoursePR2026/Fig/Chapter_3/lecture_fig_3_17__textbook_fig_3_17_p172_weights_vs_effective_parameters.png)
>
> *Figure 3.17 (Textbook Fig. 3.17, p. 172): As $\alpha$ changes, $\gamma$ changes from near 0 to near $M$. The plotted weight paths show how parameters become active as the effective number of parameters increases. This is a useful way to explain that regularization does not merely make all weights smaller; it changes which directions of parameter space are allowed to be expressed by the data.*

## 5.8 Evidence Approximation versus Cross-Validation

Cross-validation estimates generalization performance by repeatedly splitting data into training and validation sets. Evidence approximation estimates model plausibility by integrating over parameters and maximizing marginal likelihood.

| Criterion | Cross-Validation | Evidence Approximation |
|----------|------------------|------------------------|
| Uses validation data? | Yes | No, uses all data in evidence. |
| Computational cost | Can be high for many folds and hyperparameters. | Often efficient for conjugate Gaussian models. |
| Probabilistic interpretation | Indirect | Direct marginal likelihood. |
| Prior dependence | Usually less explicit | Explicit and important. |
| Output | Selected hyperparameters/model | Selected hyperparameters/model, plus posterior. |

Neither method is universally superior. Evidence is elegant and data-efficient when the model assumptions and priors are reasonable. Cross-validation is often more robust when the probabilistic model is misspecified.

---

# §6 Limitations of Fixed Basis Functions

> 📖 Textbook §3.6 Limitations of Fixed Basis Functions

## 6.1 The Problem with Fixed Bases in High Dimensions

Linear basis-function models are powerful in one-dimensional examples, but fixed basis functions face serious limitations in high-dimensional input spaces.

If we place basis functions throughout a $D$-dimensional input region, the number of basis functions needed can grow exponentially with $D$. This is another manifestation of the curse of dimensionality.

For example, suppose we use $K$ basis functions along each input dimension. A full grid requires

$$
K^D
$$

basis functions. Even with $K=10$, this becomes impossible for moderately large $D$:

| Dimension $D$ | Number of grid basis functions $10^D$ |
|--------------|----------------------------------------|
| 1 | 10 |
| 2 | 100 |
| 3 | 1,000 |
| 10 | 10,000,000,000 |
| 100 | impossible to enumerate |

This is why naive fixed basis expansions do not scale to modern high-dimensional data such as images, speech, text embeddings, or sensor arrays.

## 6.2 Fixed Basis Functions Do Not Adapt to Data

A fixed basis model chooses basis functions before seeing the target structure. This can be wasteful. Many basis functions may be placed in regions with little data, while important regions may not receive enough resolution.

The ideal representation should adapt to the data distribution and the target function. This motivates two major directions in the rest of the course.

## 6.3 Roadmap to Later Chapters

Chapter 3 ends by pointing toward more flexible approaches.

| Limitation of Fixed Bases | Later Solution | Course Location |
|---------------------------|----------------|-----------------|
| Too many basis functions in high dimensions | Learn adaptive hidden representations | Neural networks, Chapter 5 |
| Explicit basis construction can be expensive | Use dual/kernel representations | Kernel methods, Chapter 6 |
| Need sparse decision functions | Use sparse kernel machines | SVM/RVM, Chapter 7 |
| Need distributions over functions | Use Gaussian processes | Chapter 6 |

Thus Chapter 3 is not the final answer to regression. It is the mathematical foundation for later models.

## 6.4 What Students Should Remember

The central lessons are:

1. A model can be nonlinear in inputs while linear in weights.
2. Least squares is maximum likelihood under Gaussian noise.
3. Regularization controls complexity and has a Bayesian interpretation as a prior.
4. Bias-variance explains why both underfitting and overfitting are harmful.
5. Bayesian linear regression gives posterior and predictive uncertainty.
6. Evidence provides a Bayesian way to compare models and tune hyperparameters.
7. Fixed basis functions become limited in high dimensions, motivating neural networks and kernels.

---

# §7 Chapter Summary, Figure Checklist, Exercises, and Teaching Flow

## 7.1 Chapter Summary

Chapter 3 develops linear regression from three complementary perspectives.

First, from the optimization perspective, it studies least squares, normal equations, pseudo-inverse solutions, stochastic gradient updates, and regularized objectives.

Second, from the frequentist generalization perspective, it studies the bias-variance decomposition and explains the trade-off between rigid models and overly sensitive models.

Third, from the Bayesian perspective, it places a prior over weights, derives a Gaussian posterior, computes the predictive distribution, introduces equivalent kernels, and uses evidence for model comparison and hyperparameter estimation.

The chapter is conceptually important because many later models preserve the same structure while changing one component:

| Later Model | What Changes? | What Remains from Chapter 3? |
|-------------|---------------|------------------------------|
| Logistic regression | Target distribution becomes Bernoulli/multinomial | Linear basis-function structure. |
| Neural networks | Basis functions become adaptive | Weighted combinations and regularization. |
| Kernel ridge regression | Basis functions are implicit | Equivalent kernel and linear smoother view. |
| Gaussian processes | Distribution is over functions | Bayesian predictive uncertainty. |
| Relevance vector machines | Sparse Bayesian prior over weights | Evidence and effective parameters. |

## 7.2 Figure Checklist

All figures used in this lecture are screenshots/crops from the uploaded textbook PDF. Each filename records both the lecture figure number and the original textbook figure number.

| Lecture Figure | Textbook Figure | Topic | File |
|----------------|-----------------|-------|------|
| 3.1 | 3.1 | Polynomial, Gaussian, sigmoidal basis functions | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_1__textbook_fig_3_1_p140_basis_functions.png` |
| 3.2 | 3.2 | Geometry of least-squares projection | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_2__textbook_fig_3_2_p143_least_squares_geometry.png` |
| 3.3 | 3.3 | $q$-norm regularization contours | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_3__textbook_fig_3_3_p145_regularization_q_norm_contours.png` |
| 3.4 | 3.4 | Lasso sparsity and constrained optimization | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_4__textbook_fig_3_4_p146_lasso_sparsity_constraint_geometry.png` |
| 3.5 | 3.5 | Bias-variance examples under different $\lambda$ | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_5__textbook_fig_3_5_p150_bias_variance_lambda_fits.png` |
| 3.6 | 3.6 | Bias, variance, and test error curves | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_6__textbook_fig_3_6_p151_bias_variance_curves.png` |
| 3.7 | 3.7 | Sequential Bayesian learning | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_7__textbook_fig_3_7_p155_sequential_bayesian_linear_regression.png` |
| 3.8 | 3.8 | Bayesian predictive distribution | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_8__textbook_fig_3_8_p157_predictive_distribution_examples.png` |
| 3.9 | 3.9 | Posterior samples of regression functions | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_9__textbook_fig_3_9_p158_posterior_function_samples.png` |
| 3.10 | 3.10 | Equivalent kernel for Gaussian basis functions | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_10__textbook_fig_3_10_p159_equivalent_kernel_gaussian_basis.png` |
| 3.11 | 3.11 | Equivalent kernels for polynomial/sigmoidal bases | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_11__textbook_fig_3_11_p160_equivalent_kernel_polynomial_sigmoidal.png` |
| 3.12 | 3.12 | Evidence approximation and Occam factor | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_12__textbook_fig_3_12_p163_evidence_occam_factor.png` |
| 3.13 | 3.13 | Evidence and model complexity | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_13__textbook_fig_3_13_p164_evidence_model_complexity.png` |
| 3.14 | 3.14 | Evidence versus polynomial order | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_14__textbook_fig_3_14_p168_model_evidence_polynomial_order.png` |
| 3.15 | 3.15 | Effective parameter directions | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_15__textbook_fig_3_15_p170_effective_parameters_likelihood_prior.png` |
| 3.16 | 3.16 | Evidence optimization over $\alpha$ | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_16__textbook_fig_3_16_p172_evidence_alpha_gamma_curves.png` |
| 3.17 | 3.17 | Weight paths versus effective number of parameters | `./CoursePR2026/Fig/Chapter_3/lecture_fig_3_17__textbook_fig_3_17_p172_weights_vs_effective_parameters.png` |

## 7.3 Exercise Checklist

| Lecture Exercise | Textbook Exercise | Topic | File |
|------------------|-------------------|-------|------|
| 3.3 | 3.3 | Weighted least squares and data weights | `./CoursePR2026/Fig/Chapter_3/lecture_ex_3_3__textbook_ex_3_3_p174.png` |
| 3.5 | 3.5 | Regularization as constrained optimization | `./CoursePR2026/Fig/Chapter_3/lecture_ex_3_5__textbook_ex_3_5_p174.png` |

## 7.4 Suggested Teaching Flow

A practical lecture sequence is:

1. Begin with the distinction between linear in $\mathbf{x}$ and linear in $\mathbf{w}$.
2. Use Figure 3.1 to show that basis functions determine model shape.
3. Derive least squares from the Gaussian likelihood.
4. Explain the design matrix and normal equations.
5. Work through Exercise 3.3 to show how ordinary least squares changes when examples have different reliability.
6. Use Figure 3.2 to interpret least squares as projection.
7. Introduce regularization and show Figures 3.3-3.4 for ridge versus lasso geometry.
8. Work through Exercise 3.5 to connect penalized and constrained views of regularization.
9. Use Figures 3.5-3.6 to explain bias and variance.
10. Transition to Bayesian regression by asking: instead of one weight vector, why not maintain uncertainty over weights?
11. Use Figure 3.7 to explain sequential posterior updating.
12. Derive the predictive distribution and explain Figures 3.8-3.9.
13. Use Figures 3.10-3.11 briefly to foreshadow kernels and Gaussian processes.
14. Explain evidence and Occam factors conceptually using Figures 3.12-3.14.
15. Treat the detailed evidence approximation equations and Figures 3.15-3.17 as optional depth.
16. Close by explaining why fixed bases do not scale and how this motivates later chapters.

## 7.5 Key Equations to Put on the Board

The following equations are the minimum board set for this chapter.

### Linear basis-function model

$$
y(\mathbf{x},\mathbf{w})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}).
$$

### Gaussian likelihood

$$
p(t\mid\mathbf{x},\mathbf{w},\beta)
=\mathcal{N}\bigl(t\mid\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}),\beta^{-1}\bigr).
$$

### Least-squares solution

$$
\mathbf{w}_{\mathrm{ML}}
=(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{t}.
$$

### Ridge solution

$$
\mathbf{w}
=(\lambda\mathbf{I}+\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{t}.
$$

### Bias-variance-noise decomposition

$$
\mathbb{E}[\text{squared error}]
=\text{bias}^2+\text{variance}+\text{noise}.
$$

### Bayesian posterior

$$
\mathbf{S}_N^{-1}=\alpha\mathbf{I}+\beta\boldsymbol{\Phi}^T\boldsymbol{\Phi},
\qquad
\mathbf{m}_N=\beta\mathbf{S}_N\boldsymbol{\Phi}^T\mathbf{t}.
$$

### Predictive distribution

$$
p(t\mid\mathbf{x},\mathbf{t},\alpha,\beta)
=\mathcal{N}\bigl(t\mid\mathbf{m}_N^T\boldsymbol{\phi}(\mathbf{x}),\sigma_N^2(\mathbf{x})\bigr),
$$

$$
\sigma_N^2(\mathbf{x})=\beta^{-1}+\boldsymbol{\phi}(\mathbf{x})^T\mathbf{S}_N\boldsymbol{\phi}(\mathbf{x}).
$$

### Evidence

$$
p(\mathbf{t}\mid\alpha,\beta)
=\int p(\mathbf{t}\mid\mathbf{w},\beta)p(\mathbf{w}\mid\alpha)\,d\mathbf{w}.
$$

### Effective number of parameters

$$
\gamma=\sum_i\frac{\lambda_i}{\alpha+\lambda_i}.
$$

## 7.6 Common Student Misunderstandings

| Misunderstanding | Correction |
|------------------|------------|
| “Linear model means a straight line.” | It means linear in parameters. Basis functions can make the curve nonlinear in the input. |
| “Least squares is arbitrary.” | It is ML under Gaussian noise. Changing the noise model changes the loss. |
| “Regularization just makes weights small.” | It controls model complexity and can be interpreted as a prior. For lasso, it can also create sparsity. |
| “Low training error means good model.” | Low training error may mean overfitting. Generalization depends on bias, variance, and noise. |
| “Bayesian regression only gives another fitted curve.” | It gives a posterior over weights and a predictive distribution over targets. |
| “Evidence always chooses the simplest model.” | Evidence balances fit and complexity; it often prefers intermediate complexity. |
| “The number of parameters is always $M$.” | In the evidence framework, the effective number of parameters is $\gamma$, which can be smaller than $M$. |

## 7.7 One-Sentence Takeaway

Linear models for regression are the simplest setting in which we can see the full machine-learning pipeline: representation through basis functions, fitting by likelihood, complexity control by regularization or priors, uncertainty through Bayesian prediction, and model comparison through evidence.
