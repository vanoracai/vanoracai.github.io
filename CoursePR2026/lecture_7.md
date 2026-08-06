---
layout: course
title: PRML Lecture 7
---

# Pattern Recognition and Machine Learning
## Chapter 7: Sparse Kernel Machines

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 7 Sparse Kernel Machines (§7.1-§7.2)

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Maximum-Margin Classification: The Core SVM Idea](#1-maximum-margin-classification-the-core-svm-idea)
3. [§2 Hard-Margin SVM: From Geometry to the Dual Problem](#2-hard-margin-svm-from-geometry-to-the-dual-problem)
4. [§3 Soft-Margin SVM for Overlapping Classes](#3-soft-margin-svm-for-overlapping-classes)
5. [§4 Loss Functions, Probabilities, and Multiclass SVMs](#4-loss-functions-probabilities-and-multiclass-svms)
6. [§5 Support Vector Regression](#5-support-vector-regression)
7. [§6 Computational Learning Theory: A Gentle Introduction](#6-computational-learning-theory-a-gentle-introduction)
8. [§7 Relevance Vector Machines for Regression](#7-relevance-vector-machines-for-regression)
9. [§8 Why the RVM Becomes Sparse](#8-why-the-rvm-becomes-sparse)
10. [§9 Relevance Vector Machines for Classification](#9-relevance-vector-machines-for-classification)
11. [§10 Worked Examples and Textbook Exercises](#10-worked-examples-and-textbook-exercises)
12. [§11 Chapter Summary, Figure Checklist, and Teaching Flow](#11-chapter-summary-figure-checklist-and-teaching-flow)

---

## Notation and Variable Definitions

This chapter introduces two families of sparse kernel machines:

- the **support vector machine (SVM)**, based on margin maximization and convex optimization;
- the **relevance vector machine (RVM)**, based on Bayesian linear models, automatic relevance determination, and evidence maximization.

The word **sparse** has a very specific meaning here:

> After training, prediction depends on only a relatively small subset of the training examples or basis functions.

This is different from saying that the input vector itself contains many zeros.

### Core Classification Notation

| Symbol | Definition |
|---|---|
| $\mathbf{x}_n$ | The $n$th training input vector. |
| $t_n\in\{-1,+1\}$ | Binary class label used by the SVM. |
| $\boldsymbol{\phi}(\mathbf{x})$ | Feature-space representation of input $\mathbf{x}$. It may be very high-dimensional or implicit. |
| $\mathbf{w}$ | Weight vector defining a linear separator in feature space. |
| $b$ | Bias/intercept term. |
| $y(\mathbf{x})$ | Discriminant score, usually $y(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b$. |
| $k(\mathbf{x},\mathbf{x}')$ | Kernel function, $k(\mathbf{x},\mathbf{x}')=\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}')$. |
| $a_n$ | Lagrange multiplier associated with the margin constraint for point $n$. |
| $\xi_n$ | Slack variable measuring how much point $n$ violates the margin requirement. |
| $C$ | Soft-margin penalty parameter controlling the trade-off between margin size and margin violations. |
| $\nu$ | Alternative complexity parameter used by $\nu$-SVM formulations. |
| $\mathcal{S}$ | Set of support-vector indices. |

### Regression Notation

| Symbol | Definition |
|---|---|
| $t_n\in\mathbb{R}$ | Continuous regression target. |
| $\varepsilon$ | Width of the insensitive tube in support vector regression. |
| $\xi_n,\widehat{\xi}_n$ | Positive deviations above and below the $\varepsilon$-tube. |
| $a_n,\widehat{a}_n$ | Dual variables for the upper and lower tube constraints. |

### RVM and Bayesian Notation

| Symbol | Definition |
|---|---|
| $\Phi$ | Design matrix with entries $\Phi_{ni}=\phi_i(\mathbf{x}_n)$. |
| $\boldsymbol{\alpha}=(\alpha_1,\ldots,\alpha_M)^T$ | Individual precision hyperparameters for the weights. |
| $\mathbf{A}=\operatorname{diag}(\alpha_1,\ldots,\alpha_M)$ | Diagonal matrix of weight precisions. |
| $\beta$ | Noise precision, equal to inverse noise variance. |
| $\mathbf{m}$ | Posterior mean of the RVM weights. |
| $\boldsymbol{\Sigma}$ | Posterior covariance of the RVM weights. |
| $\gamma_i$ | Effective degree of determination of weight $w_i$, $\gamma_i=1-\alpha_i\Sigma_{ii}$. |
| relevance vector | A training point whose associated kernel basis function survives ARD pruning. |

> **Teaching focus.** The conceptual pipeline for this chapter is
>
> $$
> \text{large kernel expansion}
> \longrightarrow
> \text{sparsity mechanism}
> \longrightarrow
> \text{few active training points}
> \longrightarrow
> \text{fast prediction}.
> $$
>
> The SVM obtains sparsity from the geometry of the margin and KKT conditions. The RVM obtains sparsity from Bayesian hyperparameters that drive unnecessary weights to zero.

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch. 7 opening, pp. 325-326

## 0.1 Why Sparse Kernel Machines Are Needed

In Chapter 6, a kernel model often used all $N$ training points when making a prediction. A typical kernel predictor has the form

$$
y(\mathbf{x})=\sum_{n=1}^{N}c_n k(\mathbf{x},\mathbf{x}_n).
$$

This is flexible, but it creates two practical costs.

### Training cost

The model may need an $N\times N$ Gram matrix:

$$
\mathbf{K}_{nm}=k(\mathbf{x}_n,\mathbf{x}_m).
$$

Storing this matrix requires roughly $O(N^2)$ memory, and solving the corresponding optimization problem may require up to $O(N^3)$ computation in a naive implementation.

### Prediction cost

For every new input $\mathbf{x}$, the model may need to evaluate the kernel against every training point:

$$
k(\mathbf{x},\mathbf{x}_1),\ldots,k(\mathbf{x},\mathbf{x}_N).
$$

If only a small number of coefficients are nonzero, prediction becomes

$$
y(\mathbf{x})=\sum_{n\in\mathcal{S}}c_n k(\mathbf{x},\mathbf{x}_n),
$$

where $|\mathcal{S}|\ll N$.

That is the central goal of sparse kernel machines.

## 0.2 Two Routes to Sparsity

| Model | Sparsity mechanism | Optimization character | Output |
|---|---|---|---|
| **SVM** | KKT conditions make most dual coefficients exactly zero. | Convex quadratic program. | Decision score/class label; probability requires an extra calibration stage. |
| **RVM** | ARD evidence maximization sends many precisions $\alpha_i$ to infinity, shrinking weights to zero. | Nonconvex evidence optimization. | Posterior predictive distribution. |

The two methods can look similar at prediction time:

$$
\text{SVM: }y(\mathbf{x})=\sum_{n\in\mathcal{S}}a_nt_nk(\mathbf{x},\mathbf{x}_n)+b,
$$

$$
\text{RVM: }y(\mathbf{x})=\sum_{n\in\mathcal{R}}w_nk(\mathbf{x},\mathbf{x}_n)+b.
$$

But the reasons that the sets $\mathcal{S}$ and $\mathcal{R}$ are small are fundamentally different.

## 0.3 Recommended Learning Order

A good order for learning this chapter is:

1. Understand the geometry of a separating hyperplane.
2. Derive the hard-margin SVM from the geometric margin.
3. Convert the constrained primal problem into a kernelized dual problem.
4. Use KKT conditions to explain support vectors and sparsity.
5. Add slack variables for nonseparable data.
6. Compare hinge loss with logistic loss.
7. Extend the same idea to regression using an $\varepsilon$-insensitive tube.
8. Introduce the RVM as a Bayesian sparse linear model.
9. Understand how individual precision parameters prune basis functions.
10. Compare SVM and RVM rather than treating one as universally superior.

---

# §1 Maximum-Margin Classification: The Core SVM Idea

> 📖 Textbook §7.1, pp. 326-328

## 1.1 Binary Linear Classification in Feature Space

We begin with a two-class classification problem. Each target is encoded as

$$
t_n\in\{-1,+1\}.
$$

We use a linear discriminant in feature space:

$$
y(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b.
$$

The classification rule is

$$
\widehat{t}(\mathbf{x})=
\begin{cases}
+1, & y(\mathbf{x})>0,\\
-1, & y(\mathbf{x})<0.
\end{cases}
$$

The decision boundary is the hyperplane

$$
y(\mathbf{x})=0.
$$

A training point is correctly classified exactly when

$$
t_ny(\mathbf{x}_n)>0.
$$

Why does this compact condition work?

- If $t_n=+1$, then correctness requires $y(\mathbf{x}_n)>0$, so the product is positive.
- If $t_n=-1$, then correctness requires $y(\mathbf{x}_n)<0$, and the product of two negative values is again positive.

Thus one inequality handles both classes.

## 1.2 Why Merely Separating the Data Is Not Enough

If the classes are linearly separable, many hyperplanes may classify every training point correctly. The perceptron can find one such hyperplane, but the answer depends on initialization and data order.

The SVM asks a stronger question:

> Among all separating hyperplanes, which one stays as far as possible from the closest training points?

The distance from the boundary to the closest training point is the **margin**.

> ![Figure 7.1](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_1__textbook_fig_7_1__p327.png)
>
> *Figure 7.1 (Textbook Fig. 7.1, p. 327): Many boundaries may separate the two classes, but maximizing the distance to the closest points selects a particular boundary. The circled points that determine the solution are the support vectors.*

The central intuition is:

- A small-margin boundary is fragile. A small perturbation of a nearby point may change the decision.
- A large-margin boundary leaves a wider safety region between the classes.
- Only the points closest to the boundary determine the maximum-margin solution.

## 1.3 Distance from a Point to a Hyperplane

The formula

$$
\frac{|y(\mathbf{x})|}{\|\mathbf{w}\|}
$$

for the perpendicular distance to the hyperplane is worth deriving slowly.

Take any point $\mathbf{x}_0$ on the decision boundary, so

$$
\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_0)+b=0.
$$

The vector $\mathbf{w}$ is normal to the hyperplane. Therefore a unit normal vector is

$$
\frac{\mathbf{w}}{\|\mathbf{w}\|}.
$$

Any point $\boldsymbol{\phi}(\mathbf{x})$ can be decomposed into a component parallel to the hyperplane and a perpendicular displacement. The signed perpendicular distance $r$ satisfies

$$
\boldsymbol{\phi}(\mathbf{x})
=
\boldsymbol{\phi}(\mathbf{x}_0)
+r\frac{\mathbf{w}}{\|\mathbf{w}\|}
+\text{parallel component}.
$$

Multiplying by $\mathbf{w}^T$ removes the parallel component:

$$
\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})
=
\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_0)
+r\|\mathbf{w}\|.
$$

Using

$$
\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_0)=-b,
$$

we obtain

$$
\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b=r\|\mathbf{w}\|.
$$

Therefore

$$
r=\frac{y(\mathbf{x})}{\|\mathbf{w}\|},
$$

and the unsigned distance is

$$
\boxed{\operatorname{dist}(\mathbf{x},y=0)=\frac{|y(\mathbf{x})|}{\|\mathbf{w}\|}.}
$$

For a correctly classified training point, $t_ny(\mathbf{x}_n)>0$, so its distance can be written without an absolute value:

$$
\frac{t_ny(\mathbf{x}_n)}{\|\mathbf{w}\|}.
$$

## 1.4 The Geometric Margin

The closest training point has distance

$$
\rho
=
\min_n\frac{t_n(\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)+b)}{\|\mathbf{w}\|}.
$$

The maximum-margin problem is therefore

$$
\max_{\mathbf{w},b}
\left[
\frac{1}{\|\mathbf{w}\|}
\min_n t_n(\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)+b)
\right].
$$

This expression is correct but inconvenient because it contains both a minimum and a ratio. The next step removes this difficulty using a scale invariance.

## 1.5 Scale Invariance and the Canonical Representation

If we multiply both $\mathbf{w}$ and $b$ by any positive constant $\kappa$, then

$$
y(\mathbf{x})\rightarrow \kappa y(\mathbf{x}).
$$

The decision boundary does not change because

$$
y(\mathbf{x})=0
\quad\Longleftrightarrow\quad
\kappa y(\mathbf{x})=0.
$$

The geometric distance also does not change:

$$
\frac{|\kappa y(\mathbf{x})|}{\|\kappa\mathbf{w}\|}
=
\frac{|y(\mathbf{x})|}{\|\mathbf{w}\|}.
$$

We can therefore choose a convenient scaling. The canonical choice makes the closest point satisfy

$$
t_n(\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)+b)=1.
$$

Then every training point satisfies

$$
\boxed{t_n(\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)+b)\geq 1.}
$$

Under this scaling, the closest point has distance

$$
\rho=\frac{1}{\|\mathbf{w}\|}.
$$

The two supporting hyperplanes are

$$
y(\mathbf{x})=+1,
\qquad
y(\mathbf{x})=-1.
$$

Each lies at distance $1/\|\mathbf{w}\|$ from the decision boundary, so the full width between them is

$$
\frac{2}{\|\mathbf{w}\|}.
$$

> **Common convention warning.** Some sources call $1/\|\mathbf{w}\|$ the margin, while others call $2/\|\mathbf{w}\|$ the margin width. Always check which convention is being used.

---

# §2 Hard-Margin SVM: From Geometry to the Dual Problem

> 📖 Textbook §7.1, pp. 328-331

## 2.1 The Primal Optimization Problem

Maximizing $1/\|\mathbf{w}\|$ is equivalent to minimizing $\|\mathbf{w}\|$. For mathematical convenience, we minimize one half of the squared norm:

$$
\boxed{
\min_{\mathbf{w},b}\frac{1}{2}\|\mathbf{w}\|^2
}
$$

subject to

$$
\boxed{
t_n(\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)+b)\geq 1,
\qquad n=1,\ldots,N.
}
$$

This is a convex quadratic program:

- the objective is quadratic and convex;
- each constraint is linear in $\mathbf{w}$ and $b$;
- the feasible region is convex.

Therefore every local optimum is a global optimum.

## 2.2 Why We Introduce Lagrange Multipliers

The constraints couple all training points to the same $\mathbf{w}$ and $b$. The Lagrangian combines the objective and constraints into one function:

$$
L(\mathbf{w},b,\mathbf{a})
=
\frac{1}{2}\|\mathbf{w}\|^2
-
\sum_{n=1}^{N}a_n
\left[
 t_n(\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)+b)-1
\right],
$$

with

$$
a_n\geq 0.
$$

The sign is negative because the constraint is written as

$$
t_ny_n-1\geq 0,
$$

and we are minimizing over the primal variables while maximizing over the nonnegative multipliers.

## 2.3 Stationarity with Respect to the Weight Vector

Differentiate the Lagrangian with respect to $\mathbf{w}$:

$$
\frac{\partial L}{\partial \mathbf{w}}
=
\mathbf{w}
-
\sum_{n=1}^{N}a_nt_n\boldsymbol{\phi}(\mathbf{x}_n).
$$

At the optimum, the derivative is zero:

$$
\mathbf{w}
-
\sum_{n=1}^{N}a_nt_n\boldsymbol{\phi}(\mathbf{x}_n)
=0.
$$

Hence

$$
\boxed{
\mathbf{w}=\sum_{n=1}^{N}a_nt_n\boldsymbol{\phi}(\mathbf{x}_n).
}
$$

This equation is the first sign of the dual representation. The optimal weight vector lies in the span of the mapped training points.

It is not necessary to search over arbitrary directions in feature space. The solution is built from the data themselves.

## 2.4 Stationarity with Respect to the Bias

Differentiate with respect to $b$:

$$
\frac{\partial L}{\partial b}
=-\sum_{n=1}^{N}a_nt_n.
$$

Set this to zero:

$$
\boxed{
\sum_{n=1}^{N}a_nt_n=0.
}
$$

This equality becomes one of the dual constraints.

## 2.5 Deriving the Dual Objective Step by Step

Start from

$$
L
=
\frac{1}{2}\|\mathbf{w}\|^2
-
\sum_n a_nt_n\mathbf{w}^T\boldsymbol{\phi}_n
-
b\sum_n a_nt_n
+
\sum_n a_n,
$$

where $\boldsymbol{\phi}_n=\boldsymbol{\phi}(\mathbf{x}_n)$.

The term involving $b$ vanishes because

$$
\sum_n a_nt_n=0.
$$

Using

$$
\mathbf{w}=\sum_m a_mt_m\boldsymbol{\phi}_m,
$$

we have

$$
\|\mathbf{w}\|^2
=
\left(
\sum_n a_nt_n\boldsymbol{\phi}_n
\right)^T
\left(
\sum_m a_mt_m\boldsymbol{\phi}_m
\right).
$$

Expanding the two sums gives

$$
\|\mathbf{w}\|^2
=
\sum_{n=1}^{N}\sum_{m=1}^{N}
a_na_mt_nt_m
\boldsymbol{\phi}_n^T\boldsymbol{\phi}_m.
$$

Similarly,

$$
\sum_n a_nt_n\mathbf{w}^T\boldsymbol{\phi}_n
=
\sum_n\sum_m a_na_mt_nt_m
\boldsymbol{\phi}_m^T\boldsymbol{\phi}_n.
$$

The two double sums are the same. Therefore

$$
L(\mathbf{a})
=
\sum_n a_n
-
\frac{1}{2}
\sum_n\sum_m
a_na_mt_nt_m
\boldsymbol{\phi}_n^T\boldsymbol{\phi}_m.
$$

Now use the kernel identity

$$
k(\mathbf{x}_n,\mathbf{x}_m)
=
\boldsymbol{\phi}_n^T\boldsymbol{\phi}_m.
$$

The dual problem is

$$
\boxed{
\max_{\mathbf{a}}
\left[
\sum_{n=1}^{N}a_n
-
\frac{1}{2}
\sum_{n=1}^{N}\sum_{m=1}^{N}
a_na_mt_nt_mk(\mathbf{x}_n,\mathbf{x}_m)
\right]
}
$$

subject to

$$
\boxed{
a_n\geq 0,
\qquad
\sum_{n=1}^{N}a_nt_n=0.
}
$$

## 2.6 Why the Dual Form Is So Important

The primal problem depends explicitly on $\boldsymbol{\phi}(\mathbf{x})$. The dual depends only on inner products

$$
\boldsymbol{\phi}(\mathbf{x}_n)^T\boldsymbol{\phi}(\mathbf{x}_m),
$$

which can be replaced by kernels. Therefore:

- the feature space need not be constructed explicitly;
- the feature space may be very high-dimensional;
- the feature space may even be infinite-dimensional;
- the same optimization machinery works with many kernel choices.

The kernel trick is not magic. It works because both training and prediction can be expressed entirely through feature-space inner products.

## 2.7 The Kernelized Decision Function

Substitute

$$
\mathbf{w}=\sum_n a_nt_n\boldsymbol{\phi}(\mathbf{x}_n)
$$

into

$$
y(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b.
$$

Then

$$
y(\mathbf{x})
=
\left(
\sum_n a_nt_n\boldsymbol{\phi}(\mathbf{x}_n)
\right)^T
\boldsymbol{\phi}(\mathbf{x})+b.
$$

Move the transpose inside the sum:

$$
y(\mathbf{x})
=
\sum_n a_nt_n
\boldsymbol{\phi}(\mathbf{x}_n)^T\boldsymbol{\phi}(\mathbf{x})+b.
$$

Therefore

$$
\boxed{
y(\mathbf{x})
=
\sum_{n=1}^{N}a_nt_nk(\mathbf{x},\mathbf{x}_n)+b.
}
$$

## 2.8 KKT Conditions and the Origin of Support Vectors

For each training point, the KKT conditions include

$$
a_n\geq 0,
$$

$$
t_ny(\mathbf{x}_n)-1\geq 0,
$$

and the complementary-slackness condition

$$
\boxed{
a_n\left[t_ny(\mathbf{x}_n)-1\right]=0.}
$$

This last condition is crucial. A product is zero only if at least one factor is zero.

### Case 1: $a_n=0$

The point contributes nothing to the prediction:

$$
a_nt_nk(\mathbf{x},\mathbf{x}_n)=0.
$$

Such a point can often be discarded after training.

### Case 2: $a_n>0$

Then complementary slackness forces

$$
t_ny(\mathbf{x}_n)=1.
$$

The point lies on one of the two margin boundaries. It is a **support vector**.

Therefore the decision function becomes

$$
\boxed{
y(\mathbf{x})
=
\sum_{n\in\mathcal{S}}a_nt_nk(\mathbf{x},\mathbf{x}_n)+b.
}
$$

This is the SVM sparsity mechanism.

> **Key idea.** Sparsity is not imposed by manually deleting data. It emerges from the optimality conditions of the constrained problem.

## 2.9 Solving for the Bias $b$

For a hard-margin support vector $\mathbf{x}_n$,

$$
t_ny(\mathbf{x}_n)=1.
$$

Because $t_n^2=1$, this is equivalent to

$$
y(\mathbf{x}_n)=t_n.
$$

Substitute the kernelized decision function:

$$
\sum_{m\in\mathcal{S}}a_mt_mk(\mathbf{x}_n,\mathbf{x}_m)+b=t_n.
$$

Hence

$$
b
=
t_n
-
\sum_{m\in\mathcal{S}}a_mt_mk(\mathbf{x}_n,\mathbf{x}_m).
$$

In exact arithmetic, any support vector gives the same $b$. Numerically, it is more stable to average over support vectors:

$$
\boxed{
b
=
\frac{1}{N_S}
\sum_{n\in\mathcal{S}}
\left[
 t_n-
 \sum_{m\in\mathcal{S}}a_mt_mk(\mathbf{x}_n,\mathbf{x}_m)
\right].
}
$$

## 2.10 A Nonlinear SVM in Input Space

> ![Figure 7.2](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_2__textbook_fig_7_2__p331.png)
>
> *Figure 7.2 (Textbook Fig. 7.2, p. 331): A Gaussian-kernel SVM produces a nonlinear decision boundary in the original two-dimensional input space. The circled support vectors determine the boundary and margin contours.*

The data in this figure are not linearly separable in the original input plane. However, the Gaussian kernel implicitly maps them to a feature space in which a linear separating hyperplane can be found.

The phrase **linear classifier in feature space** and **nonlinear classifier in input space** describes the same model from two viewpoints.

## 2.11 Geometric Interpretation of Sparsity

Imagine moving a training point that lies far from the margin. As long as it remains correctly classified and outside the margin, the optimal boundary does not change.

By contrast, moving a support vector changes the location of the margin and usually changes the decision boundary.

Thus:

- support vectors carry geometric information about the boundary;
- non-support vectors are redundant for prediction after the model is trained.

---
# §3 Soft-Margin SVM for Overlapping Classes

> 📖 Textbook §7.1.1, pp. 331-336

## 3.1 Why Hard Margins Are Often Too Strict

The hard-margin SVM requires

$$
t_ny(\mathbf{x}_n)\geq 1
$$

for every training point. This assumption fails when:

- class distributions overlap;
- labels contain mistakes;
- measurements are noisy;
- an outlier lies on the wrong side of an otherwise sensible boundary;
- the chosen kernel does not make the data perfectly separable.

Trying to force exact separation may produce an unnecessarily complicated boundary and poor generalization.

The soft-margin SVM allows violations, but charges a penalty for them.

## 3.2 Slack Variables

Introduce one nonnegative slack variable for each training point:

$$
\xi_n\geq 0.
$$

Replace the hard constraint with

$$
\boxed{
t_ny(\mathbf{x}_n)\geq 1-\xi_n.
}
$$

The value of $\xi_n$ tells us how badly point $n$ violates the margin requirement.

> ![Figure 7.3](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_3__textbook_fig_7_3__p332.png)
>
> *Figure 7.3 (Textbook Fig. 7.3, p. 332): Slack variables describe three qualitatively different situations: points outside the margin have $\xi=0$; points inside the margin but correctly classified have $0<\xi\leq 1$; misclassified points have $\xi>1$.*

Let us derive these cases.

### Correct and outside the margin

If

$$
t_ny_n\geq 1,
$$

then the point satisfies the original hard-margin constraint and we may set

$$
\xi_n=0.
$$

### Correct but inside the margin

Suppose

$$
0<t_ny_n<1.
$$

The sign is correct, so the point is classified correctly, but it is too close to the decision boundary. The minimum feasible slack is

$$
\xi_n=1-t_ny_n,
$$

which satisfies

$$
0<\xi_n<1.
$$

### On the decision boundary

If

$$
y_n=0,
$$

then

$$
t_ny_n=0,
$$

so

$$
\xi_n=1.
$$

### Misclassified

A misclassified point has

$$
t_ny_n<0.
$$

Therefore

$$
\xi_n=1-t_ny_n>1.
$$

This gives a useful interpretation:

| Slack value | Meaning |
|---|---|
| $\xi_n=0$ | Correct and on or outside the margin. |
| $0<\xi_n<1$ | Correct but inside the margin. |
| $\xi_n=1$ | Exactly on the decision boundary. |
| $\xi_n>1$ | Misclassified. |

## 3.3 The Soft-Margin Primal Problem

We now balance two goals:

1. Keep $\|\mathbf{w}\|$ small to make the margin large.
2. Keep the total slack small to avoid excessive violations.

The objective is

$$
\boxed{
\min_{\mathbf{w},b,\boldsymbol{\xi}}
\left[
\frac{1}{2}\|\mathbf{w}\|^2
+C\sum_{n=1}^{N}\xi_n
\right]
}
$$

subject to

$$
t_n(\mathbf{w}^T\boldsymbol{\phi}_n+b)\geq 1-\xi_n,
$$

$$
\xi_n\geq 0.
$$

The parameter $C>0$ controls the trade-off.

### Large $C$

A large $C$ makes margin violations expensive. The optimizer tries hard to classify training points correctly, possibly using a narrower or more irregular margin.

### Small $C$

A small $C$ tolerates more violations. The optimizer emphasizes a wide, smooth margin and may deliberately misclassify difficult points.

A useful verbal interpretation is:

> $C$ is the price paid for one unit of slack.

## 3.4 Deriving the Soft-Margin Dual

Introduce multipliers

$$
a_n\geq 0
$$

for the constraints

$$
t_ny_n-1+\xi_n\geq 0,
$$

and multipliers

$$
\mu_n\geq 0
$$

for

$$
\xi_n\geq 0.
$$

The Lagrangian is

$$
L
=
\frac{1}{2}\|\mathbf{w}\|^2
+C\sum_n\xi_n
-
\sum_n a_n[t_n(\mathbf{w}^T\boldsymbol{\phi}_n+b)-1+\xi_n]
-
\sum_n\mu_n\xi_n.
$$

Differentiate with respect to $\mathbf{w}$:

$$
\frac{\partial L}{\partial\mathbf{w}}
=
\mathbf{w}-\sum_n a_nt_n\boldsymbol{\phi}_n=0,
$$

so

$$
\mathbf{w}=\sum_n a_nt_n\boldsymbol{\phi}_n.
$$

Differentiate with respect to $b$:

$$
\frac{\partial L}{\partial b}
=-\sum_n a_nt_n=0,
$$

so

$$
\sum_n a_nt_n=0.
$$

Differentiate with respect to $\xi_n$:

$$
\frac{\partial L}{\partial \xi_n}
=C-a_n-\mu_n=0.
$$

Thus

$$
a_n=C-\mu_n.
$$

Because $\mu_n\geq 0$,

$$
a_n\leq C.
$$

Together with $a_n\geq 0$, we obtain the box constraint

$$
\boxed{0\leq a_n\leq C.}
$$

After eliminating the primal variables, the dual objective has the same algebraic form as in the hard-margin case:

$$
\boxed{
\max_{\mathbf{a}}
\left[
\sum_n a_n
-
\frac{1}{2}\sum_n\sum_m
a_na_mt_nt_mk(\mathbf{x}_n,\mathbf{x}_m)
\right]
}
$$

subject to

$$
\boxed{
0\leq a_n\leq C,
\qquad
\sum_n a_nt_n=0.
}
$$

The only formal change in the dual is the upper bound $a_n\leq C$, but this bound carries the entire effect of the soft margin.

## 3.5 KKT Interpretation of the Different Training Points

For the soft-margin problem, the KKT conditions allow us to classify training points by their dual coefficients.

### Case A: $a_n=0$

The point is not a support vector. It lies outside the margin on the correct side:

$$
t_ny_n>1.
$$

It does not appear in the final predictor.

### Case B: $0<a_n<C$

Since $a_n<C$, the equation $a_n=C-\mu_n$ implies

$$
\mu_n>0.
$$

Complementary slackness for $\mu_n\xi_n=0$ then forces

$$
\xi_n=0.
$$

Because $a_n>0$, the margin constraint is active:

$$
t_ny_n=1.
$$

These points lie exactly on a margin boundary.

### Case C: $a_n=C$

Then $\mu_n=0$, so $\xi_n$ may be positive. Such a point lies inside the margin or is misclassified.

The taxonomy is:

| Dual coefficient | Geometric position | Role |
|---|---|---|
| $a_n=0$ | Correct and outside margin | Not a support vector |
| $0<a_n<C$ | Exactly on margin | Free support vector |
| $a_n=C$, $0<\xi_n\leq1$ | Inside margin but correct | Bound support vector |
| $a_n=C$, $\xi_n>1$ | Misclassified | Bound support vector |

## 3.6 Computing the Bias in the Soft-Margin Case

For any support vector with

$$
0<a_n<C,
$$

we know $\xi_n=0$ and

$$
t_ny_n=1.
$$

Therefore

$$
b
=
t_n-
\sum_{m\in\mathcal{S}}a_mt_mk(\mathbf{x}_n,\mathbf{x}_m).
$$

A stable estimate averages only over the support vectors satisfying $0<a_n<C$:

$$
\boxed{
b
=
\frac{1}{N_M}
\sum_{n\in\mathcal{M}}
\left[
 t_n-
 \sum_{m\in\mathcal{S}}a_mt_mk(\mathbf{x}_n,\mathbf{x}_m)
\right],
}
$$

where

$$
\mathcal{M}=\{n:0<a_n<C\}.
$$

## 3.7 The $\nu$-SVM

An alternative formulation replaces $C$ with a parameter $\nu\in(0,1]$ that has a more direct interpretation.

In the classification $\nu$-SVM, $\nu$ acts as:

- an upper bound on the fraction of margin errors;
- a lower bound on the fraction of support vectors.

This is useful because “fraction of support vectors” and “fraction of violations” are often easier to reason about than the numerical scale of $C$.

> ![Figure 7.4](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_4__textbook_fig_7_4__p335.png)
>
> *Figure 7.4 (Textbook Fig. 7.4, p. 335): A $\nu$-SVM on a nonseparable two-dimensional data set. Circled points are support vectors. Some points fall inside the margin or on the wrong side of the decision boundary.*

The exact dual form differs from the standard $C$-SVM, but the essential lesson is not the algebraic rearrangement. It is the interpretation of $\nu$ as a bound on important fractions of the training set.

## 3.8 Training Algorithms: Why SMO Matters

The dual problem contains $N$ variables and an $N\times N$ kernel matrix. A naive quadratic-programming solver may be impractical for large $N$.

Three historical strategies are:

### Chunking

Solve a smaller quadratic program involving a subset of variables, identify nonzero multipliers, and gradually expand the active set.

### Decomposition methods

Repeatedly optimize a fixed-size working set. This controls memory usage even when the full data set is large.

### Sequential Minimal Optimization (SMO)

Optimize only two multipliers at a time.

Why two? The equality constraint

$$
\sum_n a_nt_n=0
$$

means that changing one multiplier alone would usually violate feasibility. Changing two allows the constraint to be maintained.

With two variables, the subproblem has an analytical solution, so a general numerical QP solver is not needed at every step.

> **Teaching note.** The exact SMO heuristics are implementation details. The conceptual lesson is that sparsity helps prediction, but training still initially depends on the whole data set and requires specialized optimization.

---

# §4 Loss Functions, Probabilities, and Multiclass SVMs

> 📖 Textbook §7.1.2-§7.1.3, pp. 336-339

## 4.1 From Slack Variables to Hinge Loss

For a point with signed score

$$
z_n=t_ny_n,
$$

the smallest feasible slack is

$$
\xi_n=
\begin{cases}
0, & z_n\geq1,\\
1-z_n, & z_n<1.
\end{cases}
$$

This is exactly the positive-part function

$$
\boxed{
\xi_n=[1-z_n]_+=[1-t_ny_n]_+,
}
$$

where

$$
[u]_+=\max(0,u).
$$

Substituting the optimal slack into the primal objective gives the regularized hinge-loss form:

$$
\boxed{
\min_{\mathbf{w},b}
\left[
C\sum_{n=1}^{N}[1-t_ny_n]_+
+
\frac{1}{2}\|\mathbf{w}\|^2
\right].
}
$$

Equivalently, after rescaling,

$$
\sum_n E_{\mathrm{hinge}}(t_ny_n)+\lambda\|\mathbf{w}\|^2.
$$

## 4.2 Comparing Hinge, Logistic, 0-1, and Squared Loss

> ![Figure 7.5](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_5__textbook_fig_7_5__p337.png)
>
> *Figure 7.5 (Textbook Fig. 7.5, p. 337): The hinge loss used by SVMs is compared with logistic loss, the idealized 0-1 misclassification loss, and squared loss. Hinge and logistic losses are convex surrogates for the discontinuous 0-1 loss.*

Let

$$
z=ty.
$$

A positive $z$ means correct classification, and a negative $z$ means misclassification.

### 0-1 loss

$$
E_{0/1}(z)=
\begin{cases}
1, & z\leq0,\\
0, & z>0.
\end{cases}
$$

This directly counts mistakes but is discontinuous and leads to difficult nonconvex optimization.

### Hinge loss

$$
E_{\mathrm{hinge}}(z)=[1-z]_+.
$$

It is convex, piecewise linear, and exactly zero once $z\geq1$.

### Logistic loss

For $t\in\{-1,+1\}$ and

$$
p(t\mid y)=\sigma(ty),
$$

the negative log likelihood is

$$
\boxed{
E_{\mathrm{logistic}}(z)=\ln(1+e^{-z}).
}
$$

This loss is smooth and always positive, although it approaches zero as $z\rightarrow\infty$.

### Squared loss

A squared error such as

$$
E_{\mathrm{square}}(z)=(1-z)^2
$$

continues increasing when $z$ becomes very large. Therefore it can spend excessive effort moving already-correct points even farther from the boundary.

## 4.3 Why Hinge Loss Produces Sparsity

For points satisfying

$$
t_ny_n>1,
$$

the hinge loss is flat:

$$
E_{\mathrm{hinge}}(t_ny_n)=0.
$$

Its derivative is also zero. Such points exert no local pressure on the solution.

Logistic loss, in contrast, becomes small but not exactly flat. Every point can continue to affect the optimum, even when it is confidently classified.

This flat region explains why the SVM solution can depend on only support vectors.

## 4.4 SVM Scores Are Not Automatically Probabilities

The output

$$
y(\mathbf{x})
$$

is a signed decision score. It is not generally equal to a log posterior odds, and

$$
\sigma(y(\mathbf{x}))
$$

is not automatically a calibrated probability.

A common calibration method fits

$$
\boxed{
p(t=1\mid\mathbf{x})
=
\sigma(Ay(\mathbf{x})+B)
}
$$

using a separate calibration set. This is often called **Platt scaling**.

The calibration data should be independent of the data used to fit the SVM, or cross-validated predictions should be used. Otherwise, the fitted probabilities can be severely overconfident.

## 4.5 SVM versus Logistic Regression

| Property | SVM | Logistic regression |
|---|---|---|
| Data loss | Hinge loss | Logistic cross-entropy |
| Output | Decision score | Posterior model $p(t\mid\mathbf{x})$ |
| Sparsity | Usually sparse in the dual | Usually all points affect optimum |
| Optimization | Convex QP | Convex smooth optimization for linear model |
| Probability | Requires calibration | Built into the model |
| Emphasis | Margin and boundary points | Probabilistic fit to all points |

Neither is universally better. The appropriate choice depends on whether probability calibration, sparsity, optimization structure, and robustness are important for the application.

## 4.6 Multiclass SVMs

The standard SVM is fundamentally binary. For $K>2$ classes, common constructions combine binary classifiers.

### One-versus-rest

Train $K$ classifiers. Classifier $k$ separates class $C_k$ from all other classes.

A common prediction rule is

$$
\widehat{k}=\arg\max_k y_k(\mathbf{x}).
$$

Problems include:

- severe class imbalance;
- scores from independently trained classifiers may not have comparable scales;
- multiple classifiers may simultaneously claim the point;
- there is no single unified probabilistic model.

### One-versus-one

Train one classifier for every pair of classes:

$$
\frac{K(K-1)}{2}
$$

classifiers in total.

At test time, each pairwise classifier votes. The class with the most votes wins.

This often works well in practice, but training and storage scale quadratically with $K$, and voting can produce ties or ambiguous outcomes.

### DAGSVM

The same pairwise classifiers can be arranged in a directed acyclic graph. Only $K-1$ classifiers are evaluated for one test example, although the model still stores all pairwise classifiers.

### Single-machine formulations

Unified multiclass margin objectives also exist. They avoid some inconsistencies but lead to larger optimization problems.

> **Practical message.** Multiclass SVM extensions are useful engineering constructions, but they are less conceptually clean than binary SVMs or a directly probabilistic softmax model.

---

# §5 Support Vector Regression

> 📖 Textbook §7.1.4, pp. 339-344

## 5.1 From Classification Margins to a Regression Tube

In ordinary least-squares regression, every residual contributes to the objective:

$$
\frac{1}{2}\sum_{n=1}^{N}(y_n-t_n)^2.
$$

Support vector regression (SVR) instead says:

> Errors smaller than a chosen tolerance $\varepsilon$ are acceptable and receive zero loss.

The $\varepsilon$-insensitive loss is

$$
\boxed{
E_{\varepsilon}(y-t)
=
\begin{cases}
0, & |y-t|\leq\varepsilon,\\
|y-t|-\varepsilon, & |y-t|>\varepsilon.
\end{cases}
}
$$

> ![Figure 7.6](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_6__textbook_fig_7_6__p340.png)
>
> *Figure 7.6 (Textbook Fig. 7.6, p. 340): The $\varepsilon$-insensitive loss is flat inside a tolerance interval and increases linearly outside it. The quadratic loss is shown for comparison.*

The flat region is the regression analogue of the flat part of the hinge loss. It is the source of sparsity.

## 5.2 The $\varepsilon$-Tube

For a prediction function

$$
y(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b,
$$

the acceptable tube is

$$
y(\mathbf{x})-\varepsilon
\leq t
\leq
y(\mathbf{x})+\varepsilon.
$$

Points inside the tube have zero loss.

> ![Figure 7.7](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_7__textbook_fig_7_7__p341.png)
>
> *Figure 7.7 (Textbook Fig. 7.7, p. 341): The regression curve is surrounded by an $\varepsilon$-insensitive tube. Two slack variables represent deviations above and below the tube.*

We need two slack variables because a target can violate the tube in two directions.

- $\xi_n>0$ when the target lies above the tube;
- $\widehat{\xi}_n>0$ when the target lies below the tube.

The constraints are

$$
t_n\leq y_n+\varepsilon+\xi_n,
$$

$$
t_n\geq y_n-\varepsilon-\widehat{\xi}_n,
$$

with

$$
\xi_n\geq0,
\qquad
\widehat{\xi}_n\geq0.
$$

## 5.3 The SVR Primal Objective

The optimization problem is

$$
\boxed{
\min_{\mathbf{w},b,\boldsymbol{\xi},\widehat{\boldsymbol{\xi}}}
\left[
\frac{1}{2}\|\mathbf{w}\|^2
+C\sum_{n=1}^{N}(\xi_n+\widehat{\xi}_n)
\right]
}
$$

subject to the tube constraints above.

Interpretation:

- $\|\mathbf{w}\|^2/2$ encourages a smooth, low-complexity function in feature space;
- $C$ controls the price of deviations outside the tube;
- $\varepsilon$ determines how much error is ignored.

These two hyperparameters play different roles:

| Parameter | Main effect |
|---|---|
| $C$ | How strongly large outside-tube deviations are penalized. |
| $\varepsilon$ | Width of the zero-loss region; strongly affects the number of support vectors. |

A larger $\varepsilon$ generally creates a wider tube and fewer support vectors, but can increase bias.

## 5.4 The SVR Dual

Introduce dual variables $a_n$ and $\widehat{a}_n$. After forming the Lagrangian and eliminating primal variables, the dual is

$$
\boxed{
\max_{\mathbf{a},\widehat{\mathbf{a}}}
\left[
-
\frac{1}{2}
\sum_{n=1}^{N}\sum_{m=1}^{N}
(a_n-\widehat{a}_n)(a_m-\widehat{a}_m)
k(\mathbf{x}_n,\mathbf{x}_m)
ight.
}
$$

$$
\boxed{
\left.
-
\varepsilon\sum_{n=1}^{N}(a_n+\widehat{a}_n)
+
\sum_{n=1}^{N}(a_n-\widehat{a}_n)t_n
\right]
}
$$

subject to

$$
\boxed{
0\leq a_n\leq C,
\qquad
0\leq\widehat{a}_n\leq C,
}
$$

and

$$
\boxed{
\sum_{n=1}^{N}(a_n-\widehat{a}_n)=0.
}
$$

The optimal weight vector is

$$
\mathbf{w}
=
\sum_{n=1}^{N}(a_n-\widehat{a}_n)
\boldsymbol{\phi}(\mathbf{x}_n).
$$

Therefore the regression predictor is

$$
\boxed{
y(\mathbf{x})
=
\sum_{n=1}^{N}(a_n-\widehat{a}_n)
k(\mathbf{x},\mathbf{x}_n)+b.
}
$$

Only points with

$$
a_n-\widehat{a}_n\neq0
$$

contribute to prediction. These are the regression support vectors.

## 5.5 Which Points Become Regression Support Vectors?

### Strictly inside the tube

If

$$
|t_n-y_n|<\varepsilon,
$$

then

$$
a_n=\widehat{a}_n=0.
$$

The point is ignored by the final predictor.

### On the upper tube boundary

If

$$
t_n-y_n=\varepsilon,
$$

then typically

$$
0<a_n<C,
\qquad
\widehat{a}_n=0.
$$

### On the lower tube boundary

If

$$
y_n-t_n=\varepsilon,
$$

then typically

$$
a_n=0,
\qquad
0<\widehat{a}_n<C.
$$

### Outside the tube

A point beyond the tube generally has one multiplier equal to $C$ and a positive slack variable.

This is directly analogous to the classification SVM:

- points comfortably correct are inactive;
- boundary and violating points determine the solution.

## 5.6 Computing the Bias in SVR

For a point with

$$
0<a_n<C,
$$

the target lies on the upper tube boundary:

$$
t_n=y_n+\varepsilon.
$$

Therefore

$$
\boxed{
b
=
t_n-\varepsilon
-
\sum_m(a_m-\widehat{a}_m)k(\mathbf{x}_n,\mathbf{x}_m).
}
$$

For a point with

$$
0<\widehat{a}_n<C,
$$

the target lies on the lower boundary:

$$
t_n=y_n-\varepsilon,
$$

so

$$
\boxed{
b
=
t_n+\varepsilon
-
\sum_m(a_m-\widehat{a}_m)k(\mathbf{x}_n,\mathbf{x}_m).
}
$$

As usual, averaging over eligible support vectors improves numerical stability.

## 5.7 $\nu$-SVR

In ordinary SVR, the user chooses $\varepsilon$ directly. The $\nu$-SVR formulation instead uses $\nu$ to control useful fractions of points.

The textbook result is:

- at most approximately $\nu N$ points lie outside the insensitive tube;
- at least approximately $\nu N$ points become support vectors.

The tube width is then adjusted as part of the optimization rather than fixed in advance.

## 5.8 Textbook Sinusoidal Regression Example

> ![Figure 7.8](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_8__textbook_fig_7_8__p344.png)
>
> *Figure 7.8 (Textbook Fig. 7.8, p. 344): A Gaussian-kernel $\nu$-SVR fitted to noisy sinusoidal data. The red curve is the prediction, the shaded region is the insensitive tube, and circled points are support vectors.*

Notice that points comfortably inside the tube do not need to be retained. The support vectors are concentrated on or beyond the tube boundaries.

This illustrates the practical meaning of sparse regression:

> The model remembers the observations that define the tube, not every observation in the data set.

---
# §6 Computational Learning Theory: A Gentle Introduction

> 📖 Textbook §7.1.5, pp. 344-345  
> This section is introduction only and can be shortened first when teaching time is limited.

## 6.1 The Question Behind PAC Learning

Computational learning theory asks questions such as:

- How many training examples are needed?
- How complex can the hypothesis class be before generalization becomes difficult?
- Under what assumptions can a learning algorithm probably produce an approximately correct predictor?

The classic PAC phrase stands for **probably approximately correct**.

Suppose a learned classifier is denoted

$$
f(\mathbf{x};D),
$$

where $D$ is the random training set. Let its true error be

$$
R(f)=\mathbb{E}_{\mathbf{x},t}
\left[
\mathbb{I}(f(\mathbf{x};D)\neq t)
\right].
$$

PAC learning asks for a training-set size large enough that

$$
R(f)<\varepsilon
$$

with probability at least

$$
1-\delta.
$$

The two parameters have different meanings:

| Parameter | Meaning |
|---|---|
| $\varepsilon$ | Maximum acceptable true error. Smaller is more demanding. |
| $\delta$ | Probability of failure. Smaller means greater confidence. |

Thus “probably approximately correct” means:

- **probably:** with probability at least $1-\delta$;
- **approximately correct:** the true error is below $\varepsilon$.

## 6.2 VC Dimension

The Vapnik-Chervonenkis dimension is a measure of the capacity of a hypothesis class.

Very informally:

> The VC dimension is the largest number of points that the model class can label in every possible binary way.

To **shatter** a set of points means that, for every assignment of $+1$ and $-1$ labels to those points, at least one classifier in the hypothesis class realizes that labeling.

A higher VC dimension means a more flexible hypothesis class.

## 6.3 Why the Bounds Are Often Loose

Classical PAC bounds are distribution-free. They must hold for every allowed data distribution. This makes them conservative.

Real data often have helpful regularities:

- smooth decision regions;
- clustered observations;
- low-dimensional structure;
- natural invariances;
- benign noise patterns.

Worst-case theory cannot exploit these regularities unless they are explicitly added as assumptions.

Therefore PAC and VC theory are valuable for understanding capacity and generalization conceptually, but textbook bounds are often too loose to directly select a training-set size in a practical application.

## 6.4 Connection to Maximum Margin

The maximum-margin idea can be interpreted as controlling effective model complexity. Even in a very rich feature space, a large-margin separator may have better generalization behavior than an arbitrary separating hyperplane.

This helps explain an important point:

> Model complexity is not determined only by the raw number of feature-space dimensions or parameters. The geometry and norm of the chosen solution also matter.

---

# §7 Relevance Vector Machines for Regression

> 📖 Textbook §7.2-§7.2.1, pp. 345-349

## 7.1 Why Introduce the RVM?

Support vector machines are powerful, but the textbook highlights several limitations:

1. The standard SVM output is a decision score, not a posterior probability.
2. Binary SVMs require extra constructions for multiclass problems.
3. Hyperparameters such as $C$, $\nu$, and $\varepsilon$ are usually selected by cross-validation.
4. SVM kernel functions must satisfy positive-definiteness requirements.
5. The number of support vectors can still be large.

The relevance vector machine uses a Bayesian formulation to address many of these issues.

Its prediction function can resemble an SVM:

$$
y(\mathbf{x})
=
\sum_{n=1}^{N}w_nk(\mathbf{x},\mathbf{x}_n)+b.
$$

However, the coefficients are learned through a Bayesian linear model with one precision hyperparameter per weight.

## 7.2 Start from Bayesian Linear Regression

Assume the regression model

$$
t_n=y(\mathbf{x}_n)+\epsilon_n,
$$

with

$$
\epsilon_n\sim\mathcal{N}(0,\beta^{-1}).
$$

The linear basis-function model is

$$
y(\mathbf{x})
=
\sum_{i=1}^{M}w_i\phi_i(\mathbf{x})
=
\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}).
$$

For all targets together,

$$
\mathbf{t}
=
\Phi\mathbf{w}+\boldsymbol{\epsilon},
$$

where

$$
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\beta^{-1}\mathbf{I}).
$$

Therefore the likelihood is

$$
\boxed{
p(\mathbf{t}\mid\mathbf{w},\beta)
=
\mathcal{N}(\mathbf{t}\mid\Phi\mathbf{w},\beta^{-1}\mathbf{I}).
}
$$

## 7.3 Kernel Basis Functions

To mirror the SVM, choose one basis function centered on each training point:

$$
\phi_n(\mathbf{x})=k(\mathbf{x},\mathbf{x}_n).
$$

Including a bias basis, the model becomes

$$
y(\mathbf{x})
=
b+
\sum_{n=1}^{N}w_nk(\mathbf{x},\mathbf{x}_n).
$$

The RVM analysis is actually more general than this kernel-centered choice. The basis functions need not:

- be centered on training points;
- equal the number of training points;
- come from a positive-definite kernel.

This is an important difference from the SVM.

## 7.4 The ARD Prior

A standard Bayesian linear model might use one common precision $\alpha$:

$$
p(\mathbf{w}\mid\alpha)
=
\mathcal{N}(\mathbf{w}\mid\mathbf{0},\alpha^{-1}\mathbf{I}).
$$

The RVM instead gives each weight its own precision:

$$
\boxed{
p(\mathbf{w}\mid\boldsymbol{\alpha})
=
\prod_{i=1}^{M}\mathcal{N}(w_i\mid0,\alpha_i^{-1}).
}
$$

Equivalently,

$$
p(\mathbf{w}\mid\boldsymbol{\alpha})
=
\mathcal{N}(\mathbf{w}\mid\mathbf{0},\mathbf{A}^{-1}),
$$

where

$$
\mathbf{A}=\operatorname{diag}(\alpha_1,\ldots,\alpha_M).
$$

The prior variance of weight $w_i$ is

$$
\operatorname{var}(w_i)=\alpha_i^{-1}.
$$

Therefore:

- small $\alpha_i$: broad prior, weight may be large;
- large $\alpha_i$: narrow prior around zero;
- $\alpha_i\rightarrow\infty$: prior becomes concentrated at $w_i=0$.

This is the automatic relevance determination mechanism.

## 7.5 Deriving the Gaussian Posterior

The posterior is proportional to likelihood times prior:

$$
p(\mathbf{w}\mid\mathbf{t},\boldsymbol{\alpha},\beta)
\propto
p(\mathbf{t}\mid\mathbf{w},\beta)
p(\mathbf{w}\mid\boldsymbol{\alpha}).
$$

Write the negative log posterior, omitting constants independent of $\mathbf{w}$:

$$
E(\mathbf{w})
=
\frac{\beta}{2}
\|\mathbf{t}-\Phi\mathbf{w}\|^2
+
\frac{1}{2}\mathbf{w}^T\mathbf{A}\mathbf{w}.
$$

Expand the squared term:

$$
\|\mathbf{t}-\Phi\mathbf{w}\|^2
=
(\mathbf{t}-\Phi\mathbf{w})^T
(\mathbf{t}-\Phi\mathbf{w}).
$$

Therefore

$$
\|\mathbf{t}-\Phi\mathbf{w}\|^2
=
\mathbf{t}^T\mathbf{t}
-2\mathbf{w}^T\Phi^T\mathbf{t}
+
\mathbf{w}^T\Phi^T\Phi\mathbf{w}.
$$

Substitute this into $E(\mathbf{w})$:

$$
E(\mathbf{w})
=
\frac{1}{2}
\mathbf{w}^T
(\mathbf{A}+\beta\Phi^T\Phi)
\mathbf{w}
-
\beta\mathbf{w}^T\Phi^T\mathbf{t}
+
\text{constant}.
$$

A Gaussian density in $\mathbf{w}$ has exponent

$$
-\frac{1}{2}(\mathbf{w}-\mathbf{m})^T
\boldsymbol{\Sigma}^{-1}
(\mathbf{w}-\mathbf{m}).
$$

Matching the quadratic terms gives

$$
\boxed{
\boldsymbol{\Sigma}^{-1}
=
\mathbf{A}+\beta\Phi^T\Phi.
}
$$

Thus

$$
\boxed{
\boldsymbol{\Sigma}
=
(\mathbf{A}+\beta\Phi^T\Phi)^{-1}.
}
$$

Matching the linear terms gives

$$
\boldsymbol{\Sigma}^{-1}\mathbf{m}
=
\beta\Phi^T\mathbf{t}.
$$

Multiply by $\boldsymbol{\Sigma}$:

$$
\boxed{
\mathbf{m}
=
\beta\boldsymbol{\Sigma}\Phi^T\mathbf{t}.
}
$$

Therefore

$$
\boxed{
p(\mathbf{w}\mid\mathbf{t},\boldsymbol{\alpha},\beta)
=
\mathcal{N}(\mathbf{w}\mid\mathbf{m},\boldsymbol{\Sigma}).
}
$$

## 7.6 The Marginal Likelihood or Evidence

The hyperparameters $\boldsymbol{\alpha}$ and $\beta$ are learned by maximizing the evidence:

$$
p(\mathbf{t}\mid\boldsymbol{\alpha},\beta)
=
\int
p(\mathbf{t}\mid\mathbf{w},\beta)
p(\mathbf{w}\mid\boldsymbol{\alpha})
\,d\mathbf{w}.
$$

Because both distributions are Gaussian, the integral is analytic.

We can understand its covariance without performing the full integral. The target vector is

$$
\mathbf{t}=\Phi\mathbf{w}+\boldsymbol{\epsilon}.
$$

The prior gives

$$
\operatorname{cov}(\mathbf{w})=\mathbf{A}^{-1},
$$

so

$$
\operatorname{cov}(\Phi\mathbf{w})
=
\Phi\mathbf{A}^{-1}\Phi^T.
$$

The noise covariance is

$$
\operatorname{cov}(\boldsymbol{\epsilon})
=
\beta^{-1}\mathbf{I}.
$$

The two terms are independent, so their covariances add:

$$
\boxed{
\mathbf{C}
=
\beta^{-1}\mathbf{I}
+
\Phi\mathbf{A}^{-1}\Phi^T.
}
$$

Hence

$$
\boxed{
p(\mathbf{t}\mid\boldsymbol{\alpha},\beta)
=
\mathcal{N}(\mathbf{t}\mid\mathbf{0},\mathbf{C}).
}
$$

The log evidence is

$$
\boxed{
\ln p(\mathbf{t}\mid\boldsymbol{\alpha},\beta)
=
-\frac{1}{2}
\left[
N\ln(2\pi)
+
\ln|\mathbf{C}|
+
\mathbf{t}^T\mathbf{C}^{-1}\mathbf{t}
\right].
}
$$

This expression contains a natural trade-off:

- $\mathbf{t}^T\mathbf{C}^{-1}\mathbf{t}$ rewards fitting the observed targets;
- $\ln|\mathbf{C}|$ penalizes a covariance structure that spreads probability mass too broadly.

This is a Bayesian form of Occam's razor.

## 7.7 Re-estimation Equations

Define

$$
\gamma_i=1-\alpha_i\Sigma_{ii}.
$$

The evidence-stationary updates are

$$
\boxed{
\alpha_i^{\mathrm{new}}
=
\frac{\gamma_i}{m_i^2}.
}
$$

The noise precision update is

$$
\boxed{
(\beta^{\mathrm{new}})^{-1}
=
\frac{\|\mathbf{t}-\Phi\mathbf{m}\|^2}
{N-\sum_i\gamma_i}.
}
$$

## 7.8 Interpreting $\gamma_i$

The posterior variance for weight $i$ is $\Sigma_{ii}$. The product

$$
\alpha_i\Sigma_{ii}
$$

measures how strongly the posterior uncertainty resembles the prior scale. Therefore

$$
\gamma_i=1-\alpha_i\Sigma_{ii}
$$

can be interpreted as the extent to which the data determine weight $i$.

Typical interpretations are:

- $\gamma_i\approx1$: the data strongly determine the weight;
- $\gamma_i\approx0$: the weight remains effectively suppressed by the prior.

The sum

$$
\sum_i\gamma_i
$$

acts like an effective number of parameters.

## 7.9 The Iterative RVM Training Procedure

A simple evidence-maximization algorithm is:

1. Initialize $\alpha_i$ and $\beta$.
2. Compute
   $$
   \boldsymbol{\Sigma}
   =(\mathbf{A}+\beta\Phi^T\Phi)^{-1}.
   $$
3. Compute
   $$
   \mathbf{m}=\beta\boldsymbol{\Sigma}\Phi^T\mathbf{t}.
   $$
4. Compute
   $$
   \gamma_i=1-\alpha_i\Sigma_{ii}.
   $$
5. Update each $\alpha_i$ using
   $$
   \alpha_i\leftarrow\gamma_i/m_i^2.
   $$
6. Update $\beta$ from the residual error.
7. Remove basis functions whose $\alpha_i$ becomes extremely large.
8. Repeat until the evidence or parameters stabilize.

This is a type-2 maximum-likelihood or empirical-Bayes procedure: weights are integrated out, while hyperparameters are optimized.

## 7.10 Predictive Distribution

For a new input $\mathbf{x}$,

$$
p(t\mid\mathbf{x},\mathbf{w},\beta)
=
\mathcal{N}(t\mid\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}),\beta^{-1}).
$$

Average over the Gaussian posterior of $\mathbf{w}$:

$$
p(t\mid\mathbf{x},D)
=
\int
p(t\mid\mathbf{x},\mathbf{w},\beta)
p(\mathbf{w}\mid D)
\,d\mathbf{w}.
$$

The result is Gaussian:

$$
\boxed{
p(t\mid\mathbf{x},D)
=
\mathcal{N}
\left(
 t\mid
 \mathbf{m}^T\boldsymbol{\phi}(\mathbf{x}),
 \sigma^2(\mathbf{x})
\right).
}
$$

The variance is

$$
\boxed{
\sigma^2(\mathbf{x})
=
\beta^{-1}
+
\boldsymbol{\phi}(\mathbf{x})^T
\boldsymbol{\Sigma}
\boldsymbol{\phi}(\mathbf{x}).
}
$$

The two terms have distinct meanings:

| Term | Meaning |
|---|---|
| $\beta^{-1}$ | Irreducible observation noise. |
| $\boldsymbol{\phi}^T\boldsymbol{\Sigma}\boldsymbol{\phi}$ | Uncertainty caused by uncertain weights. |

## 7.11 RVM Regression Example

> ![Figure 7.9](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_9__textbook_fig_7_9__p349.png)
>
> *Figure 7.9 (Textbook Fig. 7.9, p. 349): RVM regression on the same sinusoidal data used for the SVR example. The red line is the predictive mean, the shaded region is one predictive standard deviation, and only three relevance vectors remain, compared with seven support vectors in Figure 7.8.*

This figure illustrates two major RVM advantages:

1. The output is a predictive distribution rather than only a curve.
2. The number of retained relevance vectors can be much smaller than the number of SVM support vectors.

However, the figure should not be interpreted as a universal theorem that every RVM is always more accurate or always sparser. Sparsity and performance depend on the data, basis functions, initialization, and evidence optimization.

## 7.12 A Caution About Extrapolation

For localized basis functions, all basis values may become small far from the training data:

$$
\boldsymbol{\phi}(\mathbf{x})\approx\mathbf{0}.
$$

Then the weight-uncertainty term

$$
\boldsymbol{\phi}(\mathbf{x})^T
\boldsymbol{\Sigma}
\boldsymbol{\phi}(\mathbf{x})
$$

can also become small. The RVM may therefore become misleadingly confident far away from the data.

Gaussian-process regression handles this situation differently because its predictive uncertainty is tied directly to the prior covariance over function values.

> **Common confusion.** “Bayesian” does not automatically mean that every uncertainty estimate is ideal. The quality of uncertainty still depends on the model and prior structure.

---
# §8 Why the RVM Becomes Sparse

> 📖 Textbook §7.2.2, pp. 349-353

## 8.1 The Simple Two-Target Geometric Picture

The evidence distribution is

$$
p(\mathbf{t}\mid\boldsymbol{\alpha},\beta)
=
\mathcal{N}(\mathbf{t}\mid\mathbf{0},\mathbf{C}).
$$

Consider the simplest case:

- two observed targets, so $\mathbf{t}=(t_1,t_2)^T$;
- one basis vector $\boldsymbol{\varphi}=(\phi(\mathbf{x}_1),\phi(\mathbf{x}_2))^T$;
- one precision $\alpha$;
- isotropic noise precision $\beta$.

Then

$$
\mathbf{C}
=
\beta^{-1}\mathbf{I}
+
\alpha^{-1}\boldsymbol{\varphi}\boldsymbol{\varphi}^T.
$$

The first term gives a circular covariance. The second stretches the Gaussian along the direction $\boldsymbol{\varphi}$.

> ![Figure 7.10](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_10__textbook_fig_7_10__p350.png)
>
> *Figure 7.10 (Textbook Fig. 7.10, p. 350): If a basis vector is poorly aligned with the observed target vector, giving it finite prior variance stretches the evidence distribution in an unhelpful direction and lowers the density at the observed target. Evidence maximization therefore prefers $\alpha\rightarrow\infty$ and removes that basis.*

The intuition is:

- If $\boldsymbol{\varphi}$ points toward the observed target vector, allowing variation along that direction can explain the data.
- If $\boldsymbol{\varphi}$ points in an unrelated direction, allowing variation along it wastes probability mass.
- The evidence can increase by removing the unhelpful direction, which corresponds to

$$
\alpha\rightarrow\infty.
$$

Then

$$
\alpha^{-1}\rightarrow0,
$$

and the basis contribution disappears.

## 8.2 Isolating One Hyperparameter

Write the evidence covariance as

$$
\mathbf{C}
=
\mathbf{C}_{-i}
+
\alpha_i^{-1}\boldsymbol{\varphi}_i\boldsymbol{\varphi}_i^T,
$$

where $\mathbf{C}_{-i}$ is the covariance with basis $i$ removed.

Define

$$
s_i
=
\boldsymbol{\varphi}_i^T
\mathbf{C}_{-i}^{-1}
\boldsymbol{\varphi}_i,
$$

and

$$
q_i
=
\boldsymbol{\varphi}_i^T
\mathbf{C}_{-i}^{-1}
\mathbf{t}.
$$

The textbook calls:

- $s_i$ the **sparsity** factor;
- $q_i$ the **quality** factor.

A useful informal interpretation is:

- $s_i$ describes how much new independent direction the basis offers relative to the current model;
- $q_i$ describes how well that direction aligns with the current target residual.

All dependence of the log evidence on $\alpha_i$ can be isolated as

$$
\lambda(\alpha_i)
=
\frac{1}{2}
\left[
\ln\alpha_i
-
\ln(\alpha_i+s_i)
+
\frac{q_i^2}{\alpha_i+s_i}
\right].
$$

## 8.3 Finite Weight or Pruned Weight?

Differentiate:

$$
\frac{d\lambda}{d\alpha_i}
=
\frac{
\alpha_i^{-1}s_i^2-(q_i^2-s_i)
}
{2(\alpha_i+s_i)^2}.
$$

A finite stationary point requires

$$
\alpha_i^{-1}s_i^2=q_i^2-s_i.
$$

Therefore

$$
\boxed{
\alpha_i
=
\frac{s_i^2}{q_i^2-s_i}
}
$$

only if

$$
q_i^2>s_i.
$$

If

$$
q_i^2\leq s_i,
$$

there is no positive finite maximum, and the evidence is maximized at

$$
\boxed{
\alpha_i=\infty.
}
$$

Thus the pruning rule is:

| Condition | Evidence preference | Result |
|---|---|---|
| $q_i^2>s_i$ | Finite $\alpha_i$ | Keep/add/update basis $i$ |
| $q_i^2\leq s_i$ | $\alpha_i\rightarrow\infty$ | Delete/ignore basis $i$ |

> ![Figure 7.11](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_11__textbook_fig_7_11__p352.png)
>
> *Figure 7.11 (Textbook Fig. 7.11, p. 352): When quality dominates sparsity, the one-dimensional evidence contribution has a finite maximum. When it does not, the maximum occurs at $\alpha_i=\infty$, pruning the basis function.*

## 8.4 Fast Sequential Evidence Maximization

The original simultaneous re-estimation method repeatedly inverts matrices involving all basis functions. A faster strategy changes one basis at a time.

At each iteration:

1. Evaluate the quality and sparsity statistics for candidate bases.
2. If an inactive basis satisfies $q_i^2>s_i$, add it.
3. If an active basis satisfies $q_i^2\leq s_i$, delete it.
4. Otherwise update its finite $\alpha_i$.
5. Update posterior quantities efficiently using matrix identities.

Because the active set is often small, computation scales with the number of active bases rather than the full number of candidates.

## 8.5 SVM Sparsity versus RVM Sparsity

The two methods select points for different reasons.

### SVM

A point is retained because it is geometrically important to the margin or violates the margin.

### RVM

A basis is retained because it improves the marginal likelihood after accounting for fit and uncertainty.

Therefore relevance vectors need not lie near the decision boundary. They are not simply “Bayesian support vectors.”

---

# §9 Relevance Vector Machines for Classification

> 📖 Textbook §7.2.3, pp. 353-356

## 9.1 Probabilistic Binary Classification Model

For binary classification, use targets

$$
t_n\in\{0,1\}.
$$

The linear activation is

$$
a_n=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n).
$$

The posterior class probability is modeled using a logistic sigmoid:

$$
\boxed{
y_n=p(t_n=1\mid\mathbf{x}_n,\mathbf{w})
=\sigma(a_n)
=
\frac{1}{1+e^{-a_n}}.
}
$$

The likelihood is Bernoulli:

$$
\boxed{
p(\mathbf{t}\mid\mathbf{w})
=
\prod_{n=1}^{N}
y_n^{t_n}(1-y_n)^{1-t_n}.
}
$$

The ARD prior remains

$$
p(\mathbf{w}\mid\boldsymbol{\alpha})
=
\mathcal{N}(\mathbf{w}\mid\mathbf{0},\mathbf{A}^{-1}).
$$

## 9.2 Why the Posterior Is No Longer Gaussian

In regression, the likelihood was Gaussian in $\mathbf{w}$, so multiplying by a Gaussian prior produced a Gaussian posterior.

For classification, the sigmoid likelihood is not Gaussian in $\mathbf{w}$. Therefore

$$
p(\mathbf{w}\mid\mathbf{t},\boldsymbol{\alpha})
$$

has no closed-form Gaussian expression.

The textbook uses the Laplace approximation:

1. Find the posterior mode $\mathbf{w}^{\star}$.
2. Approximate the log posterior by a quadratic function around that mode.
3. Use the corresponding Gaussian approximation.

## 9.3 Log Posterior

Ignoring constants independent of $\mathbf{w}$,

$$
\ln p(\mathbf{w}\mid\mathbf{t},\boldsymbol{\alpha})
=
\sum_{n=1}^{N}
\left[
 t_n\ln y_n
+(1-t_n)\ln(1-y_n)
\right]
-
\frac{1}{2}\mathbf{w}^T\mathbf{A}\mathbf{w}.
$$

The first term is the logistic log likelihood. The second is the ARD prior penalty.

## 9.4 Gradient

For logistic regression,

$$
\frac{\partial y_n}{\partial a_n}
=y_n(1-y_n).
$$

The familiar gradient of the log likelihood is

$$
\Phi^T(\mathbf{t}-\mathbf{y}).
$$

The prior contributes

$$
-\mathbf{A}\mathbf{w}.
$$

Therefore

$$
\boxed{
\nabla_{\mathbf{w}}
\ln p(\mathbf{w}\mid\mathbf{t},\boldsymbol{\alpha})
=
\Phi^T(\mathbf{t}-\mathbf{y})
-
\mathbf{A}\mathbf{w}.
}
$$

## 9.5 Hessian and Laplace Covariance

Define a diagonal matrix $\mathbf{B}$ with

$$
B_{nn}=y_n(1-y_n).
$$

The negative Hessian of the log posterior is

$$
\boxed{
-\nabla\nabla
\ln p(\mathbf{w}\mid\mathbf{t},\boldsymbol{\alpha})
=
\Phi^T\mathbf{B}\Phi+\mathbf{A}.
}
$$

At the posterior mode, the Laplace covariance is

$$
\boxed{
\boldsymbol{\Sigma}
=
(\Phi^T\mathbf{B}\Phi+\mathbf{A})^{-1}.
}
$$

The mode can be found by iterative reweighted least squares (IRLS) or Newton updates.

## 9.6 Evidence Re-estimation

Using the Laplace approximation to the marginal likelihood gives the same structural update as in regression:

$$
\gamma_i=1-\alpha_i\Sigma_{ii},
$$

$$
\boxed{
\alpha_i^{\mathrm{new}}
=
\frac{\gamma_i}{(w_i^{\star})^2}.
}
$$

Large $\alpha_i$ values prune weights and basis functions.

Because the approximate evidence surface is nonconvex, different initializations may produce different active sets.

## 9.7 Predictive Probability

For a new input, the desired probability is

$$
p(t=1\mid\mathbf{x},D)
=
\int
\sigma(\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}))
p(\mathbf{w}\mid D)
\,d\mathbf{w}.
$$

Under the Gaussian Laplace posterior, the activation

$$
a=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})
$$

is approximately Gaussian with

$$
\mu_a=\mathbf{w}^{\star T}\boldsymbol{\phi}(\mathbf{x}),
$$

$$
\sigma_a^2
=
\boldsymbol{\phi}(\mathbf{x})^T
\boldsymbol{\Sigma}
\boldsymbol{\phi}(\mathbf{x}).
$$

The sigmoid-Gaussian integral has no elementary closed form, but it can be evaluated numerically or approximated. The key point is that the prediction averages over parameter uncertainty rather than using only one score.

## 9.8 Classification Example

> ![Figure 7.12](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_12__textbook_fig_7_12__p356.png)
>
> *Figure 7.12 (Textbook Fig. 7.12, p. 356): The left panel shows the RVM decision boundary and circled relevance vectors. The right panel shows posterior class probability as a smooth red/blue field. Compared with the SVM in Figure 7.4, the RVM is much sparser and directly provides probabilistic outputs.*

Two visual lessons are important:

1. Relevance vectors are not necessarily concentrated on the boundary.
2. The output is not only a hard curve; it is a probability surface.

## 9.9 Multiclass RVM

For $K$ classes, define $K$ activations

$$
a_k(\mathbf{x})=\mathbf{w}_k^T\boldsymbol{\phi}(\mathbf{x}).
$$

Use the softmax function:

$$
\boxed{
y_k(\mathbf{x})
=
\frac{\exp(a_k)}{\sum_{j=1}^{K}\exp(a_j)}.
}
$$

The multiclass likelihood is

$$
p(\mathbf{T}\mid\mathbf{W})
=
\prod_{n=1}^{N}
\prod_{k=1}^{K}
y_{nk}^{t_{nk}}.
$$

A Laplace approximation can again be used, but the Hessian becomes much larger. If there are $M$ active basis functions and $K$ classes, the weight dimension is approximately $MK$, so naive second-order computations can become expensive.

## 9.10 SVM and RVM: Balanced Comparison

| Property | SVM | RVM |
|---|---|---|
| Theoretical basis | Maximum margin, convex optimization | Bayesian linear model, ARD, evidence |
| Training objective | Convex for standard SVM | Nonconvex evidence optimization |
| Global optimum | Yes for the standard QP | Not guaranteed |
| Output | Decision score | Predictive distribution |
| Sparsity | Support vectors | Relevance vectors, often fewer |
| Kernel restriction | Positive-definite kernel in standard formulation | Basis functions need not form a positive-definite kernel |
| Hyperparameters | Usually cross-validation | Often evidence maximization |
| Training speed | Mature and often faster | Can be slower and initialization-sensitive |
| Test speed | Depends on number of support vectors | Often fast because active set is small |
| Multiclass | Usually pairwise or one-vs-rest construction | Direct probabilistic softmax extension possible |

The right conclusion is not “RVM replaces SVM.” A more accurate conclusion is:

> SVM offers a robust convex margin method; RVM offers a probabilistic and often more compact Bayesian alternative, but at the cost of nonconvex and potentially slower training.

---

# §10 Worked Examples and Textbook Exercises

## 10.1 Worked Example: A One-Dimensional Hard-Margin SVM

Consider two training points:

$$
x_1=-1,\qquad t_1=-1,
$$

$$
x_2=+1,\qquad t_2=+1.
$$

Use the linear model

$$
y(x)=wx+b.
$$

The hard-margin constraints are

$$
t_1(wx_1+b)\geq1,
$$

$$
t_2(wx_2+b)\geq1.
$$

Substitute the first point:

$$
(-1)(-w+b)\geq1.
$$

Therefore

$$
w-b\geq1.
$$

Substitute the second point:

$$
(+1)(w+b)\geq1,
$$

so

$$
w+b\geq1.
$$

At the maximum-margin solution, both closest-point constraints are active:

$$
w-b=1,
$$

$$
w+b=1.
$$

Add the equations:

$$
2w=2,
$$

so

$$
w=1.
$$

Subtract them:

$$
2b=0,
$$

so

$$
b=0.
$$

The classifier is

$$
y(x)=x,
$$

and the boundary is

$$
x=0.
$$

The margin to the closest point is

$$
\frac{1}{|w|}=1.
$$

The full width between $y=-1$ and $y=+1$ is $2$.

This simple example shows that two points, one from each class, can determine the maximum-margin hyperplane.

## 10.2 Textbook Exercise 7.2: Changing the Canonical Constant

The textbook asks us to replace

$$
t_ny_n\geq1
$$

with

$$
t_ny_n\geq\gamma,
\qquad \gamma>0,
$$

and show that the geometric boundary is unchanged.

Suppose $(\mathbf{w},b)$ solves the canonical problem with right side 1. Define

$$
\mathbf{w}'=\gamma\mathbf{w},
$$

$$
b'=\gamma b.
$$

Then

$$
t_n(\mathbf{w}'^T\boldsymbol{\phi}_n+b')
=
\gamma t_n(\mathbf{w}^T\boldsymbol{\phi}_n+b)
\geq\gamma.
$$

The new decision boundary is

$$
\mathbf{w}'^T\boldsymbol{\phi}(\mathbf{x})+b'=0.
$$

Substitute the definitions:

$$
\gamma[\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b]=0.
$$

Since $\gamma>0$,

$$
\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b=0.
$$

Therefore the boundary is identical. The value 1 is a convenient scale choice, not a geometric assumption.

## 10.3 Textbook Exercise 7.3: Why Two Points Can Be Enough

Let the two feature-space points be

$$
\boldsymbol{\phi}_+,
\qquad
\boldsymbol{\phi}_-,
$$

with labels $+1$ and $-1$.

The maximum-margin boundary must be perpendicular to the line connecting them. Therefore its normal vector is parallel to

$$
\boldsymbol{\phi}_+-\boldsymbol{\phi}_-.
$$

The boundary must pass through their midpoint

$$
\frac{\boldsymbol{\phi}_++\boldsymbol{\phi}_-}{2}.
$$

Thus the hyperplane is completely determined:

- orientation from the difference vector;
- location from the midpoint.

This argument does not depend on the ambient feature-space dimension.

## 10.4 Worked Example: Interpreting Slack from the Signed Score

Suppose the true label is $t=+1$.

### Point A: $y=2.3$

$$
ty=2.3>1.
$$

Thus

$$
\xi=[1-2.3]_+=0.
$$

The point is correctly classified outside the margin.

### Point B: $y=0.4$

$$
ty=0.4.
$$

Thus

$$
\xi=1-0.4=0.6.
$$

The point is correct but inside the margin.

### Point C: $y=0$

$$
\xi=1.
$$

The point lies exactly on the decision boundary.

### Point D: $y=-0.7$

$$
ty=-0.7.
$$

Thus

$$
\xi=1-(-0.7)=1.7.
$$

The point is misclassified.

## 10.5 Worked Example: Hinge Loss versus Logistic Loss

Take a signed score

$$
z=ty=2.
$$

The hinge loss is

$$
E_{\mathrm{hinge}}(2)=[1-2]_+=0.
$$

The logistic loss is

$$
E_{\mathrm{logistic}}(2)
=
\ln(1+e^{-2})
\approx0.127.
$$

The SVM completely ignores additional improvement once the margin is satisfied. Logistic regression still receives a small benefit from increasing the score.

Now take

$$
z=-1.
$$

Then

$$
E_{\mathrm{hinge}}(-1)=2,
$$

and

$$
E_{\mathrm{logistic}}(-1)
=
\ln(1+e)
\approx1.313.
$$

Both losses penalize the mistake, but with different numerical shapes.

## 10.6 Worked Example: $\varepsilon$-Insensitive Loss

Let

$$
t=3.0,
\qquad
\varepsilon=0.5.
$$

### Prediction $y=2.7$

The absolute error is

$$
|y-t|=0.3<0.5.
$$

Therefore

$$
E_{\varepsilon}=0.
$$

### Prediction $y=2.0$

The absolute error is

$$
|2.0-3.0|=1.0.
$$

Only the amount beyond the tube is penalized:

$$
E_{\varepsilon}=1.0-0.5=0.5.
$$

### Prediction $y=4.2$

The absolute error is

$$
|4.2-3.0|=1.2,
$$

so

$$
E_{\varepsilon}=1.2-0.5=0.7.
$$

## 10.7 Worked Example: How an ARD Precision Prunes a Weight

Suppose a weight has posterior mean

$$
m_i=0.01
$$

and

$$
\gamma_i=0.8.
$$

The RVM update is

$$
\alpha_i^{\mathrm{new}}
=
\frac{\gamma_i}{m_i^2}.
$$

Substitute the values:

$$
\alpha_i^{\mathrm{new}}
=
\frac{0.8}{0.01^2}
=
\frac{0.8}{0.0001}
=8000.
$$

The corresponding prior variance is

$$
\alpha_i^{-1}=\frac{1}{8000}=0.000125.
$$

This extremely narrow prior pushes $w_i$ strongly toward zero. In later iterations, the basis is likely to be pruned.

By contrast, if

$$
m_i=0.8,
\qquad
\gamma_i=0.8,
$$

then

$$
\alpha_i^{\mathrm{new}}
=
\frac{0.8}{0.64}
=1.25,
$$

which leaves substantial prior variance and allows the basis to remain active.

## 10.8 Suggested Classroom Questions

1. Why can a point be a support vector even if it is correctly classified?
2. Why does increasing $C$ not necessarily improve test accuracy?
3. Why does a large $\varepsilon$ usually reduce the number of SVR support vectors?
4. Why are SVM scores not automatically posterior probabilities?
5. Why does $\alpha_i\rightarrow\infty$ remove an RVM basis function?
6. Why can relevance vectors be far from the classification boundary?
7. Which method has a convex training objective: SVM or RVM?
8. Which method directly represents predictive uncertainty?

---

# §11 Chapter Summary, Figure Checklist, and Teaching Flow

## 11.1 Conceptual Summary

This chapter develops two different answers to the same practical question:

> How can a flexible kernel model make predictions using only a small number of active components?

### SVM answer

Choose a maximum-margin boundary. KKT conditions force most dual coefficients to zero. The remaining support vectors determine prediction.

### RVM answer

Place an independent precision hyperparameter on every weight. Evidence maximization drives many precisions to infinity, collapsing the corresponding weights to zero. The surviving basis centers are relevance vectors.

## 11.2 Essential Equations

### Hard-margin SVM

$$
y(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b
$$

$$
\min_{\mathbf{w},b}\frac{1}{2}\|\mathbf{w}\|^2
\quad
\text{s.t.}
\quad
t_ny(\mathbf{x}_n)\geq1
$$

$$
\max_{\mathbf{a}}
\sum_n a_n
-
\frac{1}{2}\sum_{n,m}a_na_mt_nt_mk(\mathbf{x}_n,\mathbf{x}_m)
$$

$$
y(\mathbf{x})
=
\sum_{n\in\mathcal{S}}a_nt_nk(\mathbf{x},\mathbf{x}_n)+b
$$

### Soft-margin SVM

$$
\min
\frac{1}{2}\|\mathbf{w}\|^2
+C\sum_n\xi_n
$$

$$
t_ny_n\geq1-\xi_n,
\qquad
\xi_n\geq0
$$

$$
0\leq a_n\leq C
$$

### Hinge loss

$$
E_{\mathrm{hinge}}(ty)=[1-ty]_+
$$

### Support vector regression

$$
E_{\varepsilon}(y-t)
=
\max(0,|y-t|-\varepsilon)
$$

$$
y(\mathbf{x})
=
\sum_n(a_n-\widehat{a}_n)k(\mathbf{x},\mathbf{x}_n)+b
$$

### RVM posterior

$$
p(\mathbf{w}\mid\mathbf{t},\boldsymbol{\alpha},\beta)
=
\mathcal{N}(\mathbf{w}\mid\mathbf{m},\boldsymbol{\Sigma})
$$

$$
\boldsymbol{\Sigma}
=
(\mathbf{A}+\beta\Phi^T\Phi)^{-1}
$$

$$
\mathbf{m}
=
\beta\boldsymbol{\Sigma}\Phi^T\mathbf{t}
$$

### RVM evidence updates

$$
\gamma_i=1-\alpha_i\Sigma_{ii}
$$

$$
\alpha_i^{\mathrm{new}}
=
\frac{\gamma_i}{m_i^2}
$$

$$
(\beta^{\mathrm{new}})^{-1}
=
\frac{\|\mathbf{t}-\Phi\mathbf{m}\|^2}
{N-\sum_i\gamma_i}
$$

### RVM predictive distribution

$$
p(t\mid\mathbf{x},D)
=
\mathcal{N}
\left(
 t\mid\mathbf{m}^T\boldsymbol{\phi}(\mathbf{x}),
 \beta^{-1}+\boldsymbol{\phi}(\mathbf{x})^T
 \boldsymbol{\Sigma}
 \boldsymbol{\phi}(\mathbf{x})
\right)
$$

## 11.3 Common Student Confusions

| Confusion | Clarification |
|---|---|
| “The SVM maximizes distance in input space.” | It maximizes margin in feature space. The corresponding input-space boundary may be nonlinear. |
| “Every correctly classified point has zero hinge loss.” | Only points satisfying $ty\geq1$ have zero hinge loss. Correct points inside the margin still incur loss. |
| “Support vectors are misclassified points.” | Some are misclassified, but many lie exactly on the correct margin boundary. |
| “A larger $C$ always improves the model.” | Large $C$ reduces training violations but may narrow the margin and overfit. |
| “The SVM output is a probability.” | It is a decision score unless separately calibrated. |
| “SVR ignores all errors.” | It ignores only errors within $\varepsilon$; outside-tube errors are penalized. |
| “RVM sparsity is caused by KKT conditions.” | RVM sparsity comes from ARD hyperparameters and evidence maximization. |
| “A relevance vector must lie near the boundary.” | No. It is retained because its basis improves evidence, not necessarily because it is geometrically close to the boundary. |
| “Bayesian means convex.” | RVM evidence optimization is nonconvex and may have local optima. |
| “Fewer vectors always means better generalization.” | Sparsity improves storage and prediction speed, but accuracy still depends on the model and optimization. |

## 11.4 Figure Checklist

Filename convention:

```text
lecture_fig_<lecture-number>_<figure-index>__textbook_fig_<original-number>__p<textbook-page>.png
```

| Lecture Figure | Textbook Figure | File |
|---|---|---|
| Figure 7.1 | PRML Fig. 7.1 | `lecture_fig_7_1__textbook_fig_7_1__p327.png` |
| Figure 7.2 | PRML Fig. 7.2 | `lecture_fig_7_2__textbook_fig_7_2__p331.png` |
| Figure 7.3 | PRML Fig. 7.3 | `lecture_fig_7_3__textbook_fig_7_3__p332.png` |
| Figure 7.4 | PRML Fig. 7.4 | `lecture_fig_7_4__textbook_fig_7_4__p335.png` |
| Figure 7.5 | PRML Fig. 7.5 | `lecture_fig_7_5__textbook_fig_7_5__p337.png` |
| Figure 7.6 | PRML Fig. 7.6 | `lecture_fig_7_6__textbook_fig_7_6__p340.png` |
| Figure 7.7 | PRML Fig. 7.7 | `lecture_fig_7_7__textbook_fig_7_7__p341.png` |
| Figure 7.8 | PRML Fig. 7.8 | `lecture_fig_7_8__textbook_fig_7_8__p344.png` |
| Figure 7.9 | PRML Fig. 7.9 | `lecture_fig_7_9__textbook_fig_7_9__p349.png` |
| Figure 7.10 | PRML Fig. 7.10 | `lecture_fig_7_10__textbook_fig_7_10__p350.png` |
| Figure 7.11 | PRML Fig. 7.11 | `lecture_fig_7_11__textbook_fig_7_11__p352.png` |
| Figure 7.12 | PRML Fig. 7.12 | `lecture_fig_7_12__textbook_fig_7_12__p356.png` |

All figures used in this lecture were rendered and cropped from the supplied textbook PDF and saved under:

```text
./CoursePR2026/Fig/Chapter_7/
```

## 11.5 Suggested Teaching Flow

A practical teaching sequence is:

1. Revisit linear discriminants and distance to a hyperplane.
2. Use Figure 7.1 to motivate maximum margin without equations first.
3. Derive the canonical constraints and hard-margin primal problem.
4. Derive the dual slowly, emphasizing where the kernel appears.
5. Use KKT complementary slackness to explain sparsity.
6. Introduce soft margins visually with Figure 7.3 before presenting the objective.
7. Compare hinge and logistic losses using Figure 7.5.
8. Present multiclass strategies briefly rather than spending too much time on engineering variants.
9. Introduce SVR through the $\varepsilon$-tube and complete one numerical loss example.
10. Introduce the RVM as Bayesian linear regression with one precision per weight.
11. Derive the Gaussian posterior and predictive distribution.
12. Use Figures 7.10-7.11 to explain pruning intuitively and mathematically.
13. End with the SVM-RVM comparison table rather than presenting either model as universally superior.

## 11.6 Suggested Board Equations

For a limited lecture, the equations most worth deriving live are:

1. Distance to a hyperplane:
   $$
   \operatorname{dist}=\frac{|y(\mathbf{x})|}{\|\mathbf{w}\|}.
   $$
2. Canonical constraint:
   $$
   t_ny_n\geq1.
   $$
3. Hard-margin primal:
   $$
   \min\frac{1}{2}\|\mathbf{w}\|^2.
   $$
4. Stationarity result:
   $$
   \mathbf{w}=\sum_na_nt_n\boldsymbol{\phi}_n.
   $$
5. Kernelized predictor:
   $$
   y(\mathbf{x})=\sum_{n\in\mathcal{S}}a_nt_nk(\mathbf{x},\mathbf{x}_n)+b.
   $$
6. Soft-margin objective:
   $$
   \frac{1}{2}\|\mathbf{w}\|^2+C\sum_n\xi_n.
   $$
7. Hinge loss:
   $$
   [1-ty]_+.
   $$
8. $\varepsilon$-insensitive loss:
   $$
   \max(0,|y-t|-\varepsilon).
   $$
9. RVM posterior covariance and mean:
   $$
   \boldsymbol{\Sigma}=(\mathbf{A}+\beta\Phi^T\Phi)^{-1},
   \qquad
   \mathbf{m}=\beta\boldsymbol{\Sigma}\Phi^T\mathbf{t}.
   $$
10. ARD update:
   $$
   \alpha_i^{\mathrm{new}}=\gamma_i/m_i^2.
   $$

## 11.7 Bridge to Chapter 8

Chapter 7 focuses on sparse kernel predictors. Chapter 8 moves from individual predictive functions to structured probabilistic models represented by graphs.

The conceptual transition is:

> In Chapter 7, kernels control similarity and sparsity controls which training points matter. In Chapter 8, graph structure controls which variables interact and which conditional independences hold.

The next chapter will introduce:

- Bayesian networks;
- directed acyclic graphs;
- conditional independence and d-separation;
- Markov random fields;
- factor graphs;
- sum-product and max-sum message passing.
