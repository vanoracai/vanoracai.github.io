# Pattern Recognition and Machine Learning
## Lectures 6-8: Kernels, Support Vector Machines, and Graphical Models

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapters covered selectively: Ch. 6 Kernel Methods, Ch. 7 Sparse Kernel Machines, and Ch. 8 Graphical Models

---

## Teaching Scope

This combined lecture extracts only the ideas that are most useful for modern machine learning and for the later chapters of this course.

| Textbook Chapter | Main ideas retained | Topics intentionally omitted |
|---|---|---|
| **Ch. 6 Kernel Methods** | Feature maps, kernel functions, the kernel trick, common kernels, Gram matrices | Gaussian processes, GP classification, hyperparameter evidence, ARD |
| **Ch. 7 Sparse Kernel Machines** | Margin, support vectors, soft margin, hinge loss, kernelized decision boundaries | Lagrangian dual, KKT conditions, quadratic programming, SMO details, RVM |
| **Ch. 8 Graphical Models** | Observed and latent variables, directed graphs, factorization, conditional dependence, plate notation | Formal d-separation, Markov random fields, junction trees, belief propagation |

> **Teaching focus.** The purpose is not to turn students into specialists in classical kernel machines or exact graphical-model inference. The purpose is to give them a compact conceptual toolkit that remains useful in modern ML:
>
> 1. a nonlinear model can sometimes be implemented through pairwise similarities;
> 2. a classifier can be trained by protecting a margin rather than only fitting labels;
> 3. a probabilistic model can be understood by identifying variables and factorizing a joint distribution.

---

## Table of Contents

1. [§0 Roadmap: Three Ways to Add Structure](#0-roadmap-three-ways-to-add-structure)
2. [Part I — Chapter 6: Kernel Methods](#part-i--chapter-6-kernel-methods)
3. [§1 From Linear Models to Feature Maps](#1-from-linear-models-to-feature-maps)
4. [§2 Kernel Functions and the Kernel Trick](#2-kernel-functions-and-the-kernel-trick)
5. [§3 Common Kernels and Practical Interpretation](#3-common-kernels-and-practical-interpretation)
6. [§4 Kernel Worked Examples](#4-kernel-worked-examples)
7. [Part II — Chapter 7: Support Vector Machines](#part-ii--chapter-7-support-vector-machines)
8. [§5 Margin and Support Vectors](#5-margin-and-support-vectors)
9. [§6 Soft Margin and Hinge Loss](#6-soft-margin-and-hinge-loss)
10. [§7 Kernel SVM Intuition](#7-kernel-svm-intuition)
11. [§8 SVM Worked Examples](#8-svm-worked-examples)
12. [Part III — Chapter 8: Graphical Models](#part-iii--chapter-8-graphical-models)
13. [§9 Variables, Graphs, and Factorization](#9-variables-graphs-and-factorization)
14. [§10 Observed Variables, Latent Variables, and Plates](#10-observed-variables-latent-variables-and-plates)
15. [§11 Conditional Dependence and Conditional Independence](#11-conditional-dependence-and-conditional-independence)
16. [§12 Latent-Variable Models Needed Later](#12-latent-variable-models-needed-later)
17. [§13 Graphical-Model Worked Examples](#13-graphical-model-worked-examples)
18. [§14 Combined Summary and Bridge to Chapter 9](#14-combined-summary-and-bridge-to-chapter-9)

---

## Notation and Variable Definitions

### Kernel and SVM Notation

| Symbol | Definition |
|---|---|
| $\mathbf{x},\mathbf{x}'$ | Two input vectors. |
| $\boldsymbol{\phi}(\mathbf{x})$ | A feature map that transforms an input into a new feature representation. |
| $k(\mathbf{x},\mathbf{x}')$ | A kernel function measuring similarity through an inner product in feature space. |
| $K$ | Gram matrix, with $K_{nm}=k(\mathbf{x}_n,\mathbf{x}_m)$. |
| $\mathbf{w}$ | Weight vector of a linear classifier in feature space. |
| $b$ | Bias or intercept. |
| $f(\mathbf{x})$ | Classification score, usually $f(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b$. |
| $t_n\in\{-1,+1\}$ | Binary class label for training example $n$. |
| $z_n=t_nf(\mathbf{x}_n)$ | Signed classification score. Positive means correct side; larger means more confident margin. |
| $\xi_n$ | Slack variable measuring a margin violation. |
| $C$ | Soft-margin penalty parameter used in one common SVM formulation. |
| $\lambda$ | Regularization coefficient used in an equivalent regularized-loss formulation. |

### Graphical-Model Notation

| Symbol | Definition |
|---|---|
| $x$ | An observed variable in a simple latent-variable model. |
| $z$ | A latent or hidden variable. |
| $\operatorname{pa}(x_k)$ | Parent variables of node $x_k$ in a directed graph. |
| $p(x_1,\ldots,x_K)$ | Joint distribution of all variables. |
| $p(x_k\mid \operatorname{pa}(x_k))$ | Local conditional distribution associated with node $x_k$. |
| $a\perp b\mid c$ | $a$ and $b$ are conditionally independent given $c$. |
| Plate | A box indicating repeated variables or repeated factors. |
| Shaded node | An observed random variable. |
| Unshaded node | An unobserved random variable, often latent. |

---

# §0 Roadmap: Three Ways to Add Structure

The previous lectures developed regression, classification, and neural-network models. This lecture adds three complementary forms of structure.

| Idea | Main question | Typical mathematical object |
|---|---|---|
| **Kernel** | How can a simple linear algorithm behave nonlinearly? | Similarity $k(\mathbf{x},\mathbf{x}')$ |
| **Margin** | How can a classifier prefer a robust separating boundary? | Signed score $t f(\mathbf{x})$ |
| **Graphical model** | How can a complicated joint distribution be decomposed? | Product of local conditional distributions |

A compact conceptual pipeline is

$$
\mathbf{x}
\xrightarrow{\text{feature similarity}}
k(\mathbf{x},\mathbf{x}')
\xrightarrow{\text{margin learning}}
f(\mathbf{x})
\xrightarrow{\text{probabilistic structure}}
p(\text{observed},\text{latent}).
$$

These topics are historically associated with different communities, but the underlying idea is shared:

> **A good representation exposes the structure of the learning problem.**

---

# Part I — Chapter 6: Kernel Methods

> 📖 Textbook Ch. 6 opening and §6.2, especially Eq. (6.1) and Figure 6.1

# §1 From Linear Models to Feature Maps

## 1.1 A Linear Model Can Use Nonlinear Features

A linear model in the original input space has the form

$$
y(\mathbf{x})=\mathbf{w}^T\mathbf{x}+b.
$$

Its decision boundary is linear in $\mathbf{x}$. If the data cannot be separated by a line, plane, or hyperplane, one standard idea is to transform the input first:

$$
\mathbf{x}\longmapsto \boldsymbol{\phi}(\mathbf{x}).
$$

The model then becomes

$$
y(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b.
$$

The model is still linear in the transformed features $\boldsymbol{\phi}(\mathbf{x})$, but it can be nonlinear as a function of the original input $\mathbf{x}$.

### Simple one-dimensional example

Suppose

$$
\phi(x)=
\begin{bmatrix}
1\\
x\\
x^2
\end{bmatrix}.
$$

Then

$$
y(x)
=
\begin{bmatrix}
w_0&w_1&w_2
\end{bmatrix}
\begin{bmatrix}
1\\x\\x^2
\end{bmatrix}
=w_0+w_1x+w_2x^2.
$$

So a quadratic curve in the original input is just a linear function of the transformed feature vector.

## 1.2 Why Explicit Feature Maps Can Become Inconvenient

Consider an input vector with $D$ components. If we include all pairwise products, the feature vector contains terms such as

$$
x_1^2,\;x_1x_2,\;x_1x_3,\ldots,x_D^2.
$$

The number of terms grows rapidly as the degree and dimensionality increase.

For example:

| Input dimension | Polynomial degree | Approximate number of monomial features |
|---:|---:|---:|
| $D=2$ | $2$ | a few |
| $D=100$ | $2$ | thousands |
| $D=1000$ | $2$ | hundreds of thousands |
| large $D$ | high degree | potentially enormous |

The kernel idea is useful when an algorithm needs feature vectors only through their inner products.

---

# §2 Kernel Functions and the Kernel Trick

## 2.1 Definition of a Kernel

A kernel function is defined by

$$
\boxed{
k(\mathbf{x},\mathbf{x}')
=
\boldsymbol{\phi}(\mathbf{x})^T
\boldsymbol{\phi}(\mathbf{x}')
}
$$

The right-hand side is an ordinary inner product, but it is taken in the transformed feature space.

This formula gives three useful interpretations.

| Interpretation | Meaning |
|---|---|
| **Similarity** | Inputs are similar when their transformed feature vectors point in similar directions and have a large inner product. |
| **Implicit representation** | We can compute a feature-space inner product without necessarily forming every feature explicitly. |
| **Algorithm substitution** | If an algorithm uses inputs only through $\mathbf{x}^T\mathbf{x}'$, replace this inner product by $k(\mathbf{x},\mathbf{x}')$. |

> ![Figure 6.1](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_1__textbook_fig_6_1__p295.png)
>
> *Lecture Figure 6.1 (Textbook Fig. 6.1, p. 295): Different basis-function families lead to different kernel shapes. The upper row shows basis functions; the lower row shows the resulting similarity with a fixed reference input. The important message is that a kernel summarizes the combined effect of many features.*

## 2.2 The Kernel Trick

Suppose an algorithm uses only inner products such as

$$
\boldsymbol{\phi}(\mathbf{x}_n)^T\boldsymbol{\phi}(\mathbf{x}_m).
$$

If we know a kernel satisfying

$$
k(\mathbf{x}_n,\mathbf{x}_m)
=
\boldsymbol{\phi}(\mathbf{x}_n)^T\boldsymbol{\phi}(\mathbf{x}_m),
$$

then every such inner product can be replaced by a kernel evaluation.

This is the **kernel trick**:

$$
\text{explicit feature-space inner product}
\quad\longrightarrow\quad
\text{kernel evaluation}.
$$

The advantage is not that the mathematics disappears. The advantage is that the possibly large feature vector does not have to be constructed explicitly.

## 2.3 A Step-by-Step Polynomial-Kernel Example

Take two-dimensional inputs

$$
\mathbf{x}=
\begin{bmatrix}x_1\\x_2\end{bmatrix},
\qquad
\mathbf{z}=
\begin{bmatrix}z_1\\z_2\end{bmatrix}.
$$

Consider the kernel

$$
k(\mathbf{x},\mathbf{z})=(\mathbf{x}^T\mathbf{z})^2.
$$

First expand the ordinary inner product:

$$
\mathbf{x}^T\mathbf{z}=x_1z_1+x_2z_2.
$$

Now square it:

$$
(\mathbf{x}^T\mathbf{z})^2
=(x_1z_1+x_2z_2)^2.
$$

Using $(a+b)^2=a^2+2ab+b^2$,

$$
(\mathbf{x}^T\mathbf{z})^2
=x_1^2z_1^2+2x_1x_2z_1z_2+x_2^2z_2^2.
$$

Rewrite the middle term using square roots:

$$
2x_1x_2z_1z_2
=(\sqrt{2}x_1x_2)(\sqrt{2}z_1z_2).
$$

Therefore

$$
k(\mathbf{x},\mathbf{z})
=
\begin{bmatrix}
x_1^2\\
\sqrt{2}x_1x_2\\
x_2^2
\end{bmatrix}^{T}
\begin{bmatrix}
z_1^2\\
\sqrt{2}z_1z_2\\
z_2^2
\end{bmatrix}.
$$

So the implicit feature map is

$$
\boxed{
\boldsymbol{\phi}(\mathbf{x})=
\begin{bmatrix}
x_1^2\\
\sqrt{2}x_1x_2\\
x_2^2
\end{bmatrix}
}
$$

and

$$
k(\mathbf{x},\mathbf{z})
=\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{z}).
$$

### Why this matters

Computing $(\mathbf{x}^T\mathbf{z})^2$ directly is easy. Constructing and storing every second-order feature may be expensive when the original dimension is large.

## 2.4 The Gram Matrix

For training inputs $\mathbf{x}_1,\ldots,\mathbf{x}_N$, define the Gram matrix

$$
\mathbf{K}=
\begin{bmatrix}
k(\mathbf{x}_1,\mathbf{x}_1) & \cdots & k(\mathbf{x}_1,\mathbf{x}_N)\\
\vdots & \ddots & \vdots\\
k(\mathbf{x}_N,\mathbf{x}_1) & \cdots & k(\mathbf{x}_N,\mathbf{x}_N)
\end{bmatrix}.
$$

Its entries are

$$
K_{nm}=k(\mathbf{x}_n,\mathbf{x}_m).
$$

The Gram matrix can be interpreted as a pairwise similarity table.

- The diagonal entry $K_{nn}$ measures the feature-space squared length of example $n$.
- The off-diagonal entry $K_{nm}$ measures feature-space similarity between examples $n$ and $m$.
- For a valid kernel, the Gram matrix is symmetric and positive semidefinite.

For this course, the positive-semidefinite condition should be understood operationally:

> A valid kernel behaves like a genuine inner product in some feature space.

We will not prove Mercer's theorem.

---

# §3 Common Kernels and Practical Interpretation

## 3.1 Linear Kernel

$$
k(\mathbf{x},\mathbf{x}')=\mathbf{x}^T\mathbf{x}'.
$$

This is simply the ordinary inner product. It does not add nonlinear features.

Use it when:

- the input representation is already strong;
- the dimension is high and the data may be approximately linearly separable;
- computational simplicity matters.

## 3.2 Polynomial Kernel

A common form is

$$
k(\mathbf{x},\mathbf{x}')=(\mathbf{x}^T\mathbf{x}'+c)^M,
$$

where $M$ is the degree and $c\geq 0$ controls the contribution of lower-order terms.

Interpretation:

- $M=1$ gives a linear model;
- $M=2$ introduces pairwise feature interactions;
- larger $M$ introduces higher-order interactions.

A high degree can make the model very sensitive and difficult to tune, so moderate degrees are usually easier to teach and use.

## 3.3 Gaussian or RBF Kernel

$$
\boxed{
k(\mathbf{x},\mathbf{x}')
=
\exp\left(-\frac{\|\mathbf{x}-\mathbf{x}'\|^2}{2\sigma^2}\right)
}
$$

This kernel is large when two inputs are close and small when they are far apart.

### Effect of $\sigma$

| $\sigma$ | Similarity behavior | Model behavior |
|---|---|---|
| Small | Similarity drops very quickly | Very local, flexible, greater overfitting risk |
| Moderate | Nearby points influence one another | Smooth nonlinear boundary |
| Large | Many points look similar | Very smooth, may underfit |

An equivalent parameterization often uses

$$
\gamma=\frac{1}{2\sigma^2},
$$

so that

$$
k(\mathbf{x},\mathbf{x}')=\exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2).
$$

Large $\gamma$ corresponds to small $\sigma$.

## 3.4 Feature Scaling Is Essential

The RBF kernel uses Euclidean distance. Suppose one feature is measured in meters and another in millimeters. The larger numerical scale can dominate the distance even when it is not more important.

A standard preprocessing step is

$$
x_d^{\mathrm{scaled}}
=\frac{x_d-\mu_d}{s_d},
$$

where $\mu_d$ and $s_d$ are estimated using the training set.

The same transformation must be applied to validation and test data.

## 3.5 What Kernels Do and Do Not Do

A kernel does:

- define a notion of similarity;
- induce a feature space;
- allow some algorithms to use nonlinear features implicitly.

A kernel does not:

- automatically solve overfitting;
- eliminate hyperparameter selection;
- guarantee that a chosen similarity is appropriate;
- always scale well to very large $N$.

The full Gram matrix contains $N^2$ entries, so memory can become a bottleneck.

## 3.6 Connection to Modern ML

Kernel methods remain conceptually useful even when deep networks are used in practice.

- Attention mechanisms compute data-dependent similarities.
- Contrastive learning tries to make semantically related representations similar.
- Neural tangent kernels analyze some wide-network training regimes.
- Kernel approximations such as random Fourier features replace a large implicit feature space by a manageable explicit one.

The modern lesson is broader than a particular algorithm:

> **The choice of similarity strongly influences what structure a model can learn.**

---

# §4 Kernel Worked Examples

## Worked Example 4.1: Polynomial Kernel

Let

$$
\mathbf{x}=\begin{bmatrix}1\\2\end{bmatrix},
\qquad
\mathbf{z}=\begin{bmatrix}2\\1\end{bmatrix}.
$$

Use

$$
k(\mathbf{x},\mathbf{z})=(\mathbf{x}^T\mathbf{z})^2.
$$

### Step 1: Compute the ordinary inner product

$$
\mathbf{x}^T\mathbf{z}=1\cdot2+2\cdot1=4.
$$

### Step 2: Square it

$$
k(\mathbf{x},\mathbf{z})=4^2=16.
$$

### Step 3: Verify using the explicit feature map

$$
\boldsymbol{\phi}(\mathbf{x})=
\begin{bmatrix}
1^2\\\sqrt{2}(1)(2)\\2^2
\end{bmatrix}
=
\begin{bmatrix}
1\\2\sqrt{2}\\4
\end{bmatrix},
$$

and

$$
\boldsymbol{\phi}(\mathbf{z})=
\begin{bmatrix}
4\\2\sqrt{2}\\1
\end{bmatrix}.
$$

Therefore

$$
\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{z})
=1\cdot4+(2\sqrt{2})(2\sqrt{2})+4\cdot1
=4+8+4=16.
$$

Both routes give the same answer.

## Worked Example 4.2: RBF Similarity

Let

$$
\mathbf{x}=\begin{bmatrix}0\\0\end{bmatrix},
\qquad
\mathbf{x}'=\begin{bmatrix}1\\0\end{bmatrix},
\qquad
\sigma=1.
$$

First compute the squared distance:

$$
\|\mathbf{x}-\mathbf{x}'\|^2
=(0-1)^2+(0-0)^2=1.
$$

Then

$$
k(\mathbf{x},\mathbf{x}')
=\exp\left(-\frac{1}{2}\right)
\approx 0.607.
$$

If the distance doubles to $2$, the squared distance becomes $4$:

$$
k=\exp(-2)\approx 0.135.
$$

The similarity drops nonlinearly with distance.

## Quick Check

1. Why can a model be nonlinear in $\mathbf{x}$ but linear in $\mathbf{w}$?
2. What does $k(\mathbf{x},\mathbf{x}')$ compute conceptually?
3. For an RBF kernel, what happens when $\sigma$ becomes very small?
4. Why should features be scaled before distance-based kernels are used?

---

# Part II — Chapter 7: Support Vector Machines

> 📖 Textbook §7.1 and §7.1.1-§7.1.2, with emphasis on Figures 7.1, 7.2, 7.3, and 7.5

# §5 Margin and Support Vectors

## 5.1 Binary Classification Score

For binary classification, let

$$
t_n\in\{-1,+1\}.
$$

Define the score

$$
f(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b.
$$

The predicted class is

$$
\hat{t}=\operatorname{sign}(f(\mathbf{x})).
$$

The decision boundary is where the score is zero:

$$
f(\mathbf{x})=0.
$$

## 5.2 Correctness Through the Signed Score

Multiply the score by the true label:

$$
z_n=t_nf(\mathbf{x}_n).
$$

This single number summarizes both correctness and confidence.

| $z_n=t_nf(\mathbf{x}_n)$ | Interpretation |
|---:|---|
| $z_n<0$ | Misclassified |
| $z_n=0$ | Exactly on the decision boundary |
| $0<z_n<1$ | Correct side but inside the desired margin |
| $z_n=1$ | On a margin boundary |
| $z_n>1$ | Correct and outside the margin |

## 5.3 Margin Intuition

Many separating lines may classify a training set correctly. SVM prefers the one with the largest buffer between the two classes.

> ![Figure 7.1](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_1__textbook_fig_7_1__p327.png)
>
> *Lecture Figure 7.1 (Textbook Fig. 7.1, p. 327): The margin is the perpendicular distance from the decision boundary to the nearest training points. The circled points are support vectors. Moving distant points slightly does not change the boundary, but moving support vectors can change it.*

The central intuition is

> A large-margin boundary is less sensitive to small perturbations of the closest training examples.

## 5.4 Distance to a Hyperplane

For a hyperplane

$$
f(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b=0,
$$

the perpendicular distance from a point $\mathbf{x}$ to the hyperplane is

$$
\frac{|f(\mathbf{x})|}{\|\mathbf{w}\|}.
$$

For a correctly classified point with label $t$, we can write the signed distance as

$$
\frac{t f(\mathbf{x})}{\|\mathbf{w}\|}.
$$

### Canonical scaling

The same decision boundary is obtained if $\mathbf{w}$ and $b$ are multiplied by a positive constant. SVM removes this arbitrary scale by choosing the nearest points to satisfy

$$
t_nf(\mathbf{x}_n)=1.
$$

The two margin boundaries are then

$$
f(\mathbf{x})=+1
$$

and

$$
f(\mathbf{x})=-1.
$$

The distance from the central boundary to either margin boundary is

$$
\frac{1}{\|\mathbf{w}\|}.
$$

Therefore the total width between the two margin boundaries is

$$
\boxed{
\frac{2}{\|\mathbf{w}\|}
}
$$

Maximizing the margin is therefore equivalent to making $\|\mathbf{w}\|$ small while maintaining appropriate classification constraints.

We stop at this geometric statement. The dual optimization and KKT derivation are intentionally omitted.

## 5.5 Support Vectors

Support vectors are the training points that touch or violate the margin.

In the ideal separable case, they satisfy

$$
t_nf(\mathbf{x}_n)=1.
$$

In the soft-margin case, points with

$$
t_nf(\mathbf{x}_n)\leq 1
$$

are the important points for the hinge-loss objective.

Why are support vectors special?

- They are the closest points to the decision boundary.
- They determine the location of the maximum-margin boundary.
- Distant points have zero hinge loss and usually do not directly affect the final boundary.
- The decision rule is therefore sparse in terms of training examples.

---

# §6 Soft Margin and Hinge Loss

## 6.1 Why Hard Separation Is Often a Bad Goal

Real class distributions overlap. Labels can also be noisy. Forcing every training point to be correctly separated can produce an unnecessarily complicated boundary.

A soft-margin SVM allows some points to enter the margin or even be misclassified.

> ![Figure 7.3](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_3__textbook_fig_7_3__p332.png)
>
> *Lecture Figure 7.3 (Textbook Fig. 7.3, p. 332): Slack variables describe margin violations. Points with $\xi=0$ satisfy the margin, points with $0<\xi<1$ are correctly classified but inside the margin, and points with $\xi>1$ are misclassified.*

The soft-margin constraint is

$$
t_nf(\mathbf{x}_n)\geq 1-\xi_n,
\qquad
\xi_n\geq 0.
$$

The slack variable can be read directly from the signed score:

$$
\xi_n=\max(0,1-t_nf(\mathbf{x}_n)).
$$

This is exactly the hinge loss.

## 6.2 Hinge Loss

Define

$$
\boxed{
\ell_{\mathrm{hinge}}(t,f(\mathbf{x}))
=
\max(0,1-tf(\mathbf{x}))
}
$$

or, using $z=tf(\mathbf{x})$,

$$
\ell_{\mathrm{hinge}}(z)=\max(0,1-z).
$$

### Case-by-case interpretation

#### Case 1: $z\geq 1$

Then

$$
1-z\leq 0,
$$

so

$$
\ell_{\mathrm{hinge}}=0.
$$

The point is correctly classified and outside the margin.

#### Case 2: $0<z<1$

Then

$$
0<1-z<1,
$$

so

$$
\ell_{\mathrm{hinge}}=1-z.
$$

The point is correctly classified but too close to the boundary.

#### Case 3: $z=0$

Then

$$
\ell_{\mathrm{hinge}}=1.
$$

The point lies on the decision boundary.

#### Case 4: $z<0$

Then

$$
1-z>1,
$$

so the loss is larger than one. The point is misclassified, and more severe misclassification produces larger loss.

> ![Figure 7.4](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_4__textbook_fig_7_5__p337.png)
>
> *Lecture Figure 7.4 (Textbook Fig. 7.5, p. 337): The blue curve is hinge loss. It is zero once the signed score reaches one. Logistic loss is smooth and remains positive, while 0-1 loss records only whether the sign is correct.*

## 6.3 Regularized Hinge-Loss Objective

A convenient modern form of the SVM objective is

$$
\boxed{
J(\mathbf{w},b)
=
\frac{\lambda}{2}\|\mathbf{w}\|^2
+
\frac{1}{N}\sum_{n=1}^{N}
\max\left(0,1-t_nf(\mathbf{x}_n)\right)
}
$$

This objective contains two competing goals.

| Term | Purpose |
|---|---|
| $\frac{\lambda}{2}\|\mathbf{w}\|^2$ | Prefers a wide, smooth margin by keeping the weight norm small. |
| Average hinge loss | Penalizes points that lie inside the margin or on the wrong side. |

### Role of $\lambda$

- Large $\lambda$: stronger regularization, wider/smoother boundary, greater underfitting risk.
- Small $\lambda$: weaker regularization, more effort spent fitting training points, greater overfitting risk.

A common classical formulation instead uses

$$
\frac{1}{2}\|\mathbf{w}\|^2+C\sum_n \xi_n.
$$

The parameter $C$ has the opposite qualitative direction from $\lambda$:

- large $C$: margin violations are expensive;
- small $C$: violations are tolerated more easily.

## 6.4 Hinge Loss versus Logistic Loss

For $z=tf(\mathbf{x})$:

### Hinge loss

$$
\ell_{\mathrm{hinge}}(z)=\max(0,1-z).
$$

### Logistic loss

$$
\ell_{\mathrm{logistic}}(z)=\log(1+e^{-z}).
$$

| Property | Hinge loss | Logistic loss |
|---|---|---|
| Zero for sufficiently correct points | Yes, when $z\geq1$ | No |
| Smooth everywhere | No, kink at $z=1$ | Yes |
| Direct probabilistic interpretation | Not by itself | Yes, through logistic regression |
| Focus on boundary points | Strong | Softer |
| Output calibration | Usually requires extra calibration | Often more naturally probabilistic |

The two losses often produce similar boundaries, but their outputs should not be interpreted in the same way.

---

# §7 Kernel SVM Intuition

## 7.1 Linear in Feature Space, Nonlinear in Input Space

An SVM score can be written as

$$
f(\mathbf{x})
=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b.
$$

If the training procedure can be expressed through feature-space inner products, then

$$
\boldsymbol{\phi}(\mathbf{x}_n)^T\boldsymbol{\phi}(\mathbf{x})
$$

can be replaced by

$$
k(\mathbf{x}_n,\mathbf{x}).
$$

The resulting classifier has the conceptual form

$$
f(\mathbf{x})
=
\sum_{n\in\mathcal{S}}\alpha_nt_nk(\mathbf{x}_n,\mathbf{x})+b,
$$

where $\mathcal{S}$ is the set of support vectors.

We do not derive the coefficients $\alpha_n$ in this lecture. The important interpretation is:

> A prediction is formed by comparing the test input with a selected subset of influential training examples.

## 7.2 Nonlinear Decision Boundary

> ![Figure 7.2](./CoursePR2026/Fig/Chapter_7/lecture_fig_7_2__textbook_fig_7_2__p331.png)
>
> *Lecture Figure 7.2 (Textbook Fig. 7.2, p. 331): An RBF-kernel SVM creates a nonlinear decision boundary in the original two-dimensional input space. Circled training examples are support vectors, and the adjacent contours show the margin structure.*

A nonlinear boundary does not mean that the SVM abandoned linear classification. It means that the classifier is linear in an implicit feature space.

## 7.3 Effect of Kernel Width and Regularization

For an RBF SVM, two hyperparameters interact:

| Setting | Typical effect |
|---|---|
| Small $\sigma$, large $C$ | Highly local, flexible boundary; can fit noise |
| Small $\sigma$, small $C$ | Local features but more tolerance for violations |
| Large $\sigma$, large $C$ | Smooth kernel but strong pressure to fit labels |
| Large $\sigma$, small $C$ | Very smooth and tolerant; may underfit |

Use a validation set or cross-validation to select them.

## 7.4 When SVMs Are Still Attractive

SVMs can be strong when:

- the data set is small or medium-sized;
- feature engineering is meaningful;
- the input dimension is high;
- a robust classifier is needed without designing a deep architecture;
- the boundary is nonlinear but a suitable kernel is known.

Limitations include:

- training and storage can scale poorly with $N$;
- kernel and hyperparameter selection can be sensitive;
- standard SVM scores are not calibrated probabilities;
- end-to-end representation learning is less flexible than in deep networks.

---

# §8 SVM Worked Examples

## Worked Example 8.1: Hinge Loss for Four Points

Suppose the signed scores are

$$
z_n=t_nf(\mathbf{x}_n)
\in\{1.8,\;0.6,\;0,\;-0.5\}.
$$

Compute

$$
\ell_n=\max(0,1-z_n).
$$

| $z_n$ | Calculation | Hinge loss | Interpretation |
|---:|---:|---:|---|
| $1.8$ | $\max(0,1-1.8)$ | $0$ | Correct and outside margin |
| $0.6$ | $\max(0,1-0.6)$ | $0.4$ | Correct but inside margin |
| $0$ | $\max(0,1-0)$ | $1$ | On decision boundary |
| $-0.5$ | $\max(0,1+0.5)$ | $1.5$ | Misclassified |

The average hinge loss is

$$
\frac{0+0.4+1+1.5}{4}
=\frac{2.9}{4}
=0.725.
$$

## Worked Example 8.2: Margin Width

Suppose

$$
\|\mathbf{w}\|=4.
$$

The distance from the decision boundary to either margin boundary is

$$
\frac{1}{\|\mathbf{w}\|}=\frac14=0.25.
$$

The full margin width is

$$
\frac{2}{\|\mathbf{w}\|}=\frac24=0.5.
$$

If regularization leads to a smaller norm, say $\|\mathbf{w}\|=2$, the full margin becomes

$$
\frac{2}{2}=1.
$$

So reducing the norm doubles the geometric margin in this simple comparison.

## Worked Example 8.3: Identify Support-Vector Candidates

Suppose five training points have signed scores

$$
\{2.1,\;1.3,\;1.0,\;0.7,\;-0.2\}.
$$

Points with score greater than $1$ have zero hinge loss. Points with score less than or equal to $1$ are on or inside the margin.

Therefore the candidates that influence the hinge-loss term are

$$
1.0,\;0.7,\;-0.2.
$$

These correspond respectively to:

- a point on the margin;
- a correctly classified point inside the margin;
- a misclassified point.

## Quick Check

1. What quantity determines whether a point is correctly classified and whether it satisfies the margin?
2. Why do distant correctly classified points have zero hinge loss?
3. How does a support vector differ from an arbitrary training point?
4. Why can an RBF SVM produce a curved boundary in the original input space?

---

# Part III — Chapter 8: Graphical Models

> 📖 Textbook §8.1 and the introductory part of §8.2

# §9 Variables, Graphs, and Factorization

## 9.1 Why Use a Graphical Model?

A probabilistic model may contain many random variables. Writing one large joint distribution gives little visual guidance about how the variables interact.

A graphical model provides two linked descriptions:

1. a graph showing local dependence relationships;
2. a factorization of the joint probability distribution.

For this lecture, we use directed graphical models, also called Bayesian networks.

## 9.2 Nodes and Directed Edges

- A node represents a random variable.
- An arrow points from a parent variable to a child variable.
- A directed graph must not contain a directed cycle.

The graph does not replace probability equations. It organizes them.

## 9.3 Factorization Rule

For variables $x_1,\ldots,x_K$ in a directed acyclic graph,

$$
\boxed{
p(x_1,\ldots,x_K)
=
\prod_{k=1}^{K}
p(x_k\mid \operatorname{pa}(x_k))
}
$$

where $\operatorname{pa}(x_k)$ denotes the parents of node $x_k$.

### How to read this formula

For every node:

1. identify its parents;
2. write the conditional distribution of the node given its parents;
3. multiply all node-wise factors.

## 9.4 Three-Node Example

> ![Figure 8.1](./CoursePR2026/Fig/Chapter_8/lecture_fig_8_1__textbook_fig_8_1__p361.png)
>
> *Lecture Figure 8.1 (Textbook Fig. 8.1, p. 361): A directed graph over $a$, $b$, and $c$. Node $a$ has no parents, $b$ has parent $a$, and $c$ has parents $a$ and $b$.*

Read the graph one node at a time.

- For $a$, there are no parents:

$$
p(a).
$$

- For $b$, the parent is $a$:

$$
p(b\mid a).
$$

- For $c$, the parents are $a$ and $b$:

$$
p(c\mid a,b).
$$

Multiply the factors:

$$
\boxed{
p(a,b,c)=p(a)p(b\mid a)p(c\mid a,b)}.
$$

This is much easier to interpret than an unstructured table over all combinations of $a$, $b$, and $c$.

## 9.5 Factorization Does Not Automatically Mean Causation

An arrow is often convenient for describing a generative direction, but a graph learned from observational data should not automatically be interpreted as a causal graph.

For this course, read arrows primarily as instructions for factorizing a probability distribution.

---

# §10 Observed Variables, Latent Variables, and Plates

## 10.1 Observed Variable

An observed variable is a quantity whose value is available in the data.

Examples:

- an image $\mathbf{x}$;
- a class label $t$;
- an acoustic measurement;
- a sequence token $x_t$.

In the textbook's graphical notation, observed random variables are shaded.

## 10.2 Latent Variable

A latent variable is part of the model but is not directly observed.

Examples:

- cluster identity in a Gaussian mixture model;
- hidden state in an HMM;
- latent code in a variational autoencoder;
- topic assignment in a topic model.

A latent variable can explain structure in the observed data.

The simplest latent-variable factorization is

$$
\boxed{
p(x,z)=p(z)p(x\mid z)}.
$$

Read it as a two-stage generative story:

1. draw the latent variable $z$ from $p(z)$;
2. draw the observation $x$ from $p(x\mid z)$.

## 10.3 Plate Notation

When a model contains repeated observations, drawing every variable separately is inconvenient.

> ![Figure 8.2](./CoursePR2026/Fig/Chapter_8/lecture_fig_8_2__textbook_fig_8_4__p363.png)
>
> *Lecture Figure 8.2 (Textbook Fig. 8.4, p. 363): A plate compactly represents $N$ repeated target variables $t_n$ that share the same parameter $\mathbf{w}$. The box labelled $N$ means “repeat the enclosed structure for $n=1,\ldots,N$”.*

A plate indicates replication. If $t_n$ is repeated $N$ times and conditionally independent given $\mathbf{w}$, the joint distribution is

$$
p(\mathbf{t},\mathbf{w})
=p(\mathbf{w})\prod_{n=1}^{N}p(t_n\mid\mathbf{w}).
$$

## 10.4 Observed versus Latent in a Plate

> ![Figure 8.3](./CoursePR2026/Fig/Chapter_8/lecture_fig_8_3__textbook_fig_8_6__p364.png)
>
> *Lecture Figure 8.3 (Textbook Fig. 8.6, p. 364): The repeated targets $t_n$ are shaded because they are observed. The parameter vector $\mathbf{w}$ is unshaded because it is not directly observed. Small solid nodes denote fixed deterministic quantities or hyperparameters.*

A useful reading guide is:

| Visual element | Meaning |
|---|---|
| Open circle | Unobserved random variable |
| Shaded circle | Observed random variable |
| Small solid dot | Fixed parameter or deterministic quantity |
| Arrow | Conditional dependence used in factorization |
| Plate | Repeated variables/factors |

## 10.5 I.I.D. Data as a Graphical Model

Suppose $x_1,\ldots,x_N$ are conditionally independent given parameter $\theta$.

Then

$$
p(x_1,\ldots,x_N\mid\theta)
=\prod_{n=1}^{N}p(x_n\mid\theta).
$$

With a prior $p(\theta)$,

$$
p(\theta,x_1,\ldots,x_N)
=p(\theta)\prod_{n=1}^{N}p(x_n\mid\theta).
$$

This is the graphical-model form of many Bayesian learning problems.

---

# §11 Conditional Dependence and Conditional Independence

## 11.1 Ordinary Independence

Variables $a$ and $b$ are independent if

$$
p(a,b)=p(a)p(b).
$$

Equivalently,

$$
p(a\mid b)=p(a).
$$

Knowing $b$ does not change the distribution of $a$.

## 11.2 Conditional Independence

Variables $a$ and $b$ are conditionally independent given $c$ if

$$
\boxed{
p(a,b\mid c)=p(a\mid c)p(b\mid c)}.
$$

We write

$$
a\perp b\mid c.
$$

This does not necessarily mean that $a$ and $b$ are independent without knowing $c$.

## 11.3 Common-Cause Example

> ![Figure 8.4](./CoursePR2026/Fig/Chapter_8/lecture_fig_8_4__textbook_fig_8_15__p373.png)
>
> *Lecture Figure 8.4 (Textbook Fig. 8.15, p. 373): A common parent $c$ influences both $a$ and $b$. Without observing $c$, the two child variables can be statistically dependent.*

The graph factorizes as

$$
p(a,b,c)=p(c)p(a\mid c)p(b\mid c).
$$

Without conditioning on $c$,

$$
p(a,b)=\sum_c p(c)p(a\mid c)p(b\mid c),
$$

which generally does not equal $p(a)p(b)$.

### Intuitive example

Let

- $c$ = weather;
- $a$ = whether a person carries an umbrella;
- $b$ = whether the road is wet.

If weather is unknown, seeing an umbrella gives information about whether the road may be wet. Thus $a$ and $b$ are dependent.

## 11.4 Conditioning on the Common Cause

> ![Figure 8.5](./CoursePR2026/Fig/Chapter_8/lecture_fig_8_5__textbook_fig_8_16__p374.png)
>
> *Lecture Figure 8.5 (Textbook Fig. 8.16, p. 374): Once the common parent $c$ is observed, the children $a$ and $b$ become conditionally independent in this model.*

Starting from

$$
p(a,b,c)=p(c)p(a\mid c)p(b\mid c),
$$

divide by $p(c)$:

$$
p(a,b\mid c)
=\frac{p(a,b,c)}{p(c)}.
$$

Substitute the factorization:

$$
p(a,b\mid c)
=\frac{p(c)p(a\mid c)p(b\mid c)}{p(c)}.
$$

Cancel $p(c)$:

$$
\boxed{
p(a,b\mid c)=p(a\mid c)p(b\mid c)}.
$$

Therefore

$$
a\perp b\mid c.
$$

In the weather example, if the weather is already known, seeing an umbrella may provide no additional information about the road beyond the weather variable in this simplified model.

## 11.5 Chain Example

For a chain

$$
a\rightarrow c\rightarrow b,
$$

the factorization is

$$
p(a,b,c)=p(a)p(c\mid a)p(b\mid c).
$$

Given $c$, the variable $b$ no longer needs $a$:

$$
p(b\mid a,c)=p(b\mid c).
$$

Hence

$$
a\perp b\mid c.
$$

A practical interpretation is that $c$ summarizes the information passed from $a$ to $b$.

## 11.6 Why Conditional Independence Matters

Conditional independence gives:

- fewer probability factors to model;
- fewer parameters;
- simpler inference;
- a clearer interpretation of the assumptions;
- modular algorithms.

This is the main reason graphical models are useful. We do not need the formal d-separation procedure for the later course material.

---

# §12 Latent-Variable Models Needed Later

## 12.1 One Latent Variable and One Observation

The core model is

$$
p(x,z)=p(z)p(x\mid z).
$$

The observed-data distribution is obtained by marginalizing $z$:

### Discrete latent variable

$$
p(x)=\sum_z p(z)p(x\mid z).
$$

### Continuous latent variable

$$
p(x)=\int p(z)p(x\mid z)\,dz.
$$

The posterior distribution of the latent variable is

$$
p(z\mid x)
=\frac{p(z)p(x\mid z)}{p(x)}.
$$

This posterior answers:

> After observing $x$, which latent explanation $z$ is plausible?

## 12.2 Bridge to Gaussian Mixture Models

In a Gaussian mixture model:

- $z$ is a discrete component identity;
- $x$ is the observed vector;
- $p(z=k)=\pi_k$;
- $p(x\mid z=k)=\mathcal{N}(x\mid\mu_k,\Sigma_k)$.

Therefore

$$
p(x)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x\mid\mu_k,\Sigma_k).
$$

The posterior

$$
p(z=k\mid x)
$$

becomes the responsibility used by the EM algorithm.

## 12.3 Bridge to Variational Inference

Variational inference introduces an approximation

$$
q(z)\approx p(z\mid x).
$$

The graph tells us:

- which variables are latent;
- which posterior is difficult to compute;
- how a factorized approximation may be organized.

## 12.4 Bridge to Hidden Markov Models

For a sequence of hidden states $z_1,\ldots,z_T$ and observations $x_1,\ldots,x_T$,

$$
p(\mathbf{x},\mathbf{z})
=
 p(z_1)
 \prod_{t=2}^{T}p(z_t\mid z_{t-1})
 \prod_{t=1}^{T}p(x_t\mid z_t).
$$

Read it as:

1. choose the initial hidden state;
2. transition from one hidden state to the next;
3. generate each observation from the current hidden state.

This factorization is the foundation of HMM inference.

## 12.5 Bridge to Modern Latent Generative Models

The same simple idea appears in variational autoencoders:

$$
p_\theta(x,z)=p(z)p_\theta(x\mid z).
$$

The decoder models $p_\theta(x\mid z)$, while an encoder approximates the difficult posterior $p_\theta(z\mid x)$ using $q_\phi(z\mid x)$.

Thus the graphical-model notation is not merely historical. It provides the probabilistic skeleton behind many modern generative models.

---

# §13 Graphical-Model Worked Examples

## Worked Example 13.1: Factorize a Graph

Suppose the graph is

$$
a\rightarrow b,
\qquad
a\rightarrow c,
\qquad
b\rightarrow c.
$$

Identify parents:

| Node | Parents | Local factor |
|---|---|---|
| $a$ | none | $p(a)$ |
| $b$ | $a$ | $p(b\mid a)$ |
| $c$ | $a,b$ | $p(c\mid a,b)$ |

Multiply:

$$
p(a,b,c)=p(a)p(b\mid a)p(c\mid a,b).
$$

## Worked Example 13.2: Marginalize a Binary Latent Variable

Let $z\in\{0,1\}$ with

$$
p(z=1)=0.3,
\qquad
p(z=0)=0.7.
$$

Suppose $x\in\{0,1\}$ and

$$
p(x=1\mid z=1)=0.9,
$$

$$
p(x=1\mid z=0)=0.2.
$$

Compute $p(x=1)$.

Using total probability,

$$
p(x=1)
=
\sum_z p(z)p(x=1\mid z).
$$

Therefore

$$
p(x=1)
=0.3\cdot0.9+0.7\cdot0.2.
$$

Compute each term:

$$
0.3\cdot0.9=0.27,
$$

$$
0.7\cdot0.2=0.14.
$$

So

$$
\boxed{p(x=1)=0.41}.
$$

## Worked Example 13.3: Infer the Latent Variable

Continue the previous example. After observing $x=1$, compute

$$
p(z=1\mid x=1).
$$

Using Bayes' rule,

$$
p(z=1\mid x=1)
=
\frac{p(z=1)p(x=1\mid z=1)}{p(x=1)}.
$$

Substitute the numbers:

$$
p(z=1\mid x=1)
=
\frac{0.3\cdot0.9}{0.41}
=
\frac{0.27}{0.41}
\approx0.659.
$$

Although the prior probability of $z=1$ was only $0.3$, the observation $x=1$ makes $z=1$ much more plausible because $x=1$ is strongly associated with that latent state.

This is the basic inference pattern used in mixture models:

$$
\text{prior over component}
\times
\text{component likelihood}
\longrightarrow
\text{posterior responsibility}.
$$

## Worked Example 13.4: Read a Plate

Suppose a plate contains $x_n$ for $n=1,\ldots,N$, with an arrow from $\theta$ to every $x_n$.

The factorization is

$$
p(\theta,x_1,\ldots,x_N)
=p(\theta)\prod_{n=1}^{N}p(x_n\mid\theta).
$$

The product appears because the observations are conditionally independent given $\theta$.

Without conditioning on $\theta$, they need not be independent:

$$
p(x_1,\ldots,x_N)
=
\int p(\theta)\prod_{n=1}^{N}p(x_n\mid\theta)\,d\theta.
$$

A shared latent parameter can therefore create dependence among observations after it is marginalized out.

## Quick Check

1. What is the difference between an observed variable and a latent variable?
2. How do you convert a directed graph into a product of probability factors?
3. What does a plate labelled $N$ mean?
4. In the graph $c\rightarrow a$ and $c\rightarrow b$, why can $a$ and $b$ become independent after conditioning on $c$?
5. How is $p(x,z)=p(z)p(x\mid z)$ related to a Gaussian mixture model?

---

# §14 Combined Summary and Bridge to Chapter 9

## 14.1 Kernel Summary

The central equation is

$$
\boxed{k(\mathbf{x},\mathbf{x}')
=\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}')}.
$$

Remember:

- a kernel is a feature-space inner product;
- the feature space can be high-dimensional or even infinite-dimensional;
- the kernel trick avoids explicit feature construction when the algorithm needs only inner products;
- the choice of kernel defines the notion of similarity.

## 14.2 SVM Summary

The central quantities are

$$
f(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})+b
$$

and

$$
z=tf(\mathbf{x}).
$$

The hinge loss is

$$
\boxed{\ell(z)=\max(0,1-z)}.
$$

Remember:

- SVM protects a margin around the decision boundary;
- support vectors are the points on or inside the margin;
- only boundary-relevant points have positive hinge loss;
- kernels allow nonlinear boundaries in the original input space.

## 14.3 Graphical-Model Summary

The central factorization is

$$
\boxed{
p(x_1,\ldots,x_K)
=\prod_{k=1}^{K}p(x_k\mid\operatorname{pa}(x_k))}.
$$

The simplest latent-variable model is

$$
\boxed{p(x,z)=p(z)p(x\mid z)}.
$$

Remember:

- shaded nodes are observed;
- unshaded nodes are unobserved or latent;
- a plate denotes repeated variables/factors;
- conditional independence can simplify a joint distribution;
- latent variables will be inferred from observed variables.

## 14.4 What We Deliberately Did Not Learn

Students are not expected to derive or implement the following from this lecture:

- Gaussian-process regression or classification;
- SVM dual optimization;
- KKT conditions;
- quadratic-programming solvers;
- RVM evidence maximization;
- formal d-separation;
- Markov random fields;
- junction-tree inference;
- belief propagation.

These topics are valuable in specialized courses, but they are not necessary for the learning objectives here.

## 14.5 Bridge to Chapter 9: Mixture Models and EM

Chapter 9 begins with clustering and Gaussian mixtures. The graphical-model preparation from this lecture is exactly what we need.

For each observation $\mathbf{x}_n$, introduce a latent component label $\mathbf{z}_n$:

$$
p(\mathbf{x}_n,\mathbf{z}_n)
=p(\mathbf{z}_n)p(\mathbf{x}_n\mid\mathbf{z}_n).
$$

For the full data set,

$$
p(\mathbf{X},\mathbf{Z})
=
\prod_{n=1}^{N}
p(\mathbf{z}_n)p(\mathbf{x}_n\mid\mathbf{z}_n).
$$

The next lecture will answer two questions:

1. **Inference:** Given $\mathbf{x}_n$, what is the posterior probability of each hidden component $\mathbf{z}_n$?
2. **Learning:** Given uncertain component assignments, how do we estimate the component means, covariances, and mixing weights?

These two questions become the E-step and M-step of EM.

---

## Final One-Page Concept Map

| Topic | Input idea | Main equation | Main output |
|---|---|---|---|
| Kernel | Compare two examples in feature space | $k(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^T\phi(\mathbf{x}')$ | Implicit nonlinear representation |
| SVM | Protect a classification buffer | $\max(0,1-tf(\mathbf{x}))$ | Maximum-margin classifier |
| Graphical model | Decompose a joint distribution | $p(\mathbf{x})=\prod_k p(x_k\mid\operatorname{pa}(x_k))$ | Structured probabilistic model |
| Latent model | Explain observations through hidden variables | $p(x,z)=p(z)p(x\mid z)$ | Foundation for GMM, EM, VI, HMM |

> **Final message.** Kernels structure similarity, SVMs structure the decision boundary, and graphical models structure probability. These are three different answers to the same broad question: how should a learning problem be represented so that useful computation becomes possible?
