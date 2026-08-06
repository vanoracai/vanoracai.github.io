---
layout: course
title: PRML Lecture 12
---

# Pattern Recognition and Machine Learning
## Chapter 12: Continuous Latent Variables — PCA and Autoencoders

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 12 Continuous Latent Variables (§12.1-§12.4)  
> Teaching emphasis: **Principal Component Analysis (PCA)**, its probabilistic interpretation, and its connection to **linear and nonlinear autoencoders**

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Why Continuous Latent Variables?](#1-why-continuous-latent-variables)
3. [§2 Data Centering and the Covariance Matrix](#2-data-centering-and-the-covariance-matrix)
4. [§3 PCA as Maximum-Variance Projection](#3-pca-as-maximum-variance-projection)
5. [§4 Projection, Latent Coordinates, and Reconstruction](#4-projection-latent-coordinates-and-reconstruction)
6. [§5 PCA as Minimum Reconstruction Error](#5-pca-as-minimum-reconstruction-error)
7. [§6 Explained Variance and Choosing the Number of Components](#6-explained-variance-and-choosing-the-number-of-components)
8. [§7 Applications: Visualization, Compression, Whitening, and Denoising](#7-applications-visualization-compression-whitening-and-denoising)
9. [§8 Practical PCA Algorithms and High-Dimensional Data](#8-practical-pca-algorithms-and-high-dimensional-data)
10. [§9 Probabilistic PCA](#9-probabilistic-pca)
11. [§10 PCA and the Linear Autoencoder](#10-pca-and-the-linear-autoencoder)
12. [§11 Why Nonlinear Autoencoders?](#11-why-nonlinear-autoencoders)
13. [§12 Guided Exercises](#12-guided-exercises)
14. [§13 Chapter Summary](#13-chapter-summary)

---

## Notation and Variable Definitions

The central idea of this chapter is that a high-dimensional observation may be controlled by a much smaller number of hidden factors. PCA searches for the most important **linear** factors.

> **Core pipeline.** The formulas that students should remember are
>
> $$
> \mathbf{z}=\mathbf{U}_M^T(\mathbf{x}-\bar{\mathbf{x}}),
> $$
>
> followed by
>
> $$
> \widehat{\mathbf{x}}=\bar{\mathbf{x}}+\mathbf{U}_M\mathbf{z}.
> $$
>
> The first equation is the **encoder / projection**. The second equation is the **decoder / reconstruction**.

| Symbol | Definition |
|---|---|
| $N$ | Number of observations. |
| $D$ | Original data dimensionality. Each observation satisfies $\mathbf{x}_n\in\mathbb{R}^D$. |
| $M$ | Number of retained principal components, usually $M<D$. |
| $\mathbf{x}_n$ | The $n$th observed data vector. |
| $\bar{\mathbf{x}}$ | Sample mean: $\bar{\mathbf{x}}=\frac{1}{N}\sum_n\mathbf{x}_n$. |
| $\widetilde{\mathbf{x}}_n$ | Centered vector: $\widetilde{\mathbf{x}}_n=\mathbf{x}_n-\bar{\mathbf{x}}$. |
| $\mathbf{X}_c$ | Centered data matrix. Its $n$th row is $\widetilde{\mathbf{x}}_n^T$. |
| $\mathbf{S}$ | Sample covariance matrix: $\mathbf{S}=\frac{1}{N}\mathbf{X}_c^T\mathbf{X}_c$. |
| $\lambda_i$ | The $i$th eigenvalue of $\mathbf{S}$, sorted so that $\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_D\geq0$. |
| $\mathbf{u}_i$ | Unit eigenvector associated with $\lambda_i$. It is the $i$th principal direction. |
| $\mathbf{U}_M$ | Matrix containing the first $M$ principal directions: $\mathbf{U}_M=[\mathbf{u}_1,\ldots,\mathbf{u}_M]$. |
| $\mathbf{z}_n$ | $M$-dimensional latent representation of $\mathbf{x}_n$. |
| $\widehat{\mathbf{x}}_n$ | Reconstruction of $\mathbf{x}_n$ from $M$ principal components. |
| $J$ | Mean squared reconstruction distortion. |
| $r_i$ | Explained-variance ratio of component $i$: $r_i=\lambda_i/\sum_j\lambda_j$. |
| $\mathbf{W}$ | Loading matrix in probabilistic PCA. |
| $\sigma^2$ | Isotropic noise variance in probabilistic PCA. |
| $f_\theta, g_\phi$ | Encoder and decoder of an autoencoder. |

A small convention warning:

- Bishop defines the covariance using $1/N$.
- Many statistics packages use $1/(N-1)$.
- This changes the numerical eigenvalues by a common scale factor, but **does not change the principal directions**.

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch. 12 opening; §12.1-§12.4

## 0.1 What Problem Are We Solving?

Modern measurements are often high-dimensional:

- an image may contain thousands or millions of pixels;
- an antenna array may produce measurements from many channels;
- a spectrum may contain hundreds of frequency bins;
- an industrial system may contain many correlated sensor readings;
- a neural-network feature vector may contain hundreds or thousands of activations.

However, the meaningful variation can be much lower-dimensional.

For example, many pixels of a translated image change together. Many neighboring frequency bins are correlated. Many sensors respond to the same physical source. Therefore, the observed coordinates are not all independent degrees of freedom.

PCA asks:

> Can we replace a $D$-dimensional observation by an $M$-dimensional representation, with $M\ll D$, while preserving as much important variation as possible?

## 0.2 What We Will Emphasize

This lecture focuses on the parts most useful for EE students and modern machine learning:

1. data centering;
2. covariance and correlation;
3. maximum-variance projection;
4. eigenvectors and eigenvalues;
5. low-dimensional coordinates;
6. reconstruction;
7. explained variance;
8. visualization, compression, and denoising;
9. probabilistic PCA as a generative latent-variable model;
10. the relationship between PCA and autoencoders.

## 0.3 What We Will Only Introduce Briefly

The textbook contains several advanced extensions. They are valuable, but they are not central to this lecture.

| Topic | Treatment in This Lecture |
|---|---|
| Bayesian PCA | Main motivation only: automatically controlling latent dimensionality. |
| Factor analysis | One-paragraph comparison with probabilistic PCA. |
| Kernel PCA | Intuition only; no feature-space centering derivation. |
| Independent component analysis | Mentioned as a different objective; no full derivation. |
| Complex nonlinear manifold models | Replaced by a clearer connection to nonlinear autoencoders. |

---

# §1 Why Continuous Latent Variables?

> 📖 Textbook Ch. 12 opening, pp. 559-561

## 1.1 Ambient Dimension versus Intrinsic Dimension

The **ambient dimension（环境维度）** is the number of coordinates used to store an observation.

The **intrinsic dimension（内在维度）** is the number of independent factors needed to describe the meaningful variation.

Suppose a $100\times100$ image contains a digit. The ambient dimension is

$$
D=100\times100=10{,}000.
$$

Now imagine that the only changes are:

1. horizontal translation;
2. vertical translation;
3. rotation.

Then the data are controlled mainly by three continuous variables. Although each image is represented by 10,000 numbers, the collection of images is organized by roughly three underlying degrees of freedom.

> ![Figure 12.1](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_1__textbook_fig_12_1__p560.png)
>
> *Figure 12.1 (Textbook Fig. 12.1, p. 560): Copies of one digit are translated and rotated inside a larger image. The pixel space is 10,000-dimensional, but the variation is controlled by a much smaller number of latent variables.*

## 1.2 Latent Variables

A **latent variable（潜变量）** is a variable that is not directly observed but helps explain the observed data.

For the digit example:

- observed variable: pixel vector $\mathbf{x}$;
- possible latent variables: horizontal position, vertical position, and angle.

A latent-variable model tries to explain the observation using a smaller hidden representation:

$$
\mathbf{z}\in\mathbb{R}^M,
\qquad M<D.
$$

A generic generative picture is

$$
\mathbf{z}\longrightarrow\mathbf{x}.
$$

The latent variable determines the underlying structure, and the observed data may also contain noise.

## 1.3 Linear Subspace versus Nonlinear Manifold

A linear subspace is flat. In two dimensions, it is a line through the origin; after adding a mean, it becomes an affine line through the data cloud.

A nonlinear manifold can curve.

The translated-digit example is generally nonlinear in pixel space. Moving an object across a pixel causes that pixel intensity to change in a nonlinear way. PCA can only approximate such a curved structure using a flat linear subspace.

This gives the chapter roadmap:

$$
\text{PCA: linear subspace}
\quad\longrightarrow\quad
\text{autoencoder: potentially nonlinear representation}.
$$

---

# §2 Data Centering and the Covariance Matrix

> 📖 Textbook §12.1.1, equations (12.1)-(12.3)

## 2.1 Step 1: Compute the Mean

Given observations

$$
\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_N\in\mathbb{R}^D,
$$

the sample mean is

$$
\bar{\mathbf{x}}
=
\frac{1}{N}\sum_{n=1}^{N}\mathbf{x}_n.
$$

The mean is the center of the data cloud.

## 2.2 Step 2: Center Every Observation

Define

$$
\widetilde{\mathbf{x}}_n
=
\mathbf{x}_n-\bar{\mathbf{x}}.
$$

Then the centered observations have zero sample mean:

$$
\frac{1}{N}\sum_{n=1}^{N}\widetilde{\mathbf{x}}_n
=
\frac{1}{N}\sum_{n=1}^{N}(\mathbf{x}_n-\bar{\mathbf{x}})
=
\bar{\mathbf{x}}-\bar{\mathbf{x}}
=
\mathbf{0}.
$$

### Why Centering Is Essential

PCA is intended to describe **variation around the mean**, not distance from the coordinate origin.

Without centering, a large constant offset may appear to be an important direction. The first component can then point toward the mean rather than along the data variation.

> **Practical rule.** Fit the mean on the training set, store it, and subtract the same training mean from validation and test data.

Do not compute a separate test-set mean. That would use information from the test set and create data leakage.

## 2.3 Step 3: Form the Covariance Matrix

The sample covariance matrix is

$$
\mathbf{S}
=
\frac{1}{N}\sum_{n=1}^{N}
(\mathbf{x}_n-\bar{\mathbf{x}})
(\mathbf{x}_n-\bar{\mathbf{x}})^T.
$$

Using centered vectors,

$$
\mathbf{S}
=
\frac{1}{N}\sum_{n=1}^{N}
\widetilde{\mathbf{x}}_n\widetilde{\mathbf{x}}_n^T.
$$

If the centered data matrix is

$$
\mathbf{X}_c
=
\begin{bmatrix}
\widetilde{\mathbf{x}}_1^T\\
\widetilde{\mathbf{x}}_2^T\\
\vdots\\
\widetilde{\mathbf{x}}_N^T
\end{bmatrix}
\in\mathbb{R}^{N\times D},
$$

then

$$
\boxed{
\mathbf{S}=\frac{1}{N}\mathbf{X}_c^T\mathbf{X}_c
}
$$

and $\mathbf{S}\in\mathbb{R}^{D\times D}$.

## 2.4 Meaning of the Entries

For two features $i$ and $j$,

$$
S_{ij}
=
\frac{1}{N}\sum_{n=1}^{N}
(x_{ni}-\bar{x}_i)(x_{nj}-\bar{x}_j).
$$

The diagonal entries are variances:

$$
S_{ii}=\operatorname{var}[x_i].
$$

The off-diagonal entries are covariances:

$$
S_{ij}=\operatorname{cov}[x_i,x_j].
$$

Interpretation:

- $S_{ij}>0$: the two features tend to increase together;
- $S_{ij}<0$: one tends to increase when the other decreases;
- $S_{ij}\approx0$: little linear co-variation.

## 2.5 Important Properties of the Covariance Matrix

### Symmetry

$$
\mathbf{S}^T=\mathbf{S}.
$$

Therefore, its eigenvectors can be chosen to be mutually orthonormal.

### Positive Semidefiniteness

For any vector $\mathbf{a}$,

$$
\mathbf{a}^T\mathbf{S}\mathbf{a}
=
\frac{1}{N}\sum_{n=1}^{N}
(\mathbf{a}^T\widetilde{\mathbf{x}}_n)^2
\geq0.
$$

Therefore every eigenvalue satisfies

$$
\lambda_i\geq0.
$$

This property is useful because an eigenvalue will become a variance, and variance cannot be negative.

## 2.6 Principal Directions Can Be Visualized in the Original Space

For image data, each eigenvector has the same number of entries as an image. We can reshape an eigenvector into image form.

> ![Figure 12.3](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_3__textbook_fig_12_3__p566.png)
>
> *Figure 12.3 (Textbook Fig. 12.3, p. 566): The mean digit and the first four PCA eigenvectors. Each eigenvector describes a major pattern of variation around the mean image.*

The eigenvectors should not be interpreted as ordinary images. They contain positive and negative changes. Adding or subtracting an eigenvector from the mean changes the observation along one major direction of variation.

---

# §3 PCA as Maximum-Variance Projection

> 📖 Textbook §12.1.1, pp. 561-563

## 3.1 Start with One Component

We first project the centered $D$-dimensional data onto one direction.

Let $\mathbf{u}_1\in\mathbb{R}^D$ be a unit vector:

$$
\mathbf{u}_1^T\mathbf{u}_1=1.
$$

The scalar projection of $\mathbf{x}_n$ is

$$
z_{n1}
=
\mathbf{u}_1^T(\mathbf{x}_n-\bar{\mathbf{x}})
=
\mathbf{u}_1^T\widetilde{\mathbf{x}}_n.
$$

This scalar tells us where the point lies along the direction $\mathbf{u}_1$.

## 3.2 Mean of the Projected Data

Because the data are centered,

$$
\frac{1}{N}\sum_{n=1}^{N}z_{n1}
=
\frac{1}{N}\sum_{n=1}^{N}\mathbf{u}_1^T\widetilde{\mathbf{x}}_n.
$$

Move the constant vector outside the sum:

$$
\frac{1}{N}\sum_{n=1}^{N}z_{n1}
=
\mathbf{u}_1^T
\left(
\frac{1}{N}\sum_{n=1}^{N}\widetilde{\mathbf{x}}_n
\right)
=
\mathbf{u}_1^T\mathbf{0}
=0.
$$

So the projected coordinates also have zero mean.

## 3.3 Variance of the Projected Data

Since the projected mean is zero, the projected variance is

$$
\operatorname{var}[z_1]
=
\frac{1}{N}\sum_{n=1}^{N}z_{n1}^2.
$$

Substitute $z_{n1}=\mathbf{u}_1^T\widetilde{\mathbf{x}}_n$:

$$
\operatorname{var}[z_1]
=
\frac{1}{N}\sum_{n=1}^{N}
(\mathbf{u}_1^T\widetilde{\mathbf{x}}_n)^2.
$$

A scalar can be written as its transpose, so

$$
(\mathbf{u}_1^T\widetilde{\mathbf{x}}_n)^2
=
\mathbf{u}_1^T\widetilde{\mathbf{x}}_n
\widetilde{\mathbf{x}}_n^T\mathbf{u}_1.
$$

Therefore,

$$
\operatorname{var}[z_1]
=
\mathbf{u}_1^T
\left[
\frac{1}{N}\sum_{n=1}^{N}
\widetilde{\mathbf{x}}_n\widetilde{\mathbf{x}}_n^T
\right]
\mathbf{u}_1.
$$

The matrix inside the brackets is the covariance matrix $\mathbf{S}$:

$$
\boxed{
\operatorname{var}[z_1]=\mathbf{u}_1^T\mathbf{S}\mathbf{u}_1
}
$$

## 3.4 Why We Need a Constraint

If we maximize $\mathbf{u}_1^T\mathbf{S}\mathbf{u}_1$ without constraining $\mathbf{u}_1$, we could multiply $\mathbf{u}_1$ by an arbitrarily large number. The objective would then grow without bound.

We care only about direction, so we impose

$$
\mathbf{u}_1^T\mathbf{u}_1=1.
$$

## 3.5 Lagrange-Multiplier Derivation

We maximize

$$
\mathbf{u}_1^T\mathbf{S}\mathbf{u}_1
$$

subject to

$$
\mathbf{u}_1^T\mathbf{u}_1=1.
$$

Define the Lagrangian

$$
\mathcal{L}(\mathbf{u}_1,\lambda_1)
=
\mathbf{u}_1^T\mathbf{S}\mathbf{u}_1
-
\lambda_1(\mathbf{u}_1^T\mathbf{u}_1-1).
$$

Differentiate with respect to $\mathbf{u}_1$:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{u}_1}
=
2\mathbf{S}\mathbf{u}_1-2\lambda_1\mathbf{u}_1.
$$

At a stationary point,

$$
2\mathbf{S}\mathbf{u}_1-2\lambda_1\mathbf{u}_1=0.
$$

Divide by 2:

$$
\boxed{
\mathbf{S}\mathbf{u}_1=\lambda_1\mathbf{u}_1
}
$$

This is the eigenvector equation.

Therefore the maximizing direction must be an eigenvector of the covariance matrix.

## 3.6 Why the Largest Eigenvalue?

Left-multiply the eigenvector equation by $\mathbf{u}_1^T$:

$$
\mathbf{u}_1^T\mathbf{S}\mathbf{u}_1
=
\lambda_1\mathbf{u}_1^T\mathbf{u}_1.
$$

Since $\mathbf{u}_1$ is a unit vector,

$$
\mathbf{u}_1^T\mathbf{u}_1=1.
$$

Hence

$$
\mathbf{u}_1^T\mathbf{S}\mathbf{u}_1=\lambda_1.
$$

The projected variance equals the eigenvalue. Therefore, to maximize variance, choose the eigenvector with the largest eigenvalue.

> **Interpretation.**
>
> - eigenvector $\mathbf{u}_i$: direction of variation;
> - eigenvalue $\lambda_i$: amount of variance in that direction.

## 3.7 More Than One Component

After choosing $\mathbf{u}_1$, the second component must be orthogonal to the first:

$$
\mathbf{u}_2^T\mathbf{u}_1=0.
$$

Among all such directions, choose the one with maximum remaining variance. This gives the eigenvector associated with the second-largest eigenvalue.

Continue in this way. Sort the eigenvalues:

$$
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_D\geq0.
$$

Then retain

$$
\mathbf{U}_M
=
[\mathbf{u}_1,\mathbf{u}_2,\ldots,\mathbf{u}_M].
$$

Because the eigenvectors are orthonormal,

$$
\mathbf{U}_M^T\mathbf{U}_M=\mathbf{I}_M.
$$

> ![Figure 12.2](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_2__textbook_fig_12_2__p561.png)
>
> *Figure 12.2 (Textbook Fig. 12.2, p. 561): PCA chooses a lower-dimensional subspace. The same subspace can be described either as maximizing projected variance or minimizing orthogonal reconstruction error.*

## 3.8 Guided Example: Eigenvectors of a Simple Covariance Matrix

Consider

$$
\mathbf{S}
=
\begin{bmatrix}
3 & 1\\
1 & 3
\end{bmatrix}.
$$

### Step 1: Find the eigenvalues

Solve

$$
\det(\mathbf{S}-\lambda\mathbf{I})=0.
$$

Thus

$$
\det
\begin{bmatrix}
3-\lambda & 1\\
1 & 3-\lambda
\end{bmatrix}
=0.
$$

The determinant is

$$
(3-\lambda)^2-1=0.
$$

Expand:

$$
9-6\lambda+\lambda^2-1=0,
$$

so

$$
\lambda^2-6\lambda+8=0.
$$

Factor:

$$
(\lambda-4)(\lambda-2)=0.
$$

Therefore,

$$
\lambda_1=4,
\qquad
\lambda_2=2.
$$

### Step 2: Find the first eigenvector

For $\lambda_1=4$,

$$
(\mathbf{S}-4\mathbf{I})\mathbf{u}_1=0.
$$

That is,

$$
\begin{bmatrix}
-1 & 1\\
1 & -1
\end{bmatrix}
\begin{bmatrix}
u_{11}\\u_{12}
\end{bmatrix}
=0.
$$

The first row gives

$$
-u_{11}+u_{12}=0,
$$

so

$$
u_{12}=u_{11}.
$$

A direction vector is therefore $[1,1]^T$. Normalize it:

$$
\mathbf{u}_1
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\1
\end{bmatrix}.
$$

### Step 3: Find the second eigenvector

For $\lambda_2=2$, the eigenvector is proportional to $[1,-1]^T$:

$$
\mathbf{u}_2
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\-1
\end{bmatrix}.
$$

### Interpretation

- The direction $(1,1)$ has variance $4$.
- The orthogonal direction $(1,-1)$ has variance $2$.
- If we retain one component, PCA chooses $(1,1)$.

---

# §4 Projection, Latent Coordinates, and Reconstruction

> 📖 Textbook §12.1.2-§12.1.3, especially equations (12.19)-(12.20)

## 4.1 The PCA Encoder

After selecting $M$ principal directions, the latent representation is

$$
\boxed{
\mathbf{z}_n
=
\mathbf{U}_M^T(\mathbf{x}_n-\bar{\mathbf{x}})
}
$$

Shape check:

$$
\mathbf{U}_M^T\in\mathbb{R}^{M\times D},
\qquad
\mathbf{x}_n-\bar{\mathbf{x}}\in\mathbb{R}^{D},
$$

so

$$
\mathbf{z}_n\in\mathbb{R}^{M}.
$$

The $i$th coordinate is

$$
z_{ni}=\mathbf{u}_i^T(\mathbf{x}_n-\bar{\mathbf{x}}).
$$

Each latent coordinate measures how strongly the centered observation points along one principal direction.

## 4.2 The PCA Decoder

Reconstruct the centered vector by combining the retained directions:

$$
\widehat{\widetilde{\mathbf{x}}}_n
=
\mathbf{U}_M\mathbf{z}_n.
$$

Add the mean back:

$$
\boxed{
\widehat{\mathbf{x}}_n
=
\bar{\mathbf{x}}+\mathbf{U}_M\mathbf{z}_n
}
$$

Substitute the encoder formula:

$$
\widehat{\mathbf{x}}_n
=
\bar{\mathbf{x}}
+
\mathbf{U}_M\mathbf{U}_M^T
(\mathbf{x}_n-\bar{\mathbf{x}}).
$$

The matrix

$$
\mathbf{P}_M=\mathbf{U}_M\mathbf{U}_M^T
$$

is the orthogonal projection matrix onto the principal subspace.

## 4.3 Why $\mathbf{U}_M\mathbf{U}_M^T$ Is a Projection

First,

$$
\mathbf{P}_M^T
=(\mathbf{U}_M\mathbf{U}_M^T)^T
=\mathbf{U}_M\mathbf{U}_M^T
=\mathbf{P}_M.
$$

So the matrix is symmetric.

Second,

$$
\mathbf{P}_M^2
=
\mathbf{U}_M\mathbf{U}_M^T\mathbf{U}_M\mathbf{U}_M^T.
$$

Since $\mathbf{U}_M^T\mathbf{U}_M=\mathbf{I}$,

$$
\mathbf{P}_M^2
=
\mathbf{U}_M\mathbf{I}\mathbf{U}_M^T
=
\mathbf{P}_M.
$$

A matrix satisfying $\mathbf{P}^2=\mathbf{P}$ is a projection matrix.

## 4.4 Reconstruction Error

The residual is

$$
\mathbf{e}_n
=
\mathbf{x}_n-\widehat{\mathbf{x}}_n.
$$

Substitute the reconstruction:

$$
\mathbf{e}_n
=
(\mathbf{x}_n-\bar{\mathbf{x}})
-
\mathbf{U}_M\mathbf{U}_M^T(\mathbf{x}_n-\bar{\mathbf{x}}).
$$

Therefore,

$$
\boxed{
\mathbf{e}_n
=
(\mathbf{I}-\mathbf{U}_M\mathbf{U}_M^T)
(\mathbf{x}_n-\bar{\mathbf{x}})
}
$$

The residual lies in the subspace orthogonal to the retained principal directions.

## 4.5 Guided Numerical Reconstruction

Return to

$$
\mathbf{u}_1
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\1
\end{bmatrix}.
$$

Assume the mean is zero for simplicity and consider

$$
\mathbf{x}
=
\begin{bmatrix}
2\\0
\end{bmatrix}.
$$

### Encode

$$
 z
=
\mathbf{u}_1^T\mathbf{x}
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1
\end{bmatrix}
\begin{bmatrix}
2\\0
\end{bmatrix}.
$$

Thus

$$
 z=\frac{2}{\sqrt{2}}=\sqrt{2}.
$$

### Decode

$$
\widehat{\mathbf{x}}
=
\mathbf{u}_1z
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\1
\end{bmatrix}
\sqrt{2}.
$$

Therefore,

$$
\widehat{\mathbf{x}}
=
\begin{bmatrix}
1\\1
\end{bmatrix}.
$$

### Residual

$$
\mathbf{e}
=
\mathbf{x}-\widehat{\mathbf{x}}
=
\begin{bmatrix}
2\\0
\end{bmatrix}
-
\begin{bmatrix}
1\\1
\end{bmatrix}
=
\begin{bmatrix}
1\\-1
\end{bmatrix}.
$$

The residual lies along the discarded direction $\mathbf{u}_2$.

Its squared norm is

$$
\|\mathbf{e}\|^2
=1^2+(-1)^2
=2.
$$

This example shows that PCA preserves the part of the point lying along the retained direction and removes the orthogonal part.

## 4.6 Compression

Originally, storing one observation requires $D$ numbers.

After PCA, we store

$$
\mathbf{z}_n\in\mathbb{R}^M,
$$

plus the shared model parameters:

- mean $\bar{\mathbf{x}}$;
- principal-direction matrix $\mathbf{U}_M$.

When many observations share the same PCA model, the per-observation storage drops from $D$ values to $M$ values.

> ![Figure 12.5](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_5__textbook_fig_12_5__p567.png)
>
> *Figure 12.5 (Textbook Fig. 12.5, p. 567): A digit reconstructed using different numbers of principal components. Increasing $M$ improves fidelity, but reduces the compression ratio.*

The figure illustrates an important trade-off:

- small $M$: strong compression, blurry reconstruction;
- large $M$: weak compression, accurate reconstruction.

---

# §5 PCA as Minimum Reconstruction Error

> 📖 Textbook §12.1.2, pp. 563-565

The maximum-variance view is not the only definition of PCA. PCA also finds the linear subspace that minimizes mean squared reconstruction error.

## 5.1 Define the Distortion

For each observation, let $\widehat{\mathbf{x}}_n$ be its orthogonal projection onto an $M$-dimensional subspace.

Define the average squared distortion:

$$
J
=
\frac{1}{N}\sum_{n=1}^{N}
\|\mathbf{x}_n-\widehat{\mathbf{x}}_n\|^2.
$$

For a PCA subspace,

$$
\widehat{\mathbf{x}}_n
=
\bar{\mathbf{x}}+
\mathbf{U}_M\mathbf{U}_M^T
(\mathbf{x}_n-\bar{\mathbf{x}}).
$$

## 5.2 Total Variance Splits into Retained and Discarded Variance

A complete orthonormal basis contains all $D$ eigenvectors:

$$
\mathbf{U}
=
[\mathbf{u}_1,\ldots,\mathbf{u}_D].
$$

Every centered point can be expanded as

$$
\widetilde{\mathbf{x}}_n
=
\sum_{i=1}^{D}
(\mathbf{u}_i^T\widetilde{\mathbf{x}}_n)\mathbf{u}_i.
$$

PCA keeps only the first $M$ terms:

$$
\widehat{\widetilde{\mathbf{x}}}_n
=
\sum_{i=1}^{M}
(\mathbf{u}_i^T\widetilde{\mathbf{x}}_n)\mathbf{u}_i.
$$

Therefore the residual contains the discarded terms:

$$
\mathbf{e}_n
=
\sum_{i=M+1}^{D}
(\mathbf{u}_i^T\widetilde{\mathbf{x}}_n)\mathbf{u}_i.
$$

Because the eigenvectors are orthonormal, the squared residual is the sum of squared discarded coefficients:

$$
\|\mathbf{e}_n\|^2
=
\sum_{i=M+1}^{D}
(\mathbf{u}_i^T\widetilde{\mathbf{x}}_n)^2.
$$

Average over the data:

$$
J
=
\sum_{i=M+1}^{D}
\frac{1}{N}\sum_{n=1}^{N}
(\mathbf{u}_i^T\widetilde{\mathbf{x}}_n)^2.
$$

The inner average is the projected variance in direction $\mathbf{u}_i$, which equals $\lambda_i$.

Hence

$$
\boxed{
J_{\min}
=
\sum_{i=M+1}^{D}\lambda_i
}
$$

This is one of the most useful PCA results.

## 5.3 Equivalence of the Two Views

The total variance is

$$
\sum_{i=1}^{D}\lambda_i.
$$

The retained variance is

$$
\sum_{i=1}^{M}\lambda_i.
$$

The discarded variance, which equals the minimum reconstruction distortion, is

$$
\sum_{i=M+1}^{D}\lambda_i.
$$

Since total variance is fixed,

$$
\text{maximize retained variance}
$$

is equivalent to

$$
\text{minimize discarded variance}.
$$

Therefore the maximum-variance and minimum-error formulations produce the same principal subspace.

## 5.4 A Trace Form

The total data variance is

$$
\operatorname{tr}(\mathbf{S})
=
\sum_{i=1}^{D}\lambda_i.
$$

The retained variance is

$$
\operatorname{tr}(\mathbf{U}_M^T\mathbf{S}\mathbf{U}_M)
=
\sum_{i=1}^{M}\lambda_i.
$$

Therefore,

$$
J
=
\operatorname{tr}(\mathbf{S})
-
\operatorname{tr}(\mathbf{U}_M^T\mathbf{S}\mathbf{U}_M).
$$

This compact form appears frequently in optimization and representation learning.

---

# §6 Explained Variance and Choosing the Number of Components

> 📖 Textbook §12.1.3 and §12.2.3; Fig. 12.4

## 6.1 Eigenvalue Spectrum

The ordered eigenvalues

$$
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_D
$$

form the **eigenvalue spectrum（特征值谱）**.

A rapidly decreasing spectrum means that a small number of components captures most of the variation.

> ![Figure 12.4](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_4__textbook_fig_12_4__p566.png)
>
> *Figure 12.4 (Textbook Fig. 12.4, p. 566): Left: eigenvalue spectrum of the digit data. Right: reconstruction distortion given by the sum of discarded eigenvalues.*

## 6.2 Explained-Variance Ratio

The fraction of total variance explained by component $i$ is

$$
\boxed{
r_i
=
\frac{\lambda_i}{\sum_{j=1}^{D}\lambda_j}
}
$$

The ratios satisfy

$$
0\leq r_i\leq1,
$$

and

$$
\sum_{i=1}^{D}r_i=1.
$$

## 6.3 Cumulative Explained Variance

For the first $M$ components,

$$
\boxed{
R_M
=
\frac{\sum_{i=1}^{M}\lambda_i}
{\sum_{j=1}^{D}\lambda_j}
}
$$

This is the cumulative explained-variance ratio.

The unexplained fraction is

$$
1-R_M
=
\frac{\sum_{i=M+1}^{D}\lambda_i}
{\sum_{j=1}^{D}\lambda_j}.
$$

## 6.4 Example

Suppose the eigenvalues are

$$
\lambda_1=9,
\quad
\lambda_2=4,
\quad
\lambda_3=1.
$$

The total variance is

$$
9+4+1=14.
$$

The first component explains

$$
r_1=\frac{9}{14}\approx0.643.
$$

So it explains about $64.3\%$ of the variance.

The first two components explain

$$
R_2
=
\frac{9+4}{14}
=
\frac{13}{14}
\approx0.929.
$$

Thus two components explain about $92.9\%$ of the variance.

The reconstruction distortion after retaining two components is the discarded eigenvalue:

$$
J=\lambda_3=1.
$$

## 6.5 Common Ways to Choose $M$

### Method 1: Explained-Variance Threshold

Choose the smallest $M$ such that

$$
R_M\geq\tau,
$$

where $\tau$ might be $0.90$, $0.95$, or $0.99$.

There is no universal correct threshold. The choice depends on the application.

### Method 2: Scree Plot / Elbow

Plot $\lambda_i$ against $i$ and look for an elbow where the spectrum begins to flatten.

This is intuitive but can be ambiguous.

### Method 3: Validate the Downstream Task

For classification or regression, choose $M$ using validation performance rather than explained variance alone.

A low-variance direction may still contain label information. PCA is unsupervised and does not know which directions are useful for the target.

### Method 4: Reconstruction Constraint

Choose the smallest $M$ whose reconstruction error is below an acceptable engineering tolerance.

For compression, this can be more meaningful than a generic variance threshold.

## 6.6 Important Caveat

High variance does not always mean high task relevance.

A sensor may have a large nuisance variation, while a small but consistent signal is crucial for fault detection or classification. PCA may keep the nuisance direction and remove the discriminative direction.

Therefore:

> PCA is excellent for preserving global variance, but it is not automatically optimal for every supervised task.

---

# §7 Applications: Visualization, Compression, Whitening, and Denoising

> 📖 Textbook §12.1.3, pp. 565-569

## 7.1 Two-Dimensional Visualization

For visualization, set $M=2$:

$$
\mathbf{z}_n
=
\begin{bmatrix}
z_{n1}\\z_{n2}
\end{bmatrix}
=
\mathbf{U}_2^T(\mathbf{x}_n-\bar{\mathbf{x}}).
$$

Plot each observation at coordinates $(z_{n1},z_{n2})$.

> ![Figure 12.7](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_7__textbook_fig_12_8__p569.png)
>
> *Figure 12.7 (Textbook Fig. 12.8, p. 569): Oil-flow measurements projected onto the first two principal components. PCA provides a compact view of the structure in a 12-dimensional data set.*

A PCA plot can reveal:

- clusters;
- outliers;
- trajectories;
- operating regimes;
- changes across time or experimental conditions.

However, a 2D plot may omit important structure if the first two components explain only a modest fraction of the total variance.

## 7.2 Data Compression

PCA compression has two stages.

### Encoding

$$
\mathbf{z}_n=\mathbf{U}_M^T(\mathbf{x}_n-\bar{\mathbf{x}}).
$$

### Decoding

$$
\widehat{\mathbf{x}}_n=\bar{\mathbf{x}}+\mathbf{U}_M\mathbf{z}_n.
$$

The shared basis $\mathbf{U}_M$ acts like a learned transform. Compared with a fixed transform, PCA adapts to the data distribution.

### EE Examples

- compressing correlated array measurements;
- reducing the number of spectral coefficients;
- representing channel-state information;
- compressing vibration signatures;
- reducing image feature dimensionality before communication or storage.

## 7.3 Denoising

Suppose the observation is

$$
\mathbf{x}=\mathbf{s}+\boldsymbol{\epsilon},
$$

where $\mathbf{s}$ is structured signal and $\boldsymbol{\epsilon}$ is noise.

If the signal lies mainly in a low-dimensional subspace and the noise is spread across many directions, then projecting onto the leading components can suppress noise:

$$
\widehat{\mathbf{s}}
=
\bar{\mathbf{x}}+
\mathbf{U}_M\mathbf{U}_M^T
(\mathbf{x}-\bar{\mathbf{x}}).
$$

The discarded component is

$$
(\mathbf{I}-\mathbf{U}_M\mathbf{U}_M^T)
(\mathbf{x}-\bar{\mathbf{x}}).
$$

This is useful when low-variance directions are dominated by measurement noise.

### Denoising Caveat

PCA does not know which variation is noise. If useful signal has low variance, PCA can remove it.

Denoising PCA works best when:

1. signal directions are repeatable across samples;
2. signal variance is larger than noise variance;
3. the same subspace is valid for training and test data.

## 7.4 Centering, Standardization, and Whitening Are Different

These operations are often confused.

### Centering

$$
\widetilde{\mathbf{x}}
=
\mathbf{x}-\bar{\mathbf{x}}.
$$

Result: zero mean.

### Standardization

For feature $i$,

$$
y_i
=
\frac{x_i-\bar{x}_i}{s_i},
$$

where $s_i$ is its standard deviation.

Result: each original feature has approximately unit variance, but different features may remain correlated.

### PCA Rotation

$$
\mathbf{z}
=
\mathbf{U}^T(\mathbf{x}-\bar{\mathbf{x}}).
$$

The covariance in the PCA coordinate system is diagonal:

$$
\operatorname{cov}[\mathbf{z}]
=
\boldsymbol{\Lambda}.
$$

The coordinates are uncorrelated, but their variances are $\lambda_i$, not necessarily one.

### Whitening

Define

$$
\boxed{
\mathbf{y}
=
\boldsymbol{\Lambda}^{-1/2}
\mathbf{U}^T(\mathbf{x}-\bar{\mathbf{x}})
}
$$

Then

$$
\operatorname{cov}[\mathbf{y}]
=
\mathbf{I}.
$$

Whitening performs three operations:

1. center;
2. rotate into PCA coordinates;
3. divide each coordinate by its standard deviation $\sqrt{\lambda_i}$.

> ![Figure 12.6](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_6__textbook_fig_12_6__p568.png)
>
> *Figure 12.6 (Textbook Fig. 12.6, p. 568): Left: original data. Center: standardized variables. Right: whitened data with zero mean and identity covariance.*

### Numerical Caution for Whitening

If an eigenvalue is very small, division by $\sqrt{\lambda_i}$ can strongly amplify noise.

A common stabilized form is

$$
\mathbf{y}
=
(\boldsymbol{\Lambda}+\epsilon\mathbf{I})^{-1/2}
\mathbf{U}^T(\mathbf{x}-\bar{\mathbf{x}}),
$$

with small $\epsilon>0$.

## 7.5 Outlier Detection Using Reconstruction Error

A simple anomaly score is

$$
a(\mathbf{x})
=
\|\mathbf{x}-\widehat{\mathbf{x}}\|^2.
$$

A large score means that the point lies far from the learned principal subspace.

This is useful for:

- sensor faults;
- unusual spectra;
- manufacturing anomalies;
- unexpected operating conditions.

But conventional PCA considers only distance from the subspace. A point can lie close to the subspace while being far outside the region occupied by training data. Probabilistic PCA partly addresses this limitation by defining a full density model.

---

# §8 Practical PCA Algorithms and High-Dimensional Data

> 📖 Textbook §12.1.4, pp. 569-570

## 8.1 Basic Eigenvalue Algorithm

Given $\mathbf{X}\in\mathbb{R}^{N\times D}$:

1. compute the training mean $\bar{\mathbf{x}}$;
2. center the data to obtain $\mathbf{X}_c$;
3. compute $\mathbf{S}=\frac{1}{N}\mathbf{X}_c^T\mathbf{X}_c$;
4. eigendecompose $\mathbf{S}$;
5. sort eigenvalues in decreasing order;
6. retain the first $M$ eigenvectors;
7. project and reconstruct as required.

### Pseudocode

```text
Input: data matrix X with N rows and D columns

mean = average of rows of X
Xc   = X - mean
S    = (1/N) Xc^T Xc
[eigenvalues, eigenvectors] = eig(S)
sort eigenpairs by decreasing eigenvalue
U_M  = first M eigenvectors
Z    = Xc U_M
Xhat = Z U_M^T + mean
```

The matrix version of the encoder is

$$
\mathbf{Z}=\mathbf{X}_c\mathbf{U}_M,
$$

where $\mathbf{Z}\in\mathbb{R}^{N\times M}$.

The matrix reconstruction is

$$
\widehat{\mathbf{X}}
=
\mathbf{Z}\mathbf{U}_M^T
+
\mathbf{1}\bar{\mathbf{x}}^T.
$$

## 8.2 Why Explicit Covariance Can Be Expensive

The covariance matrix has size

$$
D\times D.
$$

A full eigendecomposition costs roughly

$$
O(D^3).
$$

This is unattractive when $D$ is very large.

For example, an image can have millions of dimensions, while the number of training images may be only hundreds or thousands.

## 8.3 Rank Limitation When $N<D$

After centering, $N$ data points span at most an $(N-1)$-dimensional subspace.

Therefore,

$$
\operatorname{rank}(\mathbf{X}_c)\leq N-1.
$$

Consequently,

$$
\operatorname{rank}(\mathbf{S})\leq N-1.
$$

If $N<D$, at most $N-1$ eigenvalues can be nonzero.

This means there is no benefit in asking for more than $N-1$ nonzero principal components.

## 8.4 The Smaller Gram-Matrix Trick

Instead of diagonalizing

$$
\mathbf{S}
=
\frac{1}{N}\mathbf{X}_c^T\mathbf{X}_c
\in\mathbb{R}^{D\times D},
$$

consider

$$
\mathbf{K}
=
\frac{1}{N}\mathbf{X}_c\mathbf{X}_c^T
\in\mathbb{R}^{N\times N}.
$$

Suppose

$$
\mathbf{K}\mathbf{v}_i
=
\lambda_i\mathbf{v}_i.
$$

Then the corresponding eigenvector in the original feature space is

$$
\boxed{
\mathbf{u}_i
=
\frac{1}{\sqrt{N\lambda_i}}
\mathbf{X}_c^T\mathbf{v}_i
}
$$

for $\lambda_i>0$.

This changes the eigendecomposition cost from being based on $D\times D$ to being based on $N\times N$.

It is useful when

$$
N\ll D.
$$

## 8.5 SVD: The Most Common Practical Route

Compute the singular value decomposition

$$
\mathbf{X}_c
=
\mathbf{Q}\boldsymbol{\Sigma}\mathbf{U}^T.
$$

Then the columns of $\mathbf{U}$ are the principal directions.

To see this,

$$
\mathbf{S}
=
\frac{1}{N}\mathbf{X}_c^T\mathbf{X}_c.
$$

Substitute the SVD:

$$
\mathbf{S}
=
\frac{1}{N}
(\mathbf{U}\boldsymbol{\Sigma}^T\mathbf{Q}^T)
(\mathbf{Q}\boldsymbol{\Sigma}\mathbf{U}^T).
$$

Since $\mathbf{Q}^T\mathbf{Q}=\mathbf{I}$,

$$
\mathbf{S}
=
\mathbf{U}
\left(
\frac{\boldsymbol{\Sigma}^T\boldsymbol{\Sigma}}{N}
\right)
\mathbf{U}^T.
$$

Therefore,

$$
\lambda_i=\frac{\sigma_i^2}{N},
$$

where $\sigma_i$ is the $i$th singular value.

### Why SVD Is Often Preferred

- avoids explicitly forming the covariance matrix;
- is numerically stable;
- supports truncated algorithms that compute only leading components;
- fits naturally with sparse and large data matrices.

## 8.6 Practical Checklist

Before applying PCA:

1. **Split the data first.** Fit PCA only on the training set.
2. **Center using the training mean.**
3. **Decide whether to standardize.** Features with different units may require scaling.
4. **Inspect eigenvalues and cumulative explained variance.**
5. **Validate $M$ on the downstream task.**
6. **Check reconstruction examples, not only a variance number.**
7. **Watch for outliers.** Covariance and PCA can be strongly affected by extreme points.
8. **Use SVD or truncated SVD for large problems.**
9. **Store the full preprocessing pipeline.** Mean, scale, and PCA basis must be reused at test time.

---

# §9 Probabilistic PCA

> 📖 Textbook §12.2, especially pp. 570-580

Classical PCA gives a subspace and a deterministic projection. Probabilistic PCA (PPCA) gives a generative probability model.

## 9.1 The Linear-Gaussian Latent Model

Assume an $M$-dimensional latent variable

$$
\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

Generate the observation by

$$
\boxed{
\mathbf{x}
=
\mathbf{W}\mathbf{z}
+
\boldsymbol{\mu}
+
\boldsymbol{\epsilon}
}
$$

where

$$
\boldsymbol{\epsilon}
\sim
\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I}).
$$

Equivalently,

$$
p(\mathbf{x}\mid\mathbf{z})
=
\mathcal{N}
(\mathbf{x}\mid\mathbf{W}\mathbf{z}+\boldsymbol{\mu},
\sigma^2\mathbf{I}).
$$

> ![Figure 12.8](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_8__textbook_fig_12_9__p572.png)
>
> *Figure 12.8 (Textbook Fig. 12.9, p. 572): Generative view of PPCA. First sample a latent coordinate, map it into data space, and then add isotropic Gaussian noise.*

## 9.2 Meaning of the Parameters

| Parameter | Meaning |
|---|---|
| $\boldsymbol{\mu}$ | Mean of the observed data. |
| $\mathbf{W}$ | Maps the latent vector into the observed space. Its columns span the principal subspace at the maximum-likelihood solution. |
| $\sigma^2$ | Variance of isotropic observation noise. |

## 9.3 Marginal Distribution of the Observation

Because a linear transformation of a Gaussian variable is Gaussian, $\mathbf{x}$ is Gaussian.

Its mean is

$$
\mathbb{E}[\mathbf{x}]
=
\mathbf{W}\mathbb{E}[\mathbf{z}]
+
\boldsymbol{\mu}
+
\mathbb{E}[\boldsymbol{\epsilon}].
$$

Since both random terms have zero mean,

$$
\mathbb{E}[\mathbf{x}]=\boldsymbol{\mu}.
$$

Its covariance is

$$
\operatorname{cov}[\mathbf{x}]
=
\mathbf{W}\operatorname{cov}[\mathbf{z}]\mathbf{W}^T
+
\operatorname{cov}[\boldsymbol{\epsilon}].
$$

Using $\operatorname{cov}[\mathbf{z}]=\mathbf{I}$,

$$
\boxed{
\mathbf{C}
=
\mathbf{W}\mathbf{W}^T
+
\sigma^2\mathbf{I}
}
$$

Therefore,

$$
p(\mathbf{x})
=
\mathcal{N}
(\mathbf{x}\mid\boldsymbol{\mu},\mathbf{C}).
$$

PPCA is a restricted Gaussian model:

- $\mathbf{W}\mathbf{W}^T$ captures dominant correlated variation;
- $\sigma^2\mathbf{I}$ represents equal residual noise in all directions.

## 9.4 Posterior Latent Representation

Given an observation $\mathbf{x}$, the posterior over $\mathbf{z}$ is Gaussian.

Define

$$
\mathbf{M}
=
\mathbf{W}^T\mathbf{W}+\sigma^2\mathbf{I}.
$$

Then the posterior mean is

$$
\boxed{
\mathbb{E}[\mathbf{z}\mid\mathbf{x}]
=
\mathbf{M}^{-1}
\mathbf{W}^T
(\mathbf{x}-\boldsymbol{\mu})
}
$$

This is the probabilistic counterpart of the PCA encoder.

Unlike classical PCA, PPCA also gives posterior uncertainty around the latent coordinate.

## 9.5 Connection to Ordinary PCA

Let the covariance eigenvectors be collected in $\mathbf{U}_M$, and let

$$
\boldsymbol{\Lambda}_M
=
\operatorname{diag}(\lambda_1,\ldots,\lambda_M).
$$

The maximum-likelihood loading matrix has the form

$$
\mathbf{W}_{\mathrm{ML}}
=
\mathbf{U}_M
(\boldsymbol{\Lambda}_M-\sigma^2\mathbf{I})^{1/2}
\mathbf{R},
$$

where $\mathbf{R}$ is any orthogonal rotation in latent space.

The noise estimate is

$$
\boxed{
\sigma^2_{\mathrm{ML}}
=
\frac{1}{D-M}
\sum_{i=M+1}^{D}\lambda_i
}
$$

Thus PPCA interprets the average discarded variance as isotropic noise.

The arbitrary matrix $\mathbf{R}$ means that the latent coordinate axes are not uniquely identified. Rotating the latent space does not change the observed-data distribution.

In the small-noise limit, the principal subspace becomes the ordinary PCA subspace.

## 9.6 Why Use a Probabilistic Formulation?

PPCA offers several benefits.

### Missing Values

A probabilistic model can marginalize missing measurements and use EM for estimation.

> ![Figure 12.9](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_9__textbook_fig_12_11__p580.png)
>
> *Figure 12.9 (Textbook Fig. 12.11, p. 580): PPCA visualization with complete data and with 30% of variable values omitted. EM can handle the missing measurements in a principled way.*

### Uncertainty

The latent representation is a posterior distribution rather than only a point.

### Density Estimation

PPCA defines $p(\mathbf{x})$. This allows likelihood evaluation and comparison with other probabilistic models.

### Generation

Sample

$$
\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
$$

then sample

$$
\mathbf{x}\sim
\mathcal{N}(\mathbf{W}\mathbf{z}+\boldsymbol{\mu},
\sigma^2\mathbf{I}).
$$

### Mixtures and Local Linear Models

A mixture of PPCA models can approximate a curved manifold using multiple local linear subspaces.

## 9.7 Classical PCA versus PPCA

| Aspect | Classical PCA | Probabilistic PCA |
|---|---|---|
| Output | Principal subspace | Probability distribution and latent posterior |
| Projection | Deterministic | Posterior mean with uncertainty |
| Noise | Not explicit | Isotropic Gaussian noise $\sigma^2\mathbf{I}$ |
| Missing data | Not naturally handled | Can be handled through probabilistic inference / EM |
| Generation | No explicit generative sampling rule | Yes |
| Model comparison | Usually reconstruction or explained variance | Likelihood is available |
| Main computation | Eigenvalue decomposition or SVD | Closed form or EM |

## 9.8 Factor Analysis in One Paragraph

Factor analysis uses

$$
\mathbf{x}=\mathbf{W}\mathbf{z}+\boldsymbol{\mu}+\boldsymbol{\epsilon},
$$

but allows

$$
\operatorname{cov}[\boldsymbol{\epsilon}]
=
\boldsymbol{\Psi},
$$

where $\boldsymbol{\Psi}$ is diagonal rather than $\sigma^2\mathbf{I}$.

Thus each observed feature can have its own noise variance. This is more flexible than PPCA, but the clean closed-form PCA solution is lost and iterative estimation is generally required.

## 9.9 Bayesian PCA: Main Idea Only

Bayesian PCA places priors on columns of $\mathbf{W}$. If a column is unnecessary, its prior can shrink it toward zero. This provides an automatic-relevance-determination view of selecting the effective latent dimension.

We do not derive Bayesian PCA in this lecture.

---

# §10 PCA and the Linear Autoencoder

> 📖 Textbook §12.4.2, pp. 592-594

## 10.1 Autoencoder Structure

An autoencoder is trained to reconstruct its own input.

Encoder:

$$
\mathbf{z}=f_\theta(\mathbf{x}).
$$

Decoder:

$$
\widehat{\mathbf{x}}=g_\phi(\mathbf{z}).
$$

Training objective:

$$
\min_{\theta,\phi}
\frac{1}{N}\sum_{n=1}^{N}
\|\mathbf{x}_n-
 g_\phi(f_\theta(\mathbf{x}_n))\|^2.
$$

When the bottleneck has dimension $M<D$, the network must compress the input.

> ![Figure 12.10](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_10__textbook_fig_12_18__p593.png)
>
> *Figure 12.10 (Textbook Fig. 12.18, p. 593): A shallow autoassociative network with a low-dimensional hidden layer. Under the linear setting and squared reconstruction loss, the optimum spans the PCA principal subspace.*

## 10.2 Linear Encoder and Decoder

Use centered inputs and linear maps:

$$
\mathbf{z}
=
\mathbf{W}_e(\mathbf{x}-\bar{\mathbf{x}}),
$$

$$
\widehat{\mathbf{x}}
=
\bar{\mathbf{x}}+
\mathbf{W}_d\mathbf{z}.
$$

Together,

$$
\widehat{\mathbf{x}}
=
\bar{\mathbf{x}}+
\mathbf{W}_d\mathbf{W}_e
(\mathbf{x}-\bar{\mathbf{x}}).
$$

The squared-error objective is

$$
E(\mathbf{W}_e,\mathbf{W}_d)
=
\frac{1}{N}\sum_{n=1}^{N}
\left\|
\mathbf{x}_n-
\bar{\mathbf{x}}-
\mathbf{W}_d\mathbf{W}_e
(\mathbf{x}_n-\bar{\mathbf{x}})
\right\|^2.
$$

## 10.3 PCA Is a Special Linear Autoencoder

Choose

$$
\mathbf{W}_e=\mathbf{U}_M^T,
$$

and

$$
\mathbf{W}_d=\mathbf{U}_M.
$$

Then

$$
\mathbf{z}
=
\mathbf{U}_M^T(\mathbf{x}-\bar{\mathbf{x}}),
$$

and

$$
\widehat{\mathbf{x}}
=
\bar{\mathbf{x}}+
\mathbf{U}_M\mathbf{z}.
$$

These are exactly the PCA projection and reconstruction equations.

Therefore PCA can be viewed as a linear autoencoder with:

- a bottleneck of size $M$;
- squared reconstruction error;
- encoder and decoder aligned with orthonormal principal directions.

## 10.4 The Important Theorem

For a linear autoencoder with an undercomplete bottleneck and squared reconstruction loss, the optimal reconstruction subspace is the same subspace spanned by the first $M$ principal components.

This does **not** mean that every trained weight matrix must equal $\mathbf{U}_M$ exactly.

Why not?

Suppose $\mathbf{A}\in\mathbb{R}^{M\times M}$ is invertible. We can use

$$
\mathbf{W}_e
=
\mathbf{A}\mathbf{U}_M^T,
$$

and

$$
\mathbf{W}_d
=
\mathbf{U}_M\mathbf{A}^{-1}.
$$

Then

$$
\mathbf{W}_d\mathbf{W}_e
=
\mathbf{U}_M\mathbf{A}^{-1}
\mathbf{A}\mathbf{U}_M^T
=
\mathbf{U}_M\mathbf{U}_M^T.
$$

The reconstruction is unchanged, but the latent coordinates use a different basis.

Thus the key result is:

> The optimal **subspace** matches PCA, although the hidden coordinates and weights need not be the normalized PCA eigenvectors.

## 10.5 PCA versus Linear Autoencoder

| Aspect | PCA | Linear Autoencoder |
|---|---|---|
| Objective | Maximize variance / minimize squared reconstruction error | Minimize squared reconstruction error |
| Solution | Closed form through eigenvalue decomposition or SVD | Usually optimized by gradient descent |
| Global optimum | Directly available | Same optimal subspace, but optimization may be slower |
| Basis | Orthonormal and ordered | Need not be orthogonal or ordered |
| Latent axes | Principal components | Any invertible reparameterization of the same subspace may occur |
| Training | No learning rate or epochs | Requires optimizer, learning rate, initialization, and stopping criterion |

For a purely linear problem, PCA is usually preferable because it is direct, stable, and interpretable.

---

# §11 Why Nonlinear Autoencoders?

> 📖 Textbook §12.4.2-§12.4.3, pp. 594-598

## 11.1 Limitation of a Linear Subspace

PCA approximates data using a flat $M$-dimensional subspace.

If the data lie near a curved manifold, a single flat subspace may require many dimensions to represent it accurately.

Examples include:

- object pose changes;
- rotations and translations in images;
- nonlinear sensor response;
- speech articulation;
- nonlinear dynamics;
- changing propagation environments.

## 11.2 Deep Nonlinear Autoencoder

A nonlinear encoder can be written as

$$
\mathbf{z}=f_\theta(\mathbf{x}),
$$

where $f_\theta$ contains nonlinear activation functions.

A nonlinear decoder is

$$
\widehat{\mathbf{x}}=g_\phi(\mathbf{z}).
$$

The network learns both:

1. how to map data to a low-dimensional coordinate system;
2. how to map the coordinate system back to a curved surface in the original space.

> ![Figure 12.11](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_11__textbook_fig_12_19__p594.png)
>
> *Figure 12.11 (Textbook Fig. 12.19, p. 594): Adding nonlinear hidden layers before and after the bottleneck permits nonlinear dimensionality reduction.*

## 11.3 Geometrical Interpretation

Let

$$
F_1:\mathbb{R}^{D}\rightarrow\mathbb{R}^{M}
$$

be the encoder, and

$$
F_2:\mathbb{R}^{M}\rightarrow\mathbb{R}^{D}
$$

be the decoder.

The encoder assigns low-dimensional coordinates. The decoder embeds those coordinates back into the original data space.

If $F_2$ is nonlinear, the decoded set can form a curved manifold rather than a flat plane.

> ![Figure 12.12](./CoursePR2026/Fig/Chapter_12/lecture_fig_12_12__textbook_fig_12_20__p594.png)
>
> *Figure 12.12 (Textbook Fig. 12.20, p. 594): The encoder maps a point to a low-dimensional space $S$, and the nonlinear decoder maps $S$ to a curved surface in the original data space.*

## 11.4 Why a Bottleneck Alone May Not Be Enough

An autoencoder with too much capacity can learn a near-identity mapping without learning useful structure.

Useful constraints include:

- undercomplete bottleneck $M<D$;
- weight decay;
- sparse latent activations;
- adding noise to the input;
- limiting network depth or width;
- early stopping;
- task-specific regularization.

## 11.5 Denoising Autoencoder Motivation

Instead of feeding a clean input directly, corrupt it:

$$
\widetilde{\mathbf{x}}
\sim q(\widetilde{\mathbf{x}}\mid\mathbf{x}).
$$

Train the network to reconstruct the clean input:

$$
\min_{\theta,\phi}
\mathbb{E}
\left[
\|\mathbf{x}-g_\phi(f_\theta(\widetilde{\mathbf{x}}))\|^2
\right].
$$

This discourages memorizing the identity function and encourages the model to learn stable local structure.

PCA denoising and denoising autoencoders share a common goal:

> preserve structured variation while removing unstable variation.

The difference is that PCA uses a linear subspace, whereas a nonlinear autoencoder can learn a curved representation.

## 11.6 PCA versus Nonlinear Autoencoder

| Aspect | PCA | Nonlinear Autoencoder |
|---|---|---|
| Geometry | Linear subspace | Nonlinear manifold approximation |
| Optimization | Closed-form eigen/SVD solution | Non-convex gradient-based training |
| Interpretability | Eigenvalues and orthogonal components | Latent dimensions are usually less directly interpretable |
| Data requirement | Often works well with modest data | Usually benefits from more data |
| Reconstruction | Linear | Nonlinear |
| Component ordering | Naturally ordered by variance | No automatic ordering |
| Inverse mapping | Explicit | Learned decoder |
| Generalization risk | Relatively controlled | Can overfit or learn identity-like shortcuts |

## 11.7 Kernel PCA in One Slide

Kernel PCA first maps the input implicitly into a feature space:

$$
\mathbf{x}\mapsto\boldsymbol{\phi}(\mathbf{x}),
$$

and then performs linear PCA in that feature space.

Because the feature map can be nonlinear, the resulting components correspond to nonlinear structure in the original input space.

We skip the full feature-space centering formula in this lecture.

Main limitations:

- computation depends on an $N\times N$ kernel matrix;
- scaling to very large $N$ is difficult;
- reconstruction in the original input space is not generally direct; this is the pre-image problem.

For modern representation learning, nonlinear autoencoders often provide a more flexible learned encoder and an explicit decoder.

## 11.8 ICA in One Paragraph

PCA finds uncorrelated directions ordered by variance. Independent component analysis seeks components that are statistically independent, usually by exploiting non-Gaussianity.

PCA is appropriate when the main goal is compression, visualization, or squared-error reconstruction. ICA is useful for source-separation problems such as separating mixed signals.

We do not derive ICA in this lecture.

---

# §12 Guided Exercises

## Exercise 12.1: Centering a Small Data Set

Consider three observations:

$$
\mathbf{x}_1=
\begin{bmatrix}
1\\2
\end{bmatrix},
\qquad
\mathbf{x}_2=
\begin{bmatrix}
3\\4
\end{bmatrix},
\qquad
\mathbf{x}_3=
\begin{bmatrix}
5\\6
\end{bmatrix}.
$$

### Question 1

Compute the sample mean.

### Solution

$$
\bar{\mathbf{x}}
=
\frac{1}{3}
\left(
\begin{bmatrix}1\\2\end{bmatrix}
+
\begin{bmatrix}3\\4\end{bmatrix}
+
\begin{bmatrix}5\\6\end{bmatrix}
\right).
$$

Add coordinatewise:

$$
\bar{\mathbf{x}}
=
\frac{1}{3}
\begin{bmatrix}
9\\12
\end{bmatrix}
=
\begin{bmatrix}
3\\4
\end{bmatrix}.
$$

### Question 2

Compute the centered vectors.

### Solution

$$
\widetilde{\mathbf{x}}_1
=
\begin{bmatrix}
1\\2
\end{bmatrix}
-
\begin{bmatrix}
3\\4
\end{bmatrix}
=
\begin{bmatrix}
-2\\-2
\end{bmatrix},
$$

$$
\widetilde{\mathbf{x}}_2
=
\begin{bmatrix}
0\\0
\end{bmatrix},
$$

$$
\widetilde{\mathbf{x}}_3
=
\begin{bmatrix}
2\\2
\end{bmatrix}.
$$

Check the mean:

$$
\frac{1}{3}
\left(
\begin{bmatrix}-2\\-2\end{bmatrix}
+
\begin{bmatrix}0\\0\end{bmatrix}
+
\begin{bmatrix}2\\2\end{bmatrix}
\right)
=
\begin{bmatrix}0\\0\end{bmatrix}.
$$

## Exercise 12.2: Covariance and Principal Direction

Using the centered vectors above, compute the covariance matrix with the textbook convention $1/N$.

### Solution

$$
\mathbf{S}
=
\frac{1}{3}
\sum_{n=1}^{3}
\widetilde{\mathbf{x}}_n
\widetilde{\mathbf{x}}_n^T.
$$

For the first point,

$$
\widetilde{\mathbf{x}}_1
\widetilde{\mathbf{x}}_1^T
=
\begin{bmatrix}
-2\\-2
\end{bmatrix}
\begin{bmatrix}
-2 & -2
\end{bmatrix}
=
\begin{bmatrix}
4 & 4\\
4 & 4
\end{bmatrix}.
$$

For the second point, the outer product is zero.

For the third point,

$$
\widetilde{\mathbf{x}}_3
\widetilde{\mathbf{x}}_3^T
=
\begin{bmatrix}
4 & 4\\
4 & 4
\end{bmatrix}.
$$

Therefore,

$$
\mathbf{S}
=
\frac{1}{3}
\begin{bmatrix}
8 & 8\\
8 & 8
\end{bmatrix}
=
\begin{bmatrix}
8/3 & 8/3\\
8/3 & 8/3
\end{bmatrix}.
$$

The data lie exactly along the direction $[1,1]^T$, so the first principal direction is

$$
\mathbf{u}_1
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\1
\end{bmatrix}.
$$

The orthogonal direction has zero variance.

## Exercise 12.3: Explained Variance

Suppose the eigenvalues are

$$
10,
\quad 5,
\quad 3,
\quad 1,
\quad 1.
$$

### Question 1

What fraction is explained by the first component?

### Solution

Total variance:

$$
10+5+3+1+1=20.
$$

Thus

$$
r_1=\frac{10}{20}=0.5.
$$

The first component explains $50\%$.

### Question 2

How many components are needed for at least $90\%$ cumulative explained variance?

### Solution

First component:

$$
\frac{10}{20}=0.50.
$$

First two:

$$
\frac{10+5}{20}=0.75.
$$

First three:

$$
\frac{10+5+3}{20}=0.90.
$$

Therefore,

$$
M=3.
$$

### Question 3

What is the minimum average squared reconstruction distortion for $M=3$?

### Solution

Discard the last two eigenvalues:

$$
J_{\min}=1+1=2.
$$

## Exercise 12.4: Projection and Reconstruction

Let

$$
\bar{\mathbf{x}}
=
\begin{bmatrix}
1\\1
\end{bmatrix},
\qquad
\mathbf{u}_1
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\1
\end{bmatrix},
$$

and

$$
\mathbf{x}
=
\begin{bmatrix}
3\\1
\end{bmatrix}.
$$

### Encode

Center the point:

$$
\mathbf{x}-\bar{\mathbf{x}}
=
\begin{bmatrix}
2\\0
\end{bmatrix}.
$$

Project:

$$
z
=
\mathbf{u}_1^T(\mathbf{x}-\bar{\mathbf{x}})
=
\sqrt{2}.
$$

### Decode

$$
\widehat{\mathbf{x}}
=
\bar{\mathbf{x}}+\mathbf{u}_1z.
$$

Thus

$$
\widehat{\mathbf{x}}
=
\begin{bmatrix}
1\\1
\end{bmatrix}
+
\begin{bmatrix}
1\\1
\end{bmatrix}
=
\begin{bmatrix}
2\\2
\end{bmatrix}.
$$

### Reconstruction Error

$$
\mathbf{x}-\widehat{\mathbf{x}}
=
\begin{bmatrix}
1\\-1
\end{bmatrix},
$$

and

$$
\|\mathbf{x}-\widehat{\mathbf{x}}\|^2=2.
$$

## Exercise 12.5: PCA or Autoencoder?

For each scenario, choose a reasonable first method.

### Scenario A

You have 300 samples, 80 correlated sensor features, and need an interpretable baseline.

**Answer:** Start with PCA. It is stable, fast, and gives explained variance and interpretable directions.

### Scenario B

You have millions of images and believe the data lie on a strongly curved manifold.

**Answer:** A nonlinear autoencoder may be more suitable, provided that enough data and careful validation are available.

### Scenario C

You have missing sensor values and want a likelihood-based latent model.

**Answer:** Probabilistic PCA is a natural baseline.

### Scenario D

You need to separate independent mixed sources rather than preserve maximum variance.

**Answer:** ICA is conceptually closer to the objective than PCA.

---

# §13 Chapter Summary

## 13.1 The PCA Recipe

Given $N$ observations in $D$ dimensions:

1. compute the training mean;
2. center the observations;
3. compute the covariance matrix or directly use SVD;
4. find eigenvectors and eigenvalues;
5. sort components by decreasing eigenvalue;
6. choose $M$;
7. encode by projection;
8. decode by reconstruction.

The two core equations are

$$
\boxed{
\mathbf{z}=\mathbf{U}_M^T(\mathbf{x}-\bar{\mathbf{x}})
}
$$

and

$$
\boxed{
\widehat{\mathbf{x}}=\bar{\mathbf{x}}+\mathbf{U}_M\mathbf{z}
}
$$

## 13.2 The Three Meanings of Eigenvalues

For PCA, $\lambda_i$ simultaneously represents:

1. variance of the data along $\mathbf{u}_i$;
2. contribution of component $i$ to explained variance;
3. reconstruction distortion incurred if that component is discarded.

## 13.3 The Two Equivalent Definitions

PCA finds the subspace that

$$
\text{maximizes retained variance}
$$

and equivalently

$$
\text{minimizes squared reconstruction error}.
$$

## 13.4 Connection to Representation Learning

PCA is an early and still highly useful representation-learning method.

- The encoder produces a compact feature vector.
- The decoder reconstructs the observation.
- A linear autoencoder learns the same optimal principal subspace under squared loss.
- A nonlinear autoencoder generalizes the idea to curved representations.

This creates a direct conceptual bridge:

$$
\text{PCA}
\longrightarrow
\text{linear autoencoder}
\longrightarrow
\text{nonlinear autoencoder}
\longrightarrow
\text{modern representation learning}.
$$

## 13.5 Final Practical Message

PCA should usually be one of the first baselines for high-dimensional continuous data because it is:

- simple;
- fast;
- deterministic;
- interpretable;
- useful for visualization;
- useful for compression;
- useful for denoising;
- closely connected to modern latent-representation models.

Its main limitation is equally important:

> PCA can only learn a linear subspace and preserves variance rather than task-specific information.

That limitation motivates probabilistic and nonlinear latent-variable models.

---

*End of Lecture 12*
