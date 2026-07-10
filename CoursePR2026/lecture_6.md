# Pattern Recognition and Machine Learning
## Chapter 6: Kernel Methods

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 6 Kernel Methods (§6.1-§6.4)

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Dual Representations](#1-dual-representations)
3. [§2 Constructing Kernels](#2-constructing-kernels)
4. [§3 Radial Basis Function Networks](#3-radial-basis-function-networks)
5. [§4 Gaussian Processes](#4-gaussian-processes)
6. [§5 Guided Textbook Examples and In-class Problems](#5-guided-textbook-examples-and-in-class-problems)
7. [§6 Chapter Summary, Figure Checklist, and Teaching Flow](#6-chapter-summary-figure-checklist-and-teaching-flow)

---

## Notation and Variable Definitions

Chapter 5 studied neural networks as **adaptive basis-function models**. Chapter 6 studies a different route to nonlinear learning. Instead of explicitly building many nonlinear features, we often only need their inner products. This leads to **kernel methods**.

The central idea is:

$$
\text{replace } \boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}')
\quad\text{by}\quad
k(\mathbf{x},\mathbf{x}').
$$

Here $k(\mathbf{x},\mathbf{x}')$ is called a **kernel function**. It measures similarity between two inputs, but in a way that corresponds to an inner product in some feature space.

### Generic Kernel Notation

| Symbol | Definition |
|--------|------------|
| $\mathbf{x}$ | Input vector in the original input space. |
| $D$ | Number of input dimensions. |
| $\boldsymbol{\phi}(\mathbf{x})$ | Feature-space mapping of input $\mathbf{x}$. |
| $M$ | Number of explicit basis functions or feature dimensions, when finite. |
| $k(\mathbf{x},\mathbf{x}')$ | Kernel function, usually $k(\mathbf{x},\mathbf{x}')=\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}')$. |
| $N$ | Number of training examples. |
| $\mathbf{t}$ | Target vector $(t_1,\ldots,t_N)^T$. |
| $\mathbf{K}$ | Gram matrix with entries $K_{nm}=k(\mathbf{x}_n,\mathbf{x}_m)$. |
| $\mathbf{k}(\mathbf{x})$ | Vector of kernel values between a test point and training points, with $k_n(\mathbf{x})=k(\mathbf{x}_n,\mathbf{x})$. |
| $\lambda$ | Regularization coefficient in kernel ridge regression / regularized least squares. |

### Dual-representation Notation

| Symbol | Definition |
|--------|------------|
| $\mathbf{w}$ | Weight vector in the primal feature-space model. |
| $\mathbf{a}$ | Dual coefficient vector, one coefficient per training example. |
| $\boldsymbol{\Phi}$ | Design matrix; row $n$ is $\boldsymbol{\phi}(\mathbf{x}_n)^T$. |
| $\mathbf{I}_N$ | $N\times N$ identity matrix. |
| $y(\mathbf{x})$ | Prediction at input $\mathbf{x}$. |

### Radial Basis Function Network Notation

| Symbol | Definition |
|--------|------------|
| $\phi_j(\mathbf{x})$ | Radial basis function centered at some point $\boldsymbol{\mu}_j$. |
| $h(\cdot)$ | A radial kernel profile, often Gaussian. |
| $\boldsymbol{\mu}_j$ | Center of basis function $j$. |
| $s$ or $\sigma$ | Width / scale parameter of an RBF. |
| $y(\mathbf{x})=\sum_j w_j\phi_j(\mathbf{x})$ | RBF network output. |

### Gaussian Process Notation

| Symbol | Definition |
|--------|------------|
| $y(\mathbf{x})$ | Latent function value at input $\mathbf{x}$. |
| $\mathbf{y}$ | Vector of latent function values at the training inputs. |
| $t_n$ | Noisy observed target value for training input $\mathbf{x}_n$. |
| $\beta$ | Noise precision; noise variance is $\beta^{-1}$. |
| $\mathbf{C}$ | Covariance matrix of observed targets, usually $\mathbf{C}=\mathbf{K}+\beta^{-1}\mathbf{I}_N$. |
| $c$ | Prior variance at a test point, often $c=k(\mathbf{x}_*,\mathbf{x}_*)+\beta^{-1}$. |
| $m(\mathbf{x}_*)$ | Predictive mean at a test point. |
| $\sigma^2(\mathbf{x}_*)$ | Predictive variance at a test point. |
| $\boldsymbol{\theta}$ | Kernel hyperparameters. |
| $\eta_i$ | ARD precision parameter for input dimension $i$. |

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch.6 opening; §6.1-§6.4

## 0.1 What This Chapter Is Really About

In Chapters 3 and 4, we often wrote a model as

$$
y(\mathbf{x},\mathbf{w})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}).
$$

This is a linear model in the feature vector $\boldsymbol{\phi}(\mathbf{x})$. The feature vector itself may be nonlinear in $\mathbf{x}$, but once $\boldsymbol{\phi}(\mathbf{x})$ is fixed, the model is linear in $\mathbf{w}$.

The limitation is that explicitly constructing $\boldsymbol{\phi}(\mathbf{x})$ can be difficult. Sometimes the feature vector is very high-dimensional. Sometimes it is infinite-dimensional. Sometimes the input is not even a simple vector, for example a string, a graph, a set, or a structured object.

Kernel methods give a clever shortcut. Many algorithms only need feature vectors through inner products:

$$
\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}').
$$

If we can compute this inner product directly using a kernel function

$$
k(\mathbf{x},\mathbf{x}')=\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}'),
$$

then we can work in the feature space without explicitly writing down the feature vector.

This is the **kernel trick**:

$$
\boxed{
\text{Use } k(\mathbf{x},\mathbf{x}') \text{ wherever an algorithm needs }
\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}').
}
$$

## 0.2 Memory-based Learning and Kernel Learning

The chapter begins by comparing two learning styles.

| Learning Style | What Is Stored After Training? | Prediction Uses |
|---------------|---------------------------------|-----------------|
| Parametric model | A parameter vector $\mathbf{w}$ | $\mathbf{w}$ and the new input $\mathbf{x}$ |
| Memory-based model | Training examples, or a subset of them | Similarities between $\mathbf{x}$ and stored examples |
| Kernel model | Usually training examples plus kernel coefficients | Kernel values $k(\mathbf{x}_n,\mathbf{x})$ |

Nearest-neighbour methods and Parzen density estimators are memory-based because the training examples remain part of prediction. Kernel methods are often similar: prediction at a new point depends on kernel similarities between the new point and training points.

This is not a weakness by itself. It is a different computational trade-off:

$$
\text{fast training, slower prediction}
\quad\text{versus}\quad
\text{slower training, faster prediction}.
$$

Kernel methods often require solving an $N\times N$ system involving the Gram matrix. So they are very powerful for moderate data sizes, but can become expensive when $N$ is very large.

## 0.3 Roadmap of the Chapter

Chapter 6 can be understood as answering four linked questions.

| Section | Core Question | Main Idea |
|---------|---------------|-----------|
| §6.1 Dual representations | How can a linear model be rewritten using training points instead of explicit weights? | Regularized least squares can be expressed using the Gram matrix $\mathbf{K}$. |
| §6.2 Constructing kernels | Which functions are valid kernels? | Kernels must correspond to inner products in a feature space; valid kernels can be combined to build new kernels. |
| §6.3 RBF networks | How do radial basis functions connect local similarity to regression? | Predictions become weighted averages of local basis responses. |
| §6.4 Gaussian processes | How can kernels define distributions over functions? | A kernel is a covariance function; it gives a Bayesian nonparametric regression and classification framework. |

The overall flow is:

$$
\text{feature-space inner products}
\longrightarrow
\text{kernel functions}
\longrightarrow
\text{dual prediction}
\longrightarrow
\text{Gaussian processes}.
$$

---

# §1 Dual Representations

> 📖 Textbook §6.1 Dual Representations

## 1.1 From Primal Weights to Dual Coefficients

Start from the regularized least-squares objective

$$
J(\mathbf{w})=
\frac{1}{2}\sum_{n=1}^{N}
\{\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)-t_n\}^2
+
\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}.
$$

The first term measures data fit. The second term is weight decay. The coefficient $\lambda\geq 0$ controls how strongly we penalize large weights.

The usual **primal** view is to solve for $\mathbf{w}$. But we can also show that the solution must lie in the span of the training feature vectors.

To see this, take the gradient with respect to $\mathbf{w}$.

First define the prediction error for example $n$:

$$
e_n=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)-t_n.
$$

Then the objective is

$$
J(\mathbf{w})=\frac{1}{2}\sum_{n=1}^{N}e_n^2+\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}.
$$

For one data term,

$$
\frac{\partial}{\partial \mathbf{w}}\frac{1}{2}e_n^2
=e_n\frac{\partial e_n}{\partial \mathbf{w}}.
$$

Since

$$
e_n=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)-t_n,
$$

we have

$$
\frac{\partial e_n}{\partial \mathbf{w}}=\boldsymbol{\phi}(\mathbf{x}_n).
$$

Therefore

$$
\nabla_{\mathbf{w}}J
=
\sum_{n=1}^{N}
\{\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)-t_n\}
\boldsymbol{\phi}(\mathbf{x}_n)
+
\lambda \mathbf{w}.
$$

At the optimum, the gradient is zero:

$$
\sum_{n=1}^{N}
\{\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)-t_n\}
\boldsymbol{\phi}(\mathbf{x}_n)
+
\lambda \mathbf{w}=0.
$$

Move the first term to the other side:

$$
\lambda \mathbf{w}
=
-
\sum_{n=1}^{N}
\{\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)-t_n\}
\boldsymbol{\phi}(\mathbf{x}_n).
$$

Divide by $\lambda$:

$$
\mathbf{w}
=
-\frac{1}{\lambda}
\sum_{n=1}^{N}
\{\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)-t_n\}
\boldsymbol{\phi}(\mathbf{x}_n).
$$

This has the form

$$
\mathbf{w}=\sum_{n=1}^{N}a_n\boldsymbol{\phi}(\mathbf{x}_n),
$$

where

$$
a_n=-\frac{1}{\lambda}
\{\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)-t_n\}.
$$

This is the key step. The optimal weight vector is a linear combination of training feature vectors. So instead of learning one coefficient per feature dimension, we can learn one coefficient per training example.

## 1.2 The Gram Matrix

Let $\boldsymbol{\Phi}$ be the design matrix whose $n$th row is $\boldsymbol{\phi}(\mathbf{x}_n)^T$. Then the previous result can be written compactly as

$$
\mathbf{w}=\boldsymbol{\Phi}^T\mathbf{a}.
$$

Now define the Gram matrix

$$
\mathbf{K}=\boldsymbol{\Phi}\boldsymbol{\Phi}^T.
$$

Its entries are

$$
K_{nm}
=
\boldsymbol{\phi}(\mathbf{x}_n)^T\boldsymbol{\phi}(\mathbf{x}_m)
=
k(\mathbf{x}_n,\mathbf{x}_m).
$$

So $\mathbf{K}$ is an $N\times N$ matrix of all pairwise training similarities.

It is useful to read a row of $\mathbf{K}$ as follows:

$$
\text{row } n
=
\text{similarity of } \mathbf{x}_n \text{ to every training point}.
$$

## 1.3 Solving Ridge Regression in the Dual

In the dual representation, the solution for $\mathbf{a}$ is

$$
\boxed{
\mathbf{a}=(\mathbf{K}+\lambda\mathbf{I}_N)^{-1}\mathbf{t}.
}
$$

Then prediction at a new input $\mathbf{x}$ is

$$
y(\mathbf{x})
=
\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}).
$$

Substitute $\mathbf{w}=\boldsymbol{\Phi}^T\mathbf{a}$:

$$
y(\mathbf{x})
=
(\boldsymbol{\Phi}^T\mathbf{a})^T\boldsymbol{\phi}(\mathbf{x}).
$$

Transpose the product:

$$
(\boldsymbol{\Phi}^T\mathbf{a})^T
=
\mathbf{a}^T\boldsymbol{\Phi}.
$$

Therefore

$$
y(\mathbf{x})
=
\mathbf{a}^T\boldsymbol{\Phi}\boldsymbol{\phi}(\mathbf{x}).
$$

The vector $\boldsymbol{\Phi}\boldsymbol{\phi}(\mathbf{x})$ contains inner products between every training feature vector and the test feature vector:

$$
\boldsymbol{\Phi}\boldsymbol{\phi}(\mathbf{x})
=
\begin{pmatrix}
\boldsymbol{\phi}(\mathbf{x}_1)^T\boldsymbol{\phi}(\mathbf{x})\\
\vdots\\
\boldsymbol{\phi}(\mathbf{x}_N)^T\boldsymbol{\phi}(\mathbf{x})
\end{pmatrix}
=
\begin{pmatrix}
k(\mathbf{x}_1,\mathbf{x})\\
\vdots\\
k(\mathbf{x}_N,\mathbf{x})
\end{pmatrix}
=
\mathbf{k}(\mathbf{x}).
$$

So

$$
y(\mathbf{x})=\mathbf{a}^T\mathbf{k}(\mathbf{x}).
$$

Equivalently,

$$
\boxed{
 y(\mathbf{x})
 =
 \mathbf{k}(\mathbf{x})^T
 (\mathbf{K}+\lambda\mathbf{I}_N)^{-1}
 \mathbf{t}.
}
$$

This formula is the first major result of the chapter. It says that prediction is a weighted combination of training targets, and the weights depend on kernel similarities between the new input and the training inputs.

## 1.4 Why the Dual Form Matters

At first, the dual representation may look worse than the primal representation. The primal solution often involves an $M\times M$ matrix, where $M$ is the number of basis functions. The dual solution involves an $N\times N$ matrix, where $N$ is the number of training examples.

If $N$ is much larger than $M$, the dual may be computationally more expensive.

But the dual representation has one huge advantage:

> It only uses $\boldsymbol{\phi}(\mathbf{x})$ through inner products.

This means we can use a kernel function directly and avoid explicitly constructing $\boldsymbol{\phi}(\mathbf{x})$.

This is especially useful when:

| Situation | Why Kernels Help |
|----------|------------------|
| Very high-dimensional features | We avoid explicitly storing all features. |
| Infinite-dimensional features | We can still compute the kernel value. |
| Structured inputs | Kernels can be built for strings, graphs, sets, and distributions. |
| Nonlinear decision boundaries | A linear method in feature space becomes nonlinear in input space. |

## 1.5 A Small Dimension Check

Suppose there are $N$ training examples and $M$ feature dimensions.

| Quantity | Shape |
|----------|-------|
| $\boldsymbol{\Phi}$ | $N\times M$ |
| $\boldsymbol{\Phi}^T$ | $M\times N$ |
| $\mathbf{w}$ | $M\times 1$ |
| $\mathbf{a}$ | $N\times 1$ |
| $\mathbf{K}=\boldsymbol{\Phi}\boldsymbol{\Phi}^T$ | $N\times N$ |
| $\mathbf{k}(\mathbf{x})$ | $N\times 1$ |
| $\mathbf{t}$ | $N\times 1$ |

Check the prediction formula:

$$
\mathbf{k}(\mathbf{x})^T
(\mathbf{K}+\lambda\mathbf{I}_N)^{-1}\mathbf{t}.
$$

The dimensions are

$$
(1\times N)(N\times N)(N\times 1)=1\times 1.
$$

So the prediction is a scalar, as expected.

---

# §2 Constructing Kernels

> 📖 Textbook §6.2 Constructing Kernels

## 2.1 From Basis Functions to Kernels

A kernel can be built from explicit basis functions by

$$
k(x,x')=\boldsymbol{\phi}(x)^T\boldsymbol{\phi}(x')
=
\sum_{i=1}^{M}\phi_i(x)\phi_i(x').
$$

For one-dimensional inputs, each $\phi_i(x)$ is a curve. The kernel $k(x,x')$ compares the two inputs by comparing their feature values.

> ![Figure 6.1](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_1__textbook_fig_6_1_p295_basis_functions_and_kernel_construction.png)
>
> *Figure 6.1 (Textbook Fig. 6.1, p. 295): Kernels can be constructed from basis functions. The top row shows three different families of basis functions. The bottom row shows the resulting kernel as a function of $x$ for a fixed $x'=0$. The important lesson is that different feature families induce different similarity functions.*

This figure is useful because it makes a kernel less mysterious. A kernel is not just a black-box similarity function. It can often be understood as an inner product after transforming the input into a new set of coordinates.

## 2.2 Polynomial Kernel: A Fully Worked Feature Map

Consider the kernel

$$
k(\mathbf{x},\mathbf{z})=(\mathbf{x}^T\mathbf{z})^2.
$$

For a two-dimensional input,

$$
\mathbf{x}=(x_1,x_2)^T,
\qquad
\mathbf{z}=(z_1,z_2)^T.
$$

First compute the ordinary inner product:

$$
\mathbf{x}^T\mathbf{z}=x_1z_1+x_2z_2.
$$

Now square it:

$$
(\mathbf{x}^T\mathbf{z})^2
=(x_1z_1+x_2z_2)^2.
$$

Use the identity $(a+b)^2=a^2+2ab+b^2$ with

$$
a=x_1z_1,
\qquad
b=x_2z_2.
$$

Then

$$
(x_1z_1+x_2z_2)^2
=x_1^2z_1^2+2x_1z_1x_2z_2+x_2^2z_2^2.
$$

Rearrange the middle term:

$$
2x_1z_1x_2z_2=2x_1x_2z_1z_2.
$$

Now write this as an inner product:

$$
x_1^2z_1^2+2x_1x_2z_1z_2+x_2^2z_2^2
=
(x_1^2,\sqrt{2}x_1x_2,x_2^2)
(z_1^2,\sqrt{2}z_1z_2,z_2^2)^T.
$$

So the feature map is

$$
\boxed{
\boldsymbol{\phi}(\mathbf{x})
=
(x_1^2,\sqrt{2}x_1x_2,x_2^2)^T.
}
$$

Therefore

$$
k(\mathbf{x},\mathbf{z})
=
\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{z}).
$$

This example is important because it shows exactly what the kernel trick saves us from doing. Instead of explicitly creating all quadratic features, we can compute one kernel value directly.

## 2.3 Valid Kernels and Positive Semidefinite Gram Matrices

Not every similarity function is a valid kernel. A valid kernel must correspond to an inner product in some feature space.

A practical condition is:

$$
\boxed{
\mathbf{K}\text{ must be positive semidefinite for every possible data set.}
}
$$

Positive semidefinite means that for every vector $\mathbf{c}$,

$$
\mathbf{c}^T\mathbf{K}\mathbf{c}\geq 0.
$$

This is not the same as saying that all entries of $\mathbf{K}$ are nonnegative. A matrix can contain negative entries and still be positive semidefinite.

Why does this condition make sense? If $\mathbf{K}=\boldsymbol{\Phi}\boldsymbol{\Phi}^T$, then

$$
\mathbf{c}^T\mathbf{K}\mathbf{c}
=
\mathbf{c}^T\boldsymbol{\Phi}\boldsymbol{\Phi}^T\mathbf{c}
=
(\boldsymbol{\Phi}^T\mathbf{c})^T(\boldsymbol{\Phi}^T\mathbf{c})
=
\|\boldsymbol{\Phi}^T\mathbf{c}\|^2
\geq 0.
$$

So any Gram matrix built from feature-space inner products must be positive semidefinite.

## 2.4 Kernel Construction Rules

One powerful way to build kernels is to combine simple valid kernels into more complex valid kernels.

Suppose $k_1(\mathbf{x},\mathbf{x}')$ and $k_2(\mathbf{x},\mathbf{x}')$ are valid kernels. Then many operations preserve validity.

| Construction | New Kernel |
|-------------|------------|
| Positive scaling | $k(\mathbf{x},\mathbf{x}')=c\,k_1(\mathbf{x},\mathbf{x}')$, where $c>0$ |
| Sum | $k(\mathbf{x},\mathbf{x}')=k_1(\mathbf{x},\mathbf{x}')+k_2(\mathbf{x},\mathbf{x}')$ |
| Product | $k(\mathbf{x},\mathbf{x}')=k_1(\mathbf{x},\mathbf{x}')k_2(\mathbf{x},\mathbf{x}')$ |
| Multiplication by functions | $k(\mathbf{x},\mathbf{x}')=f(\mathbf{x})k_1(\mathbf{x},\mathbf{x}')f(\mathbf{x}')$ |
| Polynomial of a kernel | $k(\mathbf{x},\mathbf{x}')=q(k_1(\mathbf{x},\mathbf{x}'))$ if $q$ has nonnegative coefficients |
| Exponential | $k(\mathbf{x},\mathbf{x}')=\exp(k_1(\mathbf{x},\mathbf{x}'))$ |

The key practical message is:

$$
\text{We can design kernels modularly.}
$$

For example, if one kernel captures shape and another captures colour, their sum or product can create a combined similarity measure.

## 2.5 Common Kernel Families

The simplest kernel is the linear kernel:

$$
k(\mathbf{x},\mathbf{x}')=\mathbf{x}^T\mathbf{x}'.
$$

A polynomial kernel has the form

$$
k(\mathbf{x},\mathbf{x}')=(\mathbf{x}^T\mathbf{x}'+c)^p,
$$

where $p$ is the polynomial degree and $c$ is a constant.

A Gaussian or radial basis function kernel has the form

$$
k(\mathbf{x},\mathbf{x}')
=
\exp\left(-\frac{\|\mathbf{x}-\mathbf{x}'\|^2}{2\sigma^2}\right).
$$

This kernel is large when $\mathbf{x}$ and $\mathbf{x}'$ are close, and small when they are far apart. The parameter $\sigma$ controls the length scale.

An exponential kernel has the form

$$
k(\mathbf{x},\mathbf{x}')
=
\exp\left(-\frac{\|\mathbf{x}-\mathbf{x}'\|}{\sigma}\right).
$$

Compared with the Gaussian kernel, it usually produces rougher sample functions in a Gaussian process.

## 2.6 Stationary and Radial Kernels

Some kernels depend only on the difference between inputs:

$$
k(\mathbf{x},\mathbf{x}')=k(\mathbf{x}-\mathbf{x}').
$$

These are called **stationary kernels**. They are translation-invariant: if we shift both inputs by the same amount, the kernel value does not change.

A more special case depends only on distance:

$$
k(\mathbf{x},\mathbf{x}')=g(\|\mathbf{x}-\mathbf{x}'\|).
$$

These are radial kernels. The Gaussian RBF kernel is the most famous example.

The intuition is simple:

$$
\text{nearby points should have similar function values.}
$$

This intuition will become central in Gaussian processes.

## 2.7 Kernels Beyond Vector Inputs

Kernels are not limited to ordinary vectors. If we can define a valid positive semidefinite similarity, we can build kernel methods for more structured objects.

Examples include:

| Input Type | Kernel Idea |
|-----------|-------------|
| Strings | Count common substrings or subsequences. |
| Sets | Compare all pairs of elements between sets. |
| Graphs | Count common walks, paths, or graph substructures. |
| Probability models | Use gradients of log likelihood, leading to Fisher kernels. |

This is one reason kernel methods became influential. They make it possible to reuse linear algorithms in complex data domains.

## 2.8 The Sigmoidal Kernel and Neural-network Connection

The textbook also discusses the sigmoidal kernel, which is related to neural networks. A typical form is

$$
k(\mathbf{x},\mathbf{x}')=	anh(a\mathbf{x}^T\mathbf{x}'+b).
$$

This resembles the activation function used in a neural network. However, not every setting of $a$ and $b$ gives a valid positive semidefinite kernel. So unlike the Gaussian kernel, the sigmoidal kernel must be used with care.

The conceptual connection is still useful:

$$
\text{a neural network learns hidden features explicitly,}
$$

whereas

$$
\text{a kernel method uses an implicit feature space through } k(\mathbf{x},\mathbf{x}').
$$

---

# §3 Radial Basis Function Networks

> 📖 Textbook §6.3 Radial Basis Function Networks; §6.3.1

## 3.1 What Is an RBF Network?

A radial basis function network uses basis functions that depend on distance from a centre. A common form is

$$
\phi_j(\mathbf{x})=
\exp\left(-\frac{\|\mathbf{x}-\boldsymbol{\mu}_j\|^2}{2s^2}\right),
$$

where $\boldsymbol{\mu}_j$ is the centre and $s$ is the width.

The model output is a linear combination of these basis functions:

$$
y(\mathbf{x})=
\sum_{j=1}^{M}w_j\phi_j(\mathbf{x}).
$$

The basis functions are local. A basis function is large when $\mathbf{x}$ is close to its centre and small when $\mathbf{x}$ is far away.

This makes RBF networks easy to interpret:

$$
\text{prediction}=	ext{weighted combination of local responses}.
$$

## 3.2 RBF Networks and Exact Interpolation

Historically, RBF methods were motivated by interpolation. Suppose each training point becomes a basis centre:

$$
\boldsymbol{\mu}_n=\mathbf{x}_n.
$$

If there are $N$ training points, we can choose $N$ basis functions. In the noiseless case, we may try to choose weights so that

$$
y(\mathbf{x}_n)=t_n,
\qquad n=1,\ldots,N.
$$

This means the fitted function passes exactly through all training points.

But in machine learning, exact interpolation is often risky because targets may contain noise. A smoother model is usually preferred. This is why regularization and probabilistic interpretations are important.

## 3.3 Normalized Basis Functions

RBF basis functions can also be normalized so that their values sum to one:

$$
\widetilde{\phi}_j(\mathbf{x})
=
\frac{\phi_j(\mathbf{x})}{\sum_{m=1}^{M}\phi_m(\mathbf{x})}.
$$

Then

$$
\sum_{j=1}^{M}\widetilde{\phi}_j(\mathbf{x})=1.
$$

This changes the interpretation. The normalized basis values behave like local weights that form a convex combination.

> ![Figure 6.2](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_2__textbook_fig_6_2_p301_gaussian_basis_functions_normalized.png)
>
> *Figure 6.2 (Textbook Fig. 6.2, p. 301): The left panel shows Gaussian basis functions. The right panel shows their normalized versions. Normalization makes the basis functions compete locally and makes their sum equal to one.*

The normalization is useful when we want the model output to behave like a local average. It avoids regions where all basis functions are small and the model output becomes poorly controlled.

## 3.4 Choosing RBF Centres

There are several simple strategies for choosing centres.

| Strategy | Description | Advantage | Disadvantage |
|----------|-------------|-----------|--------------|
| Use all training points | Set one centre at each training input. | Simple and flexible. | Can be expensive and may over-fit. |
| Random subset | Use a subset of training points as centres. | Cheaper than using all points. | May miss important regions. |
| Clustering | Use cluster centres, such as from K-means. | Represents data distribution compactly. | Requires an extra clustering step. |
| Orthogonal least squares | Select centres sequentially by reducing error. | More targeted selection. | More complex algorithm. |

The larger idea is that RBF networks sit between fully parametric models and memory-based methods. They store a set of centres, and predictions depend on distances to those centres.

## 3.5 The Nadaraya-Watson Model

The textbook derives the Nadaraya-Watson model from kernel density estimation.

Suppose we model the joint density of input $\mathbf{x}$ and target $t$ using one kernel component centred at every training pair $(\mathbf{x}_n,t_n)$:

$$
p(\mathbf{x},t)
=
\frac{1}{N}
\sum_{n=1}^{N}
f(\mathbf{x}-\mathbf{x}_n,t-t_n).
$$

For regression, we want the conditional mean

$$
\mathbb{E}[t\mid \mathbf{x}].
$$

The resulting model has the form

$$
\boxed{
 y(\mathbf{x})
 =
 \sum_{n=1}^{N}w_n(\mathbf{x})t_n
}
$$

where the weights satisfy

$$
w_n(\mathbf{x})\geq 0,
\qquad
\sum_{n=1}^{N}w_n(\mathbf{x})=1.
$$

For a Gaussian kernel in the input space, the weights take the normalized form

$$
w_n(\mathbf{x})
=
\frac{
h(\mathbf{x}-\mathbf{x}_n)
}{
\sum_{m=1}^{N}h(\mathbf{x}-\mathbf{x}_m)
}.
$$

So the prediction is a weighted average of nearby target values. Nearby examples get large weights; far examples get small weights.

> ![Figure 6.3](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_3__textbook_fig_6_3_p303_nadaraya_watson_kernel_regression.png)
>
> *Figure 6.3 (Textbook Fig. 6.3, p. 303): Nadaraya-Watson kernel regression on the sinusoidal data set. The red curve is the conditional mean, and the shaded region shows uncertainty. Each blue ellipse represents a local Gaussian component centred on a data point.*

## 3.6 Reading Figure 6.3 Slowly

Figure 6.3 is worth discussing carefully in class.

First, each training point contributes a local density bump. These bumps are drawn as blue ellipses. They are not circular because the scales of the horizontal and vertical axes differ.

Second, the red curve is not obtained by fitting global polynomial coefficients. It is obtained by local averaging. At each input $x$, the prediction is mainly influenced by nearby data points.

Third, the shaded region is wider in places where the local conditional distribution is more uncertain. This reminds us that regression can be understood probabilistically, not just as drawing a single curve.

The practical lesson is:

$$
\text{kernel regression predicts by local similarity-weighted averaging.}
$$

---

# §4 Gaussian Processes

> 📖 Textbook §6.4 Gaussian Processes; §6.4.1-§6.4.7

## 4.1 From Kernels to Distributions Over Functions

A Gaussian process is a probability distribution over functions.

This sounds abstract, so start with a simpler idea. A multivariate Gaussian distribution is a distribution over vectors:

$$
\mathbf{y}=(y_1,\ldots,y_N)^T.
$$

A Gaussian process says that for any finite set of inputs

$$
\mathbf{x}_1,\ldots,\mathbf{x}_N,
$$

the corresponding function values

$$
y(\mathbf{x}_1),\ldots,y(\mathbf{x}_N)
$$

have a joint Gaussian distribution.

A Gaussian process is written as

$$
y(\mathbf{x})\sim \mathcal{GP}(m(\mathbf{x}),k(\mathbf{x},\mathbf{x}')),
$$

where $m(\mathbf{x})$ is the mean function and $k(\mathbf{x},\mathbf{x}')$ is the covariance function.

In this chapter, the mean is often taken to be zero:

$$
m(\mathbf{x})=0.
$$

Then the kernel fully determines the prior distribution over functions.

## 4.2 Linear Regression Revisited as a Gaussian Process

Consider the linear basis-function model

$$
y(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}).
$$

Put a Gaussian prior on the weights:

$$
p(\mathbf{w})=\mathcal{N}(\mathbf{w}\mid \mathbf{0},\alpha^{-1}\mathbf{I}).
$$

Now take two inputs $\mathbf{x}$ and $\mathbf{x}'$. Their function values are

$$
y(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}),
\qquad
 y(\mathbf{x}')=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}').
$$

The covariance is

$$
\operatorname{cov}[y(\mathbf{x}),y(\mathbf{x}')]
=
\mathbb{E}[y(\mathbf{x})y(\mathbf{x}')]
-
\mathbb{E}[y(\mathbf{x})]\mathbb{E}[y(\mathbf{x}')].
$$

Because the prior mean of $\mathbf{w}$ is zero,

$$
\mathbb{E}[y(\mathbf{x})]=0.
$$

So

$$
\operatorname{cov}[y(\mathbf{x}),y(\mathbf{x}')]
=
\mathbb{E}
[(\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}))
(\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}'))].
$$

This becomes

$$
\operatorname{cov}[y(\mathbf{x}),y(\mathbf{x}')]
=
\alpha^{-1}\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}').
$$

Therefore the kernel is

$$
k(\mathbf{x},\mathbf{x}')
=
\alpha^{-1}\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}').
$$

This is a central bridge:

$$
\boxed{
\text{A Gaussian prior over weights induces a Gaussian process prior over functions.}
}
$$

## 4.3 Sample Functions and Kernel Smoothness

Different kernels produce different kinds of random functions.

> ![Figure 6.4](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_4__textbook_fig_6_4_p306_gp_samples_gaussian_vs_exponential_kernel.png)
>
> *Figure 6.4 (Textbook Fig. 6.4, p. 306): Samples from Gaussian process priors using two kernels. The Gaussian kernel produces smoother functions, while the exponential kernel produces rougher functions.*

The kernel controls how strongly function values at nearby inputs are correlated.

If the kernel changes slowly with distance, nearby points are highly correlated over longer ranges, so sampled functions are smooth.

If the kernel decays quickly or is less smooth at the origin, nearby function values can change more abruptly, so sampled functions look rougher.

## 4.4 A Flexible Gaussian Process Kernel

The textbook uses a covariance function with several hyperparameters to show how kernel parameters affect sampled functions.

A common form combines an exponential-quadratic term, a constant term, and a linear term:

$$
k(\mathbf{x}_n,\mathbf{x}_m)
=
\theta_0
\exp\left\{-\frac{\theta_1}{2}\|\mathbf{x}_n-\mathbf{x}_m\|^2\right\}
+
\theta_2
+
\theta_3\mathbf{x}_n^T\mathbf{x}_m.
$$

Each parameter has an intuitive role.

| Parameter | Rough Meaning |
|----------|---------------|
| $\theta_0$ | Overall amplitude of the smooth nonlinear component. |
| $\theta_1$ | Controls length scale; larger values usually mean faster variation. |
| $\theta_2$ | Adds a constant covariance component. |
| $\theta_3$ | Adds a linear covariance component. |

> ![Figure 6.5](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_5__textbook_fig_6_5_p308_gp_prior_samples_covariance_parameters.png)
>
> *Figure 6.5 (Textbook Fig. 6.5, p. 308): Samples from Gaussian process priors under different covariance hyperparameter settings. The titles above the plots show $(\theta_0,\theta_1,\theta_2,\theta_3)$. Changing the kernel changes the kinds of functions that are likely before seeing data.*

Figure 6.5 is useful pedagogically because it shows that Gaussian processes are not one fixed model. The kernel is the modelling choice.

## 4.5 Gaussian Process Regression: The Data Model

For regression, we assume the observed target is a noisy version of the latent function value:

$$
t_n=y_n+\epsilon_n,
$$

where

$$
y_n=y(\mathbf{x}_n),
\qquad
\epsilon_n\sim \mathcal{N}(0,\beta^{-1}).
$$

This means

$$
p(t_n\mid y_n)=\mathcal{N}(t_n\mid y_n,\beta^{-1}).
$$

For all training examples, write

$$
\mathbf{t}=(t_1,\ldots,t_N)^T,
\qquad
\mathbf{y}=(y_1,\ldots,y_N)^T.
$$

The GP prior gives

$$
p(\mathbf{y})=\mathcal{N}(\mathbf{y}\mid \mathbf{0},\mathbf{K}),
$$

where

$$
K_{nm}=k(\mathbf{x}_n,\mathbf{x}_m).
$$

Because independent Gaussian noise is added to each target, the covariance of the observed targets is

$$
\boxed{
\mathbf{C}=\mathbf{K}+\beta^{-1}\mathbf{I}_N.
}
$$

Thus the marginal distribution of the observed target vector is

$$
\boxed{
 p(\mathbf{t})=\mathcal{N}(\mathbf{t}\mid \mathbf{0},\mathbf{C}).
}
$$

> ![Figure 6.6](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_6__textbook_fig_6_6_p309_sampling_data_from_gp.png)
>
> *Figure 6.6 (Textbook Fig. 6.6, p. 309): A latent function is sampled from a Gaussian process prior. Values of the latent function are evaluated at selected inputs, and then Gaussian noise is added to produce observed targets.*

This figure separates two ideas:

1. The smooth blue curve is the latent function $y(x)$.
2. The noisy observations $t_n$ are generated by adding noise to $y(x_n)$.

The model does not assume that the observed data points lie exactly on the true function.

## 4.6 Gaussian Process Regression: Prediction

Now suppose we want to predict at a new input $\mathbf{x}_*$.

Let the observed training targets be $\mathbf{t}$, and let the new target be $t_*$. The joint distribution of $\mathbf{t}$ and $t_*$ is Gaussian:

$$
\begin{pmatrix}
\mathbf{t}\\
t_*
\end{pmatrix}
\sim
\mathcal{N}
\left(
\begin{pmatrix}
\mathbf{0}\\
0
\end{pmatrix},
\begin{pmatrix}
\mathbf{C} & \mathbf{k}\\
\mathbf{k}^T & c
\end{pmatrix}
\right).
$$

Here

$$
\mathbf{k}
=
\begin{pmatrix}
k(\mathbf{x}_1,\mathbf{x}_*)\\
\vdots\\
k(\mathbf{x}_N,\mathbf{x}_*)
\end{pmatrix},
$$

and $c$ is the variance at the test point. If we are predicting the noisy target $t_*$, then

$$
c=k(\mathbf{x}_*,\mathbf{x}_*)+\beta^{-1}.
$$

If we are predicting the latent function value $y_*$, then

$$
c=k(\mathbf{x}_*,\mathbf{x}_*).
$$

Using the conditional formula for a partitioned Gaussian, we get

$$
\boxed{
 p(t_*\mid \mathbf{t})
 =
 \mathcal{N}(t_*\mid m(\mathbf{x}_*),\sigma^2(\mathbf{x}_*)).
}
$$

The predictive mean is

$$
\boxed{
 m(\mathbf{x}_*)=\mathbf{k}^T\mathbf{C}^{-1}\mathbf{t}.
}
$$

The predictive variance is

$$
\boxed{
 \sigma^2(\mathbf{x}_*)=c-\mathbf{k}^T\mathbf{C}^{-1}\mathbf{k}.
}
$$

These two equations are among the most important equations in the chapter.

## 4.7 Interpreting the Predictive Mean

The predictive mean is

$$
m(\mathbf{x}_*)=\mathbf{k}^T\mathbf{C}^{-1}\mathbf{t}.
$$

Let

$$
\mathbf{a}=\mathbf{C}^{-1}\mathbf{t}.
$$

Then

$$
m(\mathbf{x}_*)=\mathbf{k}^T\mathbf{a}
=
\sum_{n=1}^{N}a_n k(\mathbf{x}_n,\mathbf{x}_*).
$$

This looks similar to the dual kernel regression formula from §1. The prediction is a kernel expansion centred on the training points.

The difference is probabilistic. Gaussian processes also give a predictive variance.

## 4.8 Interpreting the Predictive Variance

The predictive variance is

$$
\sigma^2(\mathbf{x}_*)=c-\mathbf{k}^T\mathbf{C}^{-1}\mathbf{k}.
$$

The first term $c$ is the prior uncertainty at the test point.

The second term

$$
\mathbf{k}^T\mathbf{C}^{-1}\mathbf{k}
$$

is the reduction in uncertainty caused by observing the training data.

If $\mathbf{x}_*$ is close to many training inputs, then $\mathbf{k}$ has large entries, and the uncertainty reduction is large.

If $\mathbf{x}_*$ is far from training inputs, then $\mathbf{k}$ has small entries, and the model remains uncertain.

> ![Figure 6.7](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_7__textbook_fig_6_7_p310_gp_regression_one_training_one_test_point.png)
>
> *Figure 6.7 (Textbook Fig. 6.7, p. 310): Gaussian process regression with one training point and one test point. Conditioning on the observed training value changes the distribution over the test value. The red ellipses show the joint Gaussian distribution, and the green curve shows the conditional distribution.*

Figure 6.7 is the simplest picture of Gaussian process regression. GP prediction is just conditioning a joint Gaussian distribution.

## 4.9 Gaussian Process Regression on the Sinusoidal Data

> ![Figure 6.8](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_8__textbook_fig_6_8_p310_gp_regression_sinusoidal_prediction.png)
>
> *Figure 6.8 (Textbook Fig. 6.8, p. 310): Gaussian process regression on the sinusoidal data set. The red curve is the predictive mean, and the shaded region shows plus and minus one standard deviation. Uncertainty grows in regions where training data are absent.*

The right side of Figure 6.8 has fewer training points. The shaded region becomes wider there. This is the main advantage of a Bayesian predictive model:

$$
\text{it tells us not only what it predicts, but also how uncertain it is.}
$$

A non-Bayesian regression curve may give a single answer everywhere. A Gaussian process gives a full predictive distribution.

## 4.10 Learning Kernel Hyperparameters

The kernel contains hyperparameters such as length scale, amplitude, and noise precision. We can learn them by maximizing the marginal likelihood.

Because

$$
p(\mathbf{t}\mid\boldsymbol{\theta})
=
\mathcal{N}(\mathbf{t}\mid\mathbf{0},\mathbf{C}),
$$

the log marginal likelihood is

$$
\ln p(\mathbf{t}\mid\boldsymbol{\theta})
=
-\frac{1}{2}\ln |\mathbf{C}|
-\frac{1}{2}\mathbf{t}^T\mathbf{C}^{-1}\mathbf{t}
-\frac{N}{2}\ln(2\pi).
$$

This expression has three parts.

| Term | Interpretation |
|------|----------------|
| $-\frac{1}{2}\mathbf{t}^T\mathbf{C}^{-1}\mathbf{t}$ | Data-fit term. It rewards explaining the targets well. |
| $-\frac{1}{2}\ln|\mathbf{C}|$ | Complexity penalty. It discourages overly flexible covariance structures. |
| $-\frac{N}{2}\ln(2\pi)$ | Normalization constant. It does not affect hyperparameter comparison when $N$ is fixed. |

The gradient with respect to a hyperparameter $\theta_i$ is

$$
\frac{\partial}{\partial \theta_i}
\ln p(\mathbf{t}\mid\boldsymbol{\theta})
=
-\frac{1}{2}\operatorname{Tr}
\left(
\mathbf{C}^{-1}
\frac{\partial \mathbf{C}}{\partial \theta_i}
\right)
+
\frac{1}{2}\mathbf{t}^T\mathbf{C}^{-1}
\frac{\partial \mathbf{C}}{\partial \theta_i}
\mathbf{C}^{-1}\mathbf{t}.
$$

This can be used by numerical optimization methods to find good kernel hyperparameters.

## 4.11 Automatic Relevance Determination

Automatic relevance determination, or ARD, gives each input dimension its own length-scale parameter.

A typical ARD kernel is

$$
k(\mathbf{x},\mathbf{x}')
=
\theta_0
\exp\left\{
-\frac{1}{2}
\sum_{i=1}^{D}
\eta_i(x_i-x_i')^2
\right\}.
$$

The parameter $\eta_i$ controls sensitivity to input dimension $i$.

If $\eta_i$ is large, then even a small change in $x_i$ can strongly reduce the kernel value. The function is sensitive to that dimension.

If $\eta_i$ is small, then changes in $x_i$ have little effect on the kernel. The function is relatively insensitive to that dimension, so that input may be irrelevant.

> ![Figure 6.9](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_9__textbook_fig_6_9_p312_ard_gp_prior_samples.png)
>
> *Figure 6.9 (Textbook Fig. 6.9, p. 312): Samples from Gaussian process priors with ARD. In the right plot, one input dimension has very small relevance, so the sampled function varies much less along that direction.*

> ![Figure 6.10](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_10__textbook_fig_6_10_p313_ard_hyperparameter_optimization.png)
>
> *Figure 6.10 (Textbook Fig. 6.10, p. 313): ARD hyperparameters during optimization. One input becomes highly relevant, one remains moderately useful, and one becomes nearly irrelevant.*

ARD is useful because it gives an automatic way to identify which inputs matter for prediction. It is a Bayesian-style alternative to manually selecting features.

## 4.12 Gaussian Processes for Classification

For classification, the target is discrete. A Gaussian process directly produces real-valued function values, so we need one extra step.

For binary classification, define a latent function

$$
a(\mathbf{x})\sim \mathcal{GP}(0,k(\mathbf{x},\mathbf{x}')).
$$

Then pass it through a sigmoid:

$$
y(\mathbf{x})=\sigma(a(\mathbf{x}))
=
\frac{1}{1+\exp(-a(\mathbf{x}))}.
$$

Now $y(\mathbf{x})$ lies between 0 and 1, so it can be interpreted as a class probability:

$$
y(\mathbf{x})=p(t=1\mid \mathbf{x}).
$$

The likelihood for one target is Bernoulli:

$$
p(t\mid a)=\sigma(a)^t\{1-\sigma(a)\}^{1-t}.
$$

> ![Figure 6.11](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_11__textbook_fig_6_11_p314_gp_classification_sigmoid_transform.png)
>
> *Figure 6.11 (Textbook Fig. 6.11, p. 314): A sample from a Gaussian process prior over latent functions $a(x)$ is transformed through a logistic sigmoid to give probabilities between 0 and 1.*

The main difficulty is that the Bernoulli likelihood is not Gaussian. Therefore the posterior over latent function values is not exactly Gaussian.

The textbook uses a Laplace approximation: find a posterior mode and approximate the posterior locally by a Gaussian.

## 4.13 Laplace Approximation for GP Classification

The posterior over latent values can be written as

$$
p(\mathbf{a}\mid\mathbf{t})
\propto
p(\mathbf{a})p(\mathbf{t}\mid\mathbf{a}).
$$

The prior $p(\mathbf{a})$ is Gaussian because it comes from a GP. The likelihood $p(\mathbf{t}\mid\mathbf{a})$ is a product of Bernoulli terms.

The posterior is therefore not Gaussian. The Laplace approximation proceeds in three steps.

1. Find the mode

$$
\mathbf{a}_{\mathrm{MAP}}
=
\operatorname*{arg\,max}_{\mathbf{a}}p(\mathbf{a}\mid\mathbf{t}).
$$

2. Compute the Hessian of the negative log posterior at the mode.

3. Approximate the posterior by a Gaussian centered at the mode.

The predictive class probability at a new input then requires integrating the sigmoid over an approximate Gaussian distribution for the new latent value.

The detailed algebra is more advanced, so for this course the important message is:

$$
\text{GP classification = latent GP + sigmoid likelihood + approximate inference.}
$$

> ![Figure 6.12](./CoursePR2026/Fig/Chapter_6/lecture_fig_6_12__textbook_fig_6_12_p319_gp_classification_decision_boundary.png)
>
> *Figure 6.12 (Textbook Fig. 6.12, p. 319): Gaussian process classification. The left panel shows training data and decision boundaries. The right panel shows the predicted posterior probability over classes, with the GP decision boundary drawn in black.*

## 4.14 Connection to Neural Networks

The final part of the chapter explains a deep connection between Gaussian processes and Bayesian neural networks.

A neural network with many hidden units can define a very flexible function. If we place a prior distribution over the network weights, then the network output becomes a random function.

In the limit of infinitely many hidden units, for certain priors and activation functions, this random function converges to a Gaussian process.

So we can read the connection as:

$$
\text{Bayesian neural network with infinite hidden units}
\quad\longrightarrow\quad
\text{Gaussian process}.
$$

This does not mean Gaussian processes and neural networks are the same in practice. Neural networks learn hidden representations, while a standard Gaussian process depends heavily on the chosen kernel. But the connection is important because it shows that kernels can be viewed as implicit priors over functions.

---

# §5 Guided Textbook Examples and In-class Problems

> 📖 Based on simple derivations and exercises around textbook §6.1-§6.4

## 5.1 Example 1: Expanding a Polynomial Kernel

**Problem.** For two-dimensional inputs, show the feature map corresponding to

$$
k(\mathbf{x},\mathbf{z})=(\mathbf{x}^T\mathbf{z})^2.
$$

**Step 1: Write the input vectors.**

$$
\mathbf{x}=(x_1,x_2)^T,
\qquad
\mathbf{z}=(z_1,z_2)^T.
$$

**Step 2: Compute the inner product.**

$$
\mathbf{x}^T\mathbf{z}=x_1z_1+x_2z_2.
$$

**Step 3: Square it.**

$$
k(\mathbf{x},\mathbf{z})=(x_1z_1+x_2z_2)^2.
$$

**Step 4: Expand carefully.**

$$
(x_1z_1+x_2z_2)^2
=(x_1z_1)^2+2(x_1z_1)(x_2z_2)+(x_2z_2)^2.
$$

So

$$
k(\mathbf{x},\mathbf{z})
=x_1^2z_1^2+2x_1x_2z_1z_2+x_2^2z_2^2.
$$

**Step 5: Put the $\mathbf{x}$ terms and $\mathbf{z}$ terms into matching vectors.**

$$
k(\mathbf{x},\mathbf{z})
=
(x_1^2,\sqrt{2}x_1x_2,x_2^2)
(z_1^2,\sqrt{2}z_1z_2,z_2^2)^T.
$$

Therefore

$$
\boxed{
\boldsymbol{\phi}(\mathbf{x})=(x_1^2,\sqrt{2}x_1x_2,x_2^2)^T.
}
$$

**Teaching point.** The kernel value can be computed without explicitly forming the feature vector, but the feature vector still exists conceptually.

## 5.2 Example 2: A Tiny Gram Matrix

**Problem.** Suppose we use the linear kernel

$$
k(x,x')=xx'
$$

with one-dimensional training inputs

$$
x_1=1,
\qquad
x_2=2.
$$

Construct the Gram matrix.

**Step 1: Use the definition.**

$$
K_{nm}=k(x_n,x_m).
$$

**Step 2: Compute each entry.**

$$
K_{11}=k(1,1)=1\cdot 1=1.
$$

$$
K_{12}=k(1,2)=1\cdot 2=2.
$$

$$
K_{21}=k(2,1)=2\cdot 1=2.
$$

$$
K_{22}=k(2,2)=2\cdot 2=4.
$$

**Step 3: Assemble the matrix.**

$$
\boxed{
\mathbf{K}=\begin{pmatrix}
1 & 2\\
2 & 4
\end{pmatrix}.
}
$$

This matrix is symmetric, as every Gram matrix from a valid kernel must be.

## 5.3 Example 3: Kernel Ridge Prediction

**Problem.** In dual ridge regression, suppose we already know

$$
\mathbf{a}=(a_1,a_2,a_3)^T.
$$

For a new input $\mathbf{x}_*$, suppose the kernel similarities to the training examples are

$$
\mathbf{k}(\mathbf{x}_*)=(0.8,0.3,0.1)^T.
$$

What is the prediction?

**Step 1: Use the dual prediction formula.**

$$
y(\mathbf{x}_*)=\mathbf{k}(\mathbf{x}_*)^T\mathbf{a}.
$$

**Step 2: Expand the dot product.**

$$
y(\mathbf{x}_*)=0.8a_1+0.3a_2+0.1a_3.
$$

**Teaching point.** Kernel prediction is a similarity-weighted combination of learned dual coefficients.

## 5.4 Example 4: Nadaraya-Watson Local Averaging

**Problem.** Suppose a test point has normalized kernel weights

$$
w_1=0.6,
\qquad
w_2=0.3,
\qquad
w_3=0.1,
$$

and the corresponding training targets are

$$
t_1=2,
\qquad
t_2=4,
\qquad
t_3=10.
$$

Compute the Nadaraya-Watson prediction.

**Step 1: Use the weighted average formula.**

$$
y(\mathbf{x})=\sum_{n=1}^{3}w_nt_n.
$$

**Step 2: Substitute the numbers.**

$$
y(\mathbf{x})=0.6\cdot 2+0.3\cdot 4+0.1\cdot 10.
$$

**Step 3: Compute each term.**

$$
0.6\cdot 2=1.2,
\qquad
0.3\cdot 4=1.2,
\qquad
0.1\cdot 10=1.0.
$$

**Step 4: Add them.**

$$
\boxed{y(\mathbf{x})=1.2+1.2+1.0=3.4.}
$$

**Teaching point.** Even though $t_3=10$ is large, its influence is small because its kernel weight is only $0.1$.

## 5.5 Example 5: One-training-point Gaussian Process Regression

This example is a simple algebraic version of Figure 6.7.

**Problem.** Suppose we have one training target $t_1$ and one test input $x_*$. The covariance matrix for the observed training target is a scalar

$$
C=k(x_1,x_1)+\beta^{-1}.
$$

Let

$$
k_*=k(x_1,x_*).
$$

Find the predictive mean.

**Step 1: Start from the general GP predictive mean.**

$$
m(x_*)=\mathbf{k}^T\mathbf{C}^{-1}\mathbf{t}.
$$

**Step 2: In the one-point case, all vectors and matrices become scalars.**

$$
\mathbf{k}=k_*,
\qquad
\mathbf{C}=C,
\qquad
\mathbf{t}=t_1.
$$

**Step 3: Substitute.**

$$
m(x_*)=k_*\,C^{-1}\,t_1.
$$

**Step 4: Since $C$ is a scalar, $C^{-1}=1/C$.**

$$
\boxed{
 m(x_*)=\frac{k(x_1,x_*)}{k(x_1,x_1)+\beta^{-1}}t_1.
}
$$

**Interpretation.** If $x_*$ is similar to $x_1$, then $k(x_1,x_*)$ is large and the prediction is strongly influenced by $t_1$. If $x_*$ is far from $x_1$, then $k(x_1,x_*)$ is small and the prediction moves toward the prior mean, which is zero here.

## 5.6 Example 6: Why GP Uncertainty Increases Away from Data

The predictive variance is

$$
\sigma^2(x_*)=c-\mathbf{k}^T\mathbf{C}^{-1}\mathbf{k}.
$$

The first term $c$ is the prior variance. The second term is the uncertainty reduction due to observed data.

If $x_*$ is far away from the training points, then

$$
\mathbf{k}\approx \mathbf{0}.
$$

Therefore

$$
\mathbf{k}^T\mathbf{C}^{-1}\mathbf{k}\approx 0.
$$

So

$$
\sigma^2(x_*)\approx c.
$$

This means the model returns to prior uncertainty far away from data.

This is exactly what Figure 6.8 shows: the shaded predictive interval becomes wider in the region where training observations are missing.

---

# §6 Chapter Summary, Figure Checklist, and Teaching Flow

## 6.1 Chapter Summary

Chapter 6 introduces kernel methods as a second major way to build nonlinear models.

The main story is:

$$
\text{linear model in feature space}
\quad\Rightarrow\quad
\text{inner products of features}
\quad\Rightarrow\quad
\text{kernel functions}
\quad\Rightarrow\quad
\text{dual prediction and Gaussian processes}.
$$

The key conceptual points are:

| Concept | Main Lesson |
|---------|-------------|
| Kernel trick | If an algorithm only uses inner products, replace them with kernels. |
| Dual representation | The solution can often be expressed using training examples instead of explicit weights. |
| Gram matrix | Stores all pairwise kernel similarities between training points. |
| Valid kernel | Must generate positive semidefinite Gram matrices. |
| RBF network | Uses local distance-based basis functions. |
| Nadaraya-Watson model | Predicts by normalized local averaging. |
| Gaussian process | Uses a kernel as a covariance function over function values. |
| GP regression | Gives both predictive mean and predictive variance. |
| Marginal likelihood | Learns kernel hyperparameters by balancing data fit and complexity. |
| ARD | Learns which input dimensions are relevant. |
| GP classification | Uses a latent GP plus sigmoid likelihood and approximate inference. |

## 6.2 What Students Should Be Able to Do

After this lecture, students should be able to:

1. Explain what a kernel function is and why it can replace feature-space inner products.
2. Derive the dual representation of regularized least-squares regression.
3. Construct a Gram matrix from a set of training inputs and a kernel.
4. Expand simple polynomial kernels into explicit feature maps.
5. Explain the positive semidefinite condition for valid kernels.
6. Interpret RBF networks as local basis-function models.
7. Derive the Nadaraya-Watson prediction as a normalized weighted average.
8. Explain what a Gaussian process is in terms of finite-dimensional Gaussian distributions.
9. Derive and interpret the GP regression predictive mean and variance.
10. Explain how kernel hyperparameters and ARD affect the behaviour of Gaussian processes.
11. Describe why GP classification requires approximate inference.
12. State the connection between infinite Bayesian neural networks and Gaussian processes.

## 6.3 Figure Checklist

All figures below are taken from the textbook PDF and saved under `./CoursePR2026/Fig/Chapter_6/`.

| Lecture Fig. | Textbook Fig. | Topic | File Path |
|--------------|---------------|-------|-----------|
| 6.1 | 6.1 | Basis functions and induced kernels | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_1__textbook_fig_6_1_p295_basis_functions_and_kernel_construction.png` |
| 6.2 | 6.2 | Gaussian basis functions and normalized basis functions | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_2__textbook_fig_6_2_p301_gaussian_basis_functions_normalized.png` |
| 6.3 | 6.3 | Nadaraya-Watson kernel regression | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_3__textbook_fig_6_3_p303_nadaraya_watson_kernel_regression.png` |
| 6.4 | 6.4 | Gaussian process samples with different kernels | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_4__textbook_fig_6_4_p306_gp_samples_gaussian_vs_exponential_kernel.png` |
| 6.5 | 6.5 | Gaussian process prior samples under different hyperparameters | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_5__textbook_fig_6_5_p308_gp_prior_samples_covariance_parameters.png` |
| 6.6 | 6.6 | Sampling noisy data from a Gaussian process | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_6__textbook_fig_6_6_p309_sampling_data_from_gp.png` |
| 6.7 | 6.7 | GP regression with one training point and one test point | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_7__textbook_fig_6_7_p310_gp_regression_one_training_one_test_point.png` |
| 6.8 | 6.8 | GP regression on the sinusoidal data set | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_8__textbook_fig_6_8_p310_gp_regression_sinusoidal_prediction.png` |
| 6.9 | 6.9 | ARD Gaussian process prior samples | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_9__textbook_fig_6_9_p312_ard_gp_prior_samples.png` |
| 6.10 | 6.10 | ARD hyperparameters during optimization | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_10__textbook_fig_6_10_p313_ard_hyperparameter_optimization.png` |
| 6.11 | 6.11 | Sigmoid transformation for GP classification | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_11__textbook_fig_6_11_p314_gp_classification_sigmoid_transform.png` |
| 6.12 | 6.12 | GP classification decision boundary and posterior probability | `./CoursePR2026/Fig/Chapter_6/lecture_fig_6_12__textbook_fig_6_12_p319_gp_classification_decision_boundary.png` |

## 6.4 Suggested Teaching Flow

A practical lecture sequence is:

1. Start by reviewing the fixed basis-function model $y(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})$.
2. Explain why explicit feature maps can be difficult or expensive.
3. Introduce the kernel function $k(\mathbf{x},\mathbf{x}')=\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}')$.
4. Derive the dual representation of regularized least squares slowly, without skipping the gradient step.
5. Define the Gram matrix and explain its dimensions.
6. Derive the prediction formula $y(\mathbf{x})=\mathbf{k}(\mathbf{x})^T(\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{t}$.
7. Use Figure 6.1 to show how basis functions induce kernel shapes.
8. Work through the polynomial kernel expansion $(\mathbf{x}^T\mathbf{z})^2$ in detail.
9. Explain positive semidefinite Gram matrices and why this condition is needed.
10. Present kernel construction rules with examples.
11. Introduce Gaussian/RBF kernels and explain distance-based similarity.
12. Use Figure 6.2 to explain normalized basis functions.
13. Introduce RBF networks as local basis-function models.
14. Derive the Nadaraya-Watson model as normalized local averaging.
15. Use Figure 6.3 to explain local regression and uncertainty.
16. Introduce Gaussian processes as distributions over functions.
17. Show how Bayesian linear regression induces a GP through the covariance function.
18. Use Figures 6.4 and 6.5 to explain how kernels control sample functions.
19. Derive GP regression: noisy targets, covariance matrix $\mathbf{C}$, joint Gaussian, conditional mean, and conditional variance.
20. Use Figure 6.6 to distinguish latent functions from noisy observations.
21. Use Figure 6.7 to explain conditioning in a joint Gaussian.
22. Use Figure 6.8 to explain predictive mean and uncertainty away from data.
23. Explain marginal likelihood for learning kernel hyperparameters.
24. Introduce ARD with Figures 6.9 and 6.10.
25. Finish with GP classification using Figures 6.11 and 6.12.
26. Briefly mention the infinite-width neural-network connection.

## 6.5 Key Equations to Put on the Board

The following equations are the minimum board set for this chapter.

### Kernel definition

$$
k(\mathbf{x},\mathbf{x}')
=
\boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\phi}(\mathbf{x}').
$$

### Gram matrix

$$
K_{nm}=k(\mathbf{x}_n,\mathbf{x}_m).
$$

### Regularized least-squares objective

$$
J(\mathbf{w})=
\frac{1}{2}\sum_{n=1}^{N}
\{\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)-t_n\}^2
+
\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}.
$$

### Dual weight representation

$$
\mathbf{w}=\boldsymbol{\Phi}^T\mathbf{a}.
$$

### Dual ridge solution

$$
\mathbf{a}=(\mathbf{K}+\lambda\mathbf{I}_N)^{-1}\mathbf{t}.
$$

### Kernel ridge prediction

$$
y(\mathbf{x})
=
\mathbf{k}(\mathbf{x})^T
(\mathbf{K}+\lambda\mathbf{I}_N)^{-1}\mathbf{t}.
$$

### Polynomial kernel feature map example

$$
(\mathbf{x}^T\mathbf{z})^2
=
(x_1^2,\sqrt{2}x_1x_2,x_2^2)
(z_1^2,\sqrt{2}z_1z_2,z_2^2)^T.
$$

### Gaussian/RBF kernel

$$
k(\mathbf{x},\mathbf{x}')
=
\exp\left(-\frac{\|\mathbf{x}-\mathbf{x}'\|^2}{2\sigma^2}\right).
$$

### Normalized RBF basis function

$$
\widetilde{\phi}_j(\mathbf{x})
=
\frac{\phi_j(\mathbf{x})}{\sum_m\phi_m(\mathbf{x})}.
$$

### Nadaraya-Watson prediction

$$
y(\mathbf{x})
=
\sum_{n=1}^{N}
\frac{h(\mathbf{x}-\mathbf{x}_n)}{\sum_m h(\mathbf{x}-\mathbf{x}_m)}t_n.
$$

### Gaussian process definition

$$
y(\mathbf{x})\sim \mathcal{GP}(m(\mathbf{x}),k(\mathbf{x},\mathbf{x}')).
$$

### GP regression covariance matrix

$$
\mathbf{C}=\mathbf{K}+\beta^{-1}\mathbf{I}_N.
$$

### GP predictive mean

$$
m(\mathbf{x}_*)=\mathbf{k}^T\mathbf{C}^{-1}\mathbf{t}.
$$

### GP predictive variance

$$
\sigma^2(\mathbf{x}_*)=c-\mathbf{k}^T\mathbf{C}^{-1}\mathbf{k}.
$$

### GP log marginal likelihood

$$
\ln p(\mathbf{t}\mid\boldsymbol{\theta})
=
-\frac{1}{2}\ln |\mathbf{C}|
-\frac{1}{2}\mathbf{t}^T\mathbf{C}^{-1}\mathbf{t}
-\frac{N}{2}\ln(2\pi).
$$

### ARD kernel

$$
k(\mathbf{x},\mathbf{x}')
=
\theta_0
\exp\left\{
-\frac{1}{2}
\sum_{i=1}^{D}
\eta_i(x_i-x_i')^2
\right\}.
$$

### GP classification likelihood

$$
p(t\mid a)=\sigma(a)^t\{1-\sigma(a)\}^{1-t}.
$$

### Infinite-width neural-network connection

$$
\text{Bayesian neural network with infinitely many hidden units}
\Rightarrow
\text{Gaussian process}.
$$
