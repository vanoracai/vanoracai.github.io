---
layout: course
title: PRML Lecture 4
---

# Pattern Recognition and Machine Learning
## Chapter 4: Linear Models for Classification

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 4 Linear Models for Classification (§4.1-§4.5)

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Discriminant Functions](#1-discriminant-functions)
3. [§2 Probabilistic Generative Models](#2-probabilistic-generative-models)
4. [§3 Probabilistic Discriminative Models](#3-probabilistic-discriminative-models)
5. [§4 The Laplace Approximation](#4-the-laplace-approximation)
6. [§5 Bayesian Logistic Regression](#5-bayesian-logistic-regression)
7. [§6 Chapter Summary, Figure Checklist, Exercises, and Teaching Flow](#6-chapter-summary-figure-checklist-exercises-and-teaching-flow)

---

## Notation and Variable Definitions

Chapter 3 studied linear models for regression, where the target variable is continuous. Chapter 4 studies the analogous family of models for **classification**, where the target is discrete:

> Given an input vector $\mathbf{x}$, assign it to one of $K$ classes $\mathcal{C}_1,\ldots,\mathcal{C}_K$.

The phrase **linear model** again has a precise meaning. A classifier may become nonlinear in the original input space after applying basis functions $\boldsymbol{\phi}(\mathbf{x})$, but the decision score is linear in the adaptive parameters $\mathbf{w}$. Classification differs from regression because the output is no longer a real-valued target; it is a class label or a posterior probability over class labels.

> **Teaching focus.** This chapter should connect directly to modern classification models. A neural network classifier still ends with:
>
> $$
> \text{features} \rightarrow \text{linear logits} \rightarrow \text{sigmoid/softmax} \rightarrow \text{cross-entropy}.
> $$
>
> For this course, the essential parts are the geometry of linear decision boundaries, why least squares is not ideal for labels, Fisher/perceptron as historical and geometric baselines, and logistic/softmax regression as the modern core. The Laplace approximation and Bayesian logistic regression are useful as a preview of approximate inference, but can be taught at the intuition level.

### Generic Classification Notation

| Symbol | Definition |
|--------|------------|
| $\mathbf{x}$ | Input vector in the original observation space. |
| $D$ | Dimensionality of $\mathbf{x}$. |
| $\mathcal{C}_k$ | The $k$th class. |
| $K$ | Number of classes. |
| $\mathcal{D}$ | Training data set, often $\{(\mathbf{x}_n,\mathbf{t}_n)\}_{n=1}^N$. |
| $N$ | Number of training examples. |
| $\mathbf{t}_n$ | Target coding for example $n$, often 1-of-$K$ coding. |
| $t_n$ | Binary target, commonly $t_n\in\{0,1\}$ for logistic regression or $t_n\in\{-1,+1\}$ for the perceptron. |
| $\mathcal{R}_k$ | Decision region assigned to class $\mathcal{C}_k$. |
| decision boundary | Surface separating two decision regions. |
| linear separability | Existence of a hyperplane that separates the classes perfectly. |

### Linear Discriminant Notation

| Symbol | Definition |
|--------|------------|
| $y(\mathbf{x})$ | Discriminant function or decision score. |
| $y_k(\mathbf{x})$ | Score for class $\mathcal{C}_k$. |
| $\mathbf{w}$ | Weight vector defining the orientation of a linear boundary. |
| $w_0$ | Bias/intercept parameter controlling the offset of the boundary. |
| $\boldsymbol{\phi}(\mathbf{x})$ | Basis-function vector / feature representation. |
| $\boldsymbol{\Phi}$ | Design matrix whose rows are feature vectors. |
| $\mathbf{W}$ | Weight matrix for multiclass linear classification. |
| $\mathbf{T}$ | Matrix of target vectors using 1-of-$K$ coding. |

### Fisher Discriminant Notation

| Symbol | Definition |
|--------|------------|
| $\mathbf{m}_k$ | Mean vector of class $\mathcal{C}_k$. |
| $N_k$ | Number of data points in class $\mathcal{C}_k$. |
| $S_W$ | Within-class scatter matrix. |
| $S_B$ | Between-class scatter matrix. |
| $J(\mathbf{w})$ | Fisher criterion, ratio of between-class to within-class scatter. |
| $\mathbf{y}$ | Low-dimensional projected feature vector in multiclass Fisher analysis. |

### Probabilistic Classification Notation

| Symbol | Definition |
|--------|------------|
| $p(\mathcal{C}_k\mid\mathbf{x})$ | Posterior probability that $\mathbf{x}$ belongs to class $\mathcal{C}_k$. |
| $p(\mathbf{x}\mid\mathcal{C}_k)$ | Class-conditional density. |
| $p(\mathcal{C}_k)$ | Class prior probability. |
| $\sigma(a)$ | Logistic sigmoid, $\sigma(a)=1/(1+\exp(-a))$. |
| $a(\mathbf{x})$ | Logit or log-odds score. |
| $\mathrm{softmax}(\mathbf{a})$ | Multiclass probability mapping from scores $a_k$ to class probabilities. |
| $\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k$ | Mean and covariance of a Gaussian class-conditional density. |

### Logistic Regression and Bayesian Notation

| Symbol | Definition |
|--------|------------|
| $E(\mathbf{w})$ | Negative log-likelihood / cross-entropy error. |
| $\mathbf{R}$ | Diagonal weight matrix in IRLS, with entries $y_n(1-y_n)$. |
| $\mathbf{H}$ | Hessian matrix of the error function. |
| $\mathbf{z}$ | Adjusted response vector in IRLS. |
| $\mathbf{w}_{\mathrm{MAP}}$ | Maximum-a-posteriori weight vector. |
| $\mathbf{S}_N$ | Approximate posterior covariance from the Laplace approximation. |
| $\kappa(\sigma^2)$ | Variance-dependent shrinkage factor in Bayesian logistic prediction. |

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch.4 opening; §4.1-§4.5

## 0.1 What This Chapter Is Really About

The goal of classification is to divide input space into regions. In a two-class problem, a classifier draws one boundary. Points on one side are assigned to class $\mathcal{C}_1$ and points on the other side to class $\mathcal{C}_2$. In a $K$-class problem, the classifier partitions the input space into $K$ decision regions.

The simplest boundary is a hyperplane:

$$
y(\mathbf{x})=\mathbf{w}^T\mathbf{x}+w_0=0.
$$

If the model predicts class labels directly from the sign or largest value of such functions, it is called a **discriminant-function approach**. If the model instead returns probabilities such as $p(\mathcal{C}_k\mid\mathbf{x})$, then it supports uncertainty-aware decisions, rejection, cost-sensitive classification, and model averaging.

This chapter can be read as answering five linked questions.

| Topic | Core Question | Long-Term Role in the Course |
|-------|---------------|-------------------------------|
| **Linear discriminants** | What is the geometry of a linear decision boundary? | Foundation for perceptrons, SVMs, and neural-network classifiers. |
| **Least squares and Fisher** | Can regression-style linear algebra be used for classification? | Shows both useful projections and failure modes of squared loss. |
| **Generative classifiers** | What posterior probabilities arise from modelling $p(\mathbf{x}\mid\mathcal{C}_k)$? | Foundation for Gaussian discriminant analysis and naive Bayes. |
| **Logistic regression** | How can we model $p(\mathcal{C}_k\mid\mathbf{x})$ directly? | Core discriminative probabilistic classifier. |
| **Laplace and Bayesian logistic regression** | How can we approximate uncertainty when the posterior is not Gaussian? | Foundation for approximate Bayesian inference in later chapters. |

The recurring pattern of the chapter is:

$$
\text{linear score}
\longrightarrow
\text{decision boundary}
\longrightarrow
\text{posterior probability}
\longrightarrow
\text{uncertainty-aware prediction}.
$$

## 0.2 Three Ways to Build a Classifier

Chapter 4 repeatedly contrasts three modelling strategies.

| Strategy | What It Models | Output | Strength | Weakness |
|----------|----------------|--------|----------|----------|
| **Discriminant function** | A direct mapping $\mathbf{x}\mapsto$ class label | Hard label | Simple and efficient | No calibrated uncertainty. |
| **Generative probabilistic model** | $p(\mathbf{x}\mid\mathcal{C}_k)$ and $p(\mathcal{C}_k)$ | Posterior via Bayes' theorem | Can use density structure and missing data | Requires modelling the input distribution. |
| **Discriminative probabilistic model** | $p(\mathcal{C}_k\mid\mathbf{x})$ directly | Posterior probability | Often fewer assumptions and fewer parameters | Usually needs iterative optimization. |

The chapter begins with direct linear decision functions. It then shows that generative Gaussian assumptions can lead naturally to logistic sigmoid and softmax posterior probabilities. Finally, it treats logistic regression as a discriminative model and introduces the Laplace approximation to make Bayesian prediction tractable.

## 0.3 The Key Transition from Chapter 3

Chapter 3 used a linear basis-function model for real-valued targets:

$$
y(\mathbf{x},\mathbf{w})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}).
$$

Chapter 4 keeps the linear score but changes the interpretation of the output. A raw linear score is not yet a probability. To obtain a valid probability, it must be passed through a suitable **activation function**:

$$
\text{binary probability:}\quad p(\mathcal{C}_1\mid\boldsymbol{\phi})=\sigma(\mathbf{w}^T\boldsymbol{\phi}),
$$

or, for multiple classes,

$$
\text{multiclass probability:}\quad
p(\mathcal{C}_k\mid\boldsymbol{\phi})
=\frac{\exp(a_k)}{\sum_j\exp(a_j)}.
$$

This is the conceptual origin of many modern classification layers. A neural network classifier, for example, often learns a nonlinear representation and then applies a linear layer followed by a softmax.

---

# §1 Discriminant Functions

> 📖 Textbook §4.1 Discriminant Functions; §4.1.1-§4.1.7

## 1.1 Basic Setup: Decision Regions and Linear Scores

A discriminant function assigns an input to a class by computing one or more scores. For two classes, the simplest rule is

$$
y(\mathbf{x})=\mathbf{w}^T\mathbf{x}+w_0,
$$

and then

$$
\begin{cases}
\mathbf{x}\in\mathcal{C}_1, & y(\mathbf{x})>0,\\
\mathbf{x}\in\mathcal{C}_2, & y(\mathbf{x})<0.
\end{cases}
$$

The decision boundary is the set of points for which $y(\mathbf{x})=0$. In $D$ dimensions this boundary is a $(D-1)$-dimensional hyperplane.

A more general classifier first transforms the input through fixed basis functions:

$$
y(\mathbf{x})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}).
$$

This is still a linear model in feature space, but it can produce nonlinear boundaries in the original input space. The same idea appeared in Chapter 3 for regression, and it becomes even more important here because many classification data sets are not linearly separable in their original coordinates.

## 1.2 Two Classes: Geometry of a Linear Discriminant

For the two-class model

$$
y(\mathbf{x})=\mathbf{w}^T\mathbf{x}+w_0,
$$

there are two geometric facts to remember.

First, $\mathbf{w}$ is normal to the decision boundary. If $\mathbf{x}_A$ and $\mathbf{x}_B$ are two points on the boundary, then

$$
\mathbf{w}^T\mathbf{x}_A+w_0=0,
\qquad
\mathbf{w}^T\mathbf{x}_B+w_0=0.
$$

Subtracting these equations gives

$$
\mathbf{w}^T(\mathbf{x}_A-\mathbf{x}_B)=0,
$$

so $\mathbf{w}$ is orthogonal to every direction lying inside the boundary.

Second, $w_0$ controls the offset of the boundary from the origin. The signed perpendicular distance from a point $\mathbf{x}$ to the boundary is

$$
r=\frac{y(\mathbf{x})}{\|\mathbf{w}\|}.
$$

This formula is useful pedagogically because it shows that a discriminant score is not just an arbitrary number. Its magnitude measures how far the point lies from the separating hyperplane, up to the scale of $\mathbf{w}$.

> ![Figure 4.1](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_1__textbook_fig_4_1_p182_linear_discriminant_geometry.png)
>
> *Figure 4.1 (Textbook Fig. 4.1, p. 182): The red line is the decision surface $y=0$. The vector $\mathbf{w}$ is perpendicular to this line, and its direction points toward the region where the score is positive. The point $\mathbf{x}$ is decomposed into a point $\mathbf{x}_\perp$ on the boundary plus a normal displacement. The signed distance is $y(\mathbf{x})/\|\mathbf{w}\|$, so the linear score carries both class information and a notion of confidence margin.*

## 1.3 Multiple Classes: Why Naive Pairwise Constructions Can Fail

For more than two classes, a tempting idea is to build several two-class classifiers and combine their outputs. There are two common constructions.

The first is **one-versus-rest**. For each class $\mathcal{C}_k$, train a classifier that separates $\mathcal{C}_k$ from all other classes. The problem is that a point can be rejected by every classifier, or accepted by more than one classifier, creating ambiguous regions.

The second is **one-versus-one**. For each pair of classes $(\mathcal{C}_j,\mathcal{C}_k)$, train a classifier that separates those two classes. This also can create regions where pairwise decisions are inconsistent.

> ![Figure 4.2](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_2__textbook_fig_4_2_p183_one_vs_rest_one_vs_one_ambiguity.png)
>
> *Figure 4.2 (Textbook Fig. 4.2, p. 183): The green regions show the key failure mode. Independent binary classifiers do not necessarily agree on a single global multiclass partition of input space. This is why it is cleaner to define one joint $K$-class model rather than trying to paste together unrelated binary decisions.*

A more coherent approach uses $K$ linear discriminant functions

$$
y_k(\mathbf{x})=\mathbf{w}_k^T\mathbf{x}+w_{k0},
\qquad k=1,\ldots,K,
$$

and assigns $\mathbf{x}$ to the class with the largest score:

$$
\operatorname*{arg\,max}_{k} y_k(\mathbf{x}).
$$

The boundary between classes $\mathcal{C}_k$ and $\mathcal{C}_j$ is given by

$$
y_k(\mathbf{x})=y_j(\mathbf{x}),
$$

or equivalently

$$
(\mathbf{w}_k-\mathbf{w}_j)^T\mathbf{x}+(w_{k0}-w_{j0})=0.
$$

Thus each pairwise boundary is still a hyperplane, but all boundaries are now coupled through a single set of class scores.

> ![Figure 4.3](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_3__textbook_fig_4_3_p184_multiclass_linear_regions_convexity.png)
>
> *Figure 4.3 (Textbook Fig. 4.3, p. 184): Multiclass linear discriminants create decision regions that are convex and singly connected. If two points lie inside the same region, then every point on the straight line between them also lies in that region. This is a useful limitation to remember: linear multiclass models cannot represent disconnected decision regions unless we first transform the inputs through nonlinear basis functions.*

## 1.4 Least Squares for Classification

Because Chapter 3 solved regression with least squares, it is natural to ask whether classification can also be solved by least squares. With 1-of-$K$ target coding, let $\mathbf{t}_n$ be a vector whose $k$th component is 1 if $\mathbf{x}_n$ belongs to class $\mathcal{C}_k$ and 0 otherwise. Define a linear output model

$$
\mathbf{y}(\mathbf{x})=\mathbf{W}^T\boldsymbol{\phi}(\mathbf{x}).
$$

The sum-of-squares error is

$$
E_D(\mathbf{W})
=\frac{1}{2}\sum_{n=1}^N
\|\mathbf{W}^T\boldsymbol{\phi}(\mathbf{x}_n)-\mathbf{t}_n\|^2.
$$

Using the design matrix $\boldsymbol{\Phi}$ and the target matrix $\mathbf{T}$, the solution has the same form as linear regression:

$$
\mathbf{W}_{\mathrm{LS}}
=(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{T}
=\boldsymbol{\Phi}^{\dagger}\mathbf{T}.
$$

At first this looks attractive because it is closed-form and computationally simple. However, least squares is usually a poor classification loss. It assumes Gaussian noise around target vectors, but class labels are discrete. It also penalizes predictions that are far beyond the correct side of the boundary, even though being “very correctly classified” should not be a problem.

> ![Figure 4.4](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_4__textbook_fig_4_4_p186_least_squares_outlier_sensitivity.png)
>
> *Figure 4.4 (Textbook Fig. 4.4, p. 186): In the left plot, least squares and logistic regression both produce reasonable boundaries. In the right plot, a cluster of extra points far from the boundary strongly moves the least-squares boundary, even though those points are already correctly classified. The squared loss keeps trying to fit the numerical target values, not just the class separation. Logistic regression is less affected because its cross-entropy loss saturates differently for confidently correct examples.*

Least squares can be even more problematic in multiclass problems.

> ![Figure 4.5](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_5__textbook_fig_4_5_p187_least_squares_multiclass_failure.png)
>
> *Figure 4.5 (Textbook Fig. 4.5, p. 187): The data are almost linearly separable by sensible class regions. The least-squares solution on the left assigns too little space to the green class, while logistic regression on the right produces a much more natural partition. This figure is important because it shows that a closed-form solution is not automatically a good classification method. The loss function must match the nature of the target.*

## 1.5 Fisher's Linear Discriminant

Fisher's linear discriminant takes a different view. Instead of fitting target values, it asks for a projection direction that makes the projected classes as separable as possible.

For two classes, project $\mathbf{x}$ onto one dimension:

$$
y=\mathbf{w}^T\mathbf{x}.
$$

Let the class means in the original space be

$$
\mathbf{m}_1=\frac{1}{N_1}\sum_{n\in\mathcal{C}_1}\mathbf{x}_n,
\qquad
\mathbf{m}_2=\frac{1}{N_2}\sum_{n\in\mathcal{C}_2}\mathbf{x}_n.
$$

The projected class means are

$$
m_1'=\mathbf{w}^T\mathbf{m}_1,
\qquad
m_2'=\mathbf{w}^T\mathbf{m}_2.
$$

A naive objective would maximize $(m_2'-m_1')^2$, but this ignores the spread of each projected class. Fisher's idea is to maximize separation between projected means while minimizing projected within-class variance.

Define the within-class scatter matrix

$$
S_W
=\sum_{n\in\mathcal{C}_1}(\mathbf{x}_n-\mathbf{m}_1)(\mathbf{x}_n-\mathbf{m}_1)^T
+\sum_{n\in\mathcal{C}_2}(\mathbf{x}_n-\mathbf{m}_2)(\mathbf{x}_n-\mathbf{m}_2)^T,
$$

and the between-class scatter matrix

$$
S_B=(\mathbf{m}_2-\mathbf{m}_1)(\mathbf{m}_2-\mathbf{m}_1)^T.
$$

The Fisher criterion is

$$
J(\mathbf{w})
=\frac{\mathbf{w}^T S_B\mathbf{w}}{\mathbf{w}^T S_W\mathbf{w}}.
$$

Maximizing this ratio gives

$$
\mathbf{w}\propto S_W^{-1}(\mathbf{m}_2-\mathbf{m}_1).
$$

> ![Figure 4.6](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_6__textbook_fig_4_6_p188_fisher_projection_class_separation.png)
>
> *Figure 4.6 (Textbook Fig. 4.6, p. 188): The line joining the class means is not necessarily a good projection direction. In the left plot, projection onto that direction creates substantial overlap in one dimension. Fisher's discriminant, shown on the right, accounts for within-class covariance and finds a projection with much cleaner separation.*

After projecting onto one dimension, we still need a threshold to classify points. Fisher's method gives a representation; it does not by itself specify a probabilistic classifier. The threshold can be chosen by minimizing training errors, by validation, or by fitting simple one-dimensional class-conditional densities in the projected space.

## 1.6 Relation Between Fisher and Least Squares

Fisher's discriminant may look very different from least squares, but for two classes there is a close relationship. With a particular choice of target coding, the least-squares solution yields a weight vector parallel to Fisher's solution.

The important teaching point is not that squared loss is generally ideal for classification. Rather, it is that several linear classification methods can be understood as different ways of choosing a useful separating direction. Least squares chooses weights to match numeric target codes. Fisher chooses weights to optimize a ratio of class separation to class spread. Logistic regression chooses weights to maximize conditional likelihood.

This comparison helps students see why the same linear algebra can appear in many different classifiers while the underlying modelling assumptions remain different.

## 1.7 Fisher's Discriminant for Multiple Classes

> **Optional depth.** The two-class Fisher idea is useful and visual. The full multiclass determinant-ratio derivation is less central for modern ML practice, so it can be summarized rather than derived in detail.

For $K$ classes, Fisher's idea generalizes by defining an overall within-class scatter and between-class scatter. Let

$$
\mathbf{m}=\frac{1}{N}\sum_{n=1}^N\mathbf{x}_n
$$

be the global mean, and let $\mathbf{m}_k$ be the mean of class $\mathcal{C}_k$. The within-class scatter is

$$
S_W=\sum_{k=1}^K\sum_{n\in\mathcal{C}_k}
(\mathbf{x}_n-\mathbf{m}_k)(\mathbf{x}_n-\mathbf{m}_k)^T,
$$

and the between-class scatter is

$$
S_B=\sum_{k=1}^K N_k(\mathbf{m}_k-\mathbf{m})(\mathbf{m}_k-\mathbf{m})^T.
$$

Instead of projecting to one dimension, we choose a projection matrix $\mathbf{W}$ and obtain

$$
\mathbf{y}=\mathbf{W}^T\mathbf{x}.
$$

A common multiclass Fisher objective is the determinant ratio

$$
J(\mathbf{W})
=\frac{|\mathbf{W}^T S_B\mathbf{W}|}{|\mathbf{W}^T S_W\mathbf{W}|}.
$$

The rank of $S_B$ is at most $K-1$, so Fisher's method can produce at most $K-1$ useful discriminant dimensions. This is a key practical point: for a ten-class problem, Fisher's criterion can at most create a nine-dimensional discriminative subspace, no matter how high-dimensional the original input is.

## 1.8 The Perceptron Algorithm

The perceptron is one of the earliest and most historically important linear classification algorithms. It uses targets $t_n\in\{-1,+1\}$ and a decision rule based on

$$
y(\mathbf{x})=f(\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})),
$$

where $f(\cdot)$ is a step function. A point is correctly classified when

$$
\mathbf{w}^T\boldsymbol{\phi}_n t_n>0.
$$

The perceptron criterion sums only over misclassified examples:

$$
E_P(\mathbf{w})
=-\sum_{n\in\mathcal{M}}\mathbf{w}^T\boldsymbol{\phi}_n t_n,
$$

where $\mathcal{M}$ is the set of misclassified points. A stochastic gradient step gives the update

$$
\mathbf{w}^{(\tau+1)}
=\mathbf{w}^{(\tau)}+\eta\boldsymbol{\phi}_n t_n.
$$

This update has an intuitive interpretation. If a positive example is misclassified, add its feature vector to $\mathbf{w}$. If a negative example is misclassified, subtract its feature vector from $\mathbf{w}$. Each update rotates or shifts the decision boundary so that the current mistaken point is more likely to be classified correctly.

> ![Figure 4.7](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_7__textbook_fig_4_7_p195_perceptron_learning_convergence.png)
>
> *Figure 4.7 (Textbook Fig. 4.7, p. 195): Each panel shows the current parameter vector and decision boundary. The circled point is misclassified, so its feature vector is used to update the weights. The boundary changes after each update. When the data are linearly separable, the perceptron convergence theorem guarantees that this process eventually finds a separating solution.*

> ![Figure 4.8](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_8__textbook_fig_4_8_p196_mark_1_perceptron_hardware.png)
>
> *Figure 4.8 (Textbook Fig. 4.8, p. 196): The perceptron was not just an abstract algorithm; it was also implemented as physical hardware. The system used a primitive image sensor, configurable feature wiring, and adaptive parameters. This figure is useful for reminding students that modern neural networks evolved from very concrete attempts to build trainable pattern-recognition machines.*

The perceptron has important limitations. It does not output probabilities, so it cannot directly express uncertainty. It converges only when the data are linearly separable in feature space. If the data are not separable, the algorithm can keep updating indefinitely. These limitations motivate probabilistic classifiers such as logistic regression.

---

# §2 Probabilistic Generative Models

> 📖 Textbook §4.2 Probabilistic Generative Models; §4.2.1-§4.2.4

## 2.1 From Class-Conditional Densities to Posterior Probabilities

A generative classifier models the class-conditional densities and class priors:

$$
p(\mathbf{x}\mid\mathcal{C}_k),
\qquad
p(\mathcal{C}_k).
$$

Bayes' theorem then gives

$$
p(\mathcal{C}_k\mid\mathbf{x})
=\frac{p(\mathbf{x}\mid\mathcal{C}_k)p(\mathcal{C}_k)}{\sum_j p(\mathbf{x}\mid\mathcal{C}_j)p(\mathcal{C}_j)}.
$$

For two classes, this can be written as a logistic sigmoid. Define the log-odds score

$$
a(\mathbf{x})
=\ln\frac{p(\mathbf{x}\mid\mathcal{C}_1)p(\mathcal{C}_1)}{p(\mathbf{x}\mid\mathcal{C}_2)p(\mathcal{C}_2)}.
$$

Then

$$
p(\mathcal{C}_1\mid\mathbf{x})=\sigma(a),
\qquad
p(\mathcal{C}_2\mid\mathbf{x})=1-\sigma(a),
$$

where

$$
\sigma(a)=\frac{1}{1+\exp(-a)}.
$$

> ![Figure 4.9](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_9__textbook_fig_4_9_p197_logistic_sigmoid_and_probit.png)
>
> *Figure 4.9 (Textbook Fig. 4.9, p. 197): The logistic sigmoid maps any real-valued score to a number between 0 and 1. Large positive scores become probabilities near 1, large negative scores become probabilities near 0, and a score of 0 corresponds to probability 0.5. The figure also compares the sigmoid with a scaled probit function, which will become useful when we discuss probit regression and Bayesian logistic prediction.*

For $K$ classes, Bayes' theorem gives the softmax form

$$
p(\mathcal{C}_k\mid\mathbf{x})
=\frac{\exp(a_k)}{\sum_j\exp(a_j)},
$$

where

$$
a_k=\ln p(\mathbf{x}\mid\mathcal{C}_k)+\ln p(\mathcal{C}_k).
$$

Thus sigmoid and softmax are not arbitrary choices. They arise naturally when posterior probabilities are written in terms of log probabilities.

## Textbook Exercise 4.7: Sigmoid Symmetry and Inverse

> ![Textbook Exercise 4.7](./CoursePR2026/Fig/Chapter_4/lecture_ex_4_7__textbook_ex_4_7_p221.png)
>
> *Textbook Exercise 4.7 (p. 221): Verify two basic identities for the logistic sigmoid.*

Start from

$$
\sigma(a)=\frac{1}{1+\exp(-a)}.
$$

For the symmetry identity,

$$
\sigma(-a)=\frac{1}{1+\exp(a)}.
$$

Also,

$$
1-\sigma(a)
=1-\frac{1}{1+\exp(-a)}
=\frac{\exp(-a)}{1+\exp(-a)}
=\frac{1}{1+\exp(a)}.
$$

Therefore

$$
\boxed{\sigma(-a)=1-\sigma(a).}
$$

For the inverse, let

$$
y=\sigma(a)=\frac{1}{1+\exp(-a)}.
$$

Then

$$
\frac{1}{y}=1+\exp(-a),
\qquad
\frac{1-y}{y}=\exp(-a).
$$

Taking logs gives

$$
\ln\frac{1-y}{y}=-a,
$$

so

$$
\boxed{a=\sigma^{-1}(y)=\ln\frac{y}{1-y}.}
$$

This inverse is called the **logit**. In words, logistic regression models the log-odds as a linear function:

$$
\ln\frac{p(\mathcal{C}_1\mid\mathbf{x})}{1-p(\mathcal{C}_1\mid\mathbf{x})}
=\mathbf{w}^T\boldsymbol{\phi}.
$$

## 2.2 Continuous Inputs: Gaussian Class-Conditional Densities

Suppose there are two classes and each class-conditional density is Gaussian with a shared covariance matrix:

$$
p(\mathbf{x}\mid\mathcal{C}_k)
=\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}).
$$

When the covariance matrix is shared, the quadratic terms in $\mathbf{x}$ cancel in the log-odds. Therefore the log-odds becomes linear:

$$
a(\mathbf{x})=\mathbf{w}^T\mathbf{x}+w_0,
$$

with

$$
\mathbf{w}=\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2),
$$

and

$$
w_0
=-\frac{1}{2}\boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1
+\frac{1}{2}\boldsymbol{\mu}_2^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2
+\ln\frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)}.
$$

This is a central result: **Gaussian class-conditionals with common covariance imply a linear decision boundary and a logistic posterior**.

> ![Figure 4.10](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_10__textbook_fig_4_10_p199_generative_gaussian_posterior_sigmoid.png)
>
> *Figure 4.10 (Textbook Fig. 4.10, p. 199): The left plot shows two class-conditional densities. The right plot shows the posterior probability $p(\mathcal{C}_1\mid\mathbf{x})$. The posterior surface has a sigmoid transition because the log-odds is linear in $\mathbf{x}$. This figure connects density modelling to the same kind of linear boundary used by discriminant functions.*

For $K$ Gaussian classes with the same covariance matrix, the posterior has a softmax form:

$$
p(\mathcal{C}_k\mid\mathbf{x})
=\frac{\exp(a_k(\mathbf{x}))}{\sum_j\exp(a_j(\mathbf{x}))},
$$

where

$$
a_k(\mathbf{x})=\mathbf{w}_k^T\mathbf{x}+w_{k0},
$$

$$
\mathbf{w}_k=\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k,
$$

and

$$
w_{k0}
=-\frac{1}{2}\boldsymbol{\mu}_k^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k
+\ln p(\mathcal{C}_k).
$$

If different classes have different covariance matrices $\boldsymbol{\Sigma}_k$, the quadratic terms no longer cancel. The resulting log-odds is quadratic in $\mathbf{x}$, so the decision boundaries become quadratic surfaces.

> ![Figure 4.11](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_11__textbook_fig_4_11_p200_linear_quadratic_decision_boundaries.png)
>
> *Figure 4.11 (Textbook Fig. 4.11, p. 200): The red and green classes share the same covariance, so the boundary between them is linear. Boundaries involving the blue class are quadratic because its covariance differs. This figure shows exactly how modelling assumptions about class covariance determine the shape of the decision boundary.*

## 2.3 Maximum-Likelihood Solution for Shared-Covariance Gaussians

For the two-class shared-covariance Gaussian model, the maximum-likelihood parameters have intuitive closed forms.

Let $N_1$ and $N_2$ be the number of examples in classes $\mathcal{C}_1$ and $\mathcal{C}_2$. The class prior estimate is

$$
\pi=\frac{N_1}{N},
\qquad
1-\pi=\frac{N_2}{N}.
$$

The class mean estimates are

$$
\boldsymbol{\mu}_1=\frac{1}{N_1}\sum_{n\in\mathcal{C}_1}\mathbf{x}_n,
\qquad
\boldsymbol{\mu}_2=\frac{1}{N_2}\sum_{n\in\mathcal{C}_2}\mathbf{x}_n.
$$

The shared covariance is a pooled covariance:

$$
\boldsymbol{\Sigma}
=\frac{N_1}{N}S_1+\frac{N_2}{N}S_2,
$$

where

$$
S_k=\frac{1}{N_k}\sum_{n\in\mathcal{C}_k}
(\mathbf{x}_n-\boldsymbol{\mu}_k)(\mathbf{x}_n-\boldsymbol{\mu}_k)^T.
$$

This solution is computationally simple, but it depends strongly on the assumption that Gaussian densities with a shared covariance are a good description of the data.

## 2.4 Discrete Features and Naive Bayes

Generative modelling is not limited to continuous Gaussian inputs. Suppose the input vector consists of binary features, such as whether each word appears in a document. A simple class-conditional model can use independent Bernoulli distributions:

$$
p(\mathbf{x}\mid\mathcal{C}_k)
=\prod_{i=1}^D \mu_{ki}^{x_i}(1-\mu_{ki})^{1-x_i}.
$$

This conditional independence assumption is the basis of naive Bayes. Even though the model may be a crude approximation, the log-odds becomes a linear function of the binary feature vector:

$$
a(\mathbf{x})
=\sum_i x_i\ln\frac{\mu_{1i}(1-\mu_{2i})}{\mu_{2i}(1-\mu_{1i})}
+\sum_i\ln\frac{1-\mu_{1i}}{1-\mu_{2i}}
+\ln\frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)}.
$$

So even with discrete inputs, generative assumptions often lead to linear discriminant functions in an appropriate representation.

## 2.5 Exponential-Family Class-Conditionals

The Gaussian and Bernoulli examples are special cases of a broader pattern. If the class-conditional densities belong to the exponential family and share suitable dispersion parameters, then the posterior probabilities often take the form of a generalized linear model.

The main teaching message is:

$$
\text{generative exponential-family assumptions}
\quad\Longrightarrow\quad
\text{linear logits in sufficient statistics}.
$$

This explains why linear classifiers appear so frequently. They are not only simple discriminative models; they also arise as posterior classifiers from common probabilistic assumptions.

---

# §3 Probabilistic Discriminative Models

> 📖 Textbook §4.3 Probabilistic Discriminative Models; §4.3.1-§4.3.6

## 3.1 Fixed Basis Functions

A discriminative probabilistic classifier models $p(\mathcal{C}_k\mid\mathbf{x})$ directly rather than modelling $p(\mathbf{x}\mid\mathcal{C}_k)$. As in regression, we can first transform the input using fixed basis functions:

$$
\boldsymbol{\phi}=\boldsymbol{\phi}(\mathbf{x}).
$$

Then a linear decision boundary in feature space corresponds to a nonlinear decision boundary in the original input space.

> ![Figure 4.12](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_12__textbook_fig_4_12_p204_nonlinear_basis_feature_space_classification.png)
>
> *Figure 4.12 (Textbook Fig. 4.12, p. 204): The left plot shows data in the original input space and two Gaussian basis functions. The right plot shows the transformed feature space. In feature space, the classes can be separated by a straight line. When mapped back to the original input space, that straight line becomes a nonlinear boundary. This is the same basis-function principle from Chapter 3, now used for classification.*

## 3.2 Logistic Regression

For two classes, logistic regression uses

$$
p(\mathcal{C}_1\mid\boldsymbol{\phi})=y(\boldsymbol{\phi})
=\sigma(\mathbf{w}^T\boldsymbol{\phi}),
$$

and

$$
p(\mathcal{C}_2\mid\boldsymbol{\phi})=1-y(\boldsymbol{\phi}).
$$

Although the model is called “regression,” it is a classification model because it predicts a Bernoulli probability. Its parameters are trained by maximizing the conditional likelihood, or equivalently minimizing the cross-entropy error

$$
E(\mathbf{w})
=-\sum_{n=1}^N
\{t_n\ln y_n+(1-t_n)\ln(1-y_n)\},
$$

where

$$
y_n=\sigma(\mathbf{w}^T\boldsymbol{\phi}_n).
$$

The gradient has a particularly elegant form:

$$
\nabla E(\mathbf{w})
=\sum_{n=1}^N (y_n-t_n)\boldsymbol{\phi}_n.
$$

This resembles the gradient of least squares, but the underlying likelihood is Bernoulli rather than Gaussian. There is no closed-form solution because the sigmoid makes the objective nonlinear in $\mathbf{w}$. However, the negative log-likelihood is convex for logistic regression, so standard optimization methods can find the unique global optimum, provided the data are not perfectly separable in an unregularized setting.

## Textbook Exercise 4.12: Derivative of the Sigmoid

> ![Textbook Exercise 4.12](./CoursePR2026/Fig/Chapter_4/lecture_ex_4_12__textbook_ex_4_12_p222.png)
>
> *Textbook Exercise 4.12 (p. 222): Verify the derivative of the logistic sigmoid.*

Start from

$$
\sigma(a)=\frac{1}{1+\exp(-a)}.
$$

Differentiate:

$$
\frac{d\sigma}{da}
=-\frac{1}{(1+\exp(-a))^2}\cdot(-\exp(-a))
=\frac{\exp(-a)}{(1+\exp(-a))^2}.
$$

Now write this in terms of $\sigma(a)$. Since

$$
\sigma(a)=\frac{1}{1+\exp(-a)}
$$

and

$$
1-\sigma(a)=\frac{\exp(-a)}{1+\exp(-a)},
$$

we get

$$
\boxed{
\frac{d\sigma(a)}{da}
=\sigma(a)\{1-\sigma(a)\}.
}
$$

This small identity is the reason logistic regression gradients become so clean.

## Textbook Exercise 4.13: Gradient of Logistic Cross-Entropy

> ![Textbook Exercise 4.13](./CoursePR2026/Fig/Chapter_4/lecture_ex_4_13__textbook_ex_4_13_p222.png)
>
> *Textbook Exercise 4.13 (p. 222): Use the sigmoid derivative to derive the logistic-regression gradient.*

For one data point, the cross-entropy error is

$$
E_n=-\{t_n\ln y_n+(1-t_n)\ln(1-y_n)\},
\qquad
y_n=\sigma(a_n),
\qquad
a_n=\mathbf{w}^T\boldsymbol{\phi}_n.
$$

First differentiate with respect to $y_n$:

$$
\frac{\partial E_n}{\partial y_n}
=-\frac{t_n}{y_n}+\frac{1-t_n}{1-y_n}.
$$

Then use

$$
\frac{\partial y_n}{\partial a_n}=y_n(1-y_n),
\qquad
\frac{\partial a_n}{\partial \mathbf{w}}=\boldsymbol{\phi}_n.
$$

Combining the terms,

$$
\frac{\partial E_n}{\partial a_n}
=
\left(
-\frac{t_n}{y_n}+\frac{1-t_n}{1-y_n}
\right)y_n(1-y_n)
=y_n-t_n.
$$

Therefore

$$
\boxed{
\nabla_{\mathbf{w}}E
=\sum_{n=1}^N (y_n-t_n)\boldsymbol{\phi}_n.
}
$$

The meaning is easy to remember: the update is driven by **prediction minus label**, multiplied by the feature vector. This is the same local error signal that appears at the output layer of a neural network with sigmoid or softmax cross-entropy.

A crucial warning: if the data are linearly separable, maximum likelihood can drive $\|\mathbf{w}\|\rightarrow\infty$. The model can keep increasing confidence without changing the classification boundary. Regularization or a Bayesian prior is therefore essential in separable or nearly separable problems.

## 3.3 Iterative Reweighted Least Squares

Logistic regression can be optimized by Newton-Raphson. The Hessian is

$$
\mathbf{H}
=\nabla\nabla E(\mathbf{w})
=\sum_{n=1}^N y_n(1-y_n)\boldsymbol{\phi}_n\boldsymbol{\phi}_n^T.
$$

In matrix form,

$$
\mathbf{H}=\boldsymbol{\Phi}^T\mathbf{R}\boldsymbol{\Phi},
$$

where $\mathbf{R}$ is diagonal with

$$
R_{nn}=y_n(1-y_n).
$$

The Newton update is

$$
\mathbf{w}^{\mathrm{new}}
=\mathbf{w}^{\mathrm{old}}-
\mathbf{H}^{-1}\nabla E(\mathbf{w}^{\mathrm{old}}).
$$

This can be rewritten as a weighted least-squares problem:

$$
\mathbf{w}^{\mathrm{new}}
=(\boldsymbol{\Phi}^T\mathbf{R}\boldsymbol{\Phi})^{-1}
\boldsymbol{\Phi}^T\mathbf{R}\mathbf{z},
$$

where the adjusted response is

$$
\mathbf{z}=\boldsymbol{\Phi}\mathbf{w}^{\mathrm{old}}
-\mathbf{R}^{-1}(\mathbf{y}-\mathbf{t}).
$$

This algorithm is called **iterative reweighted least squares** (IRLS). It is “least squares” because each Newton step solves a weighted least-squares problem, and “iterative” because the weights $R_{nn}$ depend on the current predictions.

A useful intuition is that points near the decision boundary have $y_n(1-y_n)$ close to its maximum value, so they receive large weight. Points that are already classified with very high confidence have smaller weights.

## 3.4 Multiclass Logistic Regression

For $K$ classes, logistic regression becomes softmax regression. Define scores

$$
a_k=\mathbf{w}_k^T\boldsymbol{\phi},
$$

and probabilities

$$
y_k(\boldsymbol{\phi})
=p(\mathcal{C}_k\mid\boldsymbol{\phi})
=\frac{\exp(a_k)}{\sum_j\exp(a_j)}.
$$

With 1-of-$K$ targets $t_{nk}$, the cross-entropy error is

$$
E(\mathbf{W})
=-\sum_{n=1}^N\sum_{k=1}^K t_{nk}\ln y_{nk}.
$$

The gradient for class $k$ is

$$
\nabla_{\mathbf{w}_k}E
=\sum_{n=1}^N (y_{nk}-t_{nk})\boldsymbol{\phi}_n.
$$

This is one of the most important formulas in classification. The model update is driven by **predicted probability minus target indicator**. The same pattern appears in neural-network classification, where backpropagation through a softmax cross-entropy layer starts with $y_{nk}-t_{nk}$.

## 3.5 Probit Regression

> **Optional depth.** Probit regression is useful mainly because Gaussian cumulative functions make some Bayesian integrals easier. For a first classification lecture, students only need to know that probit is another sigmoid-shaped probability link.

Logistic regression uses the logistic sigmoid as its activation function. Another possibility is the **probit** activation, which is the cumulative distribution function of a standard Gaussian:

$$
\Phi(a)=\int_{-\infty}^{a}\mathcal{N}(\theta\mid 0,1)\,d\theta.
$$

A useful latent-variable story is the following. Suppose the deterministic score $a=\mathbf{w}^T\boldsymbol{\phi}$ is compared with a random threshold $\theta$. The output is class $\mathcal{C}_1$ if $a\geq \theta$. Then

$$
p(\mathcal{C}_1\mid\boldsymbol{\phi})
=p(\theta\leq a)
=\int_{-\infty}^{a}p(\theta)\,d\theta.
$$

If $p(\theta)$ is Gaussian, this gives the probit model.

> ![Figure 4.13](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_13__textbook_fig_4_13_p211_density_and_cumulative_distribution.png)
>
> *Figure 4.13 (Textbook Fig. 4.13, p. 211): The blue curve is a density over a threshold variable. The red curve is the cumulative probability up to a given value. The value of the red curve at a score $a$ gives the probability that the threshold is below $a$. This explains why cumulative distribution functions naturally act as sigmoid-shaped activation functions.*

Logistic and probit regression often behave similarly. The logistic sigmoid is algebraically convenient for Bernoulli likelihoods, while the probit function is convenient for Gaussian integrals. This Gaussian-integral convenience becomes important in Bayesian logistic regression.

## 3.6 Canonical Link Functions

> **Optional depth.** This subsection is a short conceptual bridge to generalized linear models. The important reusable pattern is the gradient form "prediction minus target"; the exponential-family derivation can be skipped.

The Bernoulli likelihood with a logistic link is one example of a broader generalized linear model pattern. For suitable exponential-family target distributions, choosing the canonical link produces especially simple gradients.

The recurring gradient form is

$$
\nabla E
=\sum_n (\text{prediction}-\text{target})\times\text{feature}.
$$

For logistic regression this becomes

$$
\nabla E(\mathbf{w})
=\sum_n (y_n-t_n)\boldsymbol{\phi}_n.
$$

For softmax regression it becomes

$$
\nabla_{\mathbf{w}_k}E
=\sum_n (y_{nk}-t_{nk})\boldsymbol{\phi}_n.
$$

This shared structure is one reason generalized linear models are so central: different output distributions lead to different activation functions, but the optimization often retains a clean and interpretable form.

---

# §4 The Laplace Approximation

> 📖 Textbook §4.4 The Laplace Approximation; §4.4.1
>
> **Teaching choice.** Present Laplace approximation visually: approximate a complicated posterior by a Gaussian near its mode. The Taylor expansion is useful, but the main modern ML lesson is broader: exact Bayesian inference is often impossible, so we use approximations.

## 4.1 Why We Need an Approximation

In Bayesian linear regression, the Gaussian likelihood and Gaussian prior were conjugate, so the posterior over weights was exactly Gaussian. Logistic regression is different. The likelihood contains sigmoids, so a Gaussian prior does not lead to an exactly Gaussian posterior.

The posterior has the form

$$
p(\mathbf{w}\mid\mathbf{t})
\propto p(\mathbf{t}\mid\mathbf{w})p(\mathbf{w}),
$$

but the normalization and predictive integrals are generally not analytically tractable. The Laplace approximation provides a local Gaussian approximation around a mode of the distribution.

The strategy is:

$$
\text{find posterior mode}
\longrightarrow
\text{compute local curvature}
\longrightarrow
\text{replace posterior by a Gaussian}.
$$

## 4.2 One-Dimensional Laplace Approximation

Consider a distribution over a scalar variable $z$:

$$
p(z)=\frac{1}{Z}f(z).
$$

Let $z_0$ be a mode of $f(z)$. Expanding $\ln f(z)$ to second order around $z_0$ gives

$$
\ln f(z)
\simeq
\ln f(z_0)-\frac{A}{2}(z-z_0)^2,
$$

where

$$
A=-\left.\frac{d^2}{dz^2}\ln f(z)\right|_{z=z_0}.
$$

Exponentiating both sides gives a Gaussian approximation

$$
q(z)=\mathcal{N}(z\mid z_0,A^{-1}).
$$

> ![Figure 4.14](./CoursePR2026/Fig/Chapter_4/lecture_fig_4_14__textbook_fig_4_14_p215_laplace_approximation_distribution.png)
>
> *Figure 4.14 (Textbook Fig. 4.14, p. 215): The yellow distribution is skewed and non-Gaussian. The red curve is the Gaussian approximation obtained by matching the mode and local curvature. The right plot shows negative log-density: the Laplace approximation is equivalent to replacing the local shape near the minimum by a quadratic bowl. This is powerful, but also limited: it is a local approximation and may miss skewness, heavy tails, or multiple modes.*

## 4.3 Multivariate Laplace Approximation

For a vector variable $\mathbf{z}$, the same idea uses a second-order Taylor expansion around a mode $\mathbf{z}_0$:

$$
\ln f(\mathbf{z})
\simeq
\ln f(\mathbf{z}_0)
-\frac{1}{2}(\mathbf{z}-\mathbf{z}_0)^T
\mathbf{A}
(\mathbf{z}-\mathbf{z}_0),
$$

where

$$
\mathbf{A}
=-\left.\nabla\nabla\ln f(\mathbf{z})\right|_{\mathbf{z}=\mathbf{z}_0}.
$$

The approximation is

$$
q(\mathbf{z})=\mathcal{N}(\mathbf{z}\mid\mathbf{z}_0,\mathbf{A}^{-1}).
$$

The matrix $\mathbf{A}$ must be positive definite. This means the mode must be a genuine local maximum of $\ln f(\mathbf{z})$ rather than a saddle point.

## 4.4 Approximating the Evidence and BIC

The Laplace approximation also gives an approximation to the normalization constant:

$$
Z=\int f(\mathbf{z})\,d\mathbf{z}
\simeq
f(\mathbf{z}_0)(2\pi)^{M/2}|\mathbf{A}|^{-1/2}.
$$

In Bayesian model comparison, this kind of approximation is used for the model evidence. If $\mathbf{w}_{\mathrm{MAP}}$ is the posterior mode and $\mathbf{A}$ is the negative Hessian of the log posterior at that mode, then

$$
\ln p(\mathcal{D}\mid\mathcal{M})
\simeq
\ln p(\mathcal{D}\mid\mathbf{w}_{\mathrm{MAP}},\mathcal{M})
+\ln p(\mathbf{w}_{\mathrm{MAP}}\mid\mathcal{M})
+\frac{M}{2}\ln(2\pi)
-\frac{1}{2}\ln|\mathbf{A}|.
$$

A rough large-data simplification leads to the Bayesian Information Criterion (BIC):

$$
\mathrm{BIC}
=\ln p(\mathcal{D}\mid\mathbf{w}_{\mathrm{ML}})
-\frac{M}{2}\ln N.
$$

The first term rewards data fit. The second term penalizes model complexity. This is the same conceptual trade-off encountered in Chapter 3 evidence approximation: a model should explain the data well without wasting excessive parameter volume.

---

# §5 Bayesian Logistic Regression

> 📖 Textbook §4.5 Bayesian Logistic Regression; §4.5.1-§4.5.2
>
> **Teaching choice.** This section is advanced. Keep the main idea: ordinary logistic regression gives one set of weights, while Bayesian logistic regression averages over plausible weights and therefore avoids some overconfidence. The predictive approximation with $\kappa(\sigma^2)$ can be shown as a result rather than derived.

## 5.1 Posterior Over Logistic-Regression Weights

In Bayesian logistic regression, we place a prior over the weights, for example

$$
p(\mathbf{w})=\mathcal{N}(\mathbf{w}\mid\mathbf{0},\alpha^{-1}\mathbf{I}).
$$

The likelihood is Bernoulli:

$$
p(\mathbf{t}\mid\mathbf{w})
=\prod_{n=1}^N y_n^{t_n}(1-y_n)^{1-t_n},
$$

where

$$
y_n=\sigma(\mathbf{w}^T\boldsymbol{\phi}_n).
$$

The posterior is

$$
p(\mathbf{w}\mid\mathbf{t})
\propto
p(\mathbf{t}\mid\mathbf{w})p(\mathbf{w}).
$$

Unlike Bayesian linear regression, this posterior is not exactly Gaussian. We therefore approximate it using Laplace's method.

## 5.2 Laplace Approximation to the Parameter Posterior

First find the posterior mode

$$
\mathbf{w}_{\mathrm{MAP}}
=\operatorname*{arg\,max}_{\mathbf{w}}p(\mathbf{w}\mid\mathbf{t}).
$$

Equivalently, minimize the regularized negative log-likelihood

$$
E(\mathbf{w})
=-\ln p(\mathbf{t}\mid\mathbf{w})-\ln p(\mathbf{w}).
$$

For an isotropic Gaussian prior, this is cross-entropy plus weight decay:

$$
E(\mathbf{w})
= -\sum_n\{t_n\ln y_n+(1-t_n)\ln(1-y_n)\}
+\frac{\alpha}{2}\mathbf{w}^T\mathbf{w}.
$$

The Laplace posterior is

$$
q(\mathbf{w})
=\mathcal{N}(\mathbf{w}\mid\mathbf{w}_{\mathrm{MAP}},\mathbf{S}_N),
$$

where

$$
\mathbf{S}_N=\mathbf{A}^{-1},
$$

and

$$
\mathbf{A}
=\boldsymbol{\Phi}^T\mathbf{R}\boldsymbol{\Phi}+\alpha\mathbf{I}.
$$

Here $\mathbf{R}$ is evaluated at $\mathbf{w}_{\mathrm{MAP}}$.

This expression has an intuitive interpretation. The likelihood contributes curvature from the data, $\boldsymbol{\Phi}^T\mathbf{R}\boldsymbol{\Phi}$, and the prior contributes curvature $\alpha\mathbf{I}$. Stronger prior precision means smaller posterior variance.

## 5.3 Predictive Distribution

For a new input $\mathbf{x}$ with feature vector $\boldsymbol{\phi}$, the Bayesian predictive probability is

$$
p(\mathcal{C}_1\mid\boldsymbol{\phi},\mathbf{t})
=\int \sigma(\mathbf{w}^T\boldsymbol{\phi})
q(\mathbf{w})\,d\mathbf{w}.
$$

The integrand depends on $\mathbf{w}$ only through the scalar

$$
a=\mathbf{w}^T\boldsymbol{\phi}.
$$

Because $q(\mathbf{w})$ is Gaussian, $a$ is also Gaussian:

$$
a\sim\mathcal{N}(\mu_a,\sigma_a^2),
$$

with

$$
\mu_a=\mathbf{w}_{\mathrm{MAP}}^T\boldsymbol{\phi},
\qquad
\sigma_a^2=\boldsymbol{\phi}^T\mathbf{S}_N\boldsymbol{\phi}.
$$

Therefore the predictive integral becomes one-dimensional:

$$
p(\mathcal{C}_1\mid\boldsymbol{\phi},\mathbf{t})
=\int \sigma(a)\mathcal{N}(a\mid\mu_a,\sigma_a^2)\,da.
$$

This integral still has no exact closed form for the logistic sigmoid, but it can be approximated using the close relationship between the logistic sigmoid and the probit function:

$$
\int \sigma(a)\mathcal{N}(a\mid\mu_a,\sigma_a^2)\,da
\simeq
\sigma\left(\kappa(\sigma_a^2)\mu_a\right),
$$

where

$$
\kappa(\sigma^2)=\left(1+\frac{\pi\sigma^2}{8}\right)^{-1/2}.
$$

Since $0<\kappa(\sigma^2)\leq 1$, uncertainty in the weights shrinks the effective logit toward zero. Thus Bayesian averaging makes predictions less extreme near regions of high parameter uncertainty.

## 5.4 What Bayesian Logistic Regression Adds

Standard logistic regression returns one weight vector. Bayesian logistic regression returns an approximate distribution over weight vectors. This changes the interpretation of classification.

| Standard Logistic Regression | Bayesian Logistic Regression |
|------------------------------|------------------------------|
| Uses $\mathbf{w}_{\mathrm{ML}}$ or $\mathbf{w}_{\mathrm{MAP}}$. | Integrates over plausible $\mathbf{w}$. |
| Confidence depends only on distance from boundary. | Confidence also depends on posterior uncertainty. |
| Can become overconfident in sparse regions. | Predictive probabilities are moderated by uncertainty. |
| Easier optimization. | Requires approximation such as Laplace. |

This section is an important bridge to later chapters. Exact Bayesian inference is often impossible, so we approximate it. Chapter 4 introduces Laplace approximation; later chapters introduce variational inference, expectation propagation, and sampling.

---

# §6 Chapter Summary, Figure Checklist, Exercises, and Teaching Flow

## 6.1 Chapter Summary

Chapter 4 develops linear classification from direct decision rules to probabilistic Bayesian prediction.

First, from the geometric perspective, it studies hyperplane decision boundaries, signed distances, multiclass decision regions, least-squares classifiers, Fisher projections, and the perceptron algorithm.

Second, from the generative probabilistic perspective, it shows that modelling $p(\mathbf{x}\mid\mathcal{C}_k)$ and $p(\mathcal{C}_k)$ can produce logistic sigmoid or softmax posterior probabilities. Shared-covariance Gaussian models lead to linear boundaries, while different covariances lead to quadratic boundaries.

Third, from the discriminative probabilistic perspective, it introduces logistic regression, softmax regression, cross-entropy, IRLS, probit regression, and canonical link functions.

Finally, from the Bayesian perspective, it introduces the Laplace approximation and applies it to logistic regression, producing approximate posterior uncertainty and moderated predictive probabilities.

The chapter is conceptually important because many later models preserve the same structure while changing one component:

| Later Model | What Changes? | What Remains from Chapter 4? |
|-------------|---------------|------------------------------|
| Neural networks | Basis functions become learned hidden representations. | Linear logits, sigmoid/softmax outputs, cross-entropy gradients. |
| Support vector machines | Loss becomes margin-based hinge loss. | Hyperplane decision boundaries and separability geometry. |
| Kernel methods | Feature space becomes implicit. | Linear classifier in transformed feature space. |
| Bayesian neural networks | Weight posterior becomes high-dimensional and non-Gaussian. | Need for approximate Bayesian classification. |
| Variational inference | Approximation is optimized globally rather than by local curvature. | Same need to approximate intractable posteriors. |

## 6.2 Figure Checklist

All figures used in this lecture are screenshots/crops from the uploaded textbook PDF. Each filename records both the lecture figure number and the original textbook figure number.

| Lecture Figure | Textbook Figure | Topic | File |
|----------------|-----------------|-------|------|
| 4.1 | 4.1 | Geometry of a two-class linear discriminant | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_1__textbook_fig_4_1_p182_linear_discriminant_geometry.png` |
| 4.2 | 4.2 | One-versus-rest and one-versus-one ambiguity | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_2__textbook_fig_4_2_p183_one_vs_rest_one_vs_one_ambiguity.png` |
| 4.3 | 4.3 | Convex multiclass linear decision regions | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_3__textbook_fig_4_3_p184_multiclass_linear_regions_convexity.png` |
| 4.4 | 4.4 | Least-squares sensitivity to outliers | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_4__textbook_fig_4_4_p186_least_squares_outlier_sensitivity.png` |
| 4.5 | 4.5 | Least-squares failure in multiclass classification | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_5__textbook_fig_4_5_p187_least_squares_multiclass_failure.png` |
| 4.6 | 4.6 | Fisher projection and class separation | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_6__textbook_fig_4_6_p188_fisher_projection_class_separation.png` |
| 4.7 | 4.7 | Perceptron learning updates and convergence | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_7__textbook_fig_4_7_p195_perceptron_learning_convergence.png` |
| 4.8 | 4.8 | Mark 1 perceptron hardware | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_8__textbook_fig_4_8_p196_mark_1_perceptron_hardware.png` |
| 4.9 | 4.9 | Logistic sigmoid and scaled probit function | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_9__textbook_fig_4_9_p197_logistic_sigmoid_and_probit.png` |
| 4.10 | 4.10 | Gaussian class-conditionals and sigmoid posterior | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_10__textbook_fig_4_10_p199_generative_gaussian_posterior_sigmoid.png` |
| 4.11 | 4.11 | Linear and quadratic boundaries from Gaussian class-conditionals | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_11__textbook_fig_4_11_p200_linear_quadratic_decision_boundaries.png` |
| 4.12 | 4.12 | Nonlinear basis functions and feature-space classification | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_12__textbook_fig_4_12_p204_nonlinear_basis_feature_space_classification.png` |
| 4.13 | 4.13 | Density and cumulative distribution function | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_13__textbook_fig_4_13_p211_density_and_cumulative_distribution.png` |
| 4.14 | 4.14 | Laplace approximation to a non-Gaussian distribution | `./CoursePR2026/Fig/Chapter_4/lecture_fig_4_14__textbook_fig_4_14_p215_laplace_approximation_distribution.png` |

## 6.3 Exercise Checklist

| Lecture Exercise | Textbook Exercise | Topic | File |
|------------------|-------------------|-------|------|
| 4.7 | 4.7 | Sigmoid symmetry and inverse logit | `./CoursePR2026/Fig/Chapter_4/lecture_ex_4_7__textbook_ex_4_7_p221.png` |
| 4.12 | 4.12 | Derivative of the logistic sigmoid | `./CoursePR2026/Fig/Chapter_4/lecture_ex_4_12__textbook_ex_4_12_p222.png` |
| 4.13 | 4.13 | Gradient of logistic cross-entropy | `./CoursePR2026/Fig/Chapter_4/lecture_ex_4_13__textbook_ex_4_13_p222.png` |

## 6.4 Suggested Teaching Flow

A practical lecture sequence is:

1. Begin with the difference between regression targets and classification labels.
2. Introduce decision regions and the two-class linear discriminant.
3. Use Figure 4.1 to explain the geometry of $\mathbf{w}$, $w_0$, and signed distance.
4. Discuss multiclass classification and use Figure 4.2 to show why naive binary combinations can create ambiguity.
5. Use Figure 4.3 to show that joint multiclass linear discriminants produce convex regions.
6. Derive least squares for classification and immediately show Figures 4.4-4.5 as warnings.
7. Introduce Fisher's criterion and use Figure 4.6 to explain why within-class scatter matters.
8. Present the perceptron update rule and use Figures 4.7-4.8 for algorithmic and historical context.
9. Transition to probabilities by deriving posterior log-odds from Bayes' theorem.
10. Use Figure 4.9 and Exercise 4.7 to introduce sigmoid probabilities and logits.
11. Derive the shared-covariance Gaussian classifier and use Figures 4.10-4.11 to connect covariance assumptions to boundary shape.
12. Use Figure 4.12 to explain nonlinear basis functions and feature-space separability.
13. Derive logistic regression and cross-entropy.
14. Work through Exercises 4.12 and 4.13 so students see why the gradient becomes $\sum_n(y_n-t_n)\boldsymbol{\phi}_n$.
15. Explain IRLS as Newton's method rewritten as weighted least squares, but keep implementation details brief.
16. Extend to softmax regression and emphasize the $y_{nk}-t_{nk}$ gradient pattern.
17. Treat probit regression and canonical links as optional context.
18. Introduce the Laplace approximation visually with Figure 4.14.
19. Finish Bayesian logistic regression as an intuition: posterior uncertainty makes probabilities less overconfident.

## 6.5 Key Equations to Put on the Board

The following equations are the minimum board set for this chapter.

### Two-class linear discriminant

$$
y(\mathbf{x})=\mathbf{w}^T\mathbf{x}+w_0,
\qquad
r=\frac{y(\mathbf{x})}{\|\mathbf{w}\|}.
$$

### Multiclass linear decision rule

$$
y_k(\mathbf{x})=\mathbf{w}_k^T\mathbf{x}+w_{k0},
\qquad
\widehat{k}=\operatorname*{arg\,max}_{k}y_k(\mathbf{x}).
$$

### Least-squares classifier

$$
\mathbf{W}_{\mathrm{LS}}
=(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{T}.
$$

### Fisher criterion

$$
J(\mathbf{w})
=\frac{\mathbf{w}^T S_B\mathbf{w}}{\mathbf{w}^T S_W\mathbf{w}},
\qquad
\mathbf{w}\propto S_W^{-1}(\mathbf{m}_2-\mathbf{m}_1).
$$

### Perceptron update

$$
\mathbf{w}^{(\tau+1)}
=\mathbf{w}^{(\tau)}+\eta\boldsymbol{\phi}_n t_n.
$$

### Logistic sigmoid

$$
\sigma(a)=\frac{1}{1+\exp(-a)},
\qquad
p(\mathcal{C}_1\mid\mathbf{x})=\sigma(a(\mathbf{x})).
$$

### Softmax

$$
p(\mathcal{C}_k\mid\boldsymbol{\phi})
=\frac{\exp(\mathbf{w}_k^T\boldsymbol{\phi})}{\sum_j\exp(\mathbf{w}_j^T\boldsymbol{\phi})}.
$$

### Logistic-regression cross-entropy and gradient

$$
E(\mathbf{w})
=-\sum_n\{t_n\ln y_n+(1-t_n)\ln(1-y_n)\},
\qquad
\nabla E=\sum_n(y_n-t_n)\boldsymbol{\phi}_n.
$$

### IRLS Hessian

$$
\mathbf{H}=\boldsymbol{\Phi}^T\mathbf{R}\boldsymbol{\Phi},
\qquad
R_{nn}=y_n(1-y_n).
$$

### Laplace approximation

$$
q(\mathbf{z})=\mathcal{N}(\mathbf{z}\mid\mathbf{z}_0,\mathbf{A}^{-1}),
\qquad
\mathbf{A}=-\left.\nabla\nabla\ln f(\mathbf{z})\right|_{\mathbf{z}=\mathbf{z}_0}.
$$

### Bayesian logistic predictive approximation

$$
p(\mathcal{C}_1\mid\boldsymbol{\phi},\mathbf{t})
\simeq
\sigma\left(\kappa(\sigma_a^2)\mu_a\right),
$$

where

$$
\mu_a=\mathbf{w}_{\mathrm{MAP}}^T\boldsymbol{\phi},
\qquad
\sigma_a^2=\boldsymbol{\phi}^T\mathbf{S}_N\boldsymbol{\phi},
\qquad
\kappa(\sigma^2)=\left(1+\frac{\pi\sigma^2}{8}\right)^{-1/2}.
$$
