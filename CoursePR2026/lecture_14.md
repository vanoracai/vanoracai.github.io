---
layout: course
title: PRML Lecture 14
---

# Pattern Recognition and Machine Learning
## Chapter 14: Combining Models: Trees, Ensembles, Boosting, and Mixtures of Experts

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 14 Combining Models (§14.1-§14.5)  
> Teaching emphasis: practical ensemble learning for modern machine learning, especially tabular data

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Why Combine Multiple Models?](#1-why-combine-multiple-models)
3. [§2 Bayesian Model Averaging: A Brief Distinction](#2-bayesian-model-averaging-a-brief-distinction)
4. [§3 Committees, Bootstrap, and Bagging](#3-committees-bootstrap-and-bagging)
5. [§4 Decision Trees](#4-decision-trees)
6. [§5 Random Forests](#5-random-forests)
7. [§6 Boosting and AdaBoost](#6-boosting-and-adaboost)
8. [§7 Gradient Boosting](#7-gradient-boosting)
9. [§8 Conditional Mixtures and Mixtures of Experts](#8-conditional-mixtures-and-mixtures-of-experts)
10. [§9 Practical Model Selection for Tabular Data](#9-practical-model-selection-for-tabular-data)
11. [§10 Worked Examples and Textbook Exercises](#10-worked-examples-and-textbook-exercises)
12. [§11 Chapter Summary and Course Wrap-Up](#11-chapter-summary-and-course-wrap-up)

---

## Notation and Variable Definitions

This chapter uses several different kinds of model combinations. The notation below helps us keep them separate.

### Data and Individual Models

| Symbol | Definition |
|--------|------------|
| $\mathbf{x}_n$ | Input vector for training example $n$. |
| $t_n$ | Target for training example $n$. For AdaBoost, $t_n\in\{-1,+1\}$. |
| $N$ | Number of training examples. |
| $M$ | Number of models, trees, or boosting stages in an ensemble. |
| $y_m(\mathbf{x})$ | Prediction of model $m$. |
| $h_m(\mathbf{x})$ | A weak learner or regression tree added at boosting stage $m$. |
| $F_m(\mathbf{x})$ | Additive ensemble after $m$ boosting stages. |

### Ensemble Weights

| Symbol | Definition |
|--------|------------|
| $w_n^{(m)}$ | Weight assigned to data point $n$ while training boosting stage $m$. |
| $\epsilon_m$ | Weighted training error of weak learner $m$. |
| $\alpha_m$ | Weight assigned to weak learner $m$ in the final AdaBoost vote. |
| $\eta$ | Learning rate or shrinkage factor in gradient boosting. |

### Decision Trees

| Symbol | Definition |
|--------|------------|
| $R_\tau$ | Input-space region associated with leaf node $\tau$. |
| $N_\tau$ | Number of training examples in leaf $\tau$. |
| $c_\tau$ | Constant prediction made in regression leaf $\tau$. |
| $p_{\tau k}$ | Proportion of examples from class $k$ in leaf $\tau$. |
| $|T|$ | Number of leaf nodes in tree $T$. |

### Mixture Models and Experts

| Symbol | Definition |
|--------|------------|
| $K$ | Number of mixture components or experts. |
| $p_k(t\mid\mathbf{x})$ | Predictive distribution of expert $k$. |
| $\pi_k(\mathbf{x})$ | Input-dependent gating weight for expert $k$. |
| $\sum_k\pi_k(\mathbf{x})=1$ | Gating weights form a probability distribution over experts. |

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch.14 opening; §14.1-§14.5

## 0.1 What This Chapter Is Really About

Earlier chapters mostly asked:

> How can we design and train one model?

This chapter asks a different question:

> How can several imperfect models work together to produce a better prediction?

There are three main answers.

| Combination Strategy | Core Idea | Main Example |
|----------------------|-----------|--------------|
| **Average many models** | Reduce instability and variance by averaging diverse predictions. | Bagging and random forests |
| **Correct mistakes sequentially** | Add new models that focus on errors made by the current ensemble. | AdaBoost and gradient boosting |
| **Route inputs to specialized models** | Let different experts handle different parts of the input space. | Decision trees and mixtures of experts |

These strategies are still central to practical machine learning.

A particularly important lesson is:

> **For many structured and tabular data problems, tree-based ensemble methods remain among the most useful model families.**

Deep neural networks dominate many image, audio, language, and large-scale representation-learning tasks. However, tabular data often contain heterogeneous feature types, nonlinear thresholds, missing values, and irregular feature interactions. Decision trees and boosted trees naturally model many of these patterns.

## 0.2 Learning Outcomes

After this lecture, students should be able to:

1. Explain why averaging multiple unstable models can improve generalization.
2. Distinguish Bayesian model averaging from ordinary model combination.
3. Describe bootstrap aggregation and the intuition behind random forests.
4. Build and interpret a small decision tree.
5. Compute regression-leaf predictions and classification impurity measures.
6. Explain AdaBoost as sequential reweighting of difficult examples.
7. Explain gradient boosting as fitting residuals or negative gradients.
8. Compare bagging, random forests, AdaBoost, and gradient boosting.
9. Explain a mixture of experts as input-dependent soft model selection.
10. Choose a sensible tree-based baseline for a tabular machine-learning problem.

## 0.3 What We Will Not Emphasize

To keep the lecture practical and accessible, we will **not** develop the following topics in detail:

- the complete proof that AdaBoost minimizes exponential loss;
- all mathematical properties of exponential loss;
- the full EM algorithm for conditional mixtures;
- detailed updates for mixtures of linear or logistic regression models;
- the complete hierarchical mixture-of-experts learning algorithm.

These topics are mathematically valuable, but they are not essential for understanding the practical ensemble-learning ideas emphasized here.

## 0.4 A Unifying View

All methods in this chapter can be written informally as

$$
\text{final prediction}
=
\text{combine several simpler predictions}.
$$

The key question is how the combination is performed.

### Fixed average

$$
y(\mathbf{x})=\frac{1}{M}\sum_{m=1}^{M}y_m(\mathbf{x}).
$$

### Weighted average or vote

$$
y(\mathbf{x})=\sum_{m=1}^{M}\alpha_m y_m(\mathbf{x}).
$$

### Input-dependent combination

$$
p(t\mid\mathbf{x})
=
\sum_{k=1}^{K}\pi_k(\mathbf{x})p_k(t\mid\mathbf{x}).
$$

The difference between these formulas is not cosmetic. It determines how models specialize, how training is performed, and what kinds of errors the ensemble can correct.

---

# §1 Why Combine Multiple Models?

> 📖 Textbook Ch.14 opening, pp. 653-654

## 1.1 One Model Can Be Unstable

Suppose we train a deep decision tree on a data set. If we remove a few training examples and retrain, the first split may change. Once the first split changes, many later branches can also change.

This means a tree can have:

- low training error;
- strong ability to represent nonlinear patterns;
- but high sensitivity to the exact training sample.

This sensitivity is called **variance**.

An ensemble can reduce this problem by training multiple versions of the model and averaging them.

The intuition is similar to repeated measurement:

- one noisy measurement may be unreliable;
- several independent measurements can be averaged;
- positive and negative noise partly cancel.

## 1.2 Diversity Is Essential

Averaging identical models does nothing.

If

$$
y_1(\mathbf{x})=y_2(\mathbf{x})=\cdots=y_M(\mathbf{x}),
$$

then

$$
\frac{1}{M}\sum_{m=1}^{M}y_m(\mathbf{x})=y_1(\mathbf{x}).
$$

Therefore, a useful ensemble needs two properties:

1. Each member should be reasonably accurate.
2. Different members should make at least partly different errors.

This is often summarized as the **accuracy-diversity trade-off**.

A weak model that is completely random is diverse but not useful. A group of strong but identical models is accurate but gains nothing from averaging. Good ensemble methods create controlled diversity without destroying predictive quality.

## 1.3 Three Ways to Create Diversity

### Data diversity

Train each model on a different bootstrap sample.

This gives **bagging**.

### Feature diversity

At each tree split, consider only a random subset of features.

This helps create a **random forest**.

### Error-focused diversity

Train each new model to emphasize examples or residuals that the current ensemble handles poorly.

This gives **boosting**.

## 1.4 Averaging, Sequential Correction, and Routing

A useful high-level comparison is:

| Method | Models Trained | Relationship Between Models | Final Combination |
|--------|----------------|-----------------------------|------------------|
| Bagging | In parallel | Mostly independent | Average or majority vote |
| Random forest | In parallel | Bootstrap data + random features | Average or majority vote |
| AdaBoost | Sequentially | Next learner focuses on misclassified points | Weighted vote |
| Gradient boosting | Sequentially | Next learner fits current residual or negative gradient | Additive sum |
| Decision tree | One hierarchical model | Each split routes an example | One leaf prediction |
| Mixture of experts | Jointly or iteratively | Experts specialize through a gate | Soft input-dependent weighted sum |

---

# §2 Bayesian Model Averaging: A Brief Distinction

> 📖 Textbook §14.1

## 2.1 Bayesian Model Averaging

Suppose we have several candidate models indexed by $h$.

For example:

- model 1 is linear regression;
- model 2 is polynomial regression;
- model 3 is a Gaussian process.

Bayesian model averaging computes the predictive distribution

$$
p(t\mid\mathbf{x},D)
=
\sum_h p(t\mid\mathbf{x},D,h)p(h\mid D).
$$

Here:

- $p(t\mid\mathbf{x},D,h)$ is the prediction under model $h$;
- $p(h\mid D)$ is the posterior probability that model $h$ is the correct model.

The sum represents uncertainty about which **whole model** generated the complete data set.

As the data set becomes large, the posterior $p(h\mid D)$ may become concentrated on one model.

## 2.2 Model Combination Is Different

A mixture model has the form

$$
p(t\mid\mathbf{x})
=
\sum_{k=1}^{K}\pi_k(\mathbf{x})p_k(t\mid\mathbf{x}).
$$

Here, different components can explain different data points or different regions of the input space.

This is not merely uncertainty about which single model is globally correct. The model assumes that several components may all be useful.

## 2.3 A Simple Analogy

### Bayesian model averaging

> We are uncertain whether the entire data set should be modeled by model A or model B.

### Mixture or ensemble model

> Model A may be useful for some examples, while model B may be useful for other examples.

This distinction is conceptually important, but for this practical lecture we will spend most of our time on ensembles, trees, and boosting.

---

# §3 Committees, Bootstrap, and Bagging

> 📖 Textbook §14.2

## 3.1 Committee Prediction

The simplest ensemble is a committee that averages $M$ predictions:

$$
y_{\mathrm{COM}}(\mathbf{x})
=
\frac{1}{M}\sum_{m=1}^{M}y_m(\mathbf{x}).
$$

For classification, common alternatives are:

- majority vote over predicted labels;
- average predicted class probabilities, then choose the largest probability.

Averaging probabilities is usually more informative than voting only on hard labels because it preserves confidence information.

## 3.2 Bootstrap Sampling

In practice, we usually have only one training set. To train different ensemble members, we create multiple **bootstrap samples**.

Suppose the original data set contains $N$ examples.

To create one bootstrap sample:

1. Draw one example uniformly from the original data.
2. Replace it before drawing again.
3. Repeat until $N$ examples have been drawn.

Because sampling is performed with replacement:

- some original examples appear several times;
- some do not appear at all.

Each bootstrap sample is different, so models trained on them are also different.

## 3.3 Bagging

**Bagging** means **bootstrap aggregation**.

The procedure is:

1. Create $M$ bootstrap data sets.
2. Train one model on each data set.
3. Average the predictions.

For regression:

$$
y_{\mathrm{bag}}(\mathbf{x})
=
\frac{1}{M}\sum_{m=1}^{M}y_m(\mathbf{x}).
$$

For classification:

$$
\widehat{C}(\mathbf{x})
=
\operatorname{mode}\left\{
\widehat{C}_1(\mathbf{x}),\ldots,\widehat{C}_M(\mathbf{x})
\right\}.
$$

## 3.4 Why Averaging Reduces Variance

Suppose the true regression function is $h(\mathbf{x})$. Model $m$ predicts

$$
y_m(\mathbf{x})=h(\mathbf{x})+\varepsilon_m(\mathbf{x}).
$$

The committee prediction is

$$
y_{\mathrm{COM}}(\mathbf{x})
=
h(\mathbf{x})+rac{1}{M}\sum_{m=1}^{M}\varepsilon_m(\mathbf{x}).
$$

If errors have mean zero and are uncorrelated, then the variance of the average error is

$$
\operatorname{var}\left[
\frac{1}{M}\sum_{m=1}^{M}\varepsilon_m
\right]
=
\frac{1}{M^2}\sum_{m=1}^{M}\operatorname{var}[\varepsilon_m].
$$

If every model has error variance $\sigma^2$, then

$$
\operatorname{var}(\text{average error})
=
\frac{M\sigma^2}{M^2}
=
\frac{\sigma^2}{M}.
$$

This is the ideal $1/M$ reduction described in the textbook.

## 3.5 Correlated Errors: The More Realistic Case

In reality, ensemble members are trained from related data and often make correlated errors.

Suppose each pair of model errors has correlation $\rho$. Then

$$
\operatorname{var}(\text{average error})
=
\sigma^2\left[
\rho+\frac{1-\rho}{M}
\right].
$$

This formula is very useful.

### If $\rho=0$

$$
\operatorname{var}=\frac{\sigma^2}{M}.
$$

We obtain the ideal reduction.

### If $\rho=1$

$$
\operatorname{var}=\sigma^2.
$$

All models make the same errors, so averaging does nothing.

### If $0<\rho<1$

Adding more models helps, but the benefit eventually saturates near $\rho\sigma^2$.

This explains why random forests deliberately reduce correlation between trees.

## 3.6 A General Guarantee from Convexity

For squared error, averaging predictions cannot be worse than the average error of the individual models.

For one target $t$,

$$
\left(
\frac{1}{M}\sum_{m=1}^{M}y_m-t
\right)^2
\leq
\frac{1}{M}\sum_{m=1}^{M}(y_m-t)^2.
$$

This follows from Jensen's inequality because the squared function is convex.

The statement does **not** mean that the ensemble is always better than the best individual member. It means that its error is no larger than the **average** member error.

## Textbook Exercise 14.2: Ideal Committee Error Reduction

> ![Textbook Exercise 14.2](./CoursePR2026/Fig/Chapter_14/lecture_ex_14_2__textbook_ex_14_2__p674.png)
>
> *Textbook Exercise 14.2 (p. 674): Derive the ideal $1/M$ committee-error reduction when individual errors are zero-mean and uncorrelated.*

Let

$$
y_m(\mathbf{x})=h(\mathbf{x})+\varepsilon_m(\mathbf{x}).
$$

Then

$$
y_{\mathrm{COM}}(\mathbf{x})-h(\mathbf{x})
=
\frac{1}{M}\sum_{m=1}^{M}\varepsilon_m(\mathbf{x}).
$$

The expected squared error is

$$
E_{\mathrm{COM}}
=
\mathbb{E}\left[
\left(
\frac{1}{M}\sum_m\varepsilon_m
\right)^2
\right].
$$

Expand the square:

$$
E_{\mathrm{COM}}
=
\frac{1}{M^2}
\mathbb{E}\left[
\sum_m\varepsilon_m^2
+
\sum_{m\neq \ell}\varepsilon_m\varepsilon_\ell
\right].
$$

If the errors are uncorrelated, then

$$
\mathbb{E}[\varepsilon_m\varepsilon_\ell]=0,
\qquad m\neq \ell.
$$

Therefore,

$$
E_{\mathrm{COM}}
=
\frac{1}{M^2}\sum_m\mathbb{E}[\varepsilon_m^2].
$$

The average individual error is

$$
E_{\mathrm{AV}}
=
\frac{1}{M}\sum_m\mathbb{E}[\varepsilon_m^2].
$$

Hence

$$
\boxed{
E_{\mathrm{COM}}=\frac{1}{M}E_{\mathrm{AV}}
}.
$$

The critical assumption is not the algebra. It is the assumption of low error correlation.

---

# §4 Decision Trees

> 📖 Textbook §14.4

## 4.1 The Core Idea

A decision tree repeatedly asks simple questions such as

$$
x_j\leq \theta?
$$

Each question splits the current group of data into two child groups.

A new input travels from the root to a leaf:

1. evaluate the first split;
2. choose the left or right branch;
3. evaluate the next split;
4. continue until reaching a leaf;
5. use the prediction stored in that leaf.

Decision trees perform **hard routing**: each input reaches exactly one leaf.

> ![Figure 14.5](./CoursePR2026/Fig/Chapter_14/lecture_fig_14_5__textbook_fig_14_5__p663.png)
>
> *Figure 14.5 (Textbook Fig. 14.5, p. 663): Axis-aligned splits partition a two-dimensional input space into five regions. Each region can store a separate prediction.*

> ![Figure 14.6](./CoursePR2026/Fig/Chapter_14/lecture_fig_14_6__textbook_fig_14_6__p664.png)
>
> *Figure 14.6 (Textbook Fig. 14.6, p. 664): Binary tree corresponding to the regions in Figure 14.5. An input follows one path from the root to a leaf.*

## 4.2 Tree Terminology

| Term | Meaning |
|------|---------|
| Root | Top node containing the full training set |
| Internal node | A node that applies a split rule |
| Branch | Outcome of a split |
| Leaf | Final region that stores a prediction |
| Depth | Number of splits along the longest root-to-leaf path |
| Subtree | A node together with all descendants below it |

## 4.3 Regression Trees

In regression, each leaf usually predicts a constant.

Suppose leaf $R_\tau$ contains target values

$$
t_1,t_2,\ldots,t_{N_\tau}.
$$

We choose a constant $c_\tau$ to minimize

$$
Q(c_\tau)
=
\sum_{n:\mathbf{x}_n\in R_\tau}(t_n-c_\tau)^2.
$$

Differentiate:

$$
\frac{dQ}{dc_\tau}
=
2\sum_{n:\mathbf{x}_n\in R_\tau}(c_\tau-t_n).
$$

Set the derivative to zero:

$$
\sum_n(c_\tau-t_n)=0.
$$

Therefore,

$$
N_\tau c_\tau=\sum_n t_n,
$$

and

$$
\boxed{
c_\tau
=
\frac{1}{N_\tau}
\sum_{n:\mathbf{x}_n\in R_\tau}t_n
}.
$$

So a regression-tree leaf predicts the average target value of the training examples in that leaf.

## Textbook Exercise 14.10: Why a Regression Leaf Predicts the Mean

> ![Textbook Exercise 14.10](./CoursePR2026/Fig/Chapter_14/lecture_ex_14_10__textbook_ex_14_10__p675.png)
>
> *Textbook Exercise 14.10 (p. 675): Verify that the mean minimizes the sum-of-squares error for a constant prediction.*

This is the same derivation as above. It also connects decision trees back to Chapter 1:

> squared error is minimized by a conditional mean.

A leaf approximates a local conditional mean using the samples that reach that leaf.

## 4.4 Classification Trees

In classification, each leaf stores class proportions.

For class $k$ in leaf $\tau$,

$$
p_{\tau k}
=
\frac{\text{number of class-}k\text{ examples in leaf }\tau}
{N_\tau}.
$$

The predicted class is often

$$
\widehat{C}_\tau
=
\arg\max_k p_{\tau k}.
$$

The values $p_{\tau k}$ can also be used as estimated class probabilities, although probabilities from a single deep tree are often poorly calibrated.

## 4.5 Measuring Node Impurity

A good leaf should be relatively pure: most examples should belong to one class.

Two common impurity measures are entropy and the Gini index.

### Entropy impurity

$$
H_\tau
=
-\sum_{k=1}^{K}p_{\tau k}\ln p_{\tau k}.
$$

### Gini impurity

$$
G_\tau
=
\sum_{k=1}^{K}p_{\tau k}(1-p_{\tau k})
=
1-\sum_{k=1}^{K}p_{\tau k}^2.
$$

For a binary node with class probability $p$:

$$
G=2p(1-p).
$$

Both entropy and Gini impurity are:

- zero when the node contains only one class;
- largest when classes are evenly mixed.

## 4.6 Choosing a Split

Suppose a parent node has $N$ examples and a candidate split produces left and right children with $N_L$ and $N_R$ examples.

The weighted child impurity is

$$
Q_{\mathrm{children}}
=
\frac{N_L}{N}Q_L+
\frac{N_R}{N}Q_R.
$$

The impurity reduction is

$$
\Delta Q
=
Q_{\mathrm{parent}}-Q_{\mathrm{children}}.
$$

The greedy tree-growing algorithm selects the split with the largest impurity reduction, or equivalently the smallest weighted child impurity.

For regression, impurity is commonly measured by residual sum of squares or variance.

## 4.7 Greedy Tree Growth

Finding the globally optimal tree is combinatorially difficult. Practical tree algorithms use greedy search.

At one node:

1. Select a candidate feature $x_j$.
2. Consider candidate thresholds $\theta$.
3. Partition examples into $x_j\leq\theta$ and $x_j>\theta$.
4. Compute the resulting impurity or squared error.
5. Choose the best feature and threshold.
6. Repeat recursively on child nodes.

Greedy search is efficient, but it does not guarantee the globally best tree.

## 4.8 Overfitting and Pruning

A tree can keep splitting until every leaf contains very few examples. Such a tree can memorize training noise.

The textbook describes a cost-complexity objective

$$
C(T)
=
\sum_{\tau=1}^{|T|}Q_\tau(T)+\lambda|T|.
$$

Here:

- $Q_\tau(T)$ measures prediction error in leaf $\tau$;
- $|T|$ is the number of leaves;
- $\lambda$ penalizes tree complexity.

A larger $\lambda$ prefers a smaller tree.

Two practical strategies are common:

### Pre-pruning

Stop growing based on conditions such as:

- maximum depth;
- minimum examples required to split a node;
- minimum examples allowed in a leaf;
- minimum impurity reduction.

### Post-pruning

Grow a relatively large tree and then remove branches whose complexity is not justified by validation performance.

## 4.9 Strengths of Decision Trees

Decision trees are useful because they:

- represent nonlinear thresholds;
- model feature interactions automatically;
- require little feature scaling;
- can handle mixed kinds of predictive patterns;
- are easy to visualize when small;
- produce rules that humans can inspect.

For example:

> If temperature is high and blood pressure is low, follow one branch; otherwise follow another.

## 4.10 Limitations of a Single Tree

A single tree has important weaknesses:

1. **Instability:** small changes in data can produce a different tree.
2. **High variance:** a deep tree may overfit.
3. **Axis-aligned splits:** an oblique boundary may require many rectangular regions.
4. **Piecewise-constant regression:** predictions can jump abruptly across split boundaries.
5. **Greedy optimization:** early split choices may prevent a better global structure.

These limitations motivate random forests and boosting.

## Textbook Exercise 14.11: Why Gini and Entropy Can Prefer a Better Split

> ![Textbook Exercise 14.11](./CoursePR2026/Fig/Chapter_14/lecture_ex_14_11__textbook_ex_14_11__p675.png)
>
> *Textbook Exercise 14.11 (p. 675): Compare two trees that have equal misclassification rates but different entropy and Gini impurities.*

The data contain 400 examples from each class, so $N=800$.

### Tree A

The leaves contain

$$
(300,100),\qquad (100,300).
$$

Each leaf misclassifies 100 examples. Total errors:

$$
100+100=200.
$$

Misclassification rate:

$$
\frac{200}{800}=0.25.
$$

Each leaf has probabilities $(0.75,0.25)$.

Entropy:

$$
H_A
=-0.75\ln0.75-0.25\ln0.25
\approx0.5623.
$$

Gini impurity:

$$
G_A
=1-(0.75^2+0.25^2)
=0.375.
$$

### Tree B

The leaves contain

$$
(200,400),\qquad (200,0).
$$

The first leaf predicts class 2 and makes 200 errors. The second leaf is pure and makes no errors.

Total misclassification rate is again

$$
\frac{200}{800}=0.25.
$$

The first leaf contains $600$ examples with probabilities $(1/3,2/3)$. The second leaf is pure.

Weighted entropy:

$$
H_B
=
\frac{600}{800}
\left[-\frac13\ln\frac13-\frac23\ln\frac23\right]
+
\frac{200}{800}(0)
\approx0.4774.
$$

Weighted Gini impurity:

$$
G_B
=
\frac{600}{800}
\left[1-\left(\frac13\right)^2-\left(\frac23\right)^2\right]
\approx0.3333.
$$

Thus

$$
H_B<H_A,
\qquad
G_B<G_A.
$$

Entropy and Gini recognize that tree B creates one completely pure leaf, while the coarse misclassification rate cannot see this improvement.

---

# §5 Random Forests

> Modern practical extension of textbook §14.2 and §14.4

## 5.1 From Bagged Trees to Random Forests

A random forest combines two ideas:

1. Train each tree on a bootstrap sample.
2. At each split, allow the tree to consider only a random subset of features.

The first idea creates data diversity. The second idea creates feature diversity.

A random forest prediction is

### Regression

$$
\widehat{y}(\mathbf{x})
=
\frac{1}{M}\sum_{m=1}^{M}T_m(\mathbf{x}).
$$

### Classification

$$
\widehat{C}(\mathbf{x})
=
\operatorname{mode}\{T_1(\mathbf{x}),\ldots,T_M(\mathbf{x})\}.
$$

Alternatively, average class probabilities from the trees.

## 5.2 Why Random Feature Selection Helps

Suppose one feature is extremely predictive. If every tree can always consider every feature, many trees may choose the same feature near the root.

Then the trees become highly correlated.

Randomly restricting the candidate features at each split forces some trees to discover different predictive structures.

This can slightly weaken individual trees but reduce correlation enough to improve the ensemble.

The goal is not to make each tree individually optimal. The goal is to make the **forest** strong.

## 5.3 Why Deep Trees Can Work in a Forest

A deep decision tree usually has:

- low bias;
- high variance.

Bagging primarily reduces variance. Therefore random forests often use relatively deep trees and rely on averaging for stabilization.

This contrasts with gradient boosting, which usually uses shallow trees as small correction steps.

## 5.4 Out-of-Bag Evaluation

Each bootstrap sample leaves out some training examples. These are called **out-of-bag (OOB)** examples for that tree.

For each training example:

1. Find trees whose bootstrap samples did not include that example.
2. Aggregate predictions from those trees.
3. Compare with the true target.

The resulting OOB score provides an internal estimate of generalization performance without creating a separate validation set.

It is still good practice to use a proper test set for final evaluation.

## 5.5 Important Hyperparameters

| Hyperparameter | Effect |
|----------------|--------|
| Number of trees | More trees reduce Monte Carlo variability but cost more computation. |
| Maximum depth | Controls individual tree complexity. |
| Minimum leaf size | Larger leaves smooth predictions and reduce variance. |
| Number of candidate features per split | Smaller values reduce correlation but may weaken splits. |
| Class weights | Useful for imbalanced classification. |
| Bootstrap on/off | Controls whether each tree sees a resampled data set. |

## 5.6 Strengths and Weaknesses

### Strengths

- strong general-purpose baseline for tabular data;
- little feature scaling required;
- handles nonlinear interactions;
- relatively robust to noisy features;
- parallel training is possible;
- usually less sensitive than one decision tree.

### Weaknesses

- many trees reduce interpretability;
- models can be large in memory;
- regression predictions do not extrapolate smoothly beyond observed target ranges;
- probability calibration may require extra attention;
- a well-tuned gradient-boosted tree often achieves better predictive accuracy on many tabular tasks.

---

# §6 Boosting and AdaBoost

> 📖 Textbook §14.3 (§14.3.1-§14.3.2)

## 6.1 Bagging versus Boosting

Bagging trains models mostly independently. Boosting trains models sequentially.

In boosting:

1. Train a weak learner.
2. Identify what the current learner or ensemble gets wrong.
3. Train the next learner to pay more attention to those errors.
4. Combine all learners.

A weak learner is a model that performs only slightly better than random guessing, such as a decision stump.

A **decision stump** is a one-split decision tree.

> ![Figure 14.1](./CoursePR2026/Fig/Chapter_14/lecture_fig_14_1__textbook_fig_14_1__p658.png)
>
> *Figure 14.1 (Textbook Fig. 14.1, p. 658): Each base classifier is trained using weights determined by earlier classifiers. The final prediction is a weighted vote.*

## 6.2 AdaBoost Setup

Consider binary classification with

$$
t_n\in\{-1,+1\}.
$$

Each weak learner produces

$$
y_m(\mathbf{x})\in\{-1,+1\}.
$$

Initially, all examples receive equal weights:

$$
w_n^{(1)}=\frac{1}{N}.
$$

At stage $m$, train a weak learner using the weighted data.

Its weighted error is

$$
\epsilon_m
=
\frac{
\sum_{n=1}^{N}w_n^{(m)}
\mathbb{I}[y_m(\mathbf{x}_n)\neq t_n]
}{
\sum_{n=1}^{N}w_n^{(m)}
}.
$$

The learner receives vote weight

$$
\alpha_m
=
\ln\frac{1-\epsilon_m}{\epsilon_m}.
$$

Then increase the weights of misclassified points:

$$
w_n^{(m+1)}
=
w_n^{(m)}
\exp\left(
\alpha_m\mathbb{I}[y_m(\mathbf{x}_n)\neq t_n]
\right).
$$

After updating, normalize the weights so that they sum to one.

The final classifier is

$$
Y_M(\mathbf{x})
=
\operatorname{sign}\left(
\sum_{m=1}^{M}\alpha_m y_m(\mathbf{x})
\right).
$$

## 6.3 Interpreting the Learner Weight

Consider

$$
\alpha_m
=
\ln\frac{1-\epsilon_m}{\epsilon_m}.
$$

### If $\epsilon_m$ is small

The learner is accurate, so $\alpha_m$ is large and positive.

### If $\epsilon_m=0.5$

$$
\alpha_m=\ln 1=0.
$$

A random-level learner receives no vote.

### If $\epsilon_m>0.5$

$\alpha_m$ is negative. In binary classification, we can usually flip the learner's predictions to obtain error below $0.5$.

## 6.4 Equivalent AdaBoost Conventions

Some references use

$$
\alpha_m
=
\frac12\ln\frac{1-\epsilon_m}{\epsilon_m}
$$

and update weights using

$$
w_n\leftarrow w_n\exp(-\alpha_m t_n y_m(\mathbf{x}_n)).
$$

This is equivalent up to a factor-of-two convention in the ensemble score. Students should focus on the idea rather than memorizing one convention:

> better learners receive larger votes, and misclassified examples receive more weight.

## 6.5 A Numerical AdaBoost Step

Suppose there are six equally weighted examples:

$$
w_n^{(1)}=\frac16.
$$

A decision stump misclassifies two examples.

Then

$$
\epsilon_1
=2\times\frac16
=\frac13.
$$

The learner weight is

$$
\alpha_1
=
\ln\frac{1-1/3}{1/3}
=
\ln 2
\approx0.693.
$$

For a correctly classified point, the unnormalized weight remains

$$
\frac16.
$$

For a misclassified point, the weight becomes

$$
\frac16e^{0.693}
=
\frac16\cdot2
=
\frac13.
$$

Before normalization, the total weight is

$$
4\left(\frac16\right)
+
2\left(\frac13\right)
=
\frac43.
$$

After normalization:

- each correctly classified point has weight $\frac18=0.125$;
- each misclassified point has weight $\frac14=0.25$.

Thus the next learner pays twice as much attention to each misclassified point.

## 6.6 How the Boundary Evolves

> ![Figure 14.2](./CoursePR2026/Fig/Chapter_14/lecture_fig_14_2__textbook_fig_14_2__p660.png)
>
> *Figure 14.2 (Textbook Fig. 14.2, p. 660): AdaBoost combines many simple axis-aligned thresholds. Circle size indicates example weight. Misclassified examples become larger and influence the next learner more strongly.*

The figure illustrates several important ideas:

1. Each individual decision stump is very simple.
2. The combined boundary becomes increasingly flexible.
3. Difficult examples receive larger weights.
4. Many simple rules can produce a complex classifier.

## 6.7 Exponential Loss Intuition

AdaBoost can be interpreted as stagewise minimization of the exponential loss

$$
E
=
\sum_{n=1}^{N}
\exp[-t_nF(\mathbf{x}_n)].
$$

Define the signed margin

$$
z_n=t_nF(\mathbf{x}_n).
$$

- If $z_n>0$, the example is correctly classified.
- If $z_n<0$, it is misclassified.
- A large positive $z_n$ means a confident correct prediction.
- A large negative $z_n$ means a confident wrong prediction.

> ![Figure 14.3](./CoursePR2026/Fig/Chapter_14/lecture_fig_14_3__textbook_fig_14_3__p662.png)
>
> *Figure 14.3 (Textbook Fig. 14.3, p. 662): Exponential, cross-entropy, hinge, and misclassification losses as functions of signed margin $z=tF(\mathbf{x})$.*

The exponential loss grows extremely quickly for confident mistakes. This explains both a strength and a weakness of AdaBoost.

### Strength

The algorithm aggressively focuses on difficult examples.

### Weakness

A mislabeled example or severe outlier can receive enormous weight and dominate later stages.

## 6.8 Robustness and Alternative Losses

> ![Figure 14.4](./CoursePR2026/Fig/Chapter_14/lecture_fig_14_4__textbook_fig_14_4__p663.png)
>
> *Figure 14.4 (Textbook Fig. 14.4, p. 663): Absolute error grows more slowly than squared error and is therefore less sensitive to large residuals.*

The same principle applies to boosting losses.

- Squared loss strongly emphasizes large regression errors.
- Absolute loss is more robust.
- Exponential loss strongly emphasizes confident classification mistakes.
- Logistic loss grows more gently for large negative margins.

Modern boosting frameworks often allow the loss function to be chosen according to the task.

## 6.9 Practical AdaBoost Advice

AdaBoost works best when:

- weak learners are slightly better than random;
- data labels are reasonably clean;
- a sequence of simple boundaries can represent the task;
- sample weighting is supported by the base learner.

Be cautious when:

- labels contain substantial noise;
- there are severe outliers;
- the data set is extremely imbalanced without appropriate weighting;
- weak learners repeatedly focus on impossible examples.

---

# §7 Gradient Boosting

> 📖 Concept motivated by textbook §14.3.2 and Exercise 14.9; modern practical extension

## 7.1 From Reweighting Examples to Fitting Residuals

AdaBoost modifies data weights. Gradient boosting uses a broader idea:

> Add a new model that moves the current prediction in the direction that reduces the loss.

The ensemble is additive:

$$
F_M(\mathbf{x})
=
F_0(\mathbf{x})
+
\sum_{m=1}^{M}\eta\gamma_m h_m(\mathbf{x}).
$$

Here:

- $F_0$ is an initial constant prediction;
- $h_m$ is a small regression tree;
- $\gamma_m$ is the size of the correction;
- $\eta$ is the learning rate.

## 7.2 Squared-Loss Regression: Fit the Residual

Suppose the loss is

$$
L(t,F)=\frac12(t-F)^2.
$$

At stage $m-1$, the current prediction is $F_{m-1}(\mathbf{x}_n)$.

The residual is

$$
r_{nm}
=
t_n-F_{m-1}(\mathbf{x}_n).
$$

Train the next tree $h_m$ to predict these residuals.

Then update

$$
F_m(\mathbf{x})
=
F_{m-1}(\mathbf{x})
+
\eta h_m(\mathbf{x}).
$$

This is the result highlighted by textbook Exercise 14.9.

## 7.3 Why Residual Fitting Makes Sense

Suppose the current model predicts too low:

$$
t_n-F_{m-1}(\mathbf{x}_n)>0.
$$

Then the new tree should add a positive correction.

Suppose the current model predicts too high:

$$
t_n-F_{m-1}(\mathbf{x}_n)<0.
$$

Then the new tree should add a negative correction.

Each tree is not trying to solve the complete task from the beginning. It is correcting the remaining error.

## 7.4 A One-Step Gradient-Boosting Example

Consider four ordered inputs with targets

| $x$ | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|
| $t$ | 3 | 5 | 8 | 10 |

For squared loss, the best initial constant is the mean:

$$
F_0
=
\frac{3+5+8+10}{4}
=6.5.
$$

Initial residuals are

$$
r_1=3-6.5=-3.5,
$$

$$
r_2=5-6.5=-1.5,
$$

$$
r_3=8-6.5=1.5,
$$

$$
r_4=10-6.5=3.5.
$$

Fit a stump that splits between $x=2$ and $x=3$.

The mean residual on the left is

$$
\frac{-3.5-1.5}{2}=-2.5.
$$

The mean residual on the right is

$$
\frac{1.5+3.5}{2}=2.5.
$$

Let the learning rate be

$$
\eta=0.5.
$$

The updated prediction is

$$
F_1(x)
=
\begin{cases}
6.5+0.5(-2.5)=5.25, & x\leq2,\\
6.5+0.5(2.5)=7.75, & x>2.
\end{cases}
$$

The original sum of squared errors was

$$
(-3.5)^2+(-1.5)^2+(1.5)^2+(3.5)^2=29.
$$

After one small tree, it becomes

$$
(3-5.25)^2+(5-5.25)^2+(8-7.75)^2+(10-7.75)^2
=10.25.
$$

One simple correction substantially reduces the error. Later trees can fit the remaining residuals.

## 7.5 General Losses: Negative Gradients

For a general differentiable loss $L(t,F)$, define a pseudo-residual

$$
r_{nm}
=
-\left.
\frac{\partial L(t_n,F)}{\partial F}
\right|_{F=F_{m-1}(\mathbf{x}_n)}.
$$

This is the negative gradient of the loss with respect to the current prediction.

Then:

1. compute pseudo-residuals;
2. fit a tree to them;
3. add the tree to the ensemble.

This is why the method is called **gradient boosting**: it performs gradient descent in function space.

## 7.6 Classification with Logistic Loss

For binary classification, gradient boosting can minimize logistic or cross-entropy loss rather than exponential loss.

The model produces a score $F(\mathbf{x})$. A probability can be obtained through a sigmoid transformation.

Each stage fits a tree to pseudo-residuals determined by the difference between observed labels and current probabilities.

The exact implementation details vary, but the central idea is the same:

> the next tree corrects the direction in which the current probabilistic classifier is wrong.

## 7.7 Why Shallow Trees Are Common

In gradient boosting, each tree is one small update. A shallow tree often works well because it:

- captures a limited interaction pattern;
- avoids making one correction too aggressive;
- allows later trees to refine the result;
- combines naturally with a small learning rate.

A depth-1 tree models one threshold. A depth-2 or depth-3 tree can model modest feature interactions.

## 7.8 Learning Rate and Number of Trees

The update is

$$
F_m=F_{m-1}+\eta h_m.
$$

A smaller $\eta$ makes each step more conservative.

Usually:

- smaller learning rate requires more trees;
- larger learning rate trains faster but can overfit or become unstable;
- validation-based early stopping is useful.

This is similar to optimization in neural networks: step size and number of steps interact.

## 7.9 Stochastic Gradient Boosting

At each stage, a model can be trained on a random subset of examples or features.

This introduces randomness that may:

- reduce overfitting;
- reduce correlation between correction trees;
- speed up training.

The idea combines aspects of bagging and boosting.

## 7.10 AdaBoost versus Gradient Boosting

| Question | AdaBoost | Gradient Boosting |
|----------|----------|------------------|
| Main mechanism | Reweight examples | Fit negative gradients or residuals |
| Original task | Binary classification | Regression and classification |
| Classical loss | Exponential loss | User-chosen differentiable loss |
| Typical weak learner | Decision stump | Shallow regression tree |
| Outlier sensitivity | Can be high | Depends on selected loss |
| Probability interpretation | Not naturally probabilistic under exponential loss | Often uses logistic, squared, Poisson, ranking, or other losses |

AdaBoost is an elegant and historically important boosting algorithm. Gradient boosting is a more general practical framework.

## 7.11 Modern Tree-Boosting Systems

Modern implementations such as XGBoost, LightGBM, and CatBoost build on the gradient-boosting idea and add engineering and regularization techniques such as:

- efficient split search;
- second-order loss information;
- row and feature subsampling;
- explicit tree-complexity penalties;
- missing-value handling;
- specialized handling of categorical variables in some systems;
- early stopping and parallel computation.

The underlying conceptual model remains simple:

$$
\text{many small trees}
+
\text{sequential error correction}
=
\text{a strong nonlinear predictor}.
$$

---

# §8 Conditional Mixtures and Mixtures of Experts

> 📖 Textbook §14.5; emphasis on §14.5.3 intuition only

## 8.1 Why One Global Model May Be Insufficient

Suppose a regression problem has two different trends.

For some examples,

$$
t\approx a_1x+b_1.
$$

For others,

$$
t\approx a_2x+b_2.
$$

A single linear regression model averages the two trends and may fit neither one well.

A mixture model uses several component predictors.

> ![Figure 14.8](./CoursePR2026/Fig/Chapter_14/lecture_fig_14_8__textbook_fig_14_8__p670.png)
>
> *Figure 14.8 (Textbook Fig. 14.8, p. 670): A mixture of two linear regression models learns two different trends. The lower plots show responsibilities indicating which component explains each point.*

## 8.2 Multimodal Conditional Predictions

> ![Figure 14.9](./CoursePR2026/Fig/Chapter_14/lecture_fig_14_9__textbook_fig_14_9__p671.png)
>
> *Figure 14.9 (Textbook Fig. 14.9, p. 671): A mixture can represent a bimodal conditional distribution, while a single linear regression model produces only one Gaussian mode.*

This matters when one input can have several plausible outputs.

Examples include:

- future-motion prediction;
- inverse kinematics;
- ambiguous image reconstruction;
- demand forecasts with multiple regimes;
- user behavior with different latent groups.

Predicting only the conditional mean can produce an implausible answer between modes.

## 8.3 From Fixed Mixture Weights to a Gate

A simple mixture uses fixed coefficients:

$$
p(t\mid\mathbf{x})
=
\sum_{k=1}^{K}\pi_kp_k(t\mid\mathbf{x}).
$$

A mixture of experts uses input-dependent coefficients:

$$
\boxed{
p(t\mid\mathbf{x})
=
\sum_{k=1}^{K}\pi_k(\mathbf{x})p_k(t\mid\mathbf{x})
}.
$$

The functions $\pi_k(\mathbf{x})$ form the **gating network**.

The component models $p_k(t\mid\mathbf{x})$ are the **experts**.

## 8.4 Softmax Gating

A common gate uses scores $a_k(\mathbf{x})$ and a softmax:

$$
\pi_k(\mathbf{x})
=
\frac{\exp(a_k(\mathbf{x}))}
{\sum_{j=1}^{K}\exp(a_j(\mathbf{x}))}.
$$

This guarantees

$$
0\leq\pi_k(\mathbf{x})\leq1
$$

and

$$
\sum_{k=1}^{K}\pi_k(\mathbf{x})=1.
$$

## 8.5 Hard Tree versus Soft Mixture of Experts

A decision tree performs hard routing:

$$
\pi_k(\mathbf{x})\in\{0,1\}.
$$

Exactly one leaf receives the input.

A mixture of experts performs soft routing:

$$
0\leq\pi_k(\mathbf{x})\leq1.
$$

Several experts can contribute.

| Property | Decision Tree | Mixture of Experts |
|----------|---------------|--------------------|
| Routing | Hard | Soft or sparse |
| Boundary | Usually axis-aligned | Can depend on all features |
| Prediction | One leaf model | Weighted expert combination |
| Interpretability | Often high for small trees | Usually lower |
| Smoothness | Piecewise constant or piecewise simple | Can be smooth across regions |

## 8.6 Mixtures of Classifiers

> ![Figure 14.10](./CoursePR2026/Fig/Chapter_14/lecture_fig_14_10__textbook_fig_14_10__p673.png)
>
> *Figure 14.10 (Textbook Fig. 14.10, p. 673): A single logistic classifier cannot represent the class pattern well, while a mixture of two logistic models captures substantially more of the structure.*

The example shows that several simple classifiers can represent a pattern that one linear classifier cannot.

A mixture of experts goes one step further: the gate learns which classifier should dominate for each input.

## 8.7 Modern Deep-Learning Connection

In modern neural mixture-of-experts systems:

- each expert may be a neural subnetwork;
- a router computes expert scores for each input or token;
- only a small subset of experts may be activated;
- model capacity can increase without activating every parameter for every input.

The core idea is still the same as equation (14.53) in the textbook:

> different experts specialize, and a gate decides which experts should contribute.

This chapter therefore provides a direct conceptual bridge from classical ensemble learning to modern sparse expert architectures.

## 8.8 What We Skip

The textbook derives EM updates for mixtures of linear and logistic regression models. For this course, it is enough to understand:

1. A latent component indicates which expert explains an example.
2. A responsibility measures the probability that expert $k$ explains example $n$.
3. Experts are trained using responsibility-weighted data.
4. A mixture of experts makes the mixing weights depend on the input.

We do not need the full EM algebra for the practical learning objectives of this lecture.

---

# §9 Practical Model Selection for Tabular Data

## 9.1 Why Tree-Based Models Matter for Tabular Data

Tabular data often contain:

- numerical and categorical variables;
- threshold effects;
- nonlinear feature interactions;
- missing or irregular measurements;
- very different feature scales;
- moderate sample sizes;
- limited benefit from spatial or sequential inductive biases.

Trees naturally represent rules such as

> age $>45$ and blood pressure $<120$ and marker level $>\theta$.

No manual polynomial feature construction is required to model this interaction.

## 9.2 A Sensible Practical Progression

For a new supervised tabular problem:

1. Start with a simple linear or logistic baseline.
2. Train one shallow decision tree for interpretability.
3. Train a random forest as a strong stable baseline.
4. Train a gradient-boosted tree model with validation-based early stopping.
5. Compare performance, calibration, speed, and interpretability.

Do not select a model only because it is more modern or more complicated.

## 9.3 Model Comparison Table

| Model | Main Strength | Main Weakness | Good First Use |
|------|---------------|---------------|----------------|
| Linear/logistic regression | Simple, fast, interpretable coefficients | Limited nonlinear interactions | Baseline and calibrated probabilities |
| Single decision tree | Human-readable rules | High variance and instability | Explanation and debugging |
| Random forest | Robust, stable, little tuning | Large model; may be less accurate than boosting | Strong general tabular baseline |
| AdaBoost | Simple sequential ensemble | Sensitive to noisy labels/outliers | Clean binary problems and teaching |
| Gradient-boosted trees | Excellent nonlinear tabular modeling | Tuning and overfitting require care | High-performance structured-data model |
| Neural network | Flexible representation learning | Often needs more data and tuning | Large data, embeddings, multimodal or unstructured inputs |
| Mixture of experts | Specialized conditional computation | More complex training and routing | Heterogeneous regimes and large conditional models |

## 9.4 Data Leakage Still Matters More Than the Algorithm

A powerful boosting model can exploit leakage more effectively than a weak baseline.

Examples of leakage include:

- using future information to predict the past;
- computing preprocessing statistics from the full data set;
- placing repeated measurements from the same subject in both train and test sets;
- including a variable that directly encodes the target;
- selecting features using the test set.

Correct validation design is more important than a small difference between ensemble algorithms.

## 9.5 Cross-Validation Strategy

Use a split that matches deployment.

| Data Structure | Appropriate Split |
|----------------|------------------|
| Independent examples | Random train/validation/test split |
| Time series | Train on earlier times, validate on later times |
| Multiple records per person/device | Grouped split by person or device |
| Strong class imbalance | Stratified split when compatible with grouping/time |
| Small data set | Cross-validation, with a separate final test set when possible |

## 9.6 Class Imbalance

Accuracy can be misleading when one class is rare.

Consider:

- precision and recall;
- F1 score;
- area under the precision-recall curve;
- balanced accuracy;
- cost-sensitive evaluation;
- class weights or careful resampling.

The split criterion and training objective should reflect the real application cost.

## 9.7 Probability Calibration

A model can rank examples well but produce inaccurate probabilities.

For applications involving risk, triage, or decision thresholds, evaluate calibration using:

- reliability diagrams;
- Brier score;
- expected calibration error;
- calibration methods fitted on held-out data.

A high classification accuracy does not guarantee that a predicted probability of $0.9$ is correct nine times out of ten.

## 9.8 Feature Importance Requires Care

Tree ensembles can provide several notions of importance.

### Split-based importance

Measures how much a feature reduces impurity across splits.

Potential issue: it can favor variables with many possible split points.

### Permutation importance

Randomly permute one feature and measure the loss of predictive performance.

Potential issue: correlated features can substitute for each other, making each appear less important.

### Local explanation methods

Methods such as SHAP can attribute one prediction to input features, but their interpretation depends on assumptions about feature dependence and background distributions.

Feature importance is evidence about model behavior, not automatic proof of causality.

## 9.9 Hyperparameter Priorities

For random forests, start with:

- enough trees for stable validation performance;
- minimum leaf size;
- feature subsampling;
- class weights if needed.

For gradient boosting, prioritize:

- learning rate;
- number of trees with early stopping;
- tree depth or number of leaves;
- minimum leaf size;
- row and feature subsampling;
- regularization.

A small learning rate with early stopping is often safer than a very aggressive learning rate.

## 9.10 Debugging Checklist

Before extensive tuning, verify:

1. Targets and features are aligned correctly.
2. Train and validation preprocessing are separated.
3. Metrics match the application.
4. A trivial baseline is included.
5. The model can overfit a tiny subset when debugging implementation.
6. Missing-value handling is consistent.
7. Categorical variables are encoded appropriately for the chosen implementation.
8. Random seeds and data splits are recorded.
9. Validation performance is not being repeatedly optimized against the final test set.

---

# §10 Worked Examples and Textbook Exercises

## 10.1 Averaging Regression Predictions

Three models predict the same target:

$$
y_1=8,
\qquad
y_2=11,
\qquad
y_3=10.
$$

The committee prediction is

$$
y_{\mathrm{COM}}
=
\frac{8+11+10}{3}
=
\frac{29}{3}
\approx9.67.
$$

Suppose the true target is $t=10$.

Individual squared errors are

$$
(8-10)^2=4,
$$

$$
(11-10)^2=1,
$$

$$
(10-10)^2=0.
$$

Average individual squared error:

$$
\frac{4+1+0}{3}
=\frac53
\approx1.67.
$$

Committee squared error:

$$
(9.67-10)^2
\approx0.11.
$$

The average prediction is much better here because the individual errors partly cancel.

## 10.2 A Simple Regression-Tree Split

Suppose a node contains:

| $x$ | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|
| $t$ | 2 | 3 | 9 | 10 |

### Without a split

The leaf mean is

$$
\bar{t}=\frac{2+3+9+10}{4}=6.
$$

The sum of squared errors is

$$
(2-6)^2+(3-6)^2+(9-6)^2+(10-6)^2
=16+9+9+16
=50.
$$

### Split at $x\leq2$

Left targets: $2,3$.

$$
c_L=2.5.
$$

Right targets: $9,10$.

$$
c_R=9.5.
$$

New sum of squared errors:

$$
(2-2.5)^2+(3-2.5)^2+(9-9.5)^2+(10-9.5)^2
=1.
$$

The split reduces squared error from $50$ to $1$.

## 10.3 A Simple Gini Calculation

Suppose a node contains 8 class-A examples and 2 class-B examples.

Then

$$
p_A=0.8,
\qquad
p_B=0.2.
$$

Gini impurity is

$$
G
=1-(0.8^2+0.2^2)
=1-(0.64+0.04)
=0.32.
$$

If a split creates two pure child nodes, their weighted Gini impurity is zero, so the impurity reduction is $0.32$.

## 10.4 AdaBoost Weight Update

Suppose four examples have current weights

$$
(0.1,0.2,0.3,0.4).
$$

A learner misclassifies examples 1 and 2.

Weighted error:

$$
\epsilon=0.1+0.2=0.3.
$$

Learner weight:

$$
\alpha
=
\ln\frac{0.7}{0.3}
\approx0.847.
$$

Misclassified weights are multiplied by

$$
e^{0.847}\approx2.333.
$$

Unnormalized new weights:

$$
(0.2333,0.4667,0.3,0.4).
$$

Their sum is

$$
1.4.
$$

Normalized weights:

$$
(0.1667,0.3333,0.2143,0.2857).
$$

The misclassified examples now receive half of the total training weight.

## 10.5 Gradient Boosting as Residual Correction

Suppose the current predictions are

$$
F_0=(4,4,4),
$$

and targets are

$$
t=(2,5,8).
$$

Residuals are

$$
r=t-F_0=(-2,1,4).
$$

If a new tree perfectly predicts these residuals and $\eta=0.25$, then

$$
F_1
=F_0+0.25r.
$$

Therefore,

$$
F_1=(3.5,4.25,5).
$$

The ensemble moves one quarter of the way from the old predictions toward the targets. A small learning rate makes each correction conservative.

## 10.6 Concept Check: Hard and Soft Routing

Suppose two experts predict

$$
y_1(\mathbf{x})=3,
\qquad
y_2(\mathbf{x})=9.
$$

### Hard tree routing

If the tree selects expert 1,

$$
y(\mathbf{x})=3.
$$

### Soft mixture-of-experts routing

If

$$
\pi_1(\mathbf{x})=0.25,
\qquad
\pi_2(\mathbf{x})=0.75,
$$

then

$$
y(\mathbf{x})
=0.25(3)+0.75(9)
=7.5.
$$

Soft routing allows a smooth transition between specialists.

---

# §11 Chapter Summary and Course Wrap-Up

## 11.1 Conceptual Summary

This chapter can be summarized in one sentence:

> Several simple or unstable models can form a powerful predictor when their errors are diversified, their corrections are coordinated, or their expertise is routed appropriately.

| Topic | Main Lesson |
|-------|-------------|
| Bayesian model averaging | Average predictions according to posterior uncertainty over whole candidate models. |
| Committee | Average several model predictions. |
| Bagging | Use bootstrap samples to create diverse models and reduce variance. |
| Decision tree | Recursively route inputs to region-specific predictions. |
| Random forest | Average decorrelated trees created by bootstrap and feature randomness. |
| AdaBoost | Sequentially increase attention to misclassified examples. |
| Gradient boosting | Sequentially fit residuals or negative loss gradients. |
| Mixture of experts | Learn input-dependent soft routing to specialized models. |

## 11.2 The Most Important Practical Distinction

### Bagging and random forests

Primarily reduce variance by averaging many related models.

### Boosting

Primarily builds a strong additive model through sequential correction.

A useful memory aid is:

> **Bagging averages; boosting corrects.**

## 11.3 Common Student Confusions

| Confusion | Clarification |
|-----------|---------------|
| “More trees always cause overfitting.” | In a random forest, more trees usually stabilize the average. Tree depth and leaf size matter more for individual complexity. |
| “Bagging and boosting are the same because both use many trees.” | Bagging trains trees mostly independently; boosting trains them sequentially to correct errors. |
| “A decision tree is a probabilistic graphical model.” | No. It is a predictive routing structure, not a Bayesian network or Markov random field. |
| “The best split minimizes misclassification immediately.” | Entropy or Gini is usually more sensitive during tree growth. |
| “AdaBoost only counts mistakes.” | It uses weighted mistakes, and accurate learners receive larger final votes. |
| “Gradient boosting always fits ordinary residuals.” | Ordinary residuals apply to squared loss. General gradient boosting fits negative loss gradients. |
| “Feature importance proves causality.” | It only summarizes how the fitted model uses features. |
| “Neural networks have replaced boosted trees.” | Neural networks dominate many unstructured-data tasks, while boosted trees remain highly important for many tabular problems. |
| “A mixture of experts is just an average.” | The gate depends on the input, so different examples can use different combinations of experts. |

## 11.4 Suggested Board Equations

The following equations are sufficient for a clear lecture presentation.

### Board 1: Committee and bagging

$$
y_{\mathrm{COM}}(\mathbf{x})
=
\frac{1}{M}\sum_{m=1}^{M}y_m(\mathbf{x})
$$

$$
\operatorname{var}(\text{average})
=
\sigma^2\left[
\rho+\frac{1-\rho}{M}
\right]
$$

### Board 2: Regression tree

$$
c_\tau
=
\frac{1}{N_\tau}\sum_{\mathbf{x}_n\in R_\tau}t_n
$$

$$
G_\tau=1-\sum_kp_{\tau k}^2
$$

### Board 3: AdaBoost

$$
\epsilon_m
=
\sum_nw_n^{(m)}\mathbb{I}[y_m(\mathbf{x}_n)\neq t_n]
$$

$$
\alpha_m
=
\ln\frac{1-\epsilon_m}{\epsilon_m}
$$

$$
Y_M(\mathbf{x})
=
\operatorname{sign}\left(
\sum_m\alpha_my_m(\mathbf{x})
\right)
$$

### Board 4: Gradient boosting

$$
r_{nm}
=
-\left.
\frac{\partial L(t_n,F)}{\partial F}
\right|_{F=F_{m-1}(\mathbf{x}_n)}
$$

$$
F_m(\mathbf{x})
=
F_{m-1}(\mathbf{x})+\eta h_m(\mathbf{x})
$$

### Board 5: Mixture of experts

$$
p(t\mid\mathbf{x})
=
\sum_k\pi_k(\mathbf{x})p_k(t\mid\mathbf{x})
$$

## 11.5 Suggested Half-Week Teaching Flow

This chapter is designed for approximately one 100-120 minute session.

| Approximate Time | Topic |
|------------------|-------|
| 10 min | Why ensembles work: variance and diversity |
| 15 min | Bootstrap, bagging, and committee-error intuition |
| 25 min | Decision trees: splits, leaf predictions, Gini, and pruning |
| 15 min | Random forests and practical hyperparameters |
| 20 min | AdaBoost with one numerical weight-update example |
| 20 min | Gradient boosting as residual or negative-gradient fitting |
| 10 min | Mixture-of-experts intuition and modern connection |
| 5 min | Practical model-selection summary |

If time is shorter, prioritize:

1. decision trees;
2. random forests;
3. AdaBoost intuition;
4. gradient boosting;
5. tabular-data practical guidance.

Bayesian model averaging and conditional-mixture details can be treated as optional reading.

## 11.6 Figure Checklist

All displayed figures below were cropped directly from the supplied textbook PDF.

| Lecture Figure | Textbook Figure | Topic | File |
|----------------|-----------------|-------|------|
| 14.1 | 14.1 | Boosting framework | `./CoursePR2026/Fig/Chapter_14/lecture_fig_14_1__textbook_fig_14_1__p658.png` |
| 14.2 | 14.2 | Evolution of AdaBoost boundary and sample weights | `./CoursePR2026/Fig/Chapter_14/lecture_fig_14_2__textbook_fig_14_2__p660.png` |
| 14.3 | 14.3 | Exponential, cross-entropy, hinge, and classification losses | `./CoursePR2026/Fig/Chapter_14/lecture_fig_14_3__textbook_fig_14_3__p662.png` |
| 14.4 | 14.4 | Squared versus absolute error | `./CoursePR2026/Fig/Chapter_14/lecture_fig_14_4__textbook_fig_14_4__p663.png` |
| 14.5 | 14.5 | Axis-aligned tree partition | `./CoursePR2026/Fig/Chapter_14/lecture_fig_14_5__textbook_fig_14_5__p663.png` |
| 14.6 | 14.6 | Binary tree representation | `./CoursePR2026/Fig/Chapter_14/lecture_fig_14_6__textbook_fig_14_6__p664.png` |
| 14.8 | 14.8 | Mixture of two linear regression models | `./CoursePR2026/Fig/Chapter_14/lecture_fig_14_8__textbook_fig_14_8__p670.png` |
| 14.9 | 14.9 | Multimodal predictive conditional density | `./CoursePR2026/Fig/Chapter_14/lecture_fig_14_9__textbook_fig_14_9__p671.png` |
| 14.10 | 14.10 | Mixture of logistic regression models | `./CoursePR2026/Fig/Chapter_14/lecture_fig_14_10__textbook_fig_14_10__p673.png` |

Textbook exercise images:

| Exercise | Topic | File |
|----------|-------|------|
| 14.2 | Ideal committee error reduction | `./CoursePR2026/Fig/Chapter_14/lecture_ex_14_2__textbook_ex_14_2__p674.png` |
| 14.10 | Mean prediction in a regression leaf | `./CoursePR2026/Fig/Chapter_14/lecture_ex_14_10__textbook_ex_14_10__p675.png` |
| 14.11 | Misclassification, entropy, and Gini comparison | `./CoursePR2026/Fig/Chapter_14/lecture_ex_14_11__textbook_ex_14_11__p675.png` |

## 11.7 Final Course Message

Across the course, we studied probability distributions, regression, classification, neural networks, kernels, graphical models, latent variables, approximate inference, sampling, dimensionality reduction, and sequential models.

This final chapter returns to a practical principle:

> A strong machine-learning system is not always one highly complicated model. It may instead be a carefully organized collection of simpler models.

The recurring ideas are the same ones that appeared throughout the course:

- uncertainty;
- bias and variance;
- optimization of a loss function;
- latent structure;
- conditional prediction;
- regularization;
- generalization to unseen data.

Ensemble learning brings these ideas together in a form that is both conceptually elegant and highly useful in modern practice.
