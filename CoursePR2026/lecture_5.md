---
layout: course
title: PRML Lecture 5
---

# Pattern Recognition and Machine Learning
## Chapter 5: Neural Networks

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: Ch. 5 Neural Networks (§5.1-§5.7)

---

## Table of Contents

1. [§0 Learning Viewpoint and Chapter Roadmap](#0-learning-viewpoint-and-chapter-roadmap)
2. [§1 Feed-forward Network Functions](#1-feed-forward-network-functions)
3. [§2 Network Training](#2-network-training)
4. [§3 Error Backpropagation](#3-error-backpropagation)
5. [§4 The Hessian Matrix](#4-the-hessian-matrix)
6. [§5 Regularization in Neural Networks](#5-regularization-in-neural-networks)
7. [§6 Mixture Density Networks](#6-mixture-density-networks)
8. [§7 Bayesian Neural Networks](#7-bayesian-neural-networks)
9. [§8 Chapter Summary, Figure Checklist, Exercises, and Teaching Flow](#8-chapter-summary-figure-checklist-exercises-and-teaching-flow)

---

## Notation and Variable Definitions

Chapters 3 and 4 studied **linear models** for regression and classification. In those chapters, the model was linear in the adaptive parameters, even when the input was first transformed by fixed basis functions. Chapter 5 keeps the same basic supervised-learning setting, but makes one major change:

> The basis functions are no longer fixed by hand. They become **adaptive basis functions** whose parameters are learned from data.

This is the main idea behind feed-forward neural networks. A neural network is still a parametric function $y(\mathbf{x},\mathbf{w})$, but its parameters appear in several layers, and therefore the error function is usually non-convex.

> **Teaching focus.** This chapter should not become a catalogue of every neural-network trick in the textbook. For EE undergraduate and graduate students, the core classroom path is:
>
> $$
> \text{network as learned features}
> \rightarrow
> \text{task-matched loss}
> \rightarrow
> \text{backpropagation}
> \rightarrow
> \text{regularization and modern architectures}.
> $$
>
> The most important parts are feed-forward computation, sigmoid/softmax losses, backpropagation, weight decay, early stopping, data augmentation, and convolution/weight sharing. The Hessian, mixture density networks, and Bayesian neural networks are useful conceptually, but can be shortened or skipped if teaching time is limited.

### Generic Neural-Network Notation

| Symbol | Definition |
|--------|------------|
| $\mathbf{x}$ | Input vector, usually $\mathbf{x}=(x_1,\ldots,x_D)^T$. |
| $D$ | Number of input dimensions. |
| $\mathbf{t}$ | Target vector. For regression it is continuous; for classification it is a class coding. |
| $N$ | Number of training examples. |
| $\mathcal{D}$ | Training data set, usually $\{(\mathbf{x}_n,\mathbf{t}_n)\}_{n=1}^N$. |
| $\mathbf{w}$ | Vector containing all network weights and biases. |
| $W$ | Total number of adaptive parameters in the network. |
| $M$ | Number of hidden units in a simple two-layer network. |
| $K$ | Number of output units or number of classes. |
| $y_k(\mathbf{x},\mathbf{w})$ | Output of the $k$th output unit. |
| $E(\mathbf{w})$ | Error function / negative log-likelihood to be minimized. |

### Layer and Activation Notation

| Symbol | Definition |
|--------|------------|
| $a_j$ | Activation of hidden unit $j$, before applying the nonlinearity. |
| $z_j$ | Output of hidden unit $j$, after applying the hidden activation function. |
| $h(\cdot)$ | Hidden-unit activation function, often $\tanh(\cdot)$ or a logistic sigmoid in this textbook. |
| $a_k$ | Activation of output unit $k$, before applying the output activation function. |
| $f(\cdot)$ | Output activation function. It may be identity, sigmoid, or softmax depending on the task. |
| $w^{(1)}_{ji}$ | First-layer weight from input $i$ to hidden unit $j$. |
| $w^{(1)}_{j0}$ | First-layer bias for hidden unit $j$. |
| $w^{(2)}_{kj}$ | Second-layer weight from hidden unit $j$ to output unit $k$. |
| $w^{(2)}_{k0}$ | Second-layer bias for output unit $k$. |

### Backpropagation Notation

| Symbol | Definition |
|--------|------------|
| $\delta_j$ | Local derivative $\partial E/\partial a_j$ for unit $j$. |
| $z_i$ | Input to a unit, often the output of a unit in the previous layer. |
| $w_{ji}$ | Weight from unit $i$ to unit $j$. |
| $J_{ki}$ | Jacobian entry $\partial y_k/\partial x_i$. |
| $\mathbf{H}$ | Hessian matrix of second derivatives, $H_{ij}=\partial^2E/\partial w_i\partial w_j$. |
| $\nabla E$ | Gradient vector of the error function. |

### Regularization and Bayesian Notation

| Symbol | Definition |
|--------|------------|
| $\lambda$ | Weight-decay coefficient in a regularized error function. |
| $\alpha$ | Prior precision over weights in a Bayesian neural network. |
| $\beta$ | Noise precision for Gaussian regression targets. |
| $\mathbf{w}_{\mathrm{ML}}$ | Maximum-likelihood weight vector. |
| $\mathbf{w}_{\mathrm{MAP}}$ | Maximum-a-posteriori weight vector. |
| $\mathbf{A}$ | Hessian / precision matrix used in the Laplace approximation. |
| $\gamma$ | Effective number of parameters in evidence approximation. |

### Mixture Density Network Notation

| Symbol | Definition |
|--------|------------|
| $p(\mathbf{t}\mid\mathbf{x})$ | Conditional density of target given input. |
| $\pi_k(\mathbf{x})$ | Mixing coefficient of component $k$, with $\sum_k\pi_k(\mathbf{x})=1$. |
| $\boldsymbol{\mu}_k(\mathbf{x})$ | Mean of component $k$ as a function of input. |
| $\sigma_k^2(\mathbf{x})$ | Variance of component $k$ as a function of input. |
| $\gamma_{nk}$ | Responsibility of mixture component $k$ for training example $n$. |

---

# §0 Learning Viewpoint and Chapter Roadmap

> 📖 Textbook Ch.5 opening; §5.1-§5.7

## 0.1 What This Chapter Is Really About

The previous chapters introduced a very important template:

$$
y(\mathbf{x},\mathbf{w})=\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}).
$$

Here $\boldsymbol{\phi}(\mathbf{x})$ is a vector of basis functions. In Chapters 3 and 4, these basis functions were assumed to be fixed before training. For example, we might choose polynomial basis functions, Gaussian basis functions, or sigmoidal basis functions by hand.

The limitation is clear. If the basis functions are poorly chosen, then the model may fail even if we fit the output weights perfectly. Neural networks solve this problem by making the basis functions themselves trainable. A hidden unit is a learned feature detector. A layer of hidden units is a learned representation. The output layer then performs regression or classification on top of that representation.

So the key conceptual transition is:

$$
\text{fixed basis functions}
\quad\longrightarrow\quad
\text{adaptive basis functions}.
$$

A simple two-layer neural network can therefore be read as a model that first learns useful nonlinear features and then combines them to produce the output.

## 0.2 Why Neural Networks Are More Difficult Than Linear Models

Linear regression and logistic regression have relatively simple training objectives. Linear regression has a quadratic error function and a closed-form solution. Logistic regression has no closed-form solution, but its negative log-likelihood is convex.

Neural networks are different. Once we put adaptive parameters inside hidden units, the error function is no longer convex. This creates three practical issues.

| Issue | Meaning | Why It Matters |
|------|---------|----------------|
| **Non-convexity** | There can be many local minima and saddle points. | Optimization may depend on initialization. |
| **Over-fitting** | A large network can fit noise as well as signal. | Regularization and validation become essential. |
| **Derivative computation** | There may be thousands or millions of parameters. | Efficient gradients are necessary for learning. |

The central algorithmic tool in this chapter is **error backpropagation**. Backpropagation is not a separate model. It is a method for efficiently computing derivatives of an error function with respect to all network parameters.

## 0.3 Roadmap of the Chapter

This chapter can be understood as answering seven linked questions.

| Section | Core Question | Main Idea |
|---------|---------------|-----------|
| §5.1 Feed-forward network functions | What function does a neural network represent? | A composition of linear transformations and nonlinear activations. |
| §5.2 Network training | Which error functions and optimization methods are used? | Maximum likelihood gives task-matched losses; training is nonlinear optimization. |
| §5.3 Error backpropagation | How do we compute gradients efficiently? | Use the chain rule recursively from outputs back to inputs. |
| §5.4 Hessian matrix | What second-order information can we use? | Curvature helps optimization, pruning, and Bayesian approximation. |
| §5.5 Regularization | How do we control network complexity? | Weight decay, early stopping, invariance methods, convolution, and weight sharing. |
| §5.6 Mixture density networks | How can networks predict full conditional densities? | A network outputs the parameters of a conditional mixture model. |
| §5.7 Bayesian neural networks | How do we represent uncertainty over weights? | Use posterior distributions and approximations such as Laplace. |

The main pattern is:

$$
\text{network function}
\longrightarrow
\text{error function}
\longrightarrow
\text{gradient computation}
\longrightarrow
\text{regularized or Bayesian prediction}.
$$

---

# §1 Feed-forward Network Functions

> 📖 Textbook §5.1 Feed-forward Network Functions; §5.1.1

## 1.1 From Fixed Basis Functions to Adaptive Basis Functions

Recall the generalized linear model form

$$
y(\mathbf{x},\mathbf{w})=f\left(\sum_{j=1}^M w_j\phi_j(\mathbf{x})\right),
$$

where $f(\cdot)$ is an output activation function. For regression, $f$ is usually the identity function. For classification, $f$ may be a sigmoid or softmax.

The key neural-network idea is to make each basis function $\phi_j(\mathbf{x})$ itself a parametric function. In a two-layer network, hidden unit $j$ first forms a linear combination of inputs:

$$
a_j=\sum_{i=1}^D w^{(1)}_{ji}x_i+w^{(1)}_{j0}.
$$

Then it applies a nonlinear activation function:

$$
z_j=h(a_j).
$$

The output unit then forms another linear combination:

$$
a_k=\sum_{j=1}^M w^{(2)}_{kj}z_j+w^{(2)}_{k0},
$$

and applies an output activation:

$$
y_k=f(a_k).
$$

Putting these together, a two-layer network for output $k$ has the form

$$
y_k(\mathbf{x},\mathbf{w})
=f\left(
\sum_{j=1}^M w^{(2)}_{kj}
 h\left(\sum_{i=1}^D w^{(1)}_{ji}x_i+w^{(1)}_{j0}\right)
+w^{(2)}_{k0}
\right).
$$

This expression is worth reading slowly. The first layer learns nonlinear features. The second layer combines those features. This is why neural networks are more flexible than fixed-basis models.

> ![Figure 5.1](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_1__textbook_fig_5_1_p228_two_layer_network_diagram.png)
>
> *Figure 5.1 (Textbook Fig. 5.1, p. 228): A two-layer feed-forward neural network. Inputs, hidden units, and outputs are nodes. Weights are links. Biases are represented by additional fixed inputs $x_0$ and $z_0$. The green arrows show that information moves forward from input to output.*

## 1.2 Activations, Hidden Units, Weights, and Biases

A hidden unit has two parts. The activation $a_j$ is the weighted input sum before nonlinearity. The output $z_j$ is the result after applying $h(\cdot)$.

Common hidden nonlinearities include:

$$
h(a)=\tanh(a),
$$

and

$$
h(a)=\sigma(a)=\frac{1}{1+\exp(-a)}.
$$

The activation function must be nonlinear. If all hidden units were linear and all output units were linear, then the whole network would collapse into one linear model. The composition of linear maps is still linear. Nonlinearity is what allows the network to represent curved decision boundaries or nonlinear regression functions.

Bias terms are important because they allow units to shift their activation thresholds. Without biases, every hidden unit boundary would pass through the origin in its input space. Including a bias lets each hidden unit respond in a more flexible location.

## Textbook Exercise 5.8: Derivative of the Tanh Activation

> ![Textbook Exercise 5.8](./CoursePR2026/Fig/Chapter_5/lecture_ex_5_8__textbook_ex_5_8_p285.png)
>
> *Textbook Exercise 5.8 (p. 285): Derive the derivative of the $\tanh$ activation in terms of its own value.*

The hyperbolic tangent can be written as

$$
\tanh(a)=\frac{e^a-e^{-a}}{e^a+e^{-a}}.
$$

It is often easier to use the identity

$$
\tanh(a)=2\sigma(2a)-1.
$$

Differentiate both sides:

$$
\frac{d}{da}\tanh(a)
=2\cdot \frac{d\sigma(2a)}{da}
=2\cdot 2\sigma(2a)\{1-\sigma(2a)\}.
$$

So

$$
\frac{d}{da}\tanh(a)
=4\sigma(2a)\{1-\sigma(2a)\}.
$$

Now express this using $\tanh(a)$. Since

$$
\sigma(2a)=\frac{1+\tanh(a)}{2},
$$

we obtain

$$
\frac{d}{da}\tanh(a)
=4\cdot
\frac{1+\tanh(a)}{2}
\cdot
\frac{1-\tanh(a)}{2}
=1-\tanh^2(a).
$$

Therefore

$$
\boxed{\frac{d}{da}\tanh(a)=1-\tanh^2(a).}
$$

This is useful in backpropagation because once the forward pass has computed $z=\tanh(a)$, the derivative can be computed as $1-z^2$ without recomputing exponentials.

## 1.3 Output Activations Must Match the Task

The output activation should be chosen according to the target distribution.

| Task | Output Activation | Typical Error Function |
|------|-------------------|------------------------|
| Regression with Gaussian noise | Identity | Sum-of-squares error |
| Binary classification | Logistic sigmoid | Binary cross-entropy |
| Multiclass classification | Softmax | Multiclass cross-entropy |

For regression, the network output can be any real number:

$$
y_k=a_k.
$$

For binary classification, the output must be between 0 and 1:

$$
y=\sigma(a)=\frac{1}{1+\exp(-a)}.
$$

For multiclass classification, the outputs must be nonnegative and sum to 1:

$$
y_k=\frac{\exp(a_k)}{\sum_j\exp(a_j)}.
$$

The important principle is that the network output should have the same mathematical form as the quantity being predicted. A class probability should be a valid probability. A real-valued regression output should not be artificially restricted to $[0,1]$ unless the target itself requires that.

## 1.4 General Feed-forward Topologies

The two-layer network in Figure 5.1 is only the simplest common architecture. More general feed-forward networks can have additional hidden layers, skip connections, and sparse connectivity. The defining property is that there are no directed cycles. Information flows from earlier units to later units.

> ![Figure 5.2](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_2__textbook_fig_5_2_p230_general_feed_forward_topology.png)
>
> *Figure 5.2 (Textbook Fig. 5.2, p. 230): A more general feed-forward topology. The network is still acyclic, but not every layer needs to be fully connected in the simple sequential way shown in Figure 5.1. Each hidden or output unit computes a nonlinear function of the units feeding into it.*

A general feed-forward unit computes

$$
z_k=h\left(\sum_j w_{kj}z_j\right),
$$

where the sum is over units that send connections to unit $k$. Biases can be included by adding an extra input unit with fixed value 1.

The phrase **feed-forward** is important. It means the network function can be evaluated by starting from the inputs and successively computing later units. Recurrent networks, which have loops, are not covered in this chapter.

## 1.5 Universal Approximation: What It Means and What It Does Not Mean

A two-layer network with enough hidden units and suitable nonlinear activation functions can approximate a very broad class of continuous functions on compact domains. This is called a **universal approximation property**.

This result is often misunderstood. It says that a sufficiently large network can represent a good approximation. It does not say that training will easily find that approximation. It also does not say that the network will generalize well from a small data set. Approximation ability, optimization, and generalization are different questions.

> ![Figure 5.3](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_3__textbook_fig_5_3_p231_universal_approximation_examples.png)
>
> *Figure 5.3 (Textbook Fig. 5.3, p. 231): A two-layer network with three hidden units approximates several different one-dimensional functions. The dashed curves show the individual hidden-unit contributions, and the red curve is their final combination. The figure illustrates how simple hidden-unit responses can work together to form more complex functions.*

The same idea applies to classification. Hidden units can create intermediate nonlinear boundaries. The output unit then combines them into a final decision surface.

> ![Figure 5.4](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_4__textbook_fig_5_4_p232_two_class_neural_network_decision_boundary.png)
>
> *Figure 5.4 (Textbook Fig. 5.4, p. 232): A two-class neural-network classifier. The dashed blue lines show hidden-unit contours, and the red curve shows the final $y=0.5$ decision boundary. The network uses several simple hidden-unit boundaries to build a nonlinear classifier.*

The main lesson is not that every problem needs a very large network. The lesson is that hidden units act as adaptive nonlinear features. If the hidden features are useful, the output layer can solve problems that were not linearly separable in the original input space.

## 1.6 Weight-space Symmetries

Neural networks often have many equivalent parameter settings that represent the same function. Two important symmetries occur in two-layer networks.

First, if the hidden activation is odd, such as $\tanh(a)$, then changing the sign of all incoming weights and the bias of one hidden unit reverses the sign of that hidden unit's output. If we also change the sign of its outgoing weights, the final network function remains unchanged.

Second, hidden units can be permuted. If we swap two hidden units and also swap all their associated incoming and outgoing weights, the network output remains the same.

For $M$ hidden units, this creates many equivalent solutions. A common count for this kind of symmetry is

$$
M!2^M.
$$

These symmetries matter especially in Bayesian model comparison because the posterior over weights can have many equivalent modes. For ordinary training, they mostly remind us that the parameter vector itself is not unique; what matters is the function represented by the network.

---

# §2 Network Training

> 📖 Textbook §5.2 Network Training; §5.2.1-§5.2.4

## 2.1 Training as Maximum Likelihood

Neural-network training is usually formulated as minimizing an error function. In this chapter, the cleanest way to understand the error function is through maximum likelihood.

The network defines a conditional distribution $p(\mathbf{t}\mid\mathbf{x},\mathbf{w})$. The training data likelihood is

$$
p(\mathcal{D}\mid\mathbf{w})=
\prod_{n=1}^N p(\mathbf{t}_n\mid\mathbf{x}_n,\mathbf{w}).
$$

Training by maximum likelihood means choosing $\mathbf{w}$ to maximize this likelihood. Equivalently, we minimize the negative log likelihood:

$$
E(\mathbf{w})=-\ln p(\mathcal{D}\mid\mathbf{w}).
$$

Different observation models give different error functions.

## 2.2 Regression: Gaussian Noise Gives Sum-of-Squares Error

For regression, suppose the target is generated by

$$
t=y(\mathbf{x},\mathbf{w})+\epsilon,
$$

where $\epsilon$ is Gaussian noise with precision $\beta$. Then

$$
p(t\mid\mathbf{x},\mathbf{w},\beta)=
\mathcal{N}\left(t\mid y(\mathbf{x},\mathbf{w}),\beta^{-1}\right).
$$

The negative log-likelihood, ignoring constants independent of $\mathbf{w}$, becomes

$$
E_D(\mathbf{w})=\frac{1}{2}\sum_{n=1}^N
\{y(\mathbf{x}_n,\mathbf{w})-t_n\}^2.
$$

Thus the familiar sum-of-squares error is not arbitrary. It corresponds to Gaussian observation noise.

For multiple independent regression outputs,

$$
E_D(\mathbf{w})=\frac{1}{2}\sum_{n=1}^N\sum_{k=1}^K
\{y_k(\mathbf{x}_n,\mathbf{w})-t_{nk}\}^2.
$$

## Textbook Exercise 5.2: Gaussian Regression Gives Sum-of-Squares

> ![Textbook Exercise 5.2](./CoursePR2026/Fig/Chapter_5/lecture_ex_5_2__textbook_ex_5_2_p284.png)
>
> *Textbook Exercise 5.2 (p. 284): Show that maximum likelihood under a Gaussian output model is equivalent to minimizing sum-of-squares error.*

For one training example with $K$ independent output dimensions, assume

$$
p(\mathbf{t}_n\mid\mathbf{x}_n,\mathbf{w})
=\prod_{k=1}^K
\mathcal{N}(t_{nk}\mid y_k(\mathbf{x}_n,\mathbf{w}),\beta^{-1}).
$$

The likelihood for all data points is the product over $n$ and $k$. Taking the negative log gives

$$
-\ln p(\mathcal{D}\mid\mathbf{w})
=
\frac{\beta}{2}
\sum_{n=1}^N\sum_{k=1}^K
\{t_{nk}-y_k(\mathbf{x}_n,\mathbf{w})\}^2
\text{constant}.
$$

The constant and the positive scale factor $\beta$ do not change the minimizing $\mathbf{w}$. Therefore maximum likelihood is equivalent to minimizing

$$
\boxed{
E_D(\mathbf{w})
=\frac{1}{2}
\sum_{n=1}^N\sum_{k=1}^K
\{y_k(\mathbf{x}_n,\mathbf{w})-t_{nk}\}^2.
}
$$

The important lesson is the same as in linear regression: squared error is not just a habit. It encodes a Gaussian noise assumption for continuous targets.

## 2.3 Binary Classification: Sigmoid Output and Cross-Entropy

For binary classification, the target is $t_n\in\{0,1\}$. If the network output is

$$
y_n=\sigma(a_n),
$$

then $y_n$ can be interpreted as

$$
y_n=p(t_n=1\mid\mathbf{x}_n,\mathbf{w}).
$$

The Bernoulli likelihood is

$$
p(t_n\mid\mathbf{x}_n,\mathbf{w})=y_n^{t_n}(1-y_n)^{1-t_n}.
$$

The negative log-likelihood is the binary cross-entropy:

$$
E(\mathbf{w})=-\sum_{n=1}^N
\{t_n\ln y_n+(1-t_n)\ln(1-y_n)\}.
$$

This is the same classification loss used in logistic regression, except that $a_n$ is now produced by a neural network rather than by a linear model.

## 2.4 Multiclass Classification: Softmax and Cross-Entropy

For $K$ classes, the output layer uses softmax:

$$
y_k(\mathbf{x},\mathbf{w})=
\frac{\exp(a_k)}{\sum_j\exp(a_j)}.
$$

With 1-of-$K$ targets $t_{nk}$, the likelihood for one example is

$$
p(\mathbf{t}_n\mid\mathbf{x}_n,\mathbf{w})=
\prod_{k=1}^K y_{nk}^{t_{nk}}.
$$

The negative log-likelihood is

$$
E(\mathbf{w})=-\sum_{n=1}^N\sum_{k=1}^K t_{nk}\ln y_{nk}.
$$

Again, the output activation and error function form a natural pair. For sigmoid plus Bernoulli likelihood, the derivative at the output has a simple form. For softmax plus multiclass cross-entropy, the derivative also has a simple form. This is why these pairings are standard.

## Textbook Exercise 5.5: Multiclass Likelihood and Cross-Entropy

> ![Textbook Exercise 5.5](./CoursePR2026/Fig/Chapter_5/lecture_ex_5_5__textbook_ex_5_5_p285.png)
>
> *Textbook Exercise 5.5 (p. 285): Show that maximum likelihood for multiclass probabilistic outputs gives the cross-entropy loss.*

Suppose the network output has the probability interpretation

$$
y_k(\mathbf{x},\mathbf{w})=p(t_k=1\mid\mathbf{x}).
$$

For a one-hot target vector $\mathbf{t}_n$, only one component is 1. The likelihood for one example is

$$
p(\mathbf{t}_n\mid\mathbf{x}_n,\mathbf{w})
=\prod_{k=1}^K y_{nk}^{t_{nk}}.
$$

If the correct class is $c$, then $t_{nc}=1$ and all other $t_{nk}=0$, so this product simply selects $y_{nc}$, the predicted probability of the correct class.

For the whole data set,

$$
p(\mathcal{D}\mid\mathbf{w})
=\prod_{n=1}^N\prod_{k=1}^K y_{nk}^{t_{nk}}.
$$

Taking negative log likelihood gives

$$
\boxed{
E(\mathbf{w})
=-\sum_{n=1}^N\sum_{k=1}^K t_{nk}\ln y_{nk}.
}
$$

This is exactly the multiclass cross-entropy. It strongly penalizes assigning low probability to the true class. That is why it is the standard loss for softmax classifiers.

## 2.5 Parameter Optimization and Error Surfaces

Once we choose an error function, training becomes an optimization problem:

$$
\mathbf{w}^*=\operatorname*{arg\,min}_{\mathbf{w}}E(\mathbf{w}).
$$

For neural networks, $E(\mathbf{w})$ is generally non-convex. It can have many stationary points, local minima, saddle points, and flat regions. This is a major difference from ordinary linear regression.

> ![Figure 5.5](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_5__textbook_fig_5_5_p236_error_surface_local_and_global_minima.png)
>
> *Figure 5.5 (Textbook Fig. 5.5, p. 236): The error function can be viewed as a surface over weight space. The point $\mathbf{w}_A$ is a local minimum, while $\mathbf{w}_B$ is the global minimum. At a point such as $\mathbf{w}_C$, the gradient $\nabla E$ points in the direction of steepest increase, so gradient descent moves in the opposite direction.*

A stationary point satisfies

$$
\nabla E(\mathbf{w})=0.
$$

But not every stationary point is a useful minimum. It may be a maximum or a saddle point. In practice, training often uses multiple random initializations because different initial points can lead to different final solutions.

## 2.6 Local Quadratic Approximation and the Hessian

Near a point $\mathbf{w}^*$, the error function can be approximated by a second-order Taylor expansion:

$$
E(\mathbf{w})\simeq
E(\mathbf{w}^*)
+\frac{1}{2}(\mathbf{w}-\mathbf{w}^*)^T
\mathbf{H}
(\mathbf{w}-\mathbf{w}^*),
$$

where $\mathbf{H}$ is the Hessian matrix evaluated at $\mathbf{w}^*$.

If $\mathbf{H}$ is positive definite, then $\mathbf{w}^*$ is a local minimum. The eigenvectors of $\mathbf{H}$ describe the main curvature directions, and the eigenvalues describe how sharp or flat the error surface is along those directions.

> ![Figure 5.6](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_6__textbook_fig_5_6_p239_quadratic_approximation_hessian_eigenvectors.png)
>
> *Figure 5.6 (Textbook Fig. 5.6, p. 239): Near a minimum, contours of constant error are approximately ellipses. Their axes align with the Hessian eigenvectors. Large eigenvalues correspond to sharp directions; small eigenvalues correspond to flat directions.*

This view helps explain why neural-network optimization can be difficult. If the surface is very steep in one direction and very flat in another, a simple gradient step size may be too large for the steep direction and too small for the flat direction.

## 2.7 Why Gradient Information Is Essential

Suppose a network has $W$ parameters. If we try to understand the local quadratic error surface without using gradients, we would need roughly $O(W^2)$ pieces of information to locate the minimum. For large networks, this is impossible.

The gradient gives first-order information about all parameters at once. Backpropagation can compute the full gradient in time proportional to the number of weights, up to a constant factor comparable to forward evaluation. This is why gradient-based methods make neural-network training practical.

## 2.8 Gradient Descent and Stochastic Gradient Descent

The simplest gradient method is batch gradient descent:

$$
\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta\nabla E(\mathbf{w}^{(\tau)}),
$$

where $\eta$ is the learning rate.

Batch gradient descent uses all training examples before each update. This can be inefficient for large data sets. A more practical approach is **stochastic gradient descent** or **sequential gradient descent**, in which we update after one example or a small mini-batch:

$$
\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-
\eta\nabla E_n(\mathbf{w}^{(\tau)}).
$$

Stochastic updates are noisy, but the noise can be helpful. It can reduce wasted computation on redundant data and can help the optimizer move away from some shallow stationary regions. Modern deep learning uses mini-batch versions of this same basic idea.

Other optimization methods, such as conjugate gradients and quasi-Newton methods, use more information than plain gradient descent. The textbook mentions them because they can be useful for moderate-sized networks. The central requirement, however, remains the same: we need efficient gradients.

---

# §3 Error Backpropagation

> 📖 Textbook §5.3 Error Backpropagation; §5.3.1-§5.3.4

## 3.1 What Backpropagation Computes

Backpropagation computes derivatives of an error function with respect to network parameters. It is simply the chain rule applied efficiently to a computational graph.

For a weight $w_{ji}$ from unit $i$ to unit $j$, the activation of unit $j$ is

$$
a_j=\sum_i w_{ji}z_i.
$$

The derivative of the error with respect to this weight is

$$
\frac{\partial E}{\partial w_{ji}}
=
\frac{\partial E}{\partial a_j}
\frac{\partial a_j}{\partial w_{ji}}.
$$

Because

$$
\frac{\partial a_j}{\partial w_{ji}}=z_i,
$$

we get

$$
\frac{\partial E}{\partial w_{ji}}=\delta_j z_i,
$$

where

$$
\delta_j=\frac{\partial E}{\partial a_j}.
$$

This formula is the key. Once we know the $\delta$ value for each unit, the derivative for each incoming weight is just the product of the unit's $\delta$ and the input feeding that weight.

## 3.2 Output-unit Deltas

For an output unit, $\delta_k$ depends on the chosen output activation and error function. With the standard pairings, it becomes very simple.

For linear outputs with sum-of-squares error,

$$
\delta_k=y_k-t_k.
$$

For sigmoid output with binary cross-entropy,

$$
\delta=y-t.
$$

For softmax output with multiclass cross-entropy,

$$
\delta_k=y_k-t_k.
$$

This repeated pattern is important:

$$
\boxed{\text{output delta} = \text{prediction} - \text{target}.}
$$

The output layer therefore produces the error signal that will be propagated backwards through the network.

## Textbook Exercise 5.6: Sigmoid Cross-Entropy Output Delta

> ![Textbook Exercise 5.6](./CoursePR2026/Fig/Chapter_5/lecture_ex_5_6__textbook_ex_5_6_p285.png)
>
> *Textbook Exercise 5.6 (p. 285): Show that the derivative with respect to the sigmoid output activation satisfies the simple output-delta formula.*

For one binary classification example,

$$
E=-\{t\ln y+(1-t)\ln(1-y)\},
\qquad
y=\sigma(a).
$$

First differentiate the loss with respect to $y$:

$$
\frac{\partial E}{\partial y}
=-\frac{t}{y}+\frac{1-t}{1-y}.
$$

For the sigmoid,

$$
\frac{\partial y}{\partial a}=y(1-y).
$$

By the chain rule,

$$
\frac{\partial E}{\partial a}
=
\left(
-\frac{t}{y}+\frac{1-t}{1-y}
\right)y(1-y).
$$

Simplify:

$$
\frac{\partial E}{\partial a}
=-t(1-y)+(1-t)y
=y-t.
$$

Therefore the output delta is

$$
\boxed{\delta=\frac{\partial E}{\partial a}=y-t.}
$$

This is why sigmoid cross-entropy is such a convenient pairing. The output-layer error signal is just predicted probability minus target label.

## 3.3 Hidden-unit Deltas

For a hidden unit $j$, the error does not directly compare $z_j$ to a target. Instead, hidden unit $j$ influences later units. By the chain rule,

$$
\delta_j
=
\frac{\partial E}{\partial a_j}
=
\sum_k
\frac{\partial E}{\partial a_k}
\frac{\partial a_k}{\partial a_j}.
$$

Since

$$
a_k=\sum_j w_{kj}z_j
=\sum_j w_{kj}h(a_j),
$$

we have

$$
\frac{\partial a_k}{\partial a_j}=w_{kj}h'(a_j).
$$

Therefore

$$
\delta_j=h'(a_j)\sum_k w_{kj}\delta_k.
$$

This is the backpropagation formula.

> ![Figure 5.7](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_7__textbook_fig_5_7_p244_backpropagation_hidden_unit_delta.png)
>
> *Figure 5.7 (Textbook Fig. 5.7, p. 244): The hidden-unit error signal $\delta_j$ is computed from the downstream $\delta_k$ values. The blue arrow shows the forward flow of activations, and the red arrows show the backward flow of error information.*

The formula has a simple interpretation. A hidden unit receives blame from every later unit it influenced. The amount of blame is weighted by the connection strength $w_{kj}$ and scaled by the local derivative $h'(a_j)$.

## 3.4 The Four-step Backpropagation Algorithm

For one training example, backpropagation can be summarized as follows.

1. **Forward pass**: Apply the input $\mathbf{x}_n$ and compute all activations and outputs.
2. **Output deltas**: Compute $\delta_k$ for all output units using the derivative of the error with respect to output activations.
3. **Backward pass**: For each hidden layer, compute hidden deltas using

$$
\delta_j=h'(a_j)\sum_k w_{kj}\delta_k.
$$

4. **Weight derivatives**: Compute

$$
\frac{\partial E}{\partial w_{ji}}=\delta_jz_i.
$$

For a batch error function, we sum these derivatives over training examples.

## 3.5 A Simple Two-layer Example

For the two-layer network in Figure 5.1, suppose the output units are linear and the error is sum-of-squares. For output unit $k$,

$$
\delta_k=y_k-t_k.
$$

The derivative of the second-layer weight is

$$
\frac{\partial E}{\partial w^{(2)}_{kj}}=\delta_k z_j.
$$

For hidden unit $j$,

$$
\delta_j=h'(a_j)\sum_k w^{(2)}_{kj}\delta_k.
$$

The derivative of the first-layer weight is

$$
\frac{\partial E}{\partial w^{(1)}_{ji}}=\delta_j x_i.
$$

This example shows why backpropagation is efficient. The hidden-unit delta is computed once and then reused for all incoming weights to that hidden unit.

## 3.6 Efficiency and Gradient Checking

A naive numerical derivative perturbs one parameter at a time. If there are $W$ parameters, this needs $O(W)$ forward evaluations per data point. Since each forward evaluation itself costs about $O(W)$, the total cost becomes $O(W^2)$.

Backpropagation computes the full gradient in $O(W)$ time per data point. This is the fundamental reason it is practical.

However, numerical derivatives are still useful for **gradient checking**. A central-difference approximation is

$$
\frac{\partial E}{\partial w_i}
\simeq
\frac{E(w_i+\epsilon)-E(w_i-\epsilon)}{2\epsilon}.
$$

This is too expensive for routine training, but it is very useful for checking whether a backpropagation implementation is correct on small test cases.

## 3.7 The Jacobian Matrix

Backpropagation can also compute derivatives of outputs with respect to inputs. The Jacobian matrix has entries

$$
J_{ki}=\frac{\partial y_k}{\partial x_i}.
$$

The Jacobian measures how sensitive each output is to each input. This is useful in modular systems, sensitivity analysis, and error propagation.

> ![Figure 5.8](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_8__textbook_fig_5_8_p247_modular_pattern_recognition_jacobian.png)
>
> *Figure 5.8 (Textbook Fig. 5.8, p. 247): A modular pattern-recognition system. The Jacobian of a later module can be used to propagate error derivatives backward through earlier modules. This is the same chain-rule logic as ordinary backpropagation.*

The broader lesson is that backpropagation is not limited to ordinary weight gradients. It is a general way to propagate derivative information through a differentiable computational system.

---

# §4 The Hessian Matrix

> 📖 Textbook §5.4 The Hessian Matrix; §5.4.1-§5.4.6  
> This section is an introduction only. It can be shortened if teaching time is limited.
>
> **Teaching choice.** The Hessian section is useful for understanding curvature, second-order optimization, pruning, and Bayesian approximations. For a modern ML lecture, it is usually enough to explain the local quadratic picture and why full Hessians are expensive. The exact Hessian backpropagation details can be skipped.

## 4.1 Why the Hessian Matters

The gradient tells us the local slope of the error function. The Hessian tells us the local curvature:

$$
H_{ij}=\frac{\partial^2E}{\partial w_i\partial w_j}.
$$

The Hessian is useful for several reasons.

| Use | Why Curvature Helps |
|-----|---------------------|
| Second-order optimization | It tells us how to scale steps in different directions. |
| Re-training after small changes | It approximates how the optimum moves when data or regularization changes. |
| Pruning | It helps estimate which weights can be removed with little error increase. |
| Bayesian neural networks | Its inverse gives a local covariance approximation to the posterior. |
| Evidence approximation | Its determinant appears in the Laplace approximation to model evidence. |

The difficulty is computational. If the network has $W$ parameters, then the Hessian is a $W\times W$ matrix. Storing and computing it can be expensive.

## 4.2 Diagonal Approximation

The simplest approximation keeps only diagonal terms and ignores off-diagonal terms:

$$
\mathbf{H}\approx \operatorname{diag}(H_{11},\ldots,H_{WW}).
$$

This makes inversion trivial. However, neural-network Hessians are often strongly non-diagonal. Ignoring interactions between parameters can therefore be inaccurate.

This approximation is useful mainly as a computationally cheap heuristic.

## 4.3 Outer-product Approximation

For sum-of-squares errors, the Hessian contains terms involving first derivatives and second derivatives of network outputs. Near a good solution, one often approximates the Hessian by the outer product of gradients:

$$
\mathbf{H}\approx \sum_n \mathbf{b}_n\mathbf{b}_n^T,
$$

where $\mathbf{b}_n$ is a vector of first derivatives of the network output for data point $n$ with respect to the weights.

This is related to Gauss-Newton and Levenberg-Marquardt methods. It can work well when residual errors are small, but it is still an approximation.

## 4.4 Inverse Hessian and Woodbury-style Updates

Sometimes we need $\mathbf{H}^{-1}$ rather than $\mathbf{H}$. For example, the inverse Hessian appears in approximate posterior covariances. Direct inversion costs $O(W^3)$, which is expensive.

Sequential update formulas based on matrix identities can build or update an inverse Hessian more efficiently when new terms are added one at a time. The main idea is that a rank-one update to a matrix can be translated into a rank-one update to its inverse.

The important conceptual point is that curvature information is powerful but expensive. Practical methods try to approximate it or apply it indirectly.

## 4.5 Finite Differences and Exact Hessian Backpropagation

A simple way to estimate Hessian-vector or Hessian entries is to apply finite differences to gradients. For example,

$$
\frac{\partial^2E}{\partial w_i\partial w_j}
\simeq
\frac{1}{2\epsilon}
\left(
\frac{\partial E}{\partial w_i}(w_j+\epsilon)
-
\frac{\partial E}{\partial w_i}(w_j-\epsilon)
\right).
$$

This is more efficient than finite-differencing the original error for every pair, but it can still be expensive.

The textbook also describes exact Hessian evaluation using extended backpropagation. This is mathematically elegant, but the cost scales like $O(W^2)$) per pattern, so it is mainly useful for smaller networks or for theoretical understanding.

## 4.6 Fast Multiplication by the Hessian

Many algorithms do not require the full Hessian. They only need products such as

$$
\mathbf{H}\mathbf{v}
$$

for a given vector $\mathbf{v}$. Such products can be computed more efficiently than explicitly forming the whole Hessian. This is useful in second-order optimization and in approximate Bayesian methods.

For this course, the main takeaway is simple:

$$
\text{gradient} = \text{first-order direction},
\qquad
\text{Hessian} = \text{second-order curvature}.
$$

Backpropagation makes gradients cheap. Hessian information is more costly, so we usually approximate it or use it indirectly.

---

# §5 Regularization in Neural Networks

> 📖 Textbook §5.5 Regularization in Neural Networks; §5.5.1-§5.5.7
>
> **Teaching choice.** Keep this section practical. Students should understand weight decay, early stopping, data augmentation, invariance, and convolution/weight sharing. Tangent propagation and soft weight sharing are elegant but less central today, so they can be presented briefly as historical or conceptual extensions.

## 5.1 Why Regularization Is Needed

A neural network with many hidden units can represent complicated functions. This is useful when the target relationship is complicated, but dangerous when the training data are limited or noisy.

The same bias-variance trade-off from Chapter 3 appears again. A small network may under-fit. A large network may over-fit. The difference is that neural networks add non-convexity, so different random initializations may produce different fits even for the same architecture.

> ![Figure 5.9](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_9__textbook_fig_5_9_p257_network_size_sinusoidal_fits.png)
>
> *Figure 5.9 (Textbook Fig. 5.9, p. 257): Two-layer networks with different numbers of hidden units are fitted to a small sinusoidal data set. A small network is too rigid, while a larger network can represent more detailed variation. This mirrors the polynomial curve-fitting example from Chapter 1.*

Increasing model size does not always lead to smooth improvement, because training can get stuck in different local minima.

> ![Figure 5.10](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_10__textbook_fig_5_10_p257_test_error_hidden_units_local_minima.png)
>
> *Figure 5.10 (Textbook Fig. 5.10, p. 257): Test error is plotted against the number of hidden units for many random starts. The variation across starts shows the effect of local minima. This is why neural-network training often uses multiple initializations or stochastic optimization.*

A practical strategy is not simply to choose the smallest possible network. Instead, we often use a flexible network and control it with regularization.

## 5.2 Weight Decay

The simplest regularizer is weight decay:

$$
\widetilde{E}(\mathbf{w})=E(\mathbf{w})+\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}.
$$

This penalizes large weights. In Bayesian terms, it corresponds to a zero-mean isotropic Gaussian prior over weights:

$$
p(\mathbf{w}\mid\alpha)=\mathcal{N}(\mathbf{w}\mid\mathbf{0},\alpha^{-1}\mathbf{I}).
$$

The regularization coefficient controls the trade-off between data fit and parameter size. Large $\lambda$ gives smoother, simpler functions. Small $\lambda$ allows more flexible fits.

However, ordinary weight decay treats all weights as if they should have the same scale. This may be inappropriate because first-layer weights, first-layer biases, second-layer weights, and second-layer biases can play different roles.

## 5.3 Consistent Gaussian Priors

A more careful Bayesian regularization scheme can use different prior precisions for different groups of parameters. For example, we may use separate hyperparameters for:

| Parameter Group | Example Hyperparameter |
|-----------------|------------------------|
| First-layer weights | $\alpha_1^w$ |
| First-layer biases | $\alpha_1^b$ |
| Second-layer weights | $\alpha_2^w$ |
| Second-layer biases | $\alpha_2^b$ |

This allows the model to control different aspects of the function separately.

> ![Figure 5.11](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_11__textbook_fig_5_11_p260_hyperparameters_weight_priors.png)
>
> *Figure 5.11 (Textbook Fig. 5.11, p. 260): Different hyperparameters control different properties of the functions generated by a two-layer network. Some affect vertical scale; others affect horizontal variation or offsets. This shows why a single weight-decay parameter can be too crude.*

The general principle is that regularization should respect the structure of the model. If different parameters have different roles, grouping them under different priors can be more consistent than giving them all the same penalty.

## 5.4 Early Stopping

Early stopping is one of the simplest and most useful forms of regularization. During training, we monitor error on a validation set. The training error usually keeps decreasing, but the validation error may eventually increase as the model begins to over-fit. We stop training at the point of minimum validation error.

> ![Figure 5.12](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_12__textbook_fig_5_12_p261_early_stopping_training_validation_error.png)
>
> *Figure 5.12 (Textbook Fig. 5.12, p. 261): Training error decreases steadily, but validation error reaches a minimum and then begins to rise. The vertical dashed line indicates the early stopping point that gives the best validation performance.*

Early stopping can be interpreted as a form of regularization. If training begins from small weights, then early iterations often learn simple, large-scale structure first. Later iterations may fit finer details and eventually noise.

> ![Figure 5.13](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_13__textbook_fig_5_13_p262_early_stopping_weight_decay_geometry.png)
>
> *Figure 5.13 (Textbook Fig. 5.13, p. 262): In a quadratic error surface, stopping early can land near a point similar to the solution found by weight decay. Both methods prevent the weights from moving all the way to the unregularized minimum.*

Early stopping is easy to implement, but it requires a validation set. If data are scarce, holding out validation data can be costly. Cross-validation can help, but it increases computational expense.

## 5.5 Invariances

Many pattern-recognition tasks have natural invariances. For handwritten digit recognition, the class should not change if the digit is slightly shifted, scaled, or warped. For object recognition, small translations or rotations may not change the object identity.

There are four broad ways to handle invariance.

| Approach | Idea |
|----------|------|
| Data augmentation | Add transformed examples to the training set. |
| Tangent regularization | Penalize sensitivity to known transformations. |
| Invariant features | Preprocess inputs into features that are already invariant. |
| Invariant architecture | Build the invariance into the model structure, for example using convolution and pooling. |

Data augmentation is often the easiest method. We create transformed copies of training inputs and require the model to give the same target. The textbook illustrates this with synthetic warping of handwritten digits.

> ![Figure 5.14](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_14__textbook_fig_5_14_p263_synthetic_warping_digit.png)
>
> *Figure 5.14 (Textbook Fig. 5.14, p. 263): A handwritten digit is synthetically warped. The original digit is on the left; warped examples and their displacement fields are shown on the right. Such transformed data can encourage invariance to realistic input variation.*

## 5.6 Tangent Propagation

Tangent propagation is a more direct regularization method for invariance. Suppose a transformation of the input is controlled by a continuous parameter $\xi$, such as a small rotation angle. As $\xi$ changes, the input moves along a manifold in input space.

Locally, the direction of this movement is described by a tangent vector

$$
\boldsymbol{\tau}_n=\left.\frac{\partial \mathbf{s}(\mathbf{x}_n,\xi)}{\partial \xi}\right|_{\xi=0},
$$

where $\mathbf{s}(\mathbf{x},\xi)$ is the transformed input.

> ![Figure 5.15](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_15__textbook_fig_5_15_p263_tangent_manifold_transformation.png)
>
> *Figure 5.15 (Textbook Fig. 5.15, p. 263): A continuous transformation moves an input point along a manifold $\mathcal{M}$ in input space. Locally, this motion is approximated by a tangent vector $\boldsymbol{\tau}_n$.*

If the model should be invariant along this direction, then the output should not change when the input moves slightly along $\boldsymbol{\tau}_n$. Thus we penalize the directional derivative of the network output:

$$
\Omega=\frac{1}{2}\sum_n
\left\|\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\boldsymbol{\tau}_n\right\|^2.
$$

The regularized error becomes

$$
\widetilde{E}=E+\lambda\Omega.
$$

> ![Figure 5.16](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_16__textbook_fig_5_16_p265_tangent_vector_digit_rotation.png)
>
> *Figure 5.16 (Textbook Fig. 5.16, p. 265): A tangent vector approximates the effect of a small clockwise rotation of a digit. The tangent approximation is local, so it is accurate for small transformations but not for large ones.*

Tangent propagation is elegant because it uses knowledge of the desired invariance without explicitly creating many transformed copies. Its limitation is that it works locally and requires tangent vectors to be known or computed.

## 5.7 Training with Transformed Data

Training with transformed data is closely related to tangent propagation. If we generate many small transformations of each input and train the network to produce the same target, then in the limit of small transformations this encourages the output to be insensitive along transformation directions.

In other words:

$$
\text{many small augmented examples}
\quad\approx\quad
\text{penalty on tangent sensitivity}.
$$

This gives a useful conceptual link between data augmentation and regularization. Data augmentation is often easier to implement; tangent propagation gives a more analytical view of what augmentation is encouraging.

## 5.8 Convolutional Networks

Convolutional networks build some invariance into the architecture. They use three main ideas.

| Mechanism | Meaning |
|-----------|---------|
| Local receptive fields | A hidden unit looks only at a small region of the input. |
| Weight sharing | The same weights are used at many spatial locations. |
| Subsampling / pooling | Local responses are summarized to reduce sensitivity to small shifts. |

> ![Figure 5.17](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_17__textbook_fig_5_17_p268_convolutional_network_receptive_fields.png)
>
> *Figure 5.17 (Textbook Fig. 5.17, p. 268): A convolutional layer uses local receptive fields and shared weights. A subsampling layer then summarizes responses. This architecture is especially natural for images because local patterns can occur at different spatial locations.*

Weight sharing dramatically reduces the number of parameters. For example, if a feature detector uses a $5\times5$ patch, the same 25 weights can be applied across many image locations. This makes the model more compact and encourages translation equivariance. Pooling or subsampling then gives approximate translation invariance.

Although modern convolutional networks are much deeper than the simple illustration in Figure 5.17, the core principles are the same.

## 5.9 Soft Weight Sharing

Ordinary weight decay encourages weights to be near zero. Soft weight sharing encourages weights to form clusters. This is done by placing a mixture-of-Gaussians prior over weights:

$$
p(w)=\sum_{j=1}^M \pi_j\mathcal{N}(w\mid\mu_j,\sigma_j^2).
$$

Each weight is softly assigned to mixture components through responsibilities. During training, weights are encouraged to move toward learned cluster centers rather than simply toward zero.

This can produce a kind of parameter compression or structure discovery. The method is more complex than ordinary weight decay because it introduces additional mixture parameters and responsibilities, but it illustrates a broader point: regularization can express more than just “make weights small.” It can encode assumptions about useful parameter structure.

---

# §6 Mixture Density Networks

> 📖 Textbook §5.6 Mixture Density Networks
>
> **Teaching choice.** Treat mixture density networks as an optional advanced example. The important idea is that a neural network can output a whole conditional distribution, not just a mean. The derivative formulas for mixture parameters are not necessary for a first pass.

## 6.1 Why Ordinary Regression Can Fail for Inverse Problems

In ordinary regression with sum-of-squares error, the network learns the conditional mean:

$$
y(\mathbf{x})\approx \mathbb{E}[\mathbf{t}\mid\mathbf{x}].
$$

This is appropriate when $p(\mathbf{t}\mid\mathbf{x})$ is roughly unimodal and Gaussian-like. But in inverse problems, the conditional distribution can be multimodal.

A simple example is a robot arm. In the forward problem, joint angles determine the end-effector position. In the inverse problem, a desired end-effector position may correspond to multiple different joint-angle configurations.

> ![Figure 5.18](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_18__textbook_fig_5_18_p272_two_link_robot_arm_inverse_problem.png)
>
> *Figure 5.18 (Textbook Fig. 5.18, p. 272): A two-link robot arm. The forward kinematics has a unique end-effector position for given joint angles, but the inverse problem can have two solutions: elbow up and elbow down.*

If a model predicts only the conditional mean, it may output an average of two valid solutions. That average may itself be invalid.

> ![Figure 5.19](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_19__textbook_fig_5_19_p273_forward_inverse_problem_multimodality.png)
>
> *Figure 5.19 (Textbook Fig. 5.19, p. 273): A standard two-layer network fits the forward problem reasonably well, but performs poorly on the inverse problem. The inverse data are multimodal, so a single conditional mean is not enough.*

This motivates predicting a full conditional density rather than a single point estimate.

## 6.2 The Mixture Density Network Model

A mixture density network models

$$
p(\mathbf{t}\mid\mathbf{x})
$$

as a mixture distribution whose parameters are produced by a neural network. For a scalar target and Gaussian components,

$$
p(t\mid\mathbf{x})=
\sum_{k=1}^K \pi_k(\mathbf{x})
\mathcal{N}\left(t\mid\mu_k(\mathbf{x}),\sigma_k^2(\mathbf{x})\right).
$$

The network outputs the parameters of the mixture:

| Network Output | Constraint | Common Activation |
|----------------|------------|-------------------|
| Mixing coefficients $\pi_k(\mathbf{x})$ | $\pi_k\geq0$, $\sum_k\pi_k=1$ | Softmax |
| Means $\mu_k(\mathbf{x})$ | Real-valued | Identity |
| Standard deviations $\sigma_k(\mathbf{x})$ | Positive | Exponential |

> ![Figure 5.20](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_20__textbook_fig_5_20_p274_mixture_density_network_architecture.png)
>
> *Figure 5.20 (Textbook Fig. 5.20, p. 274): A mixture density network takes $\mathbf{x}$ as input and outputs the parameters $\boldsymbol{\theta}$ of a conditional mixture model. The result is a flexible conditional density $p(t\mid\mathbf{x})$ rather than a single prediction.*

The mixing coefficients can be represented as

$$
\pi_k(\mathbf{x})=
\frac{\exp(\eta_k(\mathbf{x}))}{\sum_j\exp(\eta_j(\mathbf{x}))}.
$$

The variances can be made positive by writing

$$
\sigma_k(\mathbf{x})=\exp(s_k(\mathbf{x})).
$$

The means can be direct network outputs:

$$
\mu_k(\mathbf{x})=m_k(\mathbf{x}).
$$

## 6.3 Error Function and Responsibilities

The training objective is the negative log-likelihood:

$$
E(\mathbf{w})=-\sum_{n=1}^N
\ln\left\{
\sum_{k=1}^K \pi_k(\mathbf{x}_n)
\mathcal{N}\left(t_n\mid\mu_k(\mathbf{x}_n),\sigma_k^2(\mathbf{x}_n)\right)
\right\}.
$$

As with ordinary mixture models, we can define responsibilities:

$$
\gamma_{nk}
=
\frac{
\pi_k(\mathbf{x}_n)\mathcal{N}\left(t_n\mid\mu_k(\mathbf{x}_n),\sigma_k^2(\mathbf{x}_n)\right)
}{
\sum_j \pi_j(\mathbf{x}_n)\mathcal{N}\left(t_n\mid\mu_j(\mathbf{x}_n),\sigma_j^2(\mathbf{x}_n)\right)
}.
$$

The responsibility $\gamma_{nk}$ says how much component $k$ explains target $t_n$ at input $\mathbf{x}_n$.

Training can then use backpropagation because the loss is differentiable with respect to the network outputs. The details are more involved than ordinary regression, but the structure is familiar: compute output derivatives, then backpropagate them through the network.

## 6.4 What the Trained Model Gives Us

After training, the model gives a full conditional density for every input. We can extract several quantities.

The conditional mean is

$$
\mathbb{E}[t\mid\mathbf{x}]
=\sum_{k=1}^K \pi_k(\mathbf{x})\mu_k(\mathbf{x}).
$$

The conditional variance combines within-component variance and between-component variation:

$$
\operatorname{var}[t\mid\mathbf{x}]
=\sum_{k=1}^K\pi_k(\mathbf{x})
\left\{\sigma_k^2(\mathbf{x})+\mu_k^2(\mathbf{x})\right\}
-
\left(\sum_{k=1}^K\pi_k(\mathbf{x})\mu_k(\mathbf{x})\right)^2.
$$

The conditional mode can be found by maximizing $p(t\mid\mathbf{x})$ with respect to $t$. For multimodal inverse problems, the mode may be more useful than the mean because it corresponds to one likely solution rather than an average of incompatible solutions.

> ![Figure 5.21](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_21__textbook_fig_5_21_p276_mixture_density_network_outputs.png)
>
> *Figure 5.21 (Textbook Fig. 5.21, p. 276): A mixture density network trained on the inverse problem. It learns input-dependent mixing coefficients, means, and conditional density contours. The conditional mode follows the branches of the data much better than an ordinary conditional-mean predictor.*

The core lesson is that a neural network does not have to output only a point prediction. It can output the parameters of a probabilistic model.

---

# §7 Bayesian Neural Networks

> 📖 Textbook §5.7 Bayesian Neural Networks; §5.7.1-§5.7.3
>
> **Teaching choice.** This section is advanced. Keep the main idea: standard neural networks give one fitted parameter vector, while Bayesian neural networks average over plausible weights to express uncertainty. The Laplace/evidence equations can be shown as a conceptual bridge to approximate inference rather than derived in detail.

## 7.1 Why Bayesian Neural Networks Are Hard

In Bayesian linear regression, a Gaussian prior and Gaussian likelihood produce an exact Gaussian posterior over weights. Neural networks are harder because the mapping $y(\mathbf{x},\mathbf{w})$ is nonlinear in $\mathbf{w}$. The posterior is

$$
p(\mathbf{w}\mid\mathcal{D})
\propto
p(\mathcal{D}\mid\mathbf{w})p(\mathbf{w}),
$$

but this posterior is generally non-Gaussian and multimodal.

A full Bayesian prediction requires marginalization over weights:

$$
p(\mathbf{t}\mid\mathbf{x},\mathcal{D})
=
\int p(\mathbf{t}\mid\mathbf{x},\mathbf{w})
 p(\mathbf{w}\mid\mathcal{D})\,d\mathbf{w}.
$$

This integral is usually intractable. The textbook therefore uses the Laplace approximation as a practical local approximation.

## 7.2 Laplace Approximation for Regression Networks

The first step is to find a mode of the posterior:

$$
\mathbf{w}_{\mathrm{MAP}}
=\operatorname*{arg\,max}_{\mathbf{w}}p(\mathbf{w}\mid\mathcal{D}).
$$

Equivalently, minimize the regularized error function. For Gaussian regression noise and a Gaussian prior,

$$
E(\mathbf{w})
=\frac{\beta}{2}\sum_{n=1}^N
\{y(\mathbf{x}_n,\mathbf{w})-t_n\}^2
+\frac{\alpha}{2}\mathbf{w}^T\mathbf{w}.
$$

Around $\mathbf{w}_{\mathrm{MAP}}$, approximate the posterior by a Gaussian:

$$
q(\mathbf{w})=
\mathcal{N}(\mathbf{w}\mid\mathbf{w}_{\mathrm{MAP}},\mathbf{A}^{-1}),
$$

where

$$
\mathbf{A}=\nabla\nabla E(\mathbf{w})\big|_{\mathbf{w}=\mathbf{w}_{\mathrm{MAP}}}.
$$

For a new input $\mathbf{x}$, linearize the network output around $\mathbf{w}_{\mathrm{MAP}}$:

$$
y(\mathbf{x},\mathbf{w})\simeq
 y(\mathbf{x},\mathbf{w}_{\mathrm{MAP}})
+\mathbf{g}^T(\mathbf{w}-\mathbf{w}_{\mathrm{MAP}}),
$$

where

$$
\mathbf{g}=\nabla_{\mathbf{w}}y(\mathbf{x},\mathbf{w})\big|_{\mathbf{w}=\mathbf{w}_{\mathrm{MAP}}}.
$$

Because $\mathbf{w}$ is approximately Gaussian and the output has been linearized, the predictive distribution becomes approximately Gaussian. Its variance contains both observation noise and parameter uncertainty.

## 7.3 Hyperparameter Optimization by Evidence

The hyperparameters $\alpha$ and $\beta$ control prior precision and noise precision. In the evidence framework, we choose them by maximizing the marginal likelihood:

$$
p(\mathcal{D}\mid\alpha,\beta)=
\int p(\mathcal{D}\mid\mathbf{w},\beta)p(\mathbf{w}\mid\alpha)\,d\mathbf{w}.
$$

This integral is approximated using Laplace's method. The resulting update rules have the same broad interpretation as in Bayesian linear regression:

- $\alpha$ controls how strongly weights are shrunk toward zero.
- $\beta$ controls the assumed noise level.
- The effective number of parameters $\gamma$ measures how many weight directions are well determined by the data.

For neural networks, the evidence approximation is only local because the posterior is not globally Gaussian. Still, it provides a principled way to tune regularization.

## 7.4 Bayesian Neural Networks for Classification

For binary classification, a network with sigmoid output gives

$$
y(\mathbf{x},\mathbf{w})=
\sigma(a(\mathbf{x},\mathbf{w})).
$$

The likelihood is Bernoulli and the prior over weights is often Gaussian. As in Bayesian logistic regression, the posterior over weights is approximated by Laplace.

> ![Figure 5.22](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_22__textbook_fig_5_22_p283_evidence_regularization_classification.png)
>
> *Figure 5.22 (Textbook Fig. 5.22, p. 283): Evidence-based regularization reduces over-fitting in a two-class neural-network classifier. The black curve is the unregularized maximum-likelihood result, while the red curve includes evidence-optimized regularization.*

For prediction, we need to integrate over weight uncertainty:

$$
p(t=1\mid\mathbf{x},\mathcal{D})
=
\int \sigma(a(\mathbf{x},\mathbf{w}))q(\mathbf{w})\,d\mathbf{w}.
$$

As in Chapter 4, we reduce this to a one-dimensional integral over the activation $a$. If $a$ is approximately Gaussian with mean $\mu_a$ and variance $\sigma_a^2$, then

$$
p(t=1\mid\mathbf{x},\mathcal{D})
\simeq
\sigma\left(\kappa(\sigma_a^2)\mu_a\right),
$$

where

$$
\kappa(\sigma^2)=\left(1+\frac{\pi\sigma^2}{8}\right)^{-1/2}.
$$

Because $0<\kappa(\sigma^2)\leq1$, weight uncertainty pulls the effective logit toward zero. Thus predictions become less extreme when the posterior uncertainty is large.

> ![Figure 5.23](./CoursePR2026/Fig/Chapter_5/lecture_fig_5_23__textbook_fig_5_23_p284_bayesian_neural_network_laplace_classification.png)
>
> *Figure 5.23 (Textbook Fig. 5.23, p. 284): Bayesian marginalization spreads the probability contours and makes predictions less overconfident. The $y=0.5$ boundary is largely unchanged, but probabilities away from the boundary are moderated toward 0.5.*

The main message is that Bayesian neural networks do not merely output a fitted function. They try to average over plausible functions. This is more computationally difficult, but it gives a better representation of uncertainty.

---

# §8 Chapter Summary, Figure Checklist, Exercises, and Teaching Flow

## 8.1 Chapter Summary

Chapter 5 introduces neural networks as adaptive-basis-function models.

First, feed-forward networks are built from linear combinations and nonlinear activation functions. A two-layer network learns hidden features and combines them at the output layer. Different output activations match different prediction tasks.

Second, network training is maximum-likelihood estimation. Gaussian regression gives sum-of-squares error. Bernoulli and multinomial classification give cross-entropy errors. Because hidden-layer parameters make the objective non-convex, training requires iterative optimization.

Third, backpropagation provides an efficient way to compute gradients. It uses a forward pass for activations and a backward pass for error signals. The key formulas are $\delta_j=h'(a_j)\sum_k w_{kj}\delta_k$ and $\partial E/\partial w_{ji}=\delta_jz_i$.

Fourth, Hessian information describes curvature and is useful for second-order optimization, pruning, and Bayesian approximation, but it is expensive and often approximated.

Fifth, neural networks need regularization. Weight decay, early stopping, invariance-based methods, convolutional architectures, and soft weight sharing all control model complexity in different ways.

Sixth, mixture density networks extend neural networks from point prediction to conditional density modelling. This is essential for inverse problems and multimodal target distributions.

Finally, Bayesian neural networks place distributions over weights and average predictions over plausible parameter values. Exact inference is intractable, so the chapter uses Laplace approximation and evidence ideas.

## 8.2 Conceptual Links to the Rest of the Course

| Earlier / Later Topic | Connection to Chapter 5 |
|----------------------|--------------------------|
| Chapter 3 linear regression | Neural networks replace fixed basis functions with adaptive basis functions. |
| Chapter 4 logistic regression | A neural-network classifier often ends with sigmoid or softmax regression on learned hidden features. |
| Chapter 6 kernel methods | Kernels give another way to work with nonlinear features, often implicitly rather than through hidden units. |
| Chapter 7 SVM/RVM | SVMs use margin-based sparse solutions; RVMs use Bayesian sparsity. Neural networks use compact adaptive features. |
| Chapter 10 approximate inference | Bayesian neural networks motivate approximate inference for non-Gaussian, high-dimensional posteriors. |
| Chapter 14 mixtures of experts | Mixture density networks connect neural networks to conditional mixture models. |

## 8.3 Figure Checklist

All figures used in this lecture are screenshots/crops from the uploaded textbook PDF. Each filename records both the lecture figure number and the original textbook figure number.

| Lecture Figure | Textbook Figure | Topic | File |
|----------------|-----------------|-------|------|
| 5.1 | 5.1 | Two-layer network diagram | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_1__textbook_fig_5_1_p228_two_layer_network_diagram.png` |
| 5.2 | 5.2 | General feed-forward topology | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_2__textbook_fig_5_2_p230_general_feed_forward_topology.png` |
| 5.3 | 5.3 | Universal approximation examples | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_3__textbook_fig_5_3_p231_universal_approximation_examples.png` |
| 5.4 | 5.4 | Neural-network nonlinear decision boundary | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_4__textbook_fig_5_4_p232_two_class_neural_network_decision_boundary.png` |
| 5.5 | 5.5 | Error surface with local and global minima | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_5__textbook_fig_5_5_p236_error_surface_local_and_global_minima.png` |
| 5.6 | 5.6 | Quadratic approximation and Hessian eigenvectors | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_6__textbook_fig_5_6_p239_quadratic_approximation_hessian_eigenvectors.png` |
| 5.7 | 5.7 | Backpropagation hidden-unit delta | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_7__textbook_fig_5_7_p244_backpropagation_hidden_unit_delta.png` |
| 5.8 | 5.8 | Modular system and Jacobian backpropagation | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_8__textbook_fig_5_8_p247_modular_pattern_recognition_jacobian.png` |
| 5.9 | 5.9 | Network size and sinusoidal fits | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_9__textbook_fig_5_9_p257_network_size_sinusoidal_fits.png` |
| 5.10 | 5.10 | Hidden units, test error, and local minima | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_10__textbook_fig_5_10_p257_test_error_hidden_units_local_minima.png` |
| 5.11 | 5.11 | Hyperparameters for grouped Gaussian priors | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_11__textbook_fig_5_11_p260_hyperparameters_weight_priors.png` |
| 5.12 | 5.12 | Early stopping using validation error | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_12__textbook_fig_5_12_p261_early_stopping_training_validation_error.png` |
| 5.13 | 5.13 | Early stopping and weight decay geometry | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_13__textbook_fig_5_13_p262_early_stopping_weight_decay_geometry.png` |
| 5.14 | 5.14 | Synthetic warping of a handwritten digit | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_14__textbook_fig_5_14_p263_synthetic_warping_digit.png` |
| 5.15 | 5.15 | Tangent vector along transformation manifold | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_15__textbook_fig_5_15_p263_tangent_manifold_transformation.png` |
| 5.16 | 5.16 | Tangent vector for digit rotation | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_16__textbook_fig_5_16_p265_tangent_vector_digit_rotation.png` |
| 5.17 | 5.17 | Convolutional network receptive fields | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_17__textbook_fig_5_17_p268_convolutional_network_receptive_fields.png` |
| 5.18 | 5.18 | Two-link robot arm inverse problem | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_18__textbook_fig_5_18_p272_two_link_robot_arm_inverse_problem.png` |
| 5.19 | 5.19 | Forward vs. inverse problem data | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_19__textbook_fig_5_19_p273_forward_inverse_problem_multimodality.png` |
| 5.20 | 5.20 | Mixture density network architecture | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_20__textbook_fig_5_20_p274_mixture_density_network_architecture.png` |
| 5.21 | 5.21 | MDN mixing coefficients, means, contours, and mode | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_21__textbook_fig_5_21_p276_mixture_density_network_outputs.png` |
| 5.22 | 5.22 | Evidence framework for classification regularization | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_22__textbook_fig_5_22_p283_evidence_regularization_classification.png` |
| 5.23 | 5.23 | Laplace approximation for Bayesian neural-network classification | `./CoursePR2026/Fig/Chapter_5/lecture_fig_5_23__textbook_fig_5_23_p284_bayesian_neural_network_laplace_classification.png` |

## 8.4 Exercise Checklist

| Lecture Exercise | Textbook Exercise | Topic | File |
|------------------|-------------------|-------|------|
| 5.2 | 5.2 | Gaussian regression likelihood and sum-of-squares error | `./CoursePR2026/Fig/Chapter_5/lecture_ex_5_2__textbook_ex_5_2_p284.png` |
| 5.5 | 5.5 | Multiclass likelihood and cross-entropy | `./CoursePR2026/Fig/Chapter_5/lecture_ex_5_5__textbook_ex_5_5_p285.png` |
| 5.6 | 5.6 | Sigmoid cross-entropy output delta | `./CoursePR2026/Fig/Chapter_5/lecture_ex_5_6__textbook_ex_5_6_p285.png` |
| 5.8 | 5.8 | Derivative of the $\tanh$ activation | `./CoursePR2026/Fig/Chapter_5/lecture_ex_5_8__textbook_ex_5_8_p285.png` |

## 8.5 Suggested Teaching Flow

A practical lecture sequence is:

1. Start by reviewing fixed basis-function models from Chapters 3 and 4.
2. Explain the move from fixed basis functions to adaptive hidden units.
3. Use Figure 5.1 to walk through the two-layer network computation step by step.
4. Explain hidden activations, output activations, weights, and biases, then work through Exercise 5.8 for the $\tanh$ derivative.
5. Use Figure 5.2 to show that feed-forward networks can have more general acyclic topologies.
6. Use Figures 5.3-5.4 to explain universal approximation and nonlinear decision boundaries.
7. Discuss weight-space symmetries and why equivalent parameter settings exist.
8. Derive regression, binary classification, and multiclass classification error functions from maximum likelihood.
9. Work through Exercise 5.2 for Gaussian regression and Exercise 5.5 for multiclass cross-entropy.
10. Use Figure 5.5 to introduce non-convex error surfaces and local minima.
11. Use Figure 5.6 briefly to explain the local quadratic approximation and Hessian eigenvalues.
12. Derive the output delta formula, work through Exercise 5.6, and then derive the hidden delta backpropagation formula.
13. Use Figure 5.7 to illustrate backward error propagation.
14. Present the four-step backpropagation algorithm.
15. Explain computational efficiency and gradient checking.
16. Use Figure 5.8 to introduce Jacobians in modular systems if time allows.
17. Give a short overview of Hessian approximations, emphasizing that this section can be trimmed.
18. Use Figures 5.9-5.10 to introduce over-fitting, model size, and local minima.
19. Discuss weight decay and grouped Gaussian priors using Figure 5.11.
20. Explain early stopping with Figures 5.12-5.13.
21. Discuss invariance and data augmentation; tangent propagation can be optional.
22. Introduce convolutional networks and weight sharing using Figure 5.17.
23. Introduce inverse problems with Figures 5.18-5.19, then treat MDNs as optional advanced conditional-density models.
24. Finish with Bayesian neural networks as an uncertainty concept, using Figures 5.22-5.23 without deriving every Laplace/evidence equation.

## 8.6 Key Equations to Put on the Board

The following equations are the minimum board set for this chapter.

### Two-layer neural network

$$
y_k(\mathbf{x},\mathbf{w})
=f\left(
\sum_{j=1}^M w^{(2)}_{kj}
 h\left(\sum_{i=1}^D w^{(1)}_{ji}x_i+w^{(1)}_{j0}\right)
+w^{(2)}_{k0}
\right).
$$

### Hidden activation

$$
a_j=\sum_i w^{(1)}_{ji}x_i+w^{(1)}_{j0},
\qquad
z_j=h(a_j).
$$

For $h(a)=\tanh(a)$,

$$
h'(a)=1-\tanh^2(a)=1-z^2.
$$

### Output activations

$$
y=a \quad\text{(regression)},
\qquad
 y=\sigma(a) \quad\text{(binary classification)},
$$

$$
y_k=\frac{\exp(a_k)}{\sum_j\exp(a_j)}
\quad\text{(multiclass classification)}.
$$

### Sum-of-squares error

$$
E(\mathbf{w})=\frac{1}{2}\sum_n\{y(\mathbf{x}_n,\mathbf{w})-t_n\}^2.
$$

### Binary cross-entropy

$$
E(\mathbf{w})=-\sum_n
\{t_n\ln y_n+(1-t_n)\ln(1-y_n)\}.
$$

### Multiclass cross-entropy

$$
E(\mathbf{w})=-\sum_n\sum_k t_{nk}\ln y_{nk}.
$$

### Backpropagation local derivative

$$
\delta_j=\frac{\partial E}{\partial a_j}.
$$

For the standard output-layer pairings used here,

$$
\delta_k=y_k-t_k.
$$

### Hidden-unit backpropagation formula

$$
\delta_j=h'(a_j)\sum_k w_{kj}\delta_k.
$$

### Weight derivative

$$
\frac{\partial E}{\partial w_{ji}}=\delta_j z_i.
$$

### Gradient descent

$$
\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta\nabla E(\mathbf{w}^{(\tau)}).
$$

### Quadratic approximation

$$
E(\mathbf{w})\simeq E(\mathbf{w}^*)+
\frac{1}{2}(\mathbf{w}-\mathbf{w}^*)^T\mathbf{H}(\mathbf{w}-\mathbf{w}^*).
$$

### Weight decay

$$
\widetilde{E}(\mathbf{w})=E(\mathbf{w})+\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}.
$$

### Tangent-propagation regularizer

$$
\Omega=\frac{1}{2}\sum_n
\left\|\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\boldsymbol{\tau}_n\right\|^2.
$$

### Mixture density network

$$
p(t\mid\mathbf{x})=
\sum_{k=1}^K \pi_k(\mathbf{x})
\mathcal{N}\left(t\mid\mu_k(\mathbf{x}),\sigma_k^2(\mathbf{x})\right).
$$

### MDN responsibility

$$
\gamma_{nk}
=
\frac{
\pi_k(\mathbf{x}_n)\mathcal{N}\left(t_n\mid\mu_k(\mathbf{x}_n),\sigma_k^2(\mathbf{x}_n)\right)
}{
\sum_j \pi_j(\mathbf{x}_n)\mathcal{N}\left(t_n\mid\mu_j(\mathbf{x}_n),\sigma_j^2(\mathbf{x}_n)\right)
}.
$$

### Bayesian neural-network posterior approximation

$$
q(\mathbf{w})=
\mathcal{N}(\mathbf{w}\mid\mathbf{w}_{\mathrm{MAP}},\mathbf{A}^{-1}).
$$

### Bayesian classification predictive approximation

$$
p(t=1\mid\mathbf{x},\mathcal{D})
\simeq
\sigma\left(\kappa(\sigma_a^2)\mu_a\right),
\qquad
\kappa(\sigma^2)=\left(1+\frac{\pi\sigma^2}{8}\right)^{-1/2}.
$$
