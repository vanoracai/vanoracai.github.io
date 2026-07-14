# Pattern Recognition and Machine Learning
## Chapter 13: Sequential Data — Hidden Markov Models

> 📖 Textbook: Christopher M. Bishop — *Pattern Recognition and Machine Learning*, Springer, 2006  
> Chapter covered: selected material from Ch. 13, especially §13.1, §13.2, §13.2.2, and §13.2.5  
> **Teaching scope:** a compact half-week introduction to Markov models and Hidden Markov Models (HMMs)

---

## Table of Contents

1. [§0 Learning Goals and Lecture Roadmap](#0-learning-goals-and-lecture-roadmap)
2. [§1 Why Sequential Data Need a Different Model](#1-why-sequential-data-need-a-different-model)
3. [§2 The Markov Assumption](#2-the-markov-assumption)
4. [§3 From a Markov Chain to a Hidden Markov Model](#3-from-a-markov-chain-to-a-hidden-markov-model)
5. [§4 HMM Parameters and the Joint Distribution](#4-hmm-parameters-and-the-joint-distribution)
6. [§5 The Three Central Inference Questions](#5-the-three-central-inference-questions)
7. [§6 The Forward Algorithm](#6-the-forward-algorithm)
8. [§7 Most Likely State at Each Time Step](#7-most-likely-state-at-each-time-step)
9. [§8 The Viterbi Algorithm](#8-the-viterbi-algorithm)
10. [§9 Complete Worked Example: Weather and Umbrellas](#9-complete-worked-example-weather-and-umbrellas)
11. [§10 Guided Textbook Exercises](#10-guided-textbook-exercises)
12. [§11 From HMMs to RNNs, LSTMs, and Transformers](#11-from-hmms-to-rnns-lstms-and-transformers)
13. [§12 Chapter Summary and Concept Checklist](#12-chapter-summary-and-concept-checklist)

---

## Notation and Variable Definitions

This lecture uses only a small number of symbols, but it is important to keep their roles separate. In particular, **observations are known**, whereas **hidden states are unknown**.

### Sequence Variables

| Symbol | Definition |
|---|---|
| $N$ | Length of the sequence. |
| $n$ | Time index, where $n=1,2,\ldots,N$. |
| **$x_n$** | Observation at time $n$. It is visible in the data. |
| **$z_n$** | Hidden state at time $n$. It is not directly observed. |
| $\mathbf{X}=\{x_1,\ldots,x_N\}$ | Complete observation sequence. |
| $\mathbf{Z}=\{z_1,\ldots,z_N\}$ | Complete hidden-state sequence. |
| $K$ | Number of possible hidden states. |

### HMM Parameters

| Symbol | Definition |
|---|---|
| **$\pi_k$** | Initial-state probability $p(z_1=k)$. |
| **$A_{jk}$** | Transition probability $p(z_n=k\mid z_{n-1}=j)$. The row is the previous state; the column is the new state. |
| **$b_k(x_n)$** | Emission probability or density $p(x_n\mid z_n=k)$. |
| $\boldsymbol{\theta}$ | Collection of HMM parameters, usually $\boldsymbol{\theta}=\{\boldsymbol{\pi},\mathbf{A},\text{emission parameters}\}$. |

### Dynamic-Programming Quantities

| Symbol | Definition |
|---|---|
| **$\alpha_n(k)$** | Forward quantity $p(x_1,\ldots,x_n,z_n=k)$. It sums probabilities of all partial paths ending in state $k$. |
| **$\delta_n(k)$** | Viterbi quantity: probability of the single best partial path ending in state $k$. |
| **$\psi_n(k)$** | Backpointer recording which previous state produced the maximum for $\delta_n(k)$. |
| $\widehat z_n$ | An estimated hidden state. The meaning depends on whether we estimate each time independently or estimate a complete path. |

> **Teaching focus.** The two most important dynamic-programming ideas are:
>
> $$
> \text{Forward algorithm: sum over previous paths}
> $$
>
> and
>
> $$
> \text{Viterbi algorithm: maximize over previous paths}.
> $$

---

# §0 Learning Goals and Lecture Roadmap

> 📖 Textbook Ch.13 opening; §13.1; §13.2; §13.2.2; §13.2.5

## 0.1 What Students Should Be Able to Do

After this lecture, students should be able to:

1. Explain why the i.i.d. assumption is inappropriate for many sequences.
2. State the first-order Markov assumption in words and in equations.
3. Distinguish a hidden state from an observation.
4. Interpret initial, transition, and emission probabilities.
5. Write the joint probability of one HMM state-observation path.
6. Use the forward algorithm to calculate the probability of an observation sequence.
7. Normalize forward quantities to estimate the most likely current state.
8. Use the Viterbi algorithm to find the most likely complete hidden-state sequence.
9. Explain why “the most likely state at every time” is not generally the same as “the most likely complete path.”
10. Describe the conceptual progression

$$
\text{HMM} \longrightarrow \text{RNN/LSTM} \longrightarrow \text{Transformer}.
$$

## 0.2 What We Will Not Cover

This is intentionally a focused introduction. We will not derive:

- the complete Baum-Welch or EM training algorithm;
- the full backward recursion;
- detailed scaling-factor equations;
- linear dynamical systems;
- Kalman filtering or smoothing;
- particle filters.

We will briefly mention numerical underflow and full-sequence state smoothing, but these are not the mathematical focus of the lecture.

## 0.3 Suggested Teaching Schedule

| Time | Topic |
|---|---|
| 10 minutes | Sequential data and the Markov assumption |
| 15 minutes | Hidden states, observations, transition and emission probabilities |
| 20 minutes | Forward algorithm |
| 20 minutes | Complete numerical example |
| 15 minutes | Viterbi algorithm |
| 10 minutes | HMM to RNN/LSTM/Transformer connection and summary |

The main goal is not memorizing formulas. The goal is to understand **why dynamic programming makes an apparently exponential sequence problem manageable**.

---

# §1 Why Sequential Data Need a Different Model

> 📖 Textbook Ch.13 opening, pp. 605-607

## 1.1 The i.i.d. Assumption

In many earlier models, a data set was treated as independent and identically distributed:

$$
\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_N \overset{\mathrm{i.i.d.}}{\sim} p(\mathbf{x}).
$$

Under this assumption, the probability of the complete data set factorizes as

$$
p(\mathbf{x}_1,\ldots,\mathbf{x}_N)
=
\prod_{n=1}^{N}p(\mathbf{x}_n).
$$

This is useful because every example can be processed independently.

However, it discards ordering information. If we randomly permute an i.i.d. data set, its probability does not change. For sequential data, the order is often essential.

## 1.2 Examples of Sequential Dependence

| Sequence | Why Nearby Elements Are Related |
|---|---|
| Daily weather | Rain today changes the probability of rain tomorrow. |
| Speech | Adjacent acoustic frames usually belong to the same or neighboring phonemes. |
| Handwriting | The current pen position depends strongly on the previous pen position. |
| DNA | Neighboring bases can form meaningful biological motifs. |
| Machine sensor signal | A physical system usually evolves continuously rather than jumping independently. |
| Text | The probability of the next word depends on previous words. |

A sequence model should therefore use previous information when reasoning about the current or next time step.

## 1.3 The General Chain Rule Is Too General

Probability theory always allows us to write

$$
p(x_1,x_2,\ldots,x_N)
=
p(x_1)\prod_{n=2}^{N}p(x_n\mid x_1,\ldots,x_{n-1}).
$$

This factorization is exact, but it is not yet a practical model. The conditional distribution at time $n$ can depend on an increasingly long history:

$$
x_1,\ldots,x_{n-1}.
$$

As $n$ grows, the number of possible histories grows rapidly. We therefore need a simplifying assumption.

---

# §2 The Markov Assumption

> 📖 Textbook §13.1, pp. 607-610

## 2.1 First-Order Markov Assumption

A first-order Markov model assumes that the next variable depends on the immediate previous variable, but not directly on the more distant past:

$$
p(x_n\mid x_1,\ldots,x_{n-1})
=
p(x_n\mid x_{n-1}).
$$

In words:

> **Once we know the present, the more distant past provides no additional information about the immediate future.**

This does not mean that the distant past has no effect at all. Its effect can be carried through the current state $x_{n-1}$.

> ![Figure 13.1](./CoursePR2026/Fig/Chapter_13/lecture_fig_13_1__textbook_fig_13_3__p608.png)
>
> *Figure 13.1 (Textbook Fig. 13.3, p. 608): A first-order Markov chain. Each variable depends directly only on the previous variable.*

The joint distribution becomes

$$
p(x_1,\ldots,x_N)
=
p(x_1)\prod_{n=2}^{N}p(x_n\mid x_{n-1}).
$$

This is much simpler than conditioning on the entire history.

## 2.2 A Simple Weather Interpretation

Let

$$
x_n\in\{\text{sunny},\text{rainy}\}.
$$

A first-order Markov assumption says

$$
p(x_{n+1}\mid x_1,\ldots,x_n)
=
p(x_{n+1}\mid x_n).
$$

For example,

$$
p(x_{n+1}=\text{rainy}\mid x_n=\text{rainy})=0.7.
$$

The model does not separately ask whether it rained two days ago or three days ago. That earlier history is assumed to be summarized sufficiently by today’s weather.

## 2.3 Why Not Use a Higher-Order Markov Model?

A second-order model uses

$$
p(x_n\mid x_{n-1},x_{n-2}),
$$

and an $M$th-order model uses the previous $M$ observations.

This can capture longer patterns, but the number of parameters can grow very quickly. Suppose each variable has $K$ possible values. A general $M$th-order conditional probability table must consider approximately $K^M$ possible histories.

For example, with $K=10$ states:

| Markov Order | Number of Possible Histories |
|---|---:|
| $M=1$ | $10$ |
| $M=2$ | $100$ |
| $M=3$ | $1{,}000$ |
| $M=5$ | $100{,}000$ |

This motivates a different idea:

> Instead of storing a long window of observations explicitly, introduce a compact hidden state that summarizes the relevant history.

## 2.4 Guided Derivation of the Markov Property

Starting from

$$
p(x_1,\ldots,x_N)
=
p(x_1)\prod_{m=2}^{N}p(x_m\mid x_{m-1}),
$$

consider the conditional probability

$$
p(x_n\mid x_1,\ldots,x_{n-1}).
$$

By definition,

$$
p(x_n\mid x_1,\ldots,x_{n-1})
=
\frac{p(x_1,\ldots,x_n)}{p(x_1,\ldots,x_{n-1})}.
$$

Substitute the Markov factorization into the numerator:

$$
p(x_1,\ldots,x_n)
=
p(x_1)
\left[\prod_{m=2}^{n-1}p(x_m\mid x_{m-1})\right]
p(x_n\mid x_{n-1}).
$$

The denominator is

$$
p(x_1,\ldots,x_{n-1})
=
p(x_1)
\prod_{m=2}^{n-1}p(x_m\mid x_{m-1}).
$$

The common factors cancel, leaving

$$
\boxed{
p(x_n\mid x_1,\ldots,x_{n-1})
=
p(x_n\mid x_{n-1})
}.
$$

This is the main idea behind Textbook Exercise 13.1/13.2, written without requiring graphical-model terminology.

---

# §3 From a Markov Chain to a Hidden Markov Model

> 📖 Textbook §13.1-§13.2, pp. 609-615

## 3.1 Why Introduce a Hidden State?

In many real problems, the important state of the system is not directly measurable.

Examples:

- In speech recognition, the acoustic signal is observed, but the intended phoneme is hidden.
- In activity recognition, sensor readings are observed, but the person’s activity is hidden.
- In fault diagnosis, voltage and vibration are observed, but the machine condition is hidden.
- In weather reasoning, an umbrella may be observed, but the true weather may not be directly observed.

We therefore introduce two variables at each time step:

- $z_n$: the **hidden state**;
- $x_n$: the **observation** generated from that state.

> ![Figure 13.2](./CoursePR2026/Fig/Chapter_13/lecture_fig_13_2__textbook_fig_13_5__p609.png)
>
> *Figure 13.2 (Textbook Fig. 13.5, p. 609): A state-space model. Hidden variables $z_n$ form a Markov chain, while each observation $x_n$ depends on its corresponding hidden state.*

## 3.2 The Two Conditional-Independence Assumptions

An HMM makes two central assumptions.

### Assumption 1: Hidden-State Markov Assumption

$$
p(z_n\mid z_1,\ldots,z_{n-1})
=
p(z_n\mid z_{n-1}).
$$

The next hidden state depends only on the previous hidden state.

### Assumption 2: Observation or Emission Assumption

$$
p(x_n\mid z_1,\ldots,z_N,x_1,\ldots,x_{n-1},x_{n+1},\ldots,x_N)
=
p(x_n\mid z_n).
$$

Once the current hidden state $z_n$ is known, the current observation $x_n$ does not need the other states or observations.

A simpler way to remember the model is:

$$
\boxed{z_{n-1}\rightarrow z_n\rightarrow z_{n+1}}
$$

and

$$
\boxed{z_n\rightarrow x_n}.
$$

## 3.3 Hidden Does Not Mean Unimportant

The hidden state is the model’s internal explanation of the sequence.

For example:

| Hidden State $z_n$ | Observation $x_n$ |
|---|---|
| Phoneme | Acoustic spectrum |
| Weather | Umbrella/no umbrella |
| Machine health | Sensor reading |
| Human activity | Accelerometer measurement |
| Part of a handwritten stroke | Pen coordinates |

The observation can be noisy or ambiguous. The hidden-state dynamics provide context that helps disambiguate it.

## 3.4 Why HMM Observations Can Have Long-Range Dependence

The hidden chain is first-order Markov, but the observations do not need to be first-order Markov.

The past observations influence our belief about $z_n$. The hidden state then influences $z_{n+1}$ and the future observations. Thus information can travel through the latent chain:

$$
x_1\rightarrow z_1\rightarrow z_2\rightarrow\cdots\rightarrow z_n\rightarrow x_n.
$$

The hidden state acts as a compact memory.

This is an important conceptual step toward recurrent neural networks, where a learned hidden vector also carries information forward through a sequence.

## 3.5 An HMM as a Sequential Mixture Model

A standard mixture model chooses a component independently for every data point.

An HMM also chooses a component or state at every time step, but the choice at time $n$ depends on the choice at time $n-1$.

Therefore:

- mixture model: state choices are independent across data points;
- HMM: state choices are connected by transition probabilities.

## 3.6 Textbook Application: Online Handwriting

> ![Figure 13.5](./CoursePR2026/Fig/Chapter_13/lecture_fig_13_5__textbook_fig_13_11__p615.png)
>
> *Figure 13.5 (Textbook Fig. 13.11, p. 615): Top row: real online handwritten digits. Bottom row: samples generated by an HMM. The example illustrates how a sequence model can represent different stages of a handwritten stroke.*

In an online handwriting problem, the observation at each step could contain pen coordinates or local motion. Hidden states can represent different parts of the digit, such as the upper curve, turning region, or lower stroke.

The HMM can allow the writer to spend different numbers of time steps in the same state. This gives some tolerance to local stretching or compression of the sequence.

---

# §4 HMM Parameters and the Joint Distribution

> 📖 Textbook §13.2, pp. 610-615

## 4.1 Initial-State Probabilities

The first hidden state has no previous state. We therefore define

$$
\pi_k=p(z_1=k),
$$

with

$$
\pi_k\geq 0,
\qquad
\sum_{k=1}^{K}\pi_k=1.
$$

For two weather states, an example is

$$
\boldsymbol{\pi}
=
\begin{bmatrix}
0.6 & 0.4
\end{bmatrix},
$$

meaning that the sequence begins in the sunny state with probability $0.6$ and in the rainy state with probability $0.4$.

## 4.2 Transition Probabilities

The transition probability from state $j$ to state $k$ is

$$
A_{jk}=p(z_n=k\mid z_{n-1}=j).
$$

For each previous state $j$, the probabilities of moving to all possible new states must sum to one:

$$
\sum_{k=1}^{K}A_{jk}=1.
$$

> ![Figure 13.3](./CoursePR2026/Fig/Chapter_13/lecture_fig_13_3__textbook_fig_13_6__p611.png)
>
> *Figure 13.3 (Textbook Fig. 13.6, p. 611): A transition diagram for three hidden states. An arrow from state $j$ to state $k$ represents $A_{jk}$.*

For a two-state example,

$$
\mathbf{A}
=
\begin{bmatrix}
0.7 & 0.3\\
0.4 & 0.6
\end{bmatrix}.
$$

Interpretation:

- if the current state is sunny, tomorrow is sunny with probability $0.7$ and rainy with probability $0.3$;
- if the current state is rainy, tomorrow is sunny with probability $0.4$ and rainy with probability $0.6$.

The diagonal elements $A_{11}$ and $A_{22}$ describe remaining in the same state.

## 4.3 Emission Probabilities

The emission model says how likely an observation is under each hidden state:

$$
b_k(x_n)=p(x_n\mid z_n=k).
$$

For a discrete observation, this is a probability table. For a continuous observation, it can be a probability density such as a Gaussian.

Example:

| Hidden Weather State | $p(\text{umbrella}\mid z_n)$ | $p(\text{no umbrella}\mid z_n)$ |
|---|---:|---:|
| Sunny | $0.2$ | $0.8$ |
| Rainy | $0.9$ | $0.1$ |

An umbrella strongly supports the rainy state, but it does not prove rain with certainty. Someone can carry an umbrella on a sunny day.

## 4.4 Trellis or Lattice Representation

A transition graph shows possible states and transitions. To perform sequence inference, it is useful to unfold this graph across time.

> ![Figure 13.4](./CoursePR2026/Fig/Chapter_13/lecture_fig_13_4__textbook_fig_13_7__p612.png)
>
> *Figure 13.4 (Textbook Fig. 13.7, p. 612): An HMM transition graph unfolded into a trellis. Each column is a time step, and each row is a possible hidden state.*

A path through the trellis specifies one complete hidden-state sequence:

$$
(z_1,z_2,\ldots,z_N).
$$

If there are $K$ states and $N$ time steps, then there are

$$
K^N
$$

possible hidden-state sequences before considering forbidden transitions.

For example, $K=5$ and $N=100$ gives

$$
5^{100}\approx 7.9\times 10^{69}
$$

possible paths. Direct enumeration is impossible.

## 4.5 Generative Story of an HMM

An HMM can be understood as a data-generation process.

1. Sample the first hidden state:

$$
z_1\sim p(z_1)=\boldsymbol{\pi}.
$$

2. Generate the first observation:

$$
x_1\sim p(x_1\mid z_1).
$$

3. For $n=2,\ldots,N$:
   - sample a new hidden state

$$
z_n\sim p(z_n\mid z_{n-1});
$$

   - generate the observation

$$
x_n\sim p(x_n\mid z_n).
$$

The process alternates between **state transition** and **observation emission**.

## 4.6 Joint Probability of One Complete Path

For one chosen hidden-state sequence $\mathbf{Z}$ and observation sequence $\mathbf{X}$,

$$
\boxed{
p(\mathbf{X},\mathbf{Z})
=
p(z_1)
\left[\prod_{n=2}^{N}p(z_n\mid z_{n-1})\right]
\left[\prod_{n=1}^{N}p(x_n\mid z_n)\right]
}.
$$

Using HMM notation,

$$
p(\mathbf{X},\mathbf{Z})
=
\pi_{z_1}b_{z_1}(x_1)
\prod_{n=2}^{N}
A_{z_{n-1},z_n}b_{z_n}(x_n).
$$

For a short path $z_1=a,z_2=b,z_3=c$, the probability is

$$
p(z_1=a)
\,p(x_1\mid a)
\,p(z_2=b\mid a)
\,p(x_2\mid b)
\,p(z_3=c\mid b)
\,p(x_3\mid c).
$$

> **Common mistake.** Do not multiply transition probabilities without emission probabilities. A path is evaluated using both its transitions and how well its states explain the observations.

---

# §5 The Three Central Inference Questions

> 📖 Textbook §13.2.2 and §13.2.5

An HMM is useful because it answers several different questions. These questions look similar, but they require different operations.

## 5.1 Question 1: How Probable Is the Observation Sequence?

We observe

$$
\mathbf{X}=(x_1,\ldots,x_N)
$$

but do not know the hidden states. Therefore we sum over all possible state sequences:

$$
\boxed{
p(\mathbf{X})
=
\sum_{\mathbf{Z}}p(\mathbf{X},\mathbf{Z})
}.
$$

This is called the sequence likelihood or evidence.

A direct calculation requires summing $K^N$ path probabilities. The forward algorithm performs the same sum efficiently.

## 5.2 Question 2: What Is the Most Likely State at a Particular Time?

At time $n$, using observations up to that time, we can calculate

$$
p(z_n=k\mid x_1,\ldots,x_n).
$$

The most likely current hidden state is

$$
\boxed{
\widehat z_n^{\mathrm{filter}}
=
\arg\max_k p(z_n=k\mid x_1,\ldots,x_n)
}.
$$

This is a **filtering** question. It can be answered by normalizing the forward values.

If we instead want to use the complete sequence, including future observations $x_{n+1},\ldots,x_N$, then the desired quantity is

$$
p(z_n=k\mid x_1,\ldots,x_N).
$$

That is a **smoothing** question and normally requires a backward pass. We mention this distinction but do not derive the backward algorithm in this lecture.

## 5.3 Question 3: What Is the Most Likely Complete State Sequence?

We seek one complete path:

$$
\boxed{
\widehat{\mathbf{Z}}^{\mathrm{MAP}}
=
\arg\max_{\mathbf{Z}}p(\mathbf{Z}\mid\mathbf{X})
}.
$$

Because $p(\mathbf{X})$ is constant with respect to $\mathbf{Z}$,

$$
\widehat{\mathbf{Z}}^{\mathrm{MAP}}
=
\arg\max_{\mathbf{Z}}p(\mathbf{X},\mathbf{Z}).
$$

This is the Viterbi decoding problem.

## 5.4 Sum versus Max

| Question | Operation over Hidden Paths | Algorithm |
|---|---|---|
| Probability of observations $p(\mathbf{X})$ | Sum over all paths | Forward algorithm |
| Most likely current state | Sum over paths ending in each state, then normalize | Forward filtering |
| Most likely complete path | Maximize over paths | Viterbi algorithm |

The forward and Viterbi algorithms use nearly the same trellis, but replace one operation:

$$
\sum_j
\quad\longleftrightarrow\quad
\max_j.
$$

## 5.5 Why Individual State Decisions Can Differ from the Best Path

Suppose the posterior probabilities of complete two-step paths are:

| Path | Posterior Probability |
|---|---:|
| $(A,A)$ | $0.40$ |
| $(B,C)$ | $0.35$ |
| $(B,D)$ | $0.25$ |

At time 1:

$$
p(z_1=B)=0.35+0.25=0.60,
$$

so the individually most likely state is $B$.

At time 2:

$$
p(z_2=A)=0.40,
$$

which is larger than $p(z_2=C)=0.35$ and $p(z_2=D)=0.25$.

Thus the two individual choices produce $(B,A)$, even though this path has probability zero in the table.

The most probable complete path is $(A,A)$ with probability $0.40$.

> **Key lesson.** Local decisions do not necessarily form a globally valid or globally optimal sequence.

---

# §6 The Forward Algorithm

> 📖 Textbook §13.2.2, pp. 618-623

## 6.1 Why Naive Summation Is Exponential

The sequence likelihood is

$$
p(\mathbf{X})
=
\sum_{z_1}\sum_{z_2}\cdots\sum_{z_N}
p(\mathbf{X},\mathbf{Z}).
$$

There are $K$ choices at each of $N$ time steps, producing $K^N$ complete paths.

However, many paths share the same ending state. Once different partial paths reach the same state at time $n$, their future evolution is governed by the same transition and emission probabilities. We can therefore combine them.

This is the dynamic-programming principle behind the forward algorithm.

## 6.2 Definition of the Forward Quantity

Define

$$
\boxed{
\alpha_n(k)
=
p(x_1,\ldots,x_n,z_n=k)
}.
$$

This quantity has two parts:

1. the probability of observing $x_1,\ldots,x_n$;
2. the probability that the hidden state at time $n$ is $k$.

Equivalently, $\alpha_n(k)$ is the sum of probabilities of **all partial paths that end in state $k$ at time $n$**.

## 6.3 Initialization

At time $n=1$, there is no previous transition. Therefore

$$
\boxed{
\alpha_1(k)=\pi_k b_k(x_1)
}.
$$

This has a simple interpretation:

$$
\text{probability of starting in state }k
\times
\text{probability that state }k\text{ emits }x_1.
$$

## 6.4 Recursion: Prediction Followed by Observation Update

Assume that $\alpha_{n-1}(j)$ is known for every previous state $j$.

To calculate $\alpha_n(k)$, first consider all ways of arriving at state $k$:

$$
\sum_{j=1}^{K}\alpha_{n-1}(j)A_{jk}.
$$

This is the predicted probability mass for state $k$ before using observation $x_n$.

Next multiply by the probability that state $k$ emits $x_n$:

$$
\boxed{
\alpha_n(k)
=
b_k(x_n)
\sum_{j=1}^{K}\alpha_{n-1}(j)A_{jk}
}.
$$

The two stages are:

### Stage A: Transition or Prediction

$$
\widetilde\alpha_n(k)
=
\sum_{j=1}^{K}\alpha_{n-1}(j)A_{jk}.
$$

### Stage B: Emission or Evidence Update

$$
\alpha_n(k)
=
b_k(x_n)\widetilde\alpha_n(k).
$$

> ![Figure 13.6](./CoursePR2026/Fig/Chapter_13/lecture_fig_13_6__textbook_fig_13_12__p621.png)
>
> *Figure 13.6 (Textbook Fig. 13.12, p. 621): To compute the forward value for one state at time $n$, sum contributions from every state at time $n-1$, then multiply by the emission probability of the current observation.*

## 6.5 Why the Recursion Is Correct

Start from the definition

$$
\alpha_n(k)=p(x_1,\ldots,x_n,z_n=k).
$$

Insert a sum over the previous state:

$$
\alpha_n(k)
=
\sum_{j=1}^{K}
p(x_1,\ldots,x_n,z_{n-1}=j,z_n=k).
$$

Use the HMM factorization:

$$
p(x_1,\ldots,x_n,z_{n-1}=j,z_n=k)
$$

$$
=
p(x_1,\ldots,x_{n-1},z_{n-1}=j)
\,p(z_n=k\mid z_{n-1}=j)
\,p(x_n\mid z_n=k).
$$

Recognize each term:

$$
p(x_1,\ldots,x_{n-1},z_{n-1}=j)=\alpha_{n-1}(j),
$$

$$
p(z_n=k\mid z_{n-1}=j)=A_{jk},
$$

and

$$
p(x_n\mid z_n=k)=b_k(x_n).
$$

Therefore

$$
\alpha_n(k)
=
\sum_{j=1}^{K}\alpha_{n-1}(j)A_{jk}b_k(x_n).
$$

Because $b_k(x_n)$ does not depend on $j$, move it outside the sum:

$$
\boxed{
\alpha_n(k)
=
b_k(x_n)\sum_{j=1}^{K}\alpha_{n-1}(j)A_{jk}
}.
$$

## 6.6 Termination: Observation-Sequence Probability

At the final time step, every complete hidden path ends in one of the $K$ states. Therefore

$$
\boxed{
p(\mathbf{X})
=
\sum_{k=1}^{K}\alpha_N(k)
}.
$$

This sums the probabilities of all paths grouped by their final state.

## 6.7 Forward Algorithm Summary

### Input

- initial probabilities $\boldsymbol{\pi}$;
- transition matrix $\mathbf{A}$;
- emission model $b_k(x)$;
- observations $x_1,\ldots,x_N$.

### Algorithm

1. Initialize:

$$
\alpha_1(k)=\pi_kb_k(x_1).
$$

2. For $n=2,\ldots,N$ and each state $k$:

$$
\alpha_n(k)=b_k(x_n)\sum_j\alpha_{n-1}(j)A_{jk}.
$$

3. Terminate:

$$
p(\mathbf{X})=\sum_k\alpha_N(k).
$$

## 6.8 Computational Complexity

At each time step:

- there are $K$ destination states;
- for each destination state, sum over $K$ previous states.

Thus the cost per time step is

$$
O(K^2),
$$

and the total cost is

$$
\boxed{O(NK^2)}.
$$

This is linear in sequence length $N$, rather than exponential.

If we only need the likelihood, we need to keep only the previous and current forward vectors, so memory can be reduced to

$$
O(K).
$$

## 6.9 Numerical Underflow: A Brief Practical Note

Forward values multiply many probabilities smaller than one. For long sequences, the values can become extremely small.

In practical implementations, one usually:

- normalizes at each time step and stores the normalization constants; or
- performs related calculations in a numerically stable log-space form.

The complete scaling-factor derivation is outside this lecture. The conceptual algorithm remains unchanged.

---

# §7 Most Likely State at Each Time Step

> 📖 Related to Textbook §13.2.2; filtering interpretation

## 7.1 Normalize the Forward Quantity

Recall

$$
\alpha_n(k)=p(x_1,\ldots,x_n,z_n=k).
$$

Sum over $k$:

$$
\sum_{k=1}^{K}\alpha_n(k)
=
p(x_1,\ldots,x_n).
$$

Therefore

$$
\frac{\alpha_n(k)}{\sum_{r=1}^{K}\alpha_n(r)}
=
\frac{p(x_1,\ldots,x_n,z_n=k)}{p(x_1,\ldots,x_n)}.
$$

By the definition of conditional probability,

$$
\boxed{
p(z_n=k\mid x_1,\ldots,x_n)
=
\frac{\alpha_n(k)}{\sum_{r=1}^{K}\alpha_n(r)}
}.
$$

## 7.2 Current Most Likely State

The most likely hidden state using observations available so far is

$$
\boxed{
\widehat z_n^{\mathrm{filter}}
=
\arg\max_k\alpha_n(k)
}.
$$

Normalization is not required for the argmax because every $\alpha_n(k)$ is divided by the same positive constant. However, normalization is useful when actual posterior probabilities are required.

## 7.3 Filtering versus Smoothing

| Task | Probability | Information Used |
|---|---|---|
| Filtering | $p(z_n\mid x_1,\ldots,x_n)$ | Past and current observations |
| Smoothing | $p(z_n\mid x_1,\ldots,x_N)$ | Entire observation sequence |

Forward normalization gives filtering. Smoothing can change the estimate of an earlier state after later observations are seen.

Example: an ambiguous sound at time $n$ might be interpreted differently after hearing the next word.

The textbook obtains smoothed state probabilities using the forward-backward algorithm. We do not derive the backward recursion here.

---

# §8 The Viterbi Algorithm

> 📖 Textbook §13.2.5, pp. 629-631

## 8.1 Objective

The Viterbi algorithm finds the most probable complete hidden-state sequence:

$$
\widehat{\mathbf{Z}}
=
\arg\max_{\mathbf{Z}}p(\mathbf{X},\mathbf{Z}).
$$

A naive algorithm evaluates every one of the $K^N$ paths. Viterbi uses dynamic programming to keep only the best partial path ending in each state.

> ![Figure 13.7](./CoursePR2026/Fig/Chapter_13/lecture_fig_13_7__textbook_fig_13_16__p630.png)
>
> *Figure 13.7 (Textbook Fig. 13.16, p. 630): Different paths pass through the HMM trellis. Viterbi keeps the best partial path ending at each state and discards inferior alternatives.*

## 8.2 Viterbi Quantity

Define

$$
\boxed{
\delta_n(k)
=
\max_{z_1,\ldots,z_{n-1}}
p(x_1,\ldots,x_n,z_1,\ldots,z_{n-1},z_n=k)
}.
$$

Interpretation:

> $\delta_n(k)$ is the probability of the single best partial path that explains observations up to time $n$ and ends in state $k$.

Compare this with the forward quantity:

$$
\alpha_n(k)
=
\sum_{z_1,\ldots,z_{n-1}}
p(x_1,\ldots,x_n,z_1,\ldots,z_n=k).
$$

Forward sums all partial paths. Viterbi keeps only the largest one.

## 8.3 Initialization

At time 1,

$$
\boxed{
\delta_1(k)=\pi_kb_k(x_1)
}.
$$

There is no backpointer at the first time step.

## 8.4 Recursion

To find the best path ending in state $k$ at time $n$, consider every possible previous state $j$:

$$
\delta_{n-1}(j)A_{jk}.
$$

Choose the largest predecessor contribution, then multiply by the current emission probability:

$$
\boxed{
\delta_n(k)
=
b_k(x_n)
\max_j\left[\delta_{n-1}(j)A_{jk}\right]
}.
$$

We must also remember which predecessor produced the maximum:

$$
\boxed{
\psi_n(k)
=
\arg\max_j\left[\delta_{n-1}(j)A_{jk}\right]
}.
$$

The value $\delta_n(k)$ stores the best score. The value $\psi_n(k)$ stores how to reconstruct the path.

## 8.5 Termination

At the last time step, choose the best final state:

$$
\boxed{
\widehat z_N
=
\arg\max_k\delta_N(k)
}.
$$

The probability of the best complete path is

$$
\max_k\delta_N(k).
$$

## 8.6 Backtracking

Starting from $\widehat z_N$, recover earlier states using the stored backpointers:

$$
\boxed{
\widehat z_{n-1}
=
\psi_n(\widehat z_n),
\qquad n=N,N-1,\ldots,2.
}.
$$

The forward computation finds the best score for every ending state. Backtracking converts these local records into the complete globally best path.

## 8.7 Viterbi in Log Space

Products of small probabilities can underflow. Viterbi is commonly implemented using log probabilities.

Define

$$
D_n(k)=\log\delta_n(k).
$$

Then products become sums:

$$
\boxed{
D_n(k)
=
\log b_k(x_n)
+
\max_j\left[D_{n-1}(j)+\log A_{jk}\right]
}.
$$

The backpointer is unchanged:

$$
\psi_n(k)
=
\arg\max_j\left[D_{n-1}(j)+\log A_{jk}\right].
$$

Unlike the forward algorithm, Viterbi has only maxima and products. The logarithm handles them naturally.

## 8.8 Complexity

The Viterbi recursion also examines $K$ previous states for each of $K$ current states at every time step:

$$
\boxed{O(NK^2)}.
$$

To reconstruct the path, we store $K$ backpointers at each of $N$ time steps:

$$
O(NK)
$$

memory.

## 8.9 Forward versus Viterbi

| Feature | Forward Algorithm | Viterbi Algorithm |
|---|---|---|
| Combines paths using | Sum | Maximum |
| Quantity at state $k$ | Probability of all paths ending at $k$ | Probability of best path ending at $k$ |
| Main output | $p(\mathbf{X})$ and filtering probabilities | Most likely complete state path |
| Backtracking required | No | Yes |
| Complexity | $O(NK^2)$ | $O(NK^2)$ |

A useful memory aid is:

$$
\boxed{\text{Forward} = \text{sum-product}}
$$

and

$$
\boxed{\text{Viterbi} = \text{max-product}}.
$$

---

# §9 Complete Worked Example: Weather and Umbrellas

This example answers all three central questions with a two-state HMM.

## 9.1 Model Definition

Hidden states:

$$
S=\text{sunny},
\qquad
R=\text{rainy}.
$$

Observations:

$$
U=\text{umbrella},
\qquad
N=\text{no umbrella}.
$$

Initial probabilities:

$$
\boldsymbol{\pi}
=
\begin{bmatrix}
0.6 & 0.4
\end{bmatrix}.
$$

Transition matrix:

$$
\mathbf{A}
=
\begin{bmatrix}
0.7 & 0.3\\
0.4 & 0.6
\end{bmatrix},
$$

where rows correspond to the previous state $(S,R)$ and columns correspond to the new state $(S,R)$.

Emission probabilities:

| State | $p(U\mid z)$ | $p(N\mid z)$ |
|---|---:|---:|
| $S$ | $0.2$ | $0.8$ |
| $R$ | $0.9$ | $0.1$ |

Observed sequence:

$$
\mathbf{X}=(U,U,N).
$$

We will calculate:

1. $p(U,U,N)$;
2. the most likely state after each observation;
3. the most likely complete state sequence.

## 9.2 Forward Step 1: First Umbrella Observation

For sunny:

$$
\alpha_1(S)
=
p(z_1=S)p(U\mid S)
=0.6\times 0.2
=0.12.
$$

For rainy:

$$
\alpha_1(R)
=
p(z_1=R)p(U\mid R)
=0.4\times 0.9
=0.36.
$$

The probability of observing an umbrella on day 1 is

$$
p(U)=0.12+0.36=0.48.
$$

The filtered state probabilities are

$$
p(S\mid U)=\frac{0.12}{0.48}=0.25,
$$

$$
p(R\mid U)=\frac{0.36}{0.48}=0.75.
$$

After seeing an umbrella, rain is more likely.

## 9.3 Forward Step 2: Second Umbrella Observation

### Destination State $S$

First sum the two ways of arriving at $S$:

$$
\alpha_1(S)A_{SS}+\alpha_1(R)A_{RS}
$$

$$
=0.12\times 0.7+0.36\times 0.4
$$

$$
=0.084+0.144
=0.228.
$$

Then multiply by the sunny-state umbrella emission probability:

$$
\alpha_2(S)
=0.2\times 0.228
=0.0456.
$$

### Destination State $R$

Sum the ways of arriving at $R$:

$$
\alpha_1(S)A_{SR}+\alpha_1(R)A_{RR}
$$

$$
=0.12\times 0.3+0.36\times 0.6
$$

$$
=0.036+0.216
=0.252.
$$

Multiply by the rainy-state umbrella emission probability:

$$
\alpha_2(R)
=0.9\times 0.252
=0.2268.
$$

Thus

$$
p(U,U)=0.0456+0.2268=0.2724.
$$

Filtered posterior:

$$
p(S\mid U,U)
=
\frac{0.0456}{0.2724}
\approx 0.1674,
$$

$$
p(R\mid U,U)
=
\frac{0.2268}{0.2724}
\approx 0.8326.
$$

After two umbrellas, rain is even more likely.

## 9.4 Forward Step 3: No Umbrella Observation

### Destination State $S$

Predict sunny:

$$
0.0456\times 0.7+0.2268\times 0.4
$$

$$
=0.03192+0.09072
=0.12264.
$$

Multiply by $p(N\mid S)=0.8$:

$$
\alpha_3(S)
=0.12264\times 0.8
=0.098112.
$$

### Destination State $R$

Predict rainy:

$$
0.0456\times 0.3+0.2268\times 0.6
$$

$$
=0.01368+0.13608
=0.14976.
$$

Multiply by $p(N\mid R)=0.1$:

$$
\alpha_3(R)
=0.14976\times 0.1
=0.014976.
$$

## 9.5 Answer to Question 1: Observation-Sequence Probability

Sum the final forward values:

$$
\boxed{
p(U,U,N)
=0.098112+0.014976
=0.113088
}.
$$

The probability is not the probability of one hidden path. It is the sum over all eight possible three-step hidden paths.

## 9.6 Answer to Question 2: Most Likely State at Each Time

Normalize each forward vector.

| Time | Observation | $p(S\mid x_{1:n})$ | $p(R\mid x_{1:n})$ | Most Likely State |
|---:|---|---:|---:|---|
| 1 | $U$ | $0.2500$ | $0.7500$ | $R$ |
| 2 | $U$ | $0.1674$ | $0.8326$ | $R$ |
| 3 | $N$ | $0.8676$ | $0.1324$ | $S$ |

At time 3,

$$
p(S\mid U,U,N)
=
\frac{0.098112}{0.113088}
\approx 0.8676.
$$

The no-umbrella observation strongly shifts the current-state belief toward sunny.

## 9.7 Viterbi Step 1

Initialization is the same as forward initialization:

$$
\delta_1(S)=0.6\times0.2=0.12,
$$

$$
\delta_1(R)=0.4\times0.9=0.36.
$$

## 9.8 Viterbi Step 2

### Best Path Ending in $S$

Compare the two predecessor paths:

$$
S\rightarrow S:
0.12\times0.7=0.084,
$$

$$
R\rightarrow S:
0.36\times0.4=0.144.
$$

The second is larger, so

$$
\psi_2(S)=R.
$$

Multiply by the emission probability:

$$
\delta_2(S)
=0.2\times0.144
=0.0288.
$$

### Best Path Ending in $R$

Compare

$$
S\rightarrow R:
0.12\times0.3=0.036,
$$

$$
R\rightarrow R:
0.36\times0.6=0.216.
$$

The second is larger, so

$$
\psi_2(R)=R,
$$

and

$$
\delta_2(R)
=0.9\times0.216
=0.1944.
$$

## 9.9 Viterbi Step 3

### Best Path Ending in $S$

Compare

$$
S\rightarrow S:
0.0288\times0.7=0.02016,
$$

$$
R\rightarrow S:
0.1944\times0.4=0.07776.
$$

Choose $R\rightarrow S$, then multiply by $p(N\mid S)=0.8$:

$$
\delta_3(S)
=0.07776\times0.8
=0.062208.
$$

Thus

$$
\psi_3(S)=R.
$$

### Best Path Ending in $R$

Compare

$$
S\rightarrow R:
0.0288\times0.3=0.00864,
$$

$$
R\rightarrow R:
0.1944\times0.6=0.11664.
$$

Choose $R\rightarrow R$, then multiply by $p(N\mid R)=0.1$:

$$
\delta_3(R)
=0.11664\times0.1
=0.011664.
$$

Thus

$$
\psi_3(R)=R.
$$

## 9.10 Viterbi Termination and Backtracking

The best final state is

$$
\widehat z_3
=
\arg\max\{0.062208,0.011664\}
=S.
$$

Backtrack:

$$
\widehat z_2
=
\psi_3(S)
=R,
$$

$$
\widehat z_1
=
\psi_2(R)
=R.
$$

Therefore

$$
\boxed{
\widehat{\mathbf{Z}}_{\mathrm{Viterbi}}
=(R,R,S)
}.
$$

The probability of this path and the observations is

$$
0.4\times0.9\times0.6\times0.9\times0.4\times0.8
=0.062208.
$$

## 9.11 Verify by Enumerating All Eight Paths

For this tiny example, we can verify the result manually.

| Hidden Path | Joint Probability with $(U,U,N)$ |
|---|---:|
| $(S,S,S)$ | $0.009408$ |
| $(S,S,R)$ | $0.000504$ |
| $(S,R,S)$ | $0.010368$ |
| $(S,R,R)$ | $0.001944$ |
| $(R,S,S)$ | $0.016128$ |
| $(R,S,R)$ | $0.000864$ |
| **$(R,R,S)$** | **$0.062208$** |
| $(R,R,R)$ | $0.011664$ |
| **Sum** | **$0.113088$** |

This table confirms two different results:

1. The forward result is the sum:

$$
p(U,U,N)=0.113088.
$$

2. The Viterbi result is the largest single path:

$$
\max_{\mathbf{Z}}p(\mathbf{X},\mathbf{Z})=0.062208.
$$

The likelihood must be at least as large as the best single-path probability because it includes that path plus all other paths.

## 9.12 What This Example Teaches

- Emissions connect observations to hidden states.
- Transitions prevent each observation from being interpreted independently.
- Forward sums evidence from all possible explanations.
- Filtering gives a posterior over the current state.
- Viterbi chooses one globally consistent explanation.
- In this example, the time-wise filtering modes happen to be $(R,R,S)$, the same as the Viterbi path. This agreement is not guaranteed in general.

---

# §10 Guided Textbook Exercises

> 📖 Adapted from Textbook Exercises 13.1/13.2 and 13.16, pp. 646-650

## 10.1 Exercise A: Verify the First-Order Markov Property

Consider

$$
p(x_1,\ldots,x_N)
=
p(x_1)\prod_{n=2}^{N}p(x_n\mid x_{n-1}).
$$

Show that

$$
p(x_n\mid x_1,\ldots,x_{n-1})
=
p(x_n\mid x_{n-1}).
$$

### Guided Solution

Use

$$
p(x_n\mid x_1,\ldots,x_{n-1})
=
\frac{p(x_1,\ldots,x_n)}{p(x_1,\ldots,x_{n-1})}.
$$

The numerator is

$$
p(x_1)
\prod_{m=2}^{n-1}p(x_m\mid x_{m-1})
\,p(x_n\mid x_{n-1}),
$$

and the denominator is

$$
p(x_1)
\prod_{m=2}^{n-1}p(x_m\mid x_{m-1}).
$$

After cancellation,

$$
\boxed{
p(x_n\mid x_1,\ldots,x_{n-1})
=
p(x_n\mid x_{n-1})
}.
$$

## 10.2 Exercise B: One Path Probability

Use the weather HMM from §9. Calculate

$$
p(\mathbf{X}=(U,U,N),\mathbf{Z}=(S,R,S)).
$$

### Solution

Follow the path:

$$
S\rightarrow R\rightarrow S.
$$

Multiply the initial, emission, transition, emission, transition, and emission terms:

$$
p(S)p(U\mid S)p(R\mid S)p(U\mid R)p(S\mid R)p(N\mid S).
$$

Substitute the values:

$$
0.6\times0.2\times0.3\times0.9\times0.4\times0.8.
$$

Calculate step by step:

$$
0.6\times0.2=0.12,
$$

$$
0.12\times0.3=0.036,
$$

$$
0.036\times0.9=0.0324,
$$

$$
0.0324\times0.4=0.01296,
$$

$$
0.01296\times0.8=0.010368.
$$

Therefore

$$
\boxed{p(U,U,N,S,R,S)=0.010368}.
$$

## 10.3 Exercise C: Derive the Viterbi Recursion Intuitively

Suppose we already know the best partial-path probability $\delta_{n-1}(j)$ for every previous state $j$.

To obtain the best path ending at state $k$ at time $n$:

1. extend the best path ending in each $j$ using transition $A_{jk}$;
2. select the largest extension;
3. multiply by the emission probability $b_k(x_n)$.

Therefore

$$
\boxed{
\delta_n(k)
=
b_k(x_n)
\max_j\left[\delta_{n-1}(j)A_{jk}\right]
}.
$$

This is the main result requested in Textbook Exercise 13.16, presented as a dynamic-programming argument rather than a full max-sum derivation.

## 10.4 Quick Concept Checks

### Question 1

If $A_{jk}=0$, what does this mean?

**Answer:** A transition from state $j$ to state $k$ is impossible. Viterbi will never use that edge, and forward probability mass cannot pass through it.

### Question 2

Can an emission probability alone determine the hidden state?

**Answer:** Not generally. The posterior depends on both the emission evidence and the state prediction produced by the initial and transition probabilities.

### Question 3

Why does the forward algorithm use a sum?

**Answer:** The same observation sequence can be explained by many hidden-state paths. Their probabilities must be added.

### Question 4

Why does Viterbi use a maximum?

**Answer:** It seeks one best hidden-state path rather than the total probability of all paths.

### Question 5

Is $p(\mathbf{X})$ equal to the probability of the Viterbi path?

**Answer:** No. $p(\mathbf{X})$ sums all paths; the Viterbi score is the probability of only the largest path.

---

# §11 From HMMs to RNNs, LSTMs, and Transformers

## 11.1 The Common Goal

HMMs, RNNs, LSTMs, and Transformers all model sequential data. They differ mainly in how they represent and transmit information across positions.

## 11.2 HMM: A Probabilistic Discrete State

An HMM uses a discrete hidden state:

$$
z_n\in\{1,\ldots,K\}.
$$

Its dynamics are explicitly probabilistic:

$$
p(z_n\mid z_{n-1}),
$$

and observations are generated probabilistically:

$$
p(x_n\mid z_n).
$$

Advantages:

- mathematically interpretable;
- explicit uncertainty;
- exact dynamic-programming inference;
- useful when states have clear meanings;
- can encode impossible or constrained transitions.

Limitations:

- the hidden state is usually discrete and relatively small;
- first-order transitions can be restrictive;
- simple emission models may not represent complex signals well;
- training and model design can require substantial domain structure.

## 11.3 RNN: A Learned Continuous Hidden State

A recurrent neural network replaces the discrete random state with a continuous hidden vector:

$$
\mathbf{h}_n
=
f_{\boldsymbol{\theta}}(\mathbf{h}_{n-1},\mathbf{x}_n).
$$

The hidden vector can store a distributed summary of previous inputs.

A simple RNN update is

$$
\mathbf{h}_n
=
\tanh(\mathbf{W}_h\mathbf{h}_{n-1}
+
\mathbf{W}_x\mathbf{x}_n
+
\mathbf{b}).
$$

Compared with an HMM:

- HMM state: discrete and probabilistic;
- RNN state: continuous and usually deterministic given the previous state and input;
- HMM transition/emission functions: often simple tables or distributions;
- RNN transition function: a learned nonlinear neural network.

The conceptual similarity is that both carry a state forward through time.

## 11.4 LSTM: Better Control of Long-Term Memory

A standard RNN can have difficulty preserving information over long sequences. LSTM introduces a cell state and gates:

$$
(\mathbf{c}_n,\mathbf{h}_n)
=
F_{\boldsymbol{\theta}}(\mathbf{c}_{n-1},\mathbf{h}_{n-1},\mathbf{x}_n).
$$

The gates control:

- what information to retain;
- what information to forget;
- what information to expose as output.

The LSTM still processes a sequence recurrently, one step after another, but it has a more flexible memory mechanism than a basic RNN or a small discrete HMM state.

## 11.5 Transformer: Direct Interaction Across Positions

A Transformer does not require information to pass through a single recurrent hidden state one time step at a time. Self-attention allows each position to combine information from other positions:

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
=
\operatorname{softmax}
\left(
\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{d}}
\right)
\mathbf{V}.
$$

For a non-causal encoder, each position can attend to the whole sequence. For an autoregressive Transformer, a causal mask prevents attention to future tokens.

A positional representation is needed because attention by itself does not encode order.

## 11.6 Comparison Table

| Model | Internal Representation | How Dependency Is Modelled | Typical Inference Pattern | Main Strength |
|---|---|---|---|---|
| HMM | Discrete random state $z_n$ | Explicit transition probabilities $p(z_n\mid z_{n-1})$ | Dynamic programming | Interpretability and exact probabilistic inference |
| RNN | Continuous vector $\mathbf{h}_n$ | Recurrent nonlinear update | Sequential forward pass | Flexible learned state representation |
| LSTM | Hidden vector plus memory cell | Gated recurrence | Sequential forward pass | Better long-range memory than a basic RNN |
| Transformer | Contextual representation at each position | Self-attention across positions | Parallel attention layers during training | Direct long-range interaction and scalability |

## 11.7 What Was Preserved from HMM Thinking?

Several HMM ideas remain important in modern sequence modelling.

### State or Context

All sequence models need a representation of context:

- HMM: $z_n$;
- RNN/LSTM: $\mathbf{h}_n$ or $(\mathbf{c}_n,\mathbf{h}_n)$;
- Transformer: contextual token representation after attention.

### Local Evidence and Contextual Evidence

In an HMM, an emission probability evaluates whether an observation matches a state, while transitions provide contextual consistency.

Modern neural models also combine:

- local content from the current input;
- contextual information from other positions.

### Structured Decoding

Viterbi decoding remains useful when outputs must obey structured transition constraints, even if neural networks provide the emission scores. A neural model can estimate local scores, and a dynamic-programming decoder can enforce a globally consistent path.

## 11.8 What Changed?

The main progression is an increase in representational flexibility.

### HMM

$$
\text{small discrete state}
+
\text{explicit probabilistic transitions}.
$$

### RNN/LSTM

$$
\text{learned continuous memory}
+
\text{nonlinear recurrent update}.
$$

### Transformer

$$
\text{contextual representation at every position}
+
\text{direct attention across the sequence}.
$$

This progression does not make HMMs useless. HMMs remain valuable when:

- the number of states is small;
- transitions have a clear meaning;
- uncertainty must be explicit;
- data are limited;
- structured constraints matter;
- transparent inference is preferred.

## 11.9 A Careful Historical and Conceptual Statement

It is useful to teach the progression

$$
\text{HMM}\rightarrow\text{RNN/LSTM}\rightarrow\text{Transformer},
$$

but these models are not simply newer versions of the same mathematical object.

- HMM is a probabilistic latent-variable model.
- A basic RNN is usually a deterministic neural computation graph.
- A Transformer is an attention-based architecture without recurrent state transitions.

Their shared problem is sequence modelling, while their assumptions and inference procedures differ.

---

# §12 Chapter Summary and Concept Checklist

## 12.1 The Seven Essential Equations

### First-Order Markov Assumption

$$
p(z_n\mid z_1,\ldots,z_{n-1})
=
p(z_n\mid z_{n-1}).
$$

### HMM Joint Distribution

$$
p(\mathbf{X},\mathbf{Z})
=
p(z_1)
\prod_{n=2}^{N}p(z_n\mid z_{n-1})
\prod_{n=1}^{N}p(x_n\mid z_n).
$$

### Forward Initialization

$$
\alpha_1(k)=\pi_kb_k(x_1).
$$

### Forward Recursion

$$
\alpha_n(k)
=
b_k(x_n)\sum_j\alpha_{n-1}(j)A_{jk}.
$$

### Sequence Likelihood

$$
p(\mathbf{X})=\sum_k\alpha_N(k).
$$

### Viterbi Recursion

$$
\delta_n(k)
=
b_k(x_n)\max_j\left[\delta_{n-1}(j)A_{jk}\right].
$$

### Viterbi Backpointer

$$
\psi_n(k)
=
\arg\max_j\left[\delta_{n-1}(j)A_{jk}\right].
$$

## 12.2 The Three Questions Revisited

### 1. How probable is the observation sequence?

Use the forward algorithm:

$$
p(\mathbf{X})=\sum_k\alpha_N(k).
$$

### 2. What is the most likely current hidden state?

Normalize the forward vector:

$$
p(z_n=k\mid x_{1:n})
=
\frac{\alpha_n(k)}{\sum_r\alpha_n(r)}.
$$

Then choose the largest posterior.

### 3. What is the most likely complete hidden-state sequence?

Use Viterbi recursion and backtracking:

$$
\widehat{\mathbf{Z}}
=
\arg\max_{\mathbf{Z}}p(\mathbf{X},\mathbf{Z}).
$$

## 12.3 Common Mistakes

| Mistake | Correction |
|---|---|
| Treating observations as hidden states | $x_n$ is observed; $z_n$ is latent. |
| Reading $A_{jk}$ backwards | $A_{jk}=p(z_n=k\mid z_{n-1}=j)$. |
| Forgetting the emission term | Every time step includes both transition and emission evidence. |
| Using max in the forward algorithm | Forward uses a sum over paths. |
| Using a sum in Viterbi | Viterbi keeps the maximum path. |
| Assuming the best state at each time is the best complete path | Local posterior modes need not form the global MAP path. |
| Equating Viterbi score with $p(\mathbf{X})$ | Viterbi gives one path; likelihood sums all paths. |
| Enumerating all $K^N$ paths | Use dynamic programming with $O(NK^2)$ cost. |

## 12.4 Final Conceptual Picture

An HMM separates sequence modelling into two questions:

1. **How does the hidden system evolve?**

$$
p(z_n\mid z_{n-1}).
$$

2. **How does a hidden state produce an observation?**

$$
p(x_n\mid z_n).
$$

The forward algorithm combines all possible explanations of the observations. The Viterbi algorithm selects one globally most probable explanation.

The deeper lesson extends beyond HMMs:

> Sequential models need a mechanism for carrying context across positions and a method for combining local evidence with global consistency.

HMMs do this with probabilistic states and dynamic programming. RNNs and LSTMs do it with learned recurrent memory. Transformers do it with attention across positions.

---

## End-of-Lecture One-Minute Check

A student is ready to move on if they can answer these questions without notes:

1. What is the difference between $x_n$ and $z_n$?
2. What does $A_{jk}$ mean?
3. What does $b_k(x_n)$ mean?
4. Why are there $K^N$ possible hidden paths?
5. What does $\alpha_n(k)$ sum?
6. Why does the forward algorithm use $\sum$?
7. Why does Viterbi use $\max$?
8. Why are backpointers needed?
9. Why can local state decisions disagree with the globally best path?
10. How does an HMM differ conceptually from an RNN or Transformer?
