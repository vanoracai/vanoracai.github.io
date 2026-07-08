---
layout: course
---

<style>
  .lecture-toggle {
    margin: 1rem 0;
    padding: 0 1rem 1rem;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: #ffffff;
  }

  .lecture-toggle:not([open]) {
    padding-bottom: 0;
  }

  .lecture-toggle > summary {
    display: block;
    margin: 0 -1rem;
    padding: 0.7rem 1rem;
    border-radius: 8px;
    background: #e8f1ff;
    color: #0f172a;
    cursor: pointer;
    list-style: none;
  }

  .lecture-toggle:nth-of-type(2) > summary {
    background: #e8f8f1;
  }

  .lecture-toggle:nth-of-type(3) > summary {
    background: #fff4df;
  }

  .lecture-toggle:nth-of-type(4) > summary {
    background: #f1ecff;
  }

  .lecture-toggle:nth-of-type(5) > summary {
    background: #ffeef2;
  }

  .lecture-toggle > summary::-webkit-details-marker {
    display: none;
  }

  .lecture-title {
    display: block;
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.3;
  }

  .lecture-title::before {
    content: "+";
    display: inline-block;
    width: 1.2em;
    margin-right: 0.35em;
    font-weight: 700;
  }

  .lecture-toggle[open] > summary {
    margin-bottom: 1rem;
    border-bottom: 1px solid #cbd5e1;
    border-radius: 8px 8px 0 0;
  }

  .lecture-toggle[open] .lecture-title::before {
    content: "-";
  }

  .lecture-description {
    display: block;
    margin: 0.6rem 0 0 1.55rem;
    padding: 0.55rem 0.8rem;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    background: #f8fbff;
    color: #1e293b;
    font-weight: 400;
    line-height: 1.5;
  }
</style>

# Pattern Recognition and Machine Learning

## Course Materials, 2026

This page collects lecture notes for the Pattern Recognition and Machine Learning course, following Christopher M. Bishop's *Pattern Recognition and Machine Learning*.

- [Textbook PDF](./CoursePR2026/教科书.pdf)
- [Lecture 5 package](./CoursePR2026/lecture_5_package.zip)

<details class="lecture-toggle" markdown="1">
<summary>
<span class="lecture-title">Lecture 1: Pattern Recognition and Machine Learning Introduction</span>
<span class="lecture-description">This lecture introduces the core viewpoint of PRML Chapter 1: probability as a language for reasoning under uncertainty, decision theory as a framework for making optimal choices, and information theory as a way to measure uncertainty and model comparison.</span>
</summary>

{% include_relative CoursePR2026/lecture_1.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary>
<span class="lecture-title">Lecture 2: Probability Distributions and Density Estimation</span>
<span class="lecture-description">This lecture develops the probability distributions used throughout pattern recognition, including binary and multinomial variables, Gaussian models, exponential-family structure, conjugate priors, and nonparametric density estimation.</span>
</summary>

{% include_relative CoursePR2026/lecture_2.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary>
<span class="lecture-title">Lecture 3: Linear Models for Regression</span>
<span class="lecture-description">This lecture studies linear basis function models for regression, connecting least squares, regularization, the bias-variance tradeoff, Bayesian linear regression, model comparison, and the evidence approximation.</span>
</summary>

{% include_relative CoursePR2026/lecture_3.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary>
<span class="lecture-title">Lecture 4: Linear Models for Classification</span>
<span class="lecture-description">This lecture studies linear classification models, including discriminant functions, generative and discriminative probabilistic classifiers, logistic regression, the Laplace approximation, and Bayesian logistic regression.</span>
</summary>

{% include_relative CoursePR2026/lecture_4.md %}

</details>

<details class="lecture-toggle" markdown="1">
<summary>
<span class="lecture-title">Lecture 5: Neural Networks</span>
<span class="lecture-description">This lecture studies feed-forward neural networks, including adaptive basis functions, network training, backpropagation, Hessian-based analysis, regularization, mixture density networks, and Bayesian neural networks.</span>
</summary>

{% include_relative CoursePR2026/lecture_5.md %}

</details>

<script>
  document.addEventListener("DOMContentLoaded", function () {
    const lectureToggles = Array.from(document.querySelectorAll(".lecture-toggle"));

    function showOnly(openLecture) {
      lectureToggles.forEach(function (lecture) {
        lecture.hidden = Boolean(openLecture && lecture !== openLecture);
      });
    }

    lectureToggles.forEach(function (lecture) {
      lecture.addEventListener("toggle", function () {
        if (lecture.open) {
          showOnly(lecture);
          return;
        }

        const openLecture = lectureToggles.find(function (item) {
          return item.open;
        });
        showOnly(openLecture);
      });
    });
  });
</script>
