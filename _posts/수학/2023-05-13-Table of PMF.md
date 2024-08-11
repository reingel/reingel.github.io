---
title: "Table of Probability Mass Functions"
last_modified_at: 2023-05-13
categories:
  - 수학
tags:
excerpt: ""
use_math: true
classes: wide
---

||probability mass function<br>$f(x)$|argument and parameter(s)|Moment-generating function<br>$M_{X}\left( t \right) = E\left( e^{\text{tX}} \right)$|Moments</br>$\mu = E\left( X \right) = \sum_{x}^{}{x \cdot f(x)}$<br>$\sigma^{2} = \text{Var}\left( X \right) = \sum_{x}^{}{\left( x - \mu \right)^{2} \cdot f\left( x \right)}$|Remark|Corresponding of continuous distributions|R|
|:-|:-|:-|:-|:-|:-|:-|:-|
|discrete uniform distribution|$f\left( x \right) = \frac{1}{k}$<br>for $x = x_{1},\ x_{2},\ \cdots,\ x_{k}$<br>where $x_{i} \neq x_{j}$ when $i \neq j$|$k = 1,\ 2,\ 3,\ \cdots$|$M_{X}\left( t \right) = \frac{e^{t}\left( 1 - e^{\text{kt}} \right)}{k\left( 1 - e^{t} \right)}$|If $x = 1,\ 2,\ 3,\ \cdots,\ k$<br>$\mu = \frac{k + 1}{2}$<br>$\sigma^{2} = \frac{k^{2} - 1}{12}$||||
|Bernoulli distribution|$f\left( x;\ \theta \right) = \theta^{x}\left( 1 - \theta \right)^{1 - x}$<br>for $x = 0,\ 1$|$x$ (no. of success)<br>$0 \leq \theta \leq 1$|$M_{X}\left( t \right) = 1 + \theta\left( e^{t} - 1 \right)$|$\mu = \theta$<br>$\sigma^{2} = \theta(1 - \theta)$|Bernoulli distribution with $n = 1$|||
|binomial distribution|$b\left( x;n,\ \theta \right) = \begin{pmatrix}n \\ x \\ \end{pmatrix}\theta^{x}\left( 1 - \theta \right)^{n - x}$<br>for $x = 0,\ 1,\ 2,\ \cdots,\ n$|$x$ (no. of success)<br>$n = 1, 2, 3,\cdots$<br>$0 \leq \theta \leq 1$|$M_{X}\left( t \right) = \left\lbrack 1 + \theta\left( e^{t} - 1 \right) \right\rbrack^{n}$|$\mu = \text{nθ}$<br>$\sigma^{2} = \text{nθ}(1 - \theta)$|[^1]|Normal distribution of<br>$\mu = \text{nθ}$<br>$\sigma^{2} = \text{nθ}(1 - \theta)$|\[dpqr\]binom(x, n, p)
|negative binomial distribution|$b^{*}\left( x;k,\ \theta \right) = \begin{pmatrix} x - 1 \\ k - 1 \\ \end{pmatrix}\theta^{k}\left( 1 - \theta \right)^{x - k}$<br>for $x = k,\ k + 1,\ k + 2,\ \cdots$|||$\mu = \frac{k}{\theta}$<br>$\sigma^{2} = \frac{k}{\theta}\left( \frac{1}{\theta} - 1 \right)$|[^2]|||
|geometric distribution|$g\left( x;\ \theta \right) = \theta\left( 1 - \theta \right)^{x - 1}$<br>for $x = 1,\ 2,\ 3,\ \cdots$|$x$ (no. of trials needed to get $k$ success)<br>$k = 1,\ 2,\ 3,\ \cdots$<br>$0 \leq \theta \leq 1$|$M_{X}\left( t \right) = \frac{\theta e^{t}}{1 - e^{t}\left( 1 - \theta \right)}$|$\mu = \frac{1}{\theta}$<br>$\sigma^{2} = \frac{1 - \theta}{\theta^{2}}$|negative binomial distribution with $k = 1$|Exponential distribution||
|hypergeometric distribution|$h\left( x;n,\ N,\ M \right) = \frac{\begin{pmatrix} M \\ x \\ \end{pmatrix}\begin{pmatrix} N - M \\ n - x \\ \end{pmatrix}}{\begin{pmatrix} N \\ n \\ \end{pmatrix}}$<br>for $x = 0,\ 1,\ 2,\ \cdots,\ n$, $x \leq M$ and $n - x \leq N - M$|$x$ (no. of sampled success elements)<br>$n$ (no. of sampling)<br>$N$ (no. of total elements)<br>$M$ (no. of total success elements)|*fairly complex*|$\mu = \frac{\text{nM}}{N}$<br>$\sigma^{2} = \frac{\text{nM}\left( N - M \right)\left( N - n \right)}{N^{2}\left( N - 1 \right)}$||||
|Poisson distribution|$p\left( x;\ \lambda \right) = \frac{\lambda^{x}e^{- \lambda}}{x!}$<br>for $x = 0,\ 1,\ 2,\ \cdots$|$x$ (no. of successes within a range)<br>$\lambda > 0$|$M_{X}\left( t \right) = e^{\lambda(e^{t} - 1)}$|$\mu = \lambda$<br>$\sigma^{2} = \lambda$|[^3]||\[dpqr\]pois(x, lambda)|

[^1]: $f(x;\ \theta) + f(x;\ \theta) + \cdots + f(x;\ \theta)$<br>$\rightarrow$ sum of $n$ Bernoulli distributions

[^2]: The number of trials needed to get the $k$th success = k - 1 number of successes on $x - 1$ number of trials and 1 success of Bernoulli distributions

[^3]: Infinite number of Bernoulli distributions within a range<br>($\lambda$ = the rate at which the events occur) <br>Approximation of Bernoulli distributions when $n$ is large

