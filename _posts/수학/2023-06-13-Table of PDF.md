---
title: "Table of Probability Density Functions"
last_modified_at: 2023-06-13
categories:
  - 수학
tags:
excerpt: ""
use_math: true
classes: wide
---


||probability density function|argument and parameter(s)|Moment-generating function<br>$M_{X}\left( t \right) = E(e^{\text{tX}})$|Moments<br>$\mu = E\left( X \right) = \int_{- \infty}^{\infty}{x \cdot f\left( x \right)\text{dx}}$<$\sigma^{2} = \text{Var}\left( X \right) = \int_{- \infty}^{\infty}{\left( x - \mu \right)^{2} \cdot f\left( x \right)\text{dx}}$|Random variable generator|Remark|Corresponding of discrete distributions|R|
|:-|:-|:-|:-|:-|:-|:-|:-|:-|
|uniform distribution|$u\left( x;\ \alpha,\ \beta \right)$ |$\alpha < \beta$||$\mu = \frac{\alpha + \beta}{2}$<br>$\sigma^{2} = \frac{\left( \beta - \alpha \right)^{2}}{12}$|$X=\alpha +(\beta -\alpha )U(0,1)$|The beta distribution with $\alpha = \beta = 1$.||\[dpqr\]unif(x, alpha, beta)|
|uniform distribution|$u\left( x;\ \alpha,\ \beta \right) = \left\{ \begin{matrix} \frac{1}{\beta - \alpha} & \mathrm{\text{for~}}\alpha < x < \beta \\ 0 & \mathrm{\text{elsewhere}} \\ \end{matrix} \right)$|$\alpha < \beta$||$\mu = \frac{\alpha + \beta}{2}$<br>$\sigma^{2} = \frac{\left( \beta - \alpha \right)^{2}}{12}$|$X=\alpha +(\beta -\alpha )U(0,1)$|The beta distribution with $\alpha = \beta = 1$.||\[dpqr\]unif(x, alpha, beta)|
|beta distribution|$f\left( x \right) = \left\{ \begin{matrix} \frac{\Gamma\left( \alpha + \beta \right)}{\Gamma\left( \alpha \right)\Gamma\left( \beta \right)}x^{\alpha - 1}\left( 1 - x \right)^{\beta - 1} & \mathrm{\text{for~}}0 < x < 1 \\ 0 & \mathrm{\text{elsewhere}} \\ \end{matrix} \right)$|$\alpha > 0$<br>$\beta > 0$||$\mu = \frac{\alpha}{\alpha + \beta}$<br>$\sigma^{2} = \frac{\text{αβ}}{\left( \alpha + \beta \right)^{2}\left( \alpha + \beta + 1 \right)}$||[^1]||\[dpqr\]beta(x, alpha, beta)|
|exponential distribution|$g(x;\ \lambda) = \left\{ \begin{matrix} \frac{1}{\lambda}e^{- \frac{x}{\lambda}} & \mathrm{\text{for~}}x > 0 \\ 0 & \mathrm{\text{elsewhere}} \\ \end{matrix} \right)$|$\lambda > 0$|$M_{X}\left( t \right) = \left( 1 - \text{λt} \right)^{- 1}$|$\mu = \frac{1}{\lambda}$<br>$\sigma^{2} = \frac{1}{\lambda^{2}}$|$X=-\frac{1}{\lambda}ln(1-U(0,1))$ or $X=-\frac{1}{\lambda}ln(U(0,1))$|[^2]||\[dpqr\]exp(x, lambda)|
|chi-square distribution|$f\left( x \right) = \left\{ \begin{matrix} \frac{1}{\lambda^{\nu/2}\Gamma\left( \nu/2 \right)}x^{\frac{\nu}{2} - 1}e^{- \frac{x}{2}} & \mathrm{\text{for~}}x > 0 \\ 0 & \mathrm{\text{elsewhere}} \\ \end{matrix} \right)$|$\nu > 0$ (no. of degrees of freedom = degrees of freedom)|$M_{X}\left( t \right) = \left( 1 - 2t \right)^{- \nu/2}$|$\mu = \nu$<br>$\sigma^{2} = 2\nu$||[^3]|
|gamma distribution|$g(x;\ \alpha,\ \lambda) = \left\{ \begin{matrix} \frac{1}{\lambda^{\alpha}\Gamma\left( \alpha \right)}x^{\alpha - 1}e^{- \frac{x}{\lambda}} & \mathrm{\text{for~}}x > 0 \\ 0 & \mathrm{\text{elsewhere}} \\ \end{matrix} \right)$|$\alpha > 0$<br>$\lambda > 0$|$M_{X}\left( t \right) = \left( 1 - \text{λt} \right)^{- \alpha}$|$\mu = \frac{\alpha}{\lambda}$<br>$\sigma^{2} = \frac{\alpha}{\lambda^{2}}$||[^4]||\[dpqr\]gamma(x, alpha, lambda)|
|normal distribution<br>(= Gaussian distribution)|$n(x;\ \mu,\ \sigma) = \frac{1}{\sigma\sqrt{2\pi}}e^{- \frac{1}{2}\left( \frac{x - \mu}{\sigma} \right)^{2}}$|$- \infty < \mu < \infty$<br>$\sigma > 0$|$M_{X}\left( t \right) = e^{\text{μt} + \frac{1}{2}\sigma^{2}t^{2}}$|$\mu = \mu$<br>$\sigma^{2} = \sigma^{2}$|$X=\sqrt{2}\text{erf}^{-1}(2U(0,1)-1)$|[^5]|Binomial distribution<br>$\mu = \text{np}$<br>$\sigma^{2} = \text{np}(1 - p)$|\[dpqr\]norm(x, mu, sigma)|
|standard normal distribution|$n(x;0,\ 1) = \frac{1}{\sqrt{2\pi}}e^{- \frac{1}{2}x^{2}}$||$M_{X}\left( t \right) = e^{\frac{t^{2}}{2}}$|$\mu = 0$<br>$\sigma^{2} = 1$||
|Cauchy distribution|$f\left( x \right) = \frac{\beta/\pi}{\beta^{2} + \left( x - \alpha \right)^{2}}$<br>for $- \infty < x < \infty$|$- \infty < \alpha < \infty$<br>$\beta > 0$||$\mu$ and $\sigma^{2}$ does not exist.||Normal $\div$ Normal|
|Rayleigh distribution|$f\left( x \right) = \left\{ \begin{matrix} 2\text{αx}e^{- \alpha x^{2}} & \mathrm{\text{for~}}x > 0 \\ 0 & \mathrm{\text{elsewhere}} \\ \end{matrix} \right)$|$\alpha > 0$||$\mu = \frac{1}{2}\sqrt{\frac{\pi}{\alpha}}$<br>$\sigma^{2} = \frac{1}{\alpha}\left( 1 - \frac{\pi}{4} \right)$|||
|Pareto distribution|$f\left( x \right) = \left\{ \begin{matrix} \frac{\alpha}{x^{\alpha + 1}} & \mathrm{\text{for~}}x > 1 \\ 0 & \mathrm{\text{elsewhere}} \\ \end{matrix} \right)$|$\alpha > 0$||$\mu = \frac{\alpha}{\alpha - 1}\mathrm{\text{~~~~~provided~}}\alpha > 1$<br>$\sigma^{2} = \frac{\alpha}{\left( \alpha - 1 \right)^{2}\left( \alpha - 2 \right)}\mathrm{\text{~~~provided~}}\alpha > 2$<br>\* The $r$-th moment about the origin $\mu_{r}^{'}$ exists only if $r < \alpha$|$X=(1-U(0,1))^{-\frac{1}{\alpha}}$ or $X=U(0,1)^{-\frac{1}{\alpha}}$||
|Weibull distribution|$f\left( x \right) = \left\{ \begin{matrix} kx^{\beta - 1}e^{- \alpha x^{\beta}} & \mathrm{\text{for~}}x > 0 \\ 0 & \mathrm{\text{elsewhere}} \\ \end{matrix} \right)$|$\alpha > 0$<br>$\beta > 0$||$\mu = \alpha^{- \frac{1}{\beta}}\Gamma\left( 1 + \frac{1}{\beta} \right)$||[^6]|


\* gamma function

$\Gamma\left( \alpha \right) = \int_{0}^{\infty}{x^{\alpha - 1}e^{-x}\text{dx}}\mathrm{\text{~~~for~}}\alpha > 0$

$\Gamma\left( \alpha \right) = \left( \alpha - 1 \right)!$

[^1]: In recent year, the beta distribution has found important applications in **Bayesian inference**, where parameters are looked upon as random variables, and there is a need for a fairly \"flexible\" probability density for the parameter $\theta$ of the binomial distribution, which takes on nonzero values only on the interval from 0 to 1.<br>If $\alpha > 1$ and $\beta > 1$, the beta density has a relative maximum at $x = \frac{\alpha - 1}{\alpha + \beta - 2}$

[^2]: Waiting time between random events ($\lambda$ = the rate at which the events occur) = The gamma distribution with $\alpha = 1$

[^3]: The gamma distribution with $\alpha = \nu/2$ and $\lambda = 2$. The chi-square distribution plays a very important role in sampling theory.

[^4]: The total waiting time for all $\alpha$ events (independent and identically distributed $\text{Exp}(\lambda)$) to occur distribution = $\text{Exp}\left( \lambda \right) + \text{Exp}\left( \lambda \right) + \cdots$ $\rightarrow$ sum of $\alpha$ exponential distributions = used to model positive-valued, continuous quantities whose distribution is right-skewed = As $\alpha$ increases, the gamma distribution more closely resembles the normal

[^5]: Limiting distribution of sums (and averages) of random variables (Central Limit Theorem)<br>\* 99% of the probability mass is concentrated within $3\sigma$

[^6]: Lieblein and Zelen proposed. This is useful for modeling the number of revolutions to failure. Weibull distributions with $\beta = 1$ are exponential distributions.

