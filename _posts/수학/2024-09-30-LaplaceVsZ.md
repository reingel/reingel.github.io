---
title: "Laplace Transform vs. z-Transform"
last_modified_at: 2024-09-30
categories:
  - 수학
tags:
  - Laplace Transform
  - z-Transform
excerpt: 
use_math: true
classes: wide
---

•	The **Laplace Transform** and **$z$-Transform** (along with other integral transforms such as the **Wavelet Transform**) are often studied within the framework of **functional analysis**, particularly within the context of **Hilbert spaces** (complete inner product spaces).
•	The $z$-Transform is the discrete-time analogue of the Laplace Transform, often used for analyzing discrete-time signals and systems.

*c.f.* The **Fourier Transform** can be viewed as a special case of the Laplace Transform, where the complex variable $s$ in the Laplace domain is restricted to the imaginary axis (i.e., $s=j\omega$, where $j$ is the imaginary unit and $\omega$ is the frequency).


||Laplace Transform|$z$-Transform|
|-|-|-|
| Transform | $F\left( s \right) = L\left\lbrack f\left( t \right) \right\rbrack = \int_{0}^{\infty}{f(t)e}^{- \text{st}}\text{dt}$ | $X\left( z \right) = Z\left\lbrack x\left( t \right) \right\rbrack = Z\left\lbrack x\left( \text{kT} \right) \right\rbrack = \sum_{k = 0}^{\infty}{x\left( \text{kT} \right)z^{- k}}$ |
| Inverse Transform | $f\left( t \right) = \frac{1}{2\text{πj}}\ \int_{c - j\infty}^{c + j\infty}{F\left( s \right)e^{\text{st}}}\text{ds},\ \ t > 0$<br>where $c$ is the *abscissa of convergence* for $F\left( s \right)$ | $x\left( \text{kT} \right) = Z^{- 1}\left\lbrack X\left( z \right) \right\rbrack = \frac{1}{2\text{πj}}\ \oint_{C}^{}{X\left( z \right)z^{k - 1}}\text{dz}$<br>where $C$ is a circle with its center at the origin of the $z$ plane such that all poles of $X\left( z \right)z^{k - 1}$ are inside it |
| Graphical meaning | ![](https://i.imgur.com/v0icajG.png) | ![](https://i.imgur.com/1s9LhZN.png) |
| $\sin \omega t$   | $\mathcal{L}[\sin \omega t]=\frac{\omega}{s^2 + \omega^2}$<br>![](https://i.imgur.com/z2SthPX.png) | $\mathcal{Z}[\sin \omega k T]=\frac{z \sin \omega T}{z^2 - 2 z \cos \omega T + 1} = \frac{z(e^{j\omega T} - e^{-j\omega T})/{2j}}{z^2-z(e^{j\omega T} - e^{-j\omega T})+1}$<br>![](https://i.imgur.com/q85PfUm.png) |
| $\cos \omega t$   | $\mathcal{L}[\cos \omega t]=\frac{s}{s^2+\omega^2}$<br>![](https://i.imgur.com/vlS4AwG.png) | $\mathcal{Z}[\cos \omega k T]=\frac{z(z- \cos \omega T)}{z^2 - 2 z \cos \omega T + 1} = \frac{z[z-(e^{j\omega T} + e^{-j\omega T})/{2}]}{z^2-z(e^{j\omega T} + e^{-j\omega T})+1}$<br>![](https://i.imgur.com/hVPkBiv.png) |
| $\sinh \omega t$  | $\mathcal{L}[\sinh \omega t]=\frac{\omega}{s^2 - \omega^2}$<br>![](https://i.imgur.com/uv5AHec.png) | $\mathcal{Z}[\sinh \omega k T]=\frac{z \sinh \omega T}{z^2 - 2 z \cosh \omega T + 1} = \frac{z(e^{\omega T} - e^{-\omega T})/{2}}{z^2-z(e^{\omega T} + e^{-\omega T})+1}$<br>![](https://i.imgur.com/6UMcqnu.png) |
| $\cosh \omega t$  | $\mathcal{L}[\cosh \omega t]=\frac{s}{s^2-\omega^2}$<br>![](https://i.imgur.com/CIsDSku.png) | $\mathcal{Z}[\cosh \omega k T]=\frac{z(z- \cosh \omega T)}{z^2 - 2 z \cosh \omega T + 1} = \frac{z[z-(e^{\omega T} + e^{-\omega T})/{2}]}{z^2-z(e^{\omega T} + e^{-\omega T})+1}$<br>![](https://i.imgur.com/YqMigrF.png) |
