---
title: "z-Transform"
last_modified_at: 2024-09-30
categories:
  - 수학
tags:
  - z-Transform
excerpt: 
use_math: true
classes: wide
---

## Definition

$$X\left( z \right) = Z\left\lbrack x\left( t \right) \right\rbrack = Z\left\lbrack x\left( \text{kT} \right) \right\rbrack = \sum_{k = 0}^{\infty}{x\left( \text{kT} \right)z^{- 1}}$$

It is not necessary to specify the region of $z$ over which $X(z)$ is convergent. It suffices to know that such a region exists. The $z$ transform $X(z)$ of a time function $x(t)$ obtained in this way is valid throughout the $z$ plane except at poles of $X(z)$.

## z-Transforms of Elementary Functions

### Unit-Step Function

$$x\left( t \right) = \{\begin{matrix}
1\left( t \right), & t \geq 0 \\
0, & t < 0 \\
\end{matrix}$$

$$X\left( z \right) = Z\left\lbrack 1\left( t \right) \right\rbrack = \sum_{k = 0}^{\infty}{1z^{- k}} = \sum_{k = 0}^{\infty}z^{- k} = 1 + z^{- 1} + z^{- 2} + \ldots = \frac{1}{1 - z^{- 1}}$$

### Unit-Ramp Function

$$x\left( t \right) = \{\begin{matrix}
t, & t \geq 0 \\
0, & t < 0 \\
\end{matrix}$$

#### by definition

$$X\left( z \right) = Z\left\lbrack t \right\rbrack = \sum_{k = 0}^{\infty}{x\left( \text{kT} \right)z^{- k}} = \sum_{k = 0}^{\infty}{\text{kT}z^{- k}} = T\sum_{k = 0}^{\infty}{kz^{- k}} = T\left( z^{- 1} + 2z^{- 2} + 3z^{- 3} + \ldots \right) = \frac{Tz^{- 1}}{\left( 1 - z^{- 1} \right)^{2}}$$

#### by complex differentiation

Let $x(t)$ is unit-step function, then $X(z)$ is $\frac{1}{1 - z^{- 1}}$.

$$Z\left\lbrack \text{kx}\left( k \right) \right\rbrack = - z\frac{d}{\text{dz}}X\left( z \right) = - z\frac{d}{\text{dz}}\frac{1}{1 - z^{- 1}} = - z\frac{- z^{- 2}}{\left( 1 - z^{- 1} \right)^{2}} = \frac{z^{- 1}}{\left( 1 - z^{- 1} \right)^{2}}$$

$$Z\left\lbrack \text{kTx}\left( \text{kT} \right) \right\rbrack = \text{TZ}\left\lbrack \text{kx}\left( k \right) \right\rbrack = T\frac{z^{- 1}}{\left( 1 - z^{- 1} \right)^{2}} = \frac{Tz^{- 1}}{\left( 1 - z^{- 1} \right)^{2}}$$

### Polynomial Function $\mathbf{a}^{\mathbf{k}}$

$$x\left( k \right) = \{\begin{matrix}
a^{k}, & k = 0,1,2,\ldots \\
0, & k < 0 \\
\end{matrix}$$

$$X\left( z \right) = Z\left\lbrack a^{k} \right\rbrack = \sum_{k = 0}^{\infty}{x\left( k \right)z^{- k}} = \sum_{k = 0}^{\infty}{a^{k}z^{- k}} = \sum_{k = 0}^{\infty}\left( az^{- 1} \right)^{k} = \frac{1}{1 - az^{- 1}}$$

### Exponential Function

$$x\left( t \right) = \{\begin{matrix}
e^{- \text{at}}, & t \geq 0 \\
0, & t < 0 \\
\end{matrix}$$

$$X\left( z \right) = Z\left\lbrack e^{- \text{at}} \right\rbrack = \sum_{k = 0}^{\infty}{x\left( \text{kT} \right)z^{- k}} = \sum_{k = 0}^{\infty}{e^{- \text{akT}}z^{- k}} = \sum_{k = 0}^{\infty}\left( e^{- \text{aT}}z^{- 1} \right)^{k} = \frac{1}{1 - e^{- \text{aT}}z^{- 1}}$$

### Constant Matrix

$$Z\left\lbrack \mathbf{G}^{k} \right\rbrack = \mathbf{I} + \mathbf{G}z^{- 1} + \mathbf{G}^{2}z^{- 2} + \ldots$$

$$\mathbf{G}z^{- 1}Z\left\lbrack \mathbf{G}^{k} \right\rbrack = \mathbf{G}z^{- 1} + \mathbf{G}^{2}z^{- 2} + \mathbf{G}^{3}z^{- 3} + \ldots$$

$$Z\left\lbrack \mathbf{G}^{k} \right\rbrack - \mathbf{G}z^{- 1}Z\left\lbrack \mathbf{G}^{k} \right\rbrack = \left( \mathbf{I} - \mathbf{G}z^{- 1} \right)Z\left\lbrack \mathbf{G}^{k} \right\rbrack = \mathbf{I}$$

$$Z\left\lbrack \mathbf{G}^{k} \right\rbrack = \left( \mathbf{I} - \mathbf{G}z^{- 1} \right)^{- 1} = \left( z\mathbf{I} - \mathbf{G} \right)^{- 1}z$$

$$\mathbf{G}^{k} = Z^{- 1}\left\lbrack \left( \mathbf{I} - \mathbf{G}z^{- 1} \right)^{- 1} \right\rbrack = Z^{- 1}\lbrack\left( z\mathbf{I} - \mathbf{G} \right)^{- 1}z\rbrack$$

## Time Shift

Multiplication of a $z$ transform by $z^{- n}$ has the effect of delaying the time function $x(t)$ by time $\text{nT}$. (That is, move the function to the right by time $\text{nT}$.)

$$Z\left\lbrack x\left( t - \text{nT} \right) \right\rbrack = Z\left\lbrack x\left( \text{kT} - \text{nT} \right) \right\rbrack = Z\left\lbrack x\left( \left( k - n \right)T \right) \right\rbrack = Z\left\lbrack x\left( k - n \right) \right\rbrack = z^{- n}X(z)$$

Multiplication of the $z$ transform by $z$ has the effect of advancing the signal $x(\text{kT})$ by one step (1 sampling period) and that multiplication of the $z$ transform $X(z)$ by $z^{- 1}$ has the effect of delaying the signal $x(\text{kT})$ by one step (1 sampling period).

$$Z\left\lbrack x\left( t + \text{nT} \right) \right\rbrack = Z\left\lbrack x\left( k + n \right) \right\rbrack = z^{n}X\left( z \right) - z^{n}x\left( 0 \right) - z^{n - 1}x\left( 1 \right) - z^{n - 2}x\left( 2 \right) - \ldots - \text{zx}(n - 1)$$


## Initial/Final Value Theorem

### Initial Value Theorem

$$X\left( z \right) = \sum_{k = 0}^{\infty}{x\left( k \right)z^{- k}} = x\left( 0 \right) + x\left( 1 \right)z^{- 1} + x\left( 2 \right)z^{- 2} + x\left( 3 \right)z^{- 3} + \ldots$$

Letting $z \rightarrow \infty$ in this last equation, we obtain the following equation.

$$x\left( 0 \right) = \lim_{z \rightarrow \infty}{X(z)}$$

### Final Value Theorem

$$Z\left\lbrack x\left( k \right) \right\rbrack = X\left( z \right) = \sum_{k = 0}^{\infty}{x\left( k \right)z^{- k}}$$

$$Z\left\lbrack x\left( k - 1 \right) \right\rbrack = z^{- 1}X\left( z \right) = \sum_{k = 0}^{\infty}{x\left( k - 1 \right)z^{- k}}$$

$$\sum_{k = 0}^{\infty}{x\left( k \right)z^{- k}} - \sum_{k = 0}^{\infty}{x\left( k - 1 \right)z^{- k}} = X\left( z \right) - z^{- 1}X\left( z \right) = \left( 1 - z^{- 1} \right)X(z)$$

$$\lim_{z \rightarrow 1}\left\lbrack \sum_{k = 0}^{\infty}{x\left( k \right)z^{- k}} - \sum_{k = 0}^{\infty}{x\left( k - 1 \right)z^{- k}} \right\rbrack = \left\lbrack x\left( 0 \right) - x\left( - 1 \right) \right\rbrack + \left\lbrack x\left( 1 \right) - x\left( 0 \right) \right\rbrack + \left\lbrack x\left( 2 \right) - x\left( 1 \right) \right\rbrack + \ldots = x\left( \infty \right) = \lim_{k \rightarrow \infty}{x(k)}$$

$$\lim_{k \rightarrow \infty}{x(k)} = \lim_{z \rightarrow 1}{\lbrack\left( 1 - z^{- 1} \right)X(z)}\rbrack$$


## Complex Differentiation/Integration

 

### Complex Differentiation

$$X\left( z \right) = \sum_{k = 0}^{\infty}{x\left( k \right)z^{- k}}$$

$$\frac{d}{\text{dz}}X\left( z \right) = \sum_{k = 0}^{\infty}{\left( - k \right)x\left( k \right)z^{- k - 1}}$$

$$- z\frac{d}{\text{dz}}X\left( z \right) = \sum_{k = 0}^{\infty}{\text{kx}\left( k \right)z^{- k}} = Z\lbrack\text{kx}\left( k \right)\rbrack$$

$$Z\left\lbrack \text{kx}\left( k \right) \right\rbrack = - z\frac{d}{\text{dz}}X(z)$$

### Complex Integration

Let $g\left( k \right) = \frac{x\left( k \right)}{k}$ where $x(k)/k$ is finite for $k = 0$.

$$Z\left\lbrack \frac{x\left( k \right)}{k} \right\rbrack = G\left( z \right) = \sum_{k = 0}^{\infty}{\frac{x\left( k \right)}{k}z^{- k}}$$

$$\frac{d}{\text{dz}}G\left( z \right) = \sum_{k = 0}^{\infty}{\frac{x\left( k \right)}{k}( - k)z^{- k - 1}} = - \sum_{k = 0}^{\infty}{x\left( k \right)z^{- k - 1}} = - z^{- 1}\sum_{k = 0}^{\infty}{x\left( k \right)z^{- k}} = - \frac{X\left( z \right)}{z}$$

$$\int_{z}^{\infty}{\frac{d}{\text{dz}}G\left( z \right)\text{dz}} = G\left( \infty \right) - G\left( z \right) = - \int_{z}^{\infty}{\frac{X\left( z_{1} \right)}{z_{1}}dz_{1}}$$

$$G\left( z \right) = \int_{z}^{\infty}{\frac{X\left( z_{1} \right)}{z_{1}}dz_{1}} + G(\infty)$$

$$G\left( \infty \right) = \lim_{z \rightarrow \infty}{G(z)} = g\left( 0 \right) = \lim_{k \rightarrow 0}\frac{x\left( k \right)}{k}$$

$$Z\left\lbrack \frac{x\left( k \right)}{k} \right\rbrack = \int_{z}^{\infty}{\frac{X\left( z_{1} \right)}{z_{1}}dz_{1}} + \lim_{k \rightarrow 0}\frac{x\left( k \right)}{k}$$


## Inverse z-Transform

### Inverse z-Transform

$$x\left( \text{kT} \right) = Z^{- 1}\left\lbrack X\left( z \right) \right\rbrack = \frac{1}{2\text{πj}}\ \oint_{C}^{}{X\left( z \right)z^{k - 1}}\text{dz}$$

where $C$ is a circle with its center at the origin of the z plane such that all poles of $X\left( z \right)z^{k - 1}$ are inside it

### Tip

If $X(z)$ is the form of

$$\frac{1}{1 - A}$$

$x\left( t \right) = x(\text{kT}) = Z^{- 1}\left\lbrack X\left( z \right) \right\rbrack$ will be

$$\frac{A^{k}}{z^{- k}}$$

#### Example 1

If

$$X\left( z \right) = \frac{1}{1 - z^{- 1}}$$

then,

$$x\left( \text{kT} \right) = \frac{\left( z^{- 1} \right)^{k}}{z^{- k}} = 1$$

#### Example 2

If

$$X\left( z \right) = \frac{1}{1 - e^{- \text{aT}}z^{- 1}}$$

then,

$$x\left( \text{kT} \right) = \frac{\left( e^{- \text{aT}}z^{- 1} \right)^{k}}{z^{- k}} = e^{- \text{akT}}$$

or

$$x\left( t \right) = e^{- \text{at}}$$

