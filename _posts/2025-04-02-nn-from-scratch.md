---
title: "Simple Neural Network & Softmax Function with NumPy"
layout: single
date: 2024-04-02
category: [deeplearning]
header:
  teaser: https://github.com/user-attachments/assets/478e7a0c-08ae-48a0-bbd8-8a9a9972fe8d
---
<!--more-->

## Simple Neural Network with NumPy

This is a **simple 3-layer neural network implemented using only NumPy**,  
**without deep learning frameworks** like TensorFlow or PyTorch.

Only **forward propagation** is implemented to demonstrate the core concepts of neural networks.


## Code Overview

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def soft_max(x):
    x = x - np.max(x)  # Prevent overflow
    return np.exp(x) / np.sum(np.exp(x))

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = soft_max(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
```


## Network Architecture

```
Input (2,)
↓
Affine (W1, b1)
↓
Sigmoid
↓
Affine (W2, b2)
↓
Sigmoid
↓
Affine (W3, b3)
↓
Softmax
↓
Output (2,) ← Class probabilities
```


## Preventing Overflow in Softmax

### Basic Definition

The softmax function is defined as:

<img width="215" alt="image" src="https://github.com/user-attachments/assets/478e7a0c-08ae-48a0-bbd8-8a9a9972fe8d" />

However, when the values of \( x_i \) are large,  
calculating \( e^{x_i} \) can cause **numerical overflow**.


### Solution: Subtract the Maximum Value

```python
def soft_max(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```

By subtracting the maximum value, we maintain numerical stability  
**without affecting the output probabilities**.

<img width="336" alt="image" src="https://github.com/user-attachments/assets/e78cb98b-8421-4200-a3af-1073dad4f40f" /><br>
<img width="301" alt="image" src="https://github.com/user-attachments/assets/fba9a07a-335d-4137-b443-a29fd89c9ccc" /><br>
<img width="322" alt="image" src="https://github.com/user-attachments/assets/21c46edf-fcf9-40f0-9e3f-bf644bd68db2" />

This trick ensures **stable computation while preserving the same probability distribution**.

## Softmax Function with Batch Input (3 Versions)

When handling **batch inputs (2D arrays)** in a neural network,  
softmax must be applied **row-wise** (i.e., for each sample in the batch).  
Below are three implementations:

### Version 1: Without keepdims (may raise error)

```python
def softmax_v2(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1)
        y = np.exp(x) / np.sum(np.exp(x), axis=1)
        return y
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```
axis=1 means we want to compute the max value row-wise.
However, the result of np.max(x, axis=1) has shape (2,), which doesn’t match the 2D shape of x.

```python
np.max(x, axis=1) = [3.0, 1.5]  # shape: (2,)

[[1.0, 2.0, 3.0],      -    [3.0]  ❌ error
 [0.5, 1.0, 1.5]]           [1.5]

```

### Version 2: Using Transpose

```python
def softmax_v1(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```

We transpose the input so that softmax can be applied column-wise, and then transpose it back.

After transposing:
```python
x.T =
[[1.0, 0.5],
[2.0, 1.0],
[3.0, 1.5]]

Now we compute the maximum values column-wise (axis=0):

np.max(x, axis=0) = [3.0, 1.5]

Subtracting gives:
[[-2.0, -1.0],
[-1.0, -0.5],
[ 0.0, 0.0]]
```
After applying exp and normalization (column-wise), we transpose the result back.

This version works correctly but requires two transpositions (before and after),
which may be less intuitive and slightly less efficient.

### Version 3: With keepdims=True (recommended)

```python
def softmax_v3(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```

We subtract the row-wise max values, but use keepdims=True to maintain shape compatibility.

```python
np.max(x, axis=1, keepdims=True) =
[[3.0],
[1.5]] → shape: (2, 1)

Subtracting from x:

[[1.0, 2.0, 3.0], - [[3.0]]
[0.5, 1.0, 1.5]] [[1.5]]

Result:

[[-2.0, -1.0, 0.0],
[-1.0, -0.5, 0.0]]
```

After applying exp and row-wise normalization, we get the correct softmax output.

This method is clean, safe, and the most recommended way for handling batch softmax.
