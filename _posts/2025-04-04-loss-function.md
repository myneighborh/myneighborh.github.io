---
title: "Loss Function"
layout: single
math: true
date: 2024-04-04
category: [deeplearning]
header:
  teaser: 
---
<!--more-->

# 손실 함수(Loss Function) 수식 정리

---

## 1. MSE (Mean Squared Error)

회귀(Regression) 문제에 주로 사용됩니다.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- \\( y_i \\): 실제 값  
- \\( \hat{y}_i \\): 예측 값  
- \\( n \\): 샘플 개수

---

## 2. MAE (Mean Absolute Error)

오차의 절댓값 평균입니다.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

---

## 3. Huber Loss

MSE와 MAE의 장점을 결합한 손실 함수입니다.

$$
\text{Huber}(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\\\
\delta \cdot (|y - \hat{y}| - \frac{1}{2} \delta) & \text{otherwise}
\end{cases}
$$

- \\( \delta \\): 민감도 조절 파라미터

---

## 4. Cross Entropy Loss (이진 분류)

이진 분류(Binary Classification)에서 사용됩니다.

$$
\text{Loss} = - \left( y \cdot \log(p) + (1 - y) \cdot \log(1 - p) \right)
$$

- \\( y \in \{0, 1\} \\): 실제 정답  
- \\( p \\): 예측 확률

---

## 5. Cross Entropy Loss (다중 클래스)

다중 클래스(Multi-class classification)에서 사용됩니다.

$$
\text{Loss} = - \sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)
$$

- \\( C \\): 클래스 수  
- \\( y_i \\): 정답 (one-hot)  
- \\( \hat{y}_i \\): 예측 확률 (Softmax 결과)

---

## 6. KL Divergence (Kullback–Leibler)

두 확률 분포 \\( P \\)와 \\( Q \\)의 차이를 측정합니다.

$$
D_{KL}(P \Vert Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)
$$

---

## 7. Cross Entropy와 Entropy의 관계

$$
H(P, Q) = H(P) + D_{KL}(P \Vert Q)
$$

- \\( H(P) = -\sum P(i) \log P(i) \\): 실제 분포의 엔트로피  
- \\( H(P, Q) = -\sum P(i) \log Q(i) \\): 크로스 엔트로피  
- \\( D_{KL} \\): 모델이 틀릴수록 커짐

---

## 8. Softmax 함수 (참고)

분류 문제에서 확률 벡터를 만들기 위해 사용됩니다.

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

- \\( z_i \\): 로짓 (logit) 값  
- \\( \hat{y}_i \\): 클래스 \\( i \\)의 예측 확률

---

## 요약

- 회귀 → MSE, MAE, Huber Loss  
- 분류 → Cross Entropy  
- 분포 비교 → KL Divergence  
- Softmax + Cross Entropy = 분류 문제의 대표 조합
