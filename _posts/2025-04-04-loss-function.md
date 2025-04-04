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

---

# Cross Entropy 추가 설명 및 예시

---

## Cross Entropy Loss란?

- **확률 분포 간의 차이**를 계산하는 손실 함수  
- 모델이 예측한 확률 분포가 정답에 가까울수록 손실이 작음

---

## 이진 분류 예시

- 정답: \\( y = 1 \\)
- 예측 확률: \\( p = 0.9 \\)

$$
\text{Loss} = - \log(0.9) \approx 0.105
$$

- 잘 맞힌 예측 → 손실 작음

---

- 정답: \\( y = 1 \\)
- 예측 확률: \\( p = 0.1 \\)

$$
\text{Loss} = - \log(0.1) \approx 2.302
$$

- 정답을 거의 0으로 본 예측 → 손실 큼

---

## 다중 클래스 예시

- 정답 \\( y = [1, 0, 0] \\), 예측 확률 \\( \hat{y} = [0.7, 0.2, 0.1] \\)

$$
\text{Loss} = - \log(0.7) \approx 0.357
$$

- 예측이 맞을수록 Loss는 작아짐

---

## 엔트로피가 최대일 때란?

- 예측이 전혀 구분되지 않고 모든 클래스에 균등한 확률 부여

예: \\( \hat{y} = [1/3, 1/3, 1/3] \\)

$$
\text{Loss} = - \log(1/3) \approx 1.0986
$$

> 이때 "Loss가 최대"라는 표현은 **Cross Entropy**가 아니라 **Entropy**가 최대라는 뜻임에 주의

---

## Loss(Cross Entropy)가 진짜 최대가 되는 경우는?

- 정답 클래스 확률이 **0**인 경우

$$
\text{Loss} = - \log(0) = \infty
$$

→ 완전한 오답에 대한 벌점

---

## 정리

| 상황                             | Loss 값         | 해석                          |
|----------------------------------|------------------|-------------------------------|
| 예측 확률 = 1 (정답)            | 0                | 완벽한 예측                  |
| 예측 확률 = 1/n (균등분포)      | \\( \log(n) \\)  | 가장 불확실한 예측 (Entropy 최대) |
| 예측 확률 = 0 (정답)            | 무한대           | 완전히 틀림 (Loss 최대)       |

---
