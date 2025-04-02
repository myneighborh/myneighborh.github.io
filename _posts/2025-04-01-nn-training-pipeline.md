---
title: "A Structured Overview of the Neural Network Training Pipeline"
layout: single
use_math: true
date: 2024-04-01
category: [deeplearning]
header:
  teaser: https://github.com/user-attachments/assets/19e141f0-cce3-47a3-8420-fb8d226cb223
---
<img src="https://github.com/user-attachments/assets/19e141f0-cce3-47a3-8420-fb8d226cb223" width="300"/>

---

## 1. 초기화: 무작위 가중치로 시작

- 모델은 처음에 아무것도 학습하지 않았으므로, 가중치 \( W \)와 편향 \( b \)를 무작위로 설정함  
- 초기 상태의 선형 변환 결과는 **의미 없는 예측값**을 출력함

---

## 2. 순전파 (Forward Propagation)

### 2.1 Affine Transformation + Activation Function

![image](https://github.com/user-attachments/assets/a63ce093-f866-46a3-a9c7-0576f1a7366d)

- 입력(x)에 대해 현재 가중치 (W)와 편향 (b)을 이용해 Affine 변환을 수행하고,  
  그 결과를 Activation 함수에 통과시켜 예측값을 계산함

---

## 3. 손실 함수 계산 (Loss Function)

- 실제값과 예측값의 차이를 계산  
- 예시: Mean Squared Error, Cross Entropy 등
  
![image](https://github.com/user-attachments/assets/f80a4f19-47d2-4f96-8fa2-8c06d3b3f984)

- 손실값이 클수록, 현재의 가중치가 정답과 멀다는 뜻

---

## 4. 역전파 (Backpropagation)와 가중치 업데이트

### 4.1 경사 하강법 (Gradient Descent)

- 손실을 줄이기 위해 가중치에 대한 기울기(미분)를 계산함  
- 학습률 를 사용하여 가중치를 다음과 같이 조정:

![image](https://github.com/user-attachments/assets/a99df441-6e6c-43c6-91ac-ffdad6bf74be)


- 이 과정이 **역전파(Backpropagation)**으로 불림

---

## 5. 반복 학습: 업데이트 후 재예측

- 조정된 W, b를 사용하여 다시 예측을 수행  
- 이 과정을 반복하면서 **모델의 예측 정확도가 점점 향상됨**

---



