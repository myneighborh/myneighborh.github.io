---
title: "Activation Function"
layout: single
date: 2024-03-31
category: [deeplearning]
header:
  teaser: https://github.com/user-attachments/assets/94ab4e96-a7c0-4408-bba1-56fb254cc19e
---
<!--more-->

## Activation Function

뉴런(노드) 출력값을 비선형 함수에 통과시켜주는 함수.

**왜 필요한가?**
- Activation Function이 없다면, 아무리 여러 층을 쌓아도 전체는 선형 함수로 수렴됨
- 복잡한 문제를 해결하기 위해 모델에 '비선형성'을 부여해야 함

---

## 1. Sigmoid

![Sigmoid](https://github.com/user-attachments/assets/67fa2cdb-984d-4f10-a7cb-c95a352484cf)  
<img width="162" alt="Sigmoid Detail" src="https://github.com/user-attachments/assets/d4f1393b-35b3-49d0-9f65-c563a7b1f611" />

**특징**  
- 출력 범위: 0 ~ 1  
- 확률처럼 해석 가능하여 이진 분류에 자주 사용됨

**의문**  
왜 Sigmoid는 잘 안 쓰이게 되었을까?

**문제점**  
Gradient Vanishing 현상이 발생  
→ 입력이 크거나 작을수록 기울기(미분값)가 0에 가까워지고, 역전파 시 앞쪽 레이어로 전달되지 않음

---

## 2. Tanh

![Tanh](https://github.com/user-attachments/assets/4f42b233-edb4-43c1-a643-74336b0bd8fa)  
<img width="193" alt="Tanh Detail" src="https://github.com/user-attachments/assets/61c33b67-d3f4-478c-ba8f-d47af87b6353" />

**특징**  
- 출력 범위: -1 ~ 1  
- Sigmoid보다 중심이 0이라 나은 성질을 가짐

**문제점**  
여전히 Gradient Vanishing이 발생

---

## 3. ReLU

![ReLU](https://github.com/user-attachments/assets/0354288e-6781-4b15-af36-14b626366d55)  
<img width="176" alt="ReLU Detail" src="https://github.com/user-attachments/assets/22282230-934e-4685-b949-1f1e7fb2ea3c" />

**특징**  
- x > 0이면 그대로 출력, x <= 0이면 0 출력  
- 간단하고 계산이 빠름  
- 기울기 소실 문제 없음

**의문**  
왜 음수는 0으로 처리할까?

→ 불필요한 정보 차단, 계산량 감소, 희소성 증가로 일반화에 도움됨

**문제점**  
Dying ReLU 현상: 음수 입력이 지속되면 뉴런이 죽고 다시 활성화되지 않을 수 있음

---

## 4. Leaky ReLU / PReLU

![Leaky ReLU](https://github.com/user-attachments/assets/ad0cf20e-845f-4019-8ce1-35f536e8673e)  
<img width="206" alt="PReLU Detail" src="https://github.com/user-attachments/assets/bad012b5-1841-4ce2-98b5-a8aa7736d970" />

**특징**  
- Leaky ReLU: 음수 입력도 0.01 등의 작은 기울기로 통과  
- PReLU: 음수 기울기 계수를 학습을 통해 최적화함

**의문**  
왜 음수를 완전히 0으로 막지 않을까?

→ 완전 차단 시 뉴런이 죽고 gradient 흐름이 끊기기 때문

---

## 5. ELU

![ELU](https://github.com/user-attachments/assets/7c49bceb-5bfd-4945-861e-b54777f6687d)  
<img width="273" alt="ELU Detail" src="https://github.com/user-attachments/assets/760a26d6-407f-4fdf-ad0b-fd5b7fd46a6f" />

**특징**  
- 음수 영역도 부드럽게 처리  
- 중심이 0에 가까워 안정적인 학습 가능  
- Dying ReLU 문제 완화

**문제점**  
ReLU보다 계산 복잡도 높고 처리 속도가 느림

---

## 6. Swish

![Swish](https://github.com/user-attachments/assets/605d3b35-7164-41ba-b439-eec2c4ebbc0c)  
<img width="198" alt="Swish Detail" src="https://github.com/user-attachments/assets/3ac99f8b-8286-4426-8a92-d0a3905019ed" />

**특징**  
- 함수 형태: x * sigmoid(x)  
- 부드럽고 미분 가능  
- Gradient 흐름이 자연스럽게 이어짐

**의문**  
Sigmoid를 곱하면 다시 기울기 소실이 생기지 않을까?

→ 일부 구간에서는 약해질 수 있으나, x와 곱해지므로 완전 소실은 피함

---

## 7. GELU

![GELU](https://github.com/user-attachments/assets/36f7aecc-7007-45f8-9cc3-5af132737d07)  
<img width="411" alt="GELU Detail" src="https://github.com/user-attachments/assets/6af0d5d2-edc0-4a96-8a75-58cf47603f40" />

**특징**  
- 입력값에 대해 '의미 있을 확률 × 입력값' 구조  
- 확률 기반 부드러운 스위치  
- 중심이 0, 안정적인 학습 가능  
- Transformer 계열에서 자주 사용됨

**의문**  
유의미한 정도는 어떻게 판단하는가?

→ 정규분포의 누적분포함수(CDF)를 통해 확률 계산

---

## 8. GEGLU / SwiGLU

![GEGLU](https://github.com/user-attachments/assets/8ef1f987-280c-4578-aa00-95bb7dd3fd5d)

**특징**  
- 구조: Linear1(x) × GELU(Linear2(x)) 또는 Swish(Linear2(x))  
- 한쪽은 정보 전달, 한쪽은 게이트(필터) 역할  
- GPT-4, PaLM, GLaM 등 최신 모델에서 사용

**의문**  
GPT 등 최신 모델은 어떤 활성화 함수를 사용하는가?

→ GPT-2/3는 GELU, GPT-4는 GEGLU 또는 SwiGLU로 추정됨

---

## 최종 비교표

![표](https://github.com/user-attachments/assets/94ab4e96-a7c0-4408-bba1-56fb254cc19e)

| 함수 | 출력 범위 | 중심 | 기울기 소실 | 특징 |
|------|------------|--------|---------------|--------|
| Sigmoid | 0 ~ 1 | 0.5 | 있음 | 확률처럼 해석 가능 |
| Tanh | -1 ~ 1 | 0 | 있음 | 중심 0, 안정적 |
| ReLU | 0 ~ ∞ | >0 | 없음 | 빠름, 뉴런 죽음 가능 |
| Leaky ReLU | -∞ ~ ∞ | 0 | 없음 | 죽은 뉴런 방지 |
| ELU | -α ~ ∞ | 0 | 적음 | 부드럽고 안정적 |
| Swish | -∞ ~ ∞ | ~0 | 약간 있음 | 부드럽고 효율적 |
| GELU | -∞ ~ ∞ | 0 | 거의 없음 | 확률 기반, 최신 모델 사용 |
| GEGLU / SwiGLU | -∞ ~ ∞ | 0 | 없음 | 게이팅 구조, 최신 트렌드 
