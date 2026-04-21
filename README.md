# Assignment #2: Differential Testing with DeepXplore
Reliable and Trustworthy Artificial Intelligence (26 Spring)  
서울시립대학교 일반대학원 통계데이터사이언스학과  
G202558003 김지혜

---
## 프로젝트 구조

```
├── train.py         # ResNet50 두 개를 CIFAR-10으로 학습
├── deepxplore.py    # DeepXplore 핵심 로직 (원본 DeepXplore reference 수정)
├── utils.py         # 유틸리티 함수 (원본 DeepXplore reference 수정)
├── configs.py       # 터미널 색상 설정
├── test.py          # DeepXplore 실행 및 결과 저장
├── requirements.txt
├── results/         # 시각화 결과
├── Problem2.pdf     # Problem 2 에세이 
└── reference/       # 원본 DeepXplore 코드 (참고용, git 미포함)
```

## 환경 설정
```bash
pip install -r requirements.txt
```

# 실행 방법 

## 모델 학습

```bash
python train.py
```

- CIFAR-10 데이터셋을 자동으로 다운로드합니다 (`download=True`)
- ResNet50 두 개를 서로 다른 seed/하이퍼파라미터로 독립 학습합니다
  - model1: seed=42, lr=0.01, weight_decay=5e-4
  - model2: seed=123, lr=0.05, weight_decay=1e-4
- 학습 완료 후 `models/model1.pth`, `models/model2.pth` 생성

## DeepXplore 실행

```bash
python test.py
```

- 두 모델에 대해 differential testing을 수행
- 불일치 입력 탐색 및 neuron coverage를 측정
- `results/disagreements_summary.png`에 시각화 결과를 저장

---

## 실험 결과

| 항목 | 결과 |
|------|------|
| 불일치 입력 수 | 49개 / 100 seeds |
| model1 neuron coverage | 91.69% |
| model2 neuron coverage | 80.85% |
| 평균 neuron coverage | 86.27% |

주요 불일치 패턴:
- model1=deer vs model2=bird: 7회
- model1=deer vs model2=frog: 5회
- model1=automobile vs model2=ship: 4회

---

## DeepXplore 수정 사항

원본(Pei et al., SOSP 2017)은 Keras/TF 1.x 기반이며 ImageNet + VGG16/VGG19/ResNet50 3개 모델을 사용합니다. 
본 과제에서는 아래와 같이 수정하였습니다:

| 항목 | 원본 -> 수정 |

| 프레임워크 | Keras / TF 1.x  ->  PyTorch |
| 데이터셋 | ImageNet (224×224) -> CIFAR-10 (32×32) |
| 모델 구성 | VGG16, VGG19, ResNet50 3개 -> ResNet50 2개 |
| Gradient 계산 | `K.gradients`, `K.function` -> PyTorch autograd |
| Neuron coverage 추적 | Keras intermediate layer model -> PyTorch forward hook |
| 이미지 저장 | `scipy.misc.imsave` -> `matplotlib.pyplot.imsave` |
| Python 버전 | Python 2 (`xrange`) -> Python 3 (`range`) |
| `constraint_light` 스케일 | `1e4` (이미지 범위 초과) -> `1.0` (범위 안정화) |
| Seed 입력 | `./seeds/` 폴더의 ImageNet jpeg -> CIFAR-10 test set |

---

## 참고 문헌

- Pei, K., Cao, Y., Yang, J., & Jana, S. (2017). DeepXplore: Automated Whitebox Testing of Deep Learning Systems. *SOSP 2017*.
- 원본 코드: https://github.com/peikexin9/deepxplore
