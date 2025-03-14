# xBD Road Semantic Segmentation

이 저장소는 **xBD** 데이터를 활용하여 도로(road)의 semantic segmentation을 수행하는 딥러닝 모델 구현 코드를 포함합니다.  
본 프로젝트는 데이터 전처리, 모델 학습, 평가 및 추론까지 전체 파이프라인을 제공합니다.

## 목차

- [xBD Road Semantic Segmentation](#xbd-road-semantic-segmentation)
  - [목차](#목차)
  - [프로젝트 개요](#프로젝트-개요)
  - [특징](#특징)
  - [요구사항](#요구사항)
  - [설치 방법](#설치-방법)
  - [데이터셋](#데이터셋)
  - [사용 방법](#사용-방법)
    - [Continue...](#continue)
  - [기여 방법](#기여-방법)
  - [라이선스](#라이선스)
  - [참고 자료](#참고-자료)

## 프로젝트 개요

이 프로젝트는 xBD 데이터셋을 활용하여 도로 영역을 정확하게 분할하는 semantic segmentation 모델을 구현합니다.  
모델은 딥러닝 기반 네트워크(Pytorch/TensorFlow 등)를 사용하며, 다양한 전처리 및 데이터 증강 기법을 적용하여 학습 성능을 향상시키고자 합니다.

## 특징

- **End-to-End Pipeline:** 데이터 전처리부터 모델 학습, 평가, 추론까지 전체 워크플로우 제공
- **모듈화 설계:** 각 모듈(데이터 로더, 모델, 학습 스크립트 등)이 독립적으로 구성되어 확장이 용이함
- **사용자 친화적 구성:** 구성 파일(config.yaml 등)을 통한 하이퍼파라미터 및 경로 설정
- **GPU 지원:** CUDA 지원을 통해 빠른 학습 환경 제공

## 요구사항

- Python 3.8 이상
- PyTorch (또는 사용 중인 딥러닝 프레임워크)
- CUDA (GPU 사용 시)
- 기타 Python 라이브러리: numpy, opencv-python, albumentations, matplotlib 등  
  *(자세한 내용은 `requirements.txt` 참조)*

## 설치 방법

1. 저장소 클론:
   ```bash
   git clone https://github.com/seunghyeokleeme/xBD_road_segmentation.git
   cd xBD_road_segmentation
   ```

2. 가상환경 생성 및 활성화 (선택 사항):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

3. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 데이터셋

xBD 데이터셋을 사용합니다. 데이터셋은 다음과 같이 구성되어야 합니다:

**다만, 해당 프로젝트는 building이 아닌 road를 이미지 분할을 하기 때문에 직접 레이블링이 필요합니다.**

```
data/
├── train/
│   ├── images/
│   └── targets/
├── val/
│   ├── images/
│   └── targets/
└── test/
    └── images/
```

- **다운로드:** 
  1. xBD 데이터셋은 [xBD 공식 페이지](https://xview2.org)에서 다운로드하여 직접 road를 labeling을 수행합니다.
  2. 혹은 해당 [구글드라이브 링크](https://drive.google.com/drive/folders/1y2wBg3ledu3A5C1I6-xiWdcGA0SyFrs4)에 들어가서 다운받으세요.(추천) 해당 파일들은 road labeling을 완성한 상태입니다.


## 사용 방법

### Continue...

## 기여 방법

1. 저장소를 Fork 합니다.
2. 새로운 브랜치를 생성 (`git checkout -b feature/YourFeature`).
3. 코드를 수정 및 개선합니다.
4. 변경사항을 커밋 (`git commit -m 'Add some feature'`).
5. 원격 저장소에 Push (`git push origin feature/YourFeature`).
6. Pull Request를 생성합니다.

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

## 참고 자료

- [xBD 공식 페이지](https://xview2.org)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Unet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
