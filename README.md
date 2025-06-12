# 잠재 공간 기반 적대적 얼굴 탐지 (Latent Space-Based Adversarial Face Detection)

## 1. 프로젝트 개요

이 프로젝트는 딥러닝 기반의 이미지 생성 기술, 특히 StyleGAN2를 통해 생성된 적대적 얼굴(Adversarial Classifier Evasive Images, ACAI)을 탐지하는 새로운 방법을 제안하고 구현합니다. 기존 이미지 기반 분류기가 속기 쉬운 ACAI의 특성을 고려하여, 이미지를 잠재 공간(Latent Space)으로 역변환(Inversion)한 후 잠재 벡터의 분포 특성을 분석하고, 이를 기반으로 MLP 분류기를 학습시켜 ACAI를 탐지합니다. 최종적으로는 이미지 기반 분류기와 잠재 기반 분류기를 결합한 하이브리드 탐지 구조를 구현하여 탐지 성능을 향상시키는 것을 목표로 합니다.

**주요 목표:**

- RI (Real Images)와 ACAI (Adversarial Classifier Evasive Images) 잠재 벡터의 분포 특성 분석
- RI와 ACAI를 이진 분류하는 MLP 모델 학습
- 이미지 기반 분류기와 잠재 기반 분류기를 활용한 하이브리드 탐지 시스템 구현

## 2. 프로젝트 구조

```
my_adversarial_face_detection/
├── common/
│   ├── __init__.py
│   ├── models.py
│   ├── utils.py
│   └── callbacks.py
├── scripts/
│   ├── train_mlp.py
│   ├── analyze_latents.py
│   └── hybrid_inference.py
├── data/
│   └── latents/
│       ├── RI/
│       └── ACAI/
│   └── images_for_inference/
├── models/
│   ├── mlp_classifier/
│   ├── image_classifier/
│   └── e4e_encoder/
├── results/
│   ├── latent_visualization/
│   └── mlp_training_plots/
├── README.md
└── requirements.txt
```

## 3. 환경 설정

### 3.1. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 3.2. 데이터 준비

- `data/latents/RI/`: 실제 이미지(Real Images)의 잠재 벡터
- `data/latents/ACAI/`: 적대적 분류기 회피 이미지의 잠재 벡터

**주의:** `.npy` 파일은 고용량이므로 GitHub에는 업로드하지 않고 외부 저장소에 보관할 것을 권장합니다.

- `data/images_for_inference/`: 추론용 원본 이미지 (JPG, PNG 등)

### 미리 학습된 모델 저장 위치

- 이미지 분류기 모델: `models/image_classifier/`
- e4e 인코더 모델: `models/e4e_encoder/`

## 4. 스크립트 실행 방법

### 4.1. 잠재 공간 분석 (`analyze_latents.py`)

```bash
python scripts/analyze_latents.py \
    --ri_latent_dir ./data/latents/RI/ \
    --acai_latent_dir ./data/latents/ACAI/ \
    --file_count_per_class 100 \
    --output_plot_dir ./results/latent_visualization/ \
    --seed 42
```

### 4.2. MLP 분류기 훈련 (`train_mlp.py`)

```bash
python scripts/train_mlp.py \
    --real_latent_dir ./data/latents/RI/ \
    --adv_clf_latent_dir ./data/latents/ACAI/ \
    --latent_file_count 300 \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 5e-5 \
    --dropout_rate 0.3 \
    --weight_decay 1e-4 \
    --patience 10 \
    --min_delta 0.001 \
    --output_dir ./models/mlp_classifier/ \
    --wandb_project_name Latent_Classifier \
    --experiment_name mlp_real_advclf_final_run \
    --seed 42
```

### 4.3. 하이브리드 탐지 추론 (`hybrid_inference.py`)

#### 단일 이미지

```bash
python scripts/hybrid_inference.py \
    --image_path ./data/images_for_inference/sample_real_image.jpg \
    --image_classifier_name resnet50 \
    --image_classifier_model_path ./models/image_classifier/resnet50_best_epoch.pt \
    --e4e_encoder_model_path ./models/e4e_encoder/e4e_model.pt \
    --mlp_model_path ./models/mlp_classifier/best_mlp_model_checkpoint.pt \
    --scaler_path ./models/mlp_classifier/scaler.joblib \
    --mlp_dropout_rate 0.3 \
    --device cuda
```

#### 다중 이미지 디렉토리

```bash
python scripts/hybrid_inference.py \
    --image_dir ./data/images_for_inference/ \
    --image_classifier_name resnet50 \
    --image_classifier_model_path ./models/image_classifier/resnet50_best_epoch.pt \
    --e4e_encoder_model_path ./models/e4e_encoder/e4e_model.pt \
    --mlp_model_path ./models/mlp_classifier/best_mlp_model_checkpoint.pt \
    --scaler_path ./models/mlp_classifier/scaler.joblib \
    --mlp_dropout_rate 0.3 \
    --device cuda
```

## 5. 의존성 (`requirements.txt`)

```text
torch>=1.10.0
torchvision
numpy
scikit-learn
matplotlib
tqdm
wandb
joblib
pandas
seaborn
umap-learn
Pillow
```

## 6. Reference

```text
@inproceedings{shamshad2023evading,
  title={Evading Forensic Classifiers With Attribute-Conditioned Adversarial Faces},
  author={Shamshad, Fahad and Srivatsan, Koushik and Nandakumar, Karthik},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16469--16478},
  year={2023}
}
```
