import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models 
import numpy as np
import os
from PIL import Image 
import argparse
import joblib 

# common 모듈 임포트
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from models import LatentClassifier 
from utils import load_latent_dir, get_input_dim_from_checkpoint, load_scaler 

# --- 1. 명령줄 인수 파싱 ---
parser = argparse.ArgumentParser(description='Hybrid Detection Inference for Adversarial Faces')
parser.add_argument('--image_path', type=str, default=None, 
                    help='Path to a single image file for inference.')
parser.add_argument('--image_dir', type=str, default='data/images_for_inference/', # 상대 경로 기본값
                    help='Path to a directory containing multiple image files for inference. Defaults to data/images_for_inference/.')
parser.add_argument('--image_classifier_name', type=str, default='resnet50', 
                    help='Name of the image classifier model used (e.g., resnet50, xception). Defaults to resnet50.')
parser.add_argument('--image_classifier_model_path', type=str, default='models/image_classifier/resnet50_best_epoch.pt', # 상대 경로 기본값
                    help='Path to the saved image classifier model checkpoint (.pt file). Defaults to models/image_classifier/resnet50_best_epoch.pt.')
parser.add_argument('--e4e_encoder_model_path', type=str, default='models/e4e_encoder/e4e_model.pt', # 상대 경로 기본값
                    help='Path to the e4e encoder model. (NOTE: This is a placeholder; real e4e loading requires specific library setup). Defaults to models/e4e_encoder/e4e_model.pt.')
parser.add_argument('--mlp_model_path', type=str, default='models/mlp_classifier/best_mlp_model_checkpoint.pt', # 상대 경로 기본값
                    help='Path to the saved MLP classifier model checkpoint (.pt file). Defaults to models/mlp_classifier/best_mlp_model_checkpoint.pt.')
parser.add_argument('--scaler_path', type=str, default='models/mlp_classifier/scaler.joblib', # 상대 경로 기본값
                    help='Path to the saved StandardScaler .joblib file. Defaults to models/mlp_classifier/scaler.joblib.')
parser.add_argument('--mlp_dropout_rate', type=float, default=0.3, 
                    help='Dropout rate used for the trained MLP model. Defaults to 0.3.')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to use for inference (cuda or cpu).')


# --- 2. 모델 로드 헬퍼 함수 ---

def load_image_classifier(classifier_name: str, model_path: str, device: torch.device):
    """
    미리 학습된 이미지 기반 분류기 모델을 로드합니다.
    """
    if classifier_name == 'resnet50':
        model = models.resnet50(pretrained=False) 
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif classifier_name == 'xception':
        raise NotImplementedError("Xception model loading is not implemented in this example. Please provide your Xception class definition.")
    else:
        raise ValueError(f"Unsupported image classifier: {classifier_name}. Supported: resnet50.")
    
    try:
        # torch.load 시 weights_only=False 추가 (PyTorch 2.6+ 버전 호환성)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False) 
        if 'model' in checkpoint: # 이전 train_forensic_classifier.py 스크립트 형식
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint: # MLP train_mlp.py 스크립트 형식
            model.load_state_dict(checkpoint['model_state_dict'])
        else: # 직접 state_dict만 저장된 경우
            model.load_state_dict(checkpoint)
        
        print(f"✅ Image classifier '{classifier_name}' loaded from: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading image classifier model from {model_path}: {e}")

    model.eval() 
    return model.to(device)

def load_e4e_encoder(e4e_model_path: str, device: torch.device):
    """
    e4e 인코더 모델을 로드합니다. (플레이스홀더)
    실제 구현 시 StyleGAN 및 e4e 라이브러리에 따라 다르게 구현되어야 합니다.
    """
    print(f"Loading e4e encoder from {e4e_model_path} (NOTE: Using a dummy encoder for this example).")
    class DummyE4EEncoder(nn.Module):
        def __init__(self, latent_dim: int = 9216):
            super().__init__()
            self.latent_dim = latent_dim
            # 더미 인코더는 학습이 필요 없으므로 eval() 모드
            self.eval() 

        def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
            """
            더미 인코더의 인코딩 메서드. 실제로는 이미지 텐서를 받아 잠재 벡터를 반환.
            """
            if image_tensor.ndim == 4: # (batch, channel, height, width)
                batch_size = image_tensor.shape[0]
                # 실제 e4e에서는 image_tensor를 네트워크에 통과시켜 잠재 벡터를 얻습니다.
                # 여기서는 단순히 배치 크기만큼의 무작위 텐서를 반환합니다.
                return torch.randn(batch_size, self.latent_dim).to(image_tensor.device)
            else:
                raise ValueError("DummyE4EEncoder expects image_tensor with 4 dimensions (batch, C, H, W).")

    # LatentClassifier의 input_dim (9216)과 맞춰야 함
    return DummyE4EEncoder(latent_dim=9216).to(device)

def load_mlp_classifier_and_scaler(mlp_model_path: str, scaler_path: str, dropout_rate: float, device: torch.device):
    """
    학습된 MLP 분류기 모델과 StandardScaler를 로드합니다.
    """
    # weights_only=False 추가 (PyTorch 2.6+ 버전 호환성)
    checkpoint = torch.load(mlp_model_path, map_location=device, weights_only=False) 
    
    # 모델 입력 차원 및 드롭아웃 비율 추출
    input_dim = get_input_dim_from_checkpoint(checkpoint)
    
    model = LatentClassifier(input_dim=input_dim, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    print(f"✅ MLP classifier loaded from: {mlp_model_path}")

    scaler = load_scaler(scaler_path)
    
    return model, scaler

# --- 3. 이미지 전처리 파이프라인 ---
def get_image_transforms():
    """
    이미지 분류기 입력에 필요한 전처리 변환을 반환합니다.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((224, 224)), # 이미지 분류기 입력 크기 (대부분의 CNN)
        transforms.ToTensor(),
        normalize
    ])

# --- 4. 하이브리드 탐지 파이프라인 ---
def hybrid_detect_face(image_path: str, 
                       image_classifier: nn.Module, 
                       e4e_encoder: nn.Module, 
                       mlp_classifier: nn.Module, 
                       latent_scaler: StandardScaler,
                       image_transform: transforms.Compose,
                       device: torch.device):
    """
    단일 이미지에 대해 하이브리드 탐지 파이프라인을 실행합니다.

    Args:
        image_path (str): 추론할 이미지 파일의 경로.
        image_classifier (nn.Module): 이미지 기반 감식 분류기 모델.
        e4e_encoder (nn.Module): e4e 잠재 벡터 인코더 모델.
        mlp_classifier (nn.Module): MLP 잠재 벡터 분류기 모델.
        latent_scaler (StandardScaler): 잠재 벡터 정규화를 위한 StandardScaler 객체.
        image_transform (transforms.Compose): 이미지 전처리 변환.
        device (torch.device): 추론에 사용할 장치.

    Returns:
        tuple: (final_pred_label, final_real_prob)
            final_pred_label (int): 최종 예측 레이블 (1: RI, 0: ACAI).
            final_real_prob (float): 최종 Real Image에 대한 확률.
    """
    print(f"\n--- Processing Image: {os.path.basename(image_path)} ---")
    
    # 1. 이미지 로드 및 전처리
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_transform(image).unsqueeze(0).to(device) # 배치 차원 추가
    except Exception as e:
        print(f"Error loading or transforming image {image_path}: {e}")
        return None, None

    # 2. 1단계 탐지: 이미지 기반 분류기
    # BCEWithLogitsLoss로 학습된 모델은 Logits 출력을 내므로 sigmoid를 적용
    with torch.no_grad():
        image_classifier_output_logits = image_classifier(image_tensor)
        # 이미지 분류기는 보통 1개 뉴런 출력 (로짓). sigmoid 적용하여 확률로 변환.
        image_classifier_prob_ri = torch.sigmoid(image_classifier_output_logits).item() 
    
    # 이미지 분류기의 판단 (0: ACAI, 1: RI)
    image_pred_label = 1 if image_classifier_prob_ri >= 0.5 else 0

    print(f"  Stage 1 (Image Classifier) Prediction: {'RI' if image_pred_label == 1 else 'ACAI'} (RI Prob: {image_classifier_prob_ri:.4f})")

    mlp_classifier_prob_ri = 0.5 # MLP를 사용하지 않을 경우 초기값 (불확실)

    # 3. 조건부 잠재 검증
    # 이미지 분류기가 'RI'로 판단한 경우에만 MLP 분류기를 사용합니다.
    if image_pred_label == 1: 
        print("  Image classifier predicted RI. Performing latent verification...")
        
        # a. e4e 인코더로 잠재 벡터 추출
        # e4e_encoder.encode()는 (batch_size, latent_dim) 텐서를 반환해야 함
        try:
            latent_vector_tensor = e4e_encoder.encode(image_tensor).cpu().numpy()
            latent_vector_scaled = latent_scaler.transform(latent_vector_tensor.reshape(1, -1))
            latent_vector_scaled_tensor = torch.tensor(latent_vector_scaled.astype(np.float32)).to(device)
        except Exception as e:
            print(f"  Error extracting or scaling latent vector for {image_path}: {e}")
            return None, None

        # c. MLP 분류기로 판단
        with torch.no_grad():
            mlp_classifier_output = mlp_classifier(latent_vector_scaled_tensor)
            mlp_classifier_prob_ri = mlp_classifier_output.item() # Sigmoid 출력 (0-1)

        print(f"  Stage 2 (MLP Classifier) Prediction: {'RI' if mlp_classifier_prob_ri >= 0.5 else 'ACAI'} (RI Prob: {mlp_classifier_prob_ri:.4f})")
    else:
        print("  Image classifier predicted ACAI. Skipping latent verification.")
        # 이미지 분류기가 ACAI로 판단한 경우, MLP 확률은 0.5로 두어 평균 계산에 영향 주지 않음
        # (제안서의 "soft fusion by averaging the output probabilities"에 따라 단순히 평균을 내기 위함)

    # 4. 최종 결정 (소프트 퓨전 - 평균)
    # 이미지 분류기 출력 확률 (RI에 대한 확률)과 MLP 분류기 출력 확률 (RI에 대한 확률) 평균
    final_real_prob = (image_classifier_prob_ri + mlp_classifier_prob_ri) / 2
    final_pred_label = 1 if final_real_prob >= 0.5 else 0

    print(f"  Final Combined Probability (RI): {final_real_prob:.4f}")
    print(f"  Final Detection Result: {'RI (Real)' if final_pred_label == 1 else 'ACAI (Adversarial)'}")
    
    return final_pred_label, final_real_prob

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading models and scaler...")
    image_classifier = load_image_classifier(args.image_classifier_name, args.image_classifier_model_path, device)
    e4e_encoder = load_e4e_encoder(args.e4e_encoder_model_path, device) # 플레이스홀더
    mlp_classifier, latent_scaler = load_mlp_classifier_and_scaler(
        args.mlp_model_path, args.scaler_path, args.mlp_dropout_rate, device
    )
    
    image_transform = get_image_transforms()

    if args.image_path: # 단일 이미지 추론
        hybrid_detect_face(args.image_path, image_classifier, e4e_encoder, mlp_classifier, latent_scaler, image_transform, device)
    elif args.image_dir: # 디렉토리 내 여러 이미지 추론
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(supported_formats)]
        if not image_files:
            print(f"No supported image files found in {args.image_dir}.")
        
        for img_file in sorted(image_files):
            img_path = os.path.join(args.image_dir, img_file)
            hybrid_detect_face(img_path, image_classifier, e4e_encoder, mlp_classifier, latent_scaler, image_transform, device)
    else:
        print("Please provide either --image_path or --image_dir for inference.")

    print("\nHybrid inference complete.")

