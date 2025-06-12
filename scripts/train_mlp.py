import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse
import wandb
# joblib는 common/utils.py에서 사용되므로 여기서는 직접 필요하지 않음

# common 모듈 임포트
import sys
# 현재 스크립트의 상위 폴더 (scripts/)의 상위 폴더 (프로젝트 루트)에 common 폴더가 있다고 가정
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from models import LatentClassifier 
from utils import load_latent_dir, save_scaler, get_input_dim_from_checkpoint
from callbacks import EarlyStopping

# --- 1. 명령줄 인수 파싱 ---
parser = argparse.ArgumentParser(description='Latent Space MLP Classifier Training with Regularization')
parser.add_argument('--real_latent_dir', type=str, default='data/latents/RI/', 
                    help='Path to the directory containing real latent vectors (.npy files). Defaults to data/latents/RI/.')
parser.add_argument('--adv_clf_latent_dir', type=str, default='data/latents/ACAI/', 
                    help='Path to the directory containing adversarial classifier latent vectors (.npy files). Defaults to data/latents/ACAI/.')
parser.add_argument('--latent_file_count', type=int, default=300, 
                    help='Number of latent vector files to load from each directory for training. Defaults to 300 (per class).')
parser.add_argument('--batch_size', type=int, default=16, 
                    help='Batch size for training. Defaults to 16.')
parser.add_argument('--epochs', type=int, default=50, 
                    help='Number of epochs to train the classifier. Defaults to 50.')
parser.add_argument('--learning_rate', type=float, default=5e-5, 
                    help='Learning rate for the Adam optimizer. Defaults to 5e-5.')
parser.add_argument('--dropout_rate', type=float, default=0.3, 
                    help='Dropout rate for MLP layers. Defaults to 0.3.')
parser.add_argument('--weight_decay', type=float, default=1e-4, 
                    help='Weight decay (L2 regularization) for Adam optimizer. Defaults to 1e-4.')
parser.add_argument('--patience', type=int, default=10, 
                    help='Number of epochs with no improvement after which training will be stopped (Early Stopping). Defaults to 10.')
parser.add_argument('--min_delta', type=float, default=0.001, 
                    help='Minimum change in the monitored quantity to qualify as an improvement (Early Stopping). Defaults to 0.001.')
parser.add_argument('--output_dir', type=str, default='models/mlp_classifier/', 
                    help='Path to save the output model, scaler, and plots. Defaults to models/mlp_classifier/.')
parser.add_argument('--wandb_project_name', type=str, default='Latent_Classifier', 
                    help='Name of wandb project to save the logs.')
parser.add_argument('--experiment_name', type=str, default='mlp_real_advclf_final_run', 
                    help='Name of wandb experiment to save the logs. Must be unique for new runs.')
parser.add_argument('--resume_training', action='store_true', 
                    help='Flag to resume training from checkpoint if exists.')
parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed for reproducibility. Defaults to 42.')

# --- 2. 평가 함수 (평가 지표 포함) ---
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device, criterion: nn.Module):
    """
    모델의 성능을 평가하고 다양한 지표를 반환합니다.

    Args:
        model (nn.Module): 평가할 PyTorch 모델.
        data_loader (DataLoader): 평가에 사용할 데이터 로더.
        device (torch.device): 모델과 데이터를 로드할 장치 (CPU 또는 CUDA).
        criterion (nn.Module): 손실 함수.

    Returns:
        tuple: (accuracy, avg_loss, precision, recall, f1, roc_auc) 튜플.
    """
    model.eval() # 모델을 평가 모드로 설정
    all_preds = []
    all_targets = []
    all_probs = [] # ROC-AUC를 위한 확률값 저장
    total_loss = 0
    
    with torch.no_grad(): # 기울기 계산 비활성화
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X) # Sigmoid 활성화 후의 확률값
            loss = criterion(preds, batch_y)
            total_loss += loss.item()
            
            all_preds.extend((preds > 0.5).float().cpu().numpy()) # 이진 예측값 (0 또는 1)
            all_targets.extend(batch_y.cpu().numpy()) # 실제 레이블
            all_probs.extend(preds.cpu().numpy()) # 확률값 (0.0 ~ 1.0)
    
    avg_loss = total_loss / len(data_loader)
    
    # NumPy 배열로 변환 및 평탄화
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    all_probs = np.array(all_probs).flatten()

    accuracy = np.sum(all_preds == all_targets) / len(all_targets)
    
    # 이진 분류 지표 계산 (zero_division=0 설정으로 분모 0 오류 방지)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # ROC-AUC 계산 (타겟 클래스가 1개인 경우 오류 방지)
    if len(np.unique(all_targets)) > 1:
        roc_auc = roc_auc_score(all_targets, all_probs)
    else:
        roc_auc = np.nan # 계산 불가능하면 NaN 처리

    return accuracy, avg_loss, precision, recall, f1, roc_auc

# --- 3. 데이터 로드 및 전처리 ---
def prepare_data(args: argparse.Namespace):
    """
    잠재 벡터 데이터를 로드하고 훈련/검증 세트로 분할합니다.

    Args:
        args (argparse.Namespace): 명령줄 인수 객체.

    Returns:
        tuple: (train_set_raw, val_set_raw, input_dim) 튜플.
               - train_set_raw (TensorDataset): 원본 훈련 데이터셋 (정규화 전).
               - val_set_raw (TensorDataset): 원본 검증 데이터셋 (정규화 전).
               - input_dim (int): 잠재 벡터의 차원.
    """
    print(f"Loading real latents from: {args.real_latent_dir} (count: {args.latent_file_count})")
    real_latents = load_latent_dir(args.real_latent_dir, count=args.latent_file_count)
    print(f"Loading adversarial latents from: {args.adv_clf_latent_dir} (count: {args.latent_file_count})")
    adv_clf_latents = load_latent_dir(args.adv_clf_latent_dir, count=args.latent_file_count)

    real_flat = real_latents.reshape(len(real_latents), -1)
    adv_flat = adv_clf_latents.reshape(len(adv_clf_latents), -1)

    X_combined = np.vstack([real_flat, adv_flat])
    y_combined = np.array([1]*len(real_flat) + [0]*len(adv_flat))

    X_tensor_raw = torch.tensor(X_combined.astype(np.float32))
    y_tensor = torch.tensor(y_combined.astype(np.float32)).unsqueeze(1)

    dataset = TensorDataset(X_tensor_raw, y_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    return train_set, val_set, X_combined.shape[1]

# --- 4. 훈련 함수 ---
def train_mlp_classifier(model: nn.Module, 
                         train_set_raw: TensorDataset, 
                         val_set_raw: TensorDataset, 
                         device: torch.device, 
                         args: argparse.Namespace):
    """
    MLP 분류기 모델을 훈련합니다.

    Args:
        model (nn.Module): 훈련할 PyTorch 모델.
        train_set_raw (TensorDataset): 원본 훈련 데이터셋 (정규화 전).
        val_set_raw (TensorDataset): 원본 검증 데이터셋 (정규화 전).
        device (torch.device): 모델과 데이터를 로드할 장치 (CPU 또는 CUDA).
        args (argparse.Namespace): 명령줄 인수 객체.

    Returns:
        tuple: (train_losses, val_accuracies, val_losses, val_precisions, 
                val_recalls, val_f1_scores, val_roc_aucs, final_best_accuracy) 튜플.
    """
    criterion = nn.BCELoss() # 이진 교차 엔트로피 손실 함수
    # Adam 옵티마이저에 학습률 및 가중치 감소(L2 정규화) 적용
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    start_epoch = 0
    best_val_accuracy = 0.0 # 조기 종료 및 최고 모델 저장을 위한 기준 지표

    # 체크포인트 및 Scaler 저장 경로 정의
    checkpoint_path = os.path.join(args.output_dir, 'best_mlp_model_checkpoint.pt')
    scaler_save_path = os.path.join(args.output_dir, 'scaler.joblib')

    scaler = StandardScaler() # StandardScaler 인스턴스 생성
    # 조기 종료 콜백 설정 (검증 정확도 기준으로 최대화)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode='max', verbose=True)

    # 훈련 재개 시 체크포인트 로드
    if args.resume_training and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        # weights_only=False 추가 (PyTorch 2.6+ 버전 호환성)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) 
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint['best_val_accuracy']
        # 저장된 scaler 파라미터 로드
        scaler.mean_ = checkpoint['scaler_mean']
        scaler.scale_ = checkpoint['scaler_scale']
        # StandardScaler의 n_features_in_ 속성 설정 (버전 호환성을 위해 필요할 수 있음)
        if 'n_features_in_' in checkpoint:
            scaler.n_features_in_ = checkpoint['n_features_in_']
        print(f"Resuming training from epoch {start_epoch}, best validation accuracy: {best_val_accuracy:.4f}")
    elif args.resume_training and not os.path.exists(checkpoint_path):
        print(f"⚠️ Warning: --resume_training is True but no checkpoint found at {checkpoint_path}. Starting training from scratch.")
    
    # 훈련 데이터셋으로 StandardScaler 학습 (fit)
    train_data_raw = train_set_raw.dataset.tensors[0][train_set_raw.indices].cpu().numpy()
    if not args.resume_training or not os.path.exists(checkpoint_path):
        print("Fitting StandardScaler on training data...")
        scaler.fit(train_data_raw)
        save_scaler(scaler, scaler_save_path) # 학습된 scaler 저장
    else:
        print(f"Using pre-loaded StandardScaler for transformation.")

    # 훈련 및 검증 데이터에 StandardScaler 적용 (transform)
    # TensorDataset에서 데이터 추출 시 .cpu().numpy()로 변환해야 StandardScaler에 사용 가능
    val_data_raw = val_set_raw.dataset.tensors[0][val_set_raw.indices].cpu().numpy()
    
    train_X_scaled = torch.tensor(scaler.transform(train_data_raw).astype(np.float32))
    train_y = train_set_raw.dataset.tensors[1][train_set_raw.indices]

    val_X_scaled = torch.tensor(scaler.transform(val_data_raw).astype(np.float32))
    val_y = val_set_raw.dataset.tensors[1][val_set_raw.indices]
    
    # 정규화된 데이터로 새로운 TensorDataset 및 DataLoader 생성
    train_dataset_scaled = TensorDataset(train_X_scaled, train_y)
    val_dataset_scaled = TensorDataset(val_X_scaled, val_y)

    train_loader = DataLoader(train_dataset_scaled, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_scaled, batch_size=args.batch_size)

    # 학습 지표들을 저장할 리스트
    train_losses = []
    val_accuracies = []
    val_losses = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    val_roc_aucs = []

    # 메인 학습 루프
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training Epochs"):
        model.train() # 모델을 훈련 모드로 설정
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device) # 데이터를 장치로 이동
            optimizer.zero_grad() # 이전 기울기 초기화
            outputs = model(batch_X) # 순전파
            loss = criterion(outputs, batch_y) # 손실 계산
            loss.backward() # 역전파
            optimizer.step() # 가중치 업데이트
            total_loss += loss.item() # 배치 손실 누적
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 검증 단계 (모든 지표 계산 및 기록)
        val_accuracy, avg_val_loss, val_precision, val_recall, val_f1, val_roc_auc = \
            evaluate_model(model, val_loader, device, criterion)
        
        val_accuracies.append(val_accuracy)
        val_losses.append(avg_val_loss)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)
        val_roc_aucs.append(val_roc_auc)
        
        # 콘솔에 학습 진행 상황 출력
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}")

        # WandB에 지표 로깅
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": avg_val_loss,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_score": val_f1,
            "val_roc_auc": val_roc_auc
        })

        # 조기 종료 조건 확인
        if early_stopping(val_accuracy):
            print(f"Early stopping triggered at epoch {epoch+1}. No improvement in validation accuracy for {args.patience} consecutive epochs.")
            break

        # 최고 성능 모델 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"Saving best model at epoch {epoch+1} with validation accuracy: {best_val_accuracy:.4f}")
            # 모델 상태, 옵티마이저 상태, 최고 정확도, Scaler 파라미터를 체크포인트에 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_,
                'n_features_in_': scaler.n_features_in_ # StandardScaler 로드 시 필요할 수 있는 속성
            }, checkpoint_path)
    
    # 최종 결과 반환
    return train_losses, val_accuracies, val_losses, val_precisions, \
           val_recalls, val_f1_scores, val_roc_aucs, best_val_accuracy

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    args = parser.parse_args()

    # 재현성을 위한 랜덤 시드 고정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 결과 저장 디렉토리 생성
    # models/mlp_classifier/는 output_dir 인자로 지정
    # plot 저장 폴더는 별도로 results/mlp_training_plots/ 로 지정
    plot_output_dir = os.path.join('results', 'mlp_training_plots')
    os.makedirs(plot_output_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True) # 모델/스케일러 저장 폴더

    # WandB 초기화
    wandb.init(
        project=args.wandb_project_name,
        name=args.experiment_name,
        config=vars(args), 
        id=args.experiment_name, 
        resume="allow" 
    )

    print("Preparing data...")
    train_set_raw, val_set_raw, input_dim = prepare_data(args)
    print(f"Data prepared. Input dimension: {input_dim}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LatentClassifier(input_dim=input_dim, dropout_rate=args.dropout_rate).to(device)
    print(f"Model '{model.__class__.__name__}' initialized and moved to {device}.")
    
    print("Starting training...")
    train_losses, val_accuracies, val_losses, val_precisions, val_recalls, val_f1_scores, val_roc_aucs, final_best_accuracy = \
        train_mlp_classifier(model, train_set_raw, val_set_raw, device, args)

    print(f"\nTraining finished. Best validation accuracy: {final_best_accuracy:.4f}")

    # Matplotlib을 이용한 학습 지표 플롯 저장
    plt_path = os.path.join(plot_output_dir, f"{args.experiment_name}_metrics_plot.png")
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', label='Validation Accuracy')
    plt.title(f"Validation Accuracy ({args.experiment_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', color='red', label='Validation Loss')
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='x', color='blue', linestyle='--', label='Train Loss')
    plt.title(f"Train & Validation Loss ({args.experiment_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plt_path)
    print(f"✅ Metrics plot saved to: {plt_path}")

    # WandB 세션 종료
    wandb.finish()

