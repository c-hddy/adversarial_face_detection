import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import torch # torch.Tensor 타입을 위해 임포트
import joblib # StandardScaler 저장을 위함

def load_latent_dir(dir_path: str, count: int = None) -> np.ndarray:
    """
    지정된 디렉토리에서 .npy 파일들을 로드하고 잠재 벡터를 스택합니다.
    각 .npy 파일이 2차원 배열인 경우, 평균(axis=0)을 취하여 1차원으로 만듭니다.

    Args:
        dir_path (str): .npy 파일이 있는 디렉토리 경로.
        count (int, optional): 로드할 파일의 최대 개수. None이면 모든 파일을 로드합니다.
                               기본값은 None.

    Returns:
        np.ndarray: 스택된 잠재 벡터들의 NumPy 배열.

    Raises:
        FileNotFoundError: 디렉토리가 존재하지 않거나 지정된 개수만큼 .npy 파일을 찾을 수 없을 때.
        ValueError: 로드된 잠재 벡터들의 차원이 일관되지 않을 때.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Error: Directory not found at {dir_path}. Please check the path.")
    
    # .npy 파일들을 정렬하여 가져옵니다.
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".npy")])
    if count is not None:
        files = files[:count]
    
    if not files:
        raise FileNotFoundError(f"Error: No .npy files found in {dir_path} with count limit {count} or directory is empty.")

    latents = []
    for f in files:
        file_path = os.path.join(dir_path, f)
        try:
            latent_data = np.load(file_path)
            # 잠재 벡터가 2차원 배열일 경우, 평균을 취하여 1차원으로 변환
            if latent_data.ndim == 2:
                latents.append(latent_data.mean(axis=0))
            else:
                latents.append(latent_data)
        except Exception as e:
            print(f"Warning: Could not load {file_path}. Skipping. Error: {e}")
            continue

    if not latents:
        raise ValueError(f"No valid latent vectors were loaded from {dir_path}.")

    # 모든 잠재 벡터의 차원 일관성 확인
    first_dim = latents[0].shape
    if not all(x.shape == first_dim for x in latents):
        raise ValueError(f"Latent vectors in {dir_path} have inconsistent dimensions. Expected shape: {first_dim}")
    
    return np.stack(latents)

def save_scaler(scaler: StandardScaler, path: str):
    """
    학습된 StandardScaler 객체를 지정된 경로에 저장합니다.

    Args:
        scaler (StandardScaler): 저장할 StandardScaler 객체.
        path (str): StandardScaler를 저장할 파일 경로 (.joblib).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"✅ StandardScaler saved to: {path}")

def load_scaler(path: str) -> StandardScaler:
    """
    지정된 경로에서 StandardScaler 객체를 로드합니다.

    Args:
        path (str): StandardScaler 파일 (.joblib) 경로.

    Returns:
        StandardScaler: 로드된 StandardScaler 객체.

    Raises:
        FileNotFoundError: 지정된 경로에 파일이 존재하지 않을 때.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: StandardScaler file not found at {path}.")
    scaler = joblib.load(path)
    print(f"✅ StandardScaler loaded from: {path}")
    return scaler

def get_input_dim_from_checkpoint(checkpoint: dict) -> int:
    """
    모델 체크포인트에서 입력 차원(input_dim)을 추출합니다.

    Args:
        checkpoint (dict): PyTorch 모델 체크포인트 딕셔너리.

    Returns:
        int: 모델의 입력 차원.
    """
    # 첫 번째 Linear 레이어의 가중치 텐서로부터 입력 차원을 추출
    return checkpoint['model_state_dict']['fc.0.weight'].shape[1]

