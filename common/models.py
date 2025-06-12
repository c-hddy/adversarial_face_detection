import torch.nn as nn
import torch # torch.Tensor 타입을 위해 임포트

class LatentClassifier(nn.Module):
    """
    잠재 벡터 이진 분류를 위한 다층 퍼셉트론(MLP) 모델입니다.

    Args:
        input_dim (int): 입력 잠재 벡터의 차원입니다 (예: 9216).
        dropout_rate (float): 드롭아웃 레이어에 적용할 확률입니다.
                              과적합 방지를 위해 사용됩니다.
    """
    def __init__(self, input_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        # 시퀀셜 컨테이너를 사용하여 레이어들을 순차적으로 정의합니다.
        self.fc = nn.Sequential(
            # 첫 번째 완전 연결 레이어: input_dim -> 256 뉴런
            nn.Linear(input_dim, 256),
            # 첫 번째 배치 정규화 레이어: 256 특징
            # 학습 안정성을 높이고 수렴 속도를 빠르게 합니다.
            nn.BatchNorm1d(256),
            # ReLU 활성화 함수: 비선형성을 도입합니다.
            nn.ReLU(),
            # 첫 번째 드롭아웃 레이어: dropout_rate만큼 뉴런을 무작위로 비활성화합니다.
            nn.Dropout(dropout_rate),

            # 두 번째 완전 연결 레이어: 256 -> 64 뉴런
            nn.Linear(256, 64),
            # 두 번째 배치 정규화 레이어: 64 특징
            nn.BatchNorm1d(64),
            # ReLU 활성화 함수
            nn.ReLU(),
            # 두 번째 드롭아웃 레이어
            nn.Dropout(dropout_rate),

            # 최종 출력 레이어: 64 -> 1 뉴런 (이진 분류)
            nn.Linear(64, 1),
            # 시그모이드 활성화 함수: 출력을 0과 1 사이의 확률 값으로 변환합니다.
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델의 순전파를 정의합니다.

        Args:
            x (torch.Tensor): 입력 텐서 (잠재 벡터).

        Returns:
            torch.Tensor: 모델의 예측 출력 (0과 1 사이의 확률).
        """
        return self.fc(x)