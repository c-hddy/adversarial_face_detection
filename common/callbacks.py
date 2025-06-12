class EarlyStopping:
    """
    검증 성능이 일정 기간 동안 개선되지 않을 때 학습을 조기에 중단시키는 콜백입니다.

    Args:
        patience (int): 개선이 없을 때 기다릴 에폭 수.
        min_delta (float): 개선으로 간주될 최소 변화량.
                           mode='max'일 때 (current_score - best_score) > min_delta
                           mode='min'일 때 (best_score - current_score) > min_delta
        mode (str): 모니터링할 지표의 방향 ('min' for loss, 'max' for accuracy/ROC-AUC).
        verbose (bool): 조기 종료 카운트 및 종료 메시지를 출력할지 여부.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0001, mode: str = 'max', verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # 모니터링 모드에 따라 best_score 초기값 설정
        if self.mode == 'min':
            self.val_score_sign = 1
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.val_score_sign = -1
            self.best_score = float('-inf')
        else:
            raise ValueError("Error: 'mode' must be 'min' or 'max'.")

    def __call__(self, current_score: float) -> bool:
        """
        현재 에폭의 성능 점수를 바탕으로 조기 종료 여부를 판단합니다.

        Args:
            current_score (float): 현재 에폭의 모니터링 지표 값.

        Returns:
            bool: 조기 종료가 트리거되었으면 True, 아니면 False.
        """
        if self.best_score is None:
            self.best_score = current_score
        # 현재 점수가 이전 최고 점수보다 충분히 개선되지 않았을 때
        elif (current_score * self.val_score_sign) < (self.best_score * self.val_score_sign) + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
        # 현재 점수가 이전 최고 점수보다 충분히 개선되었을 때
        else:
            # 새로운 최고 점수 업데이트 (mode에 따라)
            if (current_score * self.val_score_sign) > (self.best_score * self.val_score_sign):
                 self.best_score = current_score
            self.counter = 0 # 카운터 초기화
        return self.early_stop

