#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, brier_score_loss, log_loss
import time
import os
import optuna
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    """
    Optuna 최적화를 위한 목적 함수
    
    Parameters:
    -----------
    trial : optuna.trial.Trial
        Optuna trial 객체
    X_train, y_train : 학습 데이터셋
    X_val, y_val : 검증 데이터셋
    scale_pos_weight : 클래스 불균형 보정 가중치
    
    Returns:
    --------
    float : 검증 데이터셋의 ROC AUC 점수
    """
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'random_state': 42,
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # 검증 데이터에 대한 예측
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # ROC AUC 점수 반환
    return roc_auc_score(y_val, y_pred_proba)

def main():
    """
    Optuna와 XGBoost를 사용하여 산불 예측 모델 훈련
    """
    start_time = time.time()
    print("=== 산불 예측 모델 학습 시작 (Optuna 하이퍼파라미터 튜닝) ===")
    
    # 타임스탬프 생성 (모델 버전 관리용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 결과 디렉토리 생성
    model_dir = "../../outputs/models"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "plots"), exist_ok=True)
    
    # 데이터 로드
    input_file = "../../outputs/data/weather_data_with_wind.csv"
    print(f"데이터 로드 중: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['acq_date'])
    print(f"로드 완료: {len(df)}행, {len(df.columns)}열")
    
    # 기본 탐색적 분석
    print("\n=== 데이터 탐색 ===")
    print(f"컬럼 목록: {list(df.columns)}")
    print(f"기간: {df.acq_date.min()} ~ {df.acq_date.max()}")
    print(f"산불 발생 비율: {df.af_flag.mean()*100:.4f}% ({df.af_flag.sum()}/{len(df)})")
    
    # 피처 선택 (날씨 변수 + 풍속, 10u/10v는 제외)
    features = ['t2m', 'td2m', 'tp', 'wind10m']
    target = 'af_flag'
    
    # 결측치 확인 및 처리
    missing = df[features].isnull().sum()
    if missing.sum() > 0:
        print(f"\n결측치 발견: \n{missing[missing > 0]}")
        df = df.dropna(subset=features)
        print(f"결측치 제거 후 데이터 크기: {len(df)}")
    
    # 학습용/테스트용 분할
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
    print(f"학습 데이터 산불 비율: {y_train.mean()*100:.4f}%")
    print(f"테스트 데이터 산불 비율: {y_test.mean()*100:.4f}%")
    
    # 클래스 불균형 처리
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"불균형 보정 가중치(scale_pos_weight): {scale_pos_weight:.2f}")
    
    # 학습 데이터를 다시 학습/검증으로 분할
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    # Optuna를 사용한 하이퍼파라미터 튜닝
    print("\n=== Optuna 하이퍼파라미터 튜닝 시작 ===")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train_opt, y_train_opt, X_val, y_val, scale_pos_weight),
        n_trials=30,  # 시도 횟수 (실제 사용 시 50-100으로 증가)
        timeout=1800,  # 30분 제한 (필요에 따라 조정)
        show_progress_bar=True
    )
    
    # 최적 하이퍼파라미터 확인
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"\n최적 ROC AUC: {best_score:.4f}")
    print("최적 하이퍼파라미터:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # 최종 모델 훈련 (전체 학습 데이터 사용)
    print("\n=== 최적 하이퍼파라미터로 최종 모델 훈련 ===")
    final_params = best_params.copy()
    final_params['scale_pos_weight'] = scale_pos_weight
    final_params['objective'] = 'binary:logistic'
    final_params['eval_metric'] = 'auc'
    final_params['random_state'] = 42
    
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
        verbose=True
    )
    
    # 모델 평가
    print("\n=== 모델 평가 ===")
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = final_model.predict(X_test)
    
    # 평가 지표
    auc_score = roc_auc_score(y_test, y_pred_proba)
    brier_score = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    
    print(f"ROC AUC 점수: {auc_score:.4f}")
    print(f"Brier 점수: {brier_score:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    print("\n혼동 행렬:")
    print(cm)
    
    # 분류 보고서
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred))
    
    # 특성 중요도
    print("\n특성 중요도:")
    importance = final_model.feature_importances_
    for i, feat in enumerate(features):
        print(f"{feat}: {importance[i]:.4f}")
    
    # 하이퍼파라미터 최적화 결과 시각화
    plt.figure(figsize=(12, 8))
    
    # 파라미터 중요도
    param_importance = optuna.visualization.plot_param_importances(study)
    plt.subplot(2, 2, 1)
    plt.title("파라미터 중요도")
    plt.tight_layout()
    
    # 최적화 히스토리
    plt.subplot(2, 2, 2)
    plt.title("최적화 히스토리")
    plt.plot([trial.value for trial in study.trials], marker='o')
    plt.xlabel("Trial")
    plt.ylabel("ROC AUC")
    plt.grid(True)
    
    # 특성 중요도
    plt.subplot(2, 2, 3)
    plt.title("특성 중요도")
    plt.barh(features, importance)
    plt.tight_layout()
    
    # 저장
    plot_path = f"{model_dir}/plots/optuna_results_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Optuna 최적화 결과 시각화 저장: {plot_path}")
    
    # 모델 저장
    model_file = f"{model_dir}/xgboost_optuna_model_{timestamp}.json"
    final_model.save_model(model_file)
    print(f"모델 저장 완료: {model_file}")
    
    # 최적 하이퍼파라미터 저장
    params_file = f"{model_dir}/best_params_{timestamp}.pkl"
    joblib.dump(best_params, params_file)
    print(f"최적 하이퍼파라미터 저장 완료: {params_file}")
    
    # 소요 시간 출력
    processing_time = time.time() - start_time
    print(f"\n총 처리 시간: {processing_time/60:.2f}분")
    print("=== 모델 학습 완료 ===")

if __name__ == "__main__":
    main() 