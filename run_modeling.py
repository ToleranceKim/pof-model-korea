#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import argparse
from datetime import datetime

def check_environment():
    """환경 및 패키지 확인"""
    print("===== 산불 예측 모델 학습 실행 =====")
    print(f"시작 시간: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    print("1. 필요 패키지 확인")
    required_packages = ['xgboost', 'scikit-learn', 'optuna', 'joblib', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"오류: 필요한 패키지가 설치되어 있지 않습니다: {', '.join(missing_packages)}")
        print("다음 명령어로 필요한 패키지를 설치해주세요:")
        print("pip install -r requirements.txt")
        return False
    
    # XGBoost 버전 확인
    try:
        import xgboost as xgb
        xgb_version = xgb.__version__
        print(f"XGBoost 버전: {xgb_version}")
        
        # 버전 분해
        major, minor, *rest = xgb_version.split('.')
        version_int = int(major) * 100 + int(minor)
        
        if version_int >= 300:  # 3.0.0 이상인 경우
            print("\n경고: XGBoost 3.0.0 이상 버전은 현재 코드와 호환되지 않습니다.")
            print("다음 명령어로 호환되는 버전(1.7.6)으로 다운그레이드해주세요:")
            print("pip install xgboost==1.7.6")
            return False
        
        if version_int < 100:  # 1.0.0 미만일 경우
            print("\n경고: 설치된 XGBoost 버전이 오래되었습니다.")
            print("현재 모델 코드는 XGBoost 1.7.6 버전에 최적화되어 있습니다.")
            print("다음 명령어로 설치해주세요:")
            print("pip install xgboost==1.7.6")
            return False
    except Exception as e:
        print(f"XGBoost 버전 확인 중 오류: {e}")
        return False
    
    print("패키지 확인 완료")
    return True

def check_data_directories():
    """디렉토리 확인 및 생성"""
    print("2. 디렉토리 확인")
    
    # 데이터 파일 확인
    weather_data_path = os.path.join("outputs", "data", "weather_data_with_wind.csv")
    if not os.path.exists(weather_data_path):
        print("오류: 데이터 파일이 존재하지 않습니다.")
        print("먼저 data_collection과 preprocessing을 실행하여 데이터 파일을 생성하세요.")
        return False
    
    # 결과 디렉토리 생성
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "models", "plots"), exist_ok=True)
    
    return True

def train_basic_model():
    """기본 XGBoost 모델 훈련"""
    print()
    print("3. 기본 XGBoost 모델 훈련 실행")
    print()
    
    try:
        # 모델 훈련 스크립트 실행
        result = subprocess.run([
            sys.executable, 
            os.path.join("src", "modeling", "train_model.py")
        ], check=True)
        
        if result.returncode != 0:
            print("오류: 기본 모델 훈련 중 문제가 발생했습니다.")
            return False
            
    except subprocess.CalledProcessError:
        print("오류: 모델 훈련 스크립트 실행 중 오류 발생")
        return False
    
    return True

def run_optuna_tuning():
    """Optuna 하이퍼파라미터 튜닝"""
    print()
    print("4. Optuna 하이퍼파라미터 튜닝 실행 (옵션)")
    print("이 과정은 시간이 오래 걸릴 수 있습니다.")
    
    while True:
        response = input("Optuna 튜닝을 실행하시겠습니까? (y/n): ").strip().lower()
        if response in ['y', 'n']:
            break
        print("'y' 또는 'n'으로 응답해주세요.")
    
    if response == 'y':
        print()
        try:
            # Optuna 튜닝 스크립트 실행
            result = subprocess.run([
                sys.executable, 
                os.path.join("src", "modeling", "train_model_optuna.py")
            ], check=True)
            
            if result.returncode != 0:
                print("오류: Optuna 튜닝 중 문제가 발생했습니다.")
                return False
                
        except subprocess.CalledProcessError:
            print("오류: Optuna 튜닝 스크립트 실행 중 오류 발생")
            return False
    else:
        print("Optuna 튜닝을 건너뜁니다.")
    
    return True

def main():
    # 환경 설정 확인
    if not check_environment():
        sys.exit(1)
    
    # 디렉토리 확인
    if not check_data_directories():
        sys.exit(1)
    
    # 기본 XGBoost 모델 훈련
    if not train_basic_model():
        sys.exit(1)
    
    # Optuna 하이퍼파라미터 튜닝 (선택 사항)
    if not run_optuna_tuning():
        sys.exit(1)
    
    print()
    print("===== 모델 학습 완료 =====")
    print(f"완료 시간: {datetime.now().strftime('%H:%M:%S')}")
    print()
    print("모델 파일: outputs/models/")
    print()

if __name__ == "__main__":
    main() 