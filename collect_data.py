#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import argparse
from datetime import datetime

def check_environment():
    """환경 확인 및 설정"""
    print("=== 데이터 수집 파이프라인 시작 ===")
    print(f"시작 시간: {datetime.now().strftime('%H:%M:%S')}")
    print()

    # 필요한 패키지가 설치되어 있는지 확인
    try:
        import cdsapi
        import pandas as pd
        import numpy as np
    except ImportError:
        print("오류: 필요한 패키지가 설치되어 있지 않습니다.")
        print("pip install cdsapi pandas numpy 를 실행하여 설치하세요.")
        return False
    
    return True

def check_fire_data():
    """산불 데이터 파일 확인"""
    print("1. 산불 데이터 수집 확인...")
    fire_data_path = os.path.join("data", "raw", "DL_FIRE_M-C61_613954", "fire_archive_M-C61_613954.csv")
    
    if not os.path.exists(fire_data_path):
        print("경고: 산불 데이터 파일이 존재하지 않습니다.")
        print(f"{fire_data_path} 파일을 확인하세요.")
        return False
    
    print("산불 데이터 확인 완료.")
    return True

def collect_weather_data(start_year, end_year, start_month, end_month):
    """날씨 데이터 수집"""
    print("2. 날씨 데이터 수집 실행 중...")
    print(f"   수집 기간: {start_year}-{start_month}월 ~ {end_year}-{end_month}월")
    
    # 현재 작업 디렉토리 저장
    current_dir = os.getcwd()
    
    try:
        # 데이터 수집 스크립트 디렉토리로 이동
        os.chdir(os.path.join("src", "data_collection"))
        
        # 날씨 데이터 수집 스크립트 실행
        cmd = [
            "python", "collect_weather.py",
            "--start_year", str(start_year),
            "--end_year", str(end_year),
            "--start_month", str(start_month),
            "--end_month", str(end_month),
            "--output_dir", "..\\..\\data\\raw"
        ]
        
        # 리눅스/맥 환경에서 경로 구분자 변경
        if os.name != 'nt':
            cmd[-1] = "../../data/raw"
            
        result = subprocess.run(cmd, check=True)
        
        if result.returncode != 0:
            print("오류: 날씨 데이터 수집 실패")
            return False
            
    except subprocess.CalledProcessError:
        print("오류: 날씨 데이터 수집 스크립트 실행 중 오류 발생")
        return False
    finally:
        # 원래 디렉토리로 복귀
        os.chdir(current_dir)
    
    return True

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='날씨 데이터 수집 스크립트')
    parser.add_argument('--start_year', type=int, default=2024, help='수집 시작 연도')
    parser.add_argument('--end_year', type=int, default=2024, help='수집 종료 연도')
    parser.add_argument('--start_month', type=int, default=1, help='수집 시작 월')
    parser.add_argument('--end_month', type=int, default=12, help='수집 종료 월')
    args = parser.parse_args()
    
    # 환경 설정 확인
    if not check_environment():
        sys.exit(1)
    
    # 산불 데이터 확인
    if not check_fire_data():
        sys.exit(1)
    
    # 날씨 데이터 수집
    if not collect_weather_data(args.start_year, args.end_year, args.start_month, args.end_month):
        sys.exit(1)
    
    print()
    print("=== 데이터 수집 완료 ===")
    print(f"완료 시간: {datetime.now().strftime('%H:%M:%S')}")
    print("다음 단계: process_data.py 실행하여 데이터 전처리 진행")
    print()
    print("수집된 데이터 검증 후 전처리를 진행하세요.")
    print("- 날씨 데이터: data/raw/era5_korea_*.nc")
    print("- 산불 데이터: data/raw/DL_FIRE_M-C61_613954/fire_archive_M-C61_613954.csv")

if __name__ == "__main__":
    main() 