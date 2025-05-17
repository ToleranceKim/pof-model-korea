#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import glob
from datetime import datetime

def check_environment():
    """환경 확인 및 설정"""
    print("=== 데이터 전처리 파이프라인 시작 ===")
    print(f"시작 시간: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # 필요한 패키지가 설치되어 있는지 확인
    try:
        import pandas as pd
        import numpy as np
        import xarray as xr
    except ImportError:
        print("오류: 필요한 패키지가 설치되어 있지 않습니다.")
        print("pip install pandas numpy xarray 를 실행하여 설치하세요.")
        return False
    
    return True

def check_collected_data():
    """수집된 데이터 확인"""
    print("1. 수집된 데이터 확인 중...")
    
    # 산불 데이터 확인
    fire_data_path = os.path.join("data", "raw", "DL_FIRE_M-C61_613954", "fire_archive_M-C61_613954.csv")
    if not os.path.exists(fire_data_path):
        print("오류: 산불 데이터 파일이 존재하지 않습니다.")
        print("먼저 collect_data.py를 실행하여 데이터를 수집하세요.")
        return False
    
    # 날씨 데이터 확인
    weather_files = glob.glob(os.path.join("data", "raw", "era5_korea_*.nc"))
    if not weather_files:
        print("오류: 날씨 데이터 파일이 존재하지 않습니다.")
        print("먼저 collect_data.py를 실행하여 데이터를 수집하세요.")
        return False
    
    print("수집된 데이터 확인 완료.")
    return True

def clean_previous_outputs():
    """이전 처리 파일 삭제"""
    print("2. 이전 처리 파일 삭제 중...")
    
    # processed_data 디렉토리의 CSV 파일 삭제
    processed_data_files = glob.glob(os.path.join("processed_data", "*.csv"))
    for file in processed_data_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"경고: {file} 삭제 실패 - {e}")
    
    # outputs/data 디렉토리의 특정 CSV 파일 삭제
    output_files = [
        os.path.join("outputs", "data", "weather_data.csv"),
        os.path.join("outputs", "data", "weather_data_with_wind.csv")
    ]
    
    for file in output_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"경고: {file} 삭제 실패 - {e}")
    
    return True

def process_fire_data():
    """산불 데이터 전처리"""
    print("3. 산불 데이터 전처리 중...")
    
    # 현재 작업 디렉토리 저장
    current_dir = os.getcwd()
    
    try:
        # 전처리 스크립트 디렉토리로 이동
        os.chdir(os.path.join("src", "preprocessing"))
        
        # 산불 데이터 전처리 스크립트 실행
        input_path = "..\\..\\data\\raw\\DL_FIRE_M-C61_613954\\fire_archive_M-C61_613954.csv"
        output_path = "..\\..\\data\\reference\\af_flag_korea.csv"
        
        # 리눅스/맥 환경에서 경로 구분자 변경
        if os.name != 'nt':
            input_path = "../../data/raw/DL_FIRE_M-C61_613954/fire_archive_M-C61_613954.csv"
            output_path = "../../data/reference/af_flag_korea.csv"
            
        result = subprocess.run([
            "python", "process_af_flag.py",
            "--input", input_path,
            "--output", output_path
        ], check=True)
        
        if result.returncode != 0:
            print("오류: 산불 데이터 전처리 실패")
            return False
            
    except subprocess.CalledProcessError:
        print("오류: 산불 데이터 전처리 스크립트 실행 중 오류 발생")
        return False
    finally:
        # 원래 디렉토리로 복귀
        os.chdir(current_dir)
    
    return True

def process_weather_data():
    """날씨 데이터 전처리 및 풍속 계산, 데이터 병합 처리"""
    print("4. 날씨 데이터 전처리 및 풍속 계산, 데이터 병합 처리...")
    
    # 현재 작업 디렉토리 저장
    current_dir = os.getcwd()
    
    try:
        # 전처리 스크립트 디렉토리로 이동
        os.chdir(os.path.join("src", "preprocessing"))
        
        # 날씨 데이터 전처리 스크립트 실행
        data_dir = "..\\..\\data\\raw"
        output_dir = "..\\..\\processed_data"
        target_path = "..\\..\\data\\reference\\af_flag_korea.csv"
        final_output = "..\\..\\outputs\\data\\weather_data_with_wind.csv"
        
        # 리눅스/맥 환경에서 경로 구분자 변경
        if os.name != 'nt':
            data_dir = "../../data/raw"
            output_dir = "../../processed_data"
            target_path = "../../data/reference/af_flag_korea.csv"
            final_output = "../../outputs/data/weather_data_with_wind.csv"
            
        result = subprocess.run([
            "python", "process_weather.py",
            "--data_dir", data_dir,
            "--output_dir", output_dir,
            "--target_path", target_path,
            "--final_output", final_output
        ], check=True)
        
        if result.returncode != 0:
            print("오류: 날씨 데이터 전처리 실패")
            return False
            
    except subprocess.CalledProcessError:
        print("오류: 날씨 데이터 전처리 스크립트 실행 중 오류 발생")
        return False
    finally:
        # 원래 디렉토리로 복귀
        os.chdir(current_dir)
    
    return True

def check_data_dimensions():
    """차원 일치 검증"""
    print("5. 차원 일치 검증 중...")
    
    try:
        # 차원 검증 스크립트 실행
        weather_data = os.path.join("outputs", "data", "weather_data_with_wind.csv")
        af_flag = os.path.join("data", "reference", "af_flag_korea.csv")
        
        result = subprocess.run([
            "python", "check_dimensions.py",
            "--weather_data", weather_data,
            "--af_flag", af_flag
        ], check=True)
        
        if result.returncode != 0:
            print("경고: 차원 일치 검증에 문제가 발생했습니다. 데이터를 확인하세요.")
            
    except subprocess.CalledProcessError:
        print("경고: 차원 검증 스크립트 실행 중 오류 발생")
    
    return True

def main():
    # 환경 설정 확인
    if not check_environment():
        sys.exit(1)
    
    # 수집된 데이터 확인
    if not check_collected_data():
        sys.exit(1)
    
    # 이전 처리 파일 삭제
    clean_previous_outputs()
    
    # 산불 데이터 전처리
    if not process_fire_data():
        sys.exit(1)
    
    # 날씨 데이터 전처리 및 병합
    if not process_weather_data():
        sys.exit(1)
    
    # 차원 일치 검증
    check_data_dimensions()
    
    print()
    print("=== 전처리 파이프라인 완료 ===")
    print(f"완료 시간: {datetime.now().strftime('%H:%M:%S')}")
    print("결과 파일:")
    print("- 산불 데이터: data/reference/af_flag_korea.csv")
    print("- 최종 데이터: outputs/data/weather_data_with_wind.csv (풍속 계산 포함)")

if __name__ == "__main__":
    main() 