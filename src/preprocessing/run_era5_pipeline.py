#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import gc
import logging
from pathlib import Path
from process_era5 import ERA5Processor, process_era5_file, process_multiple_files_parallel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_era5_pipeline(input_dir, output_dir, start_date=None, end_date=None, 
                     multi_file_parallel=True, inner_parallel=True, interpolate=True,
                     max_workers=None, combine_results=True, test_mode=False, max_files=2,
                     force_reprocess=False):
    """
    ERA5 데이터 처리 파이프라인을 실행하는 함수
    
    Args:
        input_dir (str): 입력 디렉토리 경로 (ERA5 파일이 있는 위치)
        output_dir (str): 출력 디렉토리 경로
        start_date (str): 시작 날짜 (YYYYMM 형식), 없으면 모든 파일 처리
        end_date (str): 종료 날짜 (YYYYMM 형식), 없으면 모든 파일 처리
        multi_file_parallel (bool): 다중 파일 병렬 처리 사용 여부
        inner_parallel (bool): 파일 내부 병렬 처리 사용 여부
        interpolate (bool): 0.1° 격자로 보간 여부
        max_workers (int): 최대 작업자 수 (None이면 자동 결정)
        combine_results (bool): 처리 결과 병합 여부
        test_mode (bool): 테스트 모드 (일부 파일만 처리)
        max_files (int): 테스트 모드에서 처리할 최대 파일 수
        force_reprocess (bool): 이미 처리된 파일도 강제로 다시 처리
    
    Returns:
        bool: 성공 여부
    """
    print("="*80)
    print(f"ERA5 데이터 처리 파이프라인 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 기존 처리된 파일 삭제 (force_reprocess가 True인 경우)
    if force_reprocess:
        print("기존 처리된 파일을 삭제하고 다시 처리합니다.")
        if start_date or end_date:
            # 날짜 범위에 맞는 파일만 삭제
            start = start_date if start_date else "000000"
            end = end_date if end_date else "999999"
            
            for filename in os.listdir(output_dir):
                if filename.startswith("era5_korea_"):
                    parts = filename.split('_')
                    if len(parts) > 2:
                        date_part = parts[2].split('.')[0]
                        if start <= date_part <= end:
                            file_path = os.path.join(output_dir, filename)
                            try:
                                os.remove(file_path)
                                print(f"삭제됨: {file_path}")
                            except Exception as e:
                                print(f"파일 삭제 중 오류 발생: {file_path} - {str(e)}")
        else:
            # 결합 파일은 삭제하지 않음
            for filename in os.listdir(output_dir):
                if filename.startswith("era5_korea_") and not "_all_" in filename:
                    file_path = os.path.join(output_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"삭제됨: {file_path}")
                    except Exception as e:
                        print(f"파일 삭제 중 오류 발생: {file_path} - {str(e)}")

    # 1. 파일 목록 가져오기
    weather_files = []
    
    # 다양한 파일 형식 처리 (.nc, .zip, .csv)
    # ERA5 데이터의 경우 .nc 확장자를 가진 파일도 실제로는 ZIP 파일인 경우가 많음
    # 모든 ERA5 관련 파일은 ZIP 처리 경로로 처리
    for file_ext in ['.nc', '.zip', '.csv']:
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                if f.startswith('era5_korea_') and f.endswith(file_ext)]
        weather_files.extend(files)
    
    weather_files.sort()
    
    if not weather_files:
        print(f"입력 디렉토리에 처리할 파일이 없습니다: {input_dir}")
        return False
    
    # 2. 날짜 필터링
    if start_date or end_date:
        filtered_files = []
        
        # 날짜 범위 설정
        start = start_date if start_date else "000000"
        end = end_date if end_date else "999999"
        
        for file_path in weather_files:
            file_name = os.path.basename(file_path)
            # 파일명에서 날짜 부분 추출 (예: era5_korea_202001.nc -> 202001)
            date_part = ""
            parts = file_name.split('_')
            if len(parts) > 2:
                date_part = parts[2].split('.')[0]
            
            if start <= date_part <= end:
                filtered_files.append(file_path)
        
        weather_files = filtered_files
        print(f"날짜 필터링 후 처리할 파일: {len(weather_files)}개")
    
    # 3. 테스트 모드 처리
    if test_mode:
        print(f"테스트 모드 활성화: 최대 {max_files}개 파일만 처리합니다.")
        weather_files = weather_files[:max_files]
    
    print(f"처리할 파일 {len(weather_files)}개 발견")
    
    # 4. 파일 처리
    processed_files = []
    
    if multi_file_parallel and len(weather_files) > 1:
        # 다중 파일 병렬 처리
        processed_files = process_multiple_files_parallel(
            weather_files, 
            output_dir, 
            max_workers=max_workers,
            use_parallel_processing=inner_parallel,
            interpolate_to_01deg=interpolate
        )
    else:
        # 순차 처리
        for file_path in tqdm(weather_files, desc="처리 중"):
            result = process_era5_file(
                file_path, 
                output_dir, 
                use_parallel=inner_parallel,
                interpolate_to_01deg=interpolate
            )
            
            if result:
                processed_files.append(result)
            
            # 메모리 정리
            gc.collect()
    
    # 5. 결과 확인
    if not processed_files:
        print("처리된 파일이 없습니다.")
        return False
    
    print(f"\n총 {len(processed_files)}/{len(weather_files)} 파일 처리됨")
    
    # 6. 데이터 결합 (선택적)
    if combine_results and len(processed_files) > 1:
        print("\n"+"="*80)
        print("모든 파일 처리 완료. 데이터 결합 시작...")
        
        # 결합 파일명에 사용할 날짜 값 처리
        output_start_date = start_date if start_date else "all"
        output_end_date = end_date if end_date else "all"
        
        # 결합할 파일 목록 확인
        combined_csv_path = os.path.join(output_dir, f"era5_korea_{output_start_date}_{output_end_date}.csv")
        combined_parquet_path = os.path.join(output_dir, f"era5_korea_{output_start_date}_{output_end_date}.parquet")
        
        # 이미 결합된 파일이 있는지 확인
        if os.path.exists(combined_parquet_path) and os.path.exists(combined_csv_path):
            print(f"이미 결합된 파일이 존재합니다: {combined_parquet_path}")
            print(f"이미 결합된 파일이 존재합니다: {combined_csv_path}")
        else:
            # 모든 처리된 파일 로드 및 결합
            all_dfs = []
            
            for file_path in tqdm(processed_files, desc="파일 로드 중"):
                try:
                    # Parquet 파일 로드 (더 빠름)
                    df = pd.read_parquet(file_path)
                    all_dfs.append(df)
                    
                    # 메모리 관리
                    gc.collect()
                except Exception as e:
                    print(f"파일 로드 중 오류 발생: {file_path} - {str(e)}")
            
            if all_dfs:
                # 모든 데이터프레임 결합
                try:
                    print(f"총 {len(all_dfs)}개 데이터프레임 결합 중...")
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    print(f"결합된 데이터프레임 크기: {combined_df.shape}")
                    
                    # CSV 및 Parquet 형식으로 저장
                    print(f"결합된 데이터 저장 중: {combined_csv_path}")
                    combined_df.to_csv(combined_csv_path, index=False)
                    
                    print(f"결합된 데이터 저장 중: {combined_parquet_path}")
                    combined_df.to_parquet(combined_parquet_path, index=False)
                    
                    print("데이터 결합 완료!")
                except Exception as e:
                    print(f"데이터 결합 중 오류 발생: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("결합할 데이터가 없습니다.")
    
    # 7. 실행 시간 계산 및 출력
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n"+"="*80)
    print(f"ERA5 데이터 처리 파이프라인 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 실행 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.2f}초")
    print("="*80)
    
    return True

def main():
    # Create output directories
    base_dir = Path('data/raw')
    output_dir = Path('data/processed/era5')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ERA5 processor
    processor = ERA5Processor(base_dir=base_dir)
    
    # Process years 2019-2024
    for year in range(2019, 2025):
        try:
            logger.info(f"Starting processing for year {year}")
            output_file = processor.process_year(year)
            if output_file:
                logger.info(f"Successfully processed year {year}. Output: {output_file}")
            else:
                logger.error(f"Failed to process year {year}")
        except Exception as e:
            logger.error(f"Error processing year {year}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 