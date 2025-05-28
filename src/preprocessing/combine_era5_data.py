#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def combine_era5_data(input_dir, output_dir, date_range=None):
    """
    ERA5 보간 처리된 날씨 데이터 파일들을 결합하는 함수
    
    Args:
        input_dir (str): 입력 디렉토리 경로 (처리된 CSV 파일이 있는 곳)
        output_dir (str): 출력 디렉토리 경로
        date_range (tuple): 선택적 날짜 범위 (시작일, 종료일) - YYYYMMDD 형식
    
    Returns:
        str: 결합된 데이터 파일 경로
    """
    print(f"Combining ERA5 data files from: {input_dir}")
    
    # 입력 디렉토리 내 모든 CSV 파일 찾기
    csv_files = glob.glob(os.path.join(input_dir, 'era5_daily_*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    
    # 날짜 범위 필터링
    if date_range:
        start_date, end_date = date_range
        filtered_files = []
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            date_part = file_name.split('_')[-1].split('.')[0]  # era5_daily_YYYYMM.csv에서 YYYYMM 추출
            
            if len(date_part) >= 6:  # YYYYMM 형식인지 확인
                year_month = date_part[:6]  # YYYYMM
                if start_date <= year_month <= end_date:
                    filtered_files.append(file_path)
        
        csv_files = filtered_files
        print(f"Filtered to {len(csv_files)} files within date range {start_date} to {end_date}")
    
    # 파일이 없으면 종료
    if not csv_files:
        print("No files to process after filtering")
        return None
    
    # 모든 CSV 파일 로드 및 결합
    print("Loading and combining files...")
    all_dfs = []
    
    for file_path in csv_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"  - Shape: {df.shape}")
            
            # 기본 검증
            expected_columns = ['acq_date', 'grid_id', 'latitude', 'longitude']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  - Warning: Missing columns in {file_path}: {missing_columns}")
            
            # 날짜 형식 통일
            if 'acq_date' in df.columns:
                df['acq_date'] = pd.to_datetime(df['acq_date'])
            
            all_dfs.append(df)
        
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
    
    # 결합
    if not all_dfs:
        print("No valid data frames to combine")
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    
    # 중복 제거 (같은 날짜의 같은 grid_id 값)
    if 'acq_date' in combined_df.columns and 'grid_id' in combined_df.columns:
        before_dedup = combined_df.shape[0]
        combined_df = combined_df.drop_duplicates(subset=['acq_date', 'grid_id'])
        after_dedup = combined_df.shape[0]
        
        if before_dedup > after_dedup:
            print(f"Removed {before_dedup - after_dedup} duplicate records")
    
    # 풍속 계산 (누락된 경우)
    if '10u' in combined_df.columns and '10v' in combined_df.columns and 'wind10m' not in combined_df.columns:
        print("Calculating wind speed...")
        u = combined_df['10u']
        v = combined_df['10v']
        combined_df['wind10m'] = np.sqrt(u**2 + v**2)
    
    # 데이터 정렬
    if 'acq_date' in combined_df.columns:
        combined_df = combined_df.sort_values(['acq_date', 'grid_id'])
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명에 날짜 범위 정보 추가
    date_info = ""
    if date_range:
        date_info = f"_{date_range[0]}_{date_range[1]}"
    else:
        # 데이터에서 날짜 범위 추출
        if 'acq_date' in combined_df.columns:
            min_date = combined_df['acq_date'].min()
            max_date = combined_df['acq_date'].max()
            if pd.notnull(min_date) and pd.notnull(max_date):
                min_str = min_date.strftime('%Y%m')
                max_str = max_date.strftime('%Y%m')
                date_info = f"_{min_str}_{max_str}"
    
    # 파일 저장
    parquet_path = os.path.join(output_dir, f'era5_daily_combined{date_info}.parquet')
    combined_df.to_parquet(parquet_path, index=False)
    print(f"Saved combined data to {parquet_path}")
    
    # CSV 저장 (선택적)
    csv_path = os.path.join(output_dir, f'era5_daily_combined{date_info}.csv')
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved combined data to {csv_path}")
    
    # 데이터 시각화 및 요약
    visualize_combined_data(combined_df, output_dir, date_info)
    
    return parquet_path

def visualize_combined_data(df, output_dir, date_info):
    """
    결합된 데이터를 시각화하는 함수
    
    Args:
        df: 결합된 데이터프레임
        output_dir: 출력 디렉토리
        date_info: 날짜 정보 문자열
    """
    viz_dir = os.path.join(output_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. 데이터 요약 정보
    print("\n===== Data Summary =====")
    print(f"Records: {df.shape[0]}")
    print(f"Unique dates: {df['acq_date'].nunique()}")
    print(f"Unique grid points: {df['grid_id'].nunique()}")
    print(f"Date range: {df['acq_date'].min()} to {df['acq_date'].max()}")
    
    # 2. 결측치 시각화
    plt.figure(figsize=(12, 6))
    missing = df.isnull().sum() / len(df) * 100
    missing = missing.sort_values(ascending=False)
    sns.barplot(x=missing.index, y=missing.values)
    plt.title('Missing Values (%)')
    plt.xlabel('Variables')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'missing_values{date_info}.png'), dpi=300)
    plt.close()
    
    # 3. 격자점 분포 시각화
    if df.shape[0] <= 100000:  # 데이터가 너무 많으면 샘플링
        sample_df = df
    else:
        sample_df = df.sample(100000, random_state=42)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(sample_df['longitude'], sample_df['latitude'], 
                s=1, alpha=0.5, c='blue')
    plt.title('Spatial Distribution of Grid Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'grid_distribution{date_info}.png'), dpi=300)
    plt.close()
    
    # 4. 기온 분포 시각화
    if 't2m' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['t2m'] - 273.15, kde=True)  # 켈빈을 섭씨로 변환
        plt.title('Temperature Distribution (°C)')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'temperature_distribution{date_info}.png'), dpi=300)
        plt.close()
    
    print(f"Visualizations saved to {viz_dir}")

def main():
    """메인 함수"""
    if len(sys.argv) < 3:
        print("Usage: python combine_era5_data.py <input_dir> <output_dir> [start_date end_date]")
        print("Example: python combine_era5_data.py processed_data/era5_daily processed_data 201901 202412")
        
        # 기본값 사용
        input_dir = "processed_data/era5_daily"
        output_dir = "processed_data"
        print(f"Using default values: input_dir={input_dir}, output_dir={output_dir}")
    else:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    
    date_range = None
    if len(sys.argv) >= 5:
        date_range = (sys.argv[3], sys.argv[4])
    
    combine_era5_data(input_dir, output_dir, date_range)

if __name__ == "__main__":
    main() 