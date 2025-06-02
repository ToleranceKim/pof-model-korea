#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import argparse
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('era5_combine_datasets.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def combine_era5_datasets(older_dataset, newer_dataset, output_path, viz_dir=None):
    """
    두 개의 ERA5 데이터셋을 결합하는 함수
    
    Args:
        older_dataset (str): 이전 기간 데이터셋 경로
        newer_dataset (str): 최근 기간 데이터셋 경로
        output_path (str): 출력 파일 경로
        viz_dir (str, optional): 시각화 결과 저장 디렉토리
    
    Returns:
        str: 결합된 파일 경로
    """
    start_time = time.time()
    logger.info(f"데이터셋 결합 시작...")
    logger.info(f"이전 기간 데이터셋: {older_dataset}")
    logger.info(f"최근 기간 데이터셋: {newer_dataset}")
    
    # 파일 확장자 확인
    older_ext = os.path.splitext(older_dataset)[1].lower()
    newer_ext = os.path.splitext(newer_dataset)[1].lower()
    
    # 데이터셋 로드
    logger.info("데이터셋 로드 중...")
    
    if older_ext == '.parquet':
        df_older = pd.read_parquet(older_dataset)
    elif older_ext == '.csv':
        df_older = pd.read_csv(older_dataset)
    else:
        logger.error(f"지원되지 않는 파일 형식: {older_ext}")
        return None
    
    if newer_ext == '.parquet':
        df_newer = pd.read_parquet(newer_dataset)
    elif newer_ext == '.csv':
        df_newer = pd.read_csv(newer_dataset)
    else:
        logger.error(f"지원되지 않는 파일 형식: {newer_ext}")
        return None
    
    # 데이터셋 정보
    logger.info(f"이전 기간 데이터셋 크기: {df_older.shape}")
    logger.info(f"최근 기간 데이터셋 크기: {df_newer.shape}")
    
    # 날짜 열 확인 및 처리
    date_col = None
    for col in ['acq_date', 'date', 'time']:
        if col in df_older.columns and col in df_newer.columns:
            date_col = col
            break
    
    if date_col is None:
        logger.error("날짜 열을 찾을 수 없습니다")
        return None
    
    # 날짜 형식 변환
    if df_older[date_col].dtype == 'object':
        df_older[date_col] = pd.to_datetime(df_older[date_col])
    if df_newer[date_col].dtype == 'object':
        df_newer[date_col] = pd.to_datetime(df_newer[date_col])
    
    # 날짜 범위 확인
    older_start = df_older[date_col].min()
    older_end = df_older[date_col].max()
    newer_start = df_newer[date_col].min()
    newer_end = df_newer[date_col].max()
    
    logger.info(f"이전 기간 날짜 범위: {older_start} ~ {older_end}")
    logger.info(f"최근 기간 날짜 범위: {newer_start} ~ {newer_end}")
    
    # 중복 날짜 확인
    overlapping_dates = set(df_older[date_col]) & set(df_newer[date_col])
    if overlapping_dates:
        logger.warning(f"중복된 날짜가 {len(overlapping_dates)}개 있습니다")
        logger.warning(f"중복 날짜 예시: {list(overlapping_dates)[:5]}...")
        
        # 중복 데이터 처리 (최근 데이터 우선)
        df_older = df_older[~df_older[date_col].isin(overlapping_dates)]
        logger.info(f"중복 날짜 제거 후 이전 기간 데이터셋 크기: {df_older.shape}")
    
    # 컬럼 일관성 확인
    older_cols = set(df_older.columns)
    newer_cols = set(df_newer.columns)
    
    if older_cols != newer_cols:
        missing_in_older = newer_cols - older_cols
        missing_in_newer = older_cols - newer_cols
        
        if missing_in_older:
            logger.warning(f"이전 기간 데이터셋에 없는 열: {missing_in_older}")
            for col in missing_in_older:
                df_older[col] = np.nan
        
        if missing_in_newer:
            logger.warning(f"최근 기간 데이터셋에 없는 열: {missing_in_newer}")
            for col in missing_in_newer:
                df_newer[col] = np.nan
    
    # 데이터셋 결합
    logger.info("데이터셋 결합 중...")
    combined_df = pd.concat([df_older, df_newer], ignore_index=True)
    
    # 날짜 기준으로 정렬
    combined_df.sort_values(by=date_col, inplace=True)
    
    # 결합 결과 정보
    logger.info(f"결합된 데이터셋 크기: {combined_df.shape}")
    logger.info(f"결합된 날짜 범위: {combined_df[date_col].min()} ~ {combined_df[date_col].max()}")
    logger.info(f"고유 날짜 수: {combined_df[date_col].nunique()}")
    
    if 'grid_id' in combined_df.columns:
        logger.info(f"고유 격자점 수: {combined_df['grid_id'].nunique()}")
    
    # 결과 저장
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 확장자 없는 기본 경로
    output_base = os.path.splitext(output_path)[0]
    
    # Parquet 저장
    parquet_path = f"{output_base}.parquet"
    logger.info(f"Parquet 파일 저장 중: {parquet_path}")
    combined_df.to_parquet(parquet_path, index=False)
    
    # CSV 저장
    csv_path = f"{output_base}.csv"
    logger.info(f"CSV 파일 저장 중: {csv_path}")
    combined_df.to_csv(csv_path, index=False)
    
    # 시각화 (선택적)
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
        logger.info(f"시각화 생성 중: {viz_dir}")
        
        # 1. 연도별 데이터 수
        if date_col in combined_df.columns:
            combined_df['year'] = combined_df[date_col].dt.year
            yearly_counts = combined_df.groupby('year').size()
            
            plt.figure(figsize=(12, 6))
            yearly_counts.plot(kind='bar')
            plt.title('연도별 데이터 수')
            plt.xlabel('연도')
            plt.ylabel('데이터 수')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'yearly_data_counts.png'), dpi=300)
            plt.close()
        
        # 2. 변수별 결측치 비율
        plt.figure(figsize=(12, 6))
        missing = combined_df.isnull().sum() / len(combined_df) * 100
        missing = missing.sort_values(ascending=False)
        sns.barplot(x=missing.index, y=missing.values)
        plt.title('변수별 결측치 비율 (%)')
        plt.xlabel('변수')
        plt.ylabel('결측치 비율 (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'missing_values.png'), dpi=300)
        plt.close()
        
        # 3. 시계열 연속성 확인 (기온 등)
        if 't2m' in combined_df.columns and date_col in combined_df.columns:
            # 일별 평균 기온
            daily_temp = combined_df.groupby(date_col)['t2m'].mean()
            
            plt.figure(figsize=(14, 6))
            daily_temp.plot()
            plt.title('일별 평균 기온 시계열')
            plt.xlabel('날짜')
            plt.ylabel('기온 (K)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'temperature_timeseries.png'), dpi=300)
            plt.close()
    
    # 실행 시간 계산
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("="*80)
    logger.info(f"데이터셋 결합 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"결과 파일:")
    logger.info(f"  - {parquet_path}")
    logger.info(f"  - {csv_path}")
    logger.info(f"총 실행 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.2f}초")
    logger.info("="*80)
    
    return parquet_path

def main():
    """
    메인 함수
    """
    # 인자 파싱
    parser = argparse.ArgumentParser(description='ERA5 데이터셋 결합 스크립트')
    parser.add_argument('--older_dataset', type=str, required=True,
                        help='이전 기간 데이터셋 경로 (.parquet 또는 .csv)')
    parser.add_argument('--newer_dataset', type=str, required=True,
                        help='최근 기간 데이터셋 경로 (.parquet 또는 .csv)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='출력 파일 경로 (확장자 없이)')
    parser.add_argument('--viz_dir', type=str, default=None,
                        help='시각화 파일을 저장할 디렉토리 경로')
    
    args = parser.parse_args()
    
    # 데이터셋 결합 실행
    combined_file = combine_era5_datasets(
        args.older_dataset,
        args.newer_dataset,
        args.output_path,
        args.viz_dir
    )
    
    if combined_file:
        logger.info(f"프로그램 종료: 결합된 파일이 생성되었습니다.")
    else:
        logger.error("프로그램 종료: 데이터셋 결합 실패")

if __name__ == "__main__":
    main() 