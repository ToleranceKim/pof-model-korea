#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse
import os
import gc
from tqdm import tqdm

def join_datasets(weather_file, target_file, output_file, join_type='inner'):
    """
    전처리된 날씨 데이터와 타겟 데이터(af_flag)를 병합합니다.
    
    Parameters:
    -----------
    weather_file : str
        전처리된 날씨 데이터 파일 경로 (Parquet 형식)
    target_file : str
        타겟 데이터(af_flag) 파일 경로 (CSV 형식)
    output_file : str
        병합 결과 저장 파일 경로
    join_type : str
        조인 유형 ('inner' 또는 'right')
    """
    print("\n=== 날씨 데이터와 타겟 데이터 병합 시작 ===")
    
    # 파일 존재 확인
    for file_path, file_desc in [(weather_file, "날씨 데이터"), (target_file, "타겟 데이터")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_desc} 파일을 찾을 수 없습니다: {file_path}")
    
    # 날씨 데이터 로드
    print(f"날씨 데이터 로드 중: {weather_file}")
    weather_df = pd.read_parquet(weather_file)
    print(f"날씨 데이터 로드 완료: {weather_df.shape}")
    
    # 타겟 데이터 로드
    print(f"타겟 데이터 로드 중: {target_file}")
    target_df = pd.read_csv(target_file, parse_dates=['acq_date'])
    target_df = target_df[['acq_date', 'grid_id', 'af_flag']]
    print(f"타겟 데이터 로드 완료: {target_df.shape}")
    
    # 병합 전 grid_id 범위 확인 출력
    print(f"\n데이터 범위 정보:")
    weather_min_date = weather_df.acq_date.min()
    weather_max_date = weather_df.acq_date.max()
    print(f"날씨 데이터 grid_id 범위: {weather_df.grid_id.min()} - {weather_df.grid_id.max()}")
    print(f"타겟 데이터 grid_id 범위: {target_df.grid_id.min()} - {target_df.grid_id.max()}")
    print(f"날씨 데이터 날짜 범위: {weather_min_date} - {weather_max_date}")
    print(f"타겟 데이터 날짜 범위: {target_df.acq_date.min()} - {target_df.acq_date.max()}")
    
    # 날짜 범위 문자열 생성 (YYYY-YYYY 형식)
    date_range = f"{weather_min_date.year}-{weather_max_date.year}"
    print(f"날짜 범위 문자열: {date_range}")
    
    # 타겟 데이터의 양성 샘플 정보
    positive_samples = target_df[target_df['af_flag'] == 1]
    print(f"타겟 데이터 양성 샘플 수: {len(positive_samples)} ({len(positive_samples) / len(target_df) * 100:.4f}%)")
    
    # 데이터 병합 - inner join 방식으로 변경
    print(f"\n데이터 병합 중 ('{join_type}' join 사용)...")
    final_df = weather_df.merge(target_df, on=['acq_date', 'grid_id'], how=join_type)
    print(f"데이터 병합 완료: {final_df.shape}")
    
    # 결측치 처리
    if final_df.isnull().sum().sum() > 0:
        print("\n결측치 처리 중...")
        print("결측치 개수:", final_df.isnull().sum())
        
        # 각 컬럼별 결측치 처리
        weather_cols = ['t2m', 'td2m', '10u', '10v']
        for col in weather_cols:
            if col in final_df.columns and final_df[col].isnull().sum() > 0:
                col_mean = weather_df[col].mean()
                print(f"{col} 결측치를 평균값 {col_mean:.4f}로 대체")
                final_df[col].fillna(col_mean, inplace=True)
        
        # 강수량의 경우 0으로 대체
        if 'tp' in final_df.columns and final_df['tp'].isnull().sum() > 0:
            print("tp(강수량) 결측치를 0으로 대체")
            final_df['tp'].fillna(0, inplace=True)
    
    # af_flag 값 정리
    print("\n타겟 데이터 처리 중...")
    final_df['af_flag'] = final_df['af_flag'].fillna(0).astype('uint8')
    
    # U10, V10 풍속 성분을 결합하여 10m 풍속 크기(wind10m) 계산
    if '10u' in final_df.columns and '10v' in final_df.columns:
        print("U10, V10 풍속 성분을 결합하여 10m 풍속 크기(wind10m) 계산")
        final_df['wind10m'] = np.sqrt(final_df['10u']**2 + final_df['10v']**2)
        print(f"wind10m 변수 추가 완료: 범위 {final_df['wind10m'].min():.2f} ~ {final_df['wind10m'].max():.2f} m/s")
    
    # 최종 데이터 통계
    print("\n최종 데이터 통계:")
    print(f"데이터 크기: {final_df.shape}")
    print(f"양성 샘플 수: {final_df.af_flag.sum()} ({final_df.af_flag.mean()*100:.2f}%)")
    
    # 파일명에 날짜 범위 추가
    base_name, ext = os.path.splitext(output_file)
    output_file_with_date = f"{base_name}_{date_range}{ext}"
    print(f"\n출력 파일명에 날짜 범위 추가: {output_file_with_date}")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(os.path.abspath(output_file_with_date)), exist_ok=True)
    
    # 파일 확장자 무시하고 기본 경로 추출
    base_output = os.path.splitext(output_file_with_date)[0]
    
    # CSV와 Parquet 두 형식으로 저장
    # CSV 형식으로 저장
    csv_output = f"{base_output}.csv"
    print(f"\nCSV 형식으로 저장 중: {csv_output}")
    final_df.to_csv(csv_output, index=False)
    
    # Parquet 형식으로 저장
    parquet_output = f"{base_output}.parquet"
    print(f"Parquet 형식으로 저장 중: {parquet_output}")
    final_df.to_parquet(parquet_output, index=False)
    
    print("두 가지 형식 모두 저장 완료")
    
    print("\n=== 날씨 데이터와 타겟 데이터 병합 완료 ===")
    
    # 메모리 정리
    del weather_df
    del target_df
    del final_df
    gc.collect()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='날씨 데이터와 타겟 데이터(af_flag) 병합')
    parser.add_argument('--weather', type=str, required=True,
                        help='전처리된 날씨 데이터 파일 경로 (Parquet 형식)')
    parser.add_argument('--target', type=str, required=True,
                        help='타겟 데이터(af_flag) 파일 경로 (CSV 형식)')
    parser.add_argument('--output', type=str, required=True,
                        help='병합 결과 저장 파일 경로 (.parquet 또는 .csv)')
    parser.add_argument('--join_type', type=str, default='inner', choices=['inner', 'right'],
                        help='조인 유형 (기본값: inner)')
    
    args = parser.parse_args()
    
    try:
        success = join_datasets(args.weather, args.target, args.output, args.join_type)
        return 0 if success else 1
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main()) 