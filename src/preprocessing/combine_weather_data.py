#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import glob
import argparse
from tqdm import tqdm
import gc

def combine_weather_files(input_dir, output_file):
    """
    processed_data 디렉토리의 모든 월별 날씨 데이터 파일을 하나로 결합합니다.
    
    Parameters:
    -----------
    input_dir : str
        입력 디렉토리 경로 (processed_data)
    output_file : str
        출력 파일 경로
    """
    print("\n=== 날씨 데이터 결합 시작 ===")
    
    # 파일 목록 가져오기
    pattern = os.path.join(input_dir, "era5_korea_*.parquet")
    file_list = glob.glob(pattern)
    
    if not file_list:
        raise FileNotFoundError(f"날씨 데이터 파일을 찾을 수 없습니다: {pattern}")
    
    print(f"총 {len(file_list)}개의 날씨 데이터 파일을 결합합니다.")
    
    # 날짜순으로 정렬
    file_list.sort()
    
    # 데이터프레임을 누적하여 결합
    dfs = []
    for file_path in tqdm(file_list, desc="파일 로드 중"):
        df = pd.read_parquet(file_path)
        dfs.append(df)
        print(f"로드 완료: {file_path}, 형태: {df.shape}")
    
    # 모든 데이터프레임 결합
    print("모든 파일 결합 중...")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"결합 완료. 최종 형태: {combined_df.shape}")
    
    # 중복 제거
    initial_rows = combined_df.shape[0]
    combined_df = combined_df.drop_duplicates()
    duplicate_rows = initial_rows - combined_df.shape[0]
    print(f"중복 행 제거: {duplicate_rows}개 ({duplicate_rows/initial_rows*100:.2f}%)")
    
    # 메모리 정리
    del dfs
    gc.collect()
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 파일 확장자에 따라 저장 형식 결정
    if output_file.endswith('.csv'):
        print(f"\nCSV 형식으로 저장 중: {output_file}")
        combined_df.to_csv(output_file, index=False)
    else:
        output_file = output_file if output_file.endswith('.parquet') else f"{output_file}.parquet"
        print(f"\nParquet 형식으로 저장 중: {output_file}")
        combined_df.to_parquet(output_file, index=False)
    
    print("=== 날씨 데이터 결합 완료 ===")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='월별 날씨 데이터 파일 결합')
    parser.add_argument('--input_dir', type=str, default='processed_data',
                      help='입력 디렉토리 경로 (기본값: processed_data)')
    parser.add_argument('--output_file', type=str, default='processed_data/era5_korea_combined.parquet',
                      help='결합된 결과 저장 파일 경로 (기본값: processed_data/era5_korea_combined.parquet)')

    args = parser.parse_args()
    
    combine_weather_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main() 