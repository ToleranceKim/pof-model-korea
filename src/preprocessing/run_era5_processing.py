#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import subprocess
from datetime import datetime

def run_era5_pipeline(input_dir, output_dir, start_date=None, end_date=None, parallel=False, file_pattern='*.nc'):
    """
    ERA5 데이터 처리 파이프라인을 실행하는 함수
    
    Args:
        input_dir (str): 입력 디렉토리 경로 (원본 ERA5 파일이 있는 곳)
        output_dir (str): 출력 디렉토리 경로
        start_date (str): 시작 날짜 (YYYYMM 형식), 없으면 모든 파일 처리
        end_date (str): 종료 날짜 (YYYYMM 형식), 없으면 모든 파일 처리
        parallel (bool): 병렬 처리 여부
        file_pattern (str): 파일 패턴 (기본값: *.nc)
    
    Returns:
        bool: 성공 여부
    """
    print("="*80)
    print(f"ERA5 데이터 처리 파이프라인 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. 디렉토리 생성
    interp_dir = os.path.join(output_dir, 'era5_interp')
    os.makedirs(interp_dir, exist_ok=True)
    
    # 2. 입력 파일 목록 가져오기
    input_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not input_files:
        print(f"입력 디렉토리에 파일이 없습니다: {input_dir} (패턴: {file_pattern})")
        return False
    
    print(f"처리할 파일 {len(input_files)}개 발견")
    
    # 날짜 필터링
    if start_date or end_date:
        filtered_files = []
        
        # 날짜 범위 설정
        start = start_date if start_date else "000000"
        end = end_date if end_date else "999999"
        
        for file_path in input_files:
            file_name = os.path.basename(file_path)
            # 파일명에서 날짜 부분 추출 (예: era5_korea_202001.nc -> 202001)
            date_part = ""
            parts = file_name.split('_')
            if len(parts) > 2:  # era5_korea_YYYYMM.nc 형식 가정
                date_part = parts[2].split('.')[0]
            
            if start <= date_part <= end:
                filtered_files.append(file_path)
        
        input_files = filtered_files
        print(f"날짜 필터링 후 처리할 파일: {len(input_files)}개")
    
    if not input_files:
        print("필터링 후 처리할 파일이 없습니다.")
        return False
    
    # 3. 각 파일에 대해 process_weather_interp.py 실행
    processed_files = []
    
    for i, file_path in enumerate(input_files):
        file_name = os.path.basename(file_path)
        print(f"\n[{i+1}/{len(input_files)}] 처리 중: {file_name}")
        
        # 이미 처리된 파일인지 확인
        date_part = file_name.split('_')[2].split('.')[0] if len(file_name.split('_')) > 2 else "unknown"
        output_csv = os.path.join(interp_dir, f'era5_interp_{date_part}.csv')
        
        if os.path.exists(output_csv):
            print(f"이미 처리된 파일입니다: {output_csv}")
            processed_files.append(output_csv)
            continue
        
        # process_weather_interp.py 실행
        try:
            cmd = [sys.executable, 'src/preprocessing/process_weather_interp.py', file_path, interp_dir]
            print(f"실행: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(result.stdout)
            
            if os.path.exists(output_csv):
                processed_files.append(output_csv)
                print(f"처리 완료: {output_csv}")
            else:
                print(f"처리 실패: 결과 파일이 생성되지 않았습니다 - {output_csv}")
        
        except subprocess.CalledProcessError as e:
            print(f"처리 중 오류 발생: {str(e)}")
            print(f"표준 출력: {e.stdout}")
            print(f"오류 출력: {e.stderr}")
    
    # 4. 모든 처리가 완료되면 combine_era5_data.py 실행
    if processed_files:
        print("\n"+"="*80)
        print("모든 파일 처리 완료. 데이터 결합 시작...")
        
        try:
            cmd = [sys.executable, 'src/preprocessing/combine_era5_data.py', interp_dir, output_dir]
            
            # 날짜 범위 추가
            if start_date and end_date:
                cmd.extend([start_date, end_date])
            
            print(f"실행: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(result.stdout)
            
            # 결과 확인
            expected_output = os.path.join(output_dir, 'era5_korea.parquet')
            if os.path.exists(expected_output) or glob.glob(os.path.join(output_dir, 'era5_korea_*.parquet')):
                print("데이터 결합 완료!")
            else:
                print("데이터 결합 실패: 결과 파일이 생성되지 않았습니다.")
        
        except subprocess.CalledProcessError as e:
            print(f"데이터 결합 중 오류 발생: {str(e)}")
            print(f"표준 출력: {e.stdout}")
            print(f"오류 출력: {e.stderr}")
            return False
    
    else:
        print("처리된 파일이 없어 데이터 결합을 건너뜁니다.")
        return False
    
    print("\n"+"="*80)
    print(f"ERA5 데이터 처리 파이프라인 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="ERA5 데이터 처리 파이프라인")
    parser.add_argument("input_dir", help="입력 디렉토리 (원본 ERA5 파일 위치)")
    parser.add_argument("output_dir", help="출력 디렉토리")
    parser.add_argument("--start", help="시작 날짜 (YYYYMM 형식)")
    parser.add_argument("--end", help="종료 날짜 (YYYYMM 형식)")
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 활성화")
    parser.add_argument("--pattern", default="*.nc", help="파일 패턴 (기본값: *.nc)")
    
    args = parser.parse_args()
    
    run_era5_pipeline(args.input_dir, args.output_dir, args.start, args.end, args.parallel, args.pattern)

if __name__ == "__main__":
    main() 