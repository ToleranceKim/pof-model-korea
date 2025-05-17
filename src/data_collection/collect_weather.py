#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cdsapi
import calendar
import os
import sys
import argparse
from datetime import datetime

def collect_weather(start_year, end_year, start_month, end_month, output_dir):
    """
    ERA5-Land 날씨 데이터를 수집합니다.
    
    Parameters:
    -----------
    start_year : int
        시작 연도
    end_year : int
        종료 연도
    start_month : int
        시작 월 (1-12)
    end_month : int
        종료 월 (1-12)
    output_dir : str
        출력 디렉토리 경로
    
    Returns:
    --------
    list : 다운로드된 파일 경로 목록
    """
    print("=== 날씨 데이터 수집 시작 ===")
    
    try:
        c = cdsapi.Client()
        
        # 시작-종료 년월 범위 내의 모든 년월 조합 생성
        year_month_pairs = []
        for year in range(start_year, end_year + 1):
            month_start = start_month if year == start_year else 1
            month_end = end_month if year == end_year else 12
            
            for month in range(month_start, month_end + 1):
                year_month_pairs.append((year, month))
        
        print(f"수집 기간: {start_year}-{start_month:02d} ~ {end_year}-{end_month:02d} ({len(year_month_pairs)} 개월)")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        downloaded_files = []
        
        # BBOX 확장: 북위 39도까지 포함
        bbox = [39, 124, 33, 132]  # [북위, 서경, 남위, 동경]
        print(f"지역 범위: 북위 {bbox[0]}-{bbox[2]}도, 동경 {bbox[3]}-{bbox[1]}도")
        
        for year, month in year_month_pairs:
            # 해당 월의 실제 일수만 추리기
            max_day = calendar.monthrange(year, month)[1]
            day_list = [f"{d:02d}" for d in range(1, max_day+1)]
            
            # 시간 목록
            times = [f"{h:02d}:00" for h in range(0,24)]
            
            target_file = os.path.join(output_dir, f"era5_korea_{year}{month:02d}.nc")
            print(f"Retrieving {target_file} ...")
            
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': [
                        '2m_temperature','2m_dewpoint_temperature',
                        '10m_u_component_of_wind','10m_v_component_of_wind',
                        'total_precipitation',
                    ],
                    'product_type': 'reanalysis',
                    'year':   [f"{year}"],
                    'month':  [f"{month:02d}"],
                    'day':    day_list,
                    'time':   times,
                    'area':   bbox,
                    'format': 'netcdf'
                },
                target_file
            )
            print(f"Successfully downloaded {target_file}")
            downloaded_files.append(target_file)
        
        print("=== 날씨 데이터 수집 완료 ===")
        return downloaded_files
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='ERA5-Land 날씨 데이터 수집')
    
    # 현재 연도와 월 구하기
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # 명령행 인자 정의
    parser.add_argument('--start_year', type=int, default=current_year,
                        help='시작 연도 (기본값: 현재 연도)')
    parser.add_argument('--end_year', type=int, default=current_year,
                        help='종료 연도 (기본값: 현재 연도)')
    parser.add_argument('--start_month', type=int, default=1,
                        help='시작 월, 1-12 (기본값: 1)')
    parser.add_argument('--end_month', type=int, default=current_month,
                        help='종료 월, 1-12 (기본값: 현재 월)')
    parser.add_argument('--output_dir', type=str, default='../../data/raw',
                        help='출력 디렉토리 경로 (기본값: ../../data/raw)')
    
    args = parser.parse_args()
    
    # 인자 유효성 검사
    if not (1 <= args.start_month <= 12 and 1 <= args.end_month <= 12):
        print("오류: 월은 1에서 12 사이의 값이어야 합니다.")
        return 1
    
    if args.start_year > args.end_year:
        print("오류: 시작 연도는 종료 연도보다 작거나 같아야 합니다.")
        return 1
    
    if args.start_year == args.end_year and args.start_month > args.end_month:
        print("오류: 같은 해에서는 시작 월이 종료 월보다 작거나 같아야 합니다.")
        return 1
    
    # 데이터 수집 실행
    files = collect_weather(
        args.start_year, args.end_year,
        args.start_month, args.end_month,
        args.output_dir
    )
    
    if not files:
        print("날씨 데이터 수집에 실패했습니다.")
        return 1
    
    print(f"총 {len(files)}개 파일이 다운로드되었습니다.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 