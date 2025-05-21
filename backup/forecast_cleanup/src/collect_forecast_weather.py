#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path

def collect_forecast_weather(target_date, output_dir, variables=None, steps=None):
    """
    ECMWF 예보 데이터를 수집합니다.
    
    Parameters:
    -----------
    target_date : datetime
        수집할 예보의 기준 날짜
    output_dir : str
        출력 디렉토리 경로
    variables : list
        수집할 변수 목록 (기본값: 2t, 2d, u10, v10, tp, lightning)
    steps : list
        수집할 예보 시간(시간) 목록 (기본값: 24시간 간격으로 24~240시간, D+1에서 D+10)
    
    Returns:
    --------
    list : 다운로드된 파일 경로 목록
    """
    print("=== ECMWF 예보 데이터 수집 시작 ===")
    
    try:
        # ecmwf-opendata 패키지 임포트 시도
        try:
            from ecmwf.opendata import Client
        except ImportError:
            print("오류: 'ecmwf-opendata' 패키지가 설치되어 있지 않습니다.")
            print("아래 명령어로 패키지를 설치하세요:")
            print("pip install ecmwf-opendata")
            return []
            
        # 기본 변수 설정
        if variables is None:
            variables = ["2t", "2d", "u10", "v10", "tp", "lightning"]
            
        # 기본 예보 시간 설정 (24시간 간격으로 1-10일)
        if steps is None:
            steps = [24 * i for i in range(1, 11)]  # 24, 48, 72, ..., 240
        
        # 날짜 형식 변환
        date_str = target_date.strftime("%Y-%m-%d")
        
        # 출력 디렉토리가 절대 경로가 아니면 절대 경로로 변환
        output_dir = os.path.abspath(output_dir)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        print(f"출력 디렉토리: {output_dir}")
        
        # 클라이언트 초기화
        client = Client(source="ecmwf")
        print("ECMWF Open Data 클라이언트 초기화 성공")
        
        # 예보 시각 (00, 06, 12, 18 중 가장 최근)
        forecast_hours = [0, 6, 12, 18]
        current_hour = target_date.hour
        forecast_hour = max([h for h in forecast_hours if h <= current_hour], default=forecast_hours[-1])
        
        print(f"수집 날짜: {date_str}")
        print(f"예보 시각: {forecast_hour:02d}Z")
        print(f"수집 변수: {', '.join(variables)}")
        print(f"예보 시간: {', '.join([f'+{step}h (D+{step//24})' for step in steps])}")
        
        downloaded_files = []
        
        # 각 변수별로 데이터 다운로드
        for variable in variables:
            # 변수별 디렉토리 생성
            var_dir = os.path.join(output_dir, variable)
            os.makedirs(var_dir, exist_ok=True)
            
            for step in steps:
                # 파일명 생성
                target_file = os.path.join(var_dir, f"{variable}_{date_str}_{forecast_hour:02d}z_step{step:03d}.grib2")
                
                print(f"다운로드 중: {variable}, 예보 시간: +{step}h, 출력: {target_file}")
                
                try:
                    # ECMWF Open Data에서 데이터 다운로드
                    client.retrieve(
                        date=date_str,
                        time=forecast_hour,
                        step=step,
                        stream="oper",   # 고해상도 운영예보
                        type="fc",       # forecast
                        param=variable,
                        target=target_file
                    )
                    print(f"다운로드 완료: {target_file}")
                    downloaded_files.append(target_file)
                    
                except Exception as e:
                    print(f"변수 '{variable}', 예보 시간 +{step}h 다운로드 중 오류 발생: {e}")
                    traceback.print_exc()
        
        print(f"=== ECMWF 예보 데이터 수집 완료 ({len(downloaded_files)}/{len(variables) * len(steps)} 파일) ===")
        return downloaded_files
        
    except Exception as e:
        print(f"오류 발생: {e}")
        traceback.print_exc()
        return []

def test_forecast_availability(target_date=None, variable="2t", step=24):
    """
    ECMWF 예보 데이터의 가용성을 테스트합니다.
    
    Parameters:
    -----------
    target_date : datetime
        테스트할 예보의 기준 날짜 (기본값: 현재 날짜)
    variable : str
        테스트할 변수 (기본값: '2t', 2m 기온)
    step : int
        테스트할 예보 시간(시간) (기본값: 24, D+1)
        
    Returns:
    --------
    bool : 테스트 성공 여부
    """
    print("=== ECMWF 예보 데이터 가용성 테스트 ===")
    
    try:
        # ecmwf-opendata 패키지 임포트 시도
        try:
            from ecmwf.opendata import Client
        except ImportError:
            print("오류: 'ecmwf-opendata' 패키지가 설치되어 있지 않습니다.")
            print("아래 명령어로 패키지를 설치하세요:")
            print("pip install ecmwf-opendata")
            return False
            
        # 날짜가 지정되지 않은 경우 현재 날짜 사용
        if target_date is None:
            target_date = datetime.now()
            
        # 날짜 형식 변환
        date_str = target_date.strftime("%Y-%m-%d")
        
        # 예보 시각 (00, 06, 12, 18 중 가장 최근)
        forecast_hours = [0, 6, 12, 18]
        current_hour = target_date.hour
        forecast_hour = max([h for h in forecast_hours if h <= current_hour], default=forecast_hours[-1])
        
        print(f"테스트 날짜: {date_str}")
        print(f"예보 시각: {forecast_hour:02d}Z")
        print(f"테스트 변수: {variable}")
        print(f"예보 시간: +{step}h (D+{step//24})")
        
        # 임시 파일 경로 생성
        import tempfile
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, f"ecmwf_test_{date_str}_{forecast_hour:02d}z_{variable}_step{step:03d}.grib2")
        print(f"테스트 파일: {test_file}")
        
        # 클라이언트 초기화
        client = Client(source="ecmwf")
        print("ECMWF Open Data 클라이언트 초기화 성공")
        
        # 데이터 요청 전송
        print("데이터 요청 시작...")
        client.retrieve(
            date=date_str,
            time=forecast_hour,
            step=step,
            stream="oper",   # 고해상도 운영예보
            type="fc",       # forecast
            param=variable,
            target=test_file
        )
        
        # 파일 존재 여부 확인
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            print(f"\n테스트 성공! 데이터 요청이 정상적으로 처리되었습니다.")
            print(f"다운로드된 테스트 파일: {test_file}")
            print(f"파일 크기: {file_size/1024:.2f} KB")
            
            # 테스트 파일 삭제 여부 확인
            delete_file = input("테스트 파일을 삭제하시겠습니까? (y/n): ").lower() == 'y'
            if delete_file:
                os.remove(test_file)
                print(f"테스트 파일 삭제 완료")
            else:
                print(f"테스트 파일 유지: {test_file}")
                
            return True
        else:
            print(f"\n테스트 실패: 파일이 생성되지 않았습니다.")
            return False
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='ECMWF 예보 데이터 수집')
    
    # 현재 날짜
    current_date = datetime.now()
    
    # 명령행 인자 정의
    parser.add_argument('--date', type=str, default=current_date.strftime("%Y-%m-%d"),
                        help='예보 기준 날짜, YYYY-MM-DD 형식 (기본값: 현재 날짜)')
    parser.add_argument('--variables', type=str, nargs='+', 
                        default=["2t", "2d", "u10", "v10", "tp", "lightning"],
                        help='수집할 변수 목록 (기본값: 2t 2d u10 v10 tp lightning)')
    parser.add_argument('--start_step', type=int, default=24,
                        help='시작 예보 시간(시간) (기본값: 24, D+1)')
    parser.add_argument('--end_step', type=int, default=240,
                        help='종료 예보 시간(시간) (기본값: 240, D+10)')
    parser.add_argument('--step_interval', type=int, default=24,
                        help='예보 시간 간격(시간) (기본값: 24)')
    parser.add_argument('--output_dir', type=str, default='../../data/forecast',
                        help='출력 디렉토리 경로 (기본값: ../../data/forecast)')
    parser.add_argument('--test', action='store_true',
                        help='데이터 가용성 테스트만 수행 (기본 변수: 2t, 기본 예보 시간: 24h)')
    parser.add_argument('--test_variable', type=str, default='2t',
                        help='테스트할 변수 (기본값: 2t)')
    parser.add_argument('--test_step', type=int, default=24,
                        help='테스트할 예보 시간(시간) (기본값: 24)')
    
    args = parser.parse_args()
    
    # 날짜 파싱
    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print("오류: 날짜 형식은 YYYY-MM-DD여야 합니다.")
        return 1
        
    # 테스트 모드
    if args.test:
        success = test_forecast_availability(target_date, args.test_variable, args.test_step)
        return 0 if success else 1
        
    # 예보 시간 목록 생성
    steps = list(range(args.start_step, args.end_step + 1, args.step_interval))
    
    # 데이터 수집 실행
    files = collect_forecast_weather(target_date, args.output_dir, args.variables, steps)
    
    if not files:
        print("ECMWF 예보 데이터 수집에 실패했습니다.")
        return 1
        
    print(f"총 {len(files)}개 파일이 다운로드되었습니다.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 