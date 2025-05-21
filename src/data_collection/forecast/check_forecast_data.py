#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from datetime import datetime, timedelta
import traceback

def check_grib_file(file_path):
    """
    GRIB2 파일 메타데이터를 분석하여 예보 데이터 정보를 확인합니다.
    
    Parameters:
    -----------
    file_path : str
        검사할 GRIB2 파일 경로
    """
    try:
        # eccodes 라이브러리 임포트 시도
        try:
            from eccodes import codes_grib_new_from_file, codes_get, codes_release
        except ImportError:
            print("오류: 'eccodes' 패키지가 설치되어 있지 않습니다.")
            print("아래 명령어로 패키지를 설치하세요:")
            print("pip install eccodes")
            print("또는 conda install -c conda-forge eccodes python-eccodes")
            return False
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            print(f"오류: 파일이 존재하지 않습니다: {file_path}")
            return False
            
        print(f"===== 파일 분석: {os.path.basename(file_path)} =====")
        print(f"파일 경로: {file_path}")
        print(f"파일 크기: {os.path.getsize(file_path) / 1024:.2f} KB")
        print(f"수정 시간: {datetime.fromtimestamp(os.path.getmtime(file_path))}")
        print()
        
        # GRIB 파일 열기
        with open(file_path, 'rb') as f:
            gid = codes_grib_new_from_file(f)
            
            if gid is None:
                print("오류: GRIB 메시지를 읽을 수 없습니다.")
                return False
                
            try:
                # 기본 메타데이터 추출
                centre = codes_get(gid, 'centre')
                date = codes_get(gid, 'date')
                time = codes_get(gid, 'time')
                step = codes_get(gid, 'step')
                dataType = codes_get(gid, 'dataType')
                gridType = codes_get(gid, 'gridType')
                
                # 변수 관련 메타데이터
                shortName = codes_get(gid, 'shortName')
                name = codes_get(gid, 'name')
                units = codes_get(gid, 'units')
                
                # 날짜 계산
                basedate = datetime.strptime(f"{date}{time:04d}", "%Y%m%d%H%M")
                valid_date = basedate + timedelta(hours=step)
                
                # 결과 출력
                print("===== 메타데이터 정보 =====")
                print(f"데이터 소스(센터): {centre}")
                print(f"기준 날짜/시각: {basedate}")
                print(f"예보 시간(step): +{step}시간 (D+{step//24})")
                print(f"유효 날짜/시각: {valid_date}")
                print(f"데이터 타입: {dataType}")
                print(f"그리드 타입: {gridType}")
                print()
                
                print("===== 변수 정보 =====")
                print(f"변수 코드: {shortName}")
                print(f"변수 이름: {name}")
                print(f"단위: {units}")
                print()
                
                # 예보 데이터인지 확인
                is_forecast = dataType.lower() in ['fc', 'forecast'] and step > 0
                
                print("===== 결론 =====")
                if is_forecast:
                    days_ahead = step // 24
                    hours_remain = step % 24
                    ahead_str = f"{days_ahead}일"
                    if hours_remain > 0:
                        ahead_str += f" {hours_remain}시간"
                        
                    print(f"이 파일은 {ahead_str} 후의 미래 예보 데이터입니다.")
                    print(f"기준 시점: {basedate}")
                    print(f"예측 시점: {valid_date}")
                else:
                    print(f"이 파일은 미래 예보 데이터가 아닌 것으로 보입니다.")
                    if dataType.lower() == 'an' or dataType.lower() == 'analysis':
                        print("이 파일은 분석(analysis) 데이터입니다.")
                    
                return is_forecast
                
            finally:
                # 리소스 해제
                codes_release(gid)
                
    except Exception as e:
        print(f"오류 발생: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='GRIB2 예보 데이터 파일 검사')
    
    parser.add_argument('file', type=str, help='검사할 GRIB2 파일 경로')
    
    args = parser.parse_args()
    
    # 파일 검사 실행
    is_forecast = check_grib_file(args.file)
    
    # 종료 코드 설정
    return 0 if is_forecast else 1

if __name__ == "__main__":
    sys.exit(main()) 