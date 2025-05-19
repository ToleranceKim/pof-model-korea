#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cdsapi
import calendar
import os
import sys
import argparse
import traceback
import pprint
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

def try_load_dotenv_with_encodings():
    """
    여러 인코딩을 시도하여 .env 파일을 로드합니다.
    """
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp949', 'euc-kr', 'latin-1']
    env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env_file = env_path / '.env'
    
    print(f".env 파일 경로: {env_file}")
    
    if not env_file.exists():
        print(f"오류: .env 파일이 {env_file}에 존재하지 않습니다.")
        return False
    
    # 파일의 바이너리 내용 확인 (디버깅용)
    try:
        with open(env_file, 'rb') as f:
            content = f.read(20)  # 처음 20바이트만 읽음
            print(f".env 파일 헤더(16진수): {content.hex()}")
    except Exception as e:
        print(f"파일 헤더 읽기 실패: {e}")
    
    # 여러 인코딩으로 시도
    env_vars = {}
    for encoding in encodings:
        try:
            print(f"인코딩 '{encoding}'으로 시도 중...")
            with open(env_file, 'r', encoding=encoding) as f:
                content = f.read()
                
            # 직접 환경 변수 파싱
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                env_vars[key] = value
                
            print(f"인코딩 '{encoding}'으로 .env 파일 로드 성공")
            return True
        except UnicodeDecodeError:
            print(f"인코딩 '{encoding}'으로 읽기 실패")
        except Exception as e:
            print(f"인코딩 '{encoding}'으로 시도 중 오류: {e}")
    
    print("모든 인코딩 시도 실패")
    return False

def create_cdsapirc_from_env():
    """
    .env 파일의 환경변수를 사용하여 ~/.cdsapirc 파일을 생성합니다.
    이렇게 하면 cdsapi가 자동으로 인증 정보를 찾을 수 있습니다.
    """
    try:
        # .env 파일 로드 (여러 인코딩 시도)
        if not try_load_dotenv_with_encodings():
            print("대체 방법으로 직접 환경변수 입력을 시도합니다.")
            # 직접 입력 코드를 여기에 추가할 수 있음
            return False
        
        # 환경 변수에서 API 정보 가져오기
        cds_api_url = os.getenv('CDS_API_URL')
        cds_api_key = os.getenv('CDS_API_KEY')
        
        print(f"CDS_API_URL: {'설정됨' if cds_api_url else '설정되지 않음'}")
        print(f"CDS_API_KEY: {'설정됨' if cds_api_key else '설정되지 않음'}")
        
        if not cds_api_url or not cds_api_key:
            print("오류: CDS_API_URL 또는 CDS_API_KEY 환경변수가 설정되지 않았습니다.")
            print(".env 파일이 프로젝트 루트 디렉토리에 있는지 확인하고, 필요한 변수가 설정되어 있는지 확인하세요.")
            return False
        
        # ~/.cdsapirc 파일 경로 구성
        home_dir = str(Path.home())
        cdsapirc_path = os.path.join(home_dir, '.cdsapirc')
        
        # 파일 내용 구성
        content = f"url: {cds_api_url}\nkey: {cds_api_key}\n"
        
        # 기존 파일 확인
        if os.path.exists(cdsapirc_path):
            print(f"기존 {cdsapirc_path} 파일이 존재합니다.")
            # 기존 파일 내용 확인
            try:
                with open(cdsapirc_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                    print(f"기존 파일 내용:\n{existing_content}")
            except Exception as e:
                print(f"기존 파일 읽기 실패: {e}")
        
        # 파일에 내용 쓰기
        with open(cdsapirc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"CDS API 설정이 {cdsapirc_path}에 작성되었습니다.")
        
        # 생성된 파일 확인
        try:
            with open(cdsapirc_path, 'r', encoding='utf-8') as f:
                written_content = f.read()
                print(f"작성된 파일 내용:\n{written_content}")
        except Exception as e:
            print(f"작성된 파일 읽기 실패: {e}")
            
        return True
    except Exception as e:
        print(f"~/.cdsapirc 파일 생성 중 오류 발생: {e}")
        print(f"상세 오류: {traceback.format_exc()}")
        return False

def create_env_file():
    """
    사용자 입력으로 .env 파일을 새로 생성합니다.
    """
    try:
        print("\n=== .env 파일 새로 생성하기 ===")
        env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        env_file = env_path / '.env'
        
        cds_api_url = input("CDS API URL을 입력하세요 (기본값: https://cds.climate.copernicus.eu/api/v2): ")
        if not cds_api_url:
            cds_api_url = "https://cds.climate.copernicus.eu/api/v2"
        
        cds_api_key = input("CDS API KEY를 입력하세요: ")
        if not cds_api_key:
            print("API 키는 필수입니다.")
            return False
        
        # 파일에 내용 쓰기
        content = f"CDS_API_URL={cds_api_url}\nCDS_API_KEY={cds_api_key}\n"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f".env 파일이 {env_file}에 생성되었습니다.")
        return True
    except Exception as e:
        print(f".env 파일 생성 중 오류 발생: {e}")
        print(f"상세 오류: {traceback.format_exc()}")
        return False

def test_lightning_availability(year, month, day, variable_name):
    """
    번개 데이터의 가용성을 테스트합니다.
    
    Parameters:
    -----------
    year : int
        테스트할 연도
    month : int
        테스트할 월 (1-12)
    day : int
        테스트할 일
    variable_name : str
        테스트할 변수명
        
    Returns:
    --------
    bool : 테스트 성공 여부
    """
    try:
        # .env에서 CDS API 설정 생성
        if not create_cdsapirc_from_env():
            return False
            
        print(f"\n=== {year}년 {month:02d}월 {day:02d}일 번개 데이터 가용성 테스트 ===")
        print(f"테스트 변수: {variable_name}")
        
        try:
            c = cdsapi.Client(debug=True)
            print("CDS API 클라이언트 초기화 성공")
        except Exception as e:
            print(f"CDS API 클라이언트 초기화 중 오류 발생: {e}")
            print(f"상세 오류: {traceback.format_exc()}")
            return False
        
        # 임시 파일 경로 생성
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, f"cds_test_{year}{month:02d}{day:02d}_{variable_name}.nc")
        
        print(f"API 요청 전송 중... (최소 데이터셋 - 단일 시간점)")
        print(f"요청 파라미터:")
        request_params = {
            'variable': [variable_name],
            'product_type': 'reanalysis',
            'year': f'{year}',
            'month': f'{month:02d}',
            'day': f'{day:02d}',
            'time': ['00:00'],  # 단일 시간만 요청
            'area': [39, 124, 33, 132],
            'format': 'netcdf'
        }
        pprint.pprint(request_params)
        
        try:
            print(f"테스트 파일: {test_file}")
            print("데이터 요청 시작 (취소할 경우 Ctrl+C를 누르세요)...")
            
            # 실제 요청 전송 - 성공 여부만 확인
            result = c.retrieve(
                'reanalysis-era5-single-levels',
                request_params,
                test_file
            )
            
            print("\n✅ 테스트 성공! 데이터 요청이 정상적으로 처리되었습니다.")
            print(f"✅ {variable_name} 변수는 {year}년 {month:02d}월 {day:02d}일에 사용 가능합니다.")
            
            # 테스트 파일 정보 출력
            if os.path.exists(test_file):
                file_size = os.path.getsize(test_file)
                print(f"테스트 파일 크기: {file_size/1024:.2f} KB")
                
                # 테스트 파일 삭제 여부 확인
                delete_file = input("테스트 파일을 삭제하시겠습니까? (y/n): ").lower() == 'y'
                if delete_file:
                    os.remove(test_file)
                    print(f"테스트 파일 삭제 완료: {test_file}")
                else:
                    print(f"테스트 파일 유지: {test_file}")
            
            print("\n실제 데이터 수집을 진행하려면 --test 옵션 없이 명령을 실행하세요.")
            return True
            
        except Exception as e:
            print(f"\n❌ API 요청 실패: {e}")
            print(f"상세 오류: {traceback.format_exc()}")
            
            if "Cannot load dictionary values into a string" in str(e) or "ResourceNotFound" in str(e):
                print("\n💡 힌트: 변수명이 정확하지 않을 수 있습니다.")
                print("다음 변수명을 시도해보세요:")
                print("- 'litoti' (변수 약어)")
                print("- 'total_lightning_flash_density' (변수 전체 이름)")
            elif "FileNotFoundError" in str(e):
                print("\n💡 힌트: 요청한 날짜에 데이터가 없을 수 있습니다. 더 과거 날짜를 시도해보세요.")
            
            # 일부 성공한 경우 임시 파일 정리
            if os.path.exists(test_file):
                try:
                    os.remove(test_file)
                    print(f"임시 파일 정리: {test_file}")
                except:
                    pass
                    
            return False
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        print(f"상세 오류: {traceback.format_exc()}")
        return False

def collect_lightning(start_year, end_year, start_month, end_month, output_dir):
    """
    ERA5 단층(single-levels) 번개 데이터를 수집합니다.
    
    수집 변수: 'litoti' (Instantaneous total lightning flash density)
    단위: km-2 day-1 (제곱킬로미터당 하루에 발생하는 번개 수)
    GRIB 파라미터 ID: 228050
    
    참고: 이 파라미터는 지정된 시간의 총 번개 발생률을 제공합니다.
    구름-지상 번개(cloud-to-ground)와 구름 내 번개(intra-cloud) 모두 포함됩니다.
    
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
    print("=== 번개 데이터 수집 시작 ===")
    
    try:
        # .env에서 CDS API 설정 생성
        if not create_cdsapirc_from_env():
            return []
        
        print("CDS API 클라이언트 초기화 중...")    
        try:
            c = cdsapi.Client(debug=True)
            print("CDS API 클라이언트 초기화 성공")
        except Exception as e:
            print(f"CDS API 클라이언트 초기화 중 오류 발생: {e}")
            print(f"상세 오류: {traceback.format_exc()}")
            return []
        
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
            
            target_file = os.path.join(output_dir, f"era5_ltg_{year}{month:02d}.nc")
            print(f"Retrieving {target_file} ...")
            
            try:
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'variable': ['litoti'],  # ECMWF 파라미터 DB에 따른 정확한 변수명
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
            except Exception as e:
                print(f"{year}년 {month}월 데이터 다운로드 중 오류 발생: {e}")
                print(f"상세 오류: {traceback.format_exc()}")
                continue
        
        print("=== 번개 데이터 수집 완료 ===")
        return downloaded_files
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print(f"상세 오류: {traceback.format_exc()}")
        return []

def main():
    parser = argparse.ArgumentParser(description='ERA5 단층(single-levels) 번개 데이터 수집 (변수: litoti)')
    
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
    parser.add_argument('--check_config', action='store_true',
                        help='CDS API 설정만 확인하고 종료')
    parser.add_argument('--create_env', action='store_true',
                        help='.env 파일을 새로 생성')
    parser.add_argument('--test', action='store_true',
                        help='변수 가용성 테스트만 수행 (데이터 다운로드 없음)')
    parser.add_argument('--var', type=str, default='litoti',
                        help='테스트할 변수명 (기본값: litoti)')
    parser.add_argument('--test_date', type=str,
                        help='테스트할 날짜 (YYYY-MM-DD 형식)')
    
    args = parser.parse_args()
    
    # .env 파일 새로 생성
    if args.create_env:
        if create_env_file():
            print(".env 파일이 성공적으로 생성되었습니다.")
            return 0
        else:
            print(".env 파일 생성에 실패했습니다.")
            return 1
    
    # 데이터 가용성 테스트
    if args.test:
        if args.test_date:
            try:
                test_date = datetime.strptime(args.test_date, "%Y-%m-%d")
            except ValueError:
                print("오류: 날짜 형식은 YYYY-MM-DD여야 합니다.")
                return 1
        else:
            # 현재 날짜에서 3개월 전 데이터로 테스트
            test_date = datetime.now() - timedelta(days=90)
        
        success = test_lightning_availability(
            test_date.year, test_date.month, test_date.day, 
            args.var
        )
        return 0 if success else 1
    
    # CDS API 설정만 확인하는 옵션
    if args.check_config:
        success = create_cdsapirc_from_env()
        if success:
            print("CDS API 설정이 정상적으로 확인되었습니다.")
            try:
                c = cdsapi.Client(debug=True)
                print("CDS API 클라이언트 초기화 성공")
                return 0
            except Exception as e:
                print(f"CDS API 클라이언트 초기화 중 오류 발생: {e}")
                print(f"상세 오류: {traceback.format_exc()}")
                return 1
        else:
            print("CDS API 설정 확인에 실패했습니다.")
            print("--create_env 옵션을 사용하여 .env 파일을 새로 생성해 보세요.")
            return 1
    
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
    files = collect_lightning(
        args.start_year, args.end_year,
        args.start_month, args.end_month,
        args.output_dir
    )
    
    if not files:
        print("번개 데이터 수집에 실패했습니다.")
        return 1
    
    print(f"총 {len(files)}개 파일이 다운로드되었습니다.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 