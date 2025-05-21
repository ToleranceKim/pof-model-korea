#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import argparse
import traceback
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def collect_geff_fuel_data(output_dir=None, date_str=None, max_retries=3, retry_delay=5, days=10):
    """
    ECMWF Global ECMWF Fire Forecast (GEFF) 데이터에서
    연료 수분 함량 데이터를 수집합니다.
    
    이 함수는 미래 예측 데이터(D+0에서 D+10, 즉 오늘부터 10일 후까지)를 수집합니다.
    GEFF 데이터는 매일 갱신되며 향후 산불 예측을 위한 입력으로 사용됩니다.
    
    주의: 6일 이후(D+6 ~ D+10)의 데이터는 정확도가 급격히 떨어질 수 있습니다.
    논문 및 이전 연구에 따르면 5일 이후 예보의 정확도가 크게 감소합니다.
    
    중요: GEFF 일별 예측 데이터는 Copernicus Climate Data Store(CDS)에서 직접 접근할 수 없습니다.
    이 데이터에 접근하려면 ECMWF 회원 자격이 필요하며, MARS/WebAPI를 통해 요청해야 합니다.
    또는 EFFIS(EUMETSAT)의 온라인 폼을 통해 요청할 수 있습니다.
    
    참고: 연료 부하량(Fuel Load) 변수는 실시간 GEFF 예보에서 제공되지 않으므로
    별도의 정적 데이터셋에서 수집해야 합니다.
    
    Parameters:
    -----------
    output_dir : str, optional
        출력 디렉토리 경로 (기본값: 프로젝트 루트/data/fuel)
    date_str : str, optional
        수집할 예보의 기준 날짜 (YYYYMMDD 형식, 기본값: 현재 날짜)
    max_retries : int, optional
        다운로드 실패 시 최대 재시도 횟수 (기본값: 3)
    retry_delay : int, optional
        재시도 사이의 대기 시간(초) (기본값: 5)
    days : int, optional
        수집할 예보 일수 (기본값: 10, 최대 10일)
        
    Returns:
    --------
    list : 다운로드된 파일 경로 목록
    """
    logger.info("=== GEFF 연료 수분 함량 데이터 수집 시작 ===")
    logger.info(f"=== 미래 예측 데이터: 오늘부터 {days}일 후까지 (D+0 ~ D+{days}) ===")
    
    if days > 5:
        logger.warning("주의: 6일 이후(D+6 ~ D+10)의 데이터는 정확도가 급격히 떨어질 수 있습니다.")
        logger.warning("프로젝트 산불 예측 정확도를 위해 운영 환경에서는 5일까지의 데이터만 사용하는 것을 권장합니다.")
    
    # 최대 10일로 제한
    days = min(days, 10)
    
    try:
        # ecmwfapi 패키지 임포트 시도
        try:
            from ecmwfapi import ECMWFDataServer
        except ImportError:
            logger.error("'ecmwfapi' 패키지가 설치되어 있지 않습니다.")
            logger.error("설치 명령어: pip install ecmwfapi")
            logger.error("주의: ECMWF 데이터 접근을 위해 API 키와 회원 자격이 필요합니다.")
            logger.error("자세한 내용은 https://www.ecmwf.int/en/forecasts/access-forecasts/ecmwf-web-api 를 참조하세요.")
            return []
        
        # 출력 디렉토리 기본값 설정 (절대 경로 사용)
        if output_dir is None:
            # 프로젝트 루트 디렉토리 찾기
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
            output_dir = os.path.join(project_root, "data/fuel")
        else:
            output_dir = os.path.abspath(output_dir)
        
        # 날짜 설정 (기본값: 현재 날짜)
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        
        # 연료 수분 함량 변수 목록 (정확한 변수명 사용)
        # 참고: ECMWF MARS에서 정확한 파라미터 코드를 확인해야 함
        fuel_variables = {
            # 연료 수분 함량 (GEFF Forecast API에서 제공)
            "live_fuel_moist": "live_fuel_moisture_content",
            "dead_foliage_moist": "dead_foliage_moisture_content",
            "dead_wood_moist": "dead_wood_moisture_content"
        }
        
        # ECMWF API 파라미터 코드 매핑 (실제 코드는 MARS 카탈로그에서 확인 필요)
        # 이 값들은 예시이며, 실제 코드는 ECMWF 헬프데스크 또는 MARS 카탈로그에서 확인해야 함
        param_codes = {
            "live_fuel_moisture_content": "xxx.yyy",  # ECMWF MARS의 실제 코드로 대체 필요
            "dead_foliage_moisture_content": "xxx.yyy",  # ECMWF MARS의 실제 코드로 대체 필요
            "dead_wood_moisture_content": "xxx.yyy",  # ECMWF MARS의 실제 코드로 대체 필요
        }
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"출력 디렉토리: {output_dir}")
        
        # ECMWF 서버 초기화
        logger.info("ECMWF 데이터 서버 초기화 중...")
        try:
            server = ECMWFDataServer()
            logger.info("ECMWF 데이터 서버 초기화 성공")
        except Exception as e:
            logger.error(f"ECMWF 데이터 서버 초기화 실패: {e}")
            logger.error("ECMWF API 설정을 확인하세요. ~/.ecmwfapirc 파일이 필요합니다.")
            return []
        
        # 예보 리드타임 설정 (예보 발표일부터 0일~days일)
        lead_times = "/".join([f"{day*24}" for day in range(days+1)])  # "0/24/48/.../240"
        
        logger.info(f"기준 날짜: {date_str}")
        logger.info(f"수집 변수: {', '.join(fuel_variables.keys())}")
        logger.info(f"예보 리드타임: {lead_times.replace('/', ', ')}")
        
        # 다운로드된 파일 목록
        downloaded_files = []
        
        # 각 변수별로 데이터 다운로드
        for var_code, cds_var_name in fuel_variables.items():
            # 변수별 디렉토리 생성
            var_dir = os.path.join(output_dir, var_code)
            os.makedirs(var_dir, exist_ok=True)
            
            # 다운로드 경로 생성
            target_file = os.path.join(var_dir, f"{var_code}_{date_str}.grib")
            
            logger.info(f"다운로드 중: {var_code} ({cds_var_name}), 출력: {target_file}")
            
            # 재시도 로직 추가
            for attempt in range(max_retries):
                try:
                    # ECMWF API를 통한 데이터 다운로드
                    # 주의: 실제 매개변수는 ECMWF MARS 카탈로그에서 확인해야 함
                    server.retrieve({
                        "class": "od",
                        "dataset": "geff-realtime",  # 실제 MARS 카탈로그에서 정확한 이름 확인 필요
                        "stream": "oper",
                        "type": "fc",
                        "expver": "1",
                        "param": param_codes.get(cds_var_name, "unknown"),  # 변수에 맞는 파라미터 코드
                        "date": date_str,
                        "time": "00/06/12/18",  # 00, 06, 12, 18 UTC 모델 실행
                        "step": lead_times,    # "0/24/48/.../240"
                        "grid": "0.25/0.25",    # 0.25° x 0.25° 해상도
                        "area": "39/124/33/132",  # 한국 영역 (North/West/South/East)
                        "format": "grib"
                    }, target_file)
                    
                    logger.info(f"다운로드 완료: {target_file}")
                    downloaded_files.append(target_file)
                    
                    # GRIB 파일에서 기본 정보 출력 시도 (eccodes 필요)
                    try:
                        import eccodes
                        with open(target_file, 'rb') as f:
                            gid = eccodes.codes_grib_new_from_file(f)
                            if gid:
                                shortName = eccodes.codes_get(gid, 'shortName')
                                param = eccodes.codes_get(gid, 'param')
                                step = eccodes.codes_get(gid, 'step')
                                logger.info(f"  - 변수: {shortName} (param: {param})")
                                logger.info(f"  - 예보 시간: {step}")
                                eccodes.codes_release(gid)
                    except ImportError:
                        logger.warning("eccodes 패키지가 설치되지 않아 GRIB 파일 정보를 읽을 수 없습니다.")
                    except Exception as e:
                        logger.warning(f"다운로드된 파일 정보 읽기 실패: {e}")
                    
                    break  # 성공시 재시도 루프 종료
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"다운로드 실패 ({attempt+1}/{max_retries}): {var_code}")
                        logger.warning(f"오류: {e}")
                        logger.info(f"{retry_delay}초 후 재시도...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"다운로드 최종 실패: {var_code}")
                        logger.error(f"오류: {e}")
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(traceback.format_exc())
        
        total_expected = len(fuel_variables)
        success_rate = len(downloaded_files) / total_expected * 100 if total_expected > 0 else 0
        
        logger.info(f"=== GEFF 연료 수분 함량 데이터 수집 완료 ===")
        logger.info(f"다운로드 성공: {len(downloaded_files)}/{total_expected} 파일 ({success_rate:.1f}%)")
        
        return downloaded_files
    
    except Exception as e:
        logger.error(f"예기치 않은 오류 발생: {e}")
        logger.debug(traceback.format_exc())
        return []

def test_ecmwf_connection():
    """
    ECMWF API 연결 및 설정을 테스트합니다.
    
    Returns:
    --------
    bool : 테스트 성공 여부
    """
    logger.info("=== ECMWF 데이터 접근 테스트 ===")
    
    try:
        # ecmwfapi 패키지 임포트 시도
        try:
            from ecmwfapi import ECMWFDataServer
        except ImportError:
            logger.error("'ecmwfapi' 패키지가 설치되어 있지 않습니다.")
            logger.error("설치 명령어: pip install ecmwfapi")
            logger.error("주의: ECMWF 데이터 접근을 위해 API 키와 회원 자격이 필요합니다.")
            logger.error("자세한 내용은 https://www.ecmwf.int/en/forecasts/access-forecasts/ecmwf-web-api 를 참조하세요.")
            return False
        
        # ECMWF API 설정 파일 확인
        home_dir = os.path.expanduser("~")
        api_rc_file = os.path.join(home_dir, ".ecmwfapirc")
        
        if not os.path.exists(api_rc_file):
            logger.error(f"ECMWF API 설정 파일이 없습니다: {api_rc_file}")
            logger.error("ECMWF 웹사이트에서 API 키를 발급받아 설정 파일을 생성하세요.")
            logger.error("자세한 내용은 https://www.ecmwf.int/en/forecasts/access-forecasts/ecmwf-web-api 를 참조하세요.")
            return False
        
        # 서버 연결 테스트
        logger.info("ECMWF 데이터 서버 연결 테스트 중...")
        try:
            server = ECMWFDataServer()
            logger.info("ECMWF 데이터 서버 연결 성공")
            
            # 서버 상태 확인 (추가 기능)
            try:
                status = server.status()
                logger.info(f"서버 상태: {status}")
            except:
                logger.warning("서버 상태 확인 기능을 지원하지 않습니다.")
            
            logger.info("=== ECMWF 데이터 접근 테스트 완료 - 성공 ===")
            logger.info("주의: 이 테스트는 서버 연결만 확인하며, 데이터 접근 권한은 확인하지 않습니다.")
            logger.info("GEFF 데이터에 접근하려면 적절한 접근 권한이 필요합니다.")
            return True
            
        except Exception as e:
            logger.error(f"ECMWF 데이터 서버 연결 실패: {e}")
            return False
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        logger.debug(traceback.format_exc())
        return False

def show_alternative_methods():
    """
    GEFF 데이터에 접근할 수 있는 대체 방법을 보여줍니다.
    """
    logger.info("=== GEFF 데이터 접근 대체 방법 ===")
    logger.info("GEFF Fire Forecast(일별 fuel moisture) 데이터는 직접 접근이 제한되어 있습니다.")
    logger.info("다음과 같은 대체 방법을 고려해보세요:")
    
    logger.info("\n1. ECMWF MARS/Web API 사용 (회원 자격 필요)")
    logger.info("   - ECMWF 회원 기관만 접근 가능")
    logger.info("   - ecmwfapi 패키지를 사용하여 데이터 요청")
    logger.info("   - 필요한 파라미터 코드는 MARS 카탈로그나 ECMWF 헬프데스크에서 확인")
    
    logger.info("\n2. EFFIS(EUMETSAT) 온라인 폼 사용")
    logger.info("   - EUMETSAT/EFFIS에서 운영하는 GEFF-realtime 요청 폼 사용")
    logger.info("   - 하루 4회(00/06/12/18 UTC) 갱신되는 최대 15일 리드타임 데이터 제공")
    
    logger.info("\n3. ERA5-Land 재분석과 물리 모델 사용")
    logger.info("   - CDS에서 제공되는 ERA5-Land 재분석 데이터 활용")
    logger.info("   - McNorton & Di Giuseppe static fuel load 데이터 활용")
    logger.info("   - 자체 물리 모델로 fuel moisture 생성")
    
    logger.info("\n자세한 내용은 ECMWF 또는 EUMETSAT 웹사이트를 참조하세요.")
    logger.info("=== 대체 방법 안내 완료 ===")

def main():
    """
    명령행 인터페이스 함수
    """
    parser = argparse.ArgumentParser(description="GEFF 연료 수분 함량 데이터 수집 도구")
    
    parser.add_argument("--output", type=str, default=None,
                       help="출력 디렉토리 경로 (기본값: 프로젝트 루트/data/fuel)")
    parser.add_argument("--date", type=str, default=None,
                       help="수집할 날짜 (YYYYMMDD 형식, 기본값: 현재 날짜)")
    parser.add_argument("--retries", type=int, default=3,
                       help="다운로드 실패 시 최대 재시도 횟수 (기본값: 3)")
    parser.add_argument("--days", type=int, default=10,
                       help="수집할 예보 일수 (기본값: 10, 최대 10일, 권장 5일)")
    parser.add_argument("--test-connection", action="store_true",
                       help="ECMWF API 연결 테스트 (데이터 다운로드 없음)")
    parser.add_argument("--show-alternatives", action="store_true",
                       help="GEFF 데이터 접근 대체 방법 안내")
    
    args = parser.parse_args()
    
    # 대체 방법 안내
    if args.show_alternatives:
        show_alternative_methods()
        return 0
    
    # 연결 테스트
    if args.test_connection:
        success = test_ecmwf_connection()
        return 0 if success else 1
    
    # 주의 메시지 표시
    logger.warning("주의: GEFF 일별 예측 데이터는 Copernicus Climate Data Store(CDS)에서 직접 접근할 수 없습니다.")
    logger.warning("이 데이터에 접근하려면 ECMWF 회원 자격이 필요하며, MARS/WebAPI를 통해 요청해야 합니다.")
    logger.warning("또는 EFFIS(EUMETSAT)의 온라인 폼을 통해 요청할 수 있습니다.")
    logger.warning("대체 방법을 확인하려면 --show-alternatives 옵션을 사용하세요.")
    
    # 연결 테스트 권장
    logger.info("데이터 수집 전에 ECMWF API 연결을 테스트하려면 --test-connection 옵션을 사용하세요.")
    
    # 사용자 확인
    try:
        confirmation = input("계속 진행하시겠습니까? (y/n): ")
        if confirmation.lower() != 'y':
            logger.info("작업을 취소합니다.")
            return 0
    except Exception:
        pass  # 스크립트 모드에서 실행 시 무시
    
    # 전체 수집 모드
    files = collect_geff_fuel_data(
        output_dir=args.output,
        date_str=args.date,
        max_retries=args.retries,
        days=args.days
    )
    
    return 0 if files else 1

if __name__ == "__main__":
    sys.exit(main()) 