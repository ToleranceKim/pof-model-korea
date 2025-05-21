#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import argparse
import traceback
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def collect_ecmwf_basic_variables(target_date=None, output_dir=None, steps=None, max_retries=3, retry_delay=5):
    """
    ECMWF에서 기본 기상 예보 변수 데이터를 수집합니다.
    
    Parameters:
    -----------
    target_date : datetime, optional
        수집할 예보의 기준 날짜 (기본값: 현재 날짜)
    output_dir : str, optional
        출력 디렉토리 경로 (기본값: 프로젝트 루트/data/forecast)
    steps : list, optional
        수집할 예보 시간(시간) 목록 (기본값: 24시간 간격으로 24~240시간, D+1에서 D+10)
    max_retries : int, optional
        다운로드 실패 시 최대 재시도 횟수 (기본값: 3)
    retry_delay : int, optional
        재시도 사이의 대기 시간(초) (기본값: 5)
    
    Returns:
    --------
    list : 다운로드된 파일 경로 목록
    """
    logger.info("=== ECMWF 기본 기상 예보 변수 데이터 수집 시작 ===")
    
    try:
        # ecmwf-opendata 패키지 임포트 시도
        try:
            from ecmwf.opendata import Client
        except ImportError:
            logger.error("'ecmwf-opendata' 패키지가 설치되어 있지 않습니다.")
            logger.error("설치 명령어: pip install ecmwf-opendata")
            return []
            
        # 출력 디렉토리 기본값 설정 (절대 경로 사용)
        if output_dir is None:
            # 프로젝트 루트 디렉토리 찾기
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
            output_dir = os.path.join(project_root, "data/forecast")
        else:
            output_dir = os.path.abspath(output_dir)
            
        # 기본 예보 시간 설정 (24시간 간격으로 1-10일)
        if steps is None:
            steps = [24 * i for i in range(1, 11)]  # 24, 48, 72, ..., 240
        
        # 기본 변수 설정 (ECMWF Open Data에서 직접 수집 가능한 기상 변수만)
        # 주의: 연료 관련 변수(fuel load, moisture content)는 별도 API 필요
        variables = ["2t", "2d", "10u", "10v", "tp"]
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"출력 디렉토리: {output_dir}")
        
        # 클라이언트 초기화
        client = Client(source="ecmwf")
        logger.info("ECMWF Open Data 클라이언트 초기화 성공")
        
        # 예보 시각 고정 (시스템 시간에 의존하지 않음)
        forecast_hour = 0  # 00Z 예보 사용
        
        logger.info("가장 최신 예보 데이터를 사용합니다 (date=-1)")
        logger.info(f"예보 시각: {forecast_hour:02d}Z")
        logger.info(f"수집 변수: {', '.join(variables)}")
        logger.info(f"예보 시간: {', '.join([f'+{step}h (D+{step//24})' for step in steps])}")
        
        downloaded_files = []
        current_date = datetime.now().strftime("%Y%m%d")  # 현재 시간 기준 파일명 생성용
        
        # 각 변수별로 데이터 다운로드
        for variable in variables:
            # 변수별 디렉토리 생성
            var_dir = os.path.join(output_dir, variable)
            os.makedirs(var_dir, exist_ok=True)
            
            for step in steps:
                # 파일명 생성 (실제 날짜는 다운로드 시점에 최신 데이터가 사용됨)
                target_file = os.path.join(var_dir, f"{variable}_{current_date}_{forecast_hour:02d}z_step{step:03d}.grib2")
                
                logger.info(f"다운로드 중: {variable}, 예보 시간: +{step}h, 출력: {target_file}")
                
                # 재시도 로직 추가
                for attempt in range(max_retries):
                    try:
                        # ECMWF Open Data에서 데이터 다운로드 (date=-1 사용)
                        client.retrieve(
                            date=-1,           # 가장 최신 예보 사용
                            time=forecast_hour,
                            step=step,
                            stream="oper",     # 고해상도 운영예보
                            type="fc",         # forecast
                            param=variable,
                            target=target_file
                        )
                        logger.info(f"다운로드 완료: {target_file}")
                        downloaded_files.append(target_file)
                        break  # 성공시 재시도 루프 종료
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"다운로드 실패 ({attempt+1}/{max_retries}): {variable}, 예보 시간: +{step}h")
                            logger.warning(f"오류: {e}")
                            logger.info(f"{retry_delay}초 후 재시도...")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"다운로드 최종 실패: {variable}, 예보 시간: +{step}h")
                            logger.error(f"오류: {e}")
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(traceback.format_exc())
        
        total_expected = len(variables) * len(steps)
        success_rate = len(downloaded_files) / total_expected * 100 if total_expected > 0 else 0
        
        logger.info(f"=== ECMWF 기본 기상 예보 변수 데이터 수집 완료 ===")
        logger.info(f"다운로드 성공: {len(downloaded_files)}/{total_expected} 파일 ({success_rate:.1f}%)")
        
        return downloaded_files
        
    except Exception as e:
        logger.error(f"예기치 않은 오류 발생: {e}")
        logger.debug(traceback.format_exc())
        return []

def test_forecast_availability(target_date=None, variable="2t", step=24, clean=True):
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
    clean : bool
        테스트 파일 자동 삭제 여부 (기본값: True)
        
    Returns:
    --------
    bool : 테스트 성공 여부
    """
    logger.info("=== ECMWF 예보 데이터 가용성 테스트 ===")
    
    try:
        # ecmwf-opendata 패키지 임포트 시도
        try:
            from ecmwf.opendata import Client
        except ImportError:
            logger.error("'ecmwf-opendata' 패키지가 설치되어 있지 않습니다.")
            logger.error("설치 명령어: pip install ecmwf-opendata")
            return False
            
        # 클라이언트 초기화
        client = Client(source="ecmwf")
        logger.info("ECMWF Open Data 클라이언트 초기화 성공")
        
        # 예보 시각 고정
        forecast_hour = 0  # 00Z 예보 사용
        
        logger.info(f"테스트 변수: {variable}")
        logger.info(f"예보 시간: +{step}h (D+{step//24})")
        
        # 임시 파일 경로 생성
        import tempfile
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, f"ecmwf_test_latest_{forecast_hour:02d}z_{variable}_step{step:03d}.grib2")
        logger.info(f"테스트 파일: {test_file}")
        
        # 데이터 요청 전송 - date=-1로 가장 최신 예보 사용
        logger.info("데이터 요청 시작...")
        logger.info("가장 최신 예보 데이터를 요청합니다 (date=-1)")
        client.retrieve(
            date=-1,           # 가장 최신 예보 사용
            time=forecast_hour,
            step=step,
            stream="oper",     # 고해상도 운영예보
            type="fc",         # forecast
            param=variable,
            target=test_file
        )
        
        # 파일 존재 여부 확인
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            logger.info("테스트 성공! 데이터 요청이 정상적으로 처리되었습니다.")
            logger.info(f"다운로드된 테스트 파일: {test_file}")
            logger.info(f"파일 크기: {file_size/1024:.2f} KB")
            
            # 자동 삭제 옵션에 따라 처리
            if clean:
                os.remove(test_file)
                logger.info(f"테스트 파일 자동 삭제 완료")
            else:
                logger.info(f"테스트 파일 유지: {test_file}")
                
            return True
        else:
            logger.error("테스트 실패: 파일이 생성되지 않았습니다.")
            return False
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        logger.debug(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='ECMWF 기본 기상 예보 변수 데이터 수집')
    
    # 명령행 인자 정의
    parser.add_argument('--start_step', type=int, default=24,
                        help='시작 예보 시간(시간) (기본값: 24, D+1)')
    parser.add_argument('--end_step', type=int, default=240,
                        help='종료 예보 시간(시간) (기본값: 240, D+10)')
    parser.add_argument('--step_interval', type=int, default=24,
                        help='예보 시간 간격(시간) (기본값: 24)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 절대 경로 (기본값: 프로젝트 루트/data/forecast)')
    parser.add_argument('--test', action='store_true',
                        help='데이터 가용성 테스트만 수행')
    parser.add_argument('--test_variable', type=str, default='2t',
                        help='테스트할 변수 (기본값: 2t)')
    parser.add_argument('--test_step', type=int, default=24,
                        help='테스트할 예보 시간(시간) (기본값: 24)')
    parser.add_argument('--keep_test_file', action='store_true',
                        help='테스트 파일 유지 (기본값: 자동 삭제)')
    parser.add_argument('--retries', type=int, default=3,
                        help='다운로드 실패 시 최대 재시도 횟수 (기본값: 3)')
    parser.add_argument('--retry_delay', type=int, default=5,
                        help='재시도 사이의 대기 시간(초) (기본값: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 로그 레벨 설정
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("디버그 모드 활성화")
    
    # 시스템 정보 출력
    logger.debug(f"운영체제: {os.name}")
    logger.debug(f"Python 버전: {sys.version}")
    logger.debug(f"시스템 시간 참고(로컬): {datetime.now()}")
    logger.debug(f"최신 예보 데이터 사용 (date=-1)")
    
    # 테스트 모드
    if args.test:
        logger.info("테스트 모드로 실행합니다...")
        success = test_forecast_availability(
            None, 
            args.test_variable, 
            args.test_step,
            not args.keep_test_file
        )
        return 0 if success else 1
    
    # 매개변수 유효성 검사
    if args.start_step <= 0 or args.end_step <= 0 or args.step_interval <= 0:
        logger.error("오류: 예보 시간과 간격은 양수여야 합니다.")
        return 1
    
    if args.start_step > args.end_step:
        logger.error("오류: 시작 예보 시간이 종료 예보 시간보다 클 수 없습니다.")
        return 1
    
    # 예보 시간 목록 생성
    steps = list(range(args.start_step, args.end_step + 1, args.step_interval))
    
    # 데이터 수집 실행
    files = collect_ecmwf_basic_variables(
        None,
        args.output_dir,
        steps,
        args.retries,
        args.retry_delay
    )
    
    if not files:
        logger.error("ECMWF 기본 기상 예보 변수 데이터 수집에 실패했습니다.")
        return 1
    
    logger.info(f"총 {len(files)}개 파일이 다운로드되었습니다.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 