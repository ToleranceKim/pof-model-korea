"""
weather_fire
- 결합 데이터 : ERA5 기상 데이터 + MODIS 화재 데이터
- 결합 방식 : 시공간적 결합 (grid_id + date)
- 목적 : 육지 격자만 추출하여 모델링용 데이터셋 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from pathlib import Path

# Core modules import
import sys
sys.path.append(str(Path(__file__).parent.parent / "core"))
from validators import validate_row_consistency, validate_missing_values, validate_temporal_continuity

def integrate_weather_fire(weather_path: str, af_flag_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Step 1: Weather + AF_Flag 통합

    ERA5 기상 데이터와 MODIS 화재 플래그를 grid_id와 date 기준으로 결합합니다.

    Args:
        weather_path (str): ERA5 일별 기상 데이터 CSV 파일 경로
        af_flag_path (str): MODIS 화재 플래그 CSV 파일 경로
        output_path (Optional[str]): 결과 저장 경로 (None이면 저장 안 함)

    Returns:
        pd.DataFrame: 통합된 DataFrame (약 4,247,637 rows 예상)
    """
    
    logging.info("=== Step 1: Weather + AF_Flag 통합 시작 ===")
    
    # 1. 파일 존재 여부 확인
    weather_file = Path(weather_path)
    af_flag_file = Path(af_flag_path)
    
    if not weather_file.exists():
        raise FileNotFoundError(f"기상 데이터 파일을 찾을 수 없습니다: {weather_path}")
    if not af_flag_file.exists():
        raise FileNotFoundError(f"화재 플래그 파일을 찾을 수 없습니다: {af_flag_path}")
    
    logging.info("입력 파일 확인 완료")

    # 2. 데이터 로드
    weather = pd.read_csv(weather_path)
    af_flag = pd.read_csv(af_flag_path)
    
    logging.info(f"기상 데이터 : {len(weather):,} rows, 화재 데이터: {len(af_flag):,} rows")
    
    # 3. 필수 컬럼 확인
    required_weather_cols = ["grid_id", 'date']
    required_fire_cols = ['grid_id', 'date', 'af_flag']

    # 기상 데이터 컬럼 확인
    if not set(required_weather_cols).issubset(set(weather.columns)):
        missing = set(required_weather_cols) - set(weather.columns)
        raise ValueError(f"기상 데이터에 필수 컬럼이 누락됨 : {missing}")

    # 화재 데이터 컬럼 확인
    if not set(required_fire_cols).issubset(set(af_flag.columns)):
        missing = set(required_fire_cols) - set(af_flag.columns)
        raise ValueError(f"화재 데이터에 필수 컬럼이 누락됨: {missing}")

    logging.info("필수 컬럼 확인 완료")

    # 4. 데이터 통합
    result = pd.merge(
        weather,    # 첫 번째 DataFrame
        af_flag,    # 두 번째 DataFrame
        on=["grid_id", "date"],     # 결합 기준 컬럼
        how="inner",
        suffixes=('_weather', '_fire') # 중복 컬럼명 구분
    )

    logging.info(f"통합 완료: {len(result):,} rows")

    if output_path is not None:
        result.to_csv(output_path, index=False)
        logging.info(f"결과를 {output_path}에 저장했습니다")

    return result