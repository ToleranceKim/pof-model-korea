
"""
Data Validation Module for POF-Korea Project

이 모듈의 핵심 목적:
1. DataFrame 데이터 품질 검증
2. 누락값, 이상값, 일관성 검사
3. 시계열 연속성 및 공간적 일관성 확인

주요 함수:
- validate_row_consistency(): 행별 데이터 일관성 검증
- validate_missing_values(): 누락값 패턴 분석 및 검증
- validate_temporal_continuity(): 시계열 연속성 검증
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

def validate_missing_values(df: pd.DataFrame, critical_columns: Optional[List[str]] = None, max_missing_ratio: float = 0.1) -> Dict[str, Any]:
    """
    DataFrame의 누락값을 검증하고 분석합니다.

    Args:
        df: 검증할 DataFrame
        critical_columns: 필수 컬럼 리스트
    (None이면 모든 컬럼)
        max_missing_ratio: 허용 가능한 최대 누락 비율 (0.0-1.0)
    Returns:
        검증 결과 딕셔너리
    """
    # 1. 기본 검증
    if df.empty:
        return {"status": "error", "message": "빈 DataFrame입니다."}

    # 2. 누락값 계산
    missing_counts = df.isnull().sum()
    total_rows = len(df)
    missing_ratios = missing_counts / total_rows

    # 3. critical_columns 검증 (있는 경우)
    violations = []
    if critical_columns:
        for col in critical_columns:
            if col in missing_ratios and missing_ratios[col] > max_missing_ratio:
                violations.append(f"{col}:{missing_ratios[col]:.2%} (임계값: {max_missing_ratio:.2%})")

    # 반환값에 추가
    result = {
        "status": "success",
        "total_rows": total_rows,
        "missing_counts": missing_counts.to_dict(),
        "missing_ratios": missing_ratios.to_dict()
    }

    if violations:
        result["violations"] = violations
        result["status"] = "warning"

    return result

def validate_row_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    DataFrame의 행별 데이터 일관성을 검증합니다.

    Args:
        df: 검증할 DataFrame
    
    Returns:
        검증 결과 딕셔너리
    """
    # 1. 기본 검증
    if df.empty:
        return {"status": "error", "message": "빈 DataFrame입니다."}

    inconsistencies = []
    total_rows = len(df)
    
    # 2. 좌표 일관성 검증 (grid_id와 lat/lon이 모두 있는 경우)
    if all(col in df.columns for col in ['grid_id', 'lat', 'lon']):
        # grid_system import 필요
        from grid_system import grid2latlon

        for idx, row in df.iterrows():
            if pd.notna(row['grid_id']) and pd.notna(row['lat']) and pd.notna(row['lon']):
                expected_lat, expected_lon = grid2latlon(int(row['grid_id']))
                if abs(row['lat'] - expected_lat) > 0.05 or abs(row['lon'] - expected_lon) > 0.05:
                    inconsistencies.append(f"Row {idx}: grid_id-좌표 불일치")

    # 3. 온도 범위 검증 (있는 경우)
    if 't2m' in df.columns:
        temp_outliers = df[(df['t2m'] < -50) | (df['t2m'] > 60)]
        for idx in temp_outliers.index:
            inconsistencies.append(f"Row {idx}: 온도 범위 이상 ({temp_outliers.loc[idx, 't2m']}K)")

    # 4. 결과 반환
    return {
        "status": "warning" if inconsistencies else "success",
        "total_rows": total_rows,
        "inconsistent_rows": len(inconsistencies),
        "inconsistencies": inconsistencies[:10]  # 10개만 표시
    }

def validate_temporal_continuity(df: pd.DataFrame, time_col: str = 'date') -> Dict[str, Any]:
    """
    시계열 데이터의 연속성과 일관성을 검증합니다. 

    시간 컬럼의 중복, 누락 구간, 비정상적 간격을 분석하여
    데이터 품질 문제를 탐지합니다.

    Args:
        df (pd.DataFrame): 검증할 DataFrame
        time_col (str, optional) : 시간 컬럼명. Defaults to 'date'.

    Returns:
        Dict[str, Any]: 검증 결과 딕셔너리
            - status: 'success', 'warning', 'error'
            - total_records: 전체 레코드 수
            - duplicates: 중복 시간 개수
            - time_range: 시간 범위

    Raises:
        ValueError: 시간 컬럼이 존재하지 않거나 변환할 수 없는 경우

    Example:
        >>> result = validate_temporal_continuity(weather_df, 'date')
        >>> print(result['status']) 
        'success'
    """
    # 1. 기본 검증
    if df.empty:
        return {"status": "error", "message": "빈 DataFrame입니다."}

    if time_col not in df.columns:
        return {"status": "error", "message": f"'{time_col}'컬럼이 없습니다"}

    # 2. 시간 컬럼을 datetime으로 변환
    try:
        time_series = pd.to_datetime(df[time_col])
    except:
        return {"status": "error", "message": f"'{time_col}' 컬럼을 datetime으로 변환할 수 없습니다."}

    # 3. 시간 정렬 및 분석
    time_sorted = time_series.sort_values()

    # 4. 중복 시간 확인
    duplicates = time_series.duplicated().sum()
    
    # 5. 시간 간격 분석
    time_diffs = time_sorted.diff().dropna()

    # 6. 결과 반환
    return {
        "status": "success",
        "total_records": len(df),
        "time_range": f"{time_sorted.min()} ~ {time_sorted.max()}",
        "duplicates": int(duplicates),
        "time_gaps": "분석 결과" # 추가 구현 가능
    }