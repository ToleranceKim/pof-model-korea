"""
Grid System for POF-Korea Project

이 모듈의 핵심 목적:
1. 위경도 좌표 ↔ Grid ID 변환
2. 대량 데이터 처리를 위한 벡터화 연산
3. 데이터 무결성 검증

Grid ID 공식: grid_id = (lat_bin + 900) * 3600 + (lon_bin + 1800)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Grid 시스템 상수들
GRID_RESOLUTION = 0.1  # 격자 해상도 (도)
LAT_OFFSET = 900       # 위도 오프셋 (-90도 ~ 90도 -> 0 ~ 1799)
LON_OFFSET = 1800      # 경도 오프셋 (-180도 ~ 180도 -> 0 ~ 3599)
N_LON_BINS = 3600      # 경도 방향 총 격자 수

def latlon2grid(lat: float, lon: float) -> int:

    """
    위경도를 Grid ID로 변환합니다.

    Args:
        lat: 위도 (-90~90도)
        lon: 경도 (-180~180도)
        
    Returns:
        Grid ID (정수)

    Examples:
        >>> latlon2grid(37.5, 127.0) # 서울 4593070
    """

    # lat_bin 계산
    lat_bin = int(np.floor(lat / GRID_RESOLUTION))
    # lon_bin 계산
    lon_bin = int(np.floor(lon / GRID_RESOLUTION))
    # grid_id 계산
    grid_id = (lat_bin + LAT_OFFSET) * N_LON_BINS + (lon_bin + LON_OFFSET)

    return grid_id

def grid2latlon(grid_id: int) -> Tuple[float, float]:

    """
    Grid ID를 위경도 좌표로 변환합니다.

    Args:
        grid_id: Grid ID (정수)

    Returns:
        Tuple[float, float] (float, float)

    Examples:
        >>> grid2latlon(4593070) # 서울 grid_id (37.5, 127.0)
    """

    lat_bin = (grid_id // N_LON_BINS) - LAT_OFFSET
    lon_bin = (grid_id % N_LON_BINS) - LON_OFFSET
    lat = lat_bin * GRID_RESOLUTION
    lon = lon_bin * GRID_RESOLUTION
    
    return (lat, lon)

def latlon2grid_vectorized(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    대량 좌표를 한 번에 Grid ID로 변환합니다.
    Args:
        lats: 위도 배열
        lons: 경도 배열

    Returns:
        Grid ID 배열

    Example:
        >>> lats = np.array([37.5, 35.1])
        >>> lons = np.array([127.0, 129.0])
        >>> latlon2grid_vectorized(lats, lons)
        array([4593070, ???????]) # 부산은 몇 번일까요 ?
    """
    lat_bins = np.floor(lats / GRID_RESOLUTION).astype(int)
    lon_bins = np.floor(lons / GRID_RESOLUTION).astype(int)
    
    # 배열 크기 검증
    if len(lats) != len(lons):
        raise ValueError(f"배열 크기 불일치 : {len(lats)} vs {len(lons)}")

    # Grid_ID 계산
    grid_ids = (lat_bins + LAT_OFFSET) * N_LON_BINS + (lon_bins + LON_OFFSET)

    # 반환
    return grid_ids

def grid2latlon_vectorized(grid_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grid ID 배열을 위경도 좌표 배열로 변환합니다.

    args:
        grid_ids: Grid ID 배열

    Returns:
        Tuple[위도 배열, 경도 배열]

    Example:
        >>> grid_ids = np.array([4593070, 4506690])
        >>> lats, lons = grid2latlon_vectorized(grid_ids)
        >>> print(lats) # [37.5, 35.1] 예상
        >>> print(lons) # [127.0, 129.0] 예상
    """

    # NumPy 배열 타입 체크
    if not isinstance(grid_ids, np.ndarray):
        grid_ids = np.array(grid_ids) # 자동 변환

    # 빈 배열 처리
    if len(grid_ids) == 0:
        return np.array([]), np.array([])\

        # 음수 Grid ID 체크
    if np.any(grid_ids < 0):
        raise ValueError(f"음수 Grid ID 발견: {grid_ids[grid_ids < 0]}")

    lat_bins = (grid_ids // N_LON_BINS) - LAT_OFFSET
    lon_bins = (grid_ids % N_LON_BINS) - LON_OFFSET
    lats = lat_bins.astype(float) * GRID_RESOLUTION
    lons = lon_bins.astype(float) * GRID_RESOLUTION

    return (lats, lons)

def add_coordinates_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame에 grid_id를 기반으로 lat, lon 좌표 컬럼을 추가합니다.
    
    Args:
        df: grid_id 컬럼이 포함된 DataFrame
        
    Returns:
        lat, lon 컬럼이 추가된 DataFrame
        
    Raises:
        ValueError: grid_id 컬럼이 없거나 DataFrame이 비어있는 경우
        
    Example:
        >>> df = pd.DataFrame({'grid_id': [4593070, 4506690]})
        >>> df_with_coords = add_coordinates_to_dataframe(df)
        >>> print(df_with_coords)
        # grid_id, lat, lon 컬럼 포함
    """
    # TODO(human): DataFrame 통합 함수 구현
    pass