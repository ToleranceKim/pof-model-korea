"""
Spatial Filtering Module for POF-Korea Project

이 모듈의 핵심 목적:
1. GADM 한국 경계 데이터 로드 및 처리
2. DataFrame을 한국 영역으로 공간 필터링
3. GeoPandas 기반 고성능 공간 연산

주요 함수:
- load_korea_boundary(): GADM 한국 경계 폴리곤 로드
- filter_korea_boundary(): 한국 영역 내 데이터 필터링

Example:
    >>> boundary = load_korea_boundary()
    >>> filtered_df =
    filter_korea_boundary(weather_df)
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)


def load_korea_boundary(shapefile_path: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    GADM 한국 경계 폴리곤을 로드합니다.
    
    Args: shapefile_path:
     GADM 파일 경로 (None이 면 기본 경로 사용)

    Returns:
     한국 경계 GeoDataFrame

    Raises:
     FileNotFoundError: 경계 파일을 찾을 수 없는 경우
    """

    # 1. 기본 경로들 정의
    possible_paths = [
        "data/shapefiles/gadm41_KOR_0.json",
        "data/shapefiles/gadm41_KOR_0.geojson"
        ]

    # 2. 파일 경로 찾기
    if shapefile_path is None:
        for path in possible_paths:
            if Path(path).exists():
                shapefile_path = path
                break

    # 3. 파일 존재 확인
    if shapefile_path is None or not Path(shapefile_path).exists():
        raise FileNotFoundError("한국 경계 파일을 찾을 수 없습니다")

    # 4. GeoPandas로 로드
    korea_boundary = gpd.read_file(shapefile_path)
    return korea_boundary

def filter_korea_boundary(df: pd.DataFrame, korea_boundary: Optional[gpd.GeoDataFrame] = None) -> pd.DataFrame:
    """
    DataFrame을 한국 영역으로 공간 필터링합니다.

    Args:
        df: lat, lon 컬럼이 포함된 DataFrame
        korea_boundary: 한국 경계 GeoDataFrame
    (None이면 자동 로드)

    Returns:
        한국 영역 내 데이터만 포함된 DataFrame
    """
    # 1. 입력 검증
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("DataFrame에 'lat', 'lon'컬럼이 필요합니다.")

    # 2. 한국 경계 로드 (필요시)
    if korea_boundary is None:
        korea_boundary = load_korea_boundary()

    # 3. DataFrame을 GeoDataFrame으로 변환
    from shapely.geometry import Point
    geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # 4. 공간 조인으로 필터링
    filtered_gdf = gpd.sjoin(gdf, korea_boundary, how='inner', predicate='within')

    # 5. 원본 DataFrame 형태로 변환 (geometry 컬럼 제거)
    result_df = filtered_gdf.drop(columns=['geometry', 'index_right'])
    logger.info(f"공간 필터링: {len(df)} -> {len(result_df)} 개 데이터")
    return result_df