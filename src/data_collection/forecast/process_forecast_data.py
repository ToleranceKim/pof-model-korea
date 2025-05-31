#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import argparse
import traceback
import pandas as pd
import numpy as np
import xarray as xr
import json
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def collect_forecast_data(target_date=None, output_dir=None, steps=None, max_retries=3, retry_delay=5):
    """
    ECMWF에서 기본 기상 예보 변수 데이터를 수집합니다.
    
    Parameters:
    -----------
    target_date : str, optional
        수집할 예보의 기준 날짜 (YYYYMMDD 형식, 기본값: 현재 날짜)
    output_dir : str, optional
        출력 디렉토리 경로 (기본값: 프로젝트 루트/data/forecast)
    steps : list, optional
        수집할 예보 시간(시간) 목록 (기본값: 24시간 간격으로 24~120시간, D+1에서 D+5)
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
            
        # 기본 예보 시간 설정 (24시간 간격으로 1-5일)
        if steps is None:
            steps = [24 * i for i in range(1, 6)]  # 24, 48, 72, 96, 120
        
        # 기본 변수 설정 (ECMWF Open Data에서 직접 수집 가능한 기상 변수만)
        variables = ["2t", "2d", "10u", "10v", "tp"]
        
        # 한국 영역 설정
        lat_min, lat_max = 33, 39  # 남위, 북위
        lon_min, lon_max = 124, 132  # 서경, 동경
        logger.info(f"관심 영역: 한국 (위도 {lat_min}°-{lat_max}°, 경도 {lon_min}°-{lon_max}°)")
        
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
        
        # 날짜 설정
        if target_date is None:
            current_date = datetime.now().strftime("%Y%m%d")  # 현재 시간 기준 파일명 생성용
        else:
            current_date = target_date
            
        logger.info(f"기준 날짜: {current_date}")
        
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
                        # ECMWF Open Data에서 데이터 다운로드 (area 매개변수 제거)
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

def interpolate_to_01deg(ds, var_name, lat_min=33, lat_max=39, lon_min=124, lon_max=132):
    """
    xarray 데이터셋을 0.1° 격자로 선형 보간합니다.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        보간할 데이터셋
    var_name : str
        보간할 변수 이름
    lat_min, lat_max : float
        위도 범위
    lon_min, lon_max : float
        경도 범위
    
    Returns:
    --------
    pandas.DataFrame : 보간된 데이터프레임
    """
    try:
        logger.info(f"0.1° 격자로 선형 보간 시작: {var_name}")
        
        # 0.1° 격자 생성
        target_lat = np.arange(lat_min, lat_max + 0.01, 0.1)
        target_lon = np.arange(lon_min, lon_max + 0.01, 0.1)
        
        # 데이터 변수 선택
        da = ds[var_name]
        
        # 차원에 따른 처리
        if len(da.shape) == 2:  # [lat, lon]
            logger.info(f"2D 데이터 보간 중: {da.shape}")
            
            # xarray DataArray 형태로 만들기
            target_lat_da = xr.DataArray(target_lat, dims=['latitude'])
            target_lon_da = xr.DataArray(target_lon, dims=['longitude'])
            
            # 선형 보간 수행
            interpolated = da.interp(
                latitude=target_lat_da,
                longitude=target_lon_da,
                method='linear'
            )
            
            # DataFrame으로 변환
            rows = []
            
            for i, lat in enumerate(target_lat):
                for j, lon in enumerate(target_lon):
                    try:
                        value = float(interpolated[i, j].values)
                        row = {'latitude': lat, 'longitude': lon, var_name: value}
                        rows.append(row)
                    except (IndexError, ValueError) as e:
                        logger.debug(f"보간 값 추출 중 오류: {e} (위치: {lat}, {lon})")
                        continue
            
        elif len(da.shape) == 3 and 'time' in ds.coords:  # [time, lat, lon]
            logger.info(f"3D 데이터 보간 중: {da.shape}")
            times = pd.to_datetime(ds.time.values)
            
            # xarray DataArray 형태로 만들기
            target_lat_da = xr.DataArray(target_lat, dims=['latitude'])
            target_lon_da = xr.DataArray(target_lon, dims=['longitude'])
            
            # 선형 보간 수행
            interpolated = da.interp(
                latitude=target_lat_da,
                longitude=target_lon_da,
                method='linear'
            )
            
            # DataFrame으로 변환
            rows = []
            
            for t, time_val in enumerate(times):
                for i, lat in enumerate(target_lat):
                    for j, lon in enumerate(target_lon):
                        try:
                            value = float(interpolated[t, i, j].values)
                            row = {
                                'time': time_val,
                                'latitude': lat, 
                                'longitude': lon, 
                                var_name: value
                            }
                            rows.append(row)
                        except (IndexError, ValueError) as e:
                            logger.debug(f"보간 값 추출 중 오류: {e} (위치: {time_val}, {lat}, {lon})")
                            continue
        else:
            logger.error(f"지원하지 않는 데이터 형식: {da.shape}")
            return None
        
        # DataFrame 생성
        df = pd.DataFrame(rows)
        logger.info(f"보간된 데이터프레임 생성 완료: {len(rows)}개 레코드")
        
        return df
        
    except Exception as e:
        logger.error(f"보간 중 오류 발생: {e}")
        logger.debug(traceback.format_exc())
        return None

def grib_to_dataframe(grib_file, interpolate=True):
    """
    GRIB2 파일을 Pandas DataFrame으로 변환합니다.
    
    Parameters:
    -----------
    grib_file : str
        입력 GRIB2 파일 경로
    interpolate : bool, optional
        0.1° 격자로 선형 보간 수행 여부 (기본값: True)
    
    Returns:
    --------
    pandas.DataFrame : 변환된 데이터프레임 또는 None (변환 실패 시)
    """
    try:
        logger.info(f"GRIB2 파일 '{grib_file}' 처리 중...")
        
        # GRIB2 파일 로드
        try:
            import cfgrib
            ds = xr.open_dataset(grib_file, engine='cfgrib')
            logger.info(f"데이터셋 로드 성공: {list(ds.data_vars)}")
        except Exception as e:
            logger.error(f"GRIB2 파일 '{grib_file}' 로드 중 오류 발생: {e}")
            logger.error("cfgrib 패키지가 설치되어 있는지 확인하세요. (pip install cfgrib)")
            return None
        
        # 변수명 확인 (GRIB2 파일에는 보통 하나의 변수만 있음)
        var_names = list(ds.data_vars)
        if not var_names:
            logger.error("오류: GRIB2 파일에 변수가 없습니다.")
            return None
            
        var_name = var_names[0]
        logger.info(f"처리할 변수: {var_name}")
        
        # 위도/경도 확인
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            # 데이터 배열 가져오기
            da = ds[var_name]
            
            # 데이터 배열의 shape 확인
            logger.info(f"데이터 배열 shape: {da.shape}")
            
            # 한국 지역 경계 정의
            lat_min, lat_max = 33, 39  # 남위, 북위
            lon_min, lon_max = 124, 132  # 서경, 동경
            
            # 좌표계 정보 확인
            lats = ds.latitude.values
            lons = ds.longitude.values
            logger.info(f"위도 범위: {lats.min():.2f} ~ {lats.max():.2f}, 크기: {lats.shape}")
            logger.info(f"경도 범위: {lons.min():.2f} ~ {lons.max():.2f}, 크기: {lons.shape}")
            
            # 좌표 방향 확인 (위도가 내림차순인지)
            lat_descending = lats[0] > lats[-1] if len(lats) > 1 else False
            
            # 한국 지역 데이터 추출
            region_da = None
            
            # 두 가지 방법 시도
            try:
                # 1. slice 메서드 사용
                if lat_descending:
                    logger.info("위도가 내림차순으로 정렬되어 있습니다 (북→남)")
                    lat_sel = slice(lat_max, lat_min)
                else:
                    logger.info("위도가 오름차순으로 정렬되어 있습니다 (남→북)")
                    lat_sel = slice(lat_min, lat_max)
                
                region_da = da.sel(latitude=lat_sel, longitude=slice(lon_min, lon_max))
                logger.info(f"한국 영역 선택 완료 (slice 방식): {region_da.shape}")
                
                # 선택된 영역이 비어있으면 다른 방법 시도
                if region_da.size == 0:
                    raise ValueError("선택된 영역이 비어있습니다")
                    
            except Exception as e:
                logger.warning(f"slice 방식 선택 중 오류 발생: {e}")
                logger.warning("isel 방식으로 재시도합니다")
                
                try:
                    # 2. Boolean 마스킹 및 isel 메서드 사용
                    lat_mask = (lats >= lat_min) & (lats <= lat_max)
                    lon_mask = (lons >= lon_min) & (lons <= lon_max)
                    
                    if np.any(lat_mask) and np.any(lon_mask):
                        region_da = da.isel(latitude=lat_mask, longitude=lon_mask)
                        logger.info(f"한국 영역 선택 완료 (마스킹 방식): {region_da.shape}")
                    else:
                        logger.warning(f"유효한 마스크를 생성할 수 없습니다: lat_mask={np.sum(lat_mask)}, lon_mask={np.sum(lon_mask)}")
                        raise ValueError("유효한 마스크를 생성할 수 없습니다")
                except Exception as e2:
                    logger.warning(f"마스킹 방식 선택 중 오류 발생: {e2}")
                    logger.warning("전체 데이터를 사용합니다")
                    region_da = da
            
            # 선택된 영역이 비어있는지 최종 확인
            if region_da is None or region_da.size == 0:
                logger.warning("선택된 영역이 비어있어 원본 데이터를 사용합니다")
                region_da = da
            
            # DataFrame으로 변환할 데이터 생성
            rows = []
            
            # 선형 보간이 활성화되어 있으면 0.1° 격자로 보간
            if interpolate:
                logger.info(f"선택된 영역을 0.1° 격자로 보간합니다")
                
                # 0.1° 격자 생성
                target_lat = np.arange(lat_min, lat_max + 0.01, 0.1)
                target_lon = np.arange(lon_min, lon_max + 0.01, 0.1)
                
                # 격자 정보 로깅
                logger.info(f"보간 격자: 위도 {len(target_lat)}개, 경도 {len(target_lon)}개 포인트")
                
                # xarray DataArray 형태로 만들기
                target_lat_da = xr.DataArray(target_lat, dims=['latitude'])
                target_lon_da = xr.DataArray(target_lon, dims=['longitude'])
                
                try:
                    # 선형 보간 수행
                    logger.info("선형 보간 시작...")
                    interpolated = region_da.interp(
                        latitude=target_lat_da,
                        longitude=target_lon_da,
                        method='linear'
                    )
                    logger.info(f"선형 보간 완료: {interpolated.shape}")
                    
                    # 차원에 따른 처리
                    if len(interpolated.shape) == 2:  # [lat, lon]
                        for i, lat in enumerate(target_lat):
                            for j, lon in enumerate(target_lon):
                                try:
                                    value = float(interpolated[i, j].values)
                                    row = {'latitude': lat, 'longitude': lon, var_name: value}
                                    rows.append(row)
                                except (IndexError, ValueError) as e:
                                    logger.debug(f"보간 값 추출 중 오류: {e} (위치: {lat}, {lon})")
                                    continue
                    
                    elif len(interpolated.shape) == 3 and 'time' in region_da.coords:  # [time, lat, lon]
                        times = pd.to_datetime(region_da.time.values)
                        
                        for t, time_val in enumerate(times):
                            for i, lat in enumerate(target_lat):
                                for j, lon in enumerate(target_lon):
                                    try:
                                        value = float(interpolated[t, i, j].values)
                                        row = {
                                            'time': time_val,
                                            'latitude': lat, 
                                            'longitude': lon, 
                                            var_name: value
                                        }
                                        rows.append(row)
                                    except (IndexError, ValueError) as e:
                                        logger.debug(f"보간 값 추출 중 오류: {e} (위치: {time_val}, {lat}, {lon})")
                                        continue
                                        
                except Exception as e:
                    logger.error(f"선형 보간 중 오류 발생: {e}")
                    logger.warning("보간을 건너뛰고 원본 데이터 포인트를 사용합니다")
                    interpolate = False  # 보간 실패 시 원본 데이터 사용으로 변경
            
            # 보간하지 않거나 보간에 실패한 경우 원본 데이터 사용
            if not interpolate or not rows:
                logger.info("원본 데이터 포인트를 사용합니다")
                
                # 원본 데이터 처리
                if len(region_da.shape) == 2:  # [lat, lon]
                    region_lats = region_da.latitude.values
                    region_lons = region_da.longitude.values
                    data_values = region_da.values
                    
                    for i, lat in enumerate(region_lats):
                        for j, lon in enumerate(region_lons):
                            try:
                                # 한국 지역으로 재필터링 (확인용)
                                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                                    value = float(data_values[i, j])
                                    row = {'latitude': lat, 'longitude': lon, var_name: value}
                                    rows.append(row)
                            except (IndexError, ValueError) as e:
                                logger.debug(f"값 추출 중 오류: {e} (위치: {lat}, {lon})")
                                continue
                                
                elif len(region_da.shape) == 3 and 'time' in region_da.coords:  # [time, lat, lon]
                    region_lats = region_da.latitude.values
                    region_lons = region_da.longitude.values
                    times = pd.to_datetime(region_da.time.values)
                    data_values = region_da.values
                    
                    for t, time_val in enumerate(times):
                        for i, lat in enumerate(region_lats):
                            for j, lon in enumerate(region_lons):
                                try:
                                    # 한국 지역으로 재필터링 (확인용)
                                    if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                                        value = float(data_values[t, i, j])
                                        row = {
                                            'time': time_val,
                                            'latitude': lat, 
                                            'longitude': lon, 
                                            var_name: value
                                        }
                                        rows.append(row)
                                except (IndexError, ValueError) as e:
                                    logger.debug(f"값 추출 중 오류: {e} (위치: {time_val}, {lat}, {lon})")
                                    continue
            
            # DataFrame 생성
            df = pd.DataFrame(rows)
            logger.info(f"데이터프레임 생성 완료: {len(rows)}개 레코드")
            
        else:
            # 다른 형태의 데이터
            logger.error("비격자 형태의 데이터는 아직 지원하지 않습니다.")
            return None
            
        # 기본 정보 추가
        df['source'] = 'ecmwf-opendata'
        
        logger.info(f"변환 완료: 행: {len(df)}, 열: {len(df.columns)}")
        
        return df
        
    except Exception as e:
        import traceback
        logger.error(f"변환 중 오류 발생: {e}")
        logger.debug(traceback.format_exc())
        return None

def calculate_grid_id(df):
    """
    위도/경도를 grid_id로 변환합니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        위도/경도 정보가 포함된 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame : grid_id가 추가된 데이터프레임
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.error("위도/경도 정보가 데이터프레임에 없습니다.")
        return df
    
    logger.info("위도/경도를 grid_id로 변환 중...")
    
    # 위도/경도를 0.1° 단위로 변환하여 grid_id 계산
    df['lat_bin'] = (df['latitude'] / 0.1).astype(int)
    df['lon_bin'] = (df['longitude'] / 0.1).astype(int)
    df['grid_id'] = (df['lat_bin'] + 900) * 3600 + (df['lon_bin'] + 1800)
    
    # 임시 컬럼 삭제
    df = df.drop(columns=['lat_bin', 'lon_bin'])
    
    logger.info(f"grid_id 변환 완료: {df['grid_id'].nunique()} 개의 고유 grid_id 생성됨")
    
    return df

def calculate_wind10m(df_u10, df_v10):
    """
    u10와 v10 성분을 사용하여 wind10m 값을 계산합니다.
    
    Parameters:
    -----------
    df_u10 : pandas.DataFrame
        u10 변수가 포함된 데이터프레임
    df_v10 : pandas.DataFrame
        v10 변수가 포함된 데이터프레임
    
    Returns:
    --------
    pandas.DataFrame : wind10m 변수가 추가된 데이터프레임
    """
    logger.info("u10와 v10 성분으로 wind10m 계산 중...")
    
    # u10/v10 컬럼이 있는지 확인 (변수명이 다를 수 있음)
    u10_col = None
    for col in df_u10.columns:
        if col in ['u10', '10u']:
            u10_col = col
            break
    
    v10_col = None
    for col in df_v10.columns:
        if col in ['v10', '10v']:
            v10_col = col
            break
    
    if u10_col is None or v10_col is None:
        logger.error(f"바람 성분 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df_u10.columns)} / {list(df_v10.columns)}")
        return None
    
    # 공통 키로 데이터프레임 병합
    if 'time' in df_u10.columns and 'time' in df_v10.columns:
        merge_keys = ['grid_id', 'latitude', 'longitude', 'time']
    else:
        merge_keys = ['grid_id', 'latitude', 'longitude']
    
    # 데이터프레임 병합
    df_merged = pd.merge(df_u10, df_v10, on=merge_keys, how='inner')
    
    # 결합된 데이터프레임에서 wind10m 계산: √(u10² + v10²)
    df_merged['wind10m'] = np.sqrt(df_merged[u10_col]**2 + df_merged[v10_col]**2)
    
    # 이제 필요한 컬럼만 남기기
    cols_to_keep = merge_keys + ['wind10m']
    if 'source_x' in df_merged.columns:
        df_merged = df_merged[cols_to_keep + ['source_x']]
        df_merged = df_merged.rename(columns={'source_x': 'source'})
    else:
        df_merged = df_merged[cols_to_keep]
    
    logger.info(f"wind10m 계산 완료: 범위 {df_merged['wind10m'].min():.2f} ~ {df_merged['wind10m'].max():.2f} m/s")
    
    return df_merged

def grid_id_to_latlon(grid_id):
    """
    grid_id를 위도/경도로 변환합니다.
    
    Parameters:
    -----------
    grid_id : int
        변환할 grid_id
    
    Returns:
    --------
    tuple : (latitude, longitude)
    """
    lat_bin = ((grid_id // 3600) - 900)
    lon_bin = ((grid_id % 3600) - 1800)
    lat = lat_bin * 0.1
    lon = lon_bin * 0.1
    return lat, lon

def visualize_grid_distribution(df, valid_grid_ids, filter_stats, output_base):
    """
    grid_id의 지리적 분포를 시각화합니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        필터링된 데이터프레임
    valid_grid_ids : set
        유효한 grid_id 집합
    filter_stats : dict
        필터링 통계 정보
    output_base : str
        출력 파일 기본 경로
    """
    try:
        # 필요한 라이브러리가 있는지 확인
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError:
            logger.warning("matplotlib 라이브러리가 설치되어 있지 않아 시각화를 건너뜁니다.")
            return
        
        logger.info("격자 분포 시각화 중...")
        
        # 한국 지역 경계 설정
        min_lat, max_lat = 33, 39
        min_lon, max_lon = 124, 132
        
        # 유효한 grid_id의 위도/경도 계산
        valid_lats, valid_lons = [], []
        for gid in valid_grid_ids:
            lat, lon = grid_id_to_latlon(gid)
            valid_lats.append(lat)
            valid_lons.append(lon)
        
        # 데이터프레임에 있는 grid_id의 위도/경도
        data_grid_ids = set(df['grid_id'].unique())
        data_lats, data_lons = [], []
        for gid in data_grid_ids:
            lat, lon = grid_id_to_latlon(gid)
            data_lats.append(lat)
            data_lons.append(lon)
        
        # 누락된 grid_id의 위도/경도
        missing_grid_ids = filter_stats.get('missing_grid_ids', [])
        missing_lats, missing_lons = [], []
        for gid in missing_grid_ids:
            lat, lon = grid_id_to_latlon(gid)
            missing_lats.append(lat)
            missing_lons.append(lon)
        
        # 첫 번째 시각화: 격자 분포
        plt.figure(figsize=(10, 8))
        
        # 기본 지도 설정
        plt.scatter(valid_lons, valid_lats, s=5, c='lightgray', alpha=0.3, label='기대 grid_id')
        plt.scatter(data_lons, data_lats, s=15, c='blue', alpha=0.7, label='수집된 grid_id')
        
        # 누락된 grid_id가 있는 경우에만 표시
        if missing_lats:
            plt.scatter(missing_lons, missing_lats, s=10, c='red', alpha=0.7, label='누락된 grid_id')
        
        # 그래프 설정
        plt.title('Grid ID 분포')
        plt.xlabel('경도')
        plt.ylabel('위도')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(min_lon - 0.5, max_lon + 0.5)
        plt.ylim(min_lat - 0.5, max_lat + 0.5)
        
        # 범례 추가
        plt.legend(loc='upper right')
        
        # 통계 정보 추가
        info_text = (
            f"전체 grid_id: {filter_stats['expected_grid_ids']}\n"
            f"수집된 grid_id: {filter_stats['actual_grid_ids']}\n"
            f"누락된 grid_id: {len(missing_grid_ids)}\n"
            f"수집률: {filter_stats['actual_grid_ids']/filter_stats['expected_grid_ids']*100:.1f}%"
        )
        plt.annotate(info_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # 저장
        viz_file = f"{output_base}_grid_distribution.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"격자 분포 시각화 저장: {viz_file}")
        
    except Exception as e:
        logger.error(f"시각화 중 오류 발생: {e}")
        logger.debug(traceback.format_exc())

def filter_by_grid_ids(df, grid_ids_file=None):
    """
    지정된 grid_id 목록에 해당하는 데이터만 필터링합니다.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        필터링할 데이터프레임
    grid_ids_file : str, optional
        grid_id 목록이 저장된 JSON 파일 경로
        
    Returns:
    --------
    pandas.DataFrame : 필터링된 데이터프레임
    dict : 필터링 결과 통계 정보
    """
    logger.info("지정된 grid_id로 데이터 필터링 중...")
    
    # 필터링 결과 통계
    stats = {
        'total_records_before': int(len(df)),  # int64를 int로 변환
        'unique_grid_ids_before': int(df['grid_id'].nunique() if 'grid_id' in df.columns else 0),  # int64를 int로 변환
        'expected_grid_ids': 0,
        'actual_grid_ids': 0,
        'missing_grid_ids': [],
        'extra_grid_ids': []
    }
    
    # 기본 grid_id 파일 경로 설정
    if grid_ids_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        grid_ids_file = os.path.join(script_dir, "korea_land_grid_ids.json")
    
    # 파일에서 grid_id 목록 로드
    if os.path.exists(grid_ids_file):
        try:
            with open(grid_ids_file, 'r') as f:
                valid_grid_ids = set(json.load(f))
            logger.info(f"{len(valid_grid_ids)}개의 grid_id 로드 완료")
            stats['expected_grid_ids'] = len(valid_grid_ids)
        except Exception as e:
            logger.error(f"grid_id 파일 로드 중 오류: {e}")
            return df, stats
    else:
        # 기본 grid_id 목록 (파일이 없는 경우)
        logger.warning(f"grid_id 파일이 없습니다: {grid_ids_file}")
        logger.warning("grid_id 필터링을 건너뜁니다.")
        return df, stats
    
    # 원본 데이터 크기 기록
    original_size = len(df)
    
    # grid_id로 필터링
    if 'grid_id' in df.columns:
        # 데이터프레임에 있는 grid_id 중 유효하지 않은 것들 확인
        existing_grid_ids = set(df['grid_id'].unique().tolist())  # NumPy 배열을 리스트로 변환
        stats['extra_grid_ids'] = sorted([int(x) for x in list(existing_grid_ids - valid_grid_ids)])  # int64를 int로 변환
        
        # 필터링 수행
        df = df[df['grid_id'].isin(valid_grid_ids)]
        
        # 필터링 후 통계
        filtered_grid_ids = set(df['grid_id'].unique().tolist())  # NumPy 배열을 리스트로 변환
        stats['actual_grid_ids'] = len(filtered_grid_ids)
        stats['missing_grid_ids'] = sorted([int(x) for x in list(valid_grid_ids - filtered_grid_ids)])  # int64를 int로 변환
        
        # 결과 로그
        logger.info(f"grid_id 필터링 완료: {original_size}개 → {len(df)}개 레코드 ({len(df)/original_size*100:.1f}%)")
        logger.info(f"기대 grid_id 개수: {stats['expected_grid_ids']}, 실제 grid_id 개수: {stats['actual_grid_ids']}")
        
        if stats['missing_grid_ids']:
            missing_count = len(stats['missing_grid_ids'])
            logger.warning(f"{missing_count}개의 grid_id가 누락되었습니다 ({missing_count/stats['expected_grid_ids']*100:.1f}%)")
            if missing_count <= 10:
                logger.warning(f"누락된 grid_id: {stats['missing_grid_ids']}")
            else:
                logger.warning(f"누락된 grid_id 일부: {stats['missing_grid_ids'][:10]}...")
        else:
            logger.info("모든 기대 grid_id가 포함되었습니다.")
    else:
        logger.error("데이터프레임에 grid_id 컬럼이 없습니다.")
    
    return df, stats

def process_forecast_step(target_date, step, forecast_dir, output_dir, grid_ids_file=None, visualize=False, interpolate=True):
    """
    특정 예보 시간의 모든 변수를 처리하고 통합합니다.
    
    Parameters:
    -----------
    target_date : str
        예보 기준 날짜 (YYYYMMDD 형식)
    step : int
        예보 시간 (예: 24, 48, 72, 96, 120)
    forecast_dir : str
        예보 데이터 디렉토리 경로
    output_dir : str
        출력 디렉토리 경로
    grid_ids_file : str, optional
        grid_id 목록이 저장된 JSON 파일 경로
    visualize : bool, optional
        시각화 생성 여부 (기본값: False)
    interpolate : bool, optional
        0.1° 격자로 선형 보간 수행 여부 (기본값: True)
    
    Returns:
    --------
    str : 처리된 파일 경로 또는 None (처리 실패 시)
    """
    logger.info(f"=== 예보 시간 +{step}h (D+{step//24}) 처리 시작 ===")
    
    try:
        # 변수 목록 및 파일 경로 매핑
        var_file_mapping = {
            "2t": os.path.join(forecast_dir, "2t", f"2t_{target_date}_00z_step{step:03d}.grib2"),
            "2d": os.path.join(forecast_dir, "2d", f"2d_{target_date}_00z_step{step:03d}.grib2"),
            "10u": os.path.join(forecast_dir, "10u", f"10u_{target_date}_00z_step{step:03d}.grib2"),
            "10v": os.path.join(forecast_dir, "10v", f"10v_{target_date}_00z_step{step:03d}.grib2"),
            "tp": os.path.join(forecast_dir, "tp", f"tp_{target_date}_00z_step{step:03d}.grib2")
        }
        
        # 각 변수별 데이터프레임 저장
        dfs = {}
        
        # 각 변수별 GRIB2 파일을 데이터프레임으로 변환
        for var_name, file_path in var_file_mapping.items():
            if not os.path.exists(file_path):
                logger.warning(f"파일을 찾을 수 없음: {file_path}")
                continue
            
            # GRIB2 파일을 데이터프레임으로 변환 (선형 보간 적용 여부 전달)
            df = grib_to_dataframe(file_path, interpolate=interpolate)
            
            if df is not None:
                # grid_id 계산
                df = calculate_grid_id(df)
                dfs[var_name] = df
        
        # 모든 변수가 있는지 확인
        missing_vars = [var for var in var_file_mapping.keys() if var not in dfs]
        if missing_vars:
            logger.warning(f"일부 변수를 찾을 수 없음: {', '.join(missing_vars)}")
        
        # 바람 성분으로 wind10m 계산
        if '10u' in dfs and '10v' in dfs:
            wind_df = calculate_wind10m(dfs['10u'], dfs['10v'])
            if wind_df is not None:
                dfs['wind10m'] = wind_df
        
        # 최종 데이터프레임 초기화
        final_df = None
        
        # 변수명 매핑 정의
        var_name_mapping = {
            '2t': 't2m',
            'd2m': 'td2m',   # GRIB2 파일에서는 'd2m'으로 표시될 수 있음
            '2d': 'td2m',    # 파일명은 2d지만 내용은 d2m으로 저장될 수 있음
            'tp': 'tp',      # precip에서 tp로 변경
            'wind10m': 'wind10m',  # 이미 계산된 변수
            '10u': '10u',    # 풍향 계산 등을 위해 추가
            '10v': '10v'     # 풍향 계산 등을 위해 추가
        }
        
        # 변수별 처리 및 병합
        for var_name, df in dfs.items():
            # 변수명 표준화 (10u와 10v도 포함)
            processed_df = df.copy()
            
            # 변수명 매핑
            for orig_name, new_name in var_name_mapping.items():
                if orig_name in processed_df.columns:
                    processed_df = processed_df.rename(columns={orig_name: new_name})
            
            # 특수 처리: 강수량 단위 변환 (m -> mm)
            if 'tp' in processed_df.columns:
                processed_df['tp'] = processed_df['tp'] * 1000
            
            # 필요한 컬럼만 선택
            id_cols = ['grid_id', 'latitude', 'longitude']
            if 'time' in processed_df.columns:
                id_cols.append('time')
            
            value_cols = [col for col in processed_df.columns 
                          if col in ['t2m', 'td2m', 'tp', 'wind10m', '10u', '10v']]
            
            if value_cols:
                processed_df = processed_df[id_cols + value_cols]
                
                # 최종 데이터프레임에 병합
                if final_df is None:
                    final_df = processed_df
                else:
                    final_df = pd.merge(final_df, processed_df, on=id_cols, how='outer')
        
        # 최종 데이터프레임이 비어있는지 확인
        if final_df is None or final_df.empty:
            logger.error("처리할 데이터가 없습니다.")
            return None
        
        # 예보 리드타임 추가
        final_df['lead'] = step // 24  # 일 단위로 변환 (1, 2, 3, 4, 5)
        
        # forecast_date 추가 (예보 발표일)
        dt = datetime.strptime(target_date, '%Y%m%d')
        final_df['forecast_date'] = dt.strftime('%Y-%m-%d')
        
        # prediction_date 추가 (기존 valid_date, 예보 유효일, forecast_date + lead)
        if 'lead' in final_df.columns:
            final_df['prediction_date'] = final_df.apply(
                lambda row: (dt + timedelta(days=int(row['lead']))).strftime('%Y-%m-%d'), 
                axis=1
            )
        
        # 지정된 grid_id로 필터링
        final_df, filter_stats = filter_by_grid_ids(final_df, grid_ids_file)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 출력 파일 경로 생성
        output_base = os.path.join(output_dir, f"forecast_{target_date}_lead{step//24:d}")
        output_csv = f"{output_base}.csv"
        output_parquet = f"{output_base}.parquet"
        
        # CSV 및 Parquet 형식으로 저장
        logger.info(f"결과 저장 중: {output_csv}")
        final_df.to_csv(output_csv, index=False)
        
        logger.info(f"결과 저장 중: {output_parquet}")
        final_df.to_parquet(output_parquet, index=False)
        
        # 필터링 결과 요약 저장
        if filter_stats.get('expected_grid_ids', 0) > 0:
            try:
                # JSON 직렬화를 위해 NumPy 타입을 Python 기본 타입으로 변환
                serializable_stats = {
                    'total_records_before': int(filter_stats['total_records_before']),
                    'unique_grid_ids_before': int(filter_stats['unique_grid_ids_before']),
                    'expected_grid_ids': int(filter_stats['expected_grid_ids']),
                    'actual_grid_ids': int(filter_stats['actual_grid_ids']),
                    'missing_grid_ids': [int(x) for x in filter_stats['missing_grid_ids']],
                    'extra_grid_ids': [int(x) for x in filter_stats['extra_grid_ids']]
                }
                
                # 모든 정보를 하나의 JSON 파일에 저장 (missing_grid_ids.json 파일은 생성하지 않음)
                summary_file = f"{output_base}_filter_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(serializable_stats, f, indent=2)
                logger.info(f"필터링 결과 요약 저장: {summary_file}")
                
                # 시각화 생성 (선택적)
                if visualize and grid_ids_file is not None:
                    try:
                        # grid_id 목록 로드
                        try:
                            with open(grid_ids_file, 'r') as f:
                                valid_grid_ids = set(json.load(f))
                            visualize_grid_distribution(final_df, valid_grid_ids, serializable_stats, output_base)
                        except Exception as e:
                            logger.error(f"시각화를 위한 grid_id 로드 중 오류: {e}")
                            logger.debug(traceback.format_exc())
                    except Exception as e:
                        logger.error(f"시각화 생성 중 오류 발생: {e}")
                        logger.debug(traceback.format_exc())
            except Exception as e:
                logger.error(f"필터링 결과 저장 중 오류 발생: {e}")
                logger.debug(traceback.format_exc())
        
        logger.info(f"=== 예보 시간 +{step}h (D+{step//24}) 처리 완료 ===")
        logger.info(f"처리된 행 수: {len(final_df)}, 열 수: {len(final_df.columns)}")
        
        return output_csv
        
    except Exception as e:
        logger.error(f"예보 처리 중 오류 발생: {e}")
        logger.debug(traceback.format_exc())
        return None

def process_all_forecast_steps(target_date=None, forecast_dir=None, output_dir=None, steps=None, grid_ids_file=None, visualize=False, interpolate=True):
    """
    모든 예보 시간을 처리합니다.
    
    Parameters:
    -----------
    target_date : str, optional
        예보 기준 날짜 (YYYYMMDD 형식, 기본값: 현재 날짜)
    forecast_dir : str, optional
        예보 데이터 디렉토리 경로 (기본값: 프로젝트 루트/data/forecast)
    output_dir : str, optional
        출력 디렉토리 경로 (기본값: 프로젝트 루트/data/processed/forecast)
    steps : list, optional
        처리할 예보 시간(시간) 목록 (기본값: 24시간 간격으로 24~120시간, D+1에서 D+5)
    grid_ids_file : str, optional
        grid_id 목록이 저장된 JSON 파일 경로
    visualize : bool, optional
        시각화 생성 여부 (기본값: False)
    interpolate : bool, optional
        0.1° 격자로 선형 보간 수행 여부 (기본값: True)
    
    Returns:
    --------
    list : 처리된 파일 경로 목록
    """
    # 기본값 설정
    if target_date is None:
        target_date = datetime.now().strftime("%Y%m%d")
    
    if forecast_dir is None:
        # 프로젝트 루트 디렉토리 찾기
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
        forecast_dir = os.path.join(project_root, "data/forecast")
    
    if output_dir is None:
        # 프로젝트 루트 디렉토리 찾기
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
        output_dir = os.path.join(project_root, "data/processed/forecast")
    
    if steps is None:
        steps = [24 * i for i in range(1, 6)]  # 24, 48, 72, 96, 120
    
    logger.info(f"=== 모든 예보 시간 처리 시작 ===")
    logger.info(f"기준 날짜: {target_date}")
    logger.info(f"예보 데이터 디렉토리: {forecast_dir}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"처리할 예보 시간: {', '.join([f'+{step}h (D+{step//24})' for step in steps])}")
    
    # 모든 예보 시간 처리
    processed_files = []
    for step in steps:
        output_file = process_forecast_step(target_date, step, forecast_dir, output_dir, grid_ids_file, visualize, interpolate)
        if output_file:
            processed_files.append(output_file)
    
    logger.info(f"=== 모든 예보 시간 처리 완료 ===")
    logger.info(f"처리된 파일 수: {len(processed_files)}/{len(steps)}")
    
    return processed_files

def main():
    parser = argparse.ArgumentParser(description='ECMWF 예보 데이터 수집 및 전처리')
    
    parser.add_argument('--date', type=str, default=None,
                        help='예보 기준 날짜 (YYYYMMDD 형식, 기본값: 현재 날짜)')
    parser.add_argument('--forecast_dir', type=str, default=None,
                        help='예보 데이터 디렉토리 경로 (기본값: 프로젝트 루트/data/forecast)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 경로 (기본값: 프로젝트 루트/data/processed/forecast)')
    parser.add_argument('--steps', type=int, nargs='+', default=None,
                        help='처리할 예보 시간(시간) 목록 (기본값: 24, 48, 72, 96, 120)')
    parser.add_argument('--collect', action='store_true',
                        help='예보 데이터 수집 여부 (기본값: False)')
    parser.add_argument('--process_only', action='store_true',
                        help='기존 다운로드된 데이터만 처리 (기본값: False)')
    parser.add_argument('--grid_ids_file', type=str, default=None,
                        help='유효한 grid_id 목록이 저장된 JSON 파일 경로')
    parser.add_argument('--visualize', action='store_true',
                        help='필터링 결과 시각화 생성 (기본값: False)')
    parser.add_argument('--no_interpolate', action='store_true',
                        help='0.1° 격자로 선형 보간을 비활성화 (기본값: 보간 활성화)')
    
    args = parser.parse_args()
    
    # 예보 데이터 수집
    if args.collect and not args.process_only:
        logger.info("예보 데이터 수집 시작...")
        collect_forecast_data(
            target_date=args.date,
            output_dir=args.forecast_dir,
            steps=args.steps
        )
    
    # 예보 데이터 처리
    logger.info("예보 데이터 처리 시작...")
    process_all_forecast_steps(
        target_date=args.date,
        forecast_dir=args.forecast_dir,
        output_dir=args.output_dir,
        steps=args.steps,
        grid_ids_file=args.grid_ids_file,
        visualize=args.visualize,
        interpolate=not args.no_interpolate
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 