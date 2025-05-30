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
from pathlib import Path
from datetime import datetime, timedelta

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
        
        # 한국 영역 설정 (North/West/South/East)
        area = [39, 124, 33, 132]  # 북위, 서경, 남위, 동경
        logger.info(f"지역 설정: 북위 {area[0]}°, 서경 {area[1]}°, 남위 {area[2]}°, 동경 {area[3]}°")
        
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
                        # ECMWF Open Data에서 데이터 다운로드 (date=-1 사용)
                        client.retrieve(
                            date=-1,           # 가장 최신 예보 사용
                            time=forecast_hour,
                            step=step,
                            stream="oper",     # 고해상도 운영예보
                            type="fc",         # forecast
                            param=variable,
                            area=area,         # 한국 지역으로 제한
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

def grib_to_dataframe(grib_file):
    """
    GRIB2 파일을 Pandas DataFrame으로 변환합니다.
    
    Parameters:
    -----------
    grib_file : str
        입력 GRIB2 파일 경로
    
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
        
        # 데이터 배열 가져오기
        da = ds[var_name]
        
        # 위도/경도 확인
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            # 격자 형태의 위도/경도 데이터
            lats = ds.latitude.values
            lons = ds.longitude.values
            
            # 데이터 배열의 shape 확인
            logger.info(f"데이터 배열 shape: {da.shape}")
            
            # 2D 위도/경도 격자이면 평탄화
            rows = []
            
            # 위도/경도 평탄화 준비
            lat_vals = lats.flatten() if hasattr(lats, 'flatten') else lats
            lon_vals = lons.flatten() if hasattr(lons, 'flatten') else lons
            
            # 차원에 따른 처리
            if len(da.shape) == 2:  # [lat, lon]
                data_values = da.values
                
                for i, lat in enumerate(lat_vals):
                    for j, lon in enumerate(lon_vals):
                        try:
                            value = data_values[i, j]
                            row = {'latitude': lat, 'longitude': lon, var_name: value}
                            rows.append(row)
                        except IndexError:
                            continue
                            
            elif len(da.shape) == 3 and 'time' in ds.coords:  # [time, lat, lon]
                times = pd.to_datetime(ds.time.values)
                data_values = da.values
                
                for t, time_val in enumerate(times):
                    for i, lat in enumerate(lat_vals):
                        for j, lon in enumerate(lon_vals):
                            try:
                                value = data_values[t, i, j]
                                row = {
                                    'time': time_val,
                                    'latitude': lat, 
                                    'longitude': lon, 
                                    var_name: value
                                }
                                rows.append(row)
                            except IndexError:
                                continue
            else:
                logger.error(f"지원하지 않는 데이터 형식: {da.shape}")
                return None
                
            # DataFrame 생성
            df = pd.DataFrame(rows)
            
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

def process_forecast_step(target_date, step, forecast_dir, output_dir):
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
            
            # GRIB2 파일을 데이터프레임으로 변환
            df = grib_to_dataframe(file_path)
            
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
        
        logger.info(f"=== 예보 시간 +{step}h (D+{step//24}) 처리 완료 ===")
        logger.info(f"처리된 행 수: {len(final_df)}, 열 수: {len(final_df.columns)}")
        
        return output_csv
        
    except Exception as e:
        logger.error(f"예보 처리 중 오류 발생: {e}")
        logger.debug(traceback.format_exc())
        return None

def process_all_forecast_steps(target_date=None, forecast_dir=None, output_dir=None, steps=None):
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
        output_file = process_forecast_step(target_date, step, forecast_dir, output_dir)
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
        steps=args.steps
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 