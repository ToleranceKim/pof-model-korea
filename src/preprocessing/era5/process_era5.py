#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import gc
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('era5_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_nc_from_zip(zip_file_path, temp_dir):
    """
    ZIP 파일에서 NetCDF 파일을 추출합니다.
    
    Args:
        zip_file_path (str): ZIP 파일 경로
        temp_dir (str): 임시 디렉토리 경로
    
    Returns:
        list: 추출된 NetCDF 파일 경로 목록
    """
    # 문자열로 변환 (Path 객체 호환성 문제 해결)
    zip_file_path = str(zip_file_path)
    temp_dir = str(temp_dir)
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 추출된 파일 목록 확인
        extracted_files = os.listdir(temp_dir)
        nc_files = [os.path.join(temp_dir, f) for f in extracted_files if f.endswith('.nc')]
        
        if not nc_files:
            logger.error(f"ZIP 파일에 NetCDF 파일이 없습니다: {zip_file_path}")
            return []
        
        return nc_files
    
    except Exception as e:
        logger.error(f"ZIP 파일 압축 해제 중 오류 발생: {str(e)}")
        return []

def interpolate_era5_data(ds, var_name):
    """
    ERA5 데이터를 0.1° 격자로 선형 보간합니다.
    
    Args:
        ds (xarray.Dataset): 원본 데이터셋
        var_name (str): 보간할 변수 이름
    
    Returns:
        xarray.DataArray: 보간된 데이터
    """
    # 원본 격자 확인
    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values
    
    logger.info(f"변수 '{var_name}' 보간 중 (원본 격자: {len(lat_vals)}x{len(lon_vals)})")
    
    # 0.1° 격자 생성 (한국 지역: 33°N-39°N, 124°E-132°E)
    target_lat = np.arange(33, 39.1, 0.1)
    target_lon = np.arange(124, 132.1, 0.1)
    
    # xarray 데이터배열로 변환
    target_lat = xr.DataArray(target_lat, dims=['latitude'])
    target_lon = xr.DataArray(target_lon, dims=['longitude'])
    
    try:
        # 선형 보간 적용
        interpolated_var = ds[var_name].interp(
            latitude=target_lat,
            longitude=target_lon,
            method='linear'
        )
        
        return interpolated_var
    
    except Exception as e:
        logger.error(f"보간 중 오류 발생 (변수: {var_name}): {str(e)}")
        return None

def process_era5_file(file_path, intermediate_dir, output_dir, interpolate=True):
    """
    단일 ERA5 파일을 처리하는 함수
    
    Args:
        file_path (str): 입력 파일 경로 (.nc 또는 .zip)
        intermediate_dir (str): 중간 처리 결과 저장 디렉토리
        output_dir (str): 최종 출력 디렉토리
        interpolate (bool): 0.1° 격자로 보간할지 여부
    
    Returns:
        str: 처리된 파일 경로 또는 None (처리 실패 시)
    """
    # 모든 경로를 문자열로 변환 (Path 객체 호환성 문제 해결)
    file_path = str(file_path)
    intermediate_dir = str(intermediate_dir)
    output_dir = str(output_dir)
    
    # 파일명에서 연월 추출
    file_name = os.path.basename(file_path)
    year_month = None
    
    # 다양한 파일명 패턴 처리
    if '_' in file_name:
        parts = file_name.split('_')
        if len(parts) >= 3:
            year_month = parts[2].split('.')[0]  # era5_korea_YYYYMM.nc 패턴
    
    if not year_month and file_name.startswith('era5'):
        # 다른 패턴 시도
        try:
            year_month = ''.join(filter(str.isdigit, file_name))[:6]  # 처음 6자리 숫자를 연월로 사용
        except:
            pass
    
    if not year_month:
        logger.error(f"파일명에서 연월을 추출할 수 없습니다: {file_name}")
        return None
    
    logger.info(f"파일 처리 시작: {file_name} (연월: {year_month})")
    
    # 출력 파일 경로 설정
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    interpolated_file = os.path.join(intermediate_dir, f"era5_korea_{year_month}_0.1deg.nc")
    output_parquet = os.path.join(output_dir, f"era5_daily_{year_month}.parquet")
    output_csv = os.path.join(output_dir, f"era5_daily_{year_month}.csv")
    
    # 이미 처리된 파일이면 건너뜀
    if os.path.exists(output_parquet) and os.path.exists(output_csv):
        logger.info(f"이미 처리된 파일: {output_parquet}")
        return output_parquet
    
    start_time = time.time()
    
    try:
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        
        try:
            nc_files = []
            
            # ZIP 파일이면 압축 해제
            if file_path.endswith('.zip') or (file_path.endswith('.nc') and zipfile.is_zipfile(file_path)):
                nc_files = extract_nc_from_zip(file_path, temp_dir)
            else:
                # 일반 NetCDF 파일
                nc_files = [file_path]
            
            if not nc_files:
                logger.error(f"처리할 NetCDF 파일이 없습니다: {file_path}")
                return None
            
            # 모든 변수를 담을 딕셔너리 초기화
            all_variables = {}
            
            # 1. 보간 단계: 각 NetCDF 파일 처리
            for nc_file in nc_files:
                logger.info(f"NetCDF 파일 처리 중: {os.path.basename(nc_file)}")
                
                # NetCDF 파일 열기
                ds = xr.open_dataset(nc_file)
                logger.info(f"변수 목록: {list(ds.data_vars)}")
                
                # 각 변수별로 처리
                for var_name in ds.data_vars:
                    if interpolate:
                        # 0.1° 격자로 선형 보간
                        interpolated_var = interpolate_era5_data(ds, var_name)
                        if interpolated_var is not None:
                            all_variables[var_name] = interpolated_var
                    else:
                        # 보간 없이 원본 데이터 사용
                        all_variables[var_name] = ds[var_name]
                
                # 데이터셋 닫기
                ds.close()
            
            # 모든 변수를 하나의 데이터셋으로 결합
            if all_variables:
                combined_ds = xr.Dataset(all_variables)
                logger.info(f"결합된 데이터셋 변수: {list(combined_ds.data_vars)}")
                
                if interpolate:
                    # 보간된 결과 저장 (중간 단계)
                    logger.info(f"보간된 데이터 저장 중: {interpolated_file}")
                    combined_ds.to_netcdf(interpolated_file)
                
                # 2. 일별 통계 계산 단계
                logger.info("일별 통계 계산 중...")
                
                # 변수 매핑
                var_mapping = {
                    't2m': 't2m',    # 2m 기온
                    'd2m': 'td2m',   # 2m 이슬점 온도
                    'u10': '10u',    # 10m U 바람
                    'v10': '10v',    # 10m V 바람
                    'tp': 'tp'       # 강수량
                }
                
                # 시간 차원 확인
                time_dim = 'valid_time' if 'valid_time' in combined_ds.dims else 'time'
                
                # 시간 값 가져오기
                if time_dim in combined_ds.coords:
                    times = pd.to_datetime(combined_ds[time_dim].values)
                    logger.info(f"시간 차원: {time_dim}, 처음 몇 개 시간: {times[:5]}")
                else:
                    logger.error(f"시간 차원 '{time_dim}'이 좌표에 없습니다")
                    return None
                
                # 날짜별로 그룹화
                day_groups = {}
                for i, t in enumerate(times):
                    day = pd.Timestamp(t).date()
                    if day not in day_groups:
                        day_groups[day] = []
                    day_groups[day].append(i)
                
                logger.info(f"{len(day_groups)}개의 고유 날짜 발견")
                
                # 각 변수별 데이터프레임 생성
                all_results = []
                
                for var_name, target_name in var_mapping.items():
                    if var_name in combined_ds:
                        logger.info(f"변수 처리 중: {var_name} -> {target_name}")
                        
                        # 변수 데이터 가져오기
                        var_data = combined_ds[var_name].values
                        
                        # 날짜별 데이터 생성
                        daily_results = []
                        
                        # 위도, 경도 값 가져오기
                        lats = combined_ds.latitude.values
                        lons = combined_ds.longitude.values
                        
                        for day, indices in day_groups.items():
                            # 각 격자점마다 일별 평균/합계 계산
                            for lat_idx, lat in enumerate(lats):
                                for lon_idx, lon in enumerate(lons):
                                    # 해당 격자점의 하루동안의 값들
                                    grid_vals = [var_data[idx, lat_idx, lon_idx] for idx in indices]
                                    
                                    # 결측값이 아닌 값만 필터링
                                    valid_vals = [v for v in grid_vals if not np.isnan(v)]
                                    
                                    if valid_vals:
                                        # 일별 집계
                                        if var_name in ['t2m', 'd2m', 'u10', 'v10']:  # 평균 계산
                                            daily_val = np.mean(valid_vals)
                                        else:  # 합계 계산 (tp)
                                            daily_val = np.sum(valid_vals)
                                        
                                        # grid_id 계산 (0.1° 기준)
                                        lat_bin = int(np.floor(lat / 0.1))
                                        lon_bin = int(np.floor(lon / 0.1))
                                        grid_id = (lat_bin + 900) * 3600 + (lon_bin + 1800)
                                        
                                        daily_results.append({
                                            'acq_date': day,
                                            'grid_id': grid_id,
                                            target_name: daily_val,
                                            'latitude': lat,
                                            'longitude': lon
                                        })
                        
                        # 데이터프레임으로 변환
                        if daily_results:
                            var_df = pd.DataFrame(daily_results)
                            logger.info(f"변수 '{var_name}'에 대한 데이터프레임 생성 (형태: {var_df.shape})")
                            
                            # 필요한 컬럼만 선택
                            var_df = var_df[['acq_date', 'grid_id', 'latitude', 'longitude', target_name]]
                            all_results.append(var_df)
                        else:
                            logger.warning(f"변수 '{var_name}'에 대한 일별 결과 없음")
                        
                        # 메모리 정리
                        del var_data
                        del daily_results
                        gc.collect()
                
                # 모든 변수 병합
                if not all_results:
                    logger.error("성공적으로 처리된 변수가 없습니다")
                    return None
                
                # 첫 번째 데이터프레임을 기준으로 병합
                logger.info("데이터프레임 병합 중...")
                result_df = all_results[0]
                for i, df in enumerate(all_results[1:], 1):
                    var_name = list(var_mapping.values())[i]
                    logger.info(f"변수 '{var_name}' 데이터프레임 병합 중, 병합 전: {result_df.shape}")
                    result_df = pd.merge(result_df, df, on=['acq_date', 'grid_id', 'latitude', 'longitude'], how='outer')
                    logger.info(f"변수 '{var_name}' 병합 후: {result_df.shape}")
                
                # 결측값 확인
                null_counts = result_df.isnull().sum()
                logger.info(f"최종 데이터프레임의 결측값 개수:\n{null_counts}")
                
                # U10, V10 풍속 성분을 결합하여 10m 풍속 크기(wind10m) 계산
                if '10u' in result_df.columns and '10v' in result_df.columns:
                    logger.info("U와 V 성분에서 풍속 계산 중...")
                    result_df['wind10m'] = np.sqrt(result_df['10u']**2 + result_df['10v']**2)
                    logger.info(f"wind10m 범위: {result_df['wind10m'].min():.2f} ~ {result_df['wind10m'].max():.2f} m/s")
                
                # Parquet 형식으로 저장
                logger.info(f"Parquet 파일로 저장 중: {output_parquet}")
                result_df.to_parquet(output_parquet, index=False)
                
                # CSV 형식으로도 저장
                logger.info(f"CSV 파일로 저장 중: {output_csv}")
                result_df.to_csv(output_csv, index=False)
                
                logger.info(f"파일 저장 완료 (형태: {result_df.shape})")
                
                # 처리 시간 계산
                elapsed_time = time.time() - start_time
                logger.info(f"처리 완료: {file_name} (소요 시간: {elapsed_time:.2f}초)")
                
                return output_parquet
            else:
                logger.error("처리할 변수가 없습니다")
                return None
                
        finally:
            # 임시 디렉토리 정리
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"임시 디렉토리 정리 완료: {temp_dir}")
            except Exception as e:
                logger.warning(f"임시 디렉토리 정리 중 오류 발생: {e}")
    
    except Exception as e:
        logger.error(f"파일 처리 중 오류 발생: {file_path} - {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def combine_era5_data(input_dir, output_dir, viz_dir=None, date_range=None):
    """
    처리된 ERA5 데이터 파일들을 결합하는 함수
    
    Args:
        input_dir (str): 입력 디렉토리 경로 (일별 파일이 있는 곳)
        output_dir (str): 출력 디렉토리 경로
        viz_dir (str): 시각화 디렉토리 경로 (지정하지 않으면 output_dir/viz 사용)
        date_range (tuple): 날짜 범위 (시작일, 종료일) - YYYYMM 형식
    
    Returns:
        str: 결합된 데이터 파일 경로
    """
    # 문자열로 변환
    input_dir = str(input_dir)
    output_dir = str(output_dir)
    
    if viz_dir is None:
        viz_dir = os.path.join(output_dir, 'viz')
    else:
        viz_dir = str(viz_dir)
    
    logger.info(f"ERA5 데이터 파일 결합 중: {input_dir}")
    
    # 입력 디렉토리 내 모든 Parquet 파일 찾기
    parquet_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                    if f.startswith('era5_daily_') and f.endswith('.parquet')]
    
    if not parquet_files:
        logger.error(f"{input_dir}에 Parquet 파일이 없습니다")
        return None
    
    logger.info(f"{len(parquet_files)}개의 Parquet 파일 발견")
    
    # 날짜 범위 필터링
    if date_range:
        start_date, end_date = date_range
        filtered_files = []
        
        for file_path in parquet_files:
            file_name = os.path.basename(file_path)
            date_part = file_name.split('_')[-1].split('.')[0]  # era5_daily_YYYYMM.parquet에서 YYYYMM 추출
            
            if len(date_part) >= 6:  # YYYYMM 형식인지 확인
                year_month = date_part[:6]  # YYYYMM
                if start_date <= year_month <= end_date:
                    filtered_files.append(file_path)
        
        parquet_files = filtered_files
        logger.info(f"날짜 범위 {start_date}부터 {end_date}까지 필터링 후 {len(parquet_files)}개 파일")
    
    # 파일이 없으면 종료
    if not parquet_files:
        logger.error("필터링 후 처리할 파일이 없습니다")
        return None
    
    # 모든 Parquet 파일 로드 및 결합
    logger.info("파일 로드 및 결합 중...")
    all_dfs = []
    
    for file_path in parquet_files:
        logger.info(f"처리 중: {os.path.basename(file_path)}")
        
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"  - 형태: {df.shape}")
            
            # 기본 검증
            expected_columns = ['acq_date', 'grid_id', 'latitude', 'longitude']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"  - 경고: {file_path}에 누락된 열: {missing_columns}")
            
            # 날짜 형식 통일
            if 'acq_date' in df.columns:
                df['acq_date'] = pd.to_datetime(df['acq_date'])
            
            all_dfs.append(df)
        
        except Exception as e:
            logger.error(f"  - {file_path} 처리 중 오류 발생: {str(e)}")
    
    # 결합
    if not all_dfs:
        logger.error("결합할 유효한 데이터프레임이 없습니다")
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"결합된 데이터 형태: {combined_df.shape}")
    
    # 중복 제거 (같은 날짜의 같은 grid_id 값)
    if 'acq_date' in combined_df.columns and 'grid_id' in combined_df.columns:
        before_dedup = combined_df.shape[0]
        combined_df = combined_df.drop_duplicates(subset=['acq_date', 'grid_id'])
        after_dedup = combined_df.shape[0]
        
        if before_dedup > after_dedup:
            logger.info(f"{before_dedup - after_dedup}개의 중복 레코드 제거됨")
    
    # 풍속 계산 (누락된 경우)
    if '10u' in combined_df.columns and '10v' in combined_df.columns and 'wind10m' not in combined_df.columns:
        logger.info("풍속 계산 중...")
        u = combined_df['10u']
        v = combined_df['10v']
        combined_df['wind10m'] = np.sqrt(u**2 + v**2)
    
    # 데이터 정렬
    if 'acq_date' in combined_df.columns:
        combined_df = combined_df.sort_values(['acq_date', 'grid_id'])
    
    # 결과 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # 파일명에 날짜 범위 정보 추가
    date_info = ""
    if date_range:
        date_info = f"_{date_range[0]}_{date_range[1]}"
    else:
        # 데이터에서 날짜 범위 추출
        if 'acq_date' in combined_df.columns:
            min_date = combined_df['acq_date'].min()
            max_date = combined_df['acq_date'].max()
            if pd.notnull(min_date) and pd.notnull(max_date):
                min_str = min_date.strftime('%Y%m')
                max_str = max_date.strftime('%Y%m')
                date_info = f"_{min_str}_{max_str}"
    
    # 파일 저장
    # viz 디렉토리에 저장하던 것을 processed_data 디렉토리로 변경
    base_dir = os.path.dirname(output_dir)  # processed_data 디렉토리
    combined_parquet_path = os.path.join(base_dir, f'era5_daily_combined{date_info}.parquet')
    combined_csv_path = os.path.join(base_dir, f'era5_daily_combined{date_info}.csv')
    
    # Parquet 파일 저장
    combined_df.to_parquet(combined_parquet_path, index=False)
    logger.info(f"결합된 데이터 저장 완료: {combined_parquet_path}")
    
    # CSV 파일도 저장
    combined_df.to_csv(combined_csv_path, index=False)
    logger.info(f"결합된 데이터 CSV 저장 완료: {combined_csv_path}")
    
    # 시각화용 파일도 저장
    viz_parquet_path = os.path.join(viz_dir, f'era5_daily_all.parquet')
    combined_df.to_parquet(viz_parquet_path, index=False)
    logger.info(f"시각화용 데이터 저장 완료: {viz_parquet_path}")
    
    # 날짜 범위별 파일도 저장
    if date_info:
        range_parquet_path = os.path.join(output_dir, f'era5_daily{date_info}.parquet')
        combined_df.to_parquet(range_parquet_path, index=False)
        logger.info(f"범위별 데이터 저장 완료: {range_parquet_path}")
    
    # 데이터 시각화 및 요약
    visualize_combined_data(combined_df, viz_dir, date_info)
    
    return combined_parquet_path

def visualize_combined_data(df, viz_dir, date_info=""):
    """
    결합된 데이터를 시각화하는 함수
    
    Args:
        df: 결합된 데이터프레임
        viz_dir: 시각화 출력 디렉토리
        date_info: 날짜 정보 문자열
    """
    try:
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. 데이터 요약 정보
        logger.info("\n===== 데이터 요약 =====")
        logger.info(f"레코드: {df.shape[0]}")
        logger.info(f"고유 날짜: {df['acq_date'].nunique()}")
        logger.info(f"고유 격자점: {df['grid_id'].nunique()}")
        logger.info(f"날짜 범위: {df['acq_date'].min()} ~ {df['acq_date'].max()}")
        
        # 2. 결측치 시각화
        plt.figure(figsize=(12, 6))
        missing = df.isnull().sum() / len(df) * 100
        missing = missing.sort_values(ascending=False)
        sns.barplot(x=missing.index, y=missing.values)
        plt.title('결측값 (%)')
        plt.xlabel('변수')
        plt.ylabel('백분율')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'missing_values{date_info}.png'), dpi=300)
        plt.close()
        
        # 3. 격자점 분포 시각화
        if df.shape[0] <= 100000:  # 데이터가 너무 많으면 샘플링
            sample_df = df
        else:
            sample_df = df.sample(100000, random_state=42)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(sample_df['longitude'], sample_df['latitude'], 
                    s=1, alpha=0.5, c='blue')
        plt.title('격자점의 공간 분포')
        plt.xlabel('경도')
        plt.ylabel('위도')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'grid_distribution{date_info}.png'), dpi=300)
        plt.close()
        
        # 4. 기온 분포 시각화
        if 't2m' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.histplot(df['t2m'] - 273.15, kde=True)  # 켈빈을 섭씨로 변환
            plt.title('기온 분포 (°C)')
            plt.xlabel('기온 (°C)')
            plt.ylabel('빈도')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'temperature_distribution{date_info}.png'), dpi=300)
            plt.close()
        
        # 5. 시계열 시각화 (평균 기온)
        if 't2m' in df.columns and 'acq_date' in df.columns:
            daily_temp = df.groupby('acq_date')['t2m'].mean() - 273.15  # 켈빈을 섭씨로 변환
            plt.figure(figsize=(14, 6))
            daily_temp.plot()
            plt.title('일별 평균 기온 (°C)')
            plt.xlabel('날짜')
            plt.ylabel('평균 기온 (°C)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'temp_timeseries{date_info}.png'), dpi=300)
            plt.close()
        
        logger.info(f"시각화 저장 완료: {viz_dir}")
    
    except Exception as e:
        logger.error(f"시각화 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """
    메인 함수
    """
    # 인자 파싱
    parser = argparse.ArgumentParser(description='ERA5 데이터 처리 스크립트')
    parser.add_argument('--input_dir', type=str, default='data/raw',
                        help='원본 ERA5 파일이 있는 디렉토리 경로')
    parser.add_argument('--intermediate_dir', type=str, default='processed_data/era5',
                        help='중간 처리 파일을 저장할 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default='processed_data/era5_daily',
                        help='처리된 파일을 저장할 디렉토리 경로')
    parser.add_argument('--viz_dir', type=str, default='processed_data/viz',
                        help='시각화 파일을 저장할 디렉토리 경로')
    parser.add_argument('--start_date', type=str, default=None,
                        help='처리 시작 날짜 (YYYYMM 형식)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='처리 종료 날짜 (YYYYMM 형식)')
    parser.add_argument('--interpolate', type=bool, default=True,
                        help='0.1도 격자로 보간할지 여부')
    parser.add_argument('--combine', type=bool, default=True,
                        help='모든 파일을 결합할지 여부')
    
    args = parser.parse_args()
    
    # 디렉토리 경로 설정
    input_dir = Path(args.input_dir)
    intermediate_dir = Path(args.intermediate_dir)
    output_dir = Path(args.output_dir)
    viz_dir = Path(args.viz_dir)
    
    # 디렉토리 생성
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 입력 파일 목록 가져오기
    era5_files = []
    
    # 다양한 파일 형식 처리 (.nc, .zip)
    for file_ext in ['.nc', '.zip']:
        files = list(input_dir.glob(f'era5_korea_*{file_ext}'))
        era5_files.extend(files)
    
    era5_files.sort()
    
    if not era5_files:
        logger.error(f"입력 디렉토리에 처리할 파일이 없습니다: {input_dir}")
        return
    
    # 날짜 필터링
    if args.start_date or args.end_date:
        filtered_files = []
        
        # 날짜 범위 설정
        start = args.start_date if args.start_date else "000000"
        end = args.end_date if args.end_date else "999999"
        
        for file_path in era5_files:
            file_name = file_path.name
            # 파일명에서 날짜 부분 추출 (예: era5_korea_202001.nc -> 202001)
            date_part = ""
            parts = file_name.split('_')
            if len(parts) > 2:
                date_part = parts[2].split('.')[0]
            
            if start <= date_part <= end:
                filtered_files.append(file_path)
        
        era5_files = filtered_files
        logger.info(f"날짜 필터링 후 처리할 파일: {len(era5_files)}개")
    
    # 각 파일 순차 처리
    processed_files = []
    
    for file_path in era5_files:
        try:
            result = process_era5_file(
                file_path, 
                intermediate_dir,
                output_dir,
                interpolate=args.interpolate
            )
            
            if result:
                processed_files.append(result)
            
            # 메모리 정리
            gc.collect()
        
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {file_path} - {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 결과 확인
    if not processed_files:
        logger.error("처리된 파일이 없습니다.")
        return
    
    logger.info(f"총 {len(processed_files)}/{len(era5_files)} 파일 처리됨")
    
    # 데이터 결합 (선택적)
    if args.combine and len(processed_files) > 0:
        logger.info("모든 파일 처리 완료. 데이터 결합 시작...")
        
        date_range = None
        if args.start_date and args.end_date:
            date_range = (args.start_date, args.end_date)
        
        combined_file = combine_era5_data(
            output_dir,
            output_dir,
            viz_dir,
            date_range
        )
        
        if combined_file:
            logger.info(f"데이터 결합 완료: {combined_file}")
    
    # 실행 시간 계산 및 출력
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("="*80)
    logger.info(f"ERA5 데이터 처리 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"총 실행 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.2f}초")
    logger.info("="*80)

if __name__ == "__main__":
    main() 