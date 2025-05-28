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
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import cdsapi
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 도우미 함수
# ----------------------------------------------------------------------------

def process_day_data(args):
    """
    병렬 처리를 위한 일별 데이터 처리 함수
    외부 함수로 정의하여 병렬 처리 가능하게 함
    """
    day, indices, var_data, lat_vals, lon_vals, var_name, target_name, interpolate_to_01deg = args
    
    daily_results = []
    
    if interpolate_to_01deg:
        # 0.1° 격자 생성 (한국 지역: 33°N-39°N, 124°E-132°E)
        new_lats = np.arange(33.0, 39.1, 0.1)
        new_lons = np.arange(124.0, 132.1, 0.1)
        
        # 각 새로운 격자점에 대해 이중선형 보간 수행
        for lat in new_lats:
            for lon in new_lons:
                # grid_id 계산 (0.1° 기준)
                lat_bin = int(np.floor(lat / 0.1))
                lon_bin = int(np.floor(lon / 0.1))
                grid_id = (lat_bin + 900) * 3600 + (lon_bin + 1800)
                
                # 이 격자점에 대해 각 시간대별로 이중선형 보간
                daily_vals = []
                
                for idx in indices:
                    # 보간을 위한 4개 인접 격자점 찾기
                    lat_indices = np.searchsorted(lat_vals, lat)
                    lon_indices = np.searchsorted(lon_vals, lon)
                    
                    # 경계 처리
                    if lat_indices == 0:
                        lat_indices = 1
                    if lat_indices >= len(lat_vals):
                        lat_indices = len(lat_vals) - 1
                    if lon_indices == 0:
                        lon_indices = 1
                    if lon_indices >= len(lon_vals):
                        lon_indices = len(lon_vals) - 1
                    
                    # 인접 격자점 좌표
                    lat1 = lat_vals[lat_indices-1]
                    lat2 = lat_vals[lat_indices]
                    lon1 = lon_vals[lon_indices-1]
                    lon2 = lon_vals[lon_indices]
                    
                    # 상대 거리 계산
                    t = (lat - lat1) / (lat2 - lat1) if lat2 != lat1 else 0
                    u = (lon - lon1) / (lon2 - lon1) if lon2 != lon1 else 0
                    
                    # 인접 격자점 값
                    try:
                        Q11 = var_data[idx, lat_indices-1, lon_indices-1]
                        Q12 = var_data[idx, lat_indices-1, lon_indices]
                        Q21 = var_data[idx, lat_indices, lon_indices-1]
                        Q22 = var_data[idx, lat_indices, lon_indices]
                        
                        # 결측값 확인
                        if not (np.isnan(Q11) or np.isnan(Q12) or np.isnan(Q21) or np.isnan(Q22)):
                            # 이중선형 보간 공식
                            interp_val = (1-t)*(1-u)*Q11 + (1-t)*u*Q12 + t*(1-u)*Q21 + t*u*Q22
                            daily_vals.append(interp_val)
                    except IndexError:
                        # 인덱스 오류 발생 시 건너뜀
                        continue
                
                # 유효한 값이 있는 경우에만 처리
                if daily_vals:
                    # 일별 집계
                    if var_name in ['t2m', 'd2m', 'u10', 'v10']:  # 평균 계산
                        daily_val = np.mean(daily_vals)
                    else:  # 합계 계산 (tp)
                        daily_val = np.sum(daily_vals)
                    
                    daily_results.append({
                        'acq_date': day,
                        'grid_id': grid_id,
                        target_name: daily_val,
                        'latitude': lat,
                        'longitude': lon
                    })
    else:
        # 기존 로직: 원본 격자 사용
        for lat_idx, lat in enumerate(lat_vals):
            for lon_idx, lon in enumerate(lon_vals):
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
    
    return daily_results

# ----------------------------------------------------------------------------
# 핵심 처리 함수
# ----------------------------------------------------------------------------

def process_era5_file(file_path, output_dir, use_parallel=True, interpolate_to_01deg=True):
    """
    ERA5 파일(NetCDF 또는 CSV)을 처리하는 통합 함수
    
    Args:
        file_path (str): 입력 파일 경로 (.nc, .zip, .csv)
        output_dir (str): 출력 디렉토리 경로
        use_parallel (bool): 병렬 처리 사용 여부
        interpolate_to_01deg (bool): 0.1° 격자로 보간 여부
    
    Returns:
        str: 처리된 파일 경로 또는 None (처리 실패 시)
    """
    print(f"Processing ERA5 file: {file_path}")
    start_time = time.time()
    
    # ERA5 데이터의 경우 파일 확장자와 상관없이 ZIP 파일로 처리
    # process_weather.py와 동일한 방식으로 구현
    if file_path.endswith('.nc') or file_path.endswith('.zip'):
        return process_era5_from_zip(file_path, output_dir, use_parallel, interpolate_to_01deg)
    elif file_path.endswith('.csv'):
        return process_era5_from_csv(file_path, output_dir, use_parallel, interpolate_to_01deg)
    else:
        print(f"Unsupported file format: {file_path}")
        return None

def process_era5_from_zip(zip_file_path, output_dir, use_parallel=True, interpolate_to_01deg=True):
    """
    ZIP 파일에서 NetCDF를 추출하고 처리하는 함수
    
    Args:
        zip_file_path: ZIP 파일 경로
        output_dir: 출력 디렉토리
        use_parallel: 병렬 처리 사용 여부
        interpolate_to_01deg: 0.1° 격자로 보간 여부
    
    Returns:
        처리된 파일 경로 또는 None (처리 실패 시)
    """
    # 파일명에서 연월 추출
    file_name = os.path.basename(zip_file_path)
    
    # 연월 추출 패턴 확인
    if '_' in file_name:
        year_month = file_name.split('_')[-1].split('.')[0]  # era5_korea_YYYYMM.zip 패턴
    else:
        year_month = file_name.split('.')[0]  # 다른 패턴
    
    # 이미 처리된 파일인지 확인
    output_base = os.path.join(output_dir, f"era5_korea_{year_month}")
    output_parquet = f"{output_base}.parquet"
    output_csv = f"{output_base}.csv"
    
    if os.path.exists(output_parquet) and os.path.exists(output_csv):
        print(f"File already processed: {output_parquet} and {output_csv}")
        return output_parquet
    
    print(f"Processing ZIP file: {zip_file_path}")
    
    try:
        # 임시 폴더 생성
        temp_dir = tempfile.mkdtemp()
        nc_path = None
        
        try:
            # ZIP 파일 압축 해제
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 추출된 파일 목록 확인
            extracted_files = os.listdir(temp_dir)
            print(f"Extracted files: {extracted_files}")
            
            # NetCDF 파일 찾기
            nc_files = [f for f in extracted_files if f.endswith('.nc')]
            if not nc_files:
                print(f"No NetCDF files found in ZIP: {zip_file_path}")
                return None
            
            nc_path = os.path.join(temp_dir, nc_files[0])
            print(f"Found NetCDF file: {nc_path}")
            
            # NetCDF 파일 처리
            return process_era5_from_netcdf(nc_path, output_dir, use_parallel, interpolate_to_01deg, 
                                          from_zip=True, orig_file_path=zip_file_path)
        
        finally:
            # 임시 파일 정리
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"Temporary directory cleaned: {temp_dir}")
            except Exception as e:
                print(f"Error cleaning temporary directory: {str(e)}")
    
    except Exception as e:
        print(f"Error processing ERA5 ZIP file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_era5_from_netcdf(nc_file_path, output_dir, use_parallel=True, interpolate_to_01deg=True, 
                           from_zip=False, orig_file_path=None):
    """
    NetCDF 파일에서 ERA5 데이터를 처리하는 함수
    
    Args:
        nc_file_path: NetCDF 파일 경로
        output_dir: 출력 디렉토리
        use_parallel: 병렬 처리 사용 여부
        interpolate_to_01deg: 0.1° 격자로 보간 여부
        from_zip: ZIP 파일에서 추출된 파일인지 여부
        orig_file_path: 원본 ZIP 파일 경로 (from_zip=True인 경우)
    
    Returns:
        처리된 파일 경로 또는 None (처리 실패 시)
    """
    try:
        ds = None
        
        try:
            # 파일명에서 연월 추출
            if from_zip and orig_file_path:
                base_name = os.path.basename(orig_file_path)
                date_part = base_name.split('_')[-1].split('.')[0] if '_' in base_name else "unknown"
            else:
                base_name = os.path.basename(nc_file_path)
                date_part = base_name.split('_')[-1].split('.')[0] if '_' in base_name else "unknown"
            
            # 출력 파일 경로 설정
            output_base = os.path.join(output_dir, f"era5_korea_{date_part}")
            output_parquet = f"{output_base}.parquet"
            output_csv = f"{output_base}.csv"
            
            # 이미 처리된 파일인지 확인
            if os.path.exists(output_parquet) and os.path.exists(output_csv):
                print(f"File already processed: {output_parquet} and {output_csv}")
                return output_parquet
            
            # NetCDF 파일 열기 - process_weather.py와 동일한 방식 사용
            try:
                # 파일 내용 확인 (처음 10바이트 정도)
                with open(nc_file_path, 'rb') as f:
                    header = f.read(10)
                    print(f"File header: {header}")
                
                # 직접 파일 열기 시도
                ds = xr.open_dataset(nc_file_path)
                print("Successfully opened dataset with default engine")
            except Exception as e:
                print(f"Error opening NetCDF file: {str(e)}")
                return None
            
            print(f"Dataset opened successfully. Dimensions: {list(ds.dims)}")
            print(f"Variables: {list(ds.variables)}")
            
            # 필요한 변수 확인
            var_mapping = {
                't2m': 't2m',   # 2m 기온
                'd2m': 'td2m',  # 2m 이슬점 온도
                'u10': '10u',   # 10m U 바람
                'v10': '10v',   # 10m V 바람
                'tp': 'tp'      # 강수량
            }
            
            # 시간 차원 확인
            time_dim = None
            for dim_name in ['time', 'valid_time']:
                if dim_name in ds.dims:
                    time_dim = dim_name
                    break
            
            if time_dim is None:
                print("No time dimension found in dataset")
                return None
            
            # 시간 값 가져오기
            time_vals = pd.to_datetime(ds[time_dim].values)
            
            # 원본 좌표 정보 확인
            lat_vals = ds.latitude.values
            lon_vals = ds.longitude.values
            
            print(f"Original grid: {len(lat_vals)}x{len(lon_vals)} points")
            print(f"Latitude range: {lat_vals.min():.4f}° ~ {lat_vals.max():.4f}°")
            print(f"Longitude range: {lon_vals.min():.4f}° ~ {lon_vals.max():.4f}°")
            
            # 보간 대상 격자 (0.1°)
            if interpolate_to_01deg:
                # 0.1° 격자 (한국 지역: 33°N-39°N, 124°E-132°E)
                new_lats = np.arange(33.0, 39.1, 0.1)
                new_lons = np.arange(124.0, 132.1, 0.1)
                print(f"Will interpolate to 0.1° grid: {len(new_lats)}x{len(new_lons)} points")
            
            # 날짜별로 그룹화
            day_groups = {}
            for i, t in enumerate(time_vals):
                day = pd.Timestamp(t).date()
                if day not in day_groups:
                    day_groups[day] = []
                day_groups[day].append(i)
            
            print(f"Found {len(day_groups)} unique days")
            
            # 각 변수별 데이터프레임 생성 및 병합
            all_results = []
            
            for var_name, target_name in var_mapping.items():
                if var_name in ds:
                    print(f"\nProcessing variable: {var_name} -> {target_name}")
                    
                    # 변수 데이터 가져오기 (numpy 배열로)
                    var_data = ds[var_name].values
                    print(f"Variable shape: {var_data.shape}")
                    
                    # 병렬 처리를 위한 데이터 준비
                    day_items = list(day_groups.items())
                    
                    # 병렬 처리를 위한 인자 준비
                    process_args = [(day, indices, var_data, lat_vals, lon_vals, var_name, target_name, interpolate_to_01deg) 
                                    for day, indices in day_items]
                    
                    # 결과 저장할 리스트
                    all_daily_results = []
                    
                    if use_parallel and len(day_items) > 1:
                        # 병렬 처리 실행
                        # 사용 가능한 CPU 코어 수 확인 (전체의 80% 사용)
                        num_cores = max(1, int(mp.cpu_count() * 0.8))
                        print(f"Using {num_cores} CPU cores for parallel processing")
                        
                        with ProcessPoolExecutor(max_workers=num_cores) as executor:
                            futures = [executor.submit(process_day_data, arg) for arg in process_args]
                            
                            # 진행 상황 표시
                            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {var_name}"):
                                try:
                                    daily_results = future.result()
                                    all_daily_results.extend(daily_results)
                                except Exception as e:
                                    print(f"Error processing day: {e}")
                    else:
                        # 순차 처리
                        print(f"Processing {var_name} sequentially...")
                        for args in tqdm(process_args, desc=f"Processing {var_name}"):
                            daily_results = process_day_data(args)
                            all_daily_results.extend(daily_results)
                    
                    # 데이터프레임으로 변환
                    var_df = pd.DataFrame(all_daily_results)
                    print(f"Created dataframe for {var_name} with shape: {var_df.shape}")
                    
                    if not var_df.empty:
                        # 필요한 컬럼만 선택 (위경도 컬럼 포함)
                        var_df = var_df[['acq_date', 'grid_id', 'latitude', 'longitude', target_name]]
                        all_results.append(var_df)
                    else:
                        print(f"Warning: Empty dataframe for {var_name}")
                    
                    # 메모리 정리
                    del var_data
                    del all_daily_results
                    gc.collect()
            
            # 모든 변수 병합
            if not all_results:
                print("No variables were successfully processed")
                return None
            
            # 첫 번째 데이터프레임을 기준으로 병합
            print("\nMerging dataframes...")
            result_df = all_results[0]
            for i, df in enumerate(all_results[1:], 1):
                var_name = list(var_mapping.values())[i]
                print(f"Merging {var_name} dataframe, before merge: {result_df.shape}")
                result_df = pd.merge(result_df, df, on=['acq_date', 'grid_id', 'latitude', 'longitude'], how='outer')
                print(f"After merge with {var_name}: {result_df.shape}")
            
            # 결측값 확인
            null_counts = result_df.isnull().sum()
            print(f"\nNull value counts in final dataframe:\n{null_counts}")
            
            # U10, V10 풍속 성분을 결합하여 10m 풍속 크기(wind10m) 계산
            if '10u' in result_df.columns and '10v' in result_df.columns:
                print("Calculating wind speed from U and V components...")
                result_df['wind10m'] = np.sqrt(result_df['10u']**2 + result_df['10v']**2)
                print(f"wind10m range: {result_df['wind10m'].min():.2f} ~ {result_df['wind10m'].max():.2f} m/s")
            
            # 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # Parquet 형식으로 저장
            print(f"\nSaving to {output_parquet}")
            result_df.to_parquet(output_parquet, index=False)
            
            # CSV 형식으로도 저장
            print(f"Saving to {output_csv}")
            result_df.to_csv(output_csv, index=False)
            
            print(f"Successfully saved files with shape {result_df.shape}")
            
            return output_parquet
            
        finally:
            # 명시적으로 메모리 해제
            if ds is not None:
                try:
                    ds.close()
                except:
                    pass
                del ds
            
            # 메모리 정리
            gc.collect()
    
    except Exception as e:
        print(f"Error processing NetCDF file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_era5_from_csv(csv_file_path, output_dir, use_parallel=True, interpolate_to_01deg=True):
    """
    CSV 파일에서 ERA5 데이터를 처리하는 함수
    
    Args:
        csv_file_path: CSV 파일 경로
        output_dir: 출력 디렉토리
        use_parallel: 병렬 처리 사용 여부
        interpolate_to_01deg: 0.1° 격자로 보간 여부
    
    Returns:
        처리된 파일 경로 또는 None (처리 실패 시)
    """
    try:
        # 파일명에서 연월 추출
        file_name = os.path.basename(csv_file_path)
        date_part = file_name.split('_')[2].split('.')[0] if len(file_name.split('_')) > 2 else "unknown"
        
        # 출력 파일 경로 설정
        output_base = os.path.join(output_dir, f"era5_interp_{date_part}")
        output_parquet = f"{output_base}.parquet"
        output_csv = f"{output_base}.csv"
        
        # 이미 처리된 파일인지 확인
        if os.path.exists(output_parquet) and os.path.exists(output_csv):
            print(f"File already processed: {output_parquet} and {output_csv}")
            return output_parquet
        
        print(f"Processing CSV file: {csv_file_path}")
        
        # CSV 파일 읽기
        df = pd.read_csv(csv_file_path)
        print(f"CSV loaded. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # 날짜 열 처리
        if 'acq_date' in df.columns:
            df['acq_date'] = pd.to_datetime(df['acq_date'])
        
        # 위도/경도 범위 확인
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_min = df['latitude'].min()
            lat_max = df['latitude'].max()
            lon_min = df['longitude'].min()
            lon_max = df['longitude'].max()
            
            print(f"Original latitude range: {lat_min} to {lat_max}")
            print(f"Original longitude range: {lon_min} to {lon_max}")
            
            # 새로운 0.1° 격자 좌표 생성 (한국 지역: 33°N-39°N, 124°E-132°E)
            if interpolate_to_01deg:
                new_lats = np.arange(33.0, 39.1, 0.1)  # 33°N부터 39°N까지 0.1° 간격
                new_lons = np.arange(124.0, 132.1, 0.1)  # 124°E부터 132°E까지 0.1° 간격
                print(f"Will interpolate to 0.1° grid: {len(new_lats)}x{len(new_lons)} points")
            
            # 보간된 결과를 저장할 데이터프레임 초기화
            all_results = []
            
            # 날짜별로 처리
            day_groups = df.groupby('acq_date')
            
            # 병렬 처리 준비
            if use_parallel and len(day_groups) > 1:
                # 사용 가능한 CPU 코어 수 확인 (전체의 80% 사용)
                num_cores = max(1, int(mp.cpu_count() * 0.8))
                print(f"Using {num_cores} CPU cores for parallel processing")
                
                # 병렬 처리 함수
                def process_day_from_csv(day_data):
                    day, day_df = day_data
                    print(f"Processing day: {day}, with {len(day_df)} points")
                    
                    daily_results = []
                    
                    # 원본 좌표와 값
                    points = day_df[['longitude', 'latitude']].values
                    
                    # 보간할 변수들
                    var_names = ['t2m', 'td2m', '10u', '10v', 'tp']
                    
                    # 모든 새로운 격자점에 대한 결과
                    for lat in new_lats:
                        for lon in new_lons:
                            # grid_id 계산 (0.1° 기준)
                            lat_bin = int(np.floor(lat / 0.1))
                            lon_bin = int(np.floor(lon / 0.1))
                            grid_id = (lat_bin + 900) * 3600 + (lon_bin + 1800)
                            
                            row_data = {
                                'acq_date': day,
                                'latitude': lat,
                                'longitude': lon,
                                'grid_id': grid_id
                            }
                            
                            # 각 변수에 대해 가장 가까운 점 기반 보간
                            for var_name in var_names:
                                if var_name in day_df.columns:
                                    # 해당 지점과 가장 가까운 4개 점 찾기
                                    dists = np.sqrt((points[:, 0] - lon)**2 + (points[:, 1] - lat)**2)
                                    closest_idx = np.argsort(dists)[:4]
                                    
                                    # 가중치 계산 (거리의 역수)
                                    weights = 1.0 / (dists[closest_idx] + 1e-10)
                                    weights = weights / np.sum(weights)
                                    
                                    # 가중 평균 계산
                                    values = day_df[var_name].values[closest_idx]
                                    interp_value = np.sum(values * weights)
                                    
                                    row_data[var_name] = interp_value
                            
                            daily_results.append(row_data)
                    
                    return daily_results
                
                # 병렬 처리 실행
                with ProcessPoolExecutor(max_workers=num_cores) as executor:
                    futures = [executor.submit(process_day_from_csv, (day, group)) 
                              for day, group in day_groups]
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing days"):
                        try:
                            daily_results = future.result()
                            all_results.extend(daily_results)
                        except Exception as e:
                            print(f"Error processing day: {e}")
            
            else:
                # 순차 처리
                for day, day_df in tqdm(day_groups, desc="Processing days"):
                    print(f"Processing day: {day}, with {len(day_df)} points")
                    
                    # 원본 좌표와 값
                    points = day_df[['longitude', 'latitude']].values
                    
                    # 보간할 변수들
                    var_names = ['t2m', 'td2m', '10u', '10v', 'tp']
                    
                    # 모든 새로운 격자점에 대한 결과
                    for lat in new_lats:
                        for lon in new_lons:
                            # grid_id 계산 (0.1° 기준)
                            lat_bin = int(np.floor(lat / 0.1))
                            lon_bin = int(np.floor(lon / 0.1))
                            grid_id = (lat_bin + 900) * 3600 + (lon_bin + 1800)
                            
                            row_data = {
                                'acq_date': day,
                                'latitude': lat,
                                'longitude': lon,
                                'grid_id': grid_id
                            }
                            
                            # 각 변수에 대해 가장 가까운 점 기반 보간
                            for var_name in var_names:
                                if var_name in day_df.columns:
                                    # 해당 지점과 가장 가까운 4개 점 찾기
                                    dists = np.sqrt((points[:, 0] - lon)**2 + (points[:, 1] - lat)**2)
                                    closest_idx = np.argsort(dists)[:4]
                                    
                                    # 가중치 계산 (거리의 역수)
                                    weights = 1.0 / (dists[closest_idx] + 1e-10)
                                    weights = weights / np.sum(weights)
                                    
                                    # 가중 평균 계산
                                    values = day_df[var_name].values[closest_idx]
                                    interp_value = np.sum(values * weights)
                                    
                                    row_data[var_name] = interp_value
                            
                            all_results.append(row_data)
            
            # 풍속 계산 (10u, 10v에서 계산)
            for row in all_results:
                if '10u' in row and '10v' in row:
                    u = row['10u']
                    v = row['10v']
                    if not (np.isnan(u) or np.isnan(v)):
                        row['wind10m'] = np.sqrt(u**2 + v**2)
            
            # 결과를 데이터프레임으로 변환
            result_df = pd.DataFrame(all_results)
            
            # 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # Parquet 형식으로 저장
            print(f"\nSaving to {output_parquet}")
            result_df.to_parquet(output_parquet, index=False)
            
            # CSV 형식으로도 저장
            print(f"Saving to {output_csv}")
            result_df.to_csv(output_csv, index=False)
            
            print(f"Successfully saved files with shape {result_df.shape}")
            
            return output_parquet
        
        else:
            print("Error: Latitude or longitude columns not found in CSV file")
            return None
    
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_multiple_files_parallel(file_list, output_dir, max_workers=None, 
                                  use_parallel_processing=True, interpolate_to_01deg=True):
    """
    여러 파일을 병렬로 처리하는 함수
    
    Args:
        file_list: 처리할 파일 목록
        output_dir: 출력 디렉토리
        max_workers: 최대 작업자 수 (None이면 CPU 코어 수의 50%를 사용)
        use_parallel_processing: 각 파일 내부 처리에 병렬 처리 사용 여부
        interpolate_to_01deg: 0.1° 격자로 보간 여부
    
    Returns:
        처리된 파일 목록
    """
    if max_workers is None:
        max_workers = max(1, int(mp.cpu_count() * 0.5))
    
    print(f"Processing {len(file_list)} files using {max_workers} workers")
    
    processed_files = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 각 파일에 대한 작업 제출
        futures = {}
        for file_path in file_list:
            future = executor.submit(
                process_era5_file, 
                file_path, 
                output_dir,
                use_parallel_processing,
                interpolate_to_01deg
            )
            futures[future] = file_path
        
        # 완료된 작업 처리
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file_path = futures[future]
            try:
                result = future.result()
                if result:
                    processed_files.append(result)
                    print(f"Successfully processed: {file_path}")
                else:
                    print(f"Failed to process: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
    
    return processed_files

class ERA5Processor:
    def __init__(self, base_dir='data/raw'):
        self.base_dir = Path(base_dir)
        
        # Define target grid (0.1 degree resolution)
        self.target_lat = np.arange(33, 39, 0.1)
        self.target_lon = np.arange(124, 132, 0.1)
        
    def get_era5_files(self, year):
        """Get ERA5 files for a specific year"""
        pattern = f"era5_korea_{year}*.nc"
        files = list(self.base_dir.glob(pattern))
        if not files:
            logger.warning(f"No ERA5 files found for year {year}")
            return []
        return sorted(files)

    def interpolate_to_target_grid(self, ds):
        """Interpolate ERA5 data to 0.1 degree resolution"""
        # Create target grid
        target_lat = xr.DataArray(self.target_lat, dims=['latitude'])
        target_lon = xr.DataArray(self.target_lon, dims=['longitude'])
        
        # Interpolate each variable
        interpolated = {}
        for var in ds.data_vars:
            interpolated[var] = ds[var].interp(
                latitude=target_lat,
                longitude=target_lon,
                method='linear'
            )
        
        return xr.Dataset(interpolated)

    def calculate_daily_stats(self, ds):
        """Calculate daily statistics for each variable"""
        # Convert to daily data
        daily = ds.resample(time='D').agg({
            't2m': ['mean', 'min', 'max'],
            'd2m': ['mean', 'min', 'max'],
            'tp': 'sum',
            'sp': 'mean',
            'u10': 'mean',
            'v10': 'mean',
            'r2': 'mean',
            'tcc': 'mean'
        })
        
        # Flatten the multi-index
        daily = daily.to_dataframe().reset_index()
        
        # Rename columns
        daily.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                        for col in daily.columns]
        
        return daily

    def process_year(self, year):
        """Process ERA5 data for a specific year"""
        logger.info(f"Processing ERA5 data for {year}")
        
        # Get all files for the year
        era5_files = self.get_era5_files(year)
        if not era5_files:
            return None
            
        # Process each file
        all_data = []
        for nc_file in tqdm(era5_files, desc=f"Processing {year}"):
            try:
                # Read and process the data
                ds = xr.open_dataset(nc_file)
                
                # Interpolate to target grid
                ds_interp = self.interpolate_to_target_grid(ds)
                
                # Calculate daily statistics
                daily_stats = self.calculate_daily_stats(ds_interp)
                
                all_data.append(daily_stats)
                
                # Close the dataset
                ds.close()
                
            except Exception as e:
                logger.error(f"Error processing {nc_file}: {str(e)}")
                continue
        
        if all_data:
            # Combine all monthly data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Create output directory
            output_dir = Path('data/processed/era5')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            output_file = output_dir / f'era5_daily_{year}.csv'
            combined_data.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")
            
            return output_file
        return None

def main():
    # Input and output directories
    input_dir = Path('data/raw')
    output_dir = Path('process_data/era5')
    
    # Process all ERA5 files
    for file in sorted(input_dir.glob('era5_korea_*.nc')):
        try:
            process_era5(file, output_dir)
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main() 