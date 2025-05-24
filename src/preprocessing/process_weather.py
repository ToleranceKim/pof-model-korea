#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import zipfile
import pandas as pd
import numpy as np
import xarray as xr
import argparse
from tqdm import tqdm
import gc
import time

def process_netcdf_from_zip(zip_file_path, output_dir):
    """
    ZIP 파일에서 NetCDF를 추출하고 처리합니다.
    
    Args:
        zip_file_path: ZIP 파일 경로
        output_dir: 출력 디렉토리
        
    Returns:
        처리된 Parquet 파일 경로 또는 None (처리 실패 시)
    """
    # 파일명에서 연월 추출
    file_name = os.path.basename(zip_file_path)
    year_month = file_name.split('_')[-1].split('.')[0]
    output_base = os.path.join(output_dir, f"era5_korea_{year_month}")
    output_parquet = f"{output_base}.parquet"
    output_csv = f"{output_base}.csv"
    
    # 이미 처리된 파일이면 건너뜀
    if os.path.exists(output_parquet) and os.path.exists(output_csv):
        print(f"File already processed: {output_parquet} and {output_csv}")
        return output_parquet
    
    print(f"Processing: {zip_file_path}")
    
    try:
        # 임시 폴더 생성 (직접 관리)
        import tempfile
        temp_dir = tempfile.mkdtemp()
        nc_path = None
        ds = None
        
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
            
            # NetCDF 파일을 pandas DataFrame으로 직접 변환
            ds = xr.open_dataset(nc_path)
            print(f"Dataset opened. Dimensions: {list(ds.dims)}")
            print(f"Variables: {list(ds.variables)}")
            
            # 필요한 변수 추출
            var_mapping = {
                't2m': 't2m',   # 2m 기온
                'd2m': 'td2m',  # 2m 이슬점 온도
                'u10': '10u',   # 10m U 바람
                'v10': '10v',   # 10m V 바람
                'tp': 'tp'      # 강수량
            }
            
            # 데이터 추출 및 처리
            # 시간 차원 확인
            time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
            
            # 시간 값 가져오기
            if time_dim in ds.coords:
                times = pd.to_datetime(ds[time_dim].values)
                print(f"Time dimension: {time_dim}, values type: {type(ds[time_dim].values)}")
                print(f"First few times: {times[:5]}")
            else:
                print(f"Time dimension {time_dim} not found in coordinates")
                return None
            
            # 위도, 경도 값 가져오기
            lats = ds.latitude.values
            lons = ds.longitude.values
            
            # 먼저 기본 데이터 프레임 생성 - 모든 시간과 위도/경도 조합
            time_vals = pd.to_datetime(ds[time_dim].values)
            lat_vals = ds.latitude.values
            lon_vals = ds.longitude.values
            
            print(f"Creating base dataframe with times: {len(time_vals)}, lats: {len(lat_vals)}, lons: {len(lon_vals)}")
            
            # 각 변수별 데이터프레임 생성 및 병합
            all_results = []
            
            for var_name, target_name in var_mapping.items():
                if var_name in ds:
                    print(f"\nProcessing variable: {var_name} -> {target_name}")
                    
                    # 변수 데이터 가져오기 (numpy 배열로)
                    var_data = ds[var_name].values
                    print(f"Variable shape: {var_data.shape}")
                    
                    # 날짜별로 처리
                    day_groups = {}
                    for i, t in enumerate(time_vals):
                        day = pd.Timestamp(t).date()
                        if day not in day_groups:
                            day_groups[day] = []
                        day_groups[day].append(i)
                    
                    print(f"Found {len(day_groups)} unique days")
                    
                    # 날짜별 데이터 생성
                    daily_results = []
                    
                    for day, indices in day_groups.items():
                        print(f"Processing day: {day}, with {len(indices)} time points")
                        
                        # 각 격자점마다 일별 평균/합계 계산
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
                                    
                                    # grid_id 계산
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
                    var_df = pd.DataFrame(daily_results)
                    print(f"Created dataframe for {var_name} with shape: {var_df.shape}")
                    
                    if not var_df.empty:
                        # 필요한 컬럼만 선택
                        var_df = var_df[['acq_date', 'grid_id', target_name]]
                        all_results.append(var_df)
                    else:
                        print(f"Warning: Empty dataframe for {var_name}")
                    
                    # 메모리 정리
                    del var_data
                    del daily_results
                    gc.collect()
            
            # 모든 변수 병합
            if not all_results:
                print("No variables were successfully processed")
                return None
            
            # 각 변수 데이터프레임의 요약 출력
            for i, df in enumerate(all_results):
                var_name = list(var_mapping.values())[i]
                print(f"Dataframe {i} ({var_name}): shape={df.shape}, null values={df.isnull().sum().sum()}")
                print(f"Sample data:\n{df.head()}")
            
            # 첫 번째 데이터프레임을 기준으로 병합
            print("\nMerging dataframes...")
            result_df = all_results[0]
            for i, df in enumerate(all_results[1:], 1):
                var_name = list(var_mapping.values())[i]
                print(f"Merging {var_name} dataframe, before merge: {result_df.shape}")
                result_df = pd.merge(result_df, df, on=['acq_date', 'grid_id'], how='outer')
                print(f"After merge with {var_name}: {result_df.shape}")
            
            # 결측값 확인
            null_counts = result_df.isnull().sum()
            print(f"\nNull value counts in final dataframe:\n{null_counts}")
            
            # U10, V10 풍속 성분을 결합하여 10m 풍속 크기(wind10m) 계산
            if '10u' in result_df.columns and '10v' in result_df.columns:
                print("10u, 10v 풍속 성분을 결합하여 10m 풍속 크기(wind10m) 계산")
                result_df['wind10m'] = np.sqrt(result_df['10u']**2 + result_df['10v']**2)
                print(f"wind10m 변수 추가 완료: 범위 {result_df['wind10m'].min():.2f} ~ {result_df['wind10m'].max():.2f} m/s")
            
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
            if 'ds' in locals() and ds is not None:
                try:
                    ds.close()
                except:
                    pass
                del ds
            
            # 지연 시간 추가
            time.sleep(1)
            
            # 임시 디렉토리 정리 시도
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory: {e}")
            
            # 메모리 정리
            gc.collect()
    
    except Exception as e:
        print(f"Error processing file {zip_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='단순화된 날씨 데이터 전처리 파이프라인')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='기상 데이터 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default='../processed_data',
                        help='처리된 데이터 저장 디렉토리 경로')
    parser.add_argument('--target_path', type=str, default=None,
                        help='타겟 데이터 CSV 파일 경로 (사용하지 않음)')
    parser.add_argument('--final_output', type=str, default=None,
                        help='최종 병합된 데이터 저장 경로 (사용하지 않음)')
    parser.add_argument('--test_mode', action='store_true',
                        help='테스트 모드 (일부 파일만 처리)')
    parser.add_argument('--max_files', type=int, default=2,
                        help='테스트 모드에서 처리할 최대 파일 수')
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 기상 데이터 파일 목록 가져오기
    weather_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) 
                     if f.startswith('era5_korea_') and f.endswith('.nc')]
    weather_files.sort()
    
    # 테스트 모드인 경우 일부 파일만 처리
    if args.test_mode:
        print(f"Running in test mode with max {args.max_files} files")
        weather_files = weather_files[:args.max_files]
    
    print(f"Found {len(weather_files)} weather files")
    
    # 각 파일 처리
    processed_files = []
    for file_path in tqdm(weather_files):
        processed_file = process_netcdf_from_zip(file_path, args.output_dir)
        if processed_file:
            processed_files.append(processed_file)
        
        # 각 파일 처리 후 메모리 정리
        gc.collect()
    
    # 병합 과정 제거 - 개별 파일만 처리
    print(f"Processed {len(processed_files)} files. All files saved to {args.output_dir}")
    print("Done!")

if __name__ == '__main__':
    main() 