#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import logging
import gc
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_interpolated_file(input_file, output_dir):
    """
    보간된 ERA5 파일을 일별 통계로 처리합니다.
    
    Args:
        input_file: 입력 NetCDF 파일 경로 (0.1도 보간된 파일)
        output_dir: 출력 디렉토리
    
    Returns:
        처리된 파일 경로 또는 None (처리 실패 시)
    """
    logger.info(f"Processing interpolated file: {input_file}")
    
    # 파일명에서 연월 추출
    file_name = os.path.basename(input_file)
    year_month = file_name.split('_')[2].split('_')[0]  # era5_korea_YYYYMM_0.1deg.nc
    
    # 출력 파일 경로 설정
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_base = output_dir / f"era5_daily_{year_month}"
    output_parquet = f"{output_base}.parquet"
    output_csv = f"{output_base}.csv"
    
    # 이미 처리된 파일이면 건너뜀
    if os.path.exists(output_parquet) and os.path.exists(output_csv):
        logger.info(f"File already processed: {output_parquet} and {output_csv}")
        return output_parquet
    
    try:
        # NetCDF 파일 열기
        ds = xr.open_dataset(input_file)
        logger.info(f"Dataset opened. Variables: {list(ds.data_vars)}")
        
        # 변수 매핑
        var_mapping = {
            't2m': 't2m',   # 2m 기온
            'd2m': 'td2m',  # 2m 이슬점 온도
            'u10': '10u',   # 10m U 바람
            'v10': '10v',   # 10m V 바람
            'tp': 'tp',     # 강수량
            'sp': 'sp',     # 표면 기압
            'r2': 'r2',     # 상대 습도
            'tcc': 'tcc'    # 전운량
        }
        
        # 시간 차원 확인
        time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        
        # 시간 값 가져오기
        if time_dim in ds.coords:
            times = pd.to_datetime(ds[time_dim].values)
            logger.info(f"Time dimension: {time_dim}, first few times: {times[:5]}")
        else:
            logger.error(f"Time dimension {time_dim} not found in coordinates")
            return None
        
        # 날짜별로 그룹화
        day_groups = {}
        for i, t in enumerate(times):
            day = pd.Timestamp(t).date()
            if day not in day_groups:
                day_groups[day] = []
            day_groups[day].append(i)
        
        logger.info(f"Found {len(day_groups)} unique days")
        
        # 각 변수별 데이터프레임 생성 및 병합
        all_results = []
        
        for var_name, target_name in var_mapping.items():
            if var_name in ds:
                logger.info(f"Processing variable: {var_name} -> {target_name}")
                
                # 변수 데이터 가져오기
                var_data = ds[var_name].values
                
                # 날짜별 데이터 생성
                daily_results = []
                
                # 위도, 경도 값 가져오기
                lats = ds.latitude.values
                lons = ds.longitude.values
                
                for day, indices in tqdm(day_groups.items(), desc=f"Processing {var_name}"):
                    # 각 격자점마다 일별 평균/합계 계산
                    for lat_idx, lat in enumerate(lats):
                        for lon_idx, lon in enumerate(lons):
                            # 해당 격자점의 하루동안의 값들
                            grid_vals = [var_data[idx, lat_idx, lon_idx] for idx in indices]
                            
                            # 결측값이 아닌 값만 필터링
                            valid_vals = [v for v in grid_vals if not np.isnan(v)]
                            
                            if valid_vals:
                                # 일별 집계
                                if var_name in ['t2m', 'd2m', 'u10', 'v10', 'sp', 'r2', 'tcc']:  # 평균 계산
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
                    logger.info(f"Created dataframe for {var_name} with shape: {var_df.shape}")
                    
                    # 필요한 컬럼만 선택
                    var_df = var_df[['acq_date', 'grid_id', 'latitude', 'longitude', target_name]]
                    all_results.append(var_df)
                else:
                    logger.warning(f"No daily results for {var_name}")
                
                # 메모리 정리
                del var_data
                del daily_results
                gc.collect()
        
        # 모든 변수 병합
        if not all_results:
            logger.error("No variables were successfully processed")
            return None
        
        # 첫 번째 데이터프레임을 기준으로 병합
        logger.info("Merging dataframes...")
        result_df = all_results[0]
        for i, df in enumerate(all_results[1:], 1):
            var_name = list(var_mapping.values())[i]
            logger.info(f"Merging {var_name} dataframe, before merge: {result_df.shape}")
            result_df = pd.merge(result_df, df, on=['acq_date', 'grid_id', 'latitude', 'longitude'], how='outer')
            logger.info(f"After merge with {var_name}: {result_df.shape}")
        
        # 결측값 확인
        null_counts = result_df.isnull().sum()
        logger.info(f"Null value counts in final dataframe:\n{null_counts}")
        
        # U10, V10 풍속 성분을 결합하여 10m 풍속 크기(wind10m) 계산
        if '10u' in result_df.columns and '10v' in result_df.columns:
            logger.info("Calculating wind speed from U and V components...")
            result_df['wind10m'] = np.sqrt(result_df['10u']**2 + result_df['10v']**2)
            logger.info(f"wind10m range: {result_df['wind10m'].min():.2f} ~ {result_df['wind10m'].max():.2f} m/s")
        
        # Parquet 형식으로 저장
        logger.info(f"Saving to {output_parquet}")
        result_df.to_parquet(output_parquet, index=False)
        
        # CSV 형식으로도 저장
        logger.info(f"Saving to {output_csv}")
        result_df.to_csv(output_csv, index=False)
        
        logger.info(f"Successfully saved files with shape {result_df.shape}")
        
        return output_parquet
    
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # 메모리 정리
        if 'ds' in locals() and ds is not None:
            try:
                ds.close()
            except:
                pass
            del ds
        gc.collect()

def main():
    # 입력 및 출력 디렉토리
    input_dir = Path('processed_data/era5')
    output_dir = Path('processed_data/era5_daily')
    
    # 모든 보간된 ERA5 파일 처리
    for file in sorted(input_dir.glob('era5_korea_*_0.1deg.nc')):
        try:
            process_interpolated_file(file, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {file}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 