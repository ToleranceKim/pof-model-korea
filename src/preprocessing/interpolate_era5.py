#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import zipfile
import tempfile
import xarray as xr
import numpy as np
from pathlib import Path
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def interpolate_era5(zip_file_path, output_dir):
    """
    ZIP 파일에서 추출한 ERA5 데이터를 0.1° 그리드로 선형 보간합니다.
    """
    # 파일명에서 연월 추출
    file_name = os.path.basename(zip_file_path)
    year_month = file_name.split('_')[-1].split('.')[0]
    
    # 출력 파일 경로 설정
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"era5_korea_{year_month}_0.1deg.nc"
    
    # 이미 처리된 파일이면 건너뜀
    if output_file.exists():
        logger.info(f"File already processed: {output_file}")
        return output_file
    
    logger.info(f"Processing: {zip_file_path}")
    
    try:
        # 임시 폴더 생성
        temp_dir = tempfile.mkdtemp()
        
        try:
            # ZIP 파일 압축 해제
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 추출된 파일 목록 확인
            extracted_files = os.listdir(temp_dir)
            logger.info(f"Extracted files: {extracted_files}")
            
            # NetCDF 파일 찾기
            nc_files = [f for f in extracted_files if f.endswith('.nc')]
            if not nc_files:
                logger.error(f"No NetCDF files found in ZIP: {zip_file_path}")
                return None
            
            # 모든 변수를 담을 딕셔너리 초기화
            all_variables = {}
            
            # 각 NetCDF 파일 처리
            for nc_file in nc_files:
                nc_path = os.path.join(temp_dir, nc_file)
                logger.info(f"Processing NetCDF file: {nc_path}")
                
                # NetCDF 파일 열기
                ds = xr.open_dataset(nc_path)
                logger.info(f"Variables in {nc_file}: {list(ds.data_vars)}")
                
                # 원본 격자 정보 확인
                lat_vals = ds.latitude.values
                lon_vals = ds.longitude.values
                logger.info(f"Original grid: {len(lat_vals)}x{len(lon_vals)} points")
                
                # 0.1° 격자 생성
                target_lat = np.arange(33, 39, 0.1)
                target_lon = np.arange(124, 132, 0.1)
                
                # xarray 데이터배열로 변환
                target_lat = xr.DataArray(target_lat, dims=['latitude'])
                target_lon = xr.DataArray(target_lon, dims=['longitude'])
                
                # 각 변수별로 선형 보간 적용
                for var_name in ds.data_vars:
                    logger.info(f"Interpolating {var_name}...")
                    interpolated_var = ds[var_name].interp(
                        latitude=target_lat,
                        longitude=target_lon,
                        method='linear'
                    )
                    all_variables[var_name] = interpolated_var
                
                # 데이터셋 닫기
                ds.close()
            
            # 모든 변수를 하나의 데이터셋으로 결합
            if all_variables:
                combined_ds = xr.Dataset(all_variables)
                logger.info(f"Combined dataset with variables: {list(combined_ds.data_vars)}")
                
                # 결과 저장
                logger.info(f"Saving interpolated data to {output_file}")
                combined_ds.to_netcdf(output_file)
                logger.info(f"Successfully saved {output_file}")
                
                return output_file
            else:
                logger.error("No variables were interpolated")
                return None
            
        finally:
            # 임시 디렉토리 정리
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory: {e}")
    
    except Exception as e:
        logger.error(f"Error processing {zip_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 입력 및 출력 디렉토리
    input_dir = Path('data/raw')
    output_dir = Path('processed_data/era5')
    
    # 모든 ERA5 파일 처리
    for file in sorted(input_dir.glob('era5_korea_*.nc')):
        try:
            interpolate_era5(file, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {file}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 