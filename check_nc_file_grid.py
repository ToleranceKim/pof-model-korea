#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def analyze_grid_structure(file_path, output_dir=None):
    """
    NetCDF 파일의 격자 구조를 자세히 분석합니다.
    """
    print(f"Analyzing grid structure of: {file_path}")

    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return
    
    # 파일 헤더 확인 (바이너리 파일의 처음 몇 바이트를 읽어 형식 확인)
    with open(file_path, 'rb') as f:
        header = f.read(8)
        hex_header = ' '.join(f'{b:02X}' for b in header)
        print(f"File header (hex): {hex_header}")
        
        # NetCDF 파일의 매직 넘버 확인
        if header.startswith(b'CDF\x01') or header.startswith(b'CDF\x02'):
            print("Classic NetCDF format detected.")
        elif header.startswith(b'\x89HDF'):
            print("NetCDF 4 / HDF5 format detected.")
        elif header.startswith(b'GRIB'):
            print("GRIB format detected.")
        else:
            print("Unknown format detected. This might not be a NetCDF file.")

    # 1. 먼저 netCDF4 라이브러리 직접 사용
    try:
        import netCDF4 as nc
        print("\nTrying with netCDF4 library...")
        
        ds = nc.Dataset(file_path, 'r')
        print("Successfully opened with netCDF4!")
        
        # 기본 정보 출력
        print(f"Dimensions: {list(ds.dimensions.keys())}")
        print(f"Variables: {list(ds.variables.keys())}")
        
        # 위경도 변수 찾기
        lon_var_names = [var for var in ds.variables if 'lon' in var.lower()]
        lat_var_names = [var for var in ds.variables if 'lat' in var.lower()]
        
        if not lon_var_names or not lat_var_names:
            print("Could not identify longitude/latitude variables.")
            print(f"All variable names: {list(ds.variables.keys())}")
        else:
            print(f"Possible longitude variables: {lon_var_names}")
            print(f"Possible latitude variables: {lat_var_names}")
            
            # 첫 번째 변수 사용
            lon_var = lon_var_names[0]
            lat_var = lat_var_names[0]
            
            # 위경도 값 가져오기
            lon_values = ds.variables[lon_var][:]
            lat_values = ds.variables[lat_var][:]
            
            # 기본 분석 실행
            analyze_grid_coordinates(lon_values, lat_values, output_dir, os.path.basename(file_path))
            
        ds.close()
        print("Analysis with netCDF4 completed.")
        return
        
    except Exception as e:
        print(f"netCDF4 approach failed: {e}")
    
    # 2. xarray로 다양한 엔진 시도
    try:
        import xarray as xr
        print("\nTrying with xarray library using different engines...")
        
        engines = ['netcdf4', 'h5netcdf', 'scipy', 'cfgrib', 'zarr']
        
        for engine in engines:
            try:
                print(f"Trying engine: {engine}...")
                ds = xr.open_dataset(file_path, engine=engine)
                print(f"Successfully opened with xarray using {engine} engine!")
                
                # 기본 정보 출력
                print(f"Dimensions: {list(ds.dims.keys())}")
                print(f"Coordinates: {list(ds.coords.keys())}")
                print(f"Variables: {list(ds.data_vars.keys())}")
                
                # 위경도 좌표 확인
                lon_coords = [coord for coord in ds.coords if 'lon' in coord.lower()]
                lat_coords = [coord for coord in ds.coords if 'lat' in coord.lower()]
                
                if not lon_coords or not lat_coords:
                    print("Could not identify longitude/latitude coordinates.")
                    print(f"All coordinates: {list(ds.coords.keys())}")
                else:
                    print(f"Found longitude coordinates: {lon_coords}")
                    print(f"Found latitude coordinates: {lat_coords}")
                    
                    # 첫 번째 좌표 사용
                    lon_coord = lon_coords[0]
                    lat_coord = lat_coords[0]
                    
                    lon_values = ds[lon_coord].values
                    lat_values = ds[lat_coord].values
                    
                    # 기본 분석 실행
                    analyze_grid_coordinates(lon_values, lat_values, output_dir, os.path.basename(file_path))
                
                ds.close()
                print(f"Analysis with xarray ({engine}) completed.")
                return
                
            except Exception as e:
                print(f"Engine {engine} failed: {e}")
        
        print("All xarray engines failed.")
        
    except Exception as e:
        print(f"xarray approach failed: {e}")
    
    # 3. 라이브러리 추천
    print("\n===== Troubleshooting Recommendations =====")
    print("1. Install necessary packages:")
    print("   - For NetCDF: pip install netCDF4")
    print("   - For HDF5: pip install h5py h5netcdf")
    print("   - For GRIB: pip install cfgrib eccodes")
    print("2. Check if the file is actually a NetCDF file")
    print("3. Try using CDO or NCO command-line tools to inspect the file")
    
    print("\nFailed to analyze the file with all available methods.")

def analyze_grid_coordinates(lon_values, lat_values, output_dir=None, file_name="unknown"):
    """
    위경도 격자 좌표를 분석합니다.
    """
    try:
        print("\n===== Grid Coordinate Analysis =====")
        print(f"Longitude shape: {lon_values.shape}")
        print(f"Latitude shape: {lat_values.shape}")
        
        print(f"\nLongitude range: {np.min(lon_values):.4f}° ~ {np.max(lon_values):.4f}°")
        print(f"Latitude range: {np.min(lat_values):.4f}° ~ {np.max(lat_values):.4f}°")
        
        # 위경도 간격 계산
        print("\n===== Grid Resolution Analysis =====")
        
        # 경도 간격
        lon_sorted = np.sort(lon_values.flatten())
        lon_diffs = np.diff(lon_sorted)
        # 매우 작은 값 필터링 (부동소수점 오차)
        lon_diffs = lon_diffs[lon_diffs > 1e-10]
        lon_unique_diffs = np.unique(np.round(lon_diffs, 6))
        
        print("Longitude steps:")
        for step in lon_unique_diffs:
            count = np.sum(np.isclose(lon_diffs, step, atol=1e-6))
            print(f"  Step size {step:.6f}°: {count} occurrences ({count/len(lon_diffs)*100:.2f}%)")
        
        # 위도 간격
        lat_sorted = np.sort(lat_values.flatten())
        lat_diffs = np.diff(lat_sorted)
        # 매우 작은 값 필터링 (부동소수점 오차)
        lat_diffs = lat_diffs[lat_diffs > 1e-10]
        lat_unique_diffs = np.unique(np.round(lat_diffs, 6))
        
        print("\nLatitude steps:")
        for step in lat_unique_diffs:
            count = np.sum(np.isclose(lat_diffs, step, atol=1e-6))
            print(f"  Step size {step:.6f}°: {count} occurrences ({count/len(lat_diffs)*100:.2f}%)")
        
        # 전체 격자 크기 계산
        print(f"\nTotal grid cells: {len(lon_values.flatten()) * len(lat_values.flatten())}")
        print(f"Unique longitude values: {len(np.unique(lon_values))}")
        print(f"Unique latitude values: {len(np.unique(lat_values))}")
        
        # 위경도 값 분포 시각화
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(16, 12))
            
            # 1. 경도 간격 히스토그램
            plt.subplot(2, 2, 1)
            plt.hist(lon_diffs, bins=50)
            plt.title('Longitude Step Sizes')
            plt.xlabel('Step Size (degrees)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # 2. 위도 간격 히스토그램
            plt.subplot(2, 2, 2)
            plt.hist(lat_diffs, bins=50)
            plt.title('Latitude Step Sizes')
            plt.xlabel('Step Size (degrees)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # 3. 위경도 격자점 시각화
            plt.subplot(2, 2, 3)
            
            # 데이터 차원에 따라 처리
            if len(lon_values.shape) > 1 and len(lat_values.shape) > 1:
                # 2D 배열인 경우 (격자 형태)
                plt.scatter(lon_values.flatten(), lat_values.flatten(), s=1, alpha=0.5)
            else:
                # 1D 배열인 경우 (메쉬그리드 필요)
                if len(lon_values.shape) == 1 and len(lat_values.shape) == 1:
                    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
                    plt.scatter(lon_grid.flatten(), lat_grid.flatten(), s=1, alpha=0.5)
                else:
                    plt.scatter(lon_values, lat_values, s=1, alpha=0.5)
                
            plt.title('All Grid Points')
            plt.xlabel('Longitude (°E)')
            plt.ylabel('Latitude (°N)')
            plt.grid(True, alpha=0.3)
            
            # 4. 격자 구조 확대 시각화 (중앙 부분)
            plt.subplot(2, 2, 4)
            
            # 확대할 영역 선택
            lon_min, lon_max = np.min(lon_values), np.max(lon_values)
            lat_min, lat_max = np.min(lat_values), np.max(lat_values)
            
            # 중앙 20% 정도 선택
            lon_center = (lon_min + lon_max) / 2
            lat_center = (lat_min + lat_max) / 2
            lon_range = (lon_max - lon_min) * 0.2
            lat_range = (lat_max - lat_min) * 0.2
            
            # 확대 영역 경계
            zoom_lon_min = lon_center - lon_range/2
            zoom_lon_max = lon_center + lon_range/2
            zoom_lat_min = lat_center - lat_range/2
            zoom_lat_max = lat_center + lat_range/2
            
            # 데이터 차원에 따라 처리
            if len(lon_values.shape) > 1 and len(lat_values.shape) > 1:
                # 2D 배열인 경우
                mask = ((lon_values >= zoom_lon_min) & (lon_values <= zoom_lon_max) &
                        (lat_values >= zoom_lat_min) & (lat_values <= zoom_lat_max))
                plt.scatter(lon_values[mask], lat_values[mask], s=10)
            else:
                # 1D 배열인 경우
                if len(lon_values.shape) == 1 and len(lat_values.shape) == 1:
                    lon_zoom = lon_values[(lon_values >= zoom_lon_min) & (lon_values <= zoom_lon_max)]
                    lat_zoom = lat_values[(lat_values >= zoom_lat_min) & (lat_values <= zoom_lat_max)]
                    
                    if len(lon_zoom) > 0 and len(lat_zoom) > 0:
                        zoom_lon_grid, zoom_lat_grid = np.meshgrid(lon_zoom, lat_zoom)
                        plt.scatter(zoom_lon_grid, zoom_lat_grid, s=10)
            
            plt.title('Zoomed Grid Structure (Center Region)')
            plt.xlabel('Longitude (°E)')
            plt.ylabel('Latitude (°N)')
            plt.xlim(zoom_lon_min, zoom_lon_max)
            plt.ylim(zoom_lat_min, zoom_lat_max)
            plt.grid(True, alpha=0.3)
            
            # 저장
            file_name = file_name.replace('.nc', '')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{file_name}_grid_analysis.png'), dpi=300)
            print(f"\nVisualization saved to: {os.path.join(output_dir, f'{file_name}_grid_analysis.png')}")
        
        # 격자 문제 진단
        print("\n===== Grid Structure Diagnosis =====")
        
        # 일정한 간격인지 확인
        if len(lon_unique_diffs) == 1 and len(lat_unique_diffs) == 1:
            print("✓ The grid has uniform spacing.")
            print(f"  Longitude step: {lon_unique_diffs[0]:.6f}°")
            print(f"  Latitude step: {lat_unique_diffs[0]:.6f}°")
        else:
            print("✗ The grid does NOT have uniform spacing!")
            
            if len(lon_unique_diffs) > 1:
                print(f"  Longitude has {len(lon_unique_diffs)} different step sizes.")
                most_common_idx = np.argmax([np.sum(np.isclose(lon_diffs, step, atol=1e-6)) for step in lon_unique_diffs])
                print(f"  Most common longitude step: {lon_unique_diffs[most_common_idx]:.6f}°")
            
            if len(lat_unique_diffs) > 1:
                print(f"  Latitude has {len(lat_unique_diffs)} different step sizes.")
                most_common_idx = np.argmax([np.sum(np.isclose(lat_diffs, step, atol=1e-6)) for step in lat_unique_diffs])
                print(f"  Most common latitude step: {lat_unique_diffs[most_common_idx]:.6f}°")
        
        # 0.1도 간격과의 비교
        expected_step = 0.1
        if (len(lon_unique_diffs) > 0 and len(lat_unique_diffs) > 0 and
            np.isclose(lon_unique_diffs[0], expected_step, atol=1e-4) and 
            np.isclose(lat_unique_diffs[0], expected_step, atol=1e-4)):
            print(f"✓ The grid spacing matches the expected {expected_step}° resolution.")
        else:
            print(f"✗ The grid spacing does NOT match the expected {expected_step}° resolution!")
        
        # 격자 전처리 시 주의사항
        print("\n===== Recommendations =====")
        if len(lon_unique_diffs) > 1 or len(lat_unique_diffs) > 1:
            print("1. Be careful when processing this grid - it has non-uniform spacing.")
            print("2. Consider using actual coordinate values rather than assuming uniform spacing.")
            print("3. When calculating grid_id, use the exact longitude/latitude values rather than index-based calculations.")
        else:
            print("1. This grid has uniform spacing and can be processed with standard methods.")
            if (len(lon_unique_diffs) > 0 and len(lat_unique_diffs) > 0 and
                (not np.isclose(lon_unique_diffs[0], expected_step, atol=1e-4) or 
                 not np.isclose(lat_unique_diffs[0], expected_step, atol=1e-4))):
                print(f"2. However, note that the grid spacing is not exactly {expected_step}°.")
                print("3. Adjust your grid_id calculation to match the actual spacing.")
        
        # 위경도 값 CSV로 저장 (나중에 참조용)
        if output_dir:
            lon_df = pd.DataFrame({'longitude': lon_values.flatten()})
            lat_df = pd.DataFrame({'latitude': lat_values.flatten()})
            lon_df.to_csv(os.path.join(output_dir, f'{file_name}_longitudes.csv'), index=False)
            lat_df.to_csv(os.path.join(output_dir, f'{file_name}_latitudes.csv'), index=False)
            print(f"Coordinate values saved to CSV files in {output_dir}")
    
    except Exception as e:
        print(f"ERROR in analyzing coordinates: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze the grid structure of a NetCDF file')
    parser.add_argument('file_path', type=str, help='Path to the NetCDF file to analyze')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs/grid_analysis',
                        help='Directory to save visualization outputs (default: outputs/grid_analysis)')
    
    args = parser.parse_args()
    
    analyze_grid_structure(args.file_path, args.output_dir) 