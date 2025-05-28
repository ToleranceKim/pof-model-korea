#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

def check_dataset(file_path, name, max_rows=5):
    """데이터셋의 격자 시스템을 확인합니다."""
    print(f"\n==== {name} 데이터셋 확인 ====")
    print(f"파일: {file_path}")
    
    try:
        # 파일 확장자 확인
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print(f"지원되지 않는 파일 형식: {file_path}")
            return
        
        # 열 확인
        print(f"행 수: {len(df)}")
        print(f"열: {df.columns.tolist()}")
        
        # 격자 시스템 확인
        grid_found = False
        
        # grid_id 확인
        if 'grid_id' in df.columns:
            grid_found = True
            print("\n* grid_id 확인")
            print(f"  - 고유한 grid_id 수: {df['grid_id'].nunique()}")
            print(f"  - grid_id 범위: {df['grid_id'].min()} ~ {df['grid_id'].max()}")
            
            # grid_id 샘플
            print(f"  - grid_id 샘플: {df['grid_id'].head(max_rows).tolist()}")
            
            # grid_id 계산 공식과 일치하는지 확인 (0.1° 기준)
            if 'latitude' in df.columns and 'longitude' in df.columns:
                print("\n* grid_id 계산 확인 (0.1° 격자 기준)")
                sample = df.head(1)
                lat = sample['latitude'].iloc[0]
                lon = sample['longitude'].iloc[0]
                grid_id = sample['grid_id'].iloc[0]
                
                # 0.1° 기준 grid_id 계산
                lat_bin = int(np.floor(lat / 0.1))
                lon_bin = int(np.floor(lon / 0.1))
                calc_grid_id = (lat_bin + 900) * 3600 + (lon_bin + 1800)
                
                print(f"  - 샘플 좌표: ({lat}, {lon})")
                print(f"  - 저장된 grid_id: {grid_id}")
                print(f"  - 계산된 grid_id: {calc_grid_id}")
                print(f"  - 일치 여부: {grid_id == calc_grid_id}")
                
                # 0.25° 기준 grid_id 계산
                lat_bin_025 = int(np.floor(lat / 0.25))
                lon_bin_025 = int(np.floor(lon / 0.25))
                calc_grid_id_025 = (lat_bin_025 + 360) * 1440 + (lon_bin_025 + 720)
                print(f"  - 0.25° 기준 계산된 grid_id: {calc_grid_id_025}")
        
        # 위도/경도 확인
        if 'latitude' in df.columns and 'longitude' in df.columns:
            grid_found = True
            print("\n* 위도/경도 확인")
            print(f"  - 위도 범위: {df['latitude'].min():.4f}° ~ {df['latitude'].max():.4f}°")
            print(f"  - 경도 범위: {df['longitude'].min():.4f}° ~ {df['longitude'].max():.4f}°")
            
            # 격자 간격 추정
            try:
                lat_values = sorted(df['latitude'].unique())
                lon_values = sorted(df['longitude'].unique())
                
                if len(lat_values) > 1 and len(lon_values) > 1:
                    lat_diffs = [lat_values[i+1] - lat_values[i] for i in range(min(10, len(lat_values)-1))]
                    lon_diffs = [lon_values[i+1] - lon_values[i] for i in range(min(10, len(lon_values)-1))]
                    
                    avg_lat_diff = sum(lat_diffs) / len(lat_diffs)
                    avg_lon_diff = sum(lon_diffs) / len(lon_diffs)
                    
                    print(f"  - 추정 위도 간격: {avg_lat_diff:.4f}°")
                    print(f"  - 추정 경도 간격: {avg_lon_diff:.4f}°")
                    print(f"  - 0.1° 격자 여부: {0.09 < avg_lat_diff < 0.11 and 0.09 < avg_lon_diff < 0.11}")
                    print(f"  - 0.25° 격자 여부: {0.24 < avg_lat_diff < 0.26 and 0.24 < avg_lon_diff < 0.26}")
            except Exception as e:
                print(f"  - 격자 간격 추정 오류: {str(e)}")
        
        if not grid_found:
            print("\n* 격자 시스템을 찾을 수 없음")
            print("  - 열 확인: ", df.columns.tolist())
        
        # 데이터 샘플 출력
        print("\n* 데이터 샘플")
        print(df.head(max_rows))
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

def main():
    """여러 데이터셋의 격자 시스템을 확인합니다."""
    # 확인할 데이터셋 목록
    datasets = [
        ('data/FUEL_combined/FUEL_combined_2011_2021.parquet', 'FUEL'),
        ('data/DFMC_combined/DFMC_combined_2011_2021.parquet', 'DFMC'),
        ('data/population_density/combined_population_density_2000_2020.parquet', 'Population Density'),
        ('data/lai_low_high/lai_low_high_monthly.parquet', 'LAI'),
        ('data/lightning/lightning_KOR.parquet', 'Lightning'),
        ('processed_data/era5_korea_202001.parquet', 'ERA5')
    ]
    
    for file_path, name in datasets:
        if os.path.exists(file_path):
            check_dataset(file_path, name)
        else:
            print(f"\n==== {name} 데이터셋 확인 ====")
            print(f"파일을 찾을 수 없음: {file_path}")

if __name__ == "__main__":
    main() 