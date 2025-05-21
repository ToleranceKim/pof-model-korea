#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

def grib_to_csv(grib_file, output_file=None, include_coords=True):
    """
    GRIB2 파일을 CSV로 변환합니다.
    
    Parameters:
    -----------
    grib_file : str
        입력 GRIB2 파일 경로
    output_file : str
        출력 CSV 파일 경로 (None인 경우 .grib2 확장자를 .csv로 변경)
    include_coords : bool
        위도/경도 좌표를 포함할지 여부
        
    Returns:
    --------
    str : 생성된 CSV 파일 경로 또는 None (변환 실패 시)
    """
    try:
        print(f"GRIB2 파일 '{grib_file}' 처리 중...")
        
        # 출력 파일 경로가 지정되지 않은 경우 입력 파일과 동일한 위치에 .csv 확장자로 저장
        if output_file is None:
            output_file = os.path.splitext(grib_file)[0] + '.csv'
            
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # GRIB2 파일 로드
        try:
            import cfgrib
            ds = xr.open_dataset(grib_file, engine='cfgrib')
            print(f"데이터셋 로드 성공: {list(ds.data_vars)}")
        except Exception as e:
            print(f"GRIB2 파일 '{grib_file}' 로드 중 오류 발생: {e}")
            print("cfgrib 패키지가 설치되어 있는지 확인하세요. (pip install cfgrib)")
            return None
        
        # 데이터셋 정보 출력
        print(f"데이터셋 차원: {list(ds.dims)}")
        print(f"데이터셋 좌표: {list(ds.coords)}")
        
        # 변수명 확인 (GRIB2 파일에는 보통 하나의 변수만 있음)
        var_names = list(ds.data_vars)
        if not var_names:
            print("오류: GRIB2 파일에 변수가 없습니다.")
            return None
            
        var_name = var_names[0]
        print(f"처리할 변수: {var_name}")
        
        # 데이터를 pandas DataFrame으로 변환
        # 먼저 DataArray를 2D로 변환
        da = ds[var_name]
        
        # 날짜 및 시간 정보 추출
        if 'valid_time' in ds.coords:
            time_info = pd.to_datetime(ds.valid_time.values)
        elif 'time' in ds.coords:
            time_info = pd.to_datetime(ds.time.values)
        else:
            time_info = None
            
        print(f"예보 시간: {time_info}")
        
        # 데이터프레임으로 변환
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            # 격자 형태의 위도/경도 데이터
            lats = ds.latitude.values
            lons = ds.longitude.values
            
            # 데이터 배열의 shape 확인
            print(f"데이터 배열 shape: {da.shape}")
            
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
                            
            elif len(da.shape) == 3 and time_info is not None:  # [time, lat, lon]
                data_values = da.values
                
                for t, time_val in enumerate(time_info):
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
                print(f"지원하지 않는 데이터 형식: {da.shape}")
                return None
                
            # DataFrame 생성
            df = pd.DataFrame(rows)
            
        else:
            # 다른 형태의 데이터
            print("비격자 형태의 데이터는 아직 지원하지 않습니다.")
            return None
            
        # 기본 정보 추가
        df['source'] = 'ecmwf-opendata'
        
        # CSV로 저장
        print(f"CSV 파일 '{output_file}'로 저장 중...")
        df.to_csv(output_file, index=False)
        print(f"변환 완료: {output_file} (행: {len(df)}, 열: {len(df.columns)})")
        
        return output_file
        
    except Exception as e:
        import traceback
        print(f"변환 중 오류 발생: {e}")
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='GRIB2 파일을 CSV로 변환')
    
    parser.add_argument('--input', type=str, required=True,
                        help='입력 GRIB2 파일 경로')
    parser.add_argument('--output', type=str, default=None,
                        help='출력 CSV 파일 경로 (지정하지 않으면 입력 파일과 같은 위치에 .csv로 저장)')
    parser.add_argument('--no-coords', action='store_false', dest='include_coords',
                        help='위도/경도 좌표를 포함하지 않음')
    
    args = parser.parse_args()
    
    # 입력 파일이 존재하는지 확인
    if not os.path.exists(args.input):
        print(f"오류: 입력 파일 '{args.input}'이 존재하지 않습니다.")
        return 1
        
    # GRIB2 파일을 CSV로 변환
    output_file = grib_to_csv(args.input, args.output, args.include_coords)
    
    if output_file:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 