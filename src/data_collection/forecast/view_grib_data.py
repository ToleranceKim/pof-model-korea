#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

def view_grib_file(grib_file, variable=None, save_path=None, show=True):
    """
    GRIB2 파일을 읽고 지도에 시각화합니다.
    
    Parameters:
    -----------
    grib_file : str
        GRIB2 파일 경로
    variable : str, optional
        표시할 변수 이름 (지정하지 않으면 첫 번째 변수 사용)
    save_path : str, optional
        이미지 저장 경로 (지정하지 않으면 저장하지 않음)
    show : bool, optional
        화면에 표시 여부 (기본값: True)
    """
    try:
        # cfgrib 모듈 임포트 (설치되어 있지 않은 경우 설치 안내)
        try:
            import xarray as xr
            import cfgrib
        except ImportError:
            print("필요한 패키지가 설치되어 있지 않습니다.")
            print("다음 명령어로 패키지를 설치하세요:")
            print("pip install xarray cfgrib matplotlib cartopy")
            return
        
        # GRIB 파일 로드
        print(f"GRIB 파일 로드 중: {grib_file}")
        ds = xr.open_dataset(grib_file, engine='cfgrib')
        
        # 데이터셋 정보 출력
        print("\n--- 데이터셋 정보 ---")
        print(f"변수: {list(ds.data_vars)}")
        print(f"좌표: {list(ds.coords)}")
        print(f"시간: {ds.time.values}")
        print(f"경도 범위: {ds.longitude.min().values:.2f} ~ {ds.longitude.max().values:.2f}")
        print(f"위도 범위: {ds.latitude.min().values:.2f} ~ {ds.latitude.max().values:.2f}")

        # 표시할 변수 결정
        if variable is None:
            variable = list(ds.data_vars)[0]
            print(f"\n첫 번째 변수 '{variable}'를 사용합니다.")
        elif variable not in ds.data_vars:
            print(f"\n지정한 변수 '{variable}'가 데이터셋에 없습니다.")
            variable = list(ds.data_vars)[0]
            print(f"첫 번째 변수 '{variable}'를 대신 사용합니다.")
        else:
            print(f"\n변수 '{variable}'를 사용합니다.")
        
        # 변수 통계 출력
        var_data = ds[variable]
        print(f"\n--- 변수 '{variable}' 통계 ---")
        print(f"형태: {var_data.shape}")
        print(f"단위: {var_data.units}")
        print(f"최소값: {var_data.min().values}")
        print(f"최대값: {var_data.max().values}")
        print(f"평균값: {var_data.mean().values}")
        
        # 데이터 시각화
        print("\n데이터 시각화 중...")
        
        # 지도 투영법 설정
        proj = ccrs.PlateCarree()
        
        # 한국 근처 지역 좌표 설정 (경도: 124-132, 위도: 33-38)
        extent = [124, 132, 33, 38]
        
        # 그림 설정
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj})
        
        # 국경선, 해안선 추가
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        
        # 좌표 그리드 추가
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        
        # 지도 범위 설정
        ax.set_extent(extent, crs=proj)
        
        # 데이터 시각화
        im = var_data.plot(ax=ax, transform=proj, cmap='viridis', add_colorbar=True)
        
        # 제목 설정
        variable_info = f"{variable}"
        if hasattr(var_data, 'units'):
            variable_info += f" [{var_data.units}]"
        
        time_info = f"시간: {ds.time.dt.strftime('%Y-%m-%d %H:%M').values}"
        if 'step' in ds.coords:
            time_info += f", 예보 시간: +{ds.step.values}h"
            
        ax.set_title(f"{variable_info}\n{time_info}")
        
        # 이미지 저장
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"이미지 저장 완료: {save_path}")
        
        # 화면에 표시
        if show:
            plt.show()
        else:
            plt.close()
            
        print("시각화 완료!")
        return ds
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='GRIB 파일 시각화 도구')
    
    parser.add_argument('--file', '-f', type=str, required=True,
                        help='GRIB2 파일 경로')
    parser.add_argument('--variable', '-v', type=str, default=None,
                        help='표시할 변수 이름 (지정하지 않으면 첫 번째 변수 사용)')
    parser.add_argument('--save', '-s', type=str, default=None,
                        help='이미지 저장 경로 (지정하지 않으면 저장하지 않음)')
    parser.add_argument('--no-show', action='store_true',
                        help='화면에 표시하지 않음')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.file):
        print(f"오류: 파일 '{args.file}'이 존재하지 않습니다.")
        return 1
    
    # GRIB 파일 시각화
    view_grib_file(
        args.file,
        variable=args.variable,
        save_path=args.save,
        show=not args.no_show
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 