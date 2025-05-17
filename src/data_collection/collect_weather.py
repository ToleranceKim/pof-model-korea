#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cdsapi
import calendar
import os
import sys

def main():
    print("=== 날씨 데이터 수집 시작 ===")
    
    try:
        c = cdsapi.Client()
        
        years  = ["2025"]
        months = [f"{m:02d}" for m in range(1,5)]
        days   = [f"{d:02d}" for d in range(1,32)]
        times  = [f"{h:02d}:00" for h in range(0,24)]
        # BBOX 확장: 북위 39도까지 포함
        bbox   = [39, 124, 33, 132]  # [북위, 서경, 남위, 동경]
        
        for year in years:
            for month in months:
                # 해당 월의 실제 일수만 추리기
                max_day = calendar.monthrange(int(year), int(month))[1]
                day_list = [f"{d:02d}" for d in range(1, max_day+1)]
                
                target_file = os.path.join("..", "data", f"era5_korea_{year}{month}.nc")
                print(f"Retrieving {target_file} ...")
                
                c.retrieve(
                    'reanalysis-era5-land',
                    {
                        'variable': [
                            '2m_temperature','2m_dewpoint_temperature',
                            '10m_u_component_of_wind','10m_v_component_of_wind',
                            'total_precipitation',
                        ],
                        'product_type': 'reanalysis',
                        'year':   [year],
                        'month':  [month],
                        'day':    day_list,
                        'time':   times,
                        'area':   bbox,
                        'format': 'netcdf'
                    },
                    target_file
                )
                print(f"Successfully downloaded {target_file}")
        
        print("=== 날씨 데이터 수집 완료 ===")
        return 0
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 