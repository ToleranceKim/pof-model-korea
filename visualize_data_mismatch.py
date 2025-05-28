#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 경로 설정
weather_path = 'processed_data/era5_korea_2020_2024.parquet'  # 통합된 ERA5 데이터 (2020-2024년)
old_weather_path = 'processed_data/era5_korea_201101.csv'  # 이전 ERA5-Land 데이터 (비교용)
fire_path = 'data/af_flag/fire_archive_M-C61_615377.csv'  # 원본 산불 데이터

# 날씨 데이터 로드
print("Loading ERA5 weather data (2020-2024)...")
weather_df = pd.read_parquet(weather_path)
weather_df = weather_df[['grid_id', 'latitude', 'longitude']].drop_duplicates()

# 산불 데이터 로드
print("Loading fire data...")
fire_df = pd.read_csv(fire_path)

# 산불 데이터에 grid_id 추가
fire_df['lon_bin'] = np.floor(fire_df.longitude / 0.1).astype(int)
fire_df['lat_bin'] = np.floor(fire_df.latitude / 0.1).astype(int)
fire_df['grid_id'] = (fire_df.lat_bin + 900) * 3600 + (fire_df.lon_bin + 1800)

# 데이터 범위 통계
print("\n===== Data Range Statistics =====")
print(f"Weather data grid count: {weather_df.shape[0]}")
print(f"Fire data grid count: {fire_df.grid_id.nunique()}")
print(f"Total fire records: {fire_df.shape[0]}")

# 위경도 범위 비교
print("\n===== Longitude/Latitude Range =====")
print(f"Weather data - Longitude: {weather_df.longitude.min():.4f}° ~ {weather_df.longitude.max():.4f}° (Count: {weather_df.longitude.nunique()})")
print(f"Weather data - Latitude: {weather_df.latitude.min():.4f}° ~ {weather_df.latitude.max():.4f}° (Count: {weather_df.latitude.nunique()})")
print(f"Fire data - Longitude: {fire_df.longitude.min():.4f}° ~ {fire_df.longitude.max():.4f}°")
print(f"Fire data - Latitude: {fire_df.latitude.min():.4f}° ~ {fire_df.latitude.max():.4f}°")

# 산불 데이터와 날씨 데이터 간의 범위 비교
fire_grid_ids = set(fire_df.grid_id.unique())
weather_grid_ids = set(weather_df.grid_id.unique())
in_range = len(fire_grid_ids.intersection(weather_grid_ids))
out_range = len(fire_grid_ids - weather_grid_ids)
total_fire_grids = len(fire_grid_ids)

print("\n===== Fire Data Coverage Analysis =====")
print(f"Fire grids within weather data range: {in_range} ({in_range/total_fire_grids*100:.2f}%)")
print(f"Fire grids outside weather data range: {out_range} ({out_range/total_fire_grids*100:.2f}%)")

# 실제 수집 범위 계산
actual_collection_bbox = [
    weather_df.longitude.min(),
    weather_df.longitude.max(),
    weather_df.latitude.min(),
    weather_df.latitude.max()
]

# 의도된 수집 범위
intended_bbox = [124, 132, 33, 39]  # [서경, 동경, 남위, 북위]

# 시각화 설정
plt.figure(figsize=(10, 8))
sns.set_style('whitegrid')

# 날씨 데이터 포인트 그리기 (파란색)
plt.scatter(weather_df.longitude, weather_df.latitude, 
            color='blue', alpha=0.4, label='Weather Data', s=10)

# 산불 데이터 포인트 그리기 (빨간색)
plt.scatter(fire_df.longitude, fire_df.latitude, 
            color='red', alpha=0.6, label='Fire Data', s=10)

# 의도된 날씨 데이터 수집 범위 표시 (녹색 점선 박스)
plt.plot([intended_bbox[0], intended_bbox[1], intended_bbox[1], intended_bbox[0], intended_bbox[0]],
         [intended_bbox[2], intended_bbox[2], intended_bbox[3], intended_bbox[3], intended_bbox[2]],
         'g--', label='Intended Collection Range', linewidth=2)

# 실제 날씨 데이터 수집 범위 표시 (파란색 실선 박스)
plt.plot([actual_collection_bbox[0], actual_collection_bbox[1], actual_collection_bbox[1], actual_collection_bbox[0], actual_collection_bbox[0]],
         [actual_collection_bbox[2], actual_collection_bbox[2], actual_collection_bbox[3], actual_collection_bbox[3], actual_collection_bbox[2]],
         'b-', label='Actual Collection Range', linewidth=2)

# 그래프 꾸미기
plt.title('Spatial Distribution Comparison of Weather and Fire Data', fontsize=14)
plt.xlabel('Longitude (°E)', fontsize=12)
plt.ylabel('Latitude (°N)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 축 범위 설정
plt.xlim(123.5, 132.5)
plt.ylim(32.5, 39.5)

# 격자선 추가
plt.grid(True, linestyle='--', alpha=0.3)

# 분석 정보 추가 (하단에 노란색 박스)
plt.figtext(0.5, 0.01, 
            f'Fire data outside weather data range: {out_range}/{total_fire_grids} ({out_range/total_fire_grids*100:.2f}%)',
            ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# 저장 및 표시
output_dir = 'outputs/analysis'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'weather_fire_spatial_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"Visualization image saved to {output_dir} directory.") 