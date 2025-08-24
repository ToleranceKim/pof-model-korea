# POF-Korea 데이터 통합 방법론 가이드

## 개요
본 문서는 한국 지역 화재 확률 예측 모델(POF-Korea)의 데이터 통합 파이프라인을 상세히 기술합니다. 
이 가이드를 통해 전체 데이터 결합 과정을 재현할 수 있습니다.

---

## 1. 데이터 통합 아키텍처 개요

### 1.1 핵심 설계 원칙
- **공간 해상도**: 0.1° × 0.1° (약 11km × 11km) 그리드 체계
- **시간 해상도**: 일 단위 데이터
- **대상 지역**: 한국 영역 (33°N-39°N, 124°E-132°E)
- **기간**: 2000년 11월 2일 ~ 2024년 12월 31일

### 1.2 Grid ID 시스템
```python
def latlon2grid(lat: float, lon: float) -> int:
    """위경도를 Grid ID로 변환"""
    lat_bin = int(np.floor(lat / 0.1))
    lon_bin = int(np.floor(lon / 0.1))
    grid_id = (lat_bin + 900) * 3600 + (lon_bin + 1800)
    return grid_id

def grid2latlon(grid_id: int) -> tuple:
    """Grid ID를 위경도로 변환"""
    lat_bin = (grid_id // 3600) - 900
    lon_bin = (grid_id % 3600) - 1800
    lat = lat_bin / 10.0
    lon = lon_bin / 10.0
    return lat, lon
```

---

## 2. 데이터 소스 및 전처리

### 2.1 기본 데이터셋

| 데이터셋 | 소스 | 시간 해상도 | 공간 해상도 | 변수 |
|---------|------|------------|------------|------|
| ERA5 Weather | Copernicus CDS | 일별 | 0.1° | t2m, td2m, wind10m, tp |
| AF_Flag | MODIS/VIIRS | 일별 | 0.1° | 화재 발생 플래그 (0/1) |
| Landcover | MODIS MCD12Q1 | 연별 | 0.1° | lc_type1 (토지피복 유형) |
| Population | GPW v4 | 정적 (2020) | 0.1° | 인구밀도 (명/km²) |
| Road Density | OSM | 정적 | 0.1° | 도로밀도 (km/km²) |
| LFMC/DFMC | ECMWF | 월별 | 0.1° | 연료 습도 지표 |

### 2.2 데이터 수집 코드
```python
# ERA5 데이터 수집
python src/data_collection/collect_era5_data.py \
    --start-date 2000-01-01 \
    --end-date 2024-12-31 \
    --bbox 33,124,39,132

# MODIS 화재 데이터 수집
python src/data_collection/collect_fire_data.py \
    --source MODIS \
    --start-date 2000-11-02 \
    --end-date 2024-12-31
```

---

## 3. 단계별 데이터 통합 프로세스

### 3.1 Step 1: 기본 데이터 결합 (Weather + AF_Flag)

#### 목적
- 기상 데이터와 화재 발생 데이터를 시공간적으로 매칭
- 육지 격자만 추출하여 모델링 대상 범위 한정

#### 구현 코드
```python
import pandas as pd
import geopandas as gpd
from pathlib import Path

# 1. 데이터 로드
weather = pd.read_csv("era5_daily_combined_200001_202412.csv")
af_flag = pd.read_csv("af_flag_land.csv")

# 2. 날짜 컬럼 통일
weather.rename(columns={'acq_date': 'date'}, inplace=True)

# 3. Inner Join - 육지 격자만 선택
weather_af_land = pd.merge(
    weather, 
    af_flag,
    on=["grid_id", "date"],
    how="inner",
    suffixes=('', '_af')
)

print(f"결합 결과: {len(weather_af_land):,} rows")
# Expected: 4,247,637 rows
```

#### 설계 근거
- **INNER JOIN 선택 이유**: 
  - 화재 데이터가 존재하는 육지 격자만 모델링 대상
  - 바다/사막 등 화재 불가능 지역 자동 제외
  - 클래스 불균형 완화 (전체 격자 대비 약 30% 축소)

### 3.2 Step 2: 한국 영역 필터링

#### 목적
- 정확한 한국 행정구역 경계 내 격자만 추출
- 복잡한 해안선과 도서 지역 정확 처리

#### 구현 코드
```python
# GADM 한국 경계 폴리곤 로드
kor_boundary = gpd.read_file("gadm41_KOR_0.json")

# Grid ID를 좌표로 변환
weather_af_land[['latitude', 'longitude']] = weather_af_land['grid_id'].apply(
    lambda g: pd.Series(grid2latlon(g))
)

# GeoDataFrame 생성
gdf = gpd.GeoDataFrame(
    weather_af_land,
    geometry=gpd.points_from_xy(
        weather_af_land['longitude'], 
        weather_af_land['latitude']
    ),
    crs="EPSG:4326"
)

# 한국 경계 내부 점만 추출
korea_data = gdf.sjoin(
    kor_boundary[['geometry']], 
    how='inner',
    predicate='within'
)

# 불필요한 컬럼 제거
korea_data = korea_data.drop(columns=['geometry', 'index_right'])

print(f"한국 영역 필터링 후: {len(korea_data):,} rows")
```

#### 설계 근거
- **GADM 폴리곤 vs Bounding Box**:
  - Bounding Box: 빠르지만 "세밀한 해안선 반영 어려움"
  - GADM 폴리곤: 정확한 행정구역 경계 반영
  - 서해안 리아스식 해안 정확 처리
  - 제주도, 울릉도 등 도서 지역 명확 포함/제외

### 3.3 Step 3: 정적 변수 결합

#### 목적
- 시간 불변 공간 특성 데이터 추가
- 인구밀도, 도로밀도, 지형, 도시분율, 식생유형

#### 구현 코드
```python
# 정적 데이터 로드
population = pd.read_parquet("population_density_2020.parquet")
road = pd.read_parquet("road_density_0.1deg.parquet")
orography = pd.read_csv("orography_korea_cleaned.csv")
urban_frac = pd.read_csv("urban_frac_korea_cleaned.csv")
vegetation = pd.read_csv("vegetation_type_korea.csv")

# Left Join으로 정적 변수 추가
# (모든 타겟 격자에 정적 변수 부여, 결측 허용)
result = korea_data

for static_data, name in [
    (population, 'pop'), 
    (road, 'road'),
    (orography, 'oro'),
    (urban_frac, 'urban'),
    (vegetation, 'vege')
]:
    result = result.merge(
        static_data,
        on="grid_id",
        how="left",
        suffixes=('', f'_{name}')
    )
    print(f"{name} 결합 후 결측: {result[name].isna().sum()}")
```

#### 설계 근거
- **LEFT JOIN 선택 이유**:
  - 기본 데이터의 모든 행 보존
  - 일부 격자에 정적 데이터 없어도 모델링 진행
  - 결측값은 모델이 학습 중 처리

### 3.4 Step 4: 동적 변수 결합 (Landcover)

#### 목적
- 연도별 변화하는 토지피복 정보 추가
- 산림 격자 필터링을 위한 기준 변수

#### 구현 코드
```python
# Landcover 데이터 로드 (2001-2023)
landcover = pd.read_parquet("landcover_type1_korea_2001_2023.parquet")

# 날짜 처리
result['year'] = pd.to_datetime(result['date']).dt.year
landcover['year'] = pd.to_datetime(landcover['date']).dt.year

# 중복 처리: (grid_id, year)별 최빈값 선택
def mode(s):
    return s.value_counts().idxmax()

landcover_unique = (
    landcover[["grid_id", "year", "lc_type1"]]
    .groupby(["grid_id", "year"])
    .agg({"lc_type1": mode})
    .reset_index()
)

# 시간 범위 확장
# 2023 → 2024 복사 (최신 데이터 연장)
lc_2024 = landcover_unique[landcover_unique["year"] == 2023].copy()
lc_2024["year"] = 2024

# 2001 → 2000 복사 (11월 2일 이후만)
lc_2000 = landcover_unique[landcover_unique["year"] == 2001].copy()
lc_2000["year"] = 2000

landcover_full = pd.concat([landcover_unique, lc_2024, lc_2000])

# 결합
result = result.merge(
    landcover_full,
    on=["grid_id", "year"],
    how="left"
)

# 2000년 11월 2일 이전 제외 (MODIS 운영 시작 전)
mask_before = (result["year"] == 2000) & (result["date"] < "2000-11-02")
result.loc[mask_before, "lc_type1"] = np.nan
```

#### 설계 근거
- **중복 처리 (최빈값)**:
  - 같은 (grid_id, year)에 여러 관측값 존재
  - 최빈값으로 대표값 선택하여 1:1 매칭 보장
  
- **시간 범위 확장**:
  - 데이터 연속성 확보
  - 최소한의 가정 (최근/최초 값 복사)

### 3.5 Step 5: 산림 필터링

#### 목적
- 산불 발생 가능 지역으로 한정
- 모델 정확도 향상 및 클래스 균형 개선

#### 구현 코드
```python
# 산림 코드 정의
FOREST_CODES = [
    1,  # Evergreen Needleleaf Forests
    2,  # Evergreen Broadleaf Forests  
    3,  # Deciduous Needleleaf Forests
    4,  # Deciduous Broadleaf Forests
    5   # Mixed Forests
]

# 산림 격자만 추출
forest_data = result[result["lc_type1"].isin(FOREST_CODES)].copy()

print(f"산림 필터링 전: {len(result):,} rows")
print(f"산림 필터링 후: {len(forest_data):,} rows")
print(f"축소율: {len(forest_data)/len(result)*100:.1f}%")
# Expected: 2,102,696 rows (49.5%)
```

#### 설계 근거
- **도메인 특화**: 산불은 주로 산림에서 발생
- **노이즈 감소**: 도시/농지 등 무관한 격자 제거
- **클래스 균형**: 화재 positive 비율 0.4% → 0.8% 개선

### 3.6 Step 6: 시계열 변수 보간 (LFMC/DFMC)

#### 목적
- 연료 습도 데이터의 시공간적 결측 처리
- 계절성과 공간 연속성 보존

#### 구현 코드
```python
# LFMC 데이터 로드 (2011-2021 실측)
lfmc = pd.read_parquet("LFMC_combined_2011_2021.parquet")

# 시간 정보 추가
forest_data['month'] = pd.to_datetime(forest_data['date']).dt.month
lfmc['month'] = pd.to_datetime(lfmc['time']).dt.month

# 1차: 실측 데이터 결합
result_lfmc = forest_data.merge(
    lfmc[["grid_id", "time", "LFMC", "LFMC_low", "LFMC_high"]],
    left_on=["grid_id", "date"],
    right_on=["grid_id", "time"],
    how="left"
)

# 2차: Climatology 계산 및 적용
climatology = (
    lfmc.groupby(["grid_id", "month"])[["LFMC", "LFMC_low", "LFMC_high"]]
    .mean()
    .reset_index()
    .rename(columns=lambda x: x + "_clim" if x not in ["grid_id", "month"] else x)
)

result_lfmc = result_lfmc.merge(
    climatology,
    on=["grid_id", "month"],
    how="left"
)

# Climatology로 결측 채움
for col in ["LFMC", "LFMC_low", "LFMC_high"]:
    result_lfmc[f"{col}_filled"] = result_lfmc[col].fillna(
        result_lfmc[f"{col}_clim"]
    )

# 3차: IDW 공간 보간 (남은 결측)
def idw_interpolation(df, target_col, max_radius=21):
    """역거리 가중 보간"""
    missing_idx = df[df[target_col].isna()].index
    
    for radius in range(1, max_radius+1, 2):
        if len(missing_idx) == 0:
            break
            
        for idx in missing_idx:
            grid_id = df.loc[idx, 'grid_id']
            month = df.loc[idx, 'month']
            
            # 현재 격자의 row, col 계산
            row = (grid_id // 3600) - 900
            col = (grid_id % 3600) - 1800
            
            # 이웃 격자 탐색
            numerator = 0
            denominator = 0
            
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    dist = abs(dr) + abs(dc)  # Manhattan distance
                    if dist == 0 or dist > radius:
                        continue
                        
                    neighbor_row = row + dr
                    neighbor_col = col + dc
                    neighbor_grid = (neighbor_row + 900) * 3600 + (neighbor_col + 1800)
                    
                    # 이웃 격자의 같은 월 데이터 찾기
                    neighbor_data = df[
                        (df['grid_id'] == neighbor_grid) & 
                        (df['month'] == month) & 
                        (df[target_col].notna())
                    ]
                    
                    if not neighbor_data.empty:
                        weight = 1.0 / dist
                        value = neighbor_data[target_col].iloc[0]
                        numerator += weight * value
                        denominator += weight
            
            if denominator > 0:
                df.loc[idx, target_col] = numerator / denominator
        
        # 남은 결측 재계산
        missing_idx = df[df[target_col].isna()].index
        print(f"반경 {radius}: {len(missing_idx)} 결측 남음")

# IDW 적용
for col in ["LFMC_filled", "LFMC_low_filled", "LFMC_high_filled"]:
    idw_interpolation(result_lfmc, col)
```

#### 설계 근거
- **3단계 계층적 보간**:
  1. 실측 > 가장 신뢰할 수 있는 관측값
  2. Climatology > 계절 패턴 반영한 월별 평균
  3. IDW > 공간 연속성 기반 최종 보간
  
- **Manhattan Distance 사용**:
  - 격자 체계에 적합
  - 계산 효율성

---

## 4. 데이터 검증 체크리스트

### 4.1 행 수 일관성 검증
```python
def validate_row_consistency(original, merged):
    """결합 후 행 수 폭증 여부 확인"""
    assert original.shape[0] == merged.shape[0], \
        f"행 수 불일치: {original.shape[0]} → {merged.shape[0]}"
    print("✓ 행 수 일관성 검증 통과")
```

### 4.2 Grid ID 범위 검증
```python
def validate_grid_coverage(df):
    """한국 영역 Grid ID 범위 확인"""
    lats = []
    lons = []
    for grid_id in df['grid_id'].unique():
        lat, lon = grid2latlon(grid_id)
        lats.append(lat)
        lons.append(lon)
    
    assert min(lats) >= 33 and max(lats) <= 39, "위도 범위 벗어남"
    assert min(lons) >= 124 and max(lons) <= 132, "경도 범위 벗어남"
    print(f"✓ Grid 범위: {min(lats):.1f}°-{max(lats):.1f}°N, "
          f"{min(lons):.1f}°-{max(lons):.1f}°E")
```

### 4.3 시계열 연속성 검증
```python
def validate_temporal_continuity(df):
    """날짜 연속성 및 범위 확인"""
    df['date'] = pd.to_datetime(df['date'])
    date_range = pd.date_range(
        start='2000-11-02', 
        end='2024-12-31', 
        freq='D'
    )
    
    missing_dates = set(date_range) - set(df['date'].unique())
    if missing_dates:
        print(f"⚠ 누락된 날짜: {len(missing_dates)}일")
    else:
        print("✓ 시계열 연속성 검증 통과")
```

### 4.4 결측값 검증
```python
def validate_missing_values(df, critical_cols):
    """주요 변수 결측값 비율 확인"""
    for col in critical_cols:
        missing_pct = df[col].isna().mean() * 100
        print(f"{col}: {missing_pct:.2f}% 결측")
        if col in ['af_flag', 'grid_id', 'date']:
            assert missing_pct == 0, f"{col}에 결측값 존재"
```

### 4.5 보간 정확성 검증
```python
def validate_interpolation(df, year_range=(2011, 2021)):
    """실측 구간 보간값 정확성 확인"""
    test_period = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    # 실측값과 보간값 비교
    for col in ['LFMC', 'DFMC_Foliage', 'DFMC_Wood']:
        if f'{col}_filled' in df.columns and col in df.columns:
            mask = test_period[col].notna()
            if mask.any():
                diff = test_period.loc[mask, f'{col}_filled'] - test_period.loc[mask, col]
                assert diff.abs().max() < 0.001, f"{col} 보간 오류"
    
    print("✓ 보간 정확성 검증 통과")
```

---

## 5. 실행 스크립트

### 5.1 전체 파이프라인 실행
```bash
#!/bin/bash
# run_data_integration.sh

echo "POF-Korea 데이터 통합 파이프라인 시작"

# 1. 데이터 수집
python collect_data.py

# 2. 전처리
python preprocess_era5.py
python preprocess_fire.py
python preprocess_static.py

# 3. 데이터 통합
python integrate_weather_fire.py
python filter_korea_region.py
python add_static_variables.py
python add_landcover.py
python filter_forest.py
python interpolate_lfmc_dfmc.py

# 4. 검증
python validate_integration.py

echo "데이터 통합 완료"
```

### 5.2 Python 통합 스크립트
```python
# integrate_all.py
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_integration_pipeline():
    """전체 데이터 통합 파이프라인 실행"""
    
    # 1. 기본 데이터 결합
    logger.info("Step 1: Weather + AF_Flag 결합")
    weather_af = integrate_weather_fire()
    
    # 2. 한국 영역 필터링
    logger.info("Step 2: 한국 영역 필터링")
    korea_data = filter_korea_region(weather_af)
    
    # 3. 정적 변수 추가
    logger.info("Step 3: 정적 변수 추가")
    with_static = add_static_variables(korea_data)
    
    # 4. Landcover 추가
    logger.info("Step 4: Landcover 추가")
    with_landcover = add_landcover(with_static)
    
    # 5. 산림 필터링
    logger.info("Step 5: 산림 필터링")
    forest_data = filter_forest(with_landcover)
    
    # 6. LFMC/DFMC 보간
    logger.info("Step 6: LFMC/DFMC 보간")
    final_data = interpolate_fuel_moisture(forest_data)
    
    # 7. 검증
    logger.info("Step 7: 데이터 검증")
    validate_all(final_data)
    
    # 8. 저장
    output_path = Path("data/integrated/final_dataset.parquet")
    final_data.to_parquet(output_path, index=False)
    logger.info(f"최종 데이터셋 저장: {output_path}")
    
    return final_data

if __name__ == "__main__":
    result = run_integration_pipeline()
    print(f"통합 완료: {result.shape}")
```

---

## 6. 트러블슈팅 가이드

### 6.1 메모리 부족 문제
```python
# 청크 단위 처리
def process_in_chunks(df, chunk_size=100000):
    chunks = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start+chunk_size]
        processed = process_chunk(chunk)
        chunks.append(processed)
    return pd.concat(chunks, ignore_index=True)
```

### 6.2 Grid ID 불일치 문제
```python
# Grid ID 재계산으로 검증
def verify_grid_id(df):
    recalc_grid = df.apply(
        lambda row: latlon2grid(row['latitude'], row['longitude']), 
        axis=1
    )
    mismatches = df['grid_id'] != recalc_grid
    if mismatches.any():
        logger.warning(f"Grid ID 불일치: {mismatches.sum()}개")
```

### 6.3 시간대 문제
```python
# UTC로 통일
df['date'] = pd.to_datetime(df['date'], utc=True)
```

---

## 7. 성능 최적화 팁

### 7.1 병렬 처리
```python
from multiprocessing import Pool

def parallel_interpolation(df, n_cores=4):
    """병렬 IDW 보간"""
    chunks = np.array_split(df, n_cores)
    with Pool(n_cores) as pool:
        results = pool.map(idw_interpolation, chunks)
    return pd.concat(results)
```

### 7.2 인덱싱 최적화
```python
# 조인 전 인덱스 설정
df1.set_index(['grid_id', 'date'], inplace=True)
df2.set_index(['grid_id', 'date'], inplace=True)
result = df1.join(df2, how='inner')
```

---

## 8. 참고 문헌 및 리소스

- ECMWF Global PoF Model Documentation
- MODIS Land Cover Type Product (MCD12Q1) User Guide
- Copernicus Climate Data Store API Documentation
- GeoPandas Spatial Joins Guide
- GADM Database of Global Administrative Areas

---

## 부록: 주요 파일 경로

```yaml
data:
  raw:
    weather: "data/raw/era5/"
    fire: "data/raw/modis_fire/"
    static: "data/raw/static/"
  
  processed:
    weather_af: "data/processed/weather_af_land.parquet"
    korea_filtered: "data/processed/korea_region.parquet"
    with_static: "data/processed/with_static_vars.parquet"
    forest_only: "data/processed/forest_filtered.parquet"
  
  final:
    integrated: "data/integrated/final_dataset.parquet"
    
models:
  xgboost: "outputs/models/xgboost_pof_korea.json"
  
logs:
  integration: "logs/data_integration.log"
```

---

*마지막 업데이트: 2024년 12월*
*작성자: POF-Korea 프로젝트 팀*