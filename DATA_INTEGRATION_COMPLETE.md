# POF-Korea 데이터 통합 완전 문서
# Complete Data Integration Documentation for POF-Korea

## 목차 (Table of Contents)

1. [개요](#1-개요)
2. [변수 명세](#2-변수-명세)
3. [데이터 통합 방법론](#3-데이터-통합-방법론)
4. [라벨 통합 기준](#4-라벨-통합-기준)
5. [검증 프로토콜](#5-검증-프로토콜)
6. [실행 가이드](#6-실행-가이드)
7. [트러블슈팅](#7-트러블슈팅)
8. [부록](#8-부록)

---

# 1. 개요

본 문서는 한국 지역 화재 확률 예측 모델(POF-Korea)의 데이터 통합 전 과정을 포괄적으로 기술한 통합 문서입니다.

## 1.1 프로젝트 개요
- **목적**: ECMWF 글로벌 PoF 모델의 한국 지역 최적화
- **대상 지역**: 한국 (33°N-39°N, 124°E-132°E)
- **공간 해상도**: 0.1° × 0.1° (약 11km × 11km)
- **시간 범위**: 2000년 11월 2일 ~ 2024년 12월 31일
- **예측 변수**: 19개 (동적 14개, 준정적 3개, 정적 5개)
- **라벨**: af_flag (화재 발생 여부, 0/1)

## 1.2 핵심 설계 원칙
- **Grid ID 시스템**: 전역 고유 식별자 사용
- **시간 정합성**: UTC 기준 일 단위 집계
- **공간 필터링**: 지번 데이터 기반 육지 격자 선정 (1,007개)
- **도메인 특화**: 산림 격자 필터링 (581개)

## 1.3 Grid ID 체계
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

### 주요 Grid ID 예시
- 서울 (37.5°N, 127.0°E): 4,593,070
- 부산 (35.1°N, 129.0°E): 4,556,692
- 제주 (33.5°N, 126.5°E): 4,525,065

---

# 2. 변수 명세

## 2.1 변수 분류 체계
- **동적 변수 (Dynamic, 14개)**: 일별 또는 실시간 갱신
- **준정적 변수 (Semi-static, 3개)**: 월별 갱신
- **정적 변수 (Static, 5개)**: 고정값 또는 연별 갱신

## 2.2 동적 변수 (Dynamic Variables)

### 2.2.1 기상 변수 (Weather Variables)

| 변수명 | 설명 | 자료형 | 단위 | 범위 | 소스 | 갱신주기 |
|--------|------|--------|------|------|------|---------|
| t2m | 2m 기온 | float32 | K | 240-320 | ERA5 | 일별 |
| td2m | 2m 이슬점 온도 | float32 | K | 230-300 | ERA5 | 일별 |
| wind10m | 10m 풍속 | float32 | m/s | 0-30 | ERA5 | 일별 |
| tp | 총 강수량 | float32 | m | 0-0.5 | ERA5 | 일별 |

#### 풍속 계산
```python
wind10m = np.sqrt(u10**2 + v10**2)
```

### 2.2.2 연료 습도 변수 (Fuel Moisture)

| 변수명 | 설명 | 단위 | 범위 | 갱신주기 | 결측처리 |
|--------|------|------|------|---------|---------|
| LFMC | 살아있는 연료 습도 | % | 50-200 | 월별 | 3단계 보간 |
| LFMC_low | 하층 캐노피 습도 | % | 40-180 | 월별 | 3단계 보간 |
| LFMC_high | 상층 캐노피 습도 | % | 60-220 | 월별 | 3단계 보간 |
| DFMC_Foliage | 죽은 잎 습도 | % | 2-35 | 일별 | 3단계 보간 |
| DFMC_Wood | 죽은 나무 습도 | % | 5-40 | 일별 | 3단계 보간 |
| fuelload_dead | 죽은 연료 적재량 | kg/m² | 0-5 | 월별 | 식생별 기본값 |
| fuelload_live | 살아있는 연료 적재량 | kg/m² | 0-10 | 월별 | 식생별 기본값 |

#### 3단계 계층적 보간
1. **실측값**: 2011-2021 기간 직접 관측
2. **Climatology**: 월별 평균값
3. **IDW**: 공간 보간 (반경 21 격자, Manhattan distance)

### 2.2.3 기타 동적 변수

| 변수명 | 설명 | 단위 | 범위 | 소스 |
|--------|------|------|------|------|
| lightning_flash_rate | 번개 발생률 | flashes/km²/day | 0-10 | LIS |

## 2.3 준정적 변수 (Semi-static Variables)

| 변수명 | 설명 | 단위 | 범위 | 소스 | 갱신주기 |
|--------|------|------|------|------|---------|
| VOD-L | L-밴드 식생 광학 깊이 | 무차원 | 0-1.5 | SMOS/SMAP | 월별 |
| LAI_low | 하층 엽면적 지수 | m²/m² | 0-8 | MODIS | 8일→월 |
| LAI_high | 상층 엽면적 지수 | m²/m² | 0-10 | MODIS | 8일→월 |

## 2.4 정적 변수 (Static Variables)

| 변수명 | 설명 | 자료형 | 단위 | 범위 | 소스 | 갱신주기 |
|--------|------|--------|------|------|------|---------|
| vegetation_type | 식생 유형 | int8 | IGBP코드 | 0-17 | MODIS | 연별 |
| orography | 지형 고도 | float32 | m | 0-2000 | SRTM | 불변 |
| urban_fraction | 도시 비율 | float32 | 비율 | 0-1 | MODIS | 5년 |
| population_density | 인구 밀도 | float32 | 명/km² | 0-30000 | GPW v4 | 5년 |
| road_density | 도로 밀도 | float32 | km/km² | 0-50 | OSM | 연별 |

### 식생 유형 코드 (IGBP)
```python
FOREST_CODES = [
    1,  # Evergreen Needleleaf Forests
    2,  # Evergreen Broadleaf Forests
    3,  # Deciduous Needleleaf Forests
    4,  # Deciduous Broadleaf Forests
    5   # Mixed Forests
]
```

## 2.5 라벨 변수

| 변수명 | 설명 | 자료형 | 값 | 소스 | 기준 |
|--------|------|--------|-----|------|------|
| af_flag | 화재 발생 여부 | int8 | 0/1 | MODIS/VIIRS | Confidence ≥ 30% |

---

# 3. 데이터 통합 방법론

## 3.1 데이터 소스

| 데이터셋 | 소스 | 시간 해상도 | 공간 해상도 | 변수 |
|---------|------|------------|------------|------|
| ERA5 Weather | Copernicus CDS | 일별 | 0.1° | t2m, td2m, wind10m, tp |
| AF_Flag | MODIS/VIIRS | 일별 | 0.1° | 화재 발생 플래그 (0/1) |
| Landcover | MODIS MCD12Q1 | 연별 | 0.1° | lc_type1 (토지피복) |
| Population | GPW v4 | 5년 | 0.1° | 인구밀도 |
| Road Density | OSM | 정적 | 0.1° | 도로밀도 |
| LFMC/DFMC | ECMWF | 월별 | 0.1° | 연료 습도 |

## 3.2 통합 프로세스

### Step 1: Weather + AF_Flag 결합
```python
# INNER JOIN - 육지 격자만 선택
weather_af_land = pd.merge(
    weather,
    af_flag,
    on=["grid_id", "date"],
    how="inner",
    suffixes=('', '_af')
)
# 결과: 4,247,637 rows
```

**설계 근거**:
- 화재 데이터가 존재하는 육지 격자만 모델링
- 바다/사막 등 화재 불가능 지역 자동 제외
- 클래스 불균형 완화

### Step 2: 한국 영역 필터링
```python
# 지번 데이터 기반 육지 마스킹
land_grids = load_jibun_master()  # 1,007개 격자

# 육지만 추출하는 이유:
# 1. 바다/강 화재는 무의미
# 2. 계산 효율성 (4,800 → 1,007 격자)
# 3. 클래스 균형 개선
```

### Step 3: 정적 변수 결합
```python
# LEFT JOIN으로 정적 변수 추가
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
```

### Step 4: Landcover 동적 결합
```python
# 중복 처리: (grid_id, year)별 최빈값
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
# 2001 → 2000 복사 (11월 2일 이후만)
```

### Step 5: 산림 필터링
```python
# 산림 격자만 추출
forest_data = result[result["lc_type1"].isin(FOREST_CODES)].copy()
# 결과: 2,102,696 rows → 581개 고유 격자
```

**효과**:
- 화재 positive 비율: 0.4% → 0.8%
- 도메인 특화 모델링
- 노이즈 감소

### Step 6: LFMC/DFMC 보간
```python
def idw_interpolation(df, target_col, max_radius=21):
    """역거리 가중 보간"""
    for radius in range(1, max_radius+1, 2):
        for idx in missing_idx:
            # Manhattan distance 기반 이웃 탐색
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    dist = abs(dr) + abs(dc)
                    if dist > 0 and dist <= radius:
                        weight = 1.0 / dist
                        # 가중 평균 계산
```

---

# 4. 라벨 통합 기준

## 4.1 화재 탐지 알고리즘

### MODIS Active Fire Detection
- **센서**: Terra/Aqua MODIS
- **탐지 조건**:
  - T4μm > 310K (주간) / 305K (야간)
  - ΔT (4μm - 11μm) > 10K (주간) / 8K (야간)
  - 공간 일관성 테스트 통과
- **신뢰도 등급**:
  - Low (0-30%): 제외
  - Nominal (30-80%): **포함** ← 기준선
  - High (80-100%): 포함

### VIIRS Enhanced Fire Detection
- **센서**: Suomi-NPP/NOAA-20
- **해상도**: 375m (MODIS 1km 대비 개선)
- **운영**: 2012년부터

## 4.2 시공간 집계

### 공간 집계 (0.1° 격자)
```python
def aggregate_fire_to_grid(fire_points, grid_resolution=0.1):
    """
    원리:
    - 0.1° ≈ 11km × 11km (한국 위도)
    - 격자 내 1개 이상 화재 → af_flag = 1
    - 격자 내 0개 화재 → af_flag = 0
    """
    for grid in all_grids:
        fires_in_grid = fire_points.within(grid.bounds)
        grid.af_flag = 1 if len(fires_in_grid) > 0 else 0
    return grid_fires
```

### 시간 집계 (일 단위)
```python
def aggregate_daily(fire_data):
    """
    UTC 00:00 - 23:59 기준
    한국 시간 = UTC + 9시간
    """
    daily_fire = fire_data.groupby(['grid_id', 'date']).agg({
        'fire_count': 'sum',
        'confidence': 'max',
        'frp': 'mean'  # Fire Radiative Power
    })
    daily_fire['af_flag'] = (daily_fire['fire_count'] > 0).astype(int)
    return daily_fire
```

## 4.3 클래스 불균형 처리

### 현황
- 전체 데이터: 0.3955% positive (296/74,844)
- 산림 필터링 후: ~0.8% positive
- 극심한 불균형 (99.2% negative)

### 처리 전략
```python
# 1. 모델 레벨
scale_pos_weight = negative_count / positive_count  # ~250

# 2. 평가 지표
# Accuracy 대신 F1-score, PR-AUC 사용
```

## 4.4 한계점
1. **탐지 한계**: 구름/연기 차폐, 야간 소규모 화재
2. **시간 해상도**: 위성 통과 시각 제한 (하루 2-4회)
3. **공간 해상도**: 0.1° 격자 내 정확한 위치 불명

---

# 5. 검증 프로토콜

## 5.1 데이터 수집 검증

### ERA5 기상 데이터
- ✅ 시간 범위: 2000-01-01 ~ 2024-12-31
- ✅ 공간 범위: 33°N-39°N, 124°E-132°E
- ✅ 변수 완전성: t2m, td2m, 10u, 10v, tp
- ✅ 단위: 온도(K), 강수(m), 바람(m/s)

### MODIS/VIIRS 화재 데이터
- ✅ 시작 날짜: 2000-11-02 이후
- ✅ Confidence 필터: ≥ 30%
- ✅ 중복 제거 완료
- ✅ Type 필터: vegetation fire (0,1,2)

## 5.2 통합 검증

### Step 1: Weather + AF_Flag 검증
```python
def validate_weather_af_join(weather, af_flag, result):
    """1차 결합 검증"""
    assert len(result) == len(af_flag), "육지 격자 기준 유지 실패"
    assert result.date.min() >= '2000-11-02', "MODIS 이전 데이터 포함"
    assert result.grid_id.nunique() == 1007, "육지 격자 수 불일치"
    assert not result[['grid_id','date']].duplicated().any(), "중복 존재"
    print("✓ Weather-AF 결합 검증 통과")
```

### Step 2: 한국 영역 검증
```python
def validate_korea_filter(data_after):
    """한국 경계 필터링 검증"""
    assert data_after.latitude.between(33, 39).all()
    assert data_after.longitude.between(124, 132).all()
    print("✓ 한국 영역 필터링 검증 통과")
```

### Step 3: 정적 변수 검증
```python
def validate_static_merge(data, static_vars):
    """정적 변수 병합 검증"""
    for var in static_vars:
        missing_rate = data[var].isna().mean()
        assert missing_rate < 0.1, f"{var} 과다 결측: {missing_rate:.1%}"
    print("✓ 정적 변수 결합 검증 통과")
```

## 5.3 품질 검증

### 물리적 타당성
```python
# 온도 범위
assert (df.t2m > 200).all() and (df.t2m < 350).all()
assert df.td2m <= df.t2m  # 이슬점 ≤ 기온

# 강수량
assert (df.tp >= 0).all() and (df.tp < 1).all()

# 풍속
assert (df.wind10m >= 0).all() and (df.wind10m < 50).all()
```

### 시계열 연속성
```python
date_range = pd.date_range('2000-11-02', '2024-12-31', freq='D')
missing_dates = set(date_range) - set(df.date)
assert len(missing_dates) < 100, f"과다 누락 날짜: {len(missing_dates)}"
```

### 공간 일관성
```python
# 인접 격자 온도 차이 < 10K
for grid in grids:
    neighbors = get_neighbors(grid)
    temp_diff = abs(df[grid].t2m - df[neighbors].t2m.mean())
    assert temp_diff < 10
```

### 클래스 균형
```python
pos_rate = df.af_flag.mean()
assert 0.001 < pos_rate < 0.05, f"비정상 화재율: {pos_rate:.2%}"
```

## 5.4 자동화 검증

### 검증 스크립트
```bash
#!/bin/bash
# run_validation.sh

echo "=== POF-Korea 데이터 검증 시작 ==="

# 1. 데이터 존재 확인
python validate_data_existence.py

# 2. Grid ID 검증
python validate_grid_system.py

# 3. 시계열 검증
python validate_temporal.py

# 4. 공간 검증
python validate_spatial.py

# 5. 통합 검증
python validate_integration.py

# 6. 최종 보고서
python generate_validation_report.py

echo "=== 검증 완료 ==="
```

---

# 6. 실행 가이드

## 6.1 환경 설정

### 필수 패키지 설치
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt

# XGBoost 호환성
pip install xgboost==1.7.6  # 3.0+ 비호환
```

### API 키 설정
```bash
# .env 파일 생성
CDS_API_URL=https://cds.climate.copernicus.eu/api
CDS_API_KEY=your-api-key-here
```

## 6.2 데이터 수집

### ERA5 데이터
```python
python src/data_collection/collect_era5_data.py \
    --start-date 2000-01-01 \
    --end-date 2024-12-31 \
    --bbox 33,124,39,132
```

### MODIS 화재 데이터
```python
python src/data_collection/collect_fire_data.py \
    --source MODIS \
    --start-date 2000-11-02 \
    --end-date 2024-12-31
```

## 6.3 데이터 통합

### 전체 파이프라인
```python
# 통합 실행
python run.py

# 개별 단계
python integrate_weather_fire.py
python filter_korea_region.py
python add_static_variables.py
python add_landcover.py
python filter_forest.py
python interpolate_lfmc_dfmc.py
```

## 6.4 모델 학습

### 기본 학습
```python
python src/modeling/train_model.py
```

### 하이퍼파라미터 최적화
```python
python src/modeling/train_model_optuna.py
```

---

# 7. 트러블슈팅

## 7.1 메모리 부족

### 청크 단위 처리
```python
def process_in_chunks(df, chunk_size=100000):
    chunks = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start+chunk_size]
        processed = process_chunk(chunk)
        chunks.append(processed)
    return pd.concat(chunks, ignore_index=True)
```

## 7.2 Grid ID 불일치

### Grid ID 재계산 검증
```python
def verify_grid_id(df):
    recalc_grid = df.apply(
        lambda row: latlon2grid(row['latitude'], row['longitude']),
        axis=1
    )
    mismatches = df['grid_id'] != recalc_grid
    if mismatches.any():
        logger.warning(f"Grid ID 불일치: {mismatches.sum()}개")
```

## 7.3 시간대 문제

### UTC 통일
```python
df['date'] = pd.to_datetime(df['date'], utc=True)
```

## 7.4 Join 후 행 수 폭증

### 원인 및 해결
```python
# 문제: 중복 키로 인한 카테시안 곱
# 해결: 중복 제거 후 join
df1 = df1.drop_duplicates(['grid_id', 'date'])
df2 = df2.drop_duplicates(['grid_id', 'date'])
result = pd.merge(df1, df2, on=['grid_id', 'date'])
```

---

# 8. 부록

## 8.1 파일 경로 구조

```yaml
pof-model-korea/
├── data/
│   ├── raw/
│   │   ├── era5/            # ERA5 NetCDF 파일
│   │   ├── modis_fire/      # MODIS 화재 데이터
│   │   └── static/          # 정적 변수 데이터
│   ├── processed/
│   │   ├── weather_af_land.parquet
│   │   ├── korea_region.parquet
│   │   └── forest_filtered.parquet
│   └── integrated/
│       └── final_dataset.parquet
├── src/
│   ├── data_collection/
│   ├── preprocessing/
│   ├── data_integration/
│   └── modeling/
├── outputs/
│   ├── models/
│   │   └── xgboost_pof_korea.json
│   └── data/
│       └── weather_data_with_wind.csv
└── logs/
    └── data_integration.log
```

## 8.2 주요 상수 및 설정

```python
# Grid 시스템 상수
GRID_RESOLUTION = 0.1  # degrees
LAT_OFFSET = 900
LON_OFFSET = 1800
NLON = 3600

# 한국 영역
KOREA_BBOX = {
    'lat_min': 33.0,
    'lat_max': 39.0,
    'lon_min': 124.0,
    'lon_max': 132.0
}

# 육지 격자
LAND_GRID_COUNT = 1007
FOREST_GRID_COUNT = 581

# 산림 코드
FOREST_CODES = [1, 2, 3, 4, 5]  # IGBP 분류

# 시간 범위
START_DATE = '2000-11-02'  # MODIS 시작
END_DATE = '2024-12-31'
```

## 8.3 성능 최적화

### 병렬 처리
```python
from multiprocessing import Pool

def parallel_interpolation(df, n_cores=4):
    """병렬 IDW 보간"""
    chunks = np.array_split(df, n_cores)
    with Pool(n_cores) as pool:
        results = pool.map(idw_interpolation, chunks)
    return pd.concat(results)
```

### 인덱싱 최적화
```python
# 조인 전 인덱스 설정
df1.set_index(['grid_id', 'date'], inplace=True)
df2.set_index(['grid_id', 'date'], inplace=True)
result = df1.join(df2, how='inner')
```

### Dask 활용 (대용량 처리)
```python
import dask.dataframe as dd

# Pandas DataFrame을 Dask DataFrame으로 변환
ddf = dd.from_pandas(df, npartitions=10)

# 병렬 연산
result = ddf.groupby('grid_id').mean().compute()
```

## 8.4 검증 메트릭

### 모델 평가 지표
```python
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_auc_score,
    f1_score,
    confusion_matrix
)

def evaluate_model(y_true, y_pred_proba):
    """포괄적 모델 평가"""
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': precision_recall_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred_proba > 0.5),
    }

    cm = confusion_matrix(y_true, y_pred_proba > 0.5)
    metrics['precision'] = cm[1,1] / (cm[1,1] + cm[0,1])
    metrics['recall'] = cm[1,1] / (cm[1,1] + cm[1,0])

    return metrics
```

## 8.5 참고 문헌

1. ECMWF Global PoF Model Documentation
2. MODIS Fire Products Algorithm Theoretical Basis Document
3. GPW v4 Technical Documentation
4. IGBP Land Cover Classification System
5. Copernicus Climate Data Store API Documentation
6. GeoPandas Spatial Joins Guide
7. GADM Database of Global Administrative Areas

## 8.6 버전 이력

| 버전 | 날짜 | 변경사항 |
|------|------|----------|
| 1.0 | 2024-12-15 | 초기 데이터 통합 |
| 1.1 | 2025-01-17 | 문서 통합 및 상세화 |
| 2.0 | 2025-01-17 | 완전 통합 문서 생성 |

---

## 8.7 연락처 및 기여

- **프로젝트 리드**: POF-Korea Team
- **GitHub**: https://github.com/your-org/pof-model-korea
- **이슈 제보**: https://github.com/your-org/pof-model-korea/issues
- **최종 수정**: 2025년 1월 17일

---

# 검증 체크리스트 (Quick Reference)

## 필수 확인 사항 ✅
- [ ] Grid ID: 1,007개 (육지 격자)
- [ ] 날짜 범위: 2000-11-02 ~ 2024-12-31
- [ ] af_flag 비율: 0.3-0.5% (전체), 0.8% (산림)
- [ ] 중복 없음: `df[['grid_id','date']].duplicated().any() == False`
- [ ] 결측 < 5%: 주요 변수 결측률 확인

## 데이터 크기 확인 📊
- [ ] weather_af_land: ~4.2M rows
- [ ] forest_filtered: ~2.1M rows
- [ ] 최종 파일: < 1GB (Parquet)

## 물리적 타당성 🌡️
- [ ] 온도: 200K < t2m < 350K
- [ ] 이슬점: td2m ≤ t2m
- [ ] 강수: 0 ≤ tp < 1m
- [ ] 풍속: 0 ≤ wind10m < 50 m/s

---

*© 2025 POF-Korea Project. All rights reserved.*