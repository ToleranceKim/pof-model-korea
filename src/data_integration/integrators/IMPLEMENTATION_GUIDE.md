# Integrators 구현 가이드

## 개요
이 문서는 각 integrator 모듈의 구체적인 구현 방법을 설명합니다. 기존 노트북에서 추출한 실제 데이터 경로와 결합 로직을 기반으로 합니다.

---

## 데이터 경로 맵핑

### 기본 경로 구조
```yaml
# Windows 개발 환경 기준 (노트북에서 추출)
base_path: "C:\\Users\\USER\\Desktop\\my_git\\pof-model-korea"

# macOS 개발 환경으로 변환 시
base_path: "/Users/lord_jubin/Desktop/my_git/pof-model-korea"
```

### 실제 데이터 파일 경로

#### 1. 기본 데이터 (Step 1)
```yaml
weather:
  path: "processed_data/era5_daily_combined_200001_202412.csv"
  key_columns: ["grid_id", "date"]
  rename: {"acq_date": "date"}  # 날짜 컬럼 통일
  
af_flag:
  path: "data/af_flag/af_flag_land.csv" 
  key_columns: ["grid_id", "date"]
  
# 결합 결과
weather_af_land:
  expected_rows: 4_247_637
  output_path: "data/processed/weather_af_land.parquet"
```

#### 2. 정적 변수들 (Steps 2-4)
```yaml
population:
  path: "data/population_density/combined_population_density_2000_2020.parquet"
  key_columns: ["grid_id"]
  value_columns: ["pop_density_2020"]
  
road_density:
  path: "data/road_density/road_density_0.1deg.parquet"
  key_columns: ["grid_id"]
  value_columns: ["road_dens_km_km2"]
  
orography:
  path: "data/Orography/orography_korea_cleaned.csv"
  key_columns: ["grid_id"]
  value_columns: ["elevation_m"]
  
urban_frac:
  path: "data/Urban Frac/urban_frac_korea_cleaned.csv"
  key_columns: ["grid_id"]
  value_columns: ["urban_fraction"]
  
vegetation:
  path: "data/Vegetation/vegetation_type_korea.csv"
  key_columns: ["grid_id"]
  value_columns: ["vegetation_type"]
```

#### 3. 동적 변수들 (Step 5)
```yaml
landcover:
  path: "data/landcover/landcover_type1_korea_2001_2023.parquet"
  key_columns: ["grid_id", "year"]
  value_columns: ["lc_type1", "latitude", "longitude"]
  temporal_range: [2001, 2023]
  extensions:
    - copy_2023_to_2024: true  # 최신 데이터 연장
    - copy_2001_to_2000: true  # 초기 데이터 연장
  
forest_codes: [1, 2, 3, 4, 5]  # 산림 필터링용
```

#### 4. 연료 습도 데이터 (Steps 6-7)
```yaml
lfmc:
  path: "data/LFMC_combined/LFMC_combined_2011_2021.parquet"
  key_columns: ["grid_id", "time"]
  value_columns: ["LFMC", "LFMC_low", "LFMC_high"]
  temporal_range: [2011, 2021]
  interpolation:
    method: "3-step"  # 실측 → climatology → IDW
    idw_max_radius: 21
    
dfmc:
  path: "data/DFMC_combined/DFMC_combined_2011_2021.parquet"
  key_columns: ["grid_id", "time"]
  value_columns: ["DFMC_Foliage", "DFMC_Wood"]
  temporal_range: [2011, 2021]
  interpolation:
    method: "3-step"
    idw_max_radius: 21
```

#### 5. 공간 경계 데이터
```yaml
korea_boundary:
  path: "data/shapefiles/gadm41_KOR_0.json"
  format: "geojson"
  crs: "EPSG:4326"
  
# 대안: Bounding Box (빠르지만 부정확)
korea_bbox:
  min_lat: 33.0
  max_lat: 39.0 
  min_lon: 124.0
  max_lon: 132.0
```

---

## 모듈별 구현 가이드

### 1. weather_fire.py 구현

#### 목적
- ERA5 기상 데이터와 MODIS 화재 데이터를 시공간적으로 결합
- 육지 격자만 추출하여 바다/사막 지역 제외

#### 구현 코드 템플릿
```python
def integrate_weather_fire(
    weather_path: str,
    af_flag_path: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Step 1: Weather + AF_Flag 통합
    
    Args:
        weather_path: ERA5 일별 기상 데이터 경로
        af_flag_path: 화재 플래그 데이터 경로
        output_path: 결과 저장 경로 (optional)
    
    Returns:
        통합된 DataFrame (4,247,637 rows 예상)
    """
    logger.info("Step 1: Weather + AF_Flag 통합 시작")
    
    # 1. 데이터 로드
    weather = pd.read_csv(weather_path)
    af_flag = pd.read_csv(af_flag_path)
    
    # 2. 컬럼 이름 통일
    weather.rename(columns={'acq_date': 'date'}, inplace=True)
    
    # 3. 데이터 타입 통일
    weather['date'] = pd.to_datetime(weather['date'])
    af_flag['date'] = pd.to_datetime(af_flag['date'])
    
    # 4. INNER JOIN - 육지 격자만 선택
    result = pd.merge(
        weather, 
        af_flag,
        on=["grid_id", "date"],
        how="inner",
        suffixes=('', '_af')
    )
    
    # 5. 검증
    validate_row_consistency(weather, result, "weather")
    logger.info(f"통합 완료: {len(result):,} rows")
    
    # 6. 저장
    if output_path:
        result.to_parquet(output_path, index=False)
    
    return result
```

#### 검증 체크리스트
- [ ] 결과 행 수: 4,247,637 rows
- [ ] 필수 컬럼 존재: grid_id, date, af_flag
- [ ] 날짜 범위: 2000-11-02 ~ 2024-12-31
- [ ] Grid ID 범위: 한국 영역 내

### 2. static_vars.py 구현

#### 목적
- 시간 불변 공간 특성 데이터 추가
- 인구밀도, 도로밀도, 지형, 도시분율, 식생유형

#### 구현 코드 템플릿
```python
def add_static_variables(
    base_data: pd.DataFrame,
    static_configs: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Step 2-4: 정적 변수들 추가
    
    Args:
        base_data: 기본 데이터 (weather_af_land)
        static_configs: 정적 데이터 설정 딕셔너리
        
    Returns:
        정적 변수가 추가된 DataFrame
    """
    result = base_data.copy()
    
    for var_name, config in static_configs.items():
        logger.info(f"추가 중: {var_name}")
        
        # 데이터 로드
        if config['path'].endswith('.parquet'):
            static_data = pd.read_parquet(config['path'])
        else:
            static_data = pd.read_csv(config['path'])
        
        # LEFT JOIN으로 추가
        result = result.merge(
            static_data[config['key_columns'] + config['value_columns']],
            on=config['key_columns'],
            how="left",
            suffixes=('', f'_{var_name}')
        )
        
        # 결측값 확인
        missing_count = result[config['value_columns'][0]].isna().sum()
        logger.info(f"{var_name} 결측: {missing_count:,} ({missing_count/len(result)*100:.1f}%)")
    
    return result

# 설정 예시
STATIC_CONFIGS = {
    'population': {
        'path': 'data/population_density/combined_population_density_2000_2020.parquet',
        'key_columns': ['grid_id'],
        'value_columns': ['pop_density_2020']
    },
    'road': {
        'path': 'data/road_density/road_density_0.1deg.parquet', 
        'key_columns': ['grid_id'],
        'value_columns': ['road_dens_km_km2']
    }
    # ... 추가 정적 변수들
}
```

### 3. landcover.py 구현

#### 목적
- 연도별 토지피복 데이터 추가
- 중복 처리 (최빈값 선택)
- 시간 범위 확장 (2023→2024, 2001→2000)
- 산림 격자 필터링

#### 핵심 구현 로직
```python
def add_landcover(
    base_data: pd.DataFrame,
    landcover_path: str,
    forest_codes: List[int] = [1, 2, 3, 4, 5]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 5: Landcover 추가 + 산림 필터링
    
    Returns:
        Tuple[전체 데이터, 산림만 필터링된 데이터]
    """
    # 1. 연도 컬럼 추가
    result = base_data.copy()
    result['year'] = pd.to_datetime(result['date']).dt.year
    
    # 2. Landcover 로드
    landcover = pd.read_parquet(landcover_path)
    landcover['year'] = pd.to_datetime(landcover['date']).dt.year
    
    # 3. 중복 처리: 최빈값 선택
    def mode(s):
        return s.value_counts().idxmax()
    
    landcover_unique = (
        landcover[["grid_id", "year", "lc_type1"]]
        .groupby(["grid_id", "year"])
        .agg({"lc_type1": mode})
        .reset_index()
    )
    
    # 4. 시간 범위 확장
    # 2023 → 2024
    lc_2024 = landcover_unique[landcover_unique["year"] == 2023].copy()
    lc_2024["year"] = 2024
    
    # 2001 → 2000
    lc_2000 = landcover_unique[landcover_unique["year"] == 2001].copy() 
    lc_2000["year"] = 2000
    
    landcover_full = pd.concat([landcover_unique, lc_2024, lc_2000])
    
    # 5. 결합
    result = result.merge(
        landcover_full,
        on=["grid_id", "year"], 
        how="left"
    )
    
    # 6. 2000년 11월 2일 이전 제외
    mask_before = (result["year"] == 2000) & (result["date"] < "2000-11-02")
    result.loc[mask_before, "lc_type1"] = np.nan
    
    # 7. 산림 필터링
    forest_data = result[result["lc_type1"].isin(forest_codes)].copy()
    
    logger.info(f"전체: {len(result):,}, 산림: {len(forest_data):,} "
               f"({len(forest_data)/len(result)*100:.1f}%)")
    
    return result, forest_data
```

### 4. fuel_moisture.py 구현

#### 목적
- LFMC/DFMC 데이터의 시공간적 결측 처리
- 3단계 보간: 실측 → Climatology → IDW

#### 핵심 보간 알고리즘
```python
def interpolate_fuel_moisture(
    base_data: pd.DataFrame,
    lfmc_path: str,
    dfmc_path: str,
    max_radius: int = 21
) -> pd.DataFrame:
    """
    Steps 6-7: LFMC/DFMC 3단계 보간
    """
    result = base_data.copy()
    result['month'] = pd.to_datetime(result['date']).dt.month
    
    # LFMC 보간
    result = _interpolate_variable_group(
        result, lfmc_path, 
        ['LFMC', 'LFMC_low', 'LFMC_high'],
        max_radius
    )
    
    # DFMC 보간  
    result = _interpolate_variable_group(
        result, dfmc_path,
        ['DFMC_Foliage', 'DFMC_Wood'], 
        max_radius
    )
    
    return result

def _interpolate_variable_group(
    data: pd.DataFrame,
    source_path: str, 
    variables: List[str],
    max_radius: int
) -> pd.DataFrame:
    """단일 변수 그룹 3단계 보간"""
    
    # 1단계: 실측 데이터 결합 (2011-2021)
    source_data = pd.read_parquet(source_path)
    source_data['month'] = pd.to_datetime(source_data['time']).dt.month
    
    result = data.merge(
        source_data[['grid_id', 'time'] + variables],
        left_on=['grid_id', 'date'],
        right_on=['grid_id', 'time'],
        how='left'
    )
    
    # 2단계: Climatology 계산 및 적용
    climatology = (
        source_data.groupby(['grid_id', 'month'])[variables]
        .mean()
        .reset_index()
        .rename(columns={v: f"{v}_clim" for v in variables})
    )
    
    result = result.merge(climatology, on=['grid_id', 'month'], how='left')
    
    for var in variables:
        result[f"{var}_filled"] = result[var].fillna(result[f"{var}_clim"])
    
    # 3단계: IDW 공간 보간
    for var in variables:
        result = _idw_interpolation(
            result, f"{var}_filled", max_radius
        )
    
    # 정리
    cols_to_drop = variables + [f"{v}_clim" for v in variables]
    result = result.drop(columns=cols_to_drop, errors='ignore')
    
    return result

def _idw_interpolation(
    df: pd.DataFrame, 
    target_col: str, 
    max_radius: int
) -> pd.DataFrame:
    """IDW (역거리 가중) 공간 보간"""
    
    missing_idx = df[df[target_col].isna()].index
    
    for radius in range(1, max_radius + 1, 2):
        if len(missing_idx) == 0:
            break
            
        logger.info(f"{target_col} IDW 반경 {radius}: {len(missing_idx)} 결측 처리 중")
        
        newly_filled = []
        
        for idx in missing_idx:
            grid_id = df.loc[idx, 'grid_id']
            month = df.loc[idx, 'month']
            
            # 이웃 격자 찾기
            neighbors = get_neighbor_grids(grid_id, radius)
            
            numerator = 0
            denominator = 0
            
            for neighbor_grid in neighbors:
                if neighbor_grid == grid_id:
                    continue
                    
                # 같은 월의 이웃 데이터 찾기
                neighbor_data = df[
                    (df['grid_id'] == neighbor_grid) & 
                    (df['month'] == month) & 
                    (df[target_col].notna())
                ]
                
                if not neighbor_data.empty:
                    # Manhattan distance 계산
                    dist = _calculate_manhattan_distance(grid_id, neighbor_grid)
                    if dist > 0:
                        weight = 1.0 / dist
                        value = neighbor_data[target_col].iloc[0]
                        numerator += weight * value
                        denominator += weight
            
            if denominator > 0:
                df.loc[idx, target_col] = numerator / denominator
                newly_filled.append(idx)
        
        # 남은 결측 재계산
        missing_idx = df[df[target_col].isna()].index
        logger.info(f"반경 {radius} 완료: {len(newly_filled)} 채움, {len(missing_idx)} 남음")
    
    return df
```

---

## 설정 파일 구조

### config.py 구현 가이드
```python
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DataPaths:
    """데이터 경로 설정"""
    base_dir: Path
    
    # Step 1: 기본 데이터
    weather: Path
    af_flag: Path
    
    # Steps 2-4: 정적 변수들
    population: Path
    road_density: Path
    orography: Path
    urban_frac: Path
    vegetation: Path
    
    # Step 5: 동적 변수
    landcover: Path
    
    # Steps 6-7: 연료 습도
    lfmc: Path
    dfmc: Path
    
    # 공간 경계
    korea_boundary: Path
    
    @classmethod
    def from_base_dir(cls, base_dir: str):
        """기본 디렉토리에서 모든 경로 생성"""
        base = Path(base_dir)
        return cls(
            base_dir=base,
            weather=base / "processed_data/era5_daily_combined_200001_202412.csv",
            af_flag=base / "data/af_flag/af_flag_land.csv",
            population=base / "data/population_density/combined_population_density_2000_2020.parquet",
            road_density=base / "data/road_density/road_density_0.1deg.parquet",
            orography=base / "data/Orography/orography_korea_cleaned.csv",
            urban_frac=base / "data/Urban Frac/urban_frac_korea_cleaned.csv", 
            vegetation=base / "data/Vegetation/vegetation_type_korea.csv",
            landcover=base / "data/landcover/landcover_type1_korea_2001_2023.parquet",
            lfmc=base / "data/LFMC_combined/LFMC_combined_2011_2021.parquet",
            dfmc=base / "data/DFMC_combined/DFMC_combined_2011_2021.parquet",
            korea_boundary=base / "data/shapefiles/gadm41_KOR_0.json"
        )

@dataclass  
class IntegrationConfig:
    """통합 매개변수 설정"""
    
    # 격자 시스템
    grid_resolution: float = 0.1
    
    # 한국 영역 경계
    korea_bounds: Dict[str, float] = None
    
    # 산림 코드
    forest_codes: List[int] = None
    
    # IDW 보간
    idw_max_radius: int = 21
    
    # 시간 범위
    start_date: str = "2000-11-02"
    end_date: str = "2024-12-31"
    
    def __post_init__(self):
        if self.korea_bounds is None:
            self.korea_bounds = {
                'min_lat': 33.0, 'max_lat': 39.0,
                'min_lon': 124.0, 'max_lon': 132.0
            }
        
        if self.forest_codes is None:
            self.forest_codes = [1, 2, 3, 4, 5]

# 환경별 설정
def get_config() -> tuple[DataPaths, IntegrationConfig]:
    """현재 환경에 맞는 설정 반환"""
    
    # 환경 감지
    if os.name == 'nt':  # Windows
        base_dir = r"C:\Users\USER\Desktop\my_git\pof-model-korea"
    else:  # macOS/Linux
        base_dir = "/Users/lord_jubin/Desktop/my_git/pof-model-korea"
    
    paths = DataPaths.from_base_dir(base_dir)
    config = IntegrationConfig()
    
    return paths, config
```

---

## 구현 우선순위

### Phase 1: 핵심 유틸리티 (현재 진행 중)
- [x] Grid System (`grid_system.py`) 
- [ ] **사용자 작업**: `validate_grid_consistency` 함수 구현
- [ ] Spatial Filter (`spatial_filter.py`)
- [ ] Validators (`validators.py`)

### Phase 2: 통합 모듈
1. `weather_fire.py` - 가장 단순한 INNER JOIN
2. `static_vars.py` - 반복적인 LEFT JOIN 패턴  
3. `landcover.py` - 복잡한 전처리 + 필터링
4. `fuel_moisture.py` - 가장 복잡한 3단계 보간

### Phase 3: 파이프라인
1. `config.py` - 설정 관리
2. `pipeline.py` - 전체 오케스트레이션

---

## 테스트 전략

### 단위 테스트
```python
def test_weather_fire_integration():
    # 샘플 데이터로 테스트
    result = integrate_weather_fire(sample_weather_path, sample_af_path)
    assert len(result) > 0
    assert 'af_flag' in result.columns
    assert result['grid_id'].notna().all()

def test_idw_interpolation():
    # 인위적인 결측 데이터로 보간 정확도 테스트
    pass
```

### 통합 테스트
```python  
def test_full_pipeline():
    # 노트북 결과와 비교
    notebook_result = pd.read_parquet("expected_result.parquet")
    pipeline_result = DataIntegrationPipeline().run()
    
    pd.testing.assert_frame_equal(
        notebook_result.sort_values(['grid_id', 'date']).reset_index(drop=True),
        pipeline_result.sort_values(['grid_id', 'date']).reset_index(drop=True)
    )
```

이제 이 가이드를 기반으로 각 모듈을 차례로 구현할 수 있습니다.