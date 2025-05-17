# Global Probability-of-Fire (PoF) Replication for Korea

> 참고 논문: _Global data-driven prediction of fire activity_ (ECMWF, 2024)

> https://www.nature.com/articles/s41467-025-58097-7

---

## 1. 데이터

| 구분          | 상세 변수·지표                                | 시간 해상도         | 공간 해상도\* | 1차 출처·링크                                                | 무료 계정 / API | 수집·갱신 주기          | 이용 가능 기간          | 수집 방법 요약                          | 한국 적용 | 근거                                       |
| ------------- | --------------------------------------------- | ------------------- | ------------- | ------------------------------------------------------------ | --------------- | ----------------------- | ----------------------- | --------------------------------------- | :-------: | ------------------------------------------ |
| Weather       | 2 m T, 2 m Td, 10 m Wind, Total Precip        | 일별 (1 h→일)       | 0.1° (~9 km)  | ERA5-Land [CDS](https://doi.org/10.24381/cds.e2161bac)       | 필요            | 매일 (T-3 h 지연)       | 1950-01-01 → 현재       | `cdsapi` 요청 시 `area=[38,124,33,132]` |     o     | ERA5-Land는 전 지구 등격자 → 우리나라 포함 |
| Ignition-dyn  | Lightning flash density                       | 일별                | 0.1°          | ERA5 단층 변수                                               | 필요            | 매일                    | 2000-01-01 → 현재       | `cdsapi` 에서 변수만 변경               |     o     | ERA5 전 지구 자료, 동일 영역 지정 가능     |
| Fuel-model    | Fuel Load 4종, Fuel Moist 3종                 | 일별                | 0.1°          | Global Fuel v1.2 [Zenodo](https://doi.org/10.24381/378d1497) | 불필요          | 1 일 NRT                | 2014-01-01 → 2023-12-31 | ZIP → NetCDF                            |     o     | "daily global 9 km" 명시                   |
| Fuel-sat      | L-VOD                                         | 월별                | 25 km→0.1°    | SMOS-IC (INRAE)                                              | 불필요          | 월 NRT                  | 2010-01-01 → 2021-07-31 | `wget`+`cdo remap`                      |     o     | SMOS 문서: daily global retrievals         |
|               | LAI-low, LAI-high                             | 10-일               | 300 m→0.1°    | Copernicus LAI v2                                            | 필요            | 10-일 NRT               | 1999-04-01 → 현재       | CDS API → 리샘플                        |     o     | CGLS LAI: global 300 m 제공                |
| Ignition-stat | Vegetation Type, Urban Frac, Orography        | 고정                | 0.1°          | ECMWF ECLand Static                                          | 필요            | –                       | 2023 스냅숏             | CDS "ancillary" 다운로드                |     o     | 전 지구 정적 마스크                        |
|               | Population Density                            | 5 년                | 2.5′→0.1°     | SEDAC GPW v4                                                 | 필요            | 5 년                    | 2000·05·10·15·20        | GeoTIFF → `gdalwarp`                    |     o     | GPW: Gridded Population of the World       |
|               | Road Density                                  | 고정                | 5′→0.1°       | CIESIN GRiD-2018                                             | 불필요          | –                       | 2018 스냅숏             | GeoTIFF 직접 다운로드                   |     o     | GRiD: global road density 래스터           |
| Target        | MODIS Active Fire (MCD14 v6.1, low-conf 제외) | 궤도 4×/일 → 일집계 | 1 km→0.1°     | NASA FIRMS [SFTP](https://firms.modaps.eosdis.nasa.gov/)     | 필요            | 10 분 NRT / 월 아카이브 | 2000-02-24 → 현재       | REST bbox(`124,33,132,38`) 또는 SFTP    |     o     | FIRMS API: bbox 기반 글로벌 다운로드 지원  |

\* 모든 입력은 학습 전에 0.1°(~9 km) 격자로 리그리딩합니다.  
ERA5 확정값은 약 3 개월 지연 후 공개됩니다.

---

## 2 · 데이터셋 스키마

| 열 이름                                                                           | dtype            | 설명                                      |
| --------------------------------------------------------------------------------- | ---------------- | ----------------------------------------- |
| `date`                                                                            | `datetime64[ns]` | 샘플 날짜 (YYYY-MM-DD, UTC)               |
| `grid_id` †                                                                       | `int32`          | 0.1° 격자 인덱스 (또는 `lat`,`lon` float) |
| **Weather** `t2m`,`td2m`,`wind10m`,`precip`                                       | `float32`        | ERA5-Land 일평균(또는 일합계)             |
| **Lightning** `lightning`                                                         | `float32`        | 일 누계 낙뢰 밀도                         |
| **Fuel-Load** `live_leaf_load`,`live_wood_load`,`dead_leaf_load`,`dead_wood_load` | `float32`        | kg m⁻²                                    |
| **Fuel-Moist** `live_fuel_moist`,`dead_foliage_moist`,`dead_wood_moist`           | `float32`        | %                                         |
| **Veg-Sat** `vod_L`,`lai_low`,`lai_high`                                          | `float32`        | 보간된 월/10-일 자료                      |
| **Static** `veg_type`(`int8`), `orography`(`int16`), `urban_frac`(`float32`)      | -                | 정적 값                                   |
| **Human** `pop_dens`,`road_dens`                                                  | `float32`        | 인구/도로 밀도                            |
| **타깃** `af_flag`                                                                | `uint8`          | MODIS AF ≥ 1 → `1`, else `0`              |

† 글로벌 전체는 약 1 800 × 3 600 셀(극지 제외), 한반도 범위는 ≈ 480 셀.

- **한 행 = "하루 × 격자셀"**  
  예) 2019-05-12, grid #123456, 19 features, `af_flag=0/1`.
- **클래스 불균형** : 양성 ≈ 0.3 % → `scale_pos_weight = N_neg / N_pos` 로 보정.
- 대용량 학습용으로 Parquet 파티션(`year=`) & 음성 1/20 다운샘플을 권장.

---

## 3. 단계별 모델 변수 확장

| 단계 | 사용 변수         | 추천 학습 기간 | 성능 향상\* | 논문 근거             |
| ---- | ----------------- | -------------- | ----------- | --------------------- |
| S1   | Weather 4종       | 2003-현재      | 기준        | Fig 2 – Weather only  |
| S2   | + Lightning       | 2003-현재      | +3 pp       | Fig 2 개선            |
| S3   | + Fuel Moist 3종  | 2014-현재      | +15 pp      | Fuel only 손실 –15 pp |
| S4   | + Fuel Load 4종   | 2014-현재      | +5 pp       | Ablation 결과         |
| S5   | + Static Veg·Topo | 2014-현재      | +2 pp       | Fig 2 두 요소 조합    |
| S6   | + Population·Road | 2014-현재      | +3 pp       | 사람 발화 보정        |
| S7   | + VOD·LAI 3종     | 2014-2021      | +2 pp       | Full 대비 −3 ~ 7 pp   |
| S8   | Full 19종         | 2014-2021      | 최고        | 논문 최종 모델        |

\* pp = percentage-point, 논문 Fig 2·Extended Data 4 참조.

---

## 4. 데이터·모델 범위 확장 전략

1. Weather-only 단계는 1950-현재 전체 기간 활용 가능
2. Fuel 포함 이후는 최소 2014-현재
3. VOD·LAI는 2021-07 이후 결측 → 최근값 유지 or 대체 지표 사용
4. 한국 특화 파인튠: KMA LDAPS/KIM 예보 + 국립산림과학원 연료지도 + SGIS 인구격자 적용

---

## 5. 평가 지표 (논문 동일)

- Brier score, LogLoss, ROC-AUC, Pearson r, Calibration Error
- 얼리 스톱 10 round, 불균형 보정 `scale_pos_weight`

---

### 참고 링크

- 논문 Table 1 (19 drivers), Fig 2 (Ablation), Fig 3 (Skill), Methods § Training dataset
- Code capsule : `doi:10.24433/CO.8570224.v1` — 데이터 전처리 & XGBoost 파이프라인

---

## 6. 데이터 수집 및 전처리 이슈 해결

### 6.1 발견된 문제점

- **지리적 범위 불일치**: 타겟 데이터(af_flag_korea.csv)와 날씨 데이터의 grid_id 범위가 일치하지 않았습니다.

  - 타겟 데이터 범위: 4,488,667 - 4,632,683
  - 날씨 데이터 범위: 4,431,090 - 4,611,086
  - 산불 발생(af_flag=1) 데이터가 주로 날씨 데이터에서 누락된 영역에 집중되어 있었습니다.

- **조인 방식 문제**: 날씨 데이터 기준의 left join으로 인해 많은 af_flag=1 데이터가 누락되었습니다.
  - 타겟 데이터에는 296개의 산불 발생 데이터가 존재
  - 조인 후 최종 데이터에는 산불 발생 데이터가 0개 (전부 누락)

### 6.2 해결 방안

1. **수집 과정 수정**:

   - BBOX 범위를 확장: `[38, 124, 33, 132]` → `[39, 124, 33, 132]` (북위 39도까지)
   - 이를 통해 af_flag 데이터의 전체 범위(최대 38.6도)를 포함하도록 개선

2. **전처리 과정 수정**:
   - 병합 방식 변경: `how='left'` → `how='right'`로 변경하여 모든 타겟 데이터 보존
   - 결측치 처리: 날씨 데이터가 없는 지점의 기상 변수를 적절히 대체
     - 온도, 습도, 바람 변수: 해당 변수의 평균값으로 대체
     - 강수량: 0으로 대체

### 6.3 검증 필요 사항

- **데이터 완전성**: 최종 데이터셋에 af_flag=1인 데이터가 원본의 296개 모두 포함되었는지 확인
- **격자 간 공간적 연속성**: 실제 측정된 값과 결측치 대체값 사이에 급격한 차이가 없는지 확인
- **모델 성능 영향**: 결측치 대체로 인한 모델 예측 성능 영향 평가 필요

이러한 개선을 통해 산불 위험 예측 모델링에 필요한 완전한 데이터셋을 구축할 수 있게 되었습니다.

---

## 7. 파이프라인 최적화

### 7.1 개선 전 파이프라인 구조

기존 파이프라인은 다음 4단계로 구성되어 있었습니다:

1. 산불 데이터 전처리 (`process_af_flag.py`)
2. 날씨 데이터 수집 (`collect_weather.py`)
3. 날씨 데이터 전처리 및 산불 데이터와 결합 (`process_weather.py`) - `weather_data.csv` 생성
4. 풍속 변수(wind10m) 추가 처리 (`add_wind10m.py`) - `weather_data_with_wind.csv` 생성

이 구조에서는 두 번의 중복 처리가 발생하며, 중간 파일을 추가로 저장하는 문제가 있었습니다.

### 7.2 최적화된 파이프라인 구조

파이프라인을 다음과 같이 개선하여 효율성을 높였습니다:

1. 산불 데이터 전처리 (동일)
2. 날씨 데이터 수집 (동일)
3. 날씨 전처리와 풍속 계산, 산불 데이터 결합을 하나의 단계로 통합
   - `process_weather.py` 내에서 10u/10v → wind10m 계산 로직 통합
   - 바로 최종 파일 `weather_data_with_wind.csv` 생성

### 7.3 최적화 효과

- **수행 시간 단축**: 중복 처리 제거로 전체 파이프라인 실행 시간 단축
- **디스크 I/O 감소**: 중간 파일 생성 및 로드 과정 제거로 디스크 부하 감소
- **스토리지 효율화**: 중복 데이터 저장 방지로 디스크 공간 절약
- **코드 간소화**: 불필요한 스크립트 제거 및 로직 통합으로 코드베이스 간소화

### 7.4 실행 방법

최적화된 파이프라인은 다음 명령으로 실행할 수 있습니다:

```
run_optimized_pipeline.bat
```

이 배치 파일은 기존 기능을 모두 제공하면서 더 효율적으로 실행됩니다.
