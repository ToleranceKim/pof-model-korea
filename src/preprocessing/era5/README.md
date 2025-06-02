# ERA5 기상 데이터 전처리 가이드

이 디렉토리는 ERA5 기상 데이터를 처리하는 스크립트를 포함하고 있습니다. 산불 발생 예측 모델(PoF)의 입력 데이터로 사용됩니다.

## 스크립트 요약

| 스크립트 이름     | 주요 기능                               |
| ----------------- | --------------------------------------- |
| `process_era5.py` | ERA5 파일을 처리하는 핵심 함수들을 포함 |

## 데이터 처리 과정

ERA5 기상 데이터 처리는 다음 단계로 이루어집니다:

1. **원본 데이터 로드**: `.nc` 파일에서 데이터 추출
2. **선형 보간**: 0.25° 격자에서 0.1° 격자로 보간
3. **시간 집계**: 시간별 데이터를 일별로 집계
4. **풍속 계산**: 10u, 10v 성분을 이용해 풍속(wind10m) 계산
5. **데이터 저장 및 결합**: 결과를 저장하고 여러 파일을 하나로 결합

## 격자 범위 개선사항

기존 처리 로직에서는 부동소수점 연산 특성과 병렬 처리로 인해 격자 범위(33°N-39°N, 124°E-132°E)를 완전히 처리하지 못했습니다. 특히 경도 132.0° 부근 격자점이 누락되어 총 4,740개 격자점만 생성되었습니다.

현재 스크립트에서는 다음 방법으로 해결했습니다:

- Path 객체 호환성 문제 해결
- xarray 내장 보간 기능 사용
- 병렬 처리 대신 순차 처리 방식 채택
- 일관된 격자 생성 및 처리

이를 통해 의도된 격자 범위를 정확히 처리하며, af_flag 데이터와 완벽하게 일치하는 4,800개 grid_id를 생성합니다.

## 실행 방법

```bash
python src/preprocessing/era5/process_era5.py --input_dir data/raw --intermediate_dir processed_data/era5 --output_dir processed_data/era5_daily --viz_dir processed_data/viz --start_date 200501 --end_date 201812 --interpolate True --combine True
```

**주요 인자:**

- `--input_dir`: 원본 ERA5 파일이 있는 디렉토리 (기본값: data/raw)
- `--intermediate_dir`: 보간된 중간 파일을 저장할 디렉토리 (기본값: processed_data/era5)
- `--output_dir`: 일별 집계 파일을 저장할 디렉토리 (기본값: processed_data/era5_daily)
- `--viz_dir`: 시각화 데이터를 저장할 디렉토리 (기본값: processed_data/viz)
- `--start_date`: 처리 시작 날짜 (YYYYMM 형식)
- `--end_date`: 처리 종료 날짜 (YYYYMM 형식)
- `--interpolate`: 0.1도 격자로 보간할지 여부 (기본값: True)
- `--combine`: 모든 파일을 결합할지 여부 (기본값: True)

### 사용 예시

```bash
# 2005-2018년 데이터 처리
python src/preprocessing/era5/process_era5.py --input_dir data/raw --intermediate_dir processed_data/era5 --output_dir processed_data/era5_daily --viz_dir processed_data/viz --start_date 200501 --end_date 201812 --interpolate True --combine True

# 특정 월 데이터만 처리
python src/preprocessing/era5/process_era5.py --input_dir data/raw --intermediate_dir processed_data/era5 --output_dir processed_data/era5_daily --start_date 202001 --end_date 202012

# 보간 없이 원본 해상도 사용
python src/preprocessing/era5/process_era5.py --input_dir data/raw --output_dir processed_data/era5_daily_orig --interpolate False
```

## 출력 파일 형식

처리된 파일은 다음과 같은 형식으로 저장됩니다:

1. **보간된 중간 파일**: `processed_data/era5/era5_korea_YYYYMM_0.1deg.nc`
2. **월별 일일 데이터**: `processed_data/era5_daily/era5_daily_YYYYMM.csv` 및 `.parquet`
3. **전체 결합 파일**: `processed_data/era5_daily_combined_YYYYMM_YYYYMM.parquet` 및 `.csv`
4. **시각화용 파일**: `processed_data/viz/era5_daily_all.parquet`

결과 데이터에는 다음 열이 포함됩니다:

- `acq_date`: 데이터 날짜
- `grid_id`: 격자 ID (0.1도 격자 기준)
- `latitude`: 위도 (북위 33°-39°)
- `longitude`: 경도 (동경 124°-132°)
- `t2m`: 2m 기온 (일평균, K)
- `td2m`: 2m 이슬점 온도 (일평균, K)
- `10u`: 10m U 풍속 성분 (일평균, m/s)
- `10v`: 10m V 풍속 성분 (일평균, m/s)
- `wind10m`: 10m 풍속 (일평균, m/s, U/V 성분에서 계산)
- `tp`: 총 강수량 (일누적, mm)

## 주의사항

1. 원본 데이터가 `.nc` 확장자를 가지고 있더라도 내부적으로는 ZIP 파일일 수 있습니다.
2. 파일 처리 중 발생하는 오류는 로그 파일에 기록됩니다.
3. 이미 처리된 파일은 건너뛰므로, 필요시 출력 디렉토리의 파일을 삭제하여 재처리할 수 있습니다.
4. 대용량 파일을 처리하므로 충분한 메모리(최소 8GB 이상 권장)가 필요합니다.
