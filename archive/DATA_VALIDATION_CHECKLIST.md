# POF-Korea 데이터 검증 체크리스트

## 개요
본 문서는 POF-Korea 프로젝트의 데이터 통합 과정에서 수행해야 할 검증 항목들을 체계적으로 정리한 체크리스트입니다.

---

## ✅ 1. 데이터 수집 단계 검증

### 1.1 ERA5 기상 데이터
- [ ] **시간 범위**: 2000-01-01 ~ 2024-12-31 완전성
- [ ] **공간 범위**: 33°N-39°N, 124°E-132°E 포함 여부
- [ ] **변수 완전성**: t2m, td2m, 10u, 10v, tp 모두 존재
- [ ] **파일 무결성**: NetCDF 파일 손상 여부
- [ ] **단위 확인**:
  - 온도: Kelvin
  - 강수: meters
  - 바람: m/s

### 1.2 MODIS/VIIRS 화재 데이터
- [ ] **시작 날짜**: 2000-11-02 이후 (MODIS 운영 시작)
- [ ] **좌표 시스템**: WGS84 (EPSG:4326)
- [ ] **Confidence 필터**: ≥ 30% (nominal 이상)
- [ ] **중복 제거**: 동일 위치/시간 중복 화재 점 제거
- [ ] **Type 필터**: vegetation fire만 포함 (0, 1, 2)

### 1.3 정적/준정적 데이터
- [ ] **Population**: GPW v4 2020년 데이터 존재
- [ ] **Road Density**: OSM 최신 데이터
- [ ] **Landcover**: MODIS MCD12Q1 2001-2023 연속성
- [ ] **DEM**: SRTM 30m → 0.1° 리샘플링 완료

---

## ✅ 2. 전처리 단계 검증

### 2.1 Grid ID 변환
```python
# 검증 코드
def verify_grid_conversion():
    test_cases = [
        (37.5, 127.0, 4593070),  # 서울
        (35.1, 129.0, 4556692),  # 부산
        (33.5, 126.5, 4525065),  # 제주
    ]
    for lat, lon, expected_grid in test_cases:
        calculated = latlon2grid(lat, lon)
        assert calculated == expected_grid, f"Grid 변환 오류: {lat},{lon}"
```
- [ ] Grid ID 계산 정확성
- [ ] 역변환 일관성 (grid2latlon)
- [ ] 경계값 처리 (33.0, 39.0, 124.0, 132.0)

### 2.2 시간 정합성
- [ ] **날짜 형식 통일**: YYYY-MM-DD
- [ ] **시간대**: UTC 기준 통일
- [ ] **일 집계**: 00:00-23:59 UTC
- [ ] **윤년 처리**: 2월 29일 데이터 확인

### 2.3 공간 리샘플링
- [ ] **해상도 일치**: 모든 데이터 0.1° × 0.1°
- [ ] **보간 방법**:
  - 연속 변수: Bilinear
  - 범주 변수: Nearest neighbor
- [ ] **경계 처리**: Edge artifacts 확인

---

## ✅ 3. 데이터 통합 단계 검증

### 3.1 Weather + AF_Flag 결합
- [ ] **Join 키**: (grid_id, date) 일치
- [ ] **Join 방식**: INNER JOIN
- [ ] **결과 행 수**: 4,247,637 rows (±1%)
- [ ] **중복 확인**:
  ```python
  assert not df[['grid_id','date']].duplicated().any()
  ```

### 3.2 한국 영역 필터링
- [ ] **방법**: 지번 데이터 기반 (1,007개 격자)
- [ ] **검증**:
  - [ ] 서울 포함 (grid_id: 4593070)
  - [ ] 독도 포함 확인
  - [ ] 이어도 제외 확인
- [ ] **좌표 범위 재확인**

### 3.3 정적 변수 결합
- [ ] **Population 결합**:
  - [ ] 5년 단위 버킷팅 (2000, 2005, ..., 2020)
  - [ ] 2021+ → 2020 매핑
  - [ ] 결측률 < 5%

- [ ] **Road Density 결합**:
  - [ ] 단위: km/km²
  - [ ] 범위: 0-50
  - [ ] 도시 지역 높은 값 확인

- [ ] **Landcover 결합**:
  - [ ] 연도별 매칭
  - [ ] 2024년: 2023년 값 사용
  - [ ] 2000년: 2001년 값 사용 (11/2 이후)

### 3.4 산림 필터링
- [ ] **FOREST_CODES**: [1, 2, 3, 4, 5] 적용
- [ ] **결과 격자 수**: 581개
- [ ] **화재 비율 증가**: 0.4% → 0.8%

### 3.5 LFMC/DFMC 보간
- [ ] **보간 우선순위**:
  1. [ ] 실측값 (2011-2021)
  2. [ ] Climatology
  3. [ ] IDW (반경 21)
- [ ] **최종 결측률**: < 1%
- [ ] **값 범위 검증**:
  - LFMC: 50-200%
  - DFMC: 2-40%

---

## ✅ 4. 데이터 품질 검증

### 4.1 물리적 타당성
```python
# 온도 범위
assert (df.t2m > 200).all() and (df.t2m < 350).all()
assert df.td2m <= df.t2m  # 이슬점 ≤ 기온

# 강수량
assert (df.tp >= 0).all() and (df.tp < 1).all()

# 풍속
assert (df.wind10m >= 0).all() and (df.wind10m < 50).all()

# 습도
assert (df.LFMC > 0).all() and (df.LFMC < 300).all()
```
- [ ] 모든 변수 물리적 범위 내
- [ ] 변수 간 논리적 일관성
- [ ] 이상치 탐지 (3σ rule)

### 4.2 시계열 연속성
- [ ] **일별 데이터 완전성**:
  - 2000-11-02 ~ 2024-12-31
  - 누락 날짜 < 100일
- [ ] **급격한 변화 탐지**:
  ```python
  daily_diff = df.groupby('grid_id')['t2m'].diff()
  assert daily_diff.abs().max() < 30  # 30K 이상 일변화 이상
  ```

### 4.3 공간 일관성
- [ ] **이웃 격자 상관성**:
  ```python
  # 인접 격자 온도 차이 < 10K
  for grid in grids:
      neighbors = get_neighbors(grid)
      temp_diff = abs(df[grid].t2m - df[neighbors].t2m.mean())
      assert temp_diff < 10
  ```
- [ ] **고립 이상치**: 주변과 극단적 차이 점검

### 4.4 클래스 균형
- [ ] **전체 화재 비율**: 0.3-0.5% 범위
- [ ] **산림 화재 비율**: 0.7-1.0% 범위
- [ ] **연도별 추세**: 급격한 변화 없음

---

## ✅ 5. 최종 출력 검증

### 5.1 파일 형식
- [ ] **Parquet 형식**: 압축 효율성
- [ ] **CSV 백업**: 가독성 확보
- [ ] **메타데이터**: 컬럼 설명 파일

### 5.2 데이터 크기
- [ ] **전체 데이터**:
  - 행 수: ~4.2M (weather_af_land)
  - 열 수: 19 features + 1 label + 메타
- [ ] **산림 데이터**:
  - 행 수: ~2.1M
  - 파일 크기: < 1GB

### 5.3 버전 관리
- [ ] **파일명 규칙**:
  ```
  {dataset}_{version}_{YYYYMMDD}.parquet
  예: weather_af_forest_v1.1_20250117.parquet
  ```
- [ ] **변경 이력**: CHANGELOG.md 업데이트
- [ ] **체크섬**: MD5/SHA256 기록

---

## ✅ 6. 자동화 검증 스크립트

### 6.1 통합 검증 실행
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

### 6.2 검증 보고서 생성
```python
# generate_validation_report.py
import pandas as pd
from datetime import datetime

def generate_report(validation_results):
    """검증 결과 HTML 보고서 생성"""

    report = f"""
    <html>
    <head><title>POF-Korea Data Validation Report</title></head>
    <body>
    <h1>데이터 검증 보고서</h1>
    <p>생성일시: {datetime.now()}</p>

    <h2>검증 결과 요약</h2>
    <table border="1">
    <tr><th>검증 항목</th><th>결과</th><th>상세</th></tr>
    """

    for item, result in validation_results.items():
        status = "✅ 통과" if result['passed'] else "❌ 실패"
        report += f"""
        <tr>
            <td>{item}</td>
            <td>{status}</td>
            <td>{result['details']}</td>
        </tr>
        """

    report += """
    </table>
    </body>
    </html>
    """

    with open('validation_report.html', 'w') as f:
        f.write(report)

    return "validation_report.html"
```

---

## 📋 빠른 체크리스트 (Quick Check)

### 필수 확인 사항 (Critical)
- [ ] Grid ID 1,007개 (육지)
- [ ] 날짜 범위: 2000-11-02 ~ 2024-12-31
- [ ] af_flag 비율: 0.3-0.5%
- [ ] 중복 없음
- [ ] 결측 < 5%

### 권장 확인 사항 (Recommended)
- [ ] 물리적 범위 검증
- [ ] 시계열 연속성
- [ ] 공간 일관성
- [ ] 파일 무결성
- [ ] 문서화 완료

---

## 🔄 검증 주기

| 검증 유형 | 주기 | 담당 |
|---------|------|------|
| 데이터 수집 검증 | 매일 | 자동화 |
| 통합 파이프라인 | 주간 | 자동화 |
| 품질 전수 검사 | 월간 | 수동 |
| 모델 입력 검증 | 학습 시 | 자동화 |

---

## 📞 문제 발생 시 대응

1. **데이터 누락**:
   - 1차: 재다운로드 시도
   - 2차: 백업 소스 활용
   - 3차: 보간/대체 처리

2. **품질 이상**:
   - 이상치 플래깅
   - 원인 분석 로그
   - 필터링/보정 적용

3. **통합 실패**:
   - Join 키 재확인
   - 데이터 타입 점검
   - 메모리 최적화

---

*최종 수정: 2025년 1월 17일*
*작성: POF-Korea 프로젝트 팀*