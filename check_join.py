import pandas as pd
import os
import numpy as np

def check_data_compatibility():
    # 디렉토리 경로 설정
    weather_dir = 'processed_data/era5_daily'
    fire_file = 'data/af_flag/af_flag_full_combined.csv'
    
    # 열 정의
    weather_cols = ['acq_date', 'grid_id', 'latitude', 'longitude']
    fire_cols = ['date', 'grid_id']
    
    print("ERA5 날씨 데이터와 산불 데이터 조인 가능성 확인\n")
    
    # 산불 데이터 샘플 로드 (샘플 크기 줄임)
    print("산불 데이터 로드 중...")
    fire_df = pd.read_csv(fire_file, usecols=fire_cols, nrows=5000)
    
    print(f"산불 데이터 grid_id 범위: {fire_df.grid_id.min()} - {fire_df.grid_id.max()}")
    print(f"산불 데이터 날짜 범위: {fire_df.date.min()} - {fire_df.date.max()}")
    print(f"산불 데이터 고유 grid_id 수: {fire_df.grid_id.nunique()}")
    
    # 고유 grid_id 추출
    fire_grids = set(fire_df['grid_id'].unique())
    
    # 날씨 데이터 파일 확인
    weather_files = [f for f in os.listdir(weather_dir) if f.endswith('.csv')]
    print(f"\n날씨 데이터 파일 수: {len(weather_files)}")
    
    # 여러 파일에서 고유 grid_id 수집 (파일 수 줄임)
    all_weather_grids = set()
    lat_lon_map = {}  # grid_id -> (lat, lon) 매핑
    
    # 첫 번째 파일만 사용
    weather_file = weather_files[0]
    sample_file = os.path.join(weather_dir, weather_file)
    print(f"\n날씨 파일 '{weather_file}' 처리 중...")
    
    weather_df = pd.read_csv(sample_file, usecols=weather_cols, nrows=5000)
    
    # grid_id 범위 확인
    print(f"  grid_id 범위: {weather_df.grid_id.min()} - {weather_df.grid_id.max()}")
    print(f"  날짜 범위: {weather_df.acq_date.min()} - {weather_df.acq_date.max()}")
    print(f"  고유 grid_id 수: {weather_df.grid_id.nunique()}")
    
    # 위도/경도 간격 확인
    lat_values = sorted(weather_df['latitude'].unique())
    lon_values = sorted(weather_df['longitude'].unique())
    
    if len(lat_values) > 1:
        lat_diffs = [round(lat_values[i+1] - lat_values[i], 2) for i in range(len(lat_values)-1)]
        print(f"\n  위도 간격 샘플 (첫 5개): {lat_diffs[:5]}")
    
    if len(lon_values) > 1:
        lon_diffs = [round(lon_values[i+1] - lon_values[i], 2) for i in range(len(lon_values)-1)]
        print(f"  경도 간격 샘플 (첫 5개): {lon_diffs[:5]}")
    
    # grid_id를 수집
    all_weather_grids = set(weather_df['grid_id'].unique())
    
    # grid_id에 대응하는 위도/경도 정보 저장
    for _, row in weather_df[['grid_id', 'latitude', 'longitude']].drop_duplicates().head(10).iterrows():
        lat_lon_map[row['grid_id']] = (row['latitude'], row['longitude'])
    
    # 실제 grid_id와 좌표 관계 분석
    sample_grid_coords = list(lat_lon_map.items())[:5]
    print("\n샘플 grid_id와 좌표 관계:")
    for grid_id, (lat, lon) in sample_grid_coords:
        print(f"  grid_id: {grid_id}, 좌표: ({lat}, {lon})")
    
    # grid_id 생성 규칙 추론
    if sample_grid_coords:
        print("\ngrid_id 생성 규칙 추론:")
        for grid_id, (lat, lon) in sample_grid_coords:
            # 추론 시도
            lat_int = int(lat * 10)
            lon_int = int(lon * 10)
            computed_id = (lat_int * 10000) + lon_int
            diff = grid_id - computed_id
            print(f"  grid_id: {grid_id}, 좌표: ({lat}, {lon}), 계산: {lat_int}*10000 + {lon_int} = {computed_id}, 차이: {diff}")
    
    # grid_id 매칭 분석
    print("\n==== 데이터셋 간 grid_id 매칭 분석 ====")
    print(f"산불 데이터 고유 grid_id 수: {len(fire_grids)}")
    print(f"날씨 데이터 고유 grid_id 수 (샘플): {len(all_weather_grids)}")
    
    # 공통 grid_id 확인
    common_grids = fire_grids.intersection(all_weather_grids)
    print(f"공통 grid_id 수: {len(common_grids)} ({(len(common_grids) / len(fire_grids) * 100):.2f}% 커버리지)")
    
    # 산불 데이터에만 있는 grid_id
    only_fire_grids = fire_grids - all_weather_grids
    print(f"산불 데이터에만 있는 grid_id 수: {len(only_fire_grids)}")
    if only_fire_grids:
        print(f"예시: {sorted(list(only_fire_grids))[:5]}")
    
    # 올바른 grid_id -> (lat, lon) 변환 함수
    def grid_id_to_lat_lon(grid_id):
        lat_int = grid_id // 10000
        lon_int = grid_id % 10000
        lat = lat_int / 10
        lon = lon_int / 10
        return (lat, lon)
    
    # 누락된 grid_id 분석
    if only_fire_grids:
        print("\n==== 누락된 grid_id 상세 분석 ====")
        
        # 누락된 grid_id의 위도/경도 추정 (샘플만)
        sample_missing = sorted(list(only_fire_grids))[:5]
        missing_coords = []
        
        for grid_id in sample_missing:
            lat, lon = grid_id_to_lat_lon(grid_id)
            missing_coords.append((grid_id, lat, lon))
        
        print(f"누락된 grid_id의 위도/경도 예시:")
        for grid_id, lat, lon in missing_coords:
            print(f"  grid_id {grid_id}: 추정 좌표 ({lat}, {lon})")
        
        # 경계 지역 분석
        if lat_lon_map:
            min_lat = min(lat for lat, _ in lat_lon_map.values())
            max_lat = max(lat for lat, _ in lat_lon_map.values())
            min_lon = min(lon for _, lon in lat_lon_map.values())
            max_lon = max(lon for _, lon in lat_lon_map.values())
            
            print(f"\n날씨 데이터 위도 범위: {min_lat} - {max_lat}")
            print(f"날씨 데이터 경도 범위: {min_lon} - {max_lon}")
            
            # 누락된 grid_id가 경계 밖에 있는지 확인 (샘플만)
            out_of_bounds = 0
            for grid_id, lat, lon in missing_coords:
                if lat < min_lat or lat > max_lat or lon < min_lon or lon > max_lon:
                    out_of_bounds += 1
                    print(f"  grid_id {grid_id}: 경계 밖 ({lat}, {lon})")
        
        # 끝자리 분석 (패턴 확인)
        last_digits = [grid_id % 10 for grid_id in sample_missing]
        print(f"\n샘플 grid_id 끝자리: {last_digits}")
        
        # 간격 분석
        if len(sample_missing) > 1:
            diffs = [sample_missing[i+1] - sample_missing[i] for i in range(len(sample_missing)-1)]
            print(f"샘플 grid_id 간격: {diffs}")
    
    # 결론
    print("\n==== 결론 ====")
    if len(common_grids) / len(fire_grids) > 0.9:
        print("✅ 두 데이터셋은 grid_id 기준으로 충분히 조인 가능합니다 (90% 이상 매칭)")
    elif len(common_grids) / len(fire_grids) > 0.5:
        print("⚠️ 두 데이터셋은 부분적으로 조인 가능합니다 (50% 이상 매칭)")
    else:
        print("❌ 두 데이터셋의 grid_id 매칭률이 낮습니다 (50% 미만)")
    
    # 0.1도 보간 확인
    if 0.1 in lat_diffs[:5] or 0.1 in lon_diffs[:5]:
        print("✅ 날씨 데이터는 0.1도 단위로 올바르게 보간되었습니다.")
    else:
        print("❌ 날씨 데이터가 0.1도 단위로 보간되지 않았습니다.")
    
    # 누락 원인 추정
    if only_fire_grids:
        print("\n==== 누락 원인 추정 ====")
        
        if all(digit == 9 for digit in last_digits):
            print("✅ 모든 샘플 누락 grid_id가 '9'로 끝나는 특이한 패턴이 있습니다.")
            print("   이는 특정 좌표 변환 규칙 또는 필터링 조건 때문일 수 있습니다.")
        
        if out_of_bounds == len(missing_coords):
            print("✅ 샘플로 확인한 모든 누락 grid_id는 날씨 데이터의 지리적 경계 밖에 위치합니다.")
            print("   이는 데이터 처리 과정에서 유효 지역을 필터링하는 과정에서 이러한 grid_id가 제외되었을 가능성이 높습니다.")
        
        print("\n누락된 grid_id의 원본 데이터를 확인하고, 데이터 처리 과정에서 특정 조건이나 필터링이 적용되었는지 검토하세요.")

if __name__ == "__main__":
    check_data_compatibility() 