#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import sys
import os

def check_dimensions(weather_file, af_flag_file):
    """
    날씨 데이터와 산불 데이터의 차원 일치 여부를 검증합니다.
    
    Parameters:
    -----------
    weather_file : str
        날씨 데이터 파일 경로
    af_flag_file : str
        산불 플래그 데이터 파일 경로
    
    Returns:
    --------
    bool : 검증 성공 여부
    """
    print(f"\n=== 데이터 차원 일치 검증 시작 ===")
    
    # 파일 존재 확인
    if not os.path.exists(weather_file):
        print(f"오류: 날씨 데이터 파일을 찾을 수 없습니다: {weather_file}")
        return False
    
    if not os.path.exists(af_flag_file):
        print(f"오류: 산불 데이터 파일을 찾을 수 없습니다: {af_flag_file}")
        return False
    
    # 데이터 로드
    try:
        weather_df = pd.read_csv(weather_file, parse_dates=['acq_date'])
        print(f"날씨 데이터 로드 완료: {len(weather_df)} 행")
    except Exception as e:
        print(f"오류: 날씨 데이터 로드 실패: {e}")
        return False
    
    try:
        af_df = pd.read_csv(af_flag_file, parse_dates=['acq_date'])
        print(f"산불 데이터 로드 완료: {len(af_df)} 행")
    except Exception as e:
        print(f"오류: 산불 데이터 로드 실패: {e}")
        return False
    
    # 공통 열 확인
    common_cols = ['acq_date', 'grid_id']
    for col in common_cols:
        if col not in weather_df.columns:
            print(f"오류: 날씨 데이터에 필수 열이 없습니다: {col}")
            return False
        if col not in af_df.columns:
            print(f"오류: 산불 데이터에 필수 열이 없습니다: {col}")
            return False
    
    # 시간 범위 검증
    weather_dates = [weather_df['acq_date'].min(), weather_df['acq_date'].max()]
    af_dates = [af_df['acq_date'].min(), af_df['acq_date'].max()]
    
    print(f"\n시간 범위:")
    print(f"날씨 데이터: {weather_dates[0]} ~ {weather_dates[1]}")
    print(f"산불 데이터: {af_dates[0]} ~ {af_dates[1]}")
    
    date_issues = []
    if weather_dates[0] > af_dates[0]:
        date_issues.append(f"- 날씨 데이터 시작일이 산불 데이터보다 늦습니다: {weather_dates[0]} > {af_dates[0]}")
    if weather_dates[1] < af_dates[1]:
        date_issues.append(f"- 날씨 데이터 종료일이 산불 데이터보다 빠릅니다: {weather_dates[1]} < {af_dates[1]}")
    
    if date_issues:
        print("\n시간 범위 불일치 문제:")
        for issue in date_issues:
            print(issue)
    
    # 공간 범위 검증
    weather_grids = [weather_df['grid_id'].min(), weather_df['grid_id'].max()]
    af_grids = [af_df['grid_id'].min(), af_df['grid_id'].max()]
    
    print(f"\n공간 범위 (grid_id):")
    print(f"날씨 데이터: {weather_grids[0]} ~ {weather_grids[1]}")
    print(f"산불 데이터: {af_grids[0]} ~ {af_grids[1]}")
    
    grid_issues = []
    if weather_grids[0] > af_grids[0]:
        grid_issues.append(f"- 날씨 데이터 최소 grid_id가 산불 데이터보다 큽니다: {weather_grids[0]} > {af_grids[0]}")
    if weather_grids[1] < af_grids[1]:
        grid_issues.append(f"- 날씨 데이터 최대 grid_id가 산불 데이터보다 작습니다: {weather_grids[1]} < {af_grids[1]}")
    
    if grid_issues:
        print("\n공간 범위 불일치 문제:")
        for issue in grid_issues:
            print(issue)
    
    # 산불 데이터의 양성 샘플 검증
    af_positive = af_df[af_df['af_flag'] == 1]
    print(f"\n산불 발생 데이터 (af_flag=1): {len(af_positive)} 행 ({len(af_positive)/len(af_df)*100:.4f}%)")
    
    # 최종 병합 데이터의 양성 샘플 검증
    if 'af_flag' in weather_df.columns:
        weather_positive = weather_df[weather_df['af_flag'] == 1]
        print(f"최종 데이터의 산불 발생 샘플: {len(weather_positive)} 행 ({len(weather_positive)/len(weather_df)*100:.4f}%)")
        
        if len(weather_positive) < len(af_positive):
            print(f"경고: 최종 데이터의 산불 발생 샘플이 원본보다 적습니다: {len(weather_positive)} < {len(af_positive)}")
            print(f"손실된 산불 발생 샘플: {len(af_positive) - len(weather_positive)} 행")
    
    # 중복 확인
    weather_df_dup = weather_df.duplicated(subset=['acq_date', 'grid_id']).sum()
    af_df_dup = af_df.duplicated(subset=['acq_date', 'grid_id']).sum()
    
    print(f"\n중복 행:")
    print(f"날씨 데이터: {weather_df_dup} 행")
    print(f"산불 데이터: {af_df_dup} 행")
    
    if weather_df_dup > 0 or af_df_dup > 0:
        print("경고: 중복 행이 발견되었습니다.")
    
    # 결측치 확인
    weather_nulls = weather_df.isnull().sum()
    
    print(f"\n날씨 데이터 결측치:")
    for col in weather_nulls.index:
        if weather_nulls[col] > 0:
            print(f"- {col}: {weather_nulls[col]} 행 ({weather_nulls[col]/len(weather_df)*100:.2f}%)")
    
    # 검증 결과 요약
    has_issues = bool(date_issues or grid_issues or weather_df_dup > 0 or af_df_dup > 0)
    
    print("\n=== 검증 결과 요약 ===")
    if has_issues:
        print("경고: 데이터 차원 검증 중 일부 불일치가 발견되었습니다.")
        print("위의 문제를 확인하고 필요한 경우 데이터 수집 및 전처리 과정을 조정하세요.")
    else:
        print("성공: 모든 데이터 차원이 일치합니다.")
    
    return not has_issues

def main():
    parser = argparse.ArgumentParser(description='날씨 데이터와 산불 데이터의 차원 일치 검증')
    parser.add_argument('--weather_data', type=str, required=True,
                        help='전처리된 날씨 데이터 파일 경로')
    parser.add_argument('--af_flag', type=str, required=True,
                        help='산불 플래그 데이터 파일 경로')
    
    args = parser.parse_args()
    success = check_dimensions(args.weather_data, args.af_flag)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 