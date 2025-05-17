#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from datetime import datetime

def main():
    """전체 파이프라인 실행"""
    print("===== 전체 파이프라인 실행 시작 =====")
    print(f"시작 시간: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    print("1단계: 데이터 수집 실행")
    try:
        result = subprocess.run(["python", "collect_data.py"], check=True)
        if result.returncode != 0:
            print("데이터 수집 단계에서 오류가 발생했습니다.")
            print("오류를 해결한 후 다시 시도하세요.")
            sys.exit(1)
    except subprocess.CalledProcessError:
        print("데이터 수집 스크립트 실행 중 오류가 발생했습니다.")
        print("오류를 해결한 후 다시 시도하세요.")
        sys.exit(1)
    
    print()
    print("2단계: 데이터 전처리 실행")
    try:
        result = subprocess.run(["python", "process_data.py"], check=True)
        if result.returncode != 0:
            print("데이터 전처리 단계에서 오류가 발생했습니다.")
            print("오류를 해결한 후 다시 시도하세요.")
            sys.exit(1)
    except subprocess.CalledProcessError:
        print("데이터 전처리 스크립트 실행 중 오류가 발생했습니다.")
        print("오류를 해결한 후 다시 시도하세요.")
        sys.exit(1)
    
    print()
    print("===== 전체 파이프라인 실행 완료 =====")
    print(f"완료 시간: {datetime.now().strftime('%H:%M:%S')}")
    print()
    print("최종 결과 파일:")
    print("- 산불 데이터: data/reference/af_flag_korea.csv")
    print("- 최종 데이터: outputs/data/weather_data_with_wind.csv")

if __name__ == "__main__":
    main() 