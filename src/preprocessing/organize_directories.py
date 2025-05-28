#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """필요한 디렉토리 구조 생성"""
    dirs = [
        'data/raw',            # 원본 ERA5 데이터
        'process_data/era5',   # 0.1도 보간 중간 결과
        'processed_data/era5_daily'  # 최종 일별 통계 결과
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")

def move_files():
    """파일을 올바른 디렉토리로 이동"""
    # 보간된 파일이 process_data/era5에 있는지 확인
    interp_dir = Path('process_data/era5')
    if not interp_dir.exists():
        logger.warning(f"Directory does not exist: {interp_dir}")
        return
    
    # 일별 통계 파일이 있는지 확인
    daily_files = []
    daily_dirs = [
        Path('process_data/era5_daily'),
        Path('processed_data/era5_daily')
    ]
    
    for daily_dir in daily_dirs:
        if daily_dir.exists():
            for file_path in daily_dir.glob('era5_daily_*.parquet'):
                daily_files.append(file_path)
            for file_path in daily_dir.glob('era5_daily_*.csv'):
                daily_files.append(file_path)
    
    # 일별 통계 파일을 processed_data/era5_daily로 이동
    if daily_files:
        target_dir = Path('processed_data/era5_daily')
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in daily_files:
            if file_path.parent != target_dir:
                target_path = target_dir / file_path.name
                if not target_path.exists() or os.path.getsize(file_path) > os.path.getsize(target_path):
                    logger.info(f"Moving {file_path} to {target_path}")
                    shutil.copy2(file_path, target_path)
                    # 원본 파일이 processed_data 폴더에 없으면 삭제
                    if 'processed_data' not in str(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed original file: {file_path}")

def main():
    logger.info("Starting directory organization")
    create_directories()
    move_files()
    logger.info("Directory organization complete")

if __name__ == "__main__":
    main() 