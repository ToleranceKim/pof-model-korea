#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import binascii

def check_file_format(file_path):
    """파일의 형식을 확인합니다."""
    try:
        with open(file_path, 'rb') as f:
            # 파일의 처음 20 바이트를 읽어서 형식을 확인
            header = f.read(20)
            
            # 헤더 출력
            print(f"File: {file_path}")
            print(f"Header hex: {binascii.hexlify(header).decode()}")
            print(f"Header as ASCII: {repr(header)}")
            
            # 일반적인 파일 형식 확인
            if header.startswith(b'CDF\x01'):
                print("This is a NetCDF classic file")
            elif header.startswith(b'\x89HDF\r\n\x1a\n'):
                print("This is a NetCDF-4/HDF5 file")
            elif header.startswith(b'PK\x03\x04'):
                print("This is a ZIP file")
            elif header.startswith(b'\x1f\x8b'):
                print("This is a GZIP file")
            elif header.startswith(b'BZh'):
                print("This is a BZIP2 file")
            else:
                print("Unknown file format")
    except Exception as e:
        print(f"Error checking file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        check_file_format(file_path)
    else:
        # 기본적으로 data 디렉토리의 첫 번째 파일 확인
        data_dir = "data"
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if f.startswith('era5_korea_') and f.endswith('.nc')]
        if files:
            check_file_format(files[0])
        else:
            print("No .nc files found in data directory") 