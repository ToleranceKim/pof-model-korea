#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import os
from datetime import datetime

def grid_id_to_latlon(grid_id):
    """
    Convert grid ID to latitude and longitude.
    
    Parameters:
    -----------
    grid_id : int or array-like
        Grid ID to convert
    
    Returns:
    --------
    tuple : (latitude, longitude) tuple
    """
    # Row index = int(grid_id / 3600) - 900
    # Col index = grid_id % 3600 - 1800
    lat_idx = np.floor_divide(grid_id, 3600) - 900
    lon_idx = np.remainder(grid_id, 3600) - 1800
    
    # Calculate 0.1 degree grid center point (add 0.05 degree offset)
    lat = lat_idx * 0.1 + 0.05
    lon = lon_idx * 0.1 + 0.05
    
    return lat, lon

def visualize_af_flag_by_year(af_flag_file, output_dir, start_year=None, end_year=None):
    """
    Visualize grids with af_flag=1 on a map of Korea by year.
    
    Parameters:
    -----------
    af_flag_file : str
        Path to preprocessed af_flag data file
    output_dir : str
        Output directory path
    start_year : int, optional
        Start year (default: minimum year in data)
    end_year : int, optional
        End year (default: maximum year in data)
    """
    print(f"\n=== Visualizing af_flag data ===")
    print(f"af_flag data: {af_flag_file}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv(af_flag_file)
    print(f"Data size: {df.shape}")
    
    # Count af_flag values
    af_flag_counts = df['af_flag'].value_counts()
    print("\naf_flag value counts:")
    print(af_flag_counts)
    
    if 1 not in af_flag_counts:
        print("\nERROR: No af_flag=1 records found in the data!")
        print("Please check if the data file contains fire events.")
        return
    
    # Convert date
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    
    # Extract year
    df['year'] = df['acq_date'].dt.year
    
    # Set year range
    all_years = sorted(df['year'].unique())
    if start_year is None:
        start_year = min(all_years)
    if end_year is None:
        end_year = max(all_years)
    
    years_to_process = [y for y in all_years if start_year <= y <= end_year]
    print(f"Years to process: {years_to_process}")
    
    # Filter only af_flag=1 data
    positive_df = df[df['af_flag'] == 1].copy()
    print(f"Number of af_flag=1 records: {len(positive_df)}")
    
    if len(positive_df) == 0:
        print("ERROR: No af_flag=1 records in the specified year range!")
        return
    
    # Convert grid ID to latitude/longitude
    print("\n[2/4] Converting grid IDs to lat/lon...")
    lats, lons = grid_id_to_latlon(positive_df['grid_id'].values)
    positive_df['latitude'] = lats
    positive_df['longitude'] = lons
    
    # Visualize by year
    print("\n[3/4] Visualizing by year...")
    for year in years_to_process:
        print(f"Processing year {year}...")
        
        # Filter data for the year
        year_df = positive_df[positive_df['year'] == year]
        if len(year_df) == 0:
            print(f"  No af_flag=1 data for year {year}.")
            continue
        
        # Calculate monthly frequency
        monthly_counts = year_df.groupby(year_df['acq_date'].dt.month).size()
        
        # Map visualization
        fig = plt.figure(figsize=(15, 10))
        
        # Map settings
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([124, 132, 33, 39])  # Korea region
        
        # Add map layers
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Display grid points (heatmap style)
        heat_map = ax.hexbin(
            year_df['longitude'], 
            year_df['latitude'], 
            gridsize=50, 
            cmap='hot_r',
            mincnt=1,
            bins='log',
            extent=[124, 132, 33, 39]
        )
        
        # Display grid points (scatter plot)
        ax.scatter(
            year_df['longitude'], 
            year_df['latitude'], 
            c='red', 
            s=5,
            alpha=0.5, 
            transform=ccrs.PlateCarree(),
            label=f'af_flag=1 ({len(year_df)} points)'
        )
        
        # Add colorbar
        cbar = plt.colorbar(heat_map, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label('Fire occurrence frequency (log scale)')
        
        # Add information
        plt.title(f"Fire Distribution in {year} (af_flag=1)")
        plt.legend(loc='upper right')
        
        # Add monthly frequency chart
        ax_inset = fig.add_axes([0.15, 0.15, 0.2, 0.2])
        months = range(1, 13)
        monthly_data = [monthly_counts.get(m, 0) for m in months]
        ax_inset.bar(months, monthly_data, color='darkred')
        ax_inset.set_title('Monthly Fire Occurrences')
        ax_inset.set_xlabel('Month')
        ax_inset.set_ylabel('Count')
        ax_inset.set_xticks(months)
        
        # Save file
        output_file = os.path.join(output_dir, f"af_flag_map_{year}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Map saved: {output_file}")
    
    # Create combined year map
    print("\n[4/4] Creating combined year map...")
    
    # Map visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Map settings
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([124, 132, 33, 39])  # Korea region
    
    # Add map layers
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Use different colors for each year
    years_subset = years_to_process[-5:] if len(years_to_process) > 5 else years_to_process
    colors = plt.cm.jet(np.linspace(0, 1, len(years_subset)))
    
    # Display points for each year
    for i, year in enumerate(years_subset):
        year_df = positive_df[positive_df['year'] == year]
        if len(year_df) > 0:
            ax.scatter(
                year_df['longitude'], 
                year_df['latitude'], 
                c=[colors[i]], 
                s=15, 
                alpha=0.7,
                label=f'{year} ({len(year_df)} points)',
                transform=ccrs.PlateCarree()
            )
    
    # Add information
    if len(years_subset) == 1:
        title = f"Fire Locations in {years_subset[0]}"
    else:
        title = f"Fire Locations {years_subset[0]}-{years_subset[-1]}"
    
    plt.title(title)
    plt.legend(loc='upper right')
    
    # Save file
    output_file = os.path.join(output_dir, f"af_flag_map_combined.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined map saved: {output_file}")
    
    # Output yearly statistics
    yearly_stats = positive_df.groupby('year').size().sort_index()
    print("\nYearly af_flag=1 counts:")
    for year, count in yearly_stats.items():
        print(f"{year}: {count} points")
    
    print("\n=== af_flag data visualization complete ===")

def main():
    parser = argparse.ArgumentParser(description='af_flag data visualization')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to preprocessed af_flag data file')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations',
                        help='Output directory path (default: outputs/visualizations)')
    parser.add_argument('--start-year', type=int,
                        help='Start year')
    parser.add_argument('--end-year', type=int,
                        help='End year')
    
    args = parser.parse_args()
    
    visualize_af_flag_by_year(
        args.input,
        args.output_dir,
        args.start_year,
        args.end_year
    )

if __name__ == '__main__':
    main() 