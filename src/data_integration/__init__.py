"""
Data Integration Module for POF-Korea Project

This module provides functionality to integrate various data sources
for wildfire probability prediction in South Korea.

Main components:
- core: Core utilities (grid system, spatial operations, validation)
- integrators: Step-by-step data integration modules
- pipeline: Main integration pipeline orchestration
"""

from .pipeline import DataIntegrationPipeline
from .core.grid_system import latlon2grid, grid2latlon, validate_grid_bounds
from .core.validators import validate_row_consistency, validate_missing_values

__version__ = "0.1.0"
__all__ = [
    "DataIntegrationPipeline",
    "latlon2grid", 
    "grid2latlon",
    "validate_grid_bounds",
    "validate_row_consistency",
    "validate_missing_values"
]