#!/usr/bin/env python3
"""
Green Space Filter - NDVI-based vegetation detection tool.

This module provides simple threshold-based classification of green spaces
using NDVI calculated from multi-spectral satellite imagery.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, Dict


class GreenSpaceFilter:
    """
    NDVI-based green space classifier for 12-band satellite stacks.

    Works with stacks containing 4 bands (B02, B03, B04, B08) × 3 months.
    """

    def __init__(self, ndvi_threshold: float = 0.4, use_multi_temporal: bool = True,
                 output_dir: str = "results/green_filter"):
        """
        Initialize the Green Space Filter.

        Args:
            ndvi_threshold: NDVI threshold (0-1). Pixels above this are classified as green.
            use_multi_temporal: If True, averages NDVI across all months
            output_dir: Directory path for saving output files
        """
        self.ndvi_threshold = ndvi_threshold
        self.use_multi_temporal = use_multi_temporal
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Band indices for 12-band stack (4 bands × 3 months)
        # Band order: B02, B03, B04, B08 for April (0-3), August (4-7), November (8-11)
        self.band_indices = {
            'April': {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3},
            'August': {'B02': 4, 'B03': 5, 'B04': 6, 'B08': 7},
            'November': {'B02': 8, 'B03': 9, 'B04': 10, 'B08': 11}
        }

    def calculate_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI from NIR and Red bands.

        Args:
            nir: Near-infrared band (B08)
            red: Red band (B04)

        Returns:
            NDVI array with values between -1 and 1
        """
        nir = nir.astype(float)
        red = red.astype(float)

        # Calculate NDVI with epsilon to avoid division by zero
        ndvi = (nir - red) / (nir + red + 1e-8)

        return ndvi

    def calculate_evi(self, nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index (EVI).

        Args:
            nir: Near-infrared band (B08)
            red: Red band (B04)
            blue: Blue band (B02)

        Returns:
            EVI array
        """
        nir = nir.astype(float)
        red = red.astype(float)
        blue = blue.astype(float)

        evi = 2.5 * ((nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0))

        return evi

    def calculate_savi(self, nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
        """
        Calculate Soil-Adjusted Vegetation Index (SAVI).

        Args:
            nir: Near-infrared band (B08)
            red: Red band (B04)
            L: Soil brightness correction factor (default 0.5)

        Returns:
            SAVI array
        """
        nir = nir.astype(float)
        red = red.astype(float)

        savi = ((nir - red) / (nir + red + L)) * (1 + L)

        return savi

    def extract_bands_by_month(self, data: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract bands organized by month from the 12-band stack.

        Args:
            data: Full band stack (12 bands)

        Returns:
            Nested dict: {'April': {'B02': array, 'B03': array, ...}, 'August': {...}, ...}
        """
        months = {}

        for month, indices in self.band_indices.items():
            months[month] = {
                band_name: data[band_idx]
                for band_name, band_idx in indices.items()
            }

        return months

    def classify_green_spaces(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Classify pixels as green/non-green based on NDVI threshold.

        Args:
            data: Multi-month stack (12 bands: 4 raw bands × 3 months)

        Returns:
            Tuple of (binary_map, ndvi_maps) where:
                - binary_map: Binary classification (1=green, 0=non-green)
                - ndvi_maps: Dict of NDVI arrays by month for visualization
        """
        months_data = self.extract_bands_by_month(data)
        ndvi_maps = {}

        # Calculate NDVI for each month
        for month, bands in months_data.items():
            ndvi = self.calculate_ndvi(bands['B08'], bands['B04'])
            ndvi_maps[month] = ndvi

        # Decision logic based on configuration
        if self.use_multi_temporal:
            # Average NDVI across all months
            ndvi_stack = np.stack([ndvi_maps[m] for m in ['April', 'August', 'November']])
            ndvi_mean = np.nanmean(ndvi_stack, axis=0)
            binary_map = (ndvi_mean > self.ndvi_threshold).astype(np.uint8)
            print(f"Multi-temporal classification: mean NDVI threshold = {self.ndvi_threshold}")
        else:
            # Use August NDVI only (peak vegetation month)
            binary_map = (ndvi_maps['August'] > self.ndvi_threshold).astype(np.uint8)
            print(f"Single-month classification: August NDVI threshold = {self.ndvi_threshold}")

        return binary_map, ndvi_maps

    def load_stack(self, stack_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load a multi-month Sentinel-2 stack from GeoTIFF.

        Args:
            stack_path: Path to the multi-month stack GeoTIFF file

        Returns:
            Tuple of (data array, metadata dict)
        """
        with rasterio.open(stack_path) as src:
            data = src.read()
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'width': src.width,
                'height': src.height,
                'count': src.count
            }

        print(f"Loaded stack: {data.shape}")
        return data, metadata

    def process_city(self, stack_path: str, city_name: str) -> dict:
        """
        Complete processing pipeline for a single city.

        Args:
            stack_path: Path to the multi-month stack GeoTIFF
            city_name: Name of the city

        Returns:
            Dictionary with paths to output files
        """
        print(f"\nProcessing {city_name}...")

        # Load data
        data, metadata = self.load_stack(stack_path)

        # Classify
        binary_map, ndvi_maps = self.classify_green_spaces(data)

        # Calculate statistics
        green_pixels = np.sum(binary_map == 1)
        total_pixels = binary_map.size
        green_pct = 100 * green_pixels / total_pixels

        print(f"Green pixels: {green_pixels:,} / {total_pixels:,} ({green_pct:.2f}%)")

        return {
            'binary_map': binary_map,
            'ndvi_maps': ndvi_maps,
            'statistics': {
                'green_pixels': int(green_pixels),
                'total_pixels': int(total_pixels),
                'green_percentage': round(green_pct, 2)
            }
        }
