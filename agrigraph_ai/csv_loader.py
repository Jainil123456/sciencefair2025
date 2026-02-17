"""
CSV/Excel data loader for sensor data.
Validates and converts uploaded files to required format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict


def load_sensor_data(
    file_path: str,
    file_type: str = 'csv'
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load sensor data from CSV or Excel file.

    Expected columns:
    - x, X, or longitude: X coordinate
    - y, Y, or latitude: Y coordinate
    - NH3, ammonia: Ammonia concentration (ppm)
    - CH4, methane: Methane concentration (ppm)
    - NO2, nitrogen_dioxide: Nitrogen dioxide concentration (ppm)
    - CO, carbon_monoxide: Carbon monoxide concentration (ppm)
    - risk_score (optional): Ground truth labels [0-1]

    Args:
        file_path: Path to CSV/Excel file
        file_type: 'csv', 'xlsx', or 'xls'

    Returns:
        Tuple of:
        - locations: (num_nodes, 2) array of x, y coordinates
        - features: (num_nodes, 4) array of gas concentrations [NH3, CH4, NO2, CO]
        - risk_labels: (num_nodes,) array of risk scores (None if not provided)

    Raises:
        ValueError: If required columns missing or data validation fails
    """
    # Load file
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Extract coordinates
    x_col = next((c for c in ['x', 'longitude', 'lon'] if c in df.columns), None)
    y_col = next((c for c in ['y', 'latitude', 'lat'] if c in df.columns), None)

    if x_col is None or y_col is None:
        missing = []
        if x_col is None:
            missing.append('x/longitude')
        if y_col is None:
            missing.append('y/latitude')
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    try:
        locations = df[[x_col, y_col]].values.astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Coordinates must be numeric values: {str(e)}")

    # Extract gas concentrations
    gas_columns = {
        'NH3': ['nh3', 'ammonia'],
        'CH4': ['ch4', 'methane'],
        'NO2': ['no2', 'nitrogen_dioxide', 'nitrogendioxide'],
        'CO': ['co', 'carbon_monoxide', 'carbonmonoxide']
    }

    features = []
    for gas, possible_names in gas_columns.items():
        col = next((c for c in possible_names if c in df.columns), None)
        if col is None:
            raise ValueError(
                f"Missing required gas column: {gas} "
                f"(tried: {', '.join(possible_names)})"
            )
        try:
            features.append(df[col].values.astype(float))
        except (ValueError, TypeError) as e:
            raise ValueError(f"{gas} concentrations must be numeric: {str(e)}")

    features = np.column_stack(features)

    # Add coordinates to features: [NH3, CH4, NO2, CO, x, y]
    features_with_coords = np.column_stack([features, locations])

    # Extract risk labels if available
    risk_col = next(
        (c for c in ['risk_score', 'risk', 'label'] if c in df.columns),
        None
    )
    risk_labels = None
    if risk_col:
        try:
            risk_labels = df[risk_col].values.astype(float)
        except (ValueError, TypeError):
            print(f"Warning: Could not parse {risk_col} as numeric, ignoring labels")

    # Validate data
    if np.any(np.isnan(locations)):
        raise ValueError("Coordinates contain NaN values. Please clean your data.")
    if np.any(np.isnan(features_with_coords)):
        raise ValueError("Features contain NaN values. Please clean your data.")
    if np.any(features < 0):
        raise ValueError("Gas concentrations cannot be negative (ppm values must be >= 0)")

    if risk_labels is not None and np.any((risk_labels < 0) | (risk_labels > 1)):
        print("Warning: Risk scores outside [0, 1] range - normalizing")
        risk_labels = np.clip(risk_labels, 0, 1)

    return locations, features_with_coords, risk_labels


def validate_csv_format(file_path: str) -> Dict:
    """
    Validate CSV format and return info about the file.

    Args:
        file_path: Path to file to validate

    Returns:
        Dictionary with validation results:
        - valid (bool): Whether file is valid
        - num_rows (int): Number of data rows
        - columns (list): Column names
        - has_coordinates (bool): Has x, y or lat/lon columns
        - has_gases (bool): Has all required gas columns
        - has_labels (bool): Has risk_score/risk/label column
        - error (str): Error message if not valid
    """
    try:
        df = pd.read_csv(file_path, nrows=5)
        df.columns = df.columns.str.lower().str.strip()

        has_coords = any(c in df.columns for c in ['x', 'y', 'longitude', 'latitude', 'lon', 'lat'])
        has_gases = all(
            any(c in df.columns for c in variants)
            for variants in [
                ['nh3', 'ammonia'],
                ['ch4', 'methane'],
                ['no2', 'nitrogen_dioxide', 'nitrogendioxide'],
                ['co', 'carbon_monoxide', 'carbonmonoxide']
            ]
        )
        has_labels = any(c in df.columns for c in ['risk_score', 'risk', 'label'])

        return {
            'valid': True,
            'num_rows': len(df),
            'columns': list(df.columns),
            'has_coordinates': has_coords,
            'has_gases': has_gases,
            'has_labels': has_labels,
            'error': None
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'columns': [],
            'has_coordinates': False,
            'has_gases': False,
            'has_labels': False
        }
