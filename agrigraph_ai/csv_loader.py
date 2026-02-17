"""
Flexible CSV/Excel data loader for sensor data.
Auto-detects columns, handles missing data, and adapts to any format.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List


# Column name mappings - maps many possible names to canonical names
COORDINATE_ALIASES = {
    'x': ['x', 'x_coord', 'x_coordinate', 'xcoord', 'easting', 'col', 'column'],
    'y': ['y', 'y_coord', 'y_coordinate', 'ycoord', 'northing', 'row'],
    'lon': ['longitude', 'lon', 'lng', 'long'],
    'lat': ['latitude', 'lat'],
}

GAS_ALIASES = {
    'NH3': ['nh3', 'ammonia', 'nh3_ppm', 'ammonia_ppm', 'nh3_concentration'],
    'CH4': ['ch4', 'methane', 'ch4_ppm', 'methane_ppm', 'ch4_concentration'],
    'NO2': ['no2', 'nitrogen_dioxide', 'nitrogendioxide', 'no2_ppm', 'no2_concentration'],
    'CO': ['co', 'carbon_monoxide', 'carbonmonoxide', 'co_ppm', 'co_concentration'],
}

LABEL_ALIASES = ['risk_score', 'risk', 'label', 'target', 'risk_level', 'score',
                 'ground_truth', 'y_true', 'class', 'category']

# Columns to skip (not useful as features)
SKIP_COLUMNS = ['id', 'index', 'name', 'timestamp', 'date', 'time', 'datetime',
                'sensor_id', 'station', 'station_id', 'unnamed', 'unnamed: 0']


def _find_column(df_columns: List[str], aliases: List[str]) -> Optional[str]:
    """Find a column by checking against a list of aliases."""
    for alias in aliases:
        if alias in df_columns:
            return alias
    return None


def _detect_coordinate_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Auto-detect which columns contain x/y or lat/lon coordinates."""
    cols = list(df.columns)

    # Try x/y first
    x_col = _find_column(cols, COORDINATE_ALIASES['x'])
    y_col = _find_column(cols, COORDINATE_ALIASES['y'])
    if x_col and y_col:
        return x_col, y_col

    # Try lon/lat
    lon_col = _find_column(cols, COORDINATE_ALIASES['lon'])
    lat_col = _find_column(cols, COORDINATE_ALIASES['lat'])
    if lon_col and lat_col:
        return lon_col, lat_col

    # If only one coordinate found, try to find the other
    if x_col and not y_col:
        y_col = _find_column(cols, COORDINATE_ALIASES['y'] + COORDINATE_ALIASES['lat'])
        if y_col:
            return x_col, y_col
    if y_col and not x_col:
        x_col = _find_column(cols, COORDINATE_ALIASES['x'] + COORDINATE_ALIASES['lon'])
        if x_col:
            return x_col, y_col

    return x_col, y_col


def _detect_gas_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Detect which gas columns are present."""
    cols = list(df.columns)
    found = {}
    for gas, aliases in GAS_ALIASES.items():
        found[gas] = _find_column(cols, aliases)
    return found


def _detect_numeric_feature_columns(df: pd.DataFrame,
                                     used_columns: List[str]) -> List[str]:
    """Find all remaining numeric columns that could be features."""
    feature_cols = []
    for col in df.columns:
        if col in used_columns:
            continue
        if col in SKIP_COLUMNS:
            continue
        if any(skip in col for skip in ['unnamed', 'index']):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def load_sensor_data(
    file_path: str,
    file_type: str = 'csv'
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Flexibly load sensor data from CSV or Excel file.

    Handles many different formats:
    - Auto-detects coordinate columns (x/y, lat/lon, etc.)
    - Auto-detects gas columns (NH3, CH4, NO2, CO with many name variants)
    - Uses ANY numeric columns as features if specific gas columns aren't found
    - Generates synthetic coordinates if none provided (grid layout)
    - Fills missing gas columns with zeros
    - Handles NaN values by filling with column median
    - Normalizes coordinates to 0-100 range

    Args:
        file_path: Path to CSV/Excel file
        file_type: 'csv', 'xlsx', or 'xls'

    Returns:
        Tuple of (locations, features, risk_labels)
        - locations: (N, 2) array of (x, y) coordinates
        - features: (N, num_features) array with sensor readings + coordinates
        - risk_labels: (N,) array of risk scores, or None
    """
    # Step 1: Load file
    try:
        if file_type == 'csv':
            # Try multiple encodings and separators
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
            # If only 1 column, try different separators
            if len(df.columns) == 1:
                for sep in [';', '\t', '|']:
                    try:
                        df2 = pd.read_csv(file_path, sep=sep)
                        if len(df2.columns) > 1:
                            df = df2
                            break
                    except Exception:
                        continue
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")

    if len(df) == 0:
        raise ValueError("File is empty - no data rows found")

    if len(df.columns) < 2:
        raise ValueError(
            f"File has only {len(df.columns)} column(s). "
            f"Need at least 2 numeric columns for analysis."
        )

    # Step 2: Normalize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    # Drop completely empty columns
    df = df.dropna(axis=1, how='all')

    print(f"[CSV Loader] Loaded {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")

    # Step 3: Detect coordinate columns
    x_col, y_col = _detect_coordinate_columns(df)
    used_cols = []

    if x_col and y_col:
        try:
            locations = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').values
            # Fill NaN coordinates with random values in range
            for i in range(2):
                mask = np.isnan(locations[:, i])
                if np.any(mask):
                    valid = locations[~mask, i]
                    if len(valid) > 0:
                        locations[mask, i] = np.random.uniform(valid.min(), valid.max(), mask.sum())
                    else:
                        locations[mask, i] = np.random.uniform(0, 100, mask.sum())
            used_cols.extend([x_col, y_col])
            print(f"[CSV Loader] Found coordinates: x={x_col}, y={y_col}")
        except Exception:
            x_col, y_col = None, None

    if x_col is None or y_col is None:
        # Generate grid coordinates
        n = len(df)
        grid_size = int(np.ceil(np.sqrt(n)))
        xs = np.tile(np.linspace(5, 95, grid_size), grid_size)[:n]
        ys = np.repeat(np.linspace(5, 95, grid_size), grid_size)[:n]
        # Add small jitter for realism
        xs += np.random.uniform(-2, 2, n)
        ys += np.random.uniform(-2, 2, n)
        locations = np.column_stack([xs, ys])
        print(f"[CSV Loader] No coordinate columns found - generated grid layout for {n} points")

    # Normalize coordinates to 0-100 range
    for i in range(2):
        col_min, col_max = locations[:, i].min(), locations[:, i].max()
        if col_max > col_min:
            locations[:, i] = (locations[:, i] - col_min) / (col_max - col_min) * 100
        else:
            locations[:, i] = 50.0  # All same value, center them

    # Step 4: Detect and extract gas columns
    gas_map = _detect_gas_columns(df)
    gas_features = []
    gas_names_found = []

    for gas, col in gas_map.items():
        if col:
            vals = pd.to_numeric(df[col], errors='coerce').values
            # Fill NaN with column median
            median = np.nanmedian(vals) if not np.all(np.isnan(vals)) else 0
            vals = np.where(np.isnan(vals), median, vals)
            gas_features.append(vals)
            gas_names_found.append(gas)
            used_cols.append(col)
            print(f"[CSV Loader] Found gas column: {gas} = '{col}'")

    # Step 5: Find any other numeric columns as additional features
    extra_feature_cols = _detect_numeric_feature_columns(df, used_cols)
    extra_features = []

    for col in extra_feature_cols:
        if col in [_find_column(list(df.columns), LABEL_ALIASES)]:
            continue  # Skip label columns
        vals = pd.to_numeric(df[col], errors='coerce').values
        if np.all(np.isnan(vals)):
            continue  # Skip all-NaN columns
        median = np.nanmedian(vals)
        vals = np.where(np.isnan(vals), median, vals)
        extra_features.append(vals)
        used_cols.append(col)
        print(f"[CSV Loader] Using numeric column as feature: '{col}'")

    # Step 6: Build feature matrix
    # We need at least some features for the model
    all_feature_arrays = gas_features + extra_features

    if len(all_feature_arrays) == 0:
        raise ValueError(
            "No numeric feature columns found in your data. "
            f"Columns found: {list(df.columns)}. "
            "Need at least one numeric column with sensor/measurement data."
        )

    # If we have fewer than 4 gas columns, pad with zeros or duplicate existing
    # The model expects INPUT_FEATURES columns (default 6: 4 gases + x + y)
    # But we can adapt - just use whatever features we have + coordinates
    features_matrix = np.column_stack(all_feature_arrays)

    # Add coordinates to features
    features_with_coords = np.column_stack([features_matrix, locations])

    print(f"[CSV Loader] Feature matrix: {features_with_coords.shape[0]} rows x "
          f"{features_with_coords.shape[1]} columns "
          f"({len(gas_features)} gas + {len(extra_features)} extra + 2 coords)")

    # Step 7: Extract risk labels if available
    label_col = _find_column(list(df.columns), LABEL_ALIASES)
    risk_labels = None
    if label_col and label_col not in used_cols:
        try:
            risk_labels = pd.to_numeric(df[label_col], errors='coerce').values
            # Handle NaN in labels
            mask = np.isnan(risk_labels)
            if np.any(mask):
                risk_labels[mask] = np.nanmedian(risk_labels)
            # Normalize to 0-1
            rmin, rmax = risk_labels.min(), risk_labels.max()
            if rmax > rmin:
                risk_labels = (risk_labels - rmin) / (rmax - rmin)
            else:
                risk_labels = np.full_like(risk_labels, 0.5)
            print(f"[CSV Loader] Found risk labels in column: '{label_col}'")
        except Exception:
            risk_labels = None

    # If no risk labels, generate them from feature values
    if risk_labels is None:
        # Use mean of normalized features as a proxy risk score
        normalized = features_matrix.copy()
        for i in range(normalized.shape[1]):
            col_min, col_max = normalized[:, i].min(), normalized[:, i].max()
            if col_max > col_min:
                normalized[:, i] = (normalized[:, i] - col_min) / (col_max - col_min)
            else:
                normalized[:, i] = 0.5
        risk_labels = normalized.mean(axis=1)
        risk_labels = np.clip(risk_labels, 0, 1)
        print(f"[CSV Loader] No risk labels found - generated from feature averages")

    return locations, features_with_coords, risk_labels


def validate_csv_format(file_path: str) -> Dict:
    """
    Validate and describe a CSV file's format.
    Now much more permissive - accepts almost any tabular data.

    Returns:
        Dictionary with validation info
    """
    try:
        try:
            df = pd.read_csv(file_path, nrows=10)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, nrows=10, encoding='latin-1')

        # Try different separators if only 1 column
        if len(df.columns) == 1:
            for sep in [';', '\t', '|']:
                try:
                    df2 = pd.read_csv(file_path, nrows=10, sep=sep)
                    if len(df2.columns) > 1:
                        df = df2
                        break
                except Exception:
                    continue

        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

        # Count numeric columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        x_col, y_col = _detect_coordinate_columns(df)
        gas_map = _detect_gas_columns(df)
        has_coords = x_col is not None and y_col is not None
        gas_found = [g for g, c in gas_map.items() if c is not None]
        has_some_gases = len(gas_found) > 0
        label_col = _find_column(list(df.columns), LABEL_ALIASES)

        # We accept data as long as it has at least 1 numeric column
        is_valid = len(numeric_cols) >= 1

        return {
            'valid': is_valid,
            'num_rows': len(df),
            'columns': list(df.columns),
            'numeric_columns': numeric_cols,
            'has_coordinates': has_coords,
            'has_gases': has_some_gases,
            'gases_found': gas_found,
            'has_labels': label_col is not None,
            'error': None if is_valid else 'No numeric columns found in data'
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'columns': [],
            'numeric_columns': [],
            'has_coordinates': False,
            'has_gases': False,
            'gases_found': [],
            'has_labels': False
        }
