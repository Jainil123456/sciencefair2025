"""
Interpretation layer for converting risk predictions into human-readable alerts.
"""

import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    location_id: int
    x: float
    y: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    risk_score: float
    primary_gas: str
    primary_gas_concentration: float
    recommendation: str


def categorize_risk(
    risk_score: float,
    thresholds: Dict[str, float]
) -> str:
    """
    Categorize risk score into risk level.
    
    Args:
        risk_score: Risk score (0-1)
        thresholds: Dictionary with risk level thresholds
        
    Returns:
        Risk level string
    """
    if risk_score >= thresholds['critical']:
        return 'critical'
    elif risk_score >= thresholds['high']:
        return 'high'
    elif risk_score >= thresholds['medium']:
        return 'medium'
    else:
        return 'low'


def identify_primary_gas(
    gas_concentrations: np.ndarray,
    gas_names: List[str],
    gas_ranges: Dict[str, Tuple[float, float]]
) -> Tuple[str, float]:
    """
    Identify the gas with highest normalized concentration.
    
    Args:
        gas_concentrations: Array of gas concentrations [NH3, CH4, NO2, CO]
        gas_names: List of gas names
        gas_ranges: Dictionary with gas ranges
        
    Returns:
        Tuple of (gas_name, normalized_concentration)
    """
    normalized = []
    for i, gas_name in enumerate(gas_names):
        min_val, max_val = gas_ranges[gas_name]
        norm_val = (gas_concentrations[i] - min_val) / (max_val - min_val + 1e-8)
        normalized.append(norm_val)
    
    max_idx = np.argmax(normalized)
    return gas_names[max_idx], gas_concentrations[max_idx]


def generate_recommendation(
    risk_level: str,
    primary_gas: str
) -> str:
    """
    Generate human-readable recommendation based on risk level and primary gas.
    
    Args:
        risk_level: Risk level category
        primary_gas: Primary gas of concern
        
    Returns:
        Recommendation string
    """
    gas_descriptions = {
        'NH3': 'ammonia',
        'CH4': 'methane',
        'NO2': 'nitrogen dioxide',
        'CO': 'carbon monoxide'
    }
    
    gas_name = gas_descriptions.get(primary_gas, primary_gas)
    
    recommendations = {
        'low': f"Field conditions are normal. Monitor {gas_name} levels regularly.",
        'medium': f"Elevated {gas_name} detected. Increase monitoring frequency and check soil conditions.",
        'high': f"High {gas_name} concentration detected. Consider soil aeration and reduce fertilizer application. Consult agricultural expert.",
        'critical': f"CRITICAL: Very high {gas_name} levels detected. Immediate action required: evacuate area if necessary, contact agricultural emergency services, and implement soil remediation measures."
    }
    
    return recommendations.get(risk_level, "Monitor field conditions.")


def generate_alerts(
    locations: np.ndarray,
    risk_scores: np.ndarray,
    gas_features: np.ndarray,
    gas_names: List[str],
    gas_ranges: Dict[str, Tuple[float, float]],
    risk_thresholds: Dict[str, float],
    filter_by_level: Optional[str] = None
) -> List[RiskAlert]:
    """
    Generate risk alerts for all locations or filtered by risk level.
    
    Args:
        locations: Array of (x, y) coordinates
        risk_scores: Predicted risk scores
        gas_features: Gas concentration features [NH3, CH4, NO2, CO]
        gas_names: List of gas names
        gas_ranges: Dictionary with gas ranges
        risk_thresholds: Risk level thresholds
        filter_by_level: Optional risk level to filter by ('low', 'medium', 'high', 'critical')
        
    Returns:
        List of RiskAlert objects
    """
    alerts = []
    
    for i in range(len(locations)):
        risk_score = risk_scores[i]
        risk_level = categorize_risk(risk_score, risk_thresholds)
        
        # Filter if requested
        if filter_by_level and risk_level != filter_by_level:
            continue
        
        # Identify primary gas
        primary_gas, primary_conc = identify_primary_gas(
            gas_features[i], gas_names, gas_ranges
        )
        
        # Generate recommendation
        recommendation = generate_recommendation(risk_level, primary_gas)
        
        alert = RiskAlert(
            location_id=i,
            x=float(locations[i, 0]),
            y=float(locations[i, 1]),
            risk_level=risk_level,
            risk_score=float(risk_score),
            primary_gas=primary_gas,
            primary_gas_concentration=float(primary_conc),
            recommendation=recommendation
        )
        
        alerts.append(alert)
    
    # Sort by risk score (highest first)
    alerts.sort(key=lambda x: x.risk_score, reverse=True)
    
    return alerts


def alerts_to_json(
    alerts: List[RiskAlert],
    filepath: Optional[str] = None
) -> str:
    """
    Convert alerts to JSON format for mobile app integration.
    
    Args:
        alerts: List of RiskAlert objects
        filepath: Optional path to save JSON file
        
    Returns:
        JSON string
    """
    alerts_dict = {
        'total_alerts': len(alerts),
        'alerts': [asdict(alert) for alert in alerts]
    }
    
    json_str = json.dumps(alerts_dict, indent=2)
    
    if filepath:
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(json_str)
        print(f"Alerts saved to {filepath}")
    
    return json_str


def print_alerts_summary(
    alerts: List[RiskAlert],
    top_n: int = 10
):
    """
    Print human-readable summary of alerts.
    
    Args:
        alerts: List of RiskAlert objects
        top_n: Number of top alerts to display
    """
    print("\n" + "="*80)
    print("RISK ALERTS SUMMARY")
    print("="*80)
    
    # Count by risk level
    level_counts = {}
    for alert in alerts:
        level_counts[alert.risk_level] = level_counts.get(alert.risk_level, 0) + 1
    
    print(f"\nTotal Alerts: {len(alerts)}")
    print(f"  - Critical: {level_counts.get('critical', 0)}")
    print(f"  - High: {level_counts.get('high', 0)}")
    print(f"  - Medium: {level_counts.get('medium', 0)}")
    print(f"  - Low: {level_counts.get('low', 0)}")
    
    # Display top alerts
    print(f"\nTop {min(top_n, len(alerts))} Highest Risk Locations:")
    print("-"*80)
    
    for i, alert in enumerate(alerts[:top_n], 1):
        print(f"\n{i}. Location ID {alert.location_id} at ({alert.x:.1f}, {alert.y:.1f})")
        print(f"   Risk Level: {alert.risk_level.upper()}")
        print(f"   Risk Score: {alert.risk_score:.3f}")
        print(f"   Primary Gas: {alert.primary_gas} ({alert.primary_gas_concentration:.2f} ppm)")
        print(f"   Recommendation: {alert.recommendation}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Test interpretation
    try:
        from .config import Config
        gas_ranges = Config.GAS_RANGES
        risk_thresholds = Config.RISK_THRESHOLDS
    except ImportError:
        # Fallback if running directly
        gas_ranges = {
            'NH3': (0.0, 50.0),
            'CH4': (0.0, 100.0),
            'NO2': (0.0, 5.0),
            'CO': (0.0, 10.0)
        }
        risk_thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
    
    # Create dummy data
    num_nodes = 20
    locations = np.random.rand(num_nodes, 2) * 100
    risk_scores = np.random.rand(num_nodes)
    gas_features = np.random.rand(num_nodes, 4) * 50
    
    alerts = generate_alerts(
        locations, risk_scores, gas_features,
        ['NH3', 'CH4', 'NO2', 'CO'],
        gas_ranges,
        risk_thresholds
    )
    
    print_alerts_summary(alerts, top_n=5)
    json_output = alerts_to_json(alerts)
    print("\nJSON Output (first 500 chars):")
    print(json_output[:500])

