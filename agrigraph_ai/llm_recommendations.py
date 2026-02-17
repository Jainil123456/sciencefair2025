"""
LLM API integration for generating farmer recommendations.
Compares OpenAI GPT-4 and Anthropic Claude.
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def validate_alerts(alerts: List[Dict]) -> tuple[bool, str]:
    """
    Validate that alerts have required fields for LLM analysis.

    Args:
        alerts: List of alert dictionaries to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ['x', 'y', 'risk_level', 'risk_score', 'primary_gas', 'primary_gas_concentration']

    if not alerts:
        return False, "No alerts available for analysis"

    for i, alert in enumerate(alerts):
        for field in required_fields:
            if field not in alert:
                return False, f"Alert {i} missing required field: {field}"
            if field in ['x', 'y', 'risk_score', 'primary_gas_concentration']:
                try:
                    float(alert[field])
                except (ValueError, TypeError):
                    return False, f"Alert {i} field '{field}' is not numeric"

    return True, ""


def generate_farmer_prompt(
    alerts: List[Dict],
    field_stats: Dict
) -> str:
    """
    Generate a detailed prompt for LLMs based on soil analysis results.

    Args:
        alerts: List of risk alerts from GNN model (top 10-20)
        field_stats: Overall field statistics

    Returns:
        Formatted prompt string for LLM analysis
    """
    # Count alerts by level
    critical_count = sum(1 for a in alerts if a['risk_level'] == 'critical')
    high_count = sum(1 for a in alerts if a['risk_level'] == 'high')
    medium_count = sum(1 for a in alerts if a['risk_level'] == 'medium')
    low_count = sum(1 for a in alerts if a['risk_level'] == 'low')

    # Get primary gas concerns
    gas_counts = {}
    for alert in alerts:
        gas = alert['primary_gas']
        gas_counts[gas] = gas_counts.get(gas, 0) + 1

    primary_concern = max(gas_counts.items(), key=lambda x: x[1])[0] if gas_counts else "Unknown"

    prompt = f"""You are an expert agricultural soil scientist and farm management advisor.

SOIL SENSOR ANALYSIS RESULTS:
============================
Location: Farm field (coordinates range: x=[{field_stats.get('x_min', 0):.1f}-{field_stats.get('x_max', 100):.1f}], y=[{field_stats.get('y_min', 0):.1f}-{field_stats.get('y_max', 100):.1f}])
Number of sensor locations: {field_stats.get('num_nodes', 0)}
Model accuracy (R² score): {field_stats.get('r2_score', 0):.3f}

RISK ASSESSMENT SUMMARY:
- Critical zones: {critical_count}
- High risk zones: {high_count}
- Medium risk zones: {medium_count}
- Low risk zones: {low_count}
- Primary gas concern: {primary_concern} accumulation

DETAILED SENSOR ALERTS:
"""

    # Add detailed alerts
    for i, alert in enumerate(alerts[:15], 1):
        prompt += f"\n{i}. Location ({alert['x']:.1f}, {alert['y']:.1f}): "
        prompt += f"{alert['risk_level'].upper()} risk (score: {alert['risk_score']:.3f}) - "
        prompt += f"{alert['primary_gas']} at {alert['primary_gas_concentration']:.2f} ppm"

    prompt += """

ANALYSIS REQUEST:
=================
Based on these soil gas measurements and risk patterns, provide actionable advice:

1. **PROACTIVE REMEDIATION STEPS** (What should the farmer do immediately?)
   - Specific actions to reduce the detected gas concentrations
   - Priority order (critical areas first)
   - Estimated timeline and resources needed
   - Which remediation methods work best for this specific gas issue

2. **CROP PRODUCTION STRATEGIES** (How to optimize yields given current conditions?)
   - Best crops/varieties for these soil conditions
   - Recommended planting patterns and spacing
   - Fertilizer adjustments needed
   - Irrigation or drainage considerations
   - Pest/disease management specific to these gas levels

3. **MONITORING RECOMMENDATIONS**
   - How often to resample
   - Which areas need closest attention
   - Success metrics to track improvement

Be specific and practical - this advice will be implemented by a real farmer."""

    return prompt


def get_openai_recommendation(prompt: str) -> Dict:
    """
    Get recommendations from OpenAI GPT-4.

    Args:
        prompt: Analysis prompt for the LLM

    Returns:
        Dictionary with recommendation and metadata
    """
    try:
        import openai
        from openai import AuthenticationError, RateLimitError, APIError, Timeout

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {
                'success': False,
                'provider': 'OpenAI GPT-4',
                'error': 'API key not found. Check .env file for OPENAI_API_KEY'
            }

        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert agricultural advisor with deep knowledge of soil science, crop management, and farm economics. Provide practical, specific recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1500,
            timeout=30.0  # Prevent hanging requests
        )

        return {
            'success': True,
            'provider': 'OpenAI GPT-4',
            'model': response.model,
            'recommendation': response.choices[0].message.content,
            'tokens_used': response.usage.total_tokens,
            'finish_reason': response.choices[0].finish_reason
        }

    except Timeout:
        return {
            'success': False,
            'provider': 'OpenAI GPT-4',
            'error': 'Request timed out. OpenAI took too long to respond.'
        }
    except AuthenticationError:
        return {
            'success': False,
            'provider': 'OpenAI GPT-4',
            'error': 'Authentication failed. Invalid API key in .env'
        }
    except RateLimitError:
        return {
            'success': False,
            'provider': 'OpenAI GPT-4',
            'error': 'Rate limited. Exceeded OpenAI quota. Try again in 1 minute.'
        }
    except APIError as e:
        return {
            'success': False,
            'provider': 'OpenAI GPT-4',
            'error': f'OpenAI API error: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'provider': 'OpenAI GPT-4',
            'error': f'Unexpected error: {str(e)}'
        }


def get_anthropic_recommendation(prompt: str) -> Dict:
    """
    Get recommendations from Anthropic Claude.

    Args:
        prompt: Analysis prompt for the LLM

    Returns:
        Dictionary with recommendation and metadata
    """
    try:
        from anthropic import Anthropic, AuthenticationError, APIError

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return {
                'success': False,
                'provider': 'Anthropic Claude',
                'error': 'API key not found. Check .env file for ANTHROPIC_API_KEY'
            }

        client = Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-opus-4-1",
            max_tokens=1500,
            system="You are an expert agricultural advisor with deep knowledge of soil science, crop management, and farm economics. Provide practical, specific recommendations.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            'success': True,
            'provider': 'Anthropic Claude',
            'model': response.model,
            'recommendation': response.content[0].text,
            'tokens_used': response.usage.input_tokens + response.usage.output_tokens,
            'finish_reason': response.stop_reason
        }

    except AuthenticationError:
        return {
            'success': False,
            'provider': 'Anthropic Claude',
            'error': 'Authentication failed. Invalid API key in .env'
        }
    except APIError as e:
        return {
            'success': False,
            'provider': 'Anthropic Claude',
            'error': f'Anthropic API error: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'provider': 'Anthropic Claude',
            'error': f'Unexpected error: {str(e)}'
        }


def compare_llm_recommendations(
    alerts: List[Dict],
    field_stats: Dict
) -> Dict:
    """
    Generate and compare recommendations from multiple LLMs.

    Args:
        alerts: Risk alerts from GNN analysis
        field_stats: Field statistics

    Returns:
        Dictionary with recommendations from all LLMs
    """
    # Validate alerts before sending to LLMs
    is_valid, error_msg = validate_alerts(alerts)
    if not is_valid:
        return {
            'prompt': '',
            'gpt4': {'success': False, 'error': f'Data validation failed: {error_msg}'},
            'claude': {'success': False, 'error': f'Data validation failed: {error_msg}'},
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': {}
        }

    # Generate prompt
    prompt = generate_farmer_prompt(alerts, field_stats)

    print("Generating OpenAI GPT-4 recommendations...")
    gpt_result = get_openai_recommendation(prompt)

    print("Generating Anthropic Claude recommendations...")
    claude_result = get_anthropic_recommendation(prompt)

    return {
        'prompt': prompt,
        'gpt4': gpt_result,
        'claude': claude_result,
        'timestamp': datetime.now().isoformat(),
        'analysis_summary': {
            'num_alerts': len(alerts),
            'field_info': field_stats
        }
    }


def format_recommendation_for_display(text: str) -> str:
    """
    Format LLM recommendation text for HTML display.
    Preserves formatting and structure.

    Args:
        text: Raw recommendation text from LLM

    Returns:
        HTML-safe formatted text
    """
    import html
    # Escape HTML
    text = html.escape(text)
    # Preserve line breaks and structure
    text = text.replace('\n', '<br>')
    return text
