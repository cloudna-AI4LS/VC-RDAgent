#!/usr/bin/env python3
"""
Configuration utilities for handling variable substitution in JSON config files
"""

import json
import os
import re
import datetime
from pathlib import Path
from typing import Dict, Any


def resolve_path_variables(config: Dict[str, Any], base_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Resolve path variables in configuration using base_paths
    
    Args:
        config: Configuration dictionary
        base_paths: Dictionary of base path variables
        
    Returns:
        Configuration with resolved paths
    """
    def _resolve_value(value):
        if isinstance(value, str):
            # Replace variables in string
            for var_name, var_value in base_paths.items():
                placeholder = f"${{{var_name}}}"
                if placeholder in value:
                    value = value.replace(placeholder, var_value)
            
            # Handle timestamp variable
            if "${timestamp}" in value:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                value = value.replace("${timestamp}", timestamp)
            
            # Handle model_name variable
            if "${model_name}" in value:
                # Get from default_model_name in config
                default_model_name = config.get('model_config', {}).get('default_model_name', '')
                if default_model_name == "Qwen3API" or default_model_name.startswith("Qwen3API"):
                    # Get from qwen3API_config
                    qwen_config = config.get('qwen3API_config', {})
                    actual_model_name = qwen_config.get('model_name', 'Qwen/Qwen3-8B')
                elif default_model_name == "gemini" or default_model_name.startswith("gemini"):
                    # Get from gemini_config
                    gemini_config = config.get('gemini_config', {})
                    actual_model_name = gemini_config.get('model_name', 'gemini-2.5-flash')
                elif default_model_name == "openrouter" or default_model_name.startswith("openrouter"):
                    # Get from openrouter_config
                    openrouter_config = config.get('openrouter_config', {})
                    actual_model_name = openrouter_config.get('model_name', 'qwen/qwen3-8b:free')
                else:
                    actual_model_name = default_model_name
                
                if actual_model_name:
                    # Sanitize model name for filename (replace special characters)
                    model_name_sanitized = re.sub(r'[^\w\-_.]', '_', actual_model_name)
                    value = value.replace("${model_name}", model_name_sanitized)
            
            return value
        elif isinstance(value, list):
            # Recursively resolve list items
            return [_resolve_value(item) for item in value]
        elif isinstance(value, dict):
            # Recursively resolve dictionary values
            return {k: _resolve_value(v) for k, v in value.items()}
        else:
            return value
    
    return _resolve_value(config)


def load_config_with_variables(config_file: str) -> Dict[str, Any]:
    """
    Load configuration file and resolve path variables
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Resolved configuration dictionary
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Extract base paths
    base_paths = config.get('base_paths', {})
    
    # Resolve variables in the entire config
    resolved_config = resolve_path_variables(config, base_paths)
    
    return resolved_config


def get_file_path(config: Dict[str, Any], path_key: str, default: str = None) -> str:
    """
    Get a file path from configuration with fallback to default
    
    Args:
        config: Configuration dictionary
        path_key: Key to look for in file_paths (e.g., 'hpoa_file')
        default: Default value if not found
        
    Returns:
        Resolved file path
    """
    file_paths = config.get('file_paths', {})
    
    # Handle nested keys like 'mapping_files.phenotype_mapping'
    keys = path_key.split('.')
    value = file_paths
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    # Handle lists (return first item) or single values
    if isinstance(value, list):
        return value[0] if value else default
    else:
        return value if value else default


def validate_config_paths(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate that all file paths in configuration exist
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping path keys to existence status
    """
    validation_results = {}
    file_paths = config.get('file_paths', {})
    
    def _validate_paths(section, prefix=""):
        for key, value in section.items():
            if isinstance(value, dict):
                _validate_paths(value, f"{prefix}.{key}" if prefix else key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str) and (item.endswith('.json') or item.endswith('.csv') or item.endswith('.tsv') or item.endswith('.hpoa') or item.endswith('.jsonl')):
                        path_key = f"{prefix}.{key}[{i}]" if prefix else f"{key}[{i}]"
                        validation_results[path_key] = Path(item).exists()
            elif isinstance(value, str) and (value.endswith('.json') or value.endswith('.csv') or value.endswith('.tsv') or value.endswith('.hpoa') or value.endswith('.jsonl')):
                path_key = f"{prefix}.{key}" if prefix else key
                validation_results[path_key] = Path(value).exists()
    
    _validate_paths(file_paths)
    return validation_results


if __name__ == "__main__":
    # Test the configuration loading
    try:
        config = load_config_with_variables("inference_config.json")
        print("Configuration loaded successfully!")
        
        # Test path resolution
        hpoa_file = get_file_path(config, 'hpoa_file')
        print(f"HPOA file path: {hpoa_file}")
        
        # Validate paths
        validation = validate_config_paths(config)
        print("\nPath validation results:")
        for path_key, exists in validation.items():
            status = "✓" if exists else "✗"
            print(f"  {status} {path_key}")
            
    except Exception as e:
        print(f"Error: {e}") 