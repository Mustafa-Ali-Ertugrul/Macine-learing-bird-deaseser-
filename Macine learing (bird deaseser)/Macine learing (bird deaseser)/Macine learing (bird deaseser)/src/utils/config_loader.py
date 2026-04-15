"""Configuration loader utility."""
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """YAML configuration file loader."""
    
    @staticmethod
    def load(config_path: str | Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
