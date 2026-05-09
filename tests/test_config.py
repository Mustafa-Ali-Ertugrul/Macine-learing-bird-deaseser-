"""Tests for configuration loader."""
import pytest
from pathlib import Path
from src.utils.config_loader import ConfigLoader
from src.config import get_config, get_disease_classes, SUPPORTED_SPECIES


class TestConfigLoader:
    """Tests for ConfigLoader."""
    
    def test_load_valid_config(self):
        """Test loading valid config file."""
        config_path = Path("config/training_config.yaml")
        config = ConfigLoader.load(config_path)
        
        assert config is not None
        assert 'model' in config
        assert 'training' in config
        assert 'data' in config
    
    def test_load_nonexistent_config(self):
        """Test loading nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load("nonexistent.yaml")
    
    def test_config_structure(self):
        """Test config has expected structure."""
        config = ConfigLoader.load("config/training_config.yaml")
        
        # Check model config
        assert 'architecture' in config['model']
        assert 'pretrained' in config['model']
        
        # Check training config
        assert 'num_epochs' in config['training']
        assert 'batch_size' in config['training']
        assert 'optimizer' in config['training']
        
        # Check data config
        assert 'dataset_dir' in config['data']
        assert 'num_classes' in config['data']

    def test_cattle_species_config(self):
        """Test cattle species has independent classes and paths."""
        assert 'cattle' in SUPPORTED_SPECIES

        classes = get_disease_classes('cattle')
        config = get_config('vit_b16', 'cattle')

        assert 'Foot_and_Mouth_Disease' in classes
        assert 'Lumpy_Skin_Disease' in classes
        assert 'Mastitis' in classes
        assert 'Bovine_Tuberculosis' in classes
        assert 'Ringworm' in classes
        assert 'Digital_Dermatitis' in classes
        assert 'Bovine_Respiratory_Disease' not in classes
        assert config['class_names'] == classes
        assert config['num_classes'] == len(classes)
        assert 'models\\cattle' in config['model_save_path']
