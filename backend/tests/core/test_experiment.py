"""
Unit tests for experiment utilities.
Author: rajshash

This module provides unit tests for the experiment utilities used in financial indices prediction.
"""

import os
import sys
import unittest
import tempfile
import yaml
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.experiment import run_experiment, save_results, load_results


class TestExperiment(unittest.TestCase):
    """Test cases for experiment utilities."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = {
            'data': {
                'file_path': 'data/sample_data.csv',
                'index_col': 'Date',
                'target_col': 'Close',
                'split_ratio': 0.8,
                'test_size': 0.2,
                'sequence_length': 10
            },
            'model': {
                'type': 'lstm',
                'input_shape': (10, 5),
                'output_shape': 1,
                'lstm_units': [64, 32],
                'dense_units': [32],
                'dropout_rate': 0.2,
                'batch_size': 32,
                'epochs': 2
            },
            'training': {
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'loss': 'mse',
                'early_stopping_patience': 5
            },
            'experiment': {
                'name': 'test_experiment',
                'log_dir': self.temp_dir,
                'results_dir': self.temp_dir,
                'model_dir': self.temp_dir
            }
        }
        
        # Save config to temporary file
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Create dummy data
        np.random.seed(42)
        self.data = np.random.randn(100, 5)  # 100 samples, 5 features
        self.target = np.random.randn(100, 1)  # 100 samples, 1 target
        
        # Create a results dictionary
        self.results = {
            'test_metrics': {
                'mae': 0.1,
                'mse': 0.2,
                'rmse': 0.3,
                'r2': 0.5
            },
            'predictions': np.random.randn(20, 1),
            'true_values': np.random.randn(20, 1),
            'model_config': self.config
        }
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('torch.nn.LSTM')
    @patch('src.models.lstm.LSTMModel')
    @patch('src.data.loader.load_and_prepare_data')
    def test_run_experiment(self, mock_data_loader, mock_model, mock_lstm):
        """Test running an experiment."""
        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_data_loader.return_value = (
            # train data
            (torch.randn(80, 10, 5), torch.randn(80, 1)),
            # val data
            (torch.randn(10, 10, 5), torch.randn(10, 1)),
            # test data
            (torch.randn(10, 10, 5), torch.randn(10, 1))
        )
        
        mock_model_instance.train_model.return_value = {
            'loss': [0.1, 0.05],
            'val_loss': [0.2, 0.1]
        }
        
        mock_model_instance.evaluate.return_value = {
            'loss': 0.1,
            'mse': 0.1,
            'mae': 0.05,
            'rmse': 0.3,
            'r2': 0.8
        }
        
        mock_model_instance.predict.return_value = torch.randn(10, 1)
        
        # Run the experiment
        results = run_experiment(config_path=self.config_path)
        
        # Check the calls
        mock_data_loader.assert_called()
        mock_model.assert_called()
        mock_model_instance.train_model.assert_called()
        mock_model_instance.evaluate.assert_called()
        mock_model_instance.predict.assert_called()
        
        # Check results structure
        self.assertIn('test_metrics', results)
        self.assertIn('predictions', results)
        self.assertIn('true_values', results)
        self.assertIn('model_config', results)
    
    def test_save_and_load_results(self):
        """Test saving and loading experiment results."""
        # Save results
        results_path = save_results(self.results, self.temp_dir)
        
        # Check file exists
        self.assertTrue(os.path.exists(results_path))
        
        # Load results
        loaded_results = load_results(results_path)
        
        # Check loaded results
        self.assertEqual(loaded_results['test_metrics']['mae'], self.results['test_metrics']['mae'])
        self.assertEqual(loaded_results['test_metrics']['mse'], self.results['test_metrics']['mse'])
        self.assertEqual(loaded_results['test_metrics']['rmse'], self.results['test_metrics']['rmse'])
        self.assertEqual(loaded_results['test_metrics']['r2'], self.results['test_metrics']['r2'])
        self.assertIn('predictions', loaded_results)
        self.assertIn('true_values', loaded_results)
        self.assertIn('model_config', loaded_results)


if __name__ == '__main__':
    unittest.main() 