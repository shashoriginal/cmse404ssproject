"""
Unit tests for helper utilities
Author: rajshash

This module provides unit tests for helper utilities used in financial indices prediction.
"""

import os
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import tempfile
import yaml
import torch
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.helpers import (
    create_directory_structure, normalize_data, calculate_metrics,
    plot_training_history, plot_predictions, plot_feature_importance,
    plot_correlation_matrix, create_sequences, load_yaml_config,
    create_data_loaders
)


class TestHelperUtilities(unittest.TestCase):
    """Test cases for helper utilities."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Generate random test data
        np.random.seed(42)
        self.test_data = np.random.randn(100, 5)
        self.test_features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
        self.test_df = pd.DataFrame(self.test_data, columns=self.test_features)
        
        # Create test configuration file
        self.config_path = os.path.join(self.test_dir, 'test_config.yaml')
        self.test_config = {
            'model': {
                'type': 'lstm',
                'units': 64,
                'dropout': 0.2
            },
            'training': {
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Create test tensor data
        self.test_tensor_data = torch.tensor(self.test_data, dtype=torch.float32)
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove the temporary directory and its contents
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_create_directory_structure(self):
        """Test creation of directory structure."""
        dirs = create_directory_structure(self.test_dir)
        
        # Check that all expected directories were created
        for dir_name, dir_path in dirs.items():
            self.assertTrue(os.path.isdir(dir_path))
            self.assertTrue(os.path.exists(dir_path))
    
    def test_normalize_data(self):
        """Test data normalization with different methods."""
        # Test standard normalization
        norm_data = normalize_data(self.test_data, method='standard')
        self.assertEqual(norm_data.shape, self.test_data.shape)
        self.assertAlmostEqual(norm_data.mean(), 0, delta=1e-6)
        self.assertAlmostEqual(norm_data.std(), 1, delta=1e-6)
        
        # Test minmax normalization
        norm_data = normalize_data(self.test_data, method='minmax')
        self.assertEqual(norm_data.shape, self.test_data.shape)
        self.assertGreaterEqual(norm_data.min(), 0)
        self.assertLessEqual(norm_data.max(), 1)
        
        # Test with return_scaler=True
        norm_data, scaler = normalize_data(self.test_data, method='standard', return_scaler=True)
        self.assertEqual(norm_data.shape, self.test_data.shape)
        self.assertIsNotNone(scaler)
        
        # Test 1D array
        norm_data = normalize_data(self.test_data[:, 0], method='standard')
        self.assertEqual(norm_data.shape, (self.test_data.shape[0],))
        
        # Test with tensor input
        tensor_data = self.test_tensor_data
        norm_data = normalize_data(tensor_data.numpy(), method='standard')
        self.assertEqual(norm_data.shape, tensor_data.shape)
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            normalize_data(self.test_data, method='invalid_method')
    
    def test_calculate_metrics(self):
        """Test calculation of regression metrics."""
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.randn(50) * 0.2  # Add some noise
        
        # Test with numpy arrays
        metrics = calculate_metrics(y_true, y_pred)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2_score', metrics)
        
        # Test with torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        metrics = calculate_metrics(y_true_tensor, y_pred_tensor)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2_score', metrics)
        
        # Test with 2D arrays
        y_true_2d = np.random.randn(50, 1)
        y_pred_2d = y_true_2d + np.random.randn(50, 1) * 0.2
        metrics = calculate_metrics(y_true_2d, y_pred_2d)
        self.assertIn('mse', metrics)
        
        # Test with 2D tensors
        y_true_tensor_2d = torch.tensor(y_true_2d, dtype=torch.float32)
        y_pred_tensor_2d = torch.tensor(y_pred_2d, dtype=torch.float32)
        metrics = calculate_metrics(y_true_tensor_2d, y_pred_tensor_2d)
        self.assertIn('mse', metrics)
    
    def test_plot_training_history(self):
        """Test plotting of training history."""
        # Create a test history dictionary
        history = {
            'loss': [0.5, 0.4, 0.3, 0.2, 0.1],
            'val_loss': [0.55, 0.45, 0.35, 0.25, 0.15],
            'mae': [0.4, 0.3, 0.2, 0.1, 0.05],
            'val_mae': [0.45, 0.35, 0.25, 0.15, 0.05],
        }
        
        # Test without save path
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_training_history(history)
            mock_show.assert_called_once()
        
        # Test with save path
        save_path = os.path.join(self.test_dir, 'test_history_plot.png')
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_training_history(history, save_path=save_path)
            mock_savefig.assert_called_once_with(save_path)
        
        # Test with empty metrics
        history_no_metrics = {
            'loss': [0.5, 0.4, 0.3, 0.2, 0.1],
            'val_loss': [0.55, 0.45, 0.35, 0.25, 0.15],
        }
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_training_history(history_no_metrics)
            mock_show.assert_called_once()

        # Test with empty val_loss
        history_no_val = {
            'loss': [0.5, 0.4, 0.3, 0.2, 0.1],
            'val_loss': [],
            'mae': [0.4, 0.3, 0.2, 0.1, 0.05],
        }
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_training_history(history_no_val)
            mock_show.assert_called_once()
    
    def test_plot_predictions(self):
        """Test plotting of predictions."""
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.randn(50) * 0.2  # Add some noise
        
        # Test with numpy arrays, without save path
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_predictions(y_true, y_pred)
            mock_show.assert_called_once()
        
        # Test with torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_predictions(y_true_tensor, y_pred_tensor)
            mock_show.assert_called_once()
        
        # Test with save path
        save_path = os.path.join(self.test_dir, 'test_predictions_plot.png')
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_predictions(y_true, y_pred, save_path=save_path)
            mock_savefig.assert_called_once_with(save_path)
        
        # Test with 2D arrays
        y_true_2d = np.random.randn(50, 1)
        y_pred_2d = y_true_2d + np.random.randn(50, 1) * 0.2
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_predictions(y_true_2d, y_pred_2d)
            mock_show.assert_called_once()
    
    def test_plot_feature_importance(self):
        """Test plotting of feature importance."""
        importances = np.abs(np.random.randn(5))
        
        # Test with numpy arrays, without save path
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_feature_importance(self.test_features, importances)
            mock_show.assert_called_once()
        
        # Test with torch tensors
        importances_tensor = torch.tensor(importances, dtype=torch.float32)
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_feature_importance(self.test_features, importances_tensor)
            mock_show.assert_called_once()
        
        # Test with save path
        save_path = os.path.join(self.test_dir, 'test_importance_plot.png')
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_feature_importance(self.test_features, importances, save_path=save_path)
            mock_savefig.assert_called_once_with(save_path)
    
    def test_plot_correlation_matrix(self):
        """Test plotting of correlation matrix."""
        # Test with pandas DataFrame, without save path
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_correlation_matrix(self.test_df)
            mock_show.assert_called_once()
        
        # Test with numpy array
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_correlation_matrix(self.test_data, self.test_features)
            mock_show.assert_called_once()
        
        # Test with torch tensor
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_correlation_matrix(self.test_tensor_data, self.test_features)
            mock_show.assert_called_once()
        
        # Test with save path
        save_path = os.path.join(self.test_dir, 'test_correlation_plot.png')
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_correlation_matrix(self.test_df, save_path=save_path)
            mock_savefig.assert_called_once_with(save_path)
    
    def test_create_sequences(self):
        """Test creation of sequences from time series data."""
        # Create test data
        sequence_length = 5
        test_ts_data = np.random.randn(20, 3)
        
        # Test with numpy arrays
        X, y = create_sequences(test_ts_data, sequence_length)
        self.assertEqual(X.shape, (15, 5, 3))
        self.assertEqual(y.shape, (15, 1))
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        
        # Test with torch tensors
        test_ts_tensor = torch.tensor(test_ts_data, dtype=torch.float32)
        X, y = create_sequences(test_ts_tensor, sequence_length)
        self.assertEqual(X.shape, (15, 5, 3))
        self.assertEqual(y.shape, (15, 1))
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
    
    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        # Test with existing file
        config = load_yaml_config(self.config_path)
        self.assertEqual(config, self.test_config)
        
        # Test with nonexistent file
        with self.assertRaises(FileNotFoundError):
            load_yaml_config('nonexistent_file.yaml')
    
    def test_create_data_loaders(self):
        """Test creation of data loaders."""
        # Create test data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Test with default parameters
        train_loader, val_loader = create_data_loaders(X, y)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Check batch size
        for batch_x, batch_y in train_loader:
            self.assertLessEqual(batch_x.shape[0], 32)  # Default batch size
            break
        
        # Test with numpy arrays
        train_loader, val_loader = create_data_loaders(X, y, batch_size=16)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Test with torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        train_loader, val_loader = create_data_loaders(X_tensor, y_tensor)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Test without validation split
        train_loader, val_loader = create_data_loaders(X, y, val_split=0)
        self.assertIsNotNone(train_loader)
        self.assertIsNone(val_loader)


if __name__ == '__main__':
    unittest.main() 