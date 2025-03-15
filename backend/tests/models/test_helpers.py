"""
Unit tests for helper utilities
Author: rajshash

This module contains unit tests for the helper utilities module.
"""

import os
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import tempfile
import yaml
from unittest.mock import patch, MagicMock
import torch

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.helpers import (
    create_directory_structure,
    normalize_data,
    calculate_metrics,
    plot_training_history,
    plot_predictions,
    plot_feature_importance,
    plot_correlation_matrix,
    create_sequences,
    load_yaml_config,
    create_data_loaders
)


class TestHelperUtilities(unittest.TestCase):
    """Test case for helper utilities"""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_data = np.random.random((100, 5))
        self.feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
        
        # Create a test configuration file
        self.config_data = {
            'test_param': 'test_value',
            'nested': {
                'param1': 1,
                'param2': 'two'
            }
        }
        
        # Save the config file
        os.makedirs(os.path.join(self.test_dir, 'configs'), exist_ok=True)
        self.config_path = os.path.join(self.test_dir, 'configs', 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)
            
        # Create dummy data
        np.random.seed(42)
        self.data = np.random.randn(100, 5)
        self.data_tensor = torch.tensor(self.data, dtype=torch.float32)
            
    def tearDown(self):
        """Clean up after each test method."""
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.test_dir)
        
    def test_create_directory_structure(self):
        """Test directory structure creation."""
        # Create directory structure
        dirs = create_directory_structure(self.test_dir)
        
        # Check that all expected directories exist
        self.assertTrue(os.path.exists(dirs['data']))
        self.assertTrue(os.path.exists(dirs['raw']))
        self.assertTrue(os.path.exists(dirs['processed']))
        self.assertTrue(os.path.exists(dirs['features']))
        self.assertTrue(os.path.exists(dirs['models']))
        self.assertTrue(os.path.exists(dirs['checkpoints']))
        self.assertTrue(os.path.exists(dirs['results']))
        self.assertTrue(os.path.exists(dirs['visualizations']))
        
        # Check returned paths
        self.assertEqual(dirs['data'], os.path.join(self.test_dir, 'data'))
        self.assertEqual(dirs['raw'], os.path.join(self.test_dir, 'data', 'raw'))
        
    def test_normalize_data(self):
        """Test data normalization."""
        # Test MinMaxScaler
        normalized_data, scaler = normalize_data(self.data, method='minmax')
        
        # Check shape
        self.assertEqual(normalized_data.shape, self.data.shape)
        
        # Check range: values should be between -1 and 1 (or very close due to floating point precision)
        self.assertTrue(np.all(normalized_data >= -1.0 - 1e-10))
        self.assertTrue(np.all(normalized_data <= 1.0 + 1e-10))
        
        # Test inverse transform
        original_data = scaler.inverse_transform(normalized_data)
        np.testing.assert_allclose(original_data, self.data, rtol=1e-10)
        
        # Test StandardScaler
        normalized_data, scaler = normalize_data(self.data, method='standard')
        
        # Check shape
        self.assertEqual(normalized_data.shape, self.data.shape)
        
        # Check that mean is close to 0 and std close to 1
        self.assertAlmostEqual(normalized_data.mean(), 0, delta=0.1)
        self.assertAlmostEqual(normalized_data.std(), 1, delta=0.1)
        
        # Test inverse transform
        original_data = scaler.inverse_transform(normalized_data)
        np.testing.assert_allclose(original_data, self.data, rtol=1e-10)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            normalize_data(self.data, method='invalid')
        
        # Test with tensor
        normalized_data, scaler = normalize_data(self.data_tensor, method='minmax')
        self.assertIsInstance(normalized_data, np.ndarray)
        
    def test_calculate_metrics(self):
        """Test metric calculation."""
        # Create test data
        y_true = np.random.random(100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)  # Add some noise
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Check that all expected metrics are present
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2_score', metrics)
        
        # Check relationships between metrics
        self.assertAlmostEqual(metrics['rmse'], np.sqrt(metrics['mse']), delta=1e-10)
        self.assertGreaterEqual(metrics['r2_score'], 0.0)  # Should be positive as we're using a good predictor
        
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_training_history(self, mock_savefig, mock_show):
        """Test training history plotting."""
        # Create test history
        history = {
            'loss': [0.5, 0.3, 0.2, 0.1],
            'val_loss': [0.6, 0.4, 0.3, 0.2],
            'mae': [0.4, 0.3, 0.2, 0.1],
            'val_mae': [0.5, 0.4, 0.3, 0.2]
        }
        
        # Test without save path
        plot_training_history(history)
        
        # Check that show was called
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()
        
        # Reset mocks
        mock_show.reset_mock()
        mock_savefig.reset_mock()
        
        # Test with save path
        save_path = os.path.join(self.test_dir, 'history_plot.png')
        plot_training_history(history, save_path)
        
        # Check that show and savefig were called
        mock_show.assert_called_once()
        mock_savefig.assert_called_once_with(save_path)
        
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_predictions(self, mock_savefig, mock_show):
        """Test prediction plotting."""
        # Create test data
        y_true = np.random.random(100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)  # Add some noise
        
        # Test without save path
        plot_predictions(y_true, y_pred)
        
        # Check that show was called
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()
        
        # Reset mocks
        mock_show.reset_mock()
        mock_savefig.reset_mock()
        
        # Test with save path
        save_path = os.path.join(self.test_dir, 'predictions_plot.png')
        plot_predictions(y_true, y_pred, save_path=save_path)
        
        # Check that show and savefig were called
        mock_show.assert_called_once()
        mock_savefig.assert_called_once_with(save_path)
        
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_importance(self, mock_savefig, mock_show):
        """Test feature importance plotting."""
        # Create test data
        feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
        importances = np.array([0.1, 0.3, 0.2, 0.05, 0.35])
        
        # Test without save path
        plot_feature_importance(feature_names, importances)
        
        # Check that show was called
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()
        
        # Reset mocks
        mock_show.reset_mock()
        mock_savefig.reset_mock()
        
        # Test with save path
        save_path = os.path.join(self.test_dir, 'feature_importance_plot.png')
        plot_feature_importance(feature_names, importances, save_path=save_path)
        
        # Check that show and savefig were called
        mock_show.assert_called_once()
        mock_savefig.assert_called_once_with(save_path)
        
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_correlation_matrix(self, mock_savefig, mock_show):
        """Test correlation matrix plotting."""
        # Create test data as numpy array
        data_np = np.random.random((100, 5))
        
        # Test without feature names or save path
        plot_correlation_matrix(data_np)
        
        # Check that show was called
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()
        
        # Reset mocks
        mock_show.reset_mock()
        mock_savefig.reset_mock()
        
        # Test with feature names and save path
        feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
        save_path = os.path.join(self.test_dir, 'correlation_matrix_plot.png')
        plot_correlation_matrix(data_np, feature_names, save_path=save_path)
        
        # Check that show and savefig were called
        mock_show.assert_called_once()
        mock_savefig.assert_called_once_with(save_path)
        
        # Reset mocks
        mock_show.reset_mock()
        mock_savefig.reset_mock()
        
        # Test with pandas DataFrame
        data_df = pd.DataFrame(data_np, columns=feature_names)
        plot_correlation_matrix(data_df)
        
        # Check that show was called
        mock_show.assert_called_once()
        
    def test_create_sequences(self):
        """Test sequence creation."""
        # Test with default parameters
        sequence_length = 10
        target_col_idx = 0
        
        # NumPy array input
        X, y = create_sequences(self.data, sequence_length, target_col_idx)
        
        # Check shapes
        self.assertEqual(X.shape[0], 90)  # 100 - 10 = 90 sequences
        self.assertEqual(X.shape[1], 10)  # sequence_length = 10
        self.assertEqual(X.shape[2], 5)   # 5 features
        self.assertEqual(y.shape[0], 90)  # 90 targets (one per sequence)
        self.assertEqual(y.shape[1], 1)   # target shape should be (n_samples, 1)
        
        # Tensor input
        X, y = create_sequences(self.data_tensor, sequence_length, target_col_idx)
        
        # Check shapes
        self.assertEqual(X.shape[0], 90)
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(X.shape[2], 5)
        self.assertEqual(y.shape[0], 90)
        self.assertEqual(y.shape[1], 1)   # target shape should be (n_samples, 1)
        
        # Check type
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        
        # Test with different target column
        X, y = create_sequences(self.data, sequence_length, target_col_idx=2)
        self.assertEqual(y.shape[0], 90)
        self.assertEqual(y.shape[1], 1)  # target shape should be (n_samples, 1)
        
        # Test with invalid sequence length
        with self.assertRaises(ValueError):
            create_sequences(self.data, 101)  # sequence_length > data length
    
    def test_create_data_loaders(self):
        """Test creating data loaders."""
        # Create tensors
        X = torch.randn(100, 10, 5)  # 100 samples, 10 time steps, 5 features
        y = torch.randn(100, 1)      # 100 samples, 1 target
        
        # Test with validation split
        train_loader, val_loader = create_data_loaders(X, y, batch_size=32, val_split=0.2)
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Check batch sizes
        for batch_x, batch_y in train_loader:
            self.assertLessEqual(batch_x.shape[0], 32)
            break
        
        for batch_x, batch_y in val_loader:
            self.assertLessEqual(batch_x.shape[0], 32)
            break
        
        # Test without validation split
        train_loader, val_loader = create_data_loaders(X, y, batch_size=32, val_split=0)
        
        self.assertIsNotNone(train_loader)
        self.assertIsNone(val_loader)
        
        # Test with NumPy arrays
        X_np = X.numpy()
        y_np = y.numpy()
        
        train_loader, val_loader = create_data_loaders(X_np, y_np, batch_size=16, val_split=0.2)
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Check batch sizes
        for batch_x, batch_y in train_loader:
            self.assertLessEqual(batch_x.shape[0], 16)
            self.assertIsInstance(batch_x, torch.Tensor)
            self.assertIsInstance(batch_y, torch.Tensor)
            break
        
    def test_load_yaml_config(self):
        """Test YAML configuration loading."""
        # Load config
        config = load_yaml_config(self.config_path)
        
        # Check loaded config
        self.assertEqual(config['test_param'], 'test_value')
        self.assertEqual(config['nested']['param1'], 1)
        self.assertEqual(config['nested']['param2'], 'two')
        
        # Test with nonexistent file
        with self.assertRaises(FileNotFoundError):
            load_yaml_config(os.path.join(self.test_dir, 'nonexistent.yaml'))


if __name__ == '__main__':
    unittest.main() 