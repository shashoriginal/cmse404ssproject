"""
Unit tests for data loading utilities.
Author: rajshash

This module provides unit tests for the data loading utilities used in financial indices prediction.
"""

import os
import sys
import unittest
import tempfile
import yaml
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.loader import load_data, prepare_data, normalize_data, create_sequences, load_and_prepare_data


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading utilities."""
    
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
            }
        }
        
        # Save config to temporary file
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Create a small sample DataFrame
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 102,
            'Low': np.random.randn(100) + 98,
            'Close': np.random.randn(100) + 101,
            'Volume': np.random.randint(1000, 10000, size=100)
        })
        data.set_index('Date', inplace=True)
        
        # Save the DataFrame to a CSV file
        self.data_file = os.path.join(self.temp_dir, 'sample_data.csv')
        data.to_csv(self.data_file)
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_data(self):
        """Test loading data from a CSV file."""
        # Test with a real file
        df = load_data(self.data_file, index_col='Date')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 5)
        self.assertTrue(pd.api.types.is_datetime64_dtype(df.index))
        
        # Test with nonexistent file (should raise FileNotFoundError)
        with self.assertRaises(FileNotFoundError):
            load_data('nonexistent_file.csv')
    
    def test_prepare_data(self):
        """Test preparing data for model training."""
        # Load the test data
        df = load_data(self.data_file, index_col='Date')
        
        # Prepare the data with split
        X_train, X_test, y_train, y_test = prepare_data(
            df, target_col='Close', test_size=0.2
        )
        
        # Check the shapes
        self.assertEqual(X_train.shape[0], 80)  # 80% of data
        self.assertEqual(X_test.shape[0], 20)   # 20% of data
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)
        
        # Check that X has all columns except target
        self.assertEqual(X_train.shape[1], df.shape[1] - 1)
        
        # Test with different target column
        X_train, X_test, y_train, y_test = prepare_data(
            df, target_col='Volume', test_size=0.2
        )
        self.assertEqual(X_train.shape[1], df.shape[1] - 1)
        
        # Test without test split
        X_train, X_test, y_train, y_test = prepare_data(
            df, target_col='Close', test_size=0
        )
        self.assertEqual(X_train.shape[0], 100)  # All data
        self.assertEqual(X_test, None)
        self.assertEqual(y_test, None)
    
    def test_normalize_data(self):
        """Test normalizing data."""
        # Create sample data
        data = np.random.randn(100, 5)
        
        # Normalize the data
        normalized_data, scaler = normalize_data(data)
        
        # Check the shape
        self.assertEqual(normalized_data.shape, data.shape)
        
        # Check normalization: values should be between -1 and 1 (or very close due to floating point)
        self.assertTrue(np.all(normalized_data <= 1.0))
        self.assertTrue(np.all(normalized_data >= -1.0))
        
        # Test inverse transform
        restored_data = scaler.inverse_transform(normalized_data)
        np.testing.assert_allclose(restored_data, data, rtol=1e-10)
    
    def test_create_sequences(self):
        """Test creating sequences for time series data."""
        # Create sample data
        data = np.random.randn(100, 5)
        
        # Create sequences
        X, y = create_sequences(data, sequence_length=10, target_col_idx=0)
        
        # Check shapes
        self.assertEqual(X.shape[0], 90)  # 100 - 10 = 90 sequences
        self.assertEqual(X.shape[1], 10)  # 10 time steps
        self.assertEqual(X.shape[2], 5)   # 5 features
        self.assertEqual(y.shape[0], 90)
        
        # Test with tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)
        X, y = create_sequences(data_tensor, sequence_length=10, target_col_idx=0)
        self.assertEqual(X.shape[0], 90)
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(X.shape[2], 5)
        self.assertEqual(y.shape[0], 90)
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
    
    @patch('pandas.read_csv')
    @patch('src.data.loader.prepare_data')
    @patch('src.data.loader.normalize_data')
    @patch('src.data.loader.create_sequences')
    def test_load_and_prepare_data(self, mock_create_sequences, mock_normalize_data, 
                                mock_prepare_data, mock_read_csv):
        """Test the complete data loading and preparation pipeline."""
        # Set up mocks
        mock_df = pd.DataFrame({
            'Date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
            'Open': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 102,
            'Low': np.random.randn(100) + 98,
            'Close': np.random.randn(100) + 101,
            'Volume': np.random.randint(1000, 10000, size=100)
        })
        mock_read_csv.return_value = mock_df
        
        # Mock prepare_data
        X_train = np.random.randn(80, 4)
        X_test = np.random.randn(20, 4)
        y_train = np.random.randn(80, 1)
        y_test = np.random.randn(20, 1)
        mock_prepare_data.return_value = (X_train, X_test, y_train, y_test)
        
        # Mock normalize_data
        norm_X_train = np.random.randn(80, 4)
        norm_X_test = np.random.randn(20, 4)
        mock_scaler = MagicMock()
        mock_normalize_data.side_effect = [
            (norm_X_train, mock_scaler),
            (norm_X_test, mock_scaler)
        ]
        
        # Mock create_sequences
        seq_X_train = torch.randn(70, 10, 4)
        seq_y_train = torch.randn(70, 1)
        seq_X_test = torch.randn(10, 10, 4)
        seq_y_test = torch.randn(10, 1)
        mock_create_sequences.side_effect = [
            (seq_X_train, seq_y_train),
            (seq_X_test, seq_y_test)
        ]
        
        # Call the function with config path
        result = load_and_prepare_data(config_path=self.config_path)
        
        # Check the calls
        mock_read_csv.assert_called()
        mock_prepare_data.assert_called()
        self.assertEqual(mock_normalize_data.call_count, 2)  # Once for train, once for test
        self.assertEqual(mock_create_sequences.call_count, 2)  # Once for train, once for test
        
        # Check the result structure
        train_data, val_data, test_data = result
        self.assertEqual(len(train_data), 2)  # X_train, y_train
        self.assertEqual(len(test_data), 2)   # X_test, y_test
        
        # Check the individual tensors
        X_train, y_train = train_data
        X_test, y_test = test_data
        self.assertIsInstance(X_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertIsInstance(X_test, torch.Tensor)
        self.assertIsInstance(y_test, torch.Tensor)


if __name__ == '__main__':
    unittest.main() 