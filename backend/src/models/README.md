# Neural Network Models for Financial Indices Prediction

Author: rajshash

This directory contains the neural network model implementations for financial indices prediction, focusing on LSTM (Long Short-Term Memory) networks.

## Overview

The model architecture implements a multi-layer neural network approach for predicting financial indices as described in the project README. It consists of:

1. **Base Model** - A abstract base class that provides common functionality for all models
2. **LSTM Model** - A specific implementation using LSTM layers for time series prediction
3. **Training Utilities** - Helper functions for training and evaluating models

## Directory Structure

```
models/
├── __init__.py
├── base_model.py     # Abstract base class for all models
├── lstm.py           # LSTM model implementation
├── training.py       # Training utilities
└── README.md         # This file
```

## Usage

### Basic Usage

```python
from models.lstm import LSTMModel

# Create the model
model = LSTMModel()

# Build the model with input shape (sequence_length, n_features)
model.build_model((60, 10))

# Train the model
history = model.train(X_train, y_train, X_val, y_val)

# Evaluate the model
metrics = model.evaluate(X_test, y_test)
print(metrics)

# Make predictions
predictions = model.predict(X_test)

# Save and load the model
model.save('my_model.h5')
model.load('my_model.h5')
```

### Using Training Utilities

```python
from models.training import run_experiment

# Run a complete experiment
results = run_experiment(
    data_path='path/to/data.npy',
    config_path='path/to/config.yaml',
    output_dir='path/to/output'
)
```

### Command Line Usage

You can also run training from the command line:

```bash
python -m models.training --data path/to/data.npy --config path/to/config.yaml --output path/to/output
```

## Configuration

The model behavior is controlled by a YAML configuration file, typically located at `backend/configs/model_config.yaml`. The configuration includes:

```yaml
# Model Architecture
model:
  lstm_units: [128, 64]  # Units in each LSTM layer
  dense_units: [32]      # Units in each Dense layer
  dropout_rate: 0.3      # Dropout rate for regularization
  sequence_length: 60    # Length of input sequences
  batch_size: 32         # Batch size for training
  epochs: 100            # Number of epochs for training

# Training Configuration
training:
  optimizer: adam        # Optimizer to use
  learning_rate: 0.001   # Learning rate
  loss: mse              # Loss function
  validation_split: 0.2  # Validation split ratio
  # ... additional training configurations ...
```

## Extending the Models

To create a new model type, extend the `BaseModel` class:

```python
from models.base_model import BaseModel

class MyNewModel(BaseModel):
    def build_model(self, input_shape):
        # Implement your model architecture here
        # ...
        return self.model
```

## Running Tests

To run the tests for the models:

```bash
cd backend
python -m unittest discover -s tests/models
``` 