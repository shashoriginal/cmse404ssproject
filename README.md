# Financial Indices Prediction Using Neural Networks

🚧 **Under Development** 🚧

**CMSE 404 Project - Michigan State University**

This project implements a multi-layer neural network approach for predicting and analyzing financial indices, with a focus on the S&P 500.

## Project Structure

```
CMSE_404_Project/
│
├── backend/                 # Backend services
│   ├── data/               # Data directory
│   │   ├── raw/           # Raw data from yfinance
│   │   ├── processed/     # Processed and cleaned data
│   │   └── features/      # Engineered features
│   │
│   ├── src/               # Source code
│   │   ├── data/          # Data processing scripts
│   │   │   ├── __init__.py
│   │   │   ├── download.py    # Data download scripts
│   │   │   ├── preprocess.py  # Data preprocessing
│   │   │   └── features.py    # Feature engineering
│   │   │
│   │   ├── models/        # Model implementations
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py  # Base model class
│   │   │   ├── lstm.py        # LSTM implementation
│   │   │   └── training.py    # Training utilities
│   │   │
│   │   └── utils/        # Utility functions
│   │       ├── __init__.py
│   │       └── helpers.py
│   │
│   ├── tests/            # Unit tests
│   │   ├── __init__.py
│   │   ├── test_data.py
│   │   └── test_models.py
│   │
│   └── configs/          # Configuration files
│       ├── model_config.yaml
│       └── data_config.yaml
│
├── frontend/             # Frontend application (Vue.js)
│
├── requirements.txt      # Python dependencies
└── LICENSE              # MIT License

```

## Neural Network Architecture

```
[Input Layer]
     │
     │    ┌─────────────────────┐
     └───►│ Feature Processing  │
          │ - Price normalization
          │ - Technical indicators
          │ - Volume metrics    │
          └─────────┬───────────┘
                    │
                    ▼
┌───────────────────────────────┐
│      LSTM Layer 1 (128)       │
│  [Sequence Processing]         │
└───────────────┬───────────────┘
                │
                ▼
        ┌───────────────┐
        │  Dropout (0.3)│
        └───────┬───────┘
                │
                ▼
┌───────────────────────────────┐
│      LSTM Layer 2 (64)        │
│  [Pattern Recognition]         │
└───────────────┬───────────────┘
                │
                ▼
        ┌───────────────┐
        │  Dropout (0.3)│
        └───────┬───────┘
                │
                ▼
┌───────────────────────────────┐
│      Dense Layer (32)         │
│  [Feature Combination]        │
└───────────────┬───────────────┘
                │
                ▼
     ┌────────────────────┐
     │ Output Layer (1)   │
     │ [Price Prediction] │
     └────────────────────┘
```

## Model Components

### Input Features
1. **Price Data**
   - Opening price (normalized)
   - Closing price (normalized)
   - High price (normalized)
   - Low price (normalized)
   - Trading volume (log-transformed)

2. **Technical Indicators**
   - Moving Averages (5, 10, 20, 50 days)
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Volume-weighted average price (VWAP)

3. **Market Context**
   - VIX (Volatility Index)
   - Market breadth indicators
   - Sector performance metrics

### Model Layers
1. **Input Processing**
   - Feature normalization
   - Sequence creation (60-day windows)
   - Missing value handling

2. **LSTM Layers**
   - Layer 1: 128 units (sequence processing)
   - Layer 2: 64 units (pattern recognition)
   - Dropout layers (0.3) for regularization

3. **Dense Layers**
   - Hidden layer: 32 units with ReLU activation
   - Output layer: 1 unit (linear activation)

### Training Configuration
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 100
- Loss function: Mean Squared Error (MSE)
- Validation split: 20%
- Early stopping patience: 10

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CMSE_404_Project.git
cd CMSE_404_Project
```

2. Set up Python environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p backend/data/{raw,processed,features}
```

## Model Training

```bash
# Download data
python backend/src/data/download.py

# Preprocess data
python backend/src/data/preprocess.py

# Train model
python backend/src/models/training.py
```

## Author:

### Shashank Raj
- **Experience**: 
  - Research Assistant at Computational Optimization and Innovation Laboratory (COIN)
  - Developed high-performance algorithms for optimization problems
  - Expertise in neural network architecture design and implementation
- **Current Work**: Leading the development of deep learning models for financial prediction

## Contributing

We follow industry-standard best practices for contributions. Please read these guidelines carefully before making any contributions to the project.

### Branch Strategy

1. **Main Branches**
   - `main`: Production-ready code
   - `develop`: Integration branch for features

2. **Supporting Branches**
   - Feature branches: `feature/*`
   - Bug fix branches: `bugfix/*`
   - Hotfix branches: `hotfix/*`
   - Release branches: `release/*`

### Contribution Workflow

1. **Before Starting Work**
   ```bash
   # Clone the repository (first time only)
   git clone https://github.com/yourusername/CMSE_404_Project.git
   cd CMSE_404_Project

   # Ensure you're on develop branch
   git checkout develop

   # Get latest changes
   git fetch origin
   git pull origin develop

   # Create your feature branch
   git checkout -b feature/your-feature-name
   ```

2. **During Development**
   ```bash
   # Regularly fetch and rebase to stay up to date
   git fetch origin
   git rebase origin/develop

   # Make small, focused commits
   git add <files>
   git commit -m "feat: descriptive message"
   ```

3. **Before Pushing**
   ```bash
   # Fetch latest changes
   git fetch origin
   
   # Rebase your branch
   git rebase origin/develop
   
   # Run tests
   python -m pytest
   
   # Push your changes
   git push origin feature/your-feature-name
   ```

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, missing semi-colons, etc)
- `refactor`: Code changes that neither fix a bug nor add a feature
- `perf`: Performance improvements
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to build process or auxiliary tools

Examples:
```
feat(model): add LSTM layer to neural network
fix(data): handle missing values in preprocessing
docs(readme): update installation instructions
```

### Code Review Process

1. **Before Creating PR**
   - Ensure all tests pass
   - Update documentation if needed
   - Rebase on latest develop branch
   - Resolve any conflicts

2. **Creating Pull Request**
   - Use PR template
   - Link related issues
   - Provide clear description of changes
   - Add relevant labels
   - Request reviews from team members

3. **During Review**
   - Address all review comments
   - Keep PR updated with latest develop branch
   - Squash commits if requested

4. **Merging**
   - Ensure CI/CD checks pass
   - Get required approvals
   - Squash and merge to maintain clean history

### Development Standards

1. **Code Style**
   - Follow PEP 8 for Python code
   - Use meaningful variable and function names
   - Add docstrings for all functions and classes
   - Keep functions focused and small

2. **Testing**
   - Write unit tests for new features
   - Maintain test coverage above 80%
   - Test edge cases and error conditions
   - Update tests when modifying existing features

3. **Documentation**
   - Update README.md when adding features
   - Document API changes
   - Include inline comments for complex logic
   - Keep documentation up to date

### Branch Protection Rules

1. **Protected Branches**
   - `main` and `develop` are protected
   - Direct pushes are not allowed
   - PRs require review approvals
   - Status checks must pass

2. **Merge Requirements**
   - Clean rebase on target branch
   - All discussions resolved
   - CI/CD pipeline successful
   - Required number of approvals

## Acknowledgments

This project is part of the CMSE 404 course at Michigan State University.
