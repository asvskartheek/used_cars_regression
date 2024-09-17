# Model Card: Neural Network for Car Price Prediction
## VERSION: 1
## OVERALL RMSE: 72,813.14

## Training Data
- Source: Kaggle Playground Series - S4E9
- Size: 188,533 training samples, 125,690 test samples
- Features: 13 columns including both categorical and numerical data

### Features
1. Categorical (10): brand, model, model_year, fuel_type, engine, transmission, ext_col, int_col, accident, clean_title
2. Numerical (1): milage
3. Target Variable: price

## Data Preprocessing

### Numerical Features
- Standardization: The 'milage' feature is standardized (mean=0, std=1)
- Missing values are filled with the mean

### Categorical Features
- Label Encoding: All categorical features are label encoded
- Rare categories (frequency < 40) are grouped together
- Embedding sizes are calculated for each categorical feature

### Data Splitting
- 5-fold cross-validation is used
- Train-test split is maintained throughout the process

## Model Architecture
- Input: Two separate inputs for categorical and numerical features
- Categorical features: Embedded and flattened
- Numerical features: Direct input
- Combined features are passed through dense layers:
  - Dense layer (256 units, ReLU activation)
  - Dense layer (256 units, ReLU activation)
  - Dense layer (256 units, ReLU activation)
  - Output layer (1 unit, linear activation)

## Training Procedure
- Optimizer: Adam
- Learning rate: Custom schedule (0.001 for 2 epochs, 0.0001 for 1 epoch)
- Loss function: Mean Squared Error
- Metric: Root Mean Squared Error (RMSE)
- Batch size: 64
- Epochs: 3
- 5-fold cross-validation

## Results
- Overall CV RMSE: 72,813.14
- Individual fold RMSEs:
  1. 68,019.37
  2. 68,775.68
  3. 74,037.45
  4. 76,533.42
  5. 76,244.09

## Observations
- The model's performance varies significantly across folds, indicating potential instability or sensitivity to data distribution.

## References:
- https://www.kaggle.com/code/cdeotte/nn-starter-lb-72300-cv-72800