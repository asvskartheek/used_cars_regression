# Model Card: Neural Network for Car Price Prediction
## VERSION: 3
## OVERALL RMSE: 72,813.14

## Changes from Version 1
- Train the model for more epochs with different learning rates
- in v1 : 3 epochs: 1e-2 for 2 epochs, 1e-3 for 1 epoch
- in v3 : 10 epochs: 1e-2 for 3 epochs, 1e-3 for 3 epochs, 1e-4 for 3 epochs, 1e-5 for 1 epoch

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
Same as [v1](nn_starter_v1.md)

### Categorical Features
Same as [v1](nn_starter_v1.md)

### Data Splitting
Same as [v1](nn_starter_v1.md)

## Model Architecture
Same as [v1](nn_starter_v1.md)

## Training Procedure
- Optimizer: Adam
- Learning rate: Custom schedule (1e-2 for 3 epochs, 1e-3 for 3 epochs, 1e-4 for 3 epochs, 1e-5 for 1 epoch)
- Loss function: Mean Squared Error
- Metric: Root Mean Squared Error (RMSE)
- Batch size: 64
- Epochs: 10
- 5-fold cross-validation

## Results
- Overall CV RMSE: 73,095.99
- Individual fold RMSEs:
  1. 76,773.18
  2. 76,518.33
  3. 76,424.09
  4. 76,773.18
  5. 76,518.33


## References:
- https://www.kaggle.com/code/cdeotte/nn-starter-lb-72300-cv-72800