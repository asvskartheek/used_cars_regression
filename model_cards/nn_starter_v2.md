# Model Card: Neural Network
## VERSION: 2
## OVERALL RMSE: 72,905.41

## Training Data
- Source: Kaggle Playground Series - S4E9
- Size: 188,533 training samples, 125,690 test samples
- Features: 13 columns including both categorical and numerical data

### Features
1. Categorical (13): brand, model, model_year, fuel_type, transmission, ext_col, int_col, accident, clean_title, engine_turbo, engine_flexfuel, engine_hybrid, electric
2. Numerical (4): milage, engine_hp, engine_cc, engine_cyl
3. Target Variable: price

## Data Preprocessing

### Numerical Features
- Standardization: All numerical features are standardized (mean=0, std=1)
- Missing values are filled with the mean

### Categorical Features
- Label Encoding: All categorical features are label encoded
- Rare categories (frequency < 40) are grouped together
- Embedding sizes are calculated for each categorical feature

### Engine Feature Processing
- The 'engine' feature is decoded into multiple features: engine_hp, engine_cc, engine_cyl, engine_turbo, engine_flexfuel, engine_hybrid, electric

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
- Overall CV RMSE: 72,905.41
- Individual fold RMSEs:
  1. 68,209.18
  2. 68,877.68
  3. 73,975.63
  4. 76,647.44
  5. 76,368.67

## Observations
- The model's performance is slightly worse than the previous version (72,813.14 vs 72,905.41 RMSE).
- The performance variation across folds is still significant, indicating potential instability or sensitivity to data distribution.
- The additional processing of the 'engine' feature and inclusion of new categorical features did not lead to improved performance in this iteration.

## Changes from Version 1
- Decoded 'engine' feature into multiple new features
- Added new categorical features: engine_turbo, engine_flexfuel, engine_hybrid, electric
- Increased the number of numerical features from 1 to 4

## References:
- https://www.kaggle.com/code/cdeotte/nn-starter-lb-72300-cv-72800