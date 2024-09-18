# Model Card: FastAI Tabular Learner for Car Price Prediction
## VERSION: 4
## OVERALL RMSE: Approximately 76,060 (based on last fold)

## Training Data
- Source: Kaggle Playground Series - S4E9
- Size: 188,533 training samples
- Features: 13 columns including both categorical and numerical data

### Features
1. Categorical (11): id, brand, model, model_year, fuel_type, engine, transmission, ext_col, int_col, accident, clean_title
2. Numerical (1): milage
3. Target Variable: price

## Data Preprocessing

### Numerical Features
- Normalization: The 'milage' feature is normalized using FastAI's Normalize processor

### Categorical Features
- Categorification: All categorical features are processed using FastAI's Categorify processor
- Missing values are filled using FastAI's FillMissing processor

### Data Splitting
- 5-fold cross-validation is used

## Model Architecture
- FastAI's tabular_learner is used, which creates a neural network tailored for tabular data
- The exact architecture is determined by FastAI based on the input data

## Training Procedure
- Optimizer: AdamW (default in FastAI)
- Learning rate: Determined by FastAI's learning rate finder
- Loss function: Mean Squared Error (default for regression tasks in FastAI)
- Metric: Root Mean Squared Error (RMSE)
- Training method: fit_one_cycle for a maximum of 20 epochs
- Early stopping: Implemented with a patience of 3 epochs
- 5-fold cross-validation

## Results
- Overall performance: Mean RMSE across folds not explicitly calculated
- Last fold RMSE: 76,060.12

## Additional Notes
- The model checkpoints are saved for each fold
- Test predictions are made for each fold and averaged to create the final predictions

## References:
- FastAI documentation: https://docs.fast.ai/
- Kaggle Playground Series - S4E9