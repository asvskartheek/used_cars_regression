import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import re
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings("ignore")

def get_inference_stats():
    stats = {}

    train_df = pd.read_csv('data/train.csv')
    
    # Store statistics in dictionaries
    avg_milage_for_age_dict = train_enriched.groupby('age')['avg_milage_for_age'].mean().to_dict()
    avg_annual_milage_for_age_dict = train_enriched.groupby('age')['avg_annual_milage_for_age'].mean().to_dict()

    return avg_milage_for_age_dict, avg_annual_milage_for_age_dict

def calculate_vehicle_age_features(df):
    current_year = 2024
    df['age'] = current_year - df['model_year']
    df['annual_milage'] = df['milage'] / df['age']
    
    df['avg_milage_for_age'] = df.groupby('age')['milage'].transform('mean')
    df['avg_annual_milage_for_age'] = df.groupby('age')['annual_milage'].transform('mean')
    
    return df

def identify_luxury_brands(df):
    luxury_brands = [
        'Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land Rover', 
        'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
        'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston Martin', 'Maybach'
    ]
    df['is_luxury'] = df['brand'].isin(luxury_brands).astype(int)
    
    return df

def enrich_dataset(df):
    df = calculate_vehicle_age_features(df)
    df = identify_luxury_brands(df)
    return df

def preprocess_categorical_features(dataframe, threshold=100):
    categorical_columns = [
        'brand', 'model', 'fuel_type', 'engine', 'transmission',
        'ext_col', 'int_col', 'accident', 'clean_title'
    ]
    columns_to_reduce = ['model', 'engine', 'transmission', 'ext_col', 'int_col']
    
    for column in columns_to_reduce:
        mask = dataframe[column].value_counts(dropna=False)[dataframe[column]].values < threshold
        dataframe.loc[mask, column] = "noise"
        
    for column in categorical_columns:
        dataframe[column] = dataframe[column].fillna('missing')
        dataframe[column] = dataframe[column].astype('category')
        
    return dataframe

def train_model(X_train, y_train, X_val, y_val, model_type='LGBM', objective='MAE', cat_cols=None):
    if model_type == 'LGBM':
        params = {
            'objective': objective,
            'n_estimators': 1000,
            'random_state': 1,
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        callbacks = [lgb.log_evaluation(period=300), lgb.early_stopping(stopping_rounds=200)]
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks    
        )
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    elif model_type == 'CAT':
        params = {
            'loss_function': objective,
            'iterations': 1000,
            'random_seed': 1,
            'early_stopping_rounds': 200
        }
        train_data = Pool(data=X_train, label=y_train, cat_features=cat_cols)
        val_data = Pool(data=X_val, label=y_val, cat_features=cat_cols)
        
        model = CatBoostRegressor(**params)
        model.fit(train_data, eval_set=val_data, verbose=150)
        
        val_pred = model.predict(X_val)
    
    else:
        raise ValueError("Invalid model_type. Choose either 'LGBM' or 'CAT'.")
    
    return model, val_pred

def cross_validate_and_predict(X_features, y_target, test_data, model_type='LGBM', objective='MAE'):
    cat_cols = X_features.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns: {cat_cols}")
    
    oof_predictions = np.zeros(len(X_features))
    test_predictions = np.zeros(len(test_data))
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_features)):
        print(f"Training fold {fold + 1}/5 with {model_type}")

        X_train, X_val = X_features.iloc[train_idx], X_features.iloc[val_idx]
        y_train, y_val = y_target.iloc[train_idx], y_target.iloc[val_idx]

        model, val_pred = train_model(X_train, y_train, X_val, y_val, model_type, objective, cat_cols)
        
        if model_type == 'LGBM':
            test_pred = model.predict(test_data, num_iteration=model.best_iteration)
        elif model_type == 'CAT':
            test_pred = model.predict(test_data)
        else:
            raise ValueError("Invalid model_type. Choose either 'LGBM' or 'CAT'.")
        
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        rmse_scores.append(rmse)

        print(f'{model_type} Fold RMSE: {rmse}')
        
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / 5
    
    print(f'Mean RMSE: {np.mean(rmse_scores)}')
    return oof_predictions, test_predictions, model

def get_outlierness(X_features, y_target, test_data, model_type='LGBM'):
    # Train and predict using MAE objective
    oof_mae, test_mae, mae_model = cross_validate_and_predict(X_features, y_target, test_data, model_type, 'MAE')
    X_features['MAE_pred'] = oof_mae
    test_data['MAE_pred'] = test_mae

    # Train and predict using MSE objective
    oof_mse, test_mse, mse_model = cross_validate_and_predict(X_features, y_target, test_data, model_type, 'MSE')

    # Calculate the difference between MSE and MAE predictions (outlierness)
    X_features['outlierness'] = oof_mse - X_features['MAE_pred']
    test_data['outlierness'] = test_mse - test_data['MAE_pred']

    return X_features, test_data, mae_model, mse_model

def main():
    # Load data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Enrich datasets
    train_enriched = enrich_dataset(train)
    test_enriched = enrich_dataset(test)

    # Preprocess categorical features
    train_processed = preprocess_categorical_features(train_enriched)
    test_processed = preprocess_categorical_features(test_enriched)

    X_features = train_processed.drop('price', axis=1)
    y_target = train_processed['price']

    # Get outlierness
    X_features_with_outlierness, test_processed_with_outlierness, mae_model, mse_model = get_outlierness(X_features, y_target, test_processed, 'LGBM')

    # Train AutoGluon model
    X_features_with_outlierness['price'] = y_target
    predictor = TabularPredictor(
        label='price',
        eval_metric='rmse',
        problem_type='regression'
    ).fit(
        X_features_with_outlierness,
        presets='best_quality',
        time_limit=600,
        verbosity=2,
        num_gpus=0,
        included_model_types=['GBM', 'CAT']
    )

    # Save models
    predictor.save("AutogluonModels/ag_model")
    with open('models/mae_model.pkl', 'wb') as f:
        pickle.dump(mae_model, f)
    with open('models/mse_model.pkl', 'wb') as f:
        pickle.dump(mse_model, f)

    # Print statistics
    print("\nTraining Statistics:")
    print(f"Mean milage: {train['milage'].mean()}")
    print(f"Mean age: {train_processed['age'].mean()}")
    print(f"Mean annual_milage: {train_processed['annual_milage'].mean()}")
    print(f"Mean avg_milage_for_age: {train_processed['avg_milage_for_age'].mean()}")
    print(f"Mean avg_annual_milage_for_age: {train_processed['avg_annual_milage_for_age'].mean()}")
    print(f"Luxury brand ratio: {train_processed['is_luxury'].mean()}")

if __name__ == "__main__":
    main()