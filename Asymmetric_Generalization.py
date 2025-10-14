import os
import time
import numpy as np
import pandas as pd
import random
import warnings
from itertools import combinations
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss as _sk_pinball_loss

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.layers import Dense, Input, Flatten, Add, Subtract, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import QuantileRegressor

warnings.filterwarnings("ignore")


def train_status(status):

    if status == "cloud":
        from google.colab import drive
        drive.mount('/content/drive')
        pre_path = "/content/drive/My Drive/OrderFusion/"

    elif status == "local":
        pre_path = os.path.abspath(".") + "/"
        
    if not os.path.exists(pre_path):
        os.makedirs(pre_path)

    return pre_path


def set_random_seed(seed_value):

    # Set random seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)


def quantile_loss(q, name):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    loss.__name__ = f'{name}_label'
    return loss


def lr_schedule(epoch):

    # Decay learning rate every 10 epochs
    initial_lr = 5e-4
    decay_factor = 0.9
    decay_interval = 10
    
    num_decays = epoch // decay_interval
    return initial_lr * (decay_factor ** num_decays)


def HierarchicalQuantileHeadQ50(shared_representation, quantiles):

    # Sort quantiles and find the index of the median
    sorted_quantiles = sorted(quantiles)
    median_index = sorted_quantiles.index(50)

    # Start with the median quantile
    output_median = Dense(1, name='q50_label')(shared_representation)
    outputs = {f'q50_label': output_median}
    
    # Process quantiles above the median
    prev_output = output_median
    for q in sorted_quantiles[median_index + 1:]:
        residual = Dense(1)(shared_representation)
        residual = Lambda(tf.nn.relu)(residual)
        output = Add(name=f'q{q:02}_label')([prev_output, residual])
        outputs[f'q{q:02}_label'] = output
        prev_output = output  

    # Process quantiles below the median in reverse order
    prev_output = output_median
    for q in reversed(sorted_quantiles[:median_index]):
        residual = Dense(1)(shared_representation)
        residual = Lambda(tf.nn.relu)(residual)
        output = Subtract(name=f'q{q:02}_label')([prev_output, residual])
        outputs[f'q{q:02}_label'] = output
        prev_output = output 
    
    return [outputs[f'q{q:02}_label'] for q in quantiles]


def benchmark_MLP(hidden_dim, num_block, input_shape, quantiles):

    model_input = Input(shape=input_shape, name='input')

    # shape: (timesteps * features * sides)
    output = Flatten()(model_input)
    output = Dense(units=int(hidden_dim), activation='swish')(output)

    for _ in range(num_block):
        output = Dense(units=int(hidden_dim), activation='swish')(output)

    shared_representation = output
    model_outputs = HierarchicalQuantileHeadQ50(shared_representation, quantiles)

    model = Model(inputs=model_input, outputs=model_outputs)
    return model


def select_model(target_model, hidden_dim, num_block, input_shape, quantiles):


    if target_model == 'benchmark_MLP':
            model = benchmark_MLP(hidden_dim, num_block, input_shape, quantiles)

    else:
        raise ValueError(f"Unknown target_model: {target_model}")

    return model


def pinball_loss(y_true, y_pred, quantile):
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def compute_quantile_losses(y_true, y_pred_list, quantiles):

    quantile_losses = []
    for q, y_pred in zip(quantiles, y_pred_list):
        loss = pinball_loss(y_true, y_pred, q)
        quantile_losses.append(loss)

    avg_quantile_loss = float(np.mean(quantile_losses))
    return quantile_losses, avg_quantile_loss


def compute_regression_metrics(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def compute_quantile_crossing_rate(y_pred_array):

    n_samples, n_quantiles = y_pred_array.shape
    index_pairs = list(combinations(range(n_quantiles), 2))

    # Count how many pairs are violations for each sample
    violation_counts = np.zeros(n_samples, dtype=int)

    for i, j in index_pairs:
        violations = y_pred_array[:, i] > y_pred_array[:, j]
        violation_counts += violations.astype(int)

    # If a sample has at least one violation, mark it as 1
    sample_has_crossing = (violation_counts > 0).astype(int)
    crossing_rate = sample_has_crossing.mean()
    return crossing_rate


def prepare_df(dataframe):

    # convert time and fill nan
    df = dataframe.copy()
    df['UTC'] = pd.to_datetime(df['Date_DeliveryStart'], utc=True)
    df = df.drop(columns=['Date_DeliveryStart'])
    df.ffill(inplace=True)
    return df


def select_columns(index, resolution, suffix_mapping, prefixes):

    if index in suffix_mapping:

        # create the list of selected columns dynamically
        selected_columns = [
            f"{prefix}_{resolution}_{suffix}" 
            for suffix in suffix_mapping[index]
            for prefix in prefixes
        ]
        return selected_columns
    
    else:
        raise ValueError(f"Unsupported index type: {index}")
    

def read_data_stat(save_path, country, resolution, indice):

    # Load labels
    output = pd.read_pickle(f"{save_path}EPEX_Spot_Orderbook/Label_{resolution}_{country}_{indice}.pkl")
    output = output[['Date_DeliveryStart', f'{indice}']]

    # Load features
    input_buy = pd.read_pickle(f"{save_path}EPEX_Spot_Orderbook/Feature_Buy_{resolution}_{country}_{indice}.pkl")
    input_sell = pd.read_pickle(f"{save_path}EPEX_Spot_Orderbook/Feature_Sell_{resolution}_{country}_{indice}.pkl")
    input_buy = input_buy.rename(columns={col: f"{col}_buy" for col in input_buy.columns if col != "Date_DeliveryStart"})
    input_sell = input_sell.rename(columns={col: f"{col}_sell" for col in input_sell.columns if col != "Date_DeliveryStart"})

    # Merge on Date_DeliveryStart
    input_merge = pd.merge(input_buy, input_sell, on="Date_DeliveryStart", how="outer")

    # Merge features and labels
    merged = pd.merge(input_merge, output, on="Date_DeliveryStart", how="outer")

    # Standardize time zone
    merged['UTC'] = pd.to_datetime(merged['Date_DeliveryStart'], utc=True)
    merged = merged.drop(columns=['Date_DeliveryStart'])
    merged.ffill(inplace=True)
    return merged


def orderbook_split_stat(orderbook_df, train_start_date, split_len, input_col, output_col):

    # define train, val, test lengh
    train_len, val_len, test_len = split_len

    train_start_date_dt = pd.to_datetime(train_start_date)
    train_end_date_dt = train_start_date_dt + pd.DateOffset(months=train_len)
    val_end_date_dt = train_end_date_dt + pd.DateOffset(months=val_len)
    test_end_date_dt = val_end_date_dt + pd.DateOffset(months=test_len)

    train_start_date = train_start_date_dt.strftime('%Y-%m-%d')
    train_end_date = train_end_date_dt.strftime('%Y-%m-%d')
    val_end_date = val_end_date_dt.strftime('%Y-%m-%d')
    test_end_date = test_end_date_dt.strftime('%Y-%m-%d')

    train_df = orderbook_df[(orderbook_df['UTC'] >= train_start_date) & (orderbook_df['UTC'] < train_end_date)]
    val_df = orderbook_df[(orderbook_df['UTC'] >= train_end_date) & (orderbook_df['UTC'] < val_end_date)]
    test_df = orderbook_df[(orderbook_df['UTC'] >= val_end_date) & (orderbook_df['UTC'] < test_end_date)]

    # prepare features and labels
    X_train = train_df[input_col]
    y_train = train_df[output_col]

    X_val = val_df[input_col]
    y_val = val_df[output_col]

    X_test = test_df[input_col]
    y_test = test_df[output_col]

    return X_train, y_train, X_val, y_val, X_test, y_test


def orderbook_scale_stat(X_train, y_train, X_val, y_val, X_test, y_test):
    # Ensure y are DataFrames
    if not isinstance(y_train, pd.DataFrame):
        y_train = pd.DataFrame(y_train)
    if not isinstance(y_val, pd.DataFrame):
        y_val = pd.DataFrame(y_val)
    if not isinstance(y_test, pd.DataFrame):
        y_test = pd.DataFrame(y_test)

    # Scale y
    y_scaler = RobustScaler()
    y_train_scaled = pd.DataFrame(
        y_scaler.fit_transform(y_train),
        columns=y_train.columns,
        index=y_train.index
    )
    y_val_scaled = pd.DataFrame(
        y_scaler.transform(y_val),
        columns=y_val.columns,
        index=y_val.index
    )
    y_test_scaled = pd.DataFrame(
        y_scaler.transform(y_test),
        columns=y_test.columns,
        index=y_test.index
    )

    # Scale X
    x_scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        x_scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        x_scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        x_scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return (
        X_train_scaled, y_train_scaled.squeeze(),
        X_val_scaled, y_val_scaled.squeeze(),
        X_test_scaled, y_test_scaled.squeeze(),
        y_scaler
    )


def compute_quantile_crossing_rate_regressor(y_pred_array):
    n_samples, n_quantiles = y_pred_array.shape
    index_pairs = list(combinations(range(n_quantiles), 2))
    violation_counts = np.zeros(n_samples, dtype=int)
    for i, j in index_pairs:
        violations = y_pred_array[:, i] > y_pred_array[:, j]
        violation_counts += violations.astype(int)
    crossing_rate = (violation_counts > 0).mean()
    return crossing_rate


def evaluate_LQR(X_train, y_train, X_test, y_test, quantiles, y_scaler):

    # Fit and predict for each quantile
    y_pred_list = []
    quantile_losses = []
    rmse_list, mae_list, r2_list = [], [], []
    y_test = y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).ravel()


    start_time = time.time()
    for q in quantiles:
        model = QuantileRegressor(quantile=q, alpha=0)
        model.fit(X_train, y_train)
        y_pred_scaled = model.predict(X_test)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_pred_list.append(y_pred)

        # Metrics
        loss = pinball_loss(y_test, y_pred, q)
        quantile_losses.append(loss)

        rmse, mae, r2 = compute_regression_metrics(y_test, y_pred)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    end_time = time.time()
    inference_time = end_time - start_time

    # Quantile crossing
    y_pred_array = np.column_stack(y_pred_list)
    crossing_rate = compute_quantile_crossing_rate_regressor(y_pred_array)

    # Median metrics
    median_index = quantiles.index(0.5)
    median_rmse = rmse_list[median_index]
    median_mae = mae_list[median_index]
    median_r2 = r2_list[median_index]

    # Print results
    print(f"AQL: {np.mean(quantile_losses):.4f}, AQCR: {crossing_rate * 100:.2f}%")
    print(f"RMSE: {median_rmse:.2f}, MAE: {median_mae:.2f}, R²: {median_r2:.2f} \n")

    return {
        "quantile_losses": quantile_losses,
        "avg_quantile_loss": np.mean(quantile_losses),
        "quantile_crossing_rate": crossing_rate,
        "quantile_rmse_list": rmse_list,
        "quantile_mae_list": mae_list,
        "quantile_r2_list": r2_list,
        "median_rmse": median_rmse,
        "median_mae": median_mae,
        "median_r2": median_r2,
        "inference_time": inference_time,
        "y_test_original": y_test,
        "y_pred_list": y_pred_list
    }


def execute_LQR(data_config, model_config):
    countries, resolutions, indices, save_path, split_len, train_start_date, input_cols = data_config
    target_model, quantiles = model_config

    for country in countries: 
        for resolution in resolutions:
            for indice in indices:
                results_df = []
                for input_col in input_cols:
                    print(f'{country, resolution, indice} | {target_model} | input_col: {input_col}')
                    
                    # Read, split, and scale orderbook
                    orderbook_df = read_data_stat(save_path, country, resolution, indice)
                    X_train, y_train, X_val, y_val, X_test, y_test = orderbook_split_stat(orderbook_df, train_start_date, split_len, input_col, indice)
                    X_train, y_train, X_val, y_val, X_test, y_test, y_scaler = orderbook_scale_stat(X_train, y_train, X_val, y_val, X_test, y_test)
                    results = evaluate_LQR(X_train, y_train, X_test, y_test, quantiles, y_scaler)
                    results_df.append({
                                    'country': country, 'resolution': resolution, 'indice': indice,  'target_model': target_model, 'input_col': input_col,
                                    'avg_q_loss': results['avg_quantile_loss'], 'quantile_crossing': results['quantile_crossing_rate'], 'rmse': results['median_rmse'], 'mae': results['median_mae'], 'r2': results['median_r2'], 'inference_time': results['inference_time'],
                                    'y_test_original': results['y_test_original'], 'y_pred_list': results['y_pred_list']})

                results_df = pd.DataFrame(results_df)
                results_df.to_pickle(f"{save_path}Result/{country}_{resolution}_{indice}_{target_model}.pkl")
                results_df.to_csv(f"{save_path}Result/{country}_{resolution}_{indice}_{target_model}.csv")


def evaluate_MLP(best_model, X_test, y_test, quantiles, y_scaler):

    # Sort quantiles and scale back the true prices
    quantiles = sorted(quantiles)
    y_test_original = y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).ravel()

    # Measure inference time
    start_time = time.time()
    y_pred_scaled_list = best_model.predict(X_test)  # list/array of shape [n_quantiles, n_samples]
    end_time = time.time()
    inference_time = end_time - start_time

    # Scale back each quantile prediction
    y_pred_list = []
    for i, q in enumerate(quantiles):
        pred_rescaled = y_scaler.inverse_transform(y_pred_scaled_list[i].reshape(-1, 1)).ravel()
        y_pred_list.append(pred_rescaled)

    # Compute metrics
    quantile_losses, avg_quant_loss = compute_quantile_losses(y_test_original, y_pred_list, quantiles)
    median_index = quantiles.index(0.5)
    median_predictions = y_pred_list[median_index]
    rmse_median, mae_median, r2_median = compute_regression_metrics(y_test_original, median_predictions)
    y_pred_array = np.column_stack(y_pred_list)
    crossing_rate = compute_quantile_crossing_rate(y_pred_array)

    # Prepare results
    results = {
        'quantile_losses': quantile_losses,            
        'avg_quantile_loss': round(avg_quant_loss, 2), 
        'quantile_crossing_rate': round(crossing_rate * 100, 2),
        'median_quantile_rmse': round(rmse_median, 2),
        'median_quantile_mae': round(mae_median, 2),
        'median_quantile_r2': round(r2_median, 2),
        'inference_time': inference_time,
        'y_test_original': y_test_original,
        'y_pred_list': y_pred_list}

    print(f"AQL: {results['avg_quantile_loss']}, AQCR: {results['quantile_crossing_rate']}, RMSE: {results['median_quantile_rmse']}, MAE: {results['median_quantile_mae']}, R2: {results['median_quantile_r2']}, Inference time: {inference_time}s \n")
    
    return results


def optimize_models_MLP(X_train, y_train, X_val, y_val, exp_setup):

    hidden_dim, num_blocks, epoch, batch_size, save_path, target_model, quantiles = exp_setup
    input_shape = (X_train.shape[1],)
    model = select_model(target_model, hidden_dim, num_blocks, input_shape, quantiles)
    
    # Generate y_train_dict and y_val_dict
    y_train_dict = {f'q{q:02}_label': y_train for q in quantiles}
    y_val_dict = {f'q{q:02}_label': y_val for q in quantiles}
    quantiles_dict = {f'q{q:02}': q / 100 for q in quantiles}

    # Define quantile loss
    quantile_losses = {}
    for name, q in quantiles_dict.items():
        loss_name = f'{name}_label'
        quantile_losses[loss_name] = quantile_loss(q, name)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=quantile_losses)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Count model params
    model_paras_count = model.count_params()
    print(f"paras: {model_paras_count}")

    # Validate model
    checkpoint_path = os.path.join(f"{save_path}Model", f"{target_model}.keras")
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          monitor='val_loss',
                                          save_freq="epoch",
                                          save_best_only=True,
                                          mode='min',
                                          verbose=0)
    
    history = model.fit(X_train, y_train_dict, 
                        epochs=epoch, verbose=0,
                        validation_data=(X_val, y_val_dict),
                        callbacks=[checkpoint_callback, lr_scheduler],
                        batch_size=batch_size)

    # Load the best model with lowest val loss
    custom_objects = {f'{name}_label': quantile_loss(q, name) for name, q in quantiles_dict.items()}
    with custom_object_scope(custom_objects):
        best_model = load_model(checkpoint_path, custom_objects=custom_objects)

    return best_model, history.history



def execute_MLP(data_config, model_config):
    countries, resolutions, indices, save_path, split_len, train_start_date, input_cols = data_config
    target_model, model_shapes, epoch, batch_size, quantiles = model_config

    for country in countries: 
        for resolution in resolutions:
            for indice in indices:
                results_df = []

                for input_col in input_cols:
                    # Read, split, and scale orderbook
                    orderbook_df = read_data_stat(save_path, country, resolution, indice)
                    X_train, y_train, X_val, y_val, X_test, y_test = orderbook_split_stat(orderbook_df, train_start_date, split_len, input_col, indice)
                    X_train, y_train, X_val, y_val, X_test, y_test, y_scaler = orderbook_scale_stat(X_train, y_train, X_val, y_val, X_test, y_test)

                    for model_shape in model_shapes:
                        
                        # Get model depth and width
                        hidden_dim, num_block  = model_shape[0], model_shape[1]


                        print(f'{country, resolution, indice} | {target_model} | {input_col}')
                        
                        # Train, validate, and test model
                        exp_setup = (hidden_dim, num_block, epoch, batch_size, save_path, target_model, [int(q * 100) for q in quantiles])
                        best_model, hist_val, num_para = optimize_models_MLP(X_train, y_train, X_val, y_val, exp_setup)
                        min_val_loss = min(hist_val["val_loss"])
                        results = evaluate_MLP(best_model, X_test, y_test, quantiles, y_scaler)
                        results_df.append({
                            'country': country, 'resolution': resolution, 'indice': indice, 'model_shape': model_shape, 'target_model': target_model, 'num_para': num_para, 'min_val_loss': min_val_loss,  
                            'avg_q_loss': results['avg_quantile_loss'], 'quantile_crossing': results['quantile_crossing_rate'], 'rmse': results['median_quantile_rmse'], 'mae': results['median_quantile_mae'], 'r2': results['median_quantile_r2'], 'inference_time': results['inference_time'],
                            'y_test_original': results['y_test_original'], 'y_pred_list': results['y_pred_list']})

                results_df = pd.DataFrame(results_df)
                results_df.to_pickle(f"{save_path}Result/{country}_{resolution}_{indice}_{target_model}.pkl")
                results_df.to_csv(f"{save_path}Result/{country}_{resolution}_{indice}_{target_model}.csv")


def extract_base_features_from_columns(columns):
    return sorted(set(col.rsplit('_', 1)[0] for col in columns if '_buy' in col or '_sell' in col))

def optuna_objective(trial, optuna_results_log, country, resolution, indice, quantiles, save_path, train_start_date, split_len, epoch, batch_size):

    # Load data
    orderbook_df = read_data_stat(save_path, country, resolution, indice)

    # Filter columns based on selected base features
    base_features = sorted(set(col.rsplit('_', 1)[0] for col in orderbook_df if '_buy' in col or '_sell' in col))

    # Generate feature mask
    feature_mask = [trial.suggest_int(f"f_{i}", 0, 1) for i in range(len(base_features))]

    if sum(feature_mask) < 1:
        return float("inf")  # Skip uninformative subset

    selected_bases = [bf for bf, flag in zip(base_features, feature_mask) if flag == 1]
    
    input_col = []
    for base in selected_bases:
        input_col.extend([f"{base}_buy", f"{base}_sell"])
    
    num_features = len(input_col)
    print(f"#Features: {num_features}")

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = orderbook_split_stat(orderbook_df, train_start_date, split_len, input_col, indice)
    X_train, y_train, X_val, y_val, X_test, y_test, y_scaler = orderbook_scale_stat(X_train, y_train, X_val, y_val, X_test, y_test)


    # Modeling 
    QUANTILES = [int(q * 100) for q in quantiles]
    input_shape = (X_train.shape[1],)
    model = select_model("benchmark_MLP", 128, 3, input_shape, QUANTILES)
    
    # Generate y_train_dict and y_val_dict
    y_train_dict = {f'q{q:02}_label': y_train for q in QUANTILES}
    y_val_dict = {f'q{q:02}_label': y_val for q in QUANTILES}
    quantiles_dict = {f'q{q:02}': q / 100 for q in QUANTILES}

    # Define quantile loss
    quantile_losses = {}
    for name, q in quantiles_dict.items():
        loss_name = f'{name}_label'
        quantile_losses[loss_name] = quantile_loss(q, name)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=quantile_losses)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Count model params
    model_paras_count = model.count_params()
    #print(f"paras: {model_paras_count}")

    # Validate model
    checkpoint_path = os.path.join(f"{save_path}Model", f"benchmark_MLP.keras")
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          monitor='val_loss',
                                          save_freq="epoch",
                                          save_best_only=True,
                                          mode='min',
                                          verbose=0)
    
    history = model.fit(X_train, y_train_dict, 
                        epochs=epoch, verbose=0,
                        validation_data=(X_val, y_val_dict),
                        callbacks=[checkpoint_callback, lr_scheduler],
                        batch_size=batch_size)

    # Load the best model with lowest val loss
    custom_objects = {f'{name}_label': quantile_loss(q, name) for name, q in quantiles_dict.items()}
    with custom_object_scope(custom_objects):
        best_model = load_model(checkpoint_path, custom_objects=custom_objects)

    results = evaluate_MLP(best_model, X_test, y_test, quantiles, y_scaler)

    # save results
    optuna_results_log.append({
    'trial': trial.number,
    'features': input_col, 
    'num_features': num_features,
    'val_loss': float(min(model.history.history['val_loss'])),
    'avg_q_loss': results['avg_quantile_loss'],
    'quantile_crossing': results['quantile_crossing_rate'],
    'rmse': results['median_quantile_rmse'],
    'mae': results['median_quantile_mae'],
    'r2': results['median_quantile_r2'],
    'inference_time': results['inference_time'],
    #'y_test_original': results['y_test_original'],
    #'y_pred_list': results['y_pred_list']
})
    
    return float(min(model.history.history['val_loss']))


def prefilter_orderbook_features(orderbook_df):
    # Get all columns
    columns = orderbook_df.columns

    # Initialize keep list
    keep_cols = []
    feature_cols = []
    for col in columns:
        if col in ["Date_DeliveryStart", "UTC"]:  # always keep time columns
            keep_cols.append(col)
            #feature_cols.append(col)
            continue
        
        if col.startswith("ID"):  # label column
            keep_cols.append(col)
            continue
        
        if "_buy" in col or "_sell" in col:
            base = col.rsplit('_', 1)[0]  # e.g. "LastP_full"

            # Rule: drop FirstP/FirstV completely
            #if base.startswith("FirstP") or base.startswith("FirstV"):
            #    continue

            # Rule: keep LastP_full and LastV_full only
            if base.startswith("LastP") or base.startswith("LastV"):
                if "_full" in col:
                    keep_cols.append(col)
                    feature_cols.append(col)
                continue

            # All others: keep
            keep_cols.append(col)
            feature_cols.append(col)

    return orderbook_df[keep_cols], feature_cols


def optuna_objective(trial, optuna_results_log, country, resolution, indice, quantiles, save_path, train_start_date, split_len, epoch, batch_size):

    # Load data
    orderbook_df = read_data_stat(save_path, country, resolution, indice)
    orderbook_df = prefilter_orderbook_features(orderbook_df)

    # Filter columns based on selected base features
    base_features = sorted(set(col.rsplit('_', 1)[0] for col in orderbook_df if '_buy' in col or '_sell' in col))

    # Generate feature mask
    if trial.number < 100:
        desired_k = 1
    elif trial.number < 200:
        desired_k = 2
    elif trial.number < 300:
        desired_k = 4
    elif trial.number < 400:
        desired_k = 8
    else:
        desired_k = 16

    # Calculate inclusion probability
    p_include = desired_k / len(base_features)
    feature_mask = [1 if trial.suggest_float(f"f_{i}", 0.0, 1.0) < p_include else 0
        for i in range(len(base_features))]
    
    if sum(feature_mask) < 1:
        return float("inf")  # Skip uninformative subset

    selected_bases = [bf for bf, flag in zip(base_features, feature_mask) if flag == 1]
    
    input_col = []
    for base in selected_bases:
        input_col.extend([f"{base}_buy", f"{base}_sell"])
    
    num_features = len(input_col)
    print(f"#Features={num_features} - {selected_bases} (buy & sell)")

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = orderbook_split_stat(orderbook_df, train_start_date, split_len, input_col, indice)
    X_train, y_train, X_val, y_val, X_test, y_test, y_scaler = orderbook_scale_stat(X_train, y_train, X_val, y_val, X_test, y_test)


    # Modeling 
    QUANTILES = [int(q * 100) for q in quantiles]
    input_shape = (X_train.shape[1],)
    model = select_model("benchmark_MLP", 256, 3, input_shape, QUANTILES)
    
    # Generate y_train_dict and y_val_dict
    y_train_dict = {f'q{q:02}_label': y_train for q in QUANTILES}
    y_val_dict = {f'q{q:02}_label': y_val for q in QUANTILES}
    quantiles_dict = {f'q{q:02}': q / 100 for q in QUANTILES}

    # Define quantile loss
    quantile_losses = {}
    for name, q in quantiles_dict.items():
        loss_name = f'{name}_label'
        quantile_losses[loss_name] = quantile_loss(q, name)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=quantile_losses)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Count model params
    model_paras_count = model.count_params()
    print(f"paras: {model_paras_count}")

    # Validate model
    checkpoint_path = os.path.join(f"{save_path}Model", f"benchmark_MLP.keras")
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          monitor='val_loss',
                                          save_freq="epoch",
                                          save_best_only=True,
                                          mode='min',
                                          verbose=0)
    
    history = model.fit(X_train, y_train_dict, 
                        epochs=epoch, verbose=0,
                        validation_data=(X_val, y_val_dict),
                        callbacks=[checkpoint_callback, lr_scheduler],
                        batch_size=batch_size)

    # Load the best model with lowest val loss
    custom_objects = {f'{name}_label': quantile_loss(q, name) for name, q in quantiles_dict.items()}
    with custom_object_scope(custom_objects):
        best_model = load_model(checkpoint_path, custom_objects=custom_objects)

    results = evaluate_MLP(best_model, X_test, y_test, quantiles, y_scaler)

    # save results
    optuna_results_log.append({
    'trial': trial.number,
    'features': input_col, 
    'num_features': num_features,
    'val_loss': float(min(model.history.history['val_loss'])),
    'avg_q_loss': results['avg_quantile_loss'],
    'quantile_crossing': results['quantile_crossing_rate'],
    'rmse': results['median_quantile_rmse'],
    'mae': results['median_quantile_mae'],
    'r2': results['median_quantile_r2'],
    'inference_time': results['inference_time'],
    #'y_test_original': results['y_test_original'],
    #'y_pred_list': results['y_pred_list']
})
    
    return float(min(model.history.history['val_loss']))




# LQR + L1 regularization
def optimize_linear_quantile_regression(X_train, y_train, X_val, y_val, quantiles, alphas):
    from sklearn.linear_model import QuantileRegressor
    from sklearn.metrics import mean_pinball_loss
    
    results = []
    for q in quantiles:
        for alpha in alphas:
            model = QuantileRegressor(quantile=q, alpha=alpha)
            model.fit(X_train, y_train)

            # Predict on validation set
            y_val_pred = model.predict(X_val)
            val_loss = mean_pinball_loss(y_val, y_val_pred, alpha=q)

            # Get non-zero coefficient feature names
            coef_mask = model.coef_ != 0
            features = list(X_train.columns[coef_mask])

            print(f"  Quantile: {q}, Alpha: {alpha}, Validation Loss: {val_loss}, num_features: {len(features)}")
            
            # Append to results
            results.append({
                "quantile": q,
                "alpha": alpha,
                "val_loss": val_loss,
                "num_features": len(features),
                "features": features,
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df



def get_features(country, resolution, selection, quantile):
    if selection == "15min-VWAP":
        return ['VWAP_15_buy', 'VWAP_15_sell']

    elif selection == "last_P":
        return ['LastP_full_buy', 'LastP_full_sell']
    

    elif selection == 'filtered_all':
        if quantile == 0.1: 
            if country == 'germany' and resolution == 'h':
                return ['MaxP_full_sell', 'FirstP_1_sell', 'MinP_full_buy', 'MinP_60_buy', 'MinP_15_buy']
            elif country == 'germany' and resolution == 'qh':
                return ['MeanP_5_sell', 'MaxP_1_sell', 'MinP_180_buy', 'TradeCount_180_buy', 'MinP_60_buy', 'MinP_15_buy']
            elif country == 'austria' and resolution == 'h':
                return ['PriceVolatility_full_sell', 'DeltaP_full_sell', 'MinV_full_sell', 'FirstP_15_sell', 'FirstP_5_sell', 'MinP_5_sell', 'PctlP_45_5_sell', 'FirstP_1_sell', 'MaxP_1_sell', 'MinP_full_buy', 'MinV_full_buy', 'MinP_180_buy', 'PriceVolatility_60_buy', 'MinP_15_buy']
            elif country == 'austria' and resolution == 'qh':
                return ['PctlP_10_full_sell', 'MinP_180_sell', 'PctlP_10_180_sell', 'MinP_5_sell', 'MinP_1_sell', 'MinP_180_buy', 'MinP_15_buy', 'MinP_5_buy', 'PctlP_10_full_buy', 'PctlP_10_180_buy', 'PctlP_10_60_buy']
        
        elif quantile == 0.5:
            if country == 'germany' and resolution == 'h':
                return ['PctlP_75_15_sell', 'MaxP_5_sell', 'PctlP_90_5_sell', 'PctlP_25_60_buy', 'MinP_15_buy', 'PctlP_10_15_buy', 'PctlP_10_5_buy', 'PctlP_25_5_buy']
            elif country == 'germany' and resolution == 'qh':
                return ['PctlP_45_60_sell', 'PctlP_75_15_sell', 'PctlP_90_15_sell', 'MaxP_5_sell', 'MaxP_1_sell', 'PctlP_90_1_sell', 'PctlP_10_60_buy', 'PctlP_25_60_buy', 'MinP_15_buy', 'MinP_1_buy']
            elif country == 'austria' and resolution == 'h':
                return ['LastP_full_sell', 'PctlP_75_full_sell', 'FirstP_60_sell', 'FirstP_5_sell', 'PctlP_45_5_sell', 'FirstP_1_sell', 'MaxP_1_sell', 'FirstP_full_buy', 'PctlP_45_full_buy', 'LastP_full_buy', 'MinP_60_buy', 'MinP_15_buy', 'MinP_5_buy']
            elif country == 'austria' and resolution == 'qh':
                return ['LastP_full_sell', 'MeanP_full_sell', 'VWAP_180_sell', 'MaxP_15_sell', 'FirstP_1_sell', 'MeanP_1_sell', 'VWAP_1_sell', 'FirstP_full_buy', 'MeanP_full_buy', 'MeanP_60_buy', 'LastP_full_buy', 'MeanP_15_buy', 'MinP_15_buy', 'VWAP_1_buy', 'PctlP_10_5_buy']

        elif quantile == 0.9:
            if country == 'germany' and resolution == 'h':
                return ['MaxP_full_sell', 'PctlP_25_full_sell', 'PctlP_90_full_sell', 'MaxP_180_sell', 'MaxP_60_sell', 'MaxP_15_sell', 'PctlP_90_15_sell', 'MaxP_5_sell', 'PctlP_45_5_sell', 'PctlP_90_5_sell', 'PctlP_10_full_buy', 'PctlP_25_full_buy', 'MinP_5_buy', 'MeanV_5_buy', 'PctlP_10_5_buy']
            elif country == 'germany' and resolution == 'qh':
                return ['MaxP_180_sell', 'MaxP_60_sell', 'PctlP_90_60_sell', 'MaxP_15_sell', 'PctlP_90_15_sell', 'PctlP_90_1_sell', 'PctlP_25_15_buy', 'MinP_1_buy']
            elif country == 'austria' and resolution == 'h':
                return ['MinP_full_sell', 'MaxP_full_sell', 'PriceVolatility_full_sell', 'DeltaP_full_sell', 'MinV_full_sell', 'PctlP_25_full_sell', 'PctlP_55_full_sell', 'MinP_180_sell', 'MaxP_180_sell', 'PriceVolatility_180_sell', 'MeanV_180_sell', 'FirstP_60_sell', 'MinP_1_sell', 'MaxP_1_sell', 'PctlP_10_1_sell', 'FirstP_full_buy', 'MinP_full_buy', 'MaxP_full_buy', 'PctlP_45_full_buy', 'PctlP_75_full_buy', 'PctlP_90_full_buy', 'LastP_full_buy', 'SumV_180_buy', 'PctlP_45_180_buy', 'PctlV_55_180_buy', 'PriceVolatility_60_buy', 'DeltaP_60_buy', 'PctlP_55_60_buy', 'PctlP_75_60_buy', 'FirstP_15_buy', 'MaxP_5_buy', 'FirstV_5_buy', 'PctlP_75_5_buy']
            elif country == 'austria' and resolution == 'qh':
                return ['SumV_full_sell', 'PctlP_90_full_sell', 'MaxP_180_sell', 'LastP_full_sell', 'MaxP_1_sell', 'FirstP_full_buy', 'MaxP_full_buy', 'MaxP_180_buy', 'MaxP_1_buy', 'PctlP_90_full_buy', 'PctlP_90_15_buy']
            


    elif selection == 'top_5':
        if quantile == 0.1: 
            if country == 'germany' and resolution == 'h':
                return ['MinP_15_buy', 'MinP_60_buy', 'MinP_full_buy', 'MaxP_full_sell', 'FirstP_1_sell']
            elif country == 'germany' and resolution == 'qh':
                return ['MinP_60_buy', 'MinP_180_buy', 'MaxP_1_sell', 'MeanP_5_sell', 'MinP_15_buy']
            elif country == 'austria' and resolution == 'h':
                return ['MinP_15_buy', 'MinP_full_buy', 'PctlP_45_5_sell', 'MinP_5_sell', 'FirstP_5_sell']
            elif country == 'austria' and resolution == 'qh':
                return ['MinP_1_sell', 'PctlP_10_180_buy', 'MinP_15_buy', 'PctlP_10_full_buy', 'MinP_180_buy']
        
        elif quantile == 0.5:
            if country == 'germany' and resolution == 'h':
                return ['PctlP_90_5_sell', 'PctlP_10_15_buy', 'PctlP_75_15_sell', 'MinP_15_buy', 'PctlP_10_5_buy']
            elif country == 'germany' and resolution == 'qh':
                return ['PctlP_10_60_buy', 'PctlP_90_15_sell', 'MinP_15_buy', 'PctlP_45_60_sell', 'PctlP_25_60_buy']
            elif country == 'austria' and resolution == 'h':
                return ['MinP_5_buy', 'LastP_full_sell', 'PctlP_45_5_sell', 'LastP_full_buy', 'MaxP_1_sell']
            elif country == 'austria' and resolution == 'qh':
                return ['MeanP_full_buy', 'LastP_full_sell', 'VWAP_1_buy', 'VWAP_180_sell', 'MeanP_15_buy']

        elif quantile == 0.9:
            if country == 'germany' and resolution == 'h':
                return ['MaxP_15_sell', 'PctlP_10_full_buy', 'PctlP_10_5_buy', 'MaxP_60_sell', 'MaxP_180_sell']
            elif country == 'germany' and resolution == 'qh':
                return ['MaxP_60_sell', 'PctlP_90_60_sell', 'MinP_1_buy', 'MaxP_15_sell', 'MaxP_180_sell']
            elif country == 'austria' and resolution == 'h':
                return ['PctlP_75_60_buy', 'MaxP_180_sell', 'MaxP_1_sell', 'FirstP_15_buy', 'PctlP_10_1_sell']
            elif country == 'austria' and resolution == 'qh':
                return ['MaxP_1_buy', 'MaxP_180_buy', 'MaxP_180_sell', 'MaxP_1_sell', 'MaxP_full_buy']
    


    elif selection == 'top_1':
        if quantile == 0.1: 
            if country == 'germany' and resolution == 'h':
                return ['MinP_15_buy']
            elif country == 'germany' and resolution == 'qh':
                return ['MinP_60_buy']
            elif country == 'austria' and resolution == 'h':
                return ['MinP_15_buy']
            elif country == 'austria' and resolution == 'qh':
                return ['MinP_1_sell']
        
        elif quantile == 0.5:
            if country == 'germany' and resolution == 'h':
                return ['PctlP_90_5_sell']
            elif country == 'germany' and resolution == 'qh':
                return ['PctlP_10_60_buy']
            elif country == 'austria' and resolution == 'h':
                return ['MinP_5_buy']
            elif country == 'austria' and resolution == 'qh':
                return ['MeanP_full_buy']

        elif quantile == 0.9:
            if country == 'germany' and resolution == 'h':
                return ['MaxP_15_sell']
            elif country == 'germany' and resolution == 'qh':
                return ['MaxP_60_sell']
            elif country == 'austria' and resolution == 'h':
                return ['PctlP_75_60_buy']
            elif country == 'austria' and resolution == 'qh':
                return ['MaxP_1_buy']
            

    elif selection == 'all':
        if quantile == 0.1: 
            if country == 'germany' and resolution == 'h':
                return ['MinP_full_sell', 'MaxP_full_sell', 'PriceVolatility_full_sell', 'DeltaP_full_sell', 'LastV_full_sell', 'MaxV_full_sell', 'SumV_full_sell', 'TradeCount_full_sell', 'PctlP_10_full_sell', 'PctlV_75_full_sell', 'PctlV_90_full_sell', 'MinP_180_sell', 'PriceVolatility_180_sell', 'DeltaP_180_sell', 'Momentum_180_sell', 'SumV_180_sell', 'PctlV_10_180_sell', 'FirstP_60_sell', 'MinP_60_sell', 'FirstV_60_sell', 'TradeCount_60_sell', 'PctlP_10_60_sell', 'PctlV_10_60_sell', 'PctlV_45_60_sell', 'DeltaP_15_sell', 'TradeCount_15_sell', 'LastP_5_sell', 'MedianP_5_sell', 'MinP_5_sell', 'MinV_5_sell', 'VolumeVolatility_5_sell', 'DeltaV_5_sell', 'Momentum_5_sell', 'PctlP_90_5_sell', 'FirstP_1_sell', 'FirstV_1_sell', 'MaxV_1_sell', 'Momentum_1_sell', 'TradeCount_1_sell', 'MinP_full_buy', 'FirstV_full_buy', 'DeltaP_180_buy', 'MaxV_180_buy', 'PctlV_75_180_buy', 'MinP_60_buy', 'PriceVolatility_60_buy', 'PctlV_75_60_buy', 'MinP_15_buy', 'SumV_15_buy', 'PctlP_10_15_buy', 'PctlV_10_15_buy', 'FirstV_5_buy', 'Momentum_5_buy', 'PctlP_25_5_buy', 'PctlV_10_5_buy', 'PriceVolatility_1_buy', 'DeltaP_1_buy', 'FirstV_1_buy', 'Momentum_1_buy', 'TradeCount_1_buy', 'PctlV_25_1_buy']
            elif country == 'germany' and resolution == 'qh':
                return ['MinP_full_sell', 'TradeCount_180_sell', 'TradeCount_60_sell', 'FirstP_5_sell', 'MeanP_5_sell', 'MaxP_1_sell', 'MaxP_full_buy', 'MinP_180_buy', 'MaxP_180_buy', 'TradeCount_180_buy', 'MinP_60_buy', 'MinP_15_buy', 'MinP_1_buy', 'PctlP_10_1_buy']
            elif country == 'austria' and resolution == 'h':
                return ['PriceVolatility_full_sell', 'DeltaP_full_sell', 'LastV_full_sell', 'MeanV_full_sell', 'MinV_full_sell', 'MaxV_full_sell', 'TradeCount_full_sell', 'PctlP_10_full_sell', 'PctlV_10_full_sell', 'PctlV_25_full_sell', 'PriceVolatility_180_sell', 'DeltaP_180_sell', 'PctlV_75_180_sell', 'PriceVolatility_60_sell', 'MaxV_60_sell', 'Momentum_60_sell', 'PctlP_10_60_sell', 'FirstP_15_sell', 'MinP_15_sell', 'FirstV_15_sell', 'VolumeVolatility_15_sell', 'PctlV_45_15_sell', 'FirstP_5_sell', 'MinP_5_sell', 'PctlP_45_5_sell', 'FirstP_1_sell', 'MaxP_1_sell', 'SumV_1_sell', 'PctlV_90_1_sell', 'MinP_full_buy', 'FirstV_full_buy', 'MinV_full_buy', 'SumV_full_buy', 'TradeCount_full_buy', 'PctlV_45_full_buy', 'MinP_180_buy', 'DeltaP_180_buy', 'MinV_180_buy', 'PriceVolatility_60_buy', 'MaxV_60_buy', 'Momentum_60_buy', 'SumV_60_buy', 'TradeCount_60_buy', 'PctlV_90_60_buy', 'MinP_15_buy', 'FirstV_15_buy', 'DeltaP_5_buy', 'MinV_5_buy', 'VolumeVolatility_5_buy', 'TradeCount_1_buy']
            elif country == 'austria' and resolution == 'qh':
                return ['PctlP_10_full_sell', 'MinP_180_sell', 'PctlP_10_180_sell', 'MinP_5_sell', 'MinP_1_sell', 'TradeCount_full_buy', 'MinP_180_buy', 'MinP_15_buy', 'MinP_5_buy', 'PctlP_10_full_buy', 'PctlP_10_180_buy', 'PctlP_10_60_buy']
        
        elif quantile == 0.5:
            if country == 'germany' and resolution == 'h':
                return ['DeltaP_full_sell', 'SumV_full_sell', 'PctlV_90_full_sell', 'PctlV_25_180_sell', 'FirstV_60_sell', 'Momentum_60_sell', 'PctlV_10_60_sell', 'MedianV_15_sell', 'Momentum_15_sell', 'PctlP_10_15_sell', 'PctlP_25_15_sell', 'PctlP_75_15_sell', 'LastP_5_sell', 'MaxP_5_sell', 'MedianV_5_sell', 'Momentum_5_sell', 'PctlP_90_5_sell', 'PctlV_90_5_sell', 'PctlP_90_1_sell', 'PctlV_45_1_sell', 'MinP_full_buy', 'FirstV_full_buy', 'MaxV_full_buy', 'PctlV_10_full_buy', 'Momentum_180_buy', 'PctlP_75_180_buy', 'PctlV_25_180_buy', 'PctlV_55_180_buy', 'MedianP_60_buy', 'MinP_60_buy', 'MedianV_60_buy', 'PctlP_25_60_buy', 'PctlP_55_60_buy', 'MinP_15_buy', 'PriceVolatility_15_buy', 'MaxV_15_buy', 'PctlP_10_15_buy', 'PctlP_75_15_buy', 'PctlV_10_15_buy', 'PctlV_25_15_buy', 'PriceVolatility_5_buy', 'TradeCount_5_buy', 'PctlP_10_5_buy', 'PctlP_25_5_buy', 'PctlP_75_5_buy', 'PriceVolatility_1_buy', 'DeltaP_1_buy', 'MinV_1_buy', 'DeltaV_1_buy', 'TradeCount_1_buy']
            elif country == 'germany' and resolution == 'qh':
                return ['MinP_full_sell', 'TradeCount_180_sell', 'PctlP_45_60_sell', 'PctlP_75_15_sell', 'PctlP_90_15_sell', 'MaxP_5_sell', 'MaxP_1_sell', 'PctlP_90_1_sell', 'VWAP_180_buy', 'MinP_60_buy', 'PctlP_10_60_buy', 'PctlP_25_60_buy', 'MinP_15_buy', 'PctlP_10_15_buy', 'MinP_1_buy']
            elif country == 'austria' and resolution == 'h':
                return ['LastP_full_sell', 'PctlP_75_full_sell', 'FirstP_60_sell', 'FirstP_5_sell', 'PctlP_45_5_sell', 'FirstP_1_sell', 'MaxP_1_sell', 'FirstP_full_buy', 'PctlP_45_full_buy', 'PctlP_90_full_buy', 'FirstP_180_buy', 'LastP_180_buy', 'PctlP_45_180_buy', 'PctlP_90_180_buy', 'FirstP_60_buy', 'MinP_60_buy', 'MaxP_60_buy', 'MinP_15_buy', 'MinP_5_buy']
            elif country == 'austria' and resolution == 'qh':
                return ['LastP_full_sell', 'MeanP_full_sell', 'PctlP_55_full_sell', 'VWAP_180_sell', 'PctlP_75_60_sell', 'MaxP_15_sell', 'FirstP_1_sell', 'MeanP_1_sell', 'VWAP_1_sell', 'FirstP_full_buy', 'MeanP_full_buy', 'MeanP_60_buy', 'LastP_15_buy', 'MeanP_15_buy', 'MinP_15_buy', 'VWAP_1_buy', 'PctlP_10_5_buy']

        elif quantile == 0.9:
            if country == 'germany' and resolution == 'h':
                return ['FirstP_full_sell', 'MinP_full_sell', 'MaxP_full_sell', 'FirstV_full_sell', 'MedianV_full_sell', 'MeanV_full_sell', 'Momentum_full_sell', 'TradeCount_full_sell', 'PctlP_10_full_sell', 'PctlP_25_full_sell', 'PctlP_90_full_sell', 'PctlV_10_full_sell', 'PctlV_25_full_sell', 'PctlV_45_full_sell', 'PctlV_55_full_sell', 'PctlV_75_full_sell', 'MinP_180_sell', 'MaxP_180_sell', 'PriceVolatility_180_sell', 'DeltaP_180_sell', 'FirstV_180_sell', 'MeanV_180_sell', 'MinV_180_sell', 'VolumeVolatility_180_sell', 'PctlP_90_180_sell', 'PctlV_10_180_sell', 'PctlV_45_180_sell', 'PctlV_90_180_sell', 'FirstP_60_sell', 'MeanP_60_sell', 'MaxP_60_sell', 'FirstV_60_sell', 'MedianV_60_sell', 'MinV_60_sell', 'MaxV_60_sell', 'VolumeVolatility_60_sell', 'PctlP_10_60_sell', 'PctlP_25_60_sell', 'PctlV_10_60_sell', 'PctlV_25_60_sell', 'PctlV_45_60_sell', 'PctlV_55_60_sell', 'MinP_15_sell', 'MaxP_15_sell', 'PriceVolatility_15_sell', 'DeltaP_15_sell', 'FirstV_15_sell', 'MedianV_15_sell', 'MeanV_15_sell', 'MaxV_15_sell', 'Momentum_15_sell', 'SumV_15_sell', 'TradeCount_15_sell', 'PctlP_10_15_sell', 'PctlP_25_15_sell', 'PctlP_90_15_sell', 'PctlV_10_15_sell', 'PctlV_90_15_sell', 'MaxP_5_sell', 'DeltaP_5_sell', 'FirstV_5_sell', 'MedianV_5_sell', 'MinV_5_sell', 'VolumeVolatility_5_sell', 'Momentum_5_sell', 'SumV_5_sell', 'PctlP_10_5_sell', 'PctlP_25_5_sell', 'PctlP_45_5_sell', 'PctlP_90_5_sell', 'PctlV_10_5_sell', 'PctlV_55_5_sell', 'PctlV_75_5_sell', 'PctlV_90_5_sell', 'PriceVolatility_1_sell', 'DeltaP_1_sell', 'FirstV_1_sell', 'MinV_1_sell', 'VolumeVolatility_1_sell', 'Momentum_1_sell', 'PctlP_25_1_sell', 'PctlV_55_1_sell', 'PctlV_75_1_sell', 'PctlV_90_1_sell', 'FirstP_full_buy', 'PriceVolatility_full_buy', 'MeanV_full_buy', 'MaxV_full_buy', 'DeltaV_full_buy', 'Momentum_full_buy', 'PctlP_10_full_buy', 'PctlP_25_full_buy', 'PctlP_90_full_buy', 'PctlV_10_full_buy', 'PctlV_25_full_buy', 'PctlV_45_full_buy', 'PctlV_55_full_buy', 'PctlV_90_full_buy', 'MinP_180_buy', 'MaxP_180_buy', 'PriceVolatility_180_buy', 'DeltaP_180_buy', 'FirstV_180_buy', 'MinV_180_buy', 'MaxV_180_buy', 'Momentum_180_buy', 'SumV_180_buy', 'PctlP_75_180_buy', 'PctlP_90_180_buy', 'PctlV_10_180_buy', 'PctlV_25_180_buy', 'PctlV_45_180_buy', 'PctlV_55_180_buy', 'PctlV_75_180_buy', 'PctlV_90_180_buy', 'FirstP_60_buy', 'MaxP_60_buy', 'MaxV_60_buy', 'TradeCount_60_buy', 'PctlP_55_60_buy', 'PctlP_75_60_buy', 'PctlP_90_60_buy', 'PctlV_10_60_buy', 'PctlV_75_60_buy', 'PctlV_90_60_buy', 'FirstP_15_buy', 'MaxP_15_buy', 'PriceVolatility_15_buy', 'FirstV_15_buy', 'MedianV_15_buy', 'MeanV_15_buy', 'MinV_15_buy', 'MaxV_15_buy', 'Momentum_15_buy', 'PctlP_90_15_buy', 'PctlV_25_15_buy', 'PctlV_75_15_buy', 'PctlV_90_15_buy', 'FirstP_5_buy', 'MinP_5_buy', 'PriceVolatility_5_buy', 'DeltaP_5_buy', 'FirstV_5_buy', 'MeanV_5_buy', 'MinV_5_buy', 'VolumeVolatility_5_buy', 'TradeCount_5_buy', 'PctlP_10_5_buy', 'PctlP_25_5_buy', 'PctlP_90_5_buy', 'PctlV_25_5_buy', 'PctlV_75_5_buy', 'PctlV_90_5_buy', 'PriceVolatility_1_buy', 'DeltaP_1_buy', 'FirstV_1_buy', 'LastV_1_buy', 'MinV_1_buy', 'VolumeVolatility_1_buy', 'Momentum_1_buy', 'TradeCount_1_buy', 'PctlV_55_1_buy', 'PctlV_75_1_buy', 'PctlV_90_1_buy']
            elif country == 'germany' and resolution == 'qh':
                return ['MinP_full_sell', 'MaxP_full_sell', 'DeltaP_full_sell', 'MedianV_full_sell', 'PctlV_25_full_sell', 'MaxP_180_sell', 'MaxP_60_sell', 'PctlP_90_60_sell', 'MaxP_15_sell', 'PctlP_90_15_sell', 'MaxP_1_sell', 'PctlP_90_1_sell', 'DeltaP_full_buy', 'FirstV_full_buy', 'MeanV_full_buy', 'PctlV_25_full_buy', 'PctlV_75_full_buy', 'TradeCount_180_buy', 'PctlP_25_15_buy', 'FirstP_5_buy', 'MinP_1_buy']
            elif country == 'austria' and resolution == 'h':
                return ['MinP_full_sell', 'MaxP_full_sell', 'PriceVolatility_full_sell', 'DeltaP_full_sell', 'FirstV_full_sell', 'MedianV_full_sell', 'MinV_full_sell', 'MaxV_full_sell', 'Momentum_full_sell', 'SumV_full_sell', 'TradeCount_full_sell', 'PctlP_25_full_sell', 'PctlP_55_full_sell', 'PctlV_10_full_sell', 'PctlV_25_full_sell', 'PctlV_45_full_sell', 'PctlV_75_full_sell', 'PctlV_90_full_sell', 'MinP_180_sell', 'MaxP_180_sell', 'PriceVolatility_180_sell', 'DeltaP_180_sell', 'LastV_180_sell', 'MeanV_180_sell', 'MinV_180_sell', 'MaxV_180_sell', 'DeltaV_180_sell', 'Momentum_180_sell', 'SumV_180_sell', 'PctlV_10_180_sell', 'PctlV_25_180_sell', 'PctlV_45_180_sell', 'PctlV_55_180_sell', 'PctlV_75_180_sell', 'PctlV_90_180_sell', 'FirstP_60_sell', 'PriceVolatility_60_sell', 'FirstV_60_sell', 'MinV_60_sell', 'MaxV_60_sell', 'VolumeVolatility_60_sell', 'SumV_60_sell', 'TradeCount_60_sell', 'PctlV_10_60_sell', 'PctlV_25_60_sell', 'PctlV_55_60_sell', 'PctlV_90_60_sell', 'PriceVolatility_15_sell', 'DeltaP_15_sell', 'FirstV_15_sell', 'LastV_15_sell', 'MinV_15_sell', 'SumV_15_sell', 'TradeCount_15_sell', 'PctlV_45_15_sell', 'PctlV_75_15_sell', 'PctlV_90_15_sell', 'PriceVolatility_5_sell', 'MinV_5_sell', 'VolumeVolatility_5_sell', 'Momentum_5_sell', 'TradeCount_5_sell', 'PctlV_45_5_sell', 'PctlV_75_5_sell', 'FirstP_1_sell', 'MinP_1_sell', 'MaxP_1_sell', 'PriceVolatility_1_sell', 'DeltaP_1_sell', 'MedianV_1_sell', 'MinV_1_sell', 'VolumeVolatility_1_sell', 'DeltaV_1_sell', 'SumV_1_sell', 'PctlP_10_1_sell', 'PctlV_10_1_sell', 'PctlV_55_1_sell', 'PctlV_90_1_sell', 'FirstP_full_buy', 'MinP_full_buy', 'MaxP_full_buy', 'PriceVolatility_full_buy', 'FirstV_full_buy', 'MinV_full_buy', 'MaxV_full_buy', 'Momentum_full_buy', 'TradeCount_full_buy', 'PctlP_45_full_buy', 'PctlP_75_full_buy', 'PctlP_90_full_buy', 'PctlV_10_full_buy', 'PctlV_25_full_buy', 'PctlV_55_full_buy', 'PctlV_90_full_buy', 'LastP_180_buy', 'PriceVolatility_180_buy', 'MinV_180_buy', 'MaxV_180_buy', 'Momentum_180_buy', 'SumV_180_buy', 'TradeCount_180_buy', 'PctlP_45_180_buy', 'PctlV_25_180_buy', 'PctlV_55_180_buy', 'PctlV_90_180_buy', 'PriceVolatility_60_buy', 'DeltaP_60_buy', 'FirstV_60_buy', 'LastV_60_buy', 'MeanV_60_buy', 'MinV_60_buy', 'MaxV_60_buy', 'Momentum_60_buy', 'SumV_60_buy', 'PctlP_55_60_buy', 'PctlP_75_60_buy', 'FirstP_15_buy', 'PriceVolatility_15_buy', 'FirstV_15_buy', 'MedianV_15_buy', 'Momentum_15_buy', 'TradeCount_15_buy', 'PctlV_90_15_buy', 'MaxP_5_buy', 'FirstV_5_buy', 'MinV_5_buy', 'MaxV_5_buy', 'VolumeVolatility_5_buy', 'SumV_5_buy', 'TradeCount_5_buy', 'PctlP_75_5_buy', 'PriceVolatility_1_buy', 'FirstV_1_buy', 'Momentum_1_buy', 'SumV_1_buy', 'TradeCount_1_buy', 'PctlV_10_1_buy', 'PctlV_55_1_buy']
            elif country == 'austria' and resolution == 'qh':
                return ['MaxV_full_sell', 'SumV_full_sell', 'TradeCount_full_sell', 'PctlP_90_full_sell', 'PctlV_90_full_sell', 'MaxP_180_sell', 'TradeCount_180_sell', 'LastP_60_sell', 'MaxP_60_sell', 'MaxP_1_sell', 'FirstP_full_buy', 'LastP_full_buy', 'MaxP_full_buy', 'TradeCount_full_buy', 'MaxP_180_buy', 'TradeCount_180_buy', 'MaxP_1_buy', 'PctlP_90_full_buy', 'PctlP_90_15_buy']
            
def optimize_linear_quantile_regression_no_alphas(X_train, y_train, X_val, y_val, X_test, y_test, quantiles, y_scaler, country, resolution, selection):
    from sklearn.linear_model import QuantileRegressor
    from sklearn.metrics import mean_pinball_loss
    y_pred_list = []
    quantile_losses = []
    rmse_list, mae_list, r2_list = [], [], []
    y_test = y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).ravel()
    y_val = y_scaler.inverse_transform(np.array(y_val).reshape(-1, 1)).ravel()

    results = []
    val_losses = []
    for q in quantiles:
        filtered_features = get_features(country, resolution, selection, q)
        print(f'  Feature set: {filtered_features}')
        X_train_sub, X_val_sub, X_test_sub = X_train[filtered_features], X_val[filtered_features], X_test[filtered_features]

        model = QuantileRegressor(quantile=q, alpha=0.0)
        model.fit(X_train_sub, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val_sub)
        y_val_pred = y_scaler.inverse_transform(y_val_pred.reshape(-1, 1)).ravel()
        val_loss = pinball_loss(y_val, y_val_pred, q)
        val_losses.append(val_loss)
        # Get non-zero coefficient feature names
        #coef_mask = model.coef_ != 0
        #features = list(X_train.columns[coef_mask])

        print(f"  Quantile: {q}, Validation Loss: {val_loss}")
        
        y_pred_scaled = model.predict(X_test_sub)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_pred_list.append(y_pred)

        loss = pinball_loss(y_test, y_pred, q)
        quantile_losses.append(loss)

        rmse, mae, r2 = compute_regression_metrics(y_test, y_pred)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

        # Append to results
        results.append({
            "quantile": q,
            "val_loss": val_loss,
            "features": filtered_features,
            "quantile_loss": loss,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        })

    # Quantile crossing
    y_pred_array = np.column_stack(y_pred_list)
    crossing_rate = compute_quantile_crossing_rate_regressor(y_pred_array)

    # Median metrics
    median_index = quantiles.index(0.5)
    median_rmse = rmse_list[median_index]
    median_mae = mae_list[median_index]
    median_r2 = r2_list[median_index]
    print(f"val AQL: {np.mean(val_losses)}")
    print(f"AQL: {np.mean(quantile_losses):.4f}, AQCR: {crossing_rate * 100:.2f}%")
    print(f"RMSE: {median_rmse:.2f}, MAE: {median_mae:.2f}, R²: {median_r2:.2f} \n")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def ensure_trailing_sep(p: str) -> str:
    return p if p.endswith(("/", "\\")) else p + os.sep

def mpbl(y_true, y_pred, q):
    """Version-proof mean pinball loss (scikit-learn alpha kw-only in >=1.5)."""
    try:
        return _sk_pinball_loss(y_true, y_pred, alpha=float(q))
    except TypeError:
        return _sk_pinball_loss(y_true, y_pred, float(q))

def make_pinball(q: float):
    q = float(q)
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss


# -----------------------
# MLP builder
# -----------------------

def build_quantile_mlp(input_dim: int,
                       q: float,
                       hidden=(128, 64, 32),
                       dropout=0.10,
                       lr=1e-3,
                       seed=42) -> keras.Model:
    tf.keras.utils.set_random_seed(seed)
    x_in = keras.Input(shape=(input_dim,))
    x = x_in
    for h in hidden:
        x = layers.Dense(h, activation="swish")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)
    y_out = layers.Dense(1, activation="linear")(x)
    model = keras.Model(x_in, y_out, name=f"qMLP_{q:.2f}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=make_pinball(q))
    return model


# -----------------------
# Feature & scaling utils
# -----------------------

def align_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Ensure df has all columns in `feature_list` (missing -> 0.0) and correct order."""
    X = df.copy()
    for col in feature_list:
        if col not in X.columns:
            X[col] = 0.0
    return X[feature_list]

def fit_xy_scalers(X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[StandardScaler, StandardScaler]:
    xsc = StandardScaler()
    ysc = StandardScaler()
    Xs = xsc.fit_transform(np.asarray(X_train, dtype=np.float32))
    ys = ysc.fit_transform(np.asarray(y_train, dtype=np.float32).reshape(-1, 1)).ravel()
    return xsc, ysc

def apply_xy_scalers(xsc: StandardScaler, ysc: StandardScaler,
                     X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xs = xsc.transform(np.asarray(X, dtype=np.float32))
    ys = ysc.transform(np.asarray(y, dtype=np.float32).reshape(-1, 1)).ravel()
    return Xs, ys


# -----------------------
# Core evaluator (A→A, B→A, A+B→A)
# -----------------------

def evaluate_transfer_strategy(
    country_A: str, country_B: str,
    resolution_A: str, resolution_B: str,         # separate resolutions
    indice: str,
    read_data_stat, prefilter_orderbook_features, orderbook_split_stat,
    compute_regression_metrics, compute_quantile_crossing_rate_regressor, get_features,
    selection: str,
    quantiles: List[float],
    split_start: str = "2022-01-01",
    split_spec: Tuple[int, int, int] = (24, 6, 6),
    data_root: str = ".",
    model_dir: str = "./Models",
    mlp_cfg: Dict = None,
    seed: int = 42,
    strategy: str = "A_to_A",
) -> Dict:
    """
    Train one MLP per quantile under the specified transfer strategy and return:
      {AQL, AQCR, MAE (q=0.5), RMSE (q=0.5), R2 (q=0.5)}.
    """
    os.makedirs(model_dir, exist_ok=True)
    data_root = ensure_trailing_sep(data_root)

    if mlp_cfg is None:
        mlp_cfg = dict(hidden=(128, 64), dropout=0.10, lr=1e-3, epochs=300, batch_size=1024, seed=seed)

    # --- Load A and B with their own resolutions
    dfA = read_data_stat(data_root, country_A, resolution_A, indice)
    dfA, feature_cols_A = prefilter_orderbook_features(dfA)
    Xtr_A, ytr_A, Xva_A, yva_A, Xte_A, yte_A = orderbook_split_stat(dfA, split_start, split_spec, feature_cols_A, indice)

    dfB = read_data_stat(data_root, country_B, resolution_B, indice)
    dfB, feature_cols_B = prefilter_orderbook_features(dfB)
    Xtr_B, ytr_B, Xva_B, yva_B, Xte_B, yte_B = orderbook_split_stat(dfB, split_start, split_spec, feature_cols_B, indice)

    # --- Choose train/val/test per strategy
    if strategy == "A_to_A":
        train_X_df, train_y = Xtr_A, ytr_A
        val_X_df,   val_y   = Xva_A, yva_A
        test_X_df,  test_y  = Xte_A, yte_A
        feats_selector = lambda q: get_features(country_A, resolution_A, selection, q)
    elif strategy == "B_to_A":
        train_X_df, train_y = Xtr_B, ytr_B
        val_X_df,   val_y   = Xva_B, yva_B
        test_X_df,  test_y  = Xte_A, yte_A
        feats_selector = lambda q: get_features(country_B, resolution_B, selection, q)
    elif strategy == "AplusB_to_A":
        train_X_df = pd.concat([Xtr_A, Xtr_B], axis=0, ignore_index=True)
        train_y    = np.concatenate([ytr_A, ytr_B], axis=0)
        val_X_df   = pd.concat([Xva_A, Xva_B], axis=0, ignore_index=True)
        val_y      = np.concatenate([yva_A, yva_B], axis=0)
        test_X_df, test_y = Xte_A, yte_A
        def feats_selector(q):
            fA = get_features(country_A, resolution_A, selection, q)
            fB = get_features(country_B, resolution_B, selection, q)
            return sorted(set(fA) | set(fB))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # --- Train per quantile
    y_pred_list = []
    q_losses, val_q_losses = [], []
    rmse_list, mae_list, r2_list = [], [], []

    # pick the median index robustly (use q==0.5 if present; else nearest)
    if 0.5 in quantiles:
        median_index = list(quantiles).index(0.5)
    else:
        median_index = int(np.argmin([abs(q - 0.5) for q in quantiles]))

    for q in quantiles:
        feats = feats_selector(q)

        trX = align_features(train_X_df, feats)
        vaX = align_features(val_X_df,   feats)
        teX = align_features(test_X_df,  feats)

        xsc, ysc = fit_xy_scalers(trX, train_y)
        Xtr_s, ytr_s = apply_xy_scalers(xsc, ysc, trX, train_y)
        Xva_s, yva_s = apply_xy_scalers(xsc, ysc, vaX,  val_y)
        Xte_s, yte_s = apply_xy_scalers(xsc, ysc, teX,  test_y)

        model = build_quantile_mlp(
            input_dim=Xtr_s.shape[1],
            q=q,
            hidden=mlp_cfg.get("hidden", (128, 64)),
            dropout=mlp_cfg.get("dropout", 0.10),
            lr=mlp_cfg.get("lr", 1e-3),
            seed=mlp_cfg.get("seed", 42),
        )

        ckpt = os.path.join(
            model_dir,
            f"qMLP_best_{country_A}_{resolution_A}_to_{country_B}_{resolution_B}_{selection}_q{q:.2f}_{strategy}.keras"
        )
        cbs = [
            callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, mode="min", verbose=0),
            callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0),
        ]

        model.fit(
            Xtr_s, ytr_s,
            validation_data=(Xva_s, yva_s),
            epochs=mlp_cfg.get("epochs", 300),
            batch_size=mlp_cfg.get("batch_size", 1024),
            verbose=0,
            callbacks=cbs,
        )

        # inverse-scale validation/test for metrics
        yva_pred = ysc.inverse_transform(model.predict(Xva_s, verbose=0).reshape(-1, 1)).ravel()
        yva_true = ysc.inverse_transform(yva_s.reshape(-1, 1)).ravel()
        vloss = mpbl(yva_true, yva_pred, q)
        val_q_losses.append(vloss)

        yte_pred = ysc.inverse_transform(model.predict(Xte_s, verbose=0).reshape(-1, 1)).ravel()
        yte_true = ysc.inverse_transform(yte_s.reshape(-1, 1)).ravel()
        qloss = mpbl(yte_true, yte_pred, q)

        rmse, mae, r2 = compute_regression_metrics(yte_true, yte_pred)
        y_pred_list.append(yte_pred)
        q_losses.append(qloss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    # --- Aggregate
    y_pred_matrix = np.column_stack(y_pred_list)  # (n_samples, n_quantiles)
    aqcr = compute_quantile_crossing_rate_regressor(y_pred_matrix)
    aql  = float(np.mean(q_losses))

    return dict(
        AQL=aql,
        AQCR=float(aqcr),
        RMSE=float(rmse_list[median_index]),
        MAE=float(mae_list[median_index]),
        R2=float(r2_list[median_index]),
    )


# -----------------------
# Cross-country runner
# -----------------------

def run_cross_country_experiments(
    save_path: str,
    indice: str,
    selection: str,
    quantiles: List[float],
    resolutions: List[str],  # ['h', 'qh']
    read_data_stat=None, prefilter_orderbook_features=None, orderbook_split_stat=None,
    compute_regression_metrics=None, compute_quantile_crossing_rate_regressor=None, get_features=None,
    mlp_cfg: Dict = None,
) -> pd.DataFrame:
    """
    Fills the rows for Table \ref{tab:cross-country}.
    """
    rows = []
    data_root = ensure_trailing_sep(save_path)
    model_dir = os.path.join(save_path, "Models")

    for res in resolutions:
        
        # two targets per product type block: DE and AT
        for A, B in [("germany", "austria"), ("austria", "germany")]:
            # strategies test on domain A
            for strat, label in [
                ("A_to_A",      f"{A[:2].upper()} → {A[:2].upper()}"),
                ("B_to_A",      f"{B[:2].upper()} → {A[:2].upper()}"),
                ("AplusB_to_A", f"{A[:2].upper()} + {B[:2].upper()} → {A[:2].upper()}"),
            ]:
                r = evaluate_transfer_strategy(
                    country_A=A, country_B=B,
                    resolution_A=res, resolution_B=res,  # same product type for both countries in this block
                    indice=indice,
                    read_data_stat=read_data_stat,
                    prefilter_orderbook_features=prefilter_orderbook_features,
                    orderbook_split_stat=orderbook_split_stat,
                    compute_regression_metrics=compute_regression_metrics,
                    compute_quantile_crossing_rate_regressor=compute_quantile_crossing_rate_regressor,
                    get_features=get_features,
                    selection=selection,
                    quantiles=quantiles,
                    data_root=data_root,
                    model_dir=model_dir,
                    mlp_cfg=mlp_cfg,
                    strategy=strat,
                )
                rows.append({
                    "Product Type": "60-min" if res == "h" else "15-min",
                    "Trans. Strategy": label,
                    **r
                })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(save_path, "Result", "cross_country_results_qMLP.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# -----------------------
# Cross-product-type runner
# -----------------------

def run_cross_product_type_experiments(
    save_path: str,
    indice: str,
    selection: str,
    quantiles: List[float],
    markets: List[str],  # ['germany','austria']
    read_data_stat=None, prefilter_orderbook_features=None, orderbook_split_stat=None,
    compute_regression_metrics=None, compute_quantile_crossing_rate_regressor=None, get_features=None,
    mlp_cfg: Dict = None,
) -> pd.DataFrame:
    """
    Fills the rows for Table \ref{tab:liquidity}.
    """
    rows = []
    data_root = ensure_trailing_sep(save_path)
    model_dir = os.path.join(save_path, "Models")

    for market in markets:

        # two targets per market: 60-min target and 15-min target
        for (resA, resB) in [("h", "qh"), ("qh", "h")]:
            tgt_label = "60-min" if resA == "h" else "15-min"
            src_label = "60-min" if resB == "h" else "15-min"
            for strat, label in [
                ("A_to_A",      f"{tgt_label} → {tgt_label}"),
                ("B_to_A",      f"{src_label} → {tgt_label}"),
                ("AplusB_to_A", f"{tgt_label} + {src_label} → {tgt_label}"),
            ]:
                r = evaluate_transfer_strategy(
                    country_A=market, country_B=market,
                    resolution_A=resA, resolution_B=resB,  # different product types per strategy
                    indice=indice,
                    read_data_stat=read_data_stat,
                    prefilter_orderbook_features=prefilter_orderbook_features,
                    orderbook_split_stat=orderbook_split_stat,
                    compute_regression_metrics=compute_regression_metrics,
                    compute_quantile_crossing_rate_regressor=compute_quantile_crossing_rate_regressor,
                    get_features=get_features,
                    selection=selection,
                    quantiles=quantiles,
                    data_root=data_root,
                    model_dir=model_dir,
                    mlp_cfg=mlp_cfg,
                    strategy=strat,
                )
                rows.append({
                    "Market": market[:2].upper(),
                    "Trans. Strategy": label,
                    **r
                })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(save_path, "Result", "cross_product_type_results_qMLP.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df