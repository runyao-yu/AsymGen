import pandas as pd
import numpy as np
import os
import joblib
import warnings
from tqdm import tqdm
import gc
import sklearn
from sklearn.preprocessing import RobustScaler


'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- -- functions used for filtering matched trades  -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def add_traded_volume(df):
    """
    Calculates the traded volume per transaction within each OrderId group.
    Only rows with non-zero, non-NaN VolumeTraded remain at the end.
    """
    # Sort by OrderId then TransactionTime
    df = df.sort_values(['OrderId', 'TransactionTime'])
    
    # Identify trades
    trades_mask = df['ActionCode'].isin(['P', 'M'])
    
    # Compute the volume difference by shifting within each OrderId
    df['VolumeTraded'] = df.groupby('OrderId')['Volume'].shift(1) - df['Volume']
    
    # For the first row of each OrderId, set VolumeTraded to NaN
    is_first_in_group = df['OrderId'].ne(df['OrderId'].shift(1))
    df.loc[is_first_in_group, 'VolumeTraded'] = float('nan')
    
    # Set traded volume to 0 for non-trade rows
    df.loc[~trades_mask, 'VolumeTraded'] = 0
    
    # Drop rows where VolumeTraded is 0 or NaN
    df = df[df['VolumeTraded'].notna() & (df['VolumeTraded'] != 0)]
    
    return df

def filter_raw_data(country, year):
    base_path = f"EPEX_Spot_Orderbook/{country}/Intraday Continuous/Orders"
    
    # Only load these columns from CSV
    necessary_columns = [
        'DeliveryStart',
        'Side',
        'Product',
        'Price',
        'Volume',
        'ActionCode',
        'TransactionTime',
        'OrderId'
    ]
    
    path = os.path.join(base_path, str(year))
    
    # We will collect results in lists and concatenate once
    hour_list = []
    quarter_hour_list = []
    
    # Count the number of files for tqdm progress bar
    total_files = sum(len(files) for _, _, files in os.walk(path))
    
    with tqdm(total=total_files, desc=f"processing {year} data for {country}", unit="file") as pbar:
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                data_path = os.path.join(dirname, filename)
                
                # Read CSV with only necessary columns
                df = pd.read_csv(
                    data_path,
                    header=1,
                    dtype={'ParentId': 'Int64'},
                    usecols=necessary_columns
                )
                
                # Split into hour and quarter-hour subsets
                hour_df = df[df['Product'].isin(['Intraday_Hour_Power', 'XBID_Hour_Power'])]
                qh_df = df[df['Product'].isin(['Intraday_Quarter_Hour_Power', 'XBID_Quarter_Hour_Power'])]
                
                # Process the hour trades
                if not hour_df.empty:
                    hour_df = add_traded_volume(hour_df)
                    # Keep only partial/matched trades
                    hour_df = hour_df[hour_df['ActionCode'].isin(['P', 'M'])]
                    hour_list.append(hour_df)

                # Process the quarter-hour trades
                if not qh_df.empty:
                    qh_df = add_traded_volume(qh_df)
                    # Keep only partial/matched trades
                    qh_df = qh_df[qh_df['ActionCode'].isin(['P', 'M'])]
                    quarter_hour_list.append(qh_df)
                
                pbar.update(1)
    
    # Concatenate all hour and quarter-hour data for the year
    combined_h_df = pd.concat(hour_list, ignore_index=True) if hour_list else pd.DataFrame(columns=necessary_columns)
    combined_qh_df = pd.concat(quarter_hour_list, ignore_index=True) if quarter_hour_list else pd.DataFrame(columns=necessary_columns)
    
    # Only keep columns: [side, deliverystart, transactiontime, price, volume traded]
    keep_cols = ['Side', 'DeliveryStart', 'TransactionTime', 'Price', 'VolumeTraded']
    
    # Hourly
    combined_h_df = combined_h_df[keep_cols]
    combined_h_df.to_csv(f"{year}_h_{country}.csv", index=False)
    
    # Quarter-hourly
    combined_qh_df = combined_qh_df[keep_cols]
    combined_qh_df.to_csv(f"{year}_qh_{country}.csv", index=False)


def merge_filtered_data(resolution, country):

    df_2022 = pd.read_csv('EPEX_Spot_Orderbook/'+f"2022_{resolution}_{country}.csv")
    df_2022.reset_index(drop=True, inplace=True)
    df_2022['DeliveryStart'] = pd.to_datetime(df_2022['DeliveryStart'])
    df_2022['TransactionTime'] = pd.to_datetime(df_2022['TransactionTime'])

    df_2023 = pd.read_csv('EPEX_Spot_Orderbook/'+f"2023_{resolution}_{country}.csv")
    df_2023.reset_index(drop=True, inplace=True)
    df_2023['DeliveryStart'] = pd.to_datetime(df_2023['DeliveryStart'])
    df_2023['TransactionTime'] = pd.to_datetime(df_2023['TransactionTime'])

    df_2024 = pd.read_csv('EPEX_Spot_Orderbook/'+f"2024_{resolution}_{country}.csv")
    df_2024.reset_index(drop=True, inplace=True)
    df_2024['DeliveryStart'] = pd.to_datetime(df_2024['DeliveryStart'])
    df_2024['TransactionTime'] = pd.to_datetime(df_2024['TransactionTime'])

    df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
    df.to_pickle('EPEX_Spot_Orderbook/'+f"Filtered_{resolution}_{country}.pkl")


'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- --   functions used for extracting features  -- -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def input_extraction_enhanced(filtered_df):
    filtered_df = filtered_df.sort_values('TransactionTime')
    sum_volume = np.sum(filtered_df["VolumeTraded"])

    if filtered_df.empty or sum_volume == 0:
        return (np.nan,) * 20  # 21 features

    prices = filtered_df['Price']
    volumes = filtered_df['VolumeTraded']

    first_price = prices.iloc[0]
    last_price = prices.iloc[-1]
    median_price = np.median(prices)
    mean_price = np.mean(prices)
    min_price = np.min(prices)
    max_price = np.max(prices)
    price_volatility = np.std(prices)
    delta_price = last_price - first_price

    first_volume = volumes.iloc[0]
    last_volume = volumes.iloc[-1]
    median_volume = np.median(volumes)
    mean_volume = np.mean(volumes)
    min_volume = np.min(volumes)
    max_volume = np.max(volumes)
    volume_volatility = np.std(volumes)
    delta_volume = last_volume - first_volume

    vwap = np.average(prices, weights=volumes)
    momentum = (last_price - vwap) / vwap if vwap != 0 else np.nan
    trade_count = len(filtered_df)

    return (first_price, last_price, median_price, mean_price, min_price, max_price, price_volatility, delta_price,
            first_volume, last_volume, median_volume, mean_volume, min_volume, max_volume, volume_volatility, delta_volume,
            vwap, momentum, sum_volume, trade_count)


def input_extraction_enhanced(filtered_df):
    filtered_df = filtered_df.sort_values('TransactionTime')
    sum_volume = np.sum(filtered_df["VolumeTraded"])

    if filtered_df.empty or sum_volume == 0:
        return (np.nan,) * 32  # Updated feature count

    prices = filtered_df['Price']
    volumes = filtered_df['VolumeTraded']

    first_price = prices.iloc[0]
    last_price = prices.iloc[-1]
    median_price = np.median(prices)
    mean_price = np.mean(prices)
    min_price = np.min(prices)
    max_price = np.max(prices)
    price_volatility = np.std(prices)
    delta_price = last_price - first_price

    first_volume = volumes.iloc[0]
    last_volume = volumes.iloc[-1]
    median_volume = np.median(volumes)
    mean_volume = np.mean(volumes)
    min_volume = np.min(volumes)
    max_volume = np.max(volumes)
    volume_volatility = np.std(volumes)
    delta_volume = last_volume - first_volume

    vwap = np.average(prices, weights=volumes)
    momentum = (last_price - vwap) / vwap if vwap != 0 else np.nan
    trade_count = len(filtered_df)

    price_percentiles = np.percentile(prices, [10, 25, 45, 55, 75, 90])
    volume_percentiles = np.percentile(volumes, [10, 25, 45, 55, 75, 90])

    return (
        first_price, last_price, median_price, mean_price, min_price, max_price, price_volatility, delta_price,
        first_volume, last_volume, median_volume, mean_volume, min_volume, max_volume, volume_volatility, delta_volume,
        vwap, momentum, sum_volume, trade_count,
        *price_percentiles, *volume_percentiles
    )


def extract_features(df, indice):
    data_per_file = []

    main_w = {"ID1": 60, "ID2": 120, "ID3": 180}.get(indice)
    if main_w is None:
        print("Wrong indice, only ID1, ID2, or ID3")
        return pd.DataFrame()

    sub_windows = ['full', 180, 60, 15, 5, 1]
    total_groups = df['DeliveryStart'].nunique()

    with tqdm(total=total_groups, desc="Processing groups", unit="group") as pbar:
        for Date_DeliveryStart, group in df.groupby('DeliveryStart'):
            pbar.set_postfix_str(f"Processing date: {Date_DeliveryStart}")
            pbar.update(1)

            end_dt = Date_DeliveryStart - pd.Timedelta(minutes=main_w)
            subwindow_values = {}

            for i, sub_w in enumerate(sub_windows):
                if sub_w == 'full':
                    df_sub = group[group['TransactionTime'] <= end_dt]
                else:
                    start_dt = end_dt - pd.Timedelta(minutes=sub_w)
                    df_sub = group[(group['TransactionTime'] >= start_dt) & (group['TransactionTime'] <= end_dt)]

                values = input_extraction_enhanced(df_sub)

                if np.isnan(values[0]):
                    found_nonzero = False
                    for bigger_sw in sub_windows[:i][::-1]:
                        prev_vals = subwindow_values.get(bigger_sw, None)
                        if prev_vals and not np.isnan(prev_vals[0]):
                            values = prev_vals
                            found_nonzero = True
                            break

                    if not found_nonzero:
                        values = (np.nan,) * 20

                subwindow_values[sub_w] = values

            row_dict = {'Date_DeliveryStart': Date_DeliveryStart}

            feature_names = [
                "FirstP", "LastP", "MedianP", "MeanP", "MinP", "MaxP", "PriceVolatility", "DeltaP",
                "FirstV", "LastV", "MedianV", "MeanV", "MinV", "MaxV", "VolumeVolatility", "DeltaV",
                "VWAP", "Momentum", "SumV", "TradeCount",
                "PctlP_10", "PctlP_25", "PctlP_45", "PctlP_55", "PctlP_75", "PctlP_90",
                "PctlV_10", "PctlV_25", "PctlV_45", "PctlV_55", "PctlV_75", "PctlV_90"
            ]
            
            for sub_w in sub_windows:
                prefix = 'full' if sub_w == 'full' else str(sub_w)
                values = subwindow_values[sub_w]
                for name, val in zip(feature_names, values):
                    row_dict[f'{name}_{prefix}'] = val

            data_per_file.append(row_dict)

    result_df = pd.DataFrame(data_per_file)
    return result_df







def execute_feature_extraction(resolution, country, indice, side=True):

    # read data
    df = pd.read_pickle('EPEX_Spot_Orderbook/'+f"Filtered_{resolution}_{country}.pkl")
    df.reset_index(drop=True, inplace=True)
    
    # differentiate sides
    if side==True:
        # process buy side
        df_buy = extract_features(df[df["Side"] == "BUY"], indice)
        df_buy.to_pickle('EPEX_Spot_Orderbook/'+f"Feature_Buy_{resolution}_{country}_{indice}.pkl") 
        del df_buy

        # process sell side
        df_sell = extract_features(df[df["Side"] == "SELL"], indice)
        df_sell.to_pickle('EPEX_Spot_Orderbook/'+f"Feature_Sell_{resolution}_{country}_{indice}.pkl") 
        del df_sell

    # not differentiate sides
    elif side==False:
        df = extract_features(df, indice)
        df.to_pickle('EPEX_Spot_Orderbook/'+f"Feature_{resolution}_{country}_{indice}.pkl") 
        del df


'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- --    functions used for extracting labels   -- -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def output_extraction(filtered_df):
    filtered_df = filtered_df.sort_values('TransactionTime')
    sum_volume = np.sum(filtered_df["VolumeTraded"])
    num_trades = len(filtered_df)

    if sum_volume == 0:
        return np.nan, 0, 0
    else:
        price_weighted_avg = np.average(filtered_df['Price'], weights=filtered_df['VolumeTraded'])
        return price_weighted_avg, sum_volume, num_trades


def extract_labels(df, country, indice):
    data_per_file = []

    if indice == 'ID1':
        start_offset = 60

    elif indice == 'ID2':
        start_offset = 120

    elif indice == 'ID3':
        start_offset = 180

    else:
        start_offset = None
        print('Wrong indice, only ID1, ID2, or ID3')

    if country == 'germany':
        end_offset = 30

    elif country == 'austria':
        end_offset = 0

    else:
        end_offset = None
        print('Wrong country, only austria or germany')

    total_groups = df['DeliveryStart'].nunique()

    with tqdm(total=total_groups, desc="Extracting labels", unit="group") as pbar:
        for delivery_start, group in df.groupby('DeliveryStart'):
            pbar.update(1)
            label_row = {'Date_DeliveryStart': delivery_start}

            start_dt = delivery_start - pd.Timedelta(minutes=start_offset)
            end_dt = delivery_start - pd.Timedelta(minutes=end_offset)
            df_sub = group[(group['TransactionTime'] >= start_dt) & (group['TransactionTime'] <= end_dt)]

            vwap, sumv, num_trades = output_extraction(df_sub)
            label_row[indice] = vwap
            label_row[f'SumV_{indice}'] = sumv
            label_row[f'NumTrades_{indice}'] = num_trades

            data_per_file.append(label_row)

    return pd.DataFrame(data_per_file)


def execute_label_extraction(resolution, country, indice, side=False):
    df = pd.read_pickle('EPEX_Spot_Orderbook/' + f"Filtered_{resolution}_{country}.pkl")
    df.reset_index(drop=True, inplace=True)

    if side is True:
        # BUY side
        df_buy = extract_labels(df[df["Side"] == "BUY"], country, indice)
        df_buy.to_pickle('EPEX_Spot_Orderbook/' + f"Label_Buy_{resolution}_{country}_{indice}.pkl")
        del df_buy

        # SELL side
        df_sell = extract_labels(df[df["Side"] == "SELL"], country, indice)
        df_sell.to_pickle('EPEX_Spot_Orderbook/' + f"Label_Sell_{resolution}_{country}_{indice}.pkl")
        del df_sell

    elif side is False:
        df_labels = extract_labels(df, country, indice)
        df_labels.to_pickle('EPEX_Spot_Orderbook/' + f"Label_{resolution}_{country}_{indice}.pkl")
        del df_labels




'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- --  functions used for obtaining global price scaler  -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def fit_and_save_price_scaler(country, resolution, train_start_date, train_end_date):
    # Load and prepare data
    df = pd.read_pickle(os.path.join('EPEX_Spot_Orderbook/', f"Filtered_{resolution}_{country}.pkl"))
    df.reset_index(drop=True, inplace=True)

    # Filter training data
    df_train = df[(df['DeliveryStart'] >= train_start_date) & (df['DeliveryStart'] < train_end_date)]

    # Fit scaler on price values only
    scaler = RobustScaler()
    scaler.fit(df_train[['Price']].values)

    # Save the scaler
    scaler_path = os.path.join('EPEX_Spot_Orderbook/', f"robust_scaler_{country}_{resolution}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")