import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score

# Visualization Functions
def create_seasonal_heatmap(predictions_dict):
    """
    Creates a heatmap showing seasonal temperature patterns across years.
    """
    df = pd.DataFrame.from_dict(predictions_dict, orient='index')
    seasonal_data = df[['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(seasonal_data, annot=True, fmt='.2f', cmap='RdYlBu_r',
                xticklabels=['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'],
                yticklabels=df.index)
    plt.title('Seasonal Temperature Patterns Across Years')
    plt.ylabel('Year')
    plt.xlabel('Season')
    plt.savefig('seasonal_heatmap.png')
    plt.close()

def plot_annual_trend(predictions_dict):
    """
    Creates a line plot showing annual temperature trends with seasonal breakdown.
    """
    df = pd.DataFrame.from_dict(predictions_dict, orient='index')
    
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['ANNUAL'], 'k-', linewidth=2, label='Annual Average')
    plt.plot(df.index, df['JAN-FEB'], '--', label='Jan-Feb')
    plt.plot(df.index, df['MAR-MAY'], '--', label='Mar-May')
    plt.plot(df.index, df['JUN-SEP'], '--', label='Jun-Sep')
    plt.plot(df.index, df['OCT-DEC'], '--', label='Oct-Dec')
    
    plt.title('Temperature Trends Over Time')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('annual_trend.png')
    plt.close()

def create_seasonal_boxplot(predictions_dict):
    """
    Creates a boxplot showing temperature distribution across seasons.
    """
    df = pd.DataFrame.from_dict(predictions_dict, orient='index')
    seasonal_data = df[['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']]
    
    plt.figure(figsize=(10, 6))
    seasonal_data.boxplot()
    plt.title('Temperature Distribution by Season')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)
    plt.savefig('seasonal_boxplot.png')
    plt.close()

def create_monthly_comparison_radar(predictions_dict, year):
    """
    Creates a radar chart comparing seasonal temperatures for a specific year.
    """
    df = pd.DataFrame.from_dict(predictions_dict, orient='index')
    year_data = df.loc[year, ['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']]
    
    categories = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
    values = year_data.values
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title(f'Temperature Distribution for {year}')
    plt.savefig(f'radar_chart_{year}.png')
    plt.close()

# Model Training and Prediction Functions
def load_and_prepare_data(file_path):
    """
    Load and prepare the temperature data for modeling.
    """
    # Load the data
    df_temp = pd.read_csv(file_path)
    
    # Filter for recent years
    years_of_interest = [2020, 2019, 2018, 2017, 2016]
    df_filtered = df_temp[df_temp['YEAR'].isin(years_of_interest)]
    
    # Convert columns to numeric
    columns_to_convert = ['ANNUAL', 'JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']
    for col in columns_to_convert:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    # Drop rows with NaN values
    df_filtered = df_filtered.dropna()
    
    return df_filtered

def prepare_model_data(df_filtered):
    """
    Prepare data for model training.
    """
    X = df_filtered[['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']].values
    y = df_filtered[['ANNUAL', 'JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']].values
    
    # Normalize the data
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # Reshape X for Conv1D
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    # Split into train and test sets
    train_size = int(len(X_reshaped) * 0.8)
    X_train = X_reshaped[:train_size]
    X_test = X_reshaped[train_size:]
    y_train = y_scaled[:train_size]
    y_test = y_scaled[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def build_and_train_model(X_train, y_train, X_test, y_test):
    """
    Build and train the hybrid Conv1D-LSTM model.
    """
    model = Sequential([
        Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        LSTM(units=64, activation='relu', return_sequences=True),
        LSTM(units=32, activation='relu'),
        Dense(units=5)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    
    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[lr_scheduler]
    )
    
    return model, history

def predict_temperatures(year, model, scaler_X, scaler_y, X_last):
    """
    Predict temperatures for a specific year.
    """
    # Reshape input for prediction
    X_reshaped = X_last.reshape((1, X_last.shape[1], 1))
    
    # Make prediction
    predicted_temp = model.predict(X_reshaped)
    predicted_temp = scaler_y.inverse_transform(predicted_temp)
    
    # Format prediction results
    return {
        'YEAR': year,
        'ANNUAL': predicted_temp[0][0],
        'JAN-FEB': predicted_temp[0][1],
        'MAR-MAY': predicted_temp[0][2],
        'JUN-SEP': predicted_temp[0][3],
        'OCT-DEC': predicted_temp[0][4]
    }

def predict_multiple_years(start_year, end_year, model, scaler_X, scaler_y, X_last):
    """
    Predict temperatures for multiple years.
    """
    predictions = {}
    current_X = X_last.copy()
    
    for year in range(start_year, end_year + 1):
        # Get prediction for current year
        predicted_data = predict_temperatures(year, model, scaler_X, scaler_y, current_X)
        predictions[year] = predicted_data
        
        # Update input for next year's prediction
        current_X = scaler_X.transform([[
            predicted_data['JAN-FEB'],
            predicted_data['MAR-MAY'],
            predicted_data['JUN-SEP'],
            predicted_data['OCT-DEC']
        ]])
    
    return predictions

def visualize_all_predictions(predictions_dict):
    """
    Create all visualizations for the predictions.
    """
    create_seasonal_heatmap(predictions_dict)
    plot_annual_trend(predictions_dict)
    create_seasonal_boxplot(predictions_dict)
    
    # Create radar charts for each year
    for year in predictions_dict.keys():
        create_monthly_comparison_radar(predictions_dict, year)

def plot_loss(history):
    """
    Plots the training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()
def plot_mae(history):
    """
    Plots the training and validation Mean Absolute Error (MAE) over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('mae_plot.png')  # Save the plot as a .png file

def evaluate_model(y_test, y_pred):
    """
    Evaluate the model using MAE and R2 score.
    """
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R2 Score: {r2}')