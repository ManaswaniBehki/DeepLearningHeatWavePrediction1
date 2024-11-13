from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from your_model_script import load_and_prepare_data, prepare_model_data, build_and_train_model, predict_multiple_years, visualize_all_predictions, plot_loss, plot_mae  # Import the functions from your model file

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables to store the trained model and scaler
model = None
scaler_X = None
scaler_y = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    start_year = int(data['start_year'])
    end_year = int(data['end_year'])
    
    # Load and prepare data
    file_path = 'TEMP_ANNUAL_SEASONAL_MEAN.csv'  # Adjust the path as needed
    df_filtered = load_and_prepare_data(file_path)
    
    # Prepare model data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_model_data(df_filtered)
    
    # Build and train model (if not already trained)
    global model
    if model is None:
        model, history = build_and_train_model(X_train, y_train, X_test, y_test)
        # Optionally, save the model and plot loss
        plot_loss(history)  # Optionally plot the loss if you want to keep track of it
        plot_mae(history)

        
    # Use 2020's data as starting point for prediction
    X_2020 = [[20.79, 27.58, 28.45, 23.75]]  # Example data for 2020
    X_2020_scaled = scaler_X.transform(X_2020)
    
    # Make predictions
    predictions = predict_multiple_years(start_year, end_year, model, scaler_X, scaler_y, X_2020_scaled)

    # Prepare output in the desired format
    output = {}
    for year, pred in predictions.items():
        # Store predictions by months
        predicted_months = {
            'JAN-FEB': pred['JAN-FEB'],
            'MAR-MAY': pred['MAR-MAY'],
            'JUN-SEP': pred['JUN-SEP'],
            'OCT-DEC': pred['OCT-DEC']
        }

        # Find the hottest month
        hottest_month = max(predicted_months, key=predicted_months.get)
        hottest_temp = predicted_months[hottest_month]
        
        # Format the year prediction
        output[year] = {
            'RAINFALL': 'N/A',  # Assuming you don't have rainfall data in your prediction model
            'JAN-FEB': f"{pred['JAN-FEB']:.2f}°C",
            'MAR-MAY': f"{pred['MAR-MAY']:.2f}°C",
            'JUN-SEP': f"{pred['JUN-SEP']:.2f}°C",
            'OCT-DEC': f"{pred['OCT-DEC']:.2f}°C",
            'Hottest_Month': f"{hottest_month} with {hottest_temp:.2f}°C"
        }
    
    # Optionally generate visualizations
    visualize_all_predictions(predictions)  # Create and save visualizations
    
    # Return predictions in the formatted structure as JSON
    return jsonify({'predictions': output})


if __name__ == '__main__':
    app.run(debug=True)
