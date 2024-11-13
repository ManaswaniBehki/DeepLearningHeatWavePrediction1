import React, { useState } from 'react';
import axios from 'axios';

function PredictionForm() {
  // State to handle the input years and the prediction results
  const [startYear, setStartYear] = useState('');
  const [endYear, setEndYear] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);  // Store prediction results
  const [loading, setLoading] = useState(false);  // Loading state to show spinner

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (startYear && endYear) {
      setLoading(true);
      try {
        // Send the form data to the Flask backend
        const response = await axios.post('http://localhost:5000/predict', {
          start_year: startYear,
          end_year: endYear,
        });

        // Assuming your backend returns a prediction object with years as keys
        // Adjust according to the exact format returned by your Flask API
        const predictions = response.data.predictions; // Example format: { '2021': {...}, '2022': {...}, ... }

        // Format data to be an array (if necessary)
        const predictionArray = Object.entries(predictions).map(([year, data]) => ({
          year,
          ...data, // Spread the data to include everything for each year
        }));

        // Store the formatted result in state
        setPredictionResult(predictionArray);
      } catch (error) {
        console.error('Error fetching prediction data', error);
      } finally {
        setLoading(false);
      }
    } else {
      alert('Please fill in both the start and end years.');
    }
  };

  return (
    <div className="form-container">
      <h2>Prediction Results</h2>
      <form onSubmit={handleSubmit}>
        <div className="input-group">
          <div className="form-group">
            <label>Start Year:</label>
            <input
              type="number"
              value={startYear}
              onChange={(e) => setStartYear(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <label>End Year:</label>
            <input
              type="number"
              value={endYear}
              onChange={(e) => setEndYear(e.target.value)}
              required
            />
          </div>
        </div>
        <button type="submit" disabled={loading}>
          {loading ? 'Loading...' : 'Get Prediction'}
        </button>
      </form>

      {predictionResult && (
        <div className="prediction-result">
          <h3>Prediction for the years {startYear} to {endYear}</h3>
          <div className="prediction-list">
            {predictionResult.map((item, index) => (
              <div className="prediction-item" key={index}>
                <h4>Year: {item.year}</h4>
                <div className="prediction-details">
                  <p><strong>JAN-FEB:</strong> {item['JAN-FEB']}</p>
                  <p><strong>MAR-MAY:</strong> {item['MAR-MAY']}</p>
                  <p><strong>JUN-SEP:</strong> {item['JUN-SEP']}</p>
                  <p><strong>OCT-DEC:</strong> {item['OCT-DEC']}</p>
                  <p><strong>Hottest Month:</strong> {item.Hottest_Month}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default PredictionForm;
