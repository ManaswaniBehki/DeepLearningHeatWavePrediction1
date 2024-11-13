import React, { useState } from 'react';
import './App.css';
import PredictionForm from './components/predictionform';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handlePrediction = (data) => {
    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),  // Send the data as a JSON object
    })
      .then((response) => response.json())
      .then((result) => setPrediction(result.prediction))
      .catch((err) => setError('Error fetching prediction: ' + err.message));
  };

  return (
    <div className="App">
      <h1>Heatwave Prediction</h1>
      <PredictionForm onSubmit={handlePrediction} />
      {error && <p className="error">{error}</p>}
      {prediction && (
        <div>
          <h2>Prediction Results</h2>
          <p>{prediction}</p>
        </div>
      )}
    </div>
  );
}

export default App;
