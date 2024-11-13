const API_URL = 'http://localhost:5000/api';

export const fetchPrediction = async (startYear, endYear) => {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ startYear, endYear }), // Send data as JSON
    });
    const data = await response.json();
    return data.prediction;
  } catch (error) {
    console.error('Error fetching prediction:', error);
    throw error;
  }
};
