async function getPrediction() {
    const prices = document.getElementById('prices').value;
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prices })
    });

    const data = await response.json();
    const prediction = document.getElementById('prediction');
    prediction.textContent = `Predicted Price: ${data.predicted_price}`;
}
