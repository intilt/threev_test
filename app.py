from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import logging
import time
from prometheus_flask_exporter import PrometheusMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__)

# Enable Prometheus monitoring
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Car Damage Detection API', version='1.0.0')

# Load the trained model
MODEL_PATH = "models/car_damage_model.h5"
model = load_model(MODEL_PATH)

# Custom metrics for Prometheus
latency_metric = metrics.summary(
    'prediction_latency_seconds', 'Time taken for prediction requests', labels={'status': 'success'}
)
confidence_metric = metrics.summary(
    'prediction_confidence', 'Confidence score of predictions', labels={'result': lambda r: r.json.get('result')}
)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check API health."""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
@metrics.counter('prediction_requests_total', 'Total number of prediction requests', labels={'status': lambda r: r.status_code})
@latency_metric.time()
def predict():
    """Prediction endpoint."""
    start_time = time.time()
    logging.info("Received a prediction request.")
    
    if 'file' not in request.files:
        logging.error("No file provided in the request.")
        return jsonify({"error": "No file provided"}), 400

    try:
        # Read the image
        file = request.files['file']
        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224)).reshape(1, 224, 224, 3) / 255.0

        # Make prediction
        prediction = model.predict(img)[0][0]
        result = "damaged" if prediction < 0.5 else "undamaged"
        confidence_metric.observe(float(prediction))
        latency = time.time() - start_time

        logging.info(f"Prediction: {result}, Probability: {float(prediction)}, Latency: {latency:.2f}s")
        return jsonify({"result": result, "probability": float(prediction), "latency": latency})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)