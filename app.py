"""
Flask API wrapper for Chronos Inventory Forecasting Model
"""
from flask import Flask, request, jsonify
from chronos import Chronos2Pipeline
import numpy as np
import os

app = Flask(__name__)

# Global model instance (loaded once at startup)
pipeline = None

def load_model():
    """Load the fine-tuned model"""
    global pipeline
    if pipeline is None:
        model_path = "./finetuned_chronos_2"
        print(f"Loading model from {model_path}...")
        pipeline = Chronos2Pipeline.from_pretrained(model_path)
        print("Model loaded successfully!")
    return pipeline

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': pipeline is not None})

@app.route('/forecast', methods=['POST'])
def forecast():
    """
    Generate 30-day forecast
    
    Expected JSON input:
    {
        "historical_usage": [35.2, 32.1, ...],  # 90 days
        "past_covariates": {
            "price": [...],
            "promo_flag": [...],
            ...
        },
        "future_covariates": {
            "price": [...],
            "promo_flag": [...],
            ...
        }
    }
    """
    try:
        # Load model if not already loaded
        model = load_model()
        
        # Parse request
        data = request.get_json()
        
        # Prepare input
        input_data = {
            'target': np.array(data['historical_usage'], dtype=np.float32),
            'past_covariates': {
                key: np.array(val, dtype=np.float32 if key in ['price', 'temperature', 'traffic'] else object)
                for key, val in data.get('past_covariates', {}).items()
            },
            'future_covariates': {
                key: np.array(val, dtype=np.float32 if key in ['price'] else object)
                for key, val in data.get('future_covariates', {}).items()
            }
        }
        
        # Generate prediction
        quantiles, mean = model.predict_quantiles(
            [input_data],
            prediction_length=30,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        
        # Format response
        forecast = mean[0].numpy().flatten()
        
        return jsonify({
            'forecast_mean': forecast.tolist(),
            'forecast_low': quantiles[0][0, :, 0].numpy().tolist(),
            'forecast_median': quantiles[0][0, :, 1].numpy().tolist(),
            'forecast_high': quantiles[0][0, :, 2].numpy().tolist(),
            'total_forecast': float(forecast.sum()),
            'daily_average': float(forecast.mean())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model at startup
    load_model()
    # Run server
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)