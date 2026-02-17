"""
Flask API wrapper for Chronos Inventory Forecasting Model
"""
from flask import Flask, request, jsonify, send_file
from chronos import Chronos2Pipeline
import numpy as np
import pandas as pd
import io
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

@app.route('/forecast/csv', methods=['POST'])
def forecast_csv():
    """
    Generate forecast from CSV file upload
    
    Expected CSV columns:
    - id, category, timestamp, target, promo_flag, price, is_weekend, 
      month, season, is_festival, festival_name, temperature, stockout_flag, traffic
    
    Query parameter:
    - format=json (default) or format=csv for CSV download
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Validate required columns
        required_cols = ['timestamp', 'target', 'promo_flag', 'price', 'is_weekend', 
                        'month', 'season', 'is_festival', 'festival_name', 
                        'temperature', 'stockout_flag', 'traffic']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400
        
        # Sort by timestamp and take last 90 days as historical data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        if len(df) < 90:
            return jsonify({'error': 'Need at least 90 days of historical data'}), 400
        
        # Use last 90 days for historical data
        historical_df = df.tail(90)
        
        # Prepare input data
        input_data = {
            'target': historical_df['target'].values.astype(np.float32),
            'past_covariates': {
                'temperature': historical_df['temperature'].values.astype(np.float32),
                'stockout_flag': historical_df['stockout_flag'].values.astype(np.float32),
                'traffic': historical_df['traffic'].values.astype(np.float32),
                'promo_flag': historical_df['promo_flag'].values,
                'price': historical_df['price'].values.astype(np.float32),
                'is_weekend': historical_df['is_weekend'].values,
                'month': historical_df['month'].values,
                'season': historical_df['season'].values,
                'is_festival': historical_df['is_festival'].values,
                'festival_name': historical_df['festival_name'].values,
            },
            'future_covariates': {
                # Use last known values for future (user can customize this)
                'promo_flag': np.array([0] * 30),
                'price': np.array([historical_df['price'].iloc[-1]] * 30, dtype=np.float32),
                'is_weekend': np.array([False] * 30),
                'month': np.array([historical_df['month'].iloc[-1]] * 30),
                'season': np.array([historical_df['season'].iloc[-1]] * 30),
                'is_festival': np.array([0] * 30),
                'festival_name': np.array([''] * 30),
            }
        }
        
        # Load model if not already loaded
        model = load_model()
        
        # Generate prediction
        quantiles, mean = model.predict_quantiles(
            [input_data],
            prediction_length=30,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        
        # Format response
        forecast = mean[0].numpy().flatten()
        low = quantiles[0][0, :, 0].numpy()
        median = quantiles[0][0, :, 1].numpy()
        high = quantiles[0][0, :, 2].numpy()
        
        # Generate future dates (30 days from last timestamp)
        last_date = historical_df['timestamp'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        
        # Check output format
        output_format = request.args.get('format', 'json').lower()
        
        if output_format == 'csv':
            # Create CSV output
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'forecast_mean': forecast,
                'forecast_low_10': low,
                'forecast_median': median,
                'forecast_high_90': high
            })
            
            # Convert to CSV
            output = io.StringIO()
            forecast_df.to_csv(output, index=False)
            output.seek(0)
            
            # Return as downloadable file
            return send_file(
                io.BytesIO(output.getvalue().encode()),
                mimetype='text/csv',
                as_attachment=True,
                download_name='forecast_results.csv'
            )
        else:
            # Return JSON
            return jsonify({
                'forecast': {
                    'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                    'mean': forecast.tolist(),
                    'low_10': low.tolist(),
                    'median': median.tolist(),
                    'high_90': high.tolist()
                },
                'summary': {
                    'total_forecast': float(forecast.sum()),
                    'daily_average': float(forecast.mean()),
                    'min_prediction': float(forecast.min()),
                    'max_prediction': float(forecast.max())
                },
                'metadata': {
                    'historical_days': len(historical_df),
                    'forecast_days': 30,
                    'last_historical_date': last_date.strftime('%Y-%m-%d')
                }
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model at startup
    load_model()
    # Run server
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)