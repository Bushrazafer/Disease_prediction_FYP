"""
RoyalSoft ML Intelligence Engine - REST API
Flask-based Production API for Diabetes Prediction
"""

from flask import Flask, request, jsonify
from predict import DiabetesPredictionEngine
import json

app = Flask(__name__)
engine = DiabetesPredictionEngine()

@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "RoyalSoft ML Intelligence Engine",
        "version": "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Diabetes prediction endpoint
    
    Expected JSON payload:
    {
        "age": 45,
        "glucose": 140,
        "blood_pressure": 85,
        "skin_thickness": 25,
        "insulin": 100,
        "bmi": 32.5,
        "diabetes_pedigree": 0.8,
        "pregnancies": 2
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        result = engine.predict(data)
        
        if not result["success"]:
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information and metrics"""
    return jsonify({
        "success": True,
        "model_info": {
            "version": "1.0.0",
            "trained_on": "PIMA Diabetes Dataset",
            "accuracy": engine.metrics['accuracy'],
            "precision": engine.metrics['precision'],
            "recall": engine.metrics['recall'],
            "f1_score": engine.metrics['f1_score'],
            "auc_roc": engine.metrics['auc_roc'],
            "confusion_matrix": engine.metrics['confusion_matrix']
        },
        "features": engine.feature_names
    })

if __name__ == '__main__':
    print("=" * 60)
    print("RoyalSoft ML Intelligence Engine - API Server")
    print("=" * 60)
    print("Starting Flask API on http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /health      - Health check")
    print("  POST /predict     - Diabetes prediction")
    print("  GET  /model-info  - Model information")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
