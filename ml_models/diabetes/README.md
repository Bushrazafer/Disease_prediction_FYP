# RoyalSoft ML Intelligence Engine
## Production-Grade Diabetes Prediction System

### 🎯 Overview
Enterprise-level machine learning system for diabetes risk prediction with strict medical validation, explainability, and production-ready architecture.

### 📊 Model Performance
- **Accuracy**: 75.97%
- **Precision**: 64.41%
- **Recall**: 70.37%
- **F1-Score**: 67.26%
- **AUC-ROC**: 0.8274

### 🚀 Quick Start

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Train Model
```bash
python train_model.py
```

#### 3. Test Predictions
```bash
python test_predictions.py
```

#### 4. Run API Server (Optional)
```bash
pip install flask
python api.py
```

### 📋 Input Features (Strict Validation)

| Feature | Range | Unit |
|---------|-------|------|
| age | 18-120 | years |
| glucose | 50-400 | mg/dL |
| blood_pressure | 80-200 | mmHg |
| skin_thickness | 0-100 | mm |
| insulin | 0-300 | μU/mL |
| bmi | 10-60 | kg/m² |
| diabetes_pedigree | 0-3 | score |
| pregnancies | 0-20 | count |

### 💻 Usage Example

```python
from predict import DiabetesPredictionEngine

# Initialize engine
engine = DiabetesPredictionEngine()

# Patient data
patient = {
    "age": 45,
    "glucose": 140,
    "blood_pressure": 85,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 32.5,
    "diabetes_pedigree": 0.8,
    "pregnancies": 2
}

# Get prediction
result = engine.predict(patient)
print(result)
```

### 📤 Output Format

```json
{
  "success": true,
  "prediction": 1,
  "probability": 66.5,
  "confidence": 33.0,
  "risk_level": "medium",
  "message": "Diabetes risk detected. This is an AI-based risk estimation, not a medical diagnosis.",
  "feature_importance": [
    {"feature": "Glucose", "importance": 15.23},
    {"feature": "Bmi", "importance": 9.67}
  ],
  "recommendations": [
    {
      "priority": "high",
      "title": "Elevated Glucose Level",
      "description": "Your glucose level is in diabetic range..."
    }
  ],
  "model_info": {
    "version": "1.0.0",
    "trained_on": "PIMA Diabetes Dataset",
    "accuracy": 75.97,
    "auc_roc": 0.8274
  }
}
```

### 🔒 Security & Validation
- ✅ Strict medical range validation
- ✅ Type checking (no strings in numeric fields)
- ✅ No hallucination - only validated features
- ✅ Error-first approach
- ✅ Medical safety disclaimers

### 🎯 Risk Classification
- **Low**: < 25% probability
- **Medium-Low**: 25-49%
- **Medium**: 50-69%
- **Medium-High**: 70-85%
- **High**: > 85%

### 🧪 Testing
Comprehensive test suite covering:
- Valid predictions (high/medium/low risk)
- Input validation errors
- Edge cases (min/max values)
- Non-numeric inputs
- Missing fields

### 📁 Project Structure
```
├── diabetes.csv          # Training dataset
├── train_model.py        # Training pipeline
├── predict.py            # Prediction engine
├── api.py                # Flask REST API
├── test_predictions.py   # Test suite
├── model.pkl             # Trained model
├── scaler.pkl            # Feature scaler
├── features.pkl          # Feature names
├── metrics.pkl           # Model metrics
└── requirements.txt      # Dependencies
```

### 🔧 API Endpoints

#### Health Check
```bash
GET /health
```

#### Predict
```bash
POST /predict
Content-Type: application/json

{
  "age": 45,
  "glucose": 140,
  ...
}
```

#### Model Info
```bash
GET /model-info
```

### ⚠️ Medical Disclaimer
This is an AI-based risk estimation tool, NOT a medical diagnosis. Always consult healthcare professionals for medical advice.

### 📈 Feature Engineering
- Age × BMI interaction
- Glucose × BMI interaction
- High-risk age indicator (≥45)
- Obesity indicator (BMI ≥30)
- Prediabetic glucose range (100-125)
- Diabetic glucose level (≥126)

### 🏆 Enterprise Features
- Production-ready code
- Comprehensive error handling
- Medical validation
- Explainable AI (feature importance)
- Personalized recommendations
- Structured JSON output
- Integration-ready (Django/Mobile/ERP)

### 📝 License
RoyalSoft Enterprise ML System - Production Use
