# Depression Prediction Model

## Overview
Production-grade ML model for depression risk assessment using student mental health data.

## Model Performance
- **Accuracy**: 85.33%
- **Precision**: 89.28%
- **Recall**: 85.16%
- **F1-Score**: 87.17%
- **AUC-ROC**: 0.9248

## Features (27 total)

### Base Features (13)
| Feature | Type | Description |
|---------|------|-------------|
| gender | categorical | Male/Female |
| age | numeric | Age in years |
| academic_pressure | numeric | 1-5 scale |
| work_pressure | numeric | 0-5 scale |
| cgpa | numeric | Academic performance (0-10) |
| study_satisfaction | numeric | 1-5 scale |
| job_satisfaction | numeric | 0-5 scale |
| sleep_duration | categorical | 0-3 (Less than 5h to More than 8h) |
| dietary_habits | categorical | 0-2 (Unhealthy to Healthy) |
| suicidal_thoughts | boolean | Yes/No |
| work_study_hours | numeric | Hours per day |
| financial_stress | numeric | 1-5 scale |
| family_history | boolean | Family history of mental illness |

### Engineered Features (14)
- sleep_risk, total_pressure, satisfaction_score
- life_balance, high_risk_age, overwork
- high_financial_stress, risk_factor_count
- age_pressure_interaction, sleep_stress_interaction
- academic_risk, depression_risk_score
- protective_factors, vulnerability_index

## Usage

```python
from ml_models.Depression.predict import DepressionPredictor

predictor = DepressionPredictor()
result = predictor.predict({
    'gender': 'male',
    'age': 22,
    'academic_pressure': 4,
    'work_pressure': 0,
    'cgpa': 7.5,
    'study_satisfaction': 3,
    'job_satisfaction': 0,
    'sleep_duration': '5-6 hours',
    'dietary_habits': 'Moderate',
    'suicidal_thoughts': 'No',
    'work_study_hours': 8,
    'financial_stress': 3,
    'family_history': 'No'
})

print(f"Risk: {result['risk_level']}")
print(f"Probability: {result['probability']}%")
```

## API Endpoint
```
POST /api/predict/depression/
```

## Files
- `model.pkl` - Trained stacking ensemble model
- `scaler.pkl` - RobustScaler + PowerTransformer
- `features.pkl` - Feature names list
- `metrics.pkl` - Model performance metrics
- `train_model.py` - Training pipeline
- `predict.py` - Prediction module

## Training
```bash
python ml_models/Depression/train_model.py
```

## Important Note
This model is for informational purposes only and should not replace professional mental health evaluation.
