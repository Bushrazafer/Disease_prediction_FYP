# RoyalSoft ML Engine - Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (30 seconds)
```bash
pip install -r requirements.txt
```

### Step 2: Verify System (10 seconds)
```bash
python verify_system.py
```

### Step 3: Make a Prediction (5 seconds)
```python
from predict import DiabetesPredictionEngine

engine = DiabetesPredictionEngine()

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

result = engine.predict(patient)
print(result)
```

## 📋 Common Commands

### Training
```bash
python train_model.py          # Train new model
```

### Testing
```bash
python test_predictions.py     # Run all tests
python verify_system.py        # System check
python integration_examples.py # Integration examples
```

### API Server
```bash
python api.py                  # Start Flask server
# Access at http://localhost:5000
```

## 🔧 API Usage

### cURL Example
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "glucose": 140,
    "blood_pressure": 85,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 32.5,
    "diabetes_pedigree": 0.8,
    "pregnancies": 2
  }'
```

### Python Requests
```python
import requests

response = requests.post('http://localhost:5000/predict', json={
    "age": 45,
    "glucose": 140,
    "blood_pressure": 85,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 32.5,
    "diabetes_pedigree": 0.8,
    "pregnancies": 2
})

print(response.json())
```

### JavaScript/Fetch
```javascript
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    age: 45,
    glucose: 140,
    blood_pressure: 85,
    skin_thickness: 25,
    insulin: 100,
    bmi: 32.5,
    diabetes_pedigree: 0.8,
    pregnancies: 2
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

## 📊 Input Requirements

| Field | Type | Range | Required |
|-------|------|-------|----------|
| age | number | 18-120 | ✅ |
| glucose | number | 50-400 | ✅ |
| blood_pressure | number | 80-200 | ✅ |
| skin_thickness | number | 0-100 | ✅ |
| insulin | number | 0-300 | ✅ |
| bmi | number | 10-60 | ✅ |
| diabetes_pedigree | number | 0-3 | ✅ |
| pregnancies | number | 0-20 | ✅ |

## 📤 Output Format

```json
{
  "success": true,
  "prediction": 0 or 1,
  "probability": 0-100,
  "confidence": 0-100,
  "risk_level": "low|medium-low|medium|medium-high|high",
  "message": "...",
  "feature_importance": [...],
  "recommendations": [...],
  "model_info": {...}
}
```

## ⚡ Performance

- **Average**: 18ms per prediction
- **Throughput**: ~55 predictions/second
- **Memory**: ~50MB loaded model

## 🔍 Troubleshooting

### Model Not Found
```bash
python train_model.py  # Retrain model
```

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

### Port Already in Use
```bash
# Change port in api.py or use:
python api.py --port 5001
```

## 📚 Documentation

- **README.md** - Overview and usage
- **DEPLOYMENT_GUIDE.md** - Production deployment
- **PROJECT_SUMMARY.md** - Complete project details
- **integration_examples.py** - Code examples

## 🎯 Quick Examples

### High Risk Patient
```python
high_risk = {
    "age": 55, "glucose": 180, "blood_pressure": 90,
    "skin_thickness": 30, "insulin": 150, "bmi": 35.5,
    "diabetes_pedigree": 1.5, "pregnancies": 5
}
# Expected: High risk, probability > 70%
```

### Low Risk Patient
```python
low_risk = {
    "age": 25, "glucose": 85, "blood_pressure": 80,
    "skin_thickness": 20, "insulin": 80, "bmi": 22.0,
    "diabetes_pedigree": 0.2, "pregnancies": 0
}
# Expected: Low risk, probability < 25%
```

### Invalid Input
```python
invalid = {
    "age": 45, "glucose": 500  # Missing fields + invalid glucose
}
# Expected: Error with validation message
```

## 🚨 Error Codes

| Error | Meaning | Solution |
|-------|---------|----------|
| Missing required field | Field not provided | Include all 8 fields |
| Invalid range | Value out of bounds | Check medical ranges |
| Must be numeric | Non-numeric value | Use numbers only |
| Model not found | Model files missing | Run train_model.py |

## 💡 Tips

1. **Always validate input** before sending to API
2. **Cache the engine** - initialize once, reuse many times
3. **Log predictions** for monitoring and auditing
4. **Handle errors gracefully** - check `success` field
5. **Include disclaimer** - this is not medical diagnosis

## 📞 Support

- **Documentation**: See README.md and DEPLOYMENT_GUIDE.md
- **Examples**: Run integration_examples.py
- **Testing**: Run test_predictions.py
- **Verification**: Run verify_system.py

---

**Ready to go! 🎉**

```bash
python verify_system.py  # Verify everything works
python predict.py        # Test prediction
python api.py            # Start API server
```
