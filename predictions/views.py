"""
MediCare AI Views - Dynamic CMS + ML Model Integration
"""

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import logging
import numpy as np

from .models import (
    Prediction, DiseaseInfo, DiseasePrevention, 
    DiseaseTreatment, MLModelVersion, DISEASE_CHOICES,
    SiteSettings, AboutSection, TeamMember, ContactMessage, FAQ,
    HeroSection, HeroStat, PredictionCard, WhyChooseUs, HomepageStat, CTASection
)
from .ml_utils import (
    load_model, predict_with_model, detect_model_format,
    validate_features, prepare_features, get_risk_level,
    ModelLoadError, PredictionError
)
# Import enhanced prediction service
from .prediction_service import make_enhanced_prediction
# Import auth decorators
from .auth_views import login_required_for_prediction, api_login_required

logger = logging.getLogger(__name__)

# Disease name mapping for URL slugs
DISEASE_SLUG_MAP = {
    'diabetes': 'diabetes',
    'cardiovascular': 'cardiovascular',
    'kidney': 'kidney',
    'breast-cancer': 'breast_cancer',
    'breast_cancer': 'breast_cancer',
    'depression': 'depression',
    'obesity': 'obesity',
}

DISEASE_DISPLAY_NAMES = {
    'diabetes': 'Diabetes',
    'cardiovascular': 'Cardiovascular Disease',
    'kidney': 'Kidney Disease',
    'breast_cancer': 'Breast Cancer',
    'depression': 'Depression',
    'obesity': 'Obesity',
}


# =========================================================
# BASIC PAGE VIEWS
# =========================================================

def get_site_context():
    """Get common site context for all pages"""
    return {
        'site_settings': SiteSettings.get_settings(),
    }


def home(request):
    """Home page view with dynamic CMS content"""
    context = get_site_context()
    context['hero'] = HeroSection.get_hero()
    context['hero_stats'] = HeroStat.objects.filter(is_active=True)
    context['prediction_cards'] = PredictionCard.objects.filter(is_active=True)
    context['why_choose_us'] = WhyChooseUs.objects.filter(is_active=True)
    context['homepage_stats'] = HomepageStat.objects.filter(is_active=True)
    context['cta'] = CTASection.get_cta()
    return render(request, 'index.html', context)


def about(request):
    """About page view with CMS content"""
    context = get_site_context()
    context['about_sections'] = AboutSection.objects.filter(is_active=True)
    context['team_members'] = TeamMember.objects.filter(is_active=True)
    context['faqs'] = FAQ.objects.filter(is_active=True)
    return render(request, 'about.html', context)


def contact(request):
    """Contact page view with form submission"""
    context = get_site_context()
    
    if request.method == 'POST':
        try:
            ContactMessage.objects.create(
                name=request.POST.get('name', ''),
                email=request.POST.get('email', ''),
                phone=request.POST.get('phone', ''),
                subject=request.POST.get('subject', ''),
                message=request.POST.get('message', ''),
            )
            context['success_message'] = 'Thank you! Your message has been sent successfully.'
        except Exception as e:
            context['error_message'] = 'Sorry, there was an error sending your message. Please try again.'
    
    return render(request, 'contact.html', context)


# =========================================================
# CMS CONTENT VIEWS - DYNAMIC DISEASE INFORMATION
# =========================================================

def get_disease_content(disease_slug, content_type):
    """Helper to fetch disease content from ORM"""
    disease_name = DISEASE_SLUG_MAP.get(disease_slug, disease_slug)
    
    model_map = {
        'info': DiseaseInfo,
        'prevention': DiseasePrevention,
        'treatment': DiseaseTreatment,
    }
    
    model = model_map.get(content_type)
    if not model:
        return []
    
    return model.objects.filter(
        disease_name=disease_name,
        is_active=True
    ).order_by('order')


def disease_content_view(request, disease_slug, content_type):
    """Generic view for disease content pages"""
    disease_name = DISEASE_SLUG_MAP.get(disease_slug, disease_slug)
    display_name = DISEASE_DISPLAY_NAMES.get(disease_name, disease_name.title())
    
    content_items = get_disease_content(disease_slug, content_type)
    
    content_type_titles = {
        'info': 'Information',
        'prevention': 'Prevention Guidelines',
        'treatment': 'Treatment Options',
    }
    
    context = {
        'disease_name': disease_name,
        'disease_slug': disease_slug,
        'display_name': display_name,
        'content_type': content_type,
        'content_type_title': content_type_titles.get(content_type, content_type.title()),
        'content_items': content_items,
        'has_content': content_items.exists(),
        'fallback_message': "Content will be updated soon. Please check back later.",
    }
    
    return render(request, 'predictions/disease_content.html', context)


# Individual disease content views
def diabetes_info(request):
    return disease_content_view(request, 'diabetes', 'info')

def diabetes_prevention(request):
    return disease_content_view(request, 'diabetes', 'prevention')

def diabetes_treatment(request):
    return disease_content_view(request, 'diabetes', 'treatment')

def cardiovascular_info(request):
    return disease_content_view(request, 'cardiovascular', 'info')

def cardiovascular_prevention(request):
    return disease_content_view(request, 'cardiovascular', 'prevention')

def cardiovascular_treatment(request):
    return disease_content_view(request, 'cardiovascular', 'treatment')

def kidney_info(request):
    return disease_content_view(request, 'kidney', 'info')

def kidney_prevention(request):
    return disease_content_view(request, 'kidney', 'prevention')

def kidney_treatment(request):
    return disease_content_view(request, 'kidney', 'treatment')

def breast_cancer_info(request):
    return disease_content_view(request, 'breast_cancer', 'info')

def breast_cancer_prevention(request):
    return disease_content_view(request, 'breast_cancer', 'prevention')

def breast_cancer_treatment(request):
    return disease_content_view(request, 'breast_cancer', 'treatment')

def depression_info(request):
    return disease_content_view(request, 'depression', 'info')

def depression_prevention(request):
    return disease_content_view(request, 'depression', 'prevention')

def depression_treatment(request):
    return disease_content_view(request, 'depression', 'treatment')

def obesity_info(request):
    return disease_content_view(request, 'obesity', 'info')

def obesity_prevention(request):
    return disease_content_view(request, 'obesity', 'prevention')

def obesity_treatment(request):
    return disease_content_view(request, 'obesity', 'treatment')


# =========================================================
# PREDICTION PAGE VIEWS - DYNAMIC FORM GENERATION
# =========================================================

def get_prediction_context(disease_name):
    """Get context for prediction page including active model schema"""
    active_model = MLModelVersion.get_active_model(disease_name)
    
    context = {
        'disease_name': disease_name,
        'display_name': DISEASE_DISPLAY_NAMES.get(disease_name, disease_name.title()),
        'has_model': active_model is not None,
        'model_version': None,
        'feature_schema': [],
        'feature_types': {},
        'feature_options': {},
    }
    
    if active_model:
        context.update({
            'model_version': active_model.version,
            'feature_schema': active_model.feature_schema or [],
            'feature_types': active_model.feature_types or {},
            'feature_options': active_model.feature_options or {},
        })
    
    return context


@login_required_for_prediction
def diabetes_prediction(request):
    context = get_prediction_context('diabetes')
    context['info_url'] = 'diabetes_info'
    context['prevention_url'] = 'diabetes_prevention'
    context['treatment_url'] = 'diabetes_treatment'
    return render(request, 'predictions/diabetes.html', context)


@login_required_for_prediction
@login_required_for_prediction
def cardiovascular_prediction(request):
    context = get_prediction_context('cardiovascular')
    context['info_url'] = 'cardiovascular_info'
    context['prevention_url'] = 'cardiovascular_prevention'
    context['treatment_url'] = 'cardiovascular_treatment'
    return render(request, 'predictions/cardiovascular.html', context)


@login_required_for_prediction
def kidney_prediction(request):
    context = get_prediction_context('kidney')
    context['info_url'] = 'kidney_info'
    context['prevention_url'] = 'kidney_prevention'
    context['treatment_url'] = 'kidney_treatment'
    return render(request, 'predictions/kidney.html', context)


@login_required_for_prediction
def breast_cancer_prediction(request):
    context = get_prediction_context('breast_cancer')
    context['info_url'] = 'breast_cancer_info'
    context['prevention_url'] = 'breast_cancer_prevention'
    context['treatment_url'] = 'breast_cancer_treatment'
    # Mark as clinician-only tool
    context['clinician_only'] = True
    context['clinician_warning'] = (
        'This tool analyzes Fine Needle Aspiration (FNA) biopsy cell measurements '
        'from pathology lab reports. These are imaging-derived features from professional '
        'cell analysis, NOT patient-providable values. This tool is intended for healthcare '
        'professionals only.'
    )
    return render(request, 'predictions/breast_cancer.html', context)


@login_required_for_prediction
def depression_prediction(request):
    context = get_prediction_context('depression')
    context['info_url'] = 'depression_info'
    context['prevention_url'] = 'depression_prevention'
    context['treatment_url'] = 'depression_treatment'
    # Add crisis resources for mental health
    context['crisis_resources'] = [
        {'name': 'National Helpline (India)', 'number': '1800-599-0019'},
        {'name': 'iCall', 'number': '9152987821'},
        {'name': 'Vandrevala Foundation', 'number': '1860-2662-345'},
        {'name': 'National Suicide Prevention Lifeline (US)', 'number': '988'},
    ]
    context['show_crisis_warning'] = True
    return render(request, 'predictions/depression.html', context)


@login_required_for_prediction
def obesity_prediction(request):
    context = get_prediction_context('obesity')
    context['info_url'] = 'obesity_info'
    context['prevention_url'] = 'obesity_prevention'
    context['treatment_url'] = 'obesity_treatment'
    return render(request, 'predictions/obesity.html', context)


# =========================================================
# API ENDPOINTS - ML PREDICTION WITH DYNAMIC SCHEMA
# =========================================================

def make_prediction(disease_name, data, user=None):
    """
    Core prediction function using active ML model
    
    Returns dict with: success, prediction, probability, risk_level, message
    """
    # Get active model
    active_model = MLModelVersion.get_active_model(disease_name)
    
    if not active_model:
        # Fallback to mock prediction if no model configured
        return make_mock_prediction(disease_name, data)
    
    try:
        # Validate features against schema
        is_valid, errors = validate_features(
            data,
            active_model.feature_schema,
            active_model.feature_types
        )
        
        if not is_valid:
            return {
                'success': False,
                'error': f"Invalid input: {'; '.join(errors)}"
            }
        
        # Prepare features in correct order
        features = prepare_features(
            data,
            active_model.feature_schema,
            active_model.feature_types
        )
        
        # Load model
        model = load_model(active_model.file_path)
        model_format = detect_model_format(active_model.file_path)
        
        # Load scaler if exists (for diabetes and cardiovascular models)
        scaler = None
        power_transformer = None
        imputer = None
        
        # Special handling for CKD - features are already properly encoded
        if disease_name == 'kidney':
            # CKD features are already encoded and ready for prediction
            # Skip preprocessing as it causes issues with categorical data
            logger.info("Skipping preprocessing for CKD model - features already prepared")
            pass
        
        elif disease_name in ['diabetes', 'cardiovascular', 'breast_cancer', 'depression', 'obesity']:
            scaler_path = active_model.file_path.replace('model.pkl', 'scaler.pkl')
            imputer_path = active_model.file_path.replace('model.pkl', 'imputer.pkl')
            
            try:
                from predictions.ml_utils import load_sklearn_model, get_model_path
                scaler_data = load_sklearn_model(get_model_path(scaler_path))
                
                # Handle both old and new scaler format
                if isinstance(scaler_data, dict):
                    scaler = scaler_data.get('scaler')
                    power_transformer = scaler_data.get('power_transformer')
                else:
                    scaler = scaler_data
                
                # Load imputer if exists
                try:
                    imputer = load_sklearn_model(get_model_path(imputer_path))
                except:
                    pass
                
                # Apply imputer first
                if imputer is not None:
                    features = imputer.transform(features)
                
                if scaler is not None:
                    features = scaler.transform(features)
                if power_transformer is not None:
                    features = power_transformer.transform(features)
                    
            except Exception as e:
                logger.warning(f"Could not load scaler: {e}")
        
        # Make prediction
        prediction, probability_array = predict_with_model(model, features, model_format)
        
        # Handle multi-class obesity prediction
        if disease_name == 'obesity':
            try:
                from predictions.ml_utils import load_sklearn_model, get_model_path
                encoder_path = active_model.file_path.replace('model.pkl', 'label_encoder.pkl')
                label_encoder = load_sklearn_model(get_model_path(encoder_path))
                
                class_name = label_encoder.inverse_transform([prediction[0]])[0]
                
                # Get max probability for the predicted class
                if probability_array is not None:
                    probability = float(max(probability_array[0])) * 100
                else:
                    probability = 100.0
                
                # Determine risk level based on obesity class
                obesity_risk_map = {
                    'Insufficient_Weight': 'low',
                    'Normal_Weight': 'low',
                    'Overweight_Level_I': 'medium',
                    'Overweight_Level_II': 'medium',
                    'Obesity_Type_I': 'high',
                    'Obesity_Type_II': 'high',
                    'Obesity_Type_III': 'high'
                }
                risk_level = obesity_risk_map.get(class_name, 'medium')
                
                # Save prediction
                if user and user.is_authenticated:
                    Prediction.objects.create(
                        user=user,
                        prediction_type=disease_name,
                        input_data=data,
                        result={
                            'prediction': int(prediction[0]),
                            'class_name': class_name,
                            'probability': probability,
                            'risk_level': risk_level
                        },
                        probability=probability,
                        risk_level=risk_level,
                        model_version=active_model.name
                    )
                
                return {
                    'success': True,
                    'prediction': int(prediction[0]),
                    'class_name': class_name,
                    'display_name': class_name.replace('_', ' '),
                    'probability': round(probability, 2),
                    'risk_level': risk_level,
                    'model_version': active_model.version,
                    'message': f'Obesity Level: {class_name.replace("_", " ")}'
                }
            except Exception as e:
                logger.error(f"Error in obesity prediction: {e}")
        
        # Extract probability for binary classification
        if probability_array is not None:
            if len(probability_array.shape) > 1:
                probability = float(probability_array[0][1]) * 100
            else:
                probability = float(probability_array[0]) * 100
        else:
            probability = float(prediction[0]) * 100
        
        risk_level = get_risk_level(probability)
        
        # Save prediction to database
        if user and user.is_authenticated:
            Prediction.objects.create(
                user=user,
                prediction_type=disease_name,
                input_data=data,
                result={
                    'prediction': int(prediction[0]),
                    'probability': probability,
                    'risk_level': risk_level
                },
                probability=probability,
                risk_level=risk_level,
                model_version=active_model.name
            )
        
        return {
            'success': True,
            'prediction': int(prediction[0]),
            'probability': round(probability, 2),
            'risk_level': risk_level,
            'model_version': active_model.version,
            'message': f'Risk assessment completed. Risk level: {risk_level.title()}'
        }
        
    except ModelLoadError as e:
        logger.error(f"Model load error for {disease_name}: {e}")
        return make_mock_prediction(disease_name, data)
        
    except PredictionError as e:
        logger.error(f"Prediction error for {disease_name}: {e}")
        return {'success': False, 'error': str(e)}
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction for {disease_name}: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': 'An unexpected error occurred'}


def make_mock_prediction(disease_name, data):
    """Fallback mock prediction when no ML model is available"""
    # Simple risk calculation based on common factors
    risk_score = 0
    
    age = float(data.get('age', 30))
    if age > 50:
        risk_score += 20
    elif age > 40:
        risk_score += 10
    
    bmi = float(data.get('bmi', 0) or 0)
    if bmi > 30:
        risk_score += 25
    elif bmi > 25:
        risk_score += 15
    
    # Add some randomness for demo
    risk_score += np.random.randint(10, 30)
    
    probability = min(risk_score, 95)
    risk_level = get_risk_level(probability)
    prediction = 1 if probability > 50 else 0
    
    return {
        'success': True,
        'prediction': prediction,
        'probability': round(probability, 2),
        'risk_level': risk_level,
        'model_version': 'mock',
        'message': f'Risk assessment completed (demo mode). Risk level: {risk_level.title()}'
    }


# =========================================================
# API PREDICTION ENDPOINTS - Using Enhanced Prediction Service
# =========================================================

def _get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


@api_login_required
@require_http_methods(["POST"])
def api_diabetes_prediction(request):
    """API endpoint for diabetes prediction"""
    try:
        data = json.loads(request.body)
        result = make_enhanced_prediction(
            'diabetes', 
            data, 
            request.user,
            session_id=request.session.session_key,
            ip_address=_get_client_ip(request),
            disclaimer_acknowledged=data.get('disclaimer_acknowledged', False),
        )
        return JsonResponse(result)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@api_login_required
@require_http_methods(["POST"])
def api_cardiovascular_prediction(request):
    """API endpoint for cardiovascular prediction"""
    try:
        data = json.loads(request.body)
        result = make_enhanced_prediction(
            'cardiovascular', 
            data, 
            request.user,
            session_id=request.session.session_key,
            ip_address=_get_client_ip(request),
            disclaimer_acknowledged=data.get('disclaimer_acknowledged', False),
        )
        return JsonResponse(result)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@api_login_required
@require_http_methods(["POST"])
def api_kidney_prediction(request):
    """API endpoint for kidney disease prediction"""
    try:
        data = json.loads(request.body)
        result = make_enhanced_prediction(
            'kidney', 
            data, 
            request.user,
            session_id=request.session.session_key,
            ip_address=_get_client_ip(request),
            disclaimer_acknowledged=data.get('disclaimer_acknowledged', False),
        )
        return JsonResponse(result)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@api_login_required
@require_http_methods(["POST"])
def api_breast_cancer_prediction(request):
    """API endpoint for breast cancer prediction (Clinician-only)"""
    try:
        data = json.loads(request.body)
        result = make_enhanced_prediction(
            'breast_cancer', 
            data, 
            request.user,
            session_id=request.session.session_key,
            ip_address=_get_client_ip(request),
            disclaimer_acknowledged=data.get('disclaimer_acknowledged', False),
        )
        return JsonResponse(result)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@api_login_required
@require_http_methods(["POST"])
def api_depression_prediction(request):
    """API endpoint for depression prediction"""
    try:
        data = json.loads(request.body)
        result = make_enhanced_prediction(
            'depression', 
            data, 
            request.user,
            session_id=request.session.session_key,
            ip_address=_get_client_ip(request),
            disclaimer_acknowledged=data.get('disclaimer_acknowledged', False),
        )
        return JsonResponse(result)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@api_login_required
@require_http_methods(["POST"])
def api_obesity_prediction(request):
    """API endpoint for obesity prediction"""
    try:
        data = json.loads(request.body)
        result = make_enhanced_prediction(
            'obesity', 
            data, 
            request.user,
            session_id=request.session.session_key,
            ip_address=_get_client_ip(request),
            disclaimer_acknowledged=data.get('disclaimer_acknowledged', False),
        )
        return JsonResponse(result)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# =========================================================
# API - GET FEATURE SCHEMA FOR DYNAMIC FORMS
# =========================================================

@require_http_methods(["GET"])
def api_get_feature_schema(request, disease_name):
    """API endpoint to get feature schema for a disease model"""
    disease_name = DISEASE_SLUG_MAP.get(disease_name, disease_name)
    active_model = MLModelVersion.get_active_model(disease_name)
    
    if not active_model:
        return JsonResponse({
            'success': False,
            'has_model': False,
            'message': 'No active model configured for this disease'
        })
    
    return JsonResponse({
        'success': True,
        'has_model': True,
        'model_name': active_model.name,
        'model_version': active_model.version,
        'feature_schema': active_model.feature_schema,
        'feature_types': active_model.feature_types or {},
        'feature_options': active_model.feature_options or {},
    })


# =========================================================
# GENERIC PREDICTION ENDPOINT
# =========================================================

@api_login_required
@require_http_methods(["POST"])
def api_generic_prediction(request, disease_name):
    """Generic API endpoint for any disease prediction"""
    disease_name = DISEASE_SLUG_MAP.get(disease_name, disease_name)
    
    if disease_name not in dict(DISEASE_CHOICES):
        return JsonResponse({
            'success': False,
            'error': f'Unknown disease type: {disease_name}'
        }, status=400)
    
    try:
        data = json.loads(request.body)
        result = make_enhanced_prediction(
            disease_name, 
            data, 
            request.user,
            session_id=request.session.session_key,
            ip_address=_get_client_ip(request),
            disclaimer_acknowledged=data.get('disclaimer_acknowledged', False),
        )
        return JsonResponse(result)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
