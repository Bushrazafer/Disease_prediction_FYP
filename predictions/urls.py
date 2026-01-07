from django.urls import path
from django.shortcuts import render
from predictions import views
from predictions import auth_views
from allauth.socialaccount.providers.google.views import oauth2_login as google_login_view

urlpatterns = [
    # =========================================================
    # AUTHENTICATION PAGES
    # =========================================================
    path('login/', auth_views.login_view, name='login'),
    path('signup/', auth_views.signup_view, name='signup'),
    path('logout/', auth_views.logout_view, name='logout'),
    path('forgot-password/', auth_views.forgot_password_view, name='forgot_password'),
    path('terms/', auth_views.terms_view, name='terms'),
    path('auth/google/', google_login_view, name='google_login'),  # Google OAuth
    
    # =========================================================
    # MAIN PAGES
    # =========================================================
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    
    # Test page for frontend-backend integration
    path('test/', lambda request: render(request, 'test_frontend_backend.html'), name='test_page'),

    # =========================================================
    # PREDICTION FORM PAGES
    # =========================================================
    path('predict/diabetes/', views.diabetes_prediction, name='diabetes'),
    path('predict/cardiovascular/', views.cardiovascular_prediction, name='cardiovascular'),
    path('predict/kidney/', views.kidney_prediction, name='kidney'),
    path('predict/breast-cancer/', views.breast_cancer_prediction, name='breast_cancer'),
    path('predict/depression/', views.depression_prediction, name='depression'),
    path('predict/obesity/', views.obesity_prediction, name='obesity'),

    # =========================================================
    # API ENDPOINTS - PREDICTIONS
    # =========================================================
    path('api/predict/diabetes/', views.api_diabetes_prediction, name='api_diabetes'),
    path('api/predict/cardiovascular/', views.api_cardiovascular_prediction, name='api_cardiovascular'),
    path('api/predict/kidney/', views.api_kidney_prediction, name='api_kidney'),
    path('api/predict/breast-cancer/', views.api_breast_cancer_prediction, name='api_breast_cancer'),
    path('api/predict/depression/', views.api_depression_prediction, name='api_depression'),
    path('api/predict/obesity/', views.api_obesity_prediction, name='api_obesity'),
    
    # Generic prediction endpoint
    path('api/predict/<str:disease_name>/', views.api_generic_prediction, name='api_generic_prediction'),

    # =========================================================
    # API ENDPOINTS - FEATURE SCHEMA
    # =========================================================
    path('api/schema/<str:disease_name>/', views.api_get_feature_schema, name='api_feature_schema'),

    # =========================================================
    # DISEASE INFORMATION PAGES (CMS-DRIVEN)
    # =========================================================
    # Diabetes
    path('diabetes/info/', views.diabetes_info, name='diabetes_info'),
    path('diabetes/prevention/', views.diabetes_prevention, name='diabetes_prevention'),
    path('diabetes/treatment/', views.diabetes_treatment, name='diabetes_treatment'),

    # Cardiovascular
    path('cardiovascular/info/', views.cardiovascular_info, name='cardiovascular_info'),
    path('cardiovascular/prevention/', views.cardiovascular_prevention, name='cardiovascular_prevention'),
    path('cardiovascular/treatment/', views.cardiovascular_treatment, name='cardiovascular_treatment'),

    # Kidney
    path('kidney/info/', views.kidney_info, name='kidney_info'),
    path('kidney/prevention/', views.kidney_prevention, name='kidney_prevention'),
    path('kidney/treatment/', views.kidney_treatment, name='kidney_treatment'),

    # Breast Cancer
    path('breast-cancer/info/', views.breast_cancer_info, name='breast_cancer_info'),
    path('breast-cancer/prevention/', views.breast_cancer_prevention, name='breast_cancer_prevention'),
    path('breast-cancer/treatment/', views.breast_cancer_treatment, name='breast_cancer_treatment'),

    # Depression
    path('depression/info/', views.depression_info, name='depression_info'),
    path('depression/prevention/', views.depression_prevention, name='depression_prevention'),
    path('depression/treatment/', views.depression_treatment, name='depression_treatment'),

    # Obesity
    path('obesity/info/', views.obesity_info, name='obesity_info'),
    path('obesity/prevention/', views.obesity_prevention, name='obesity_prevention'),
    path('obesity/treatment/', views.obesity_treatment, name='obesity_treatment'),
]
