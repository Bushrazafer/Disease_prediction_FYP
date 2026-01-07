"""
MediCare AI Authentication Views
Real working backend with Django authentication
"""

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from functools import wraps
import re

# Import activity tracker
from .activity_tracker import log_login, log_logout, log_signup


def login_required_for_prediction(view_func):
    """
    Custom decorator - redirects to login if user tries to access prediction pages
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.warning(request, 'Please login or sign up to use health predictions.')
            return redirect(f'/login/?next={request.path}')
        return view_func(request, *args, **kwargs)
    return wrapper


def api_login_required(view_func):
    """
    Decorator for API endpoints - returns JSON error if not authenticated
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({
                'success': False,
                'error': 'Authentication required',
                'login_required': True,
                'message': 'Please login or sign up to use health predictions.'
            }, status=401)
        return view_func(request, *args, **kwargs)
    return wrapper


def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """
    Validate password strength
    Returns (is_valid, error_message)
    """
    if len(password) < 8:
        return False, 'Password must be at least 8 characters.'
    if not re.search(r'[A-Za-z]', password):
        return False, 'Password must contain at least one letter.'
    if not re.search(r'\d', password):
        return False, 'Password must contain at least one number.'
    return True, None


def login_view(request):
    """Handle user login - Real backend authentication"""
    # Already logged in? Go home
    if request.user.is_authenticated:
        return redirect('home')
    
    # Get redirect URL (where to go after login)
    next_url = request.GET.get('next', '')
    
    if request.method == 'POST':
        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '')
        
        # Basic validation
        if not email or not password:
            messages.error(request, 'Please enter both email and password.')
            return render(request, 'auth/login.html', {'next': next_url})
        
        # Find user by email
        try:
            user_obj = User.objects.get(email=email)
            username = user_obj.username
        except User.DoesNotExist:
            messages.error(request, 'Invalid email or password.')
            return render(request, 'auth/login.html', {'next': next_url})
        
        # Authenticate
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            if user.is_active:
                login(request, user)
                
                # Log login activity
                log_login(user, request)
                
                messages.success(request, f'Welcome back, {user.first_name or user.username}!')
                
                # Redirect to next URL or home
                if next_url and next_url.startswith('/'):
                    return redirect(next_url)
                return redirect('home')
            else:
                messages.error(request, 'Your account is disabled. Please contact support.')
        else:
            messages.error(request, 'Invalid email or password.')
    
    return render(request, 'auth/login.html', {'next': next_url})


def signup_view(request):
    """Handle user registration - Real backend with validation"""
    if request.user.is_authenticated:
        return redirect('home')
    
    next_url = request.GET.get('next', '')
    
    if request.method == 'POST':
        full_name = request.POST.get('full_name', '').strip()
        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')
        terms = request.POST.get('terms')
        
        # Store form data for re-display on error
        form_data = {
            'full_name': full_name,
            'email': email,
            'next': next_url
        }
        
        # Validation
        errors = []
        
        # Name validation
        if not full_name:
            errors.append('Full name is required.')
        elif len(full_name) < 2:
            errors.append('Please enter your full name.')
        
        # Email validation
        if not email:
            errors.append('Email is required.')
        elif not validate_email(email):
            errors.append('Please enter a valid email address.')
        elif User.objects.filter(email=email).exists():
            errors.append('An account with this email already exists. Please login instead.')
        
        # Password validation
        is_valid_password, password_error = validate_password(password)
        if not is_valid_password:
            errors.append(password_error)
        
        if password != confirm_password:
            errors.append('Passwords do not match.')
        
        # Terms validation
        if not terms:
            errors.append('You must agree to the Terms & Conditions.')
        
        # If errors, show them and return
        if errors:
            for error in errors:
                messages.error(request, error)
            return render(request, 'auth/signup.html', form_data)
        
        # Create user
        try:
            # Generate unique username from email
            username = email.split('@')[0]
            base_username = username
            counter = 1
            while User.objects.filter(username=username).exists():
                username = f"{base_username}{counter}"
                counter += 1
            
            # Split full name
            name_parts = full_name.split()
            first_name = name_parts[0] if name_parts else ''
            last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
            
            # Create the user
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )
            
            # Auto login after signup
            login(request, user)
            
            # Log signup activity
            log_signup(user, request)
            
            messages.success(request, f'Welcome to MediCare, {first_name}! Your account has been created.')
            
            # Redirect to next URL or home
            if next_url and next_url.startswith('/'):
                return redirect(next_url)
            return redirect('home')
            
        except Exception as e:
            messages.error(request, 'An error occurred while creating your account. Please try again.')
            return render(request, 'auth/signup.html', form_data)
    
    return render(request, 'auth/signup.html', {'next': next_url})


def logout_view(request):
    """Handle user logout"""
    if request.user.is_authenticated:
        # Log logout activity before logging out
        log_logout(request.user, request)
        logout(request)
        messages.info(request, 'You have been logged out successfully.')
    return redirect('home')


def forgot_password_view(request):
    """Handle forgot password request"""
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        email = request.POST.get('email', '').strip().lower()
        
        # Always show same message for security (don't reveal if email exists)
        messages.success(
            request, 
            'If an account exists with this email, you will receive a password reset link shortly.'
        )
        
        # In production, you would:
        # 1. Check if user exists
        # 2. Generate password reset token
        # 3. Send email with reset link
        # For now, just log it
        if User.objects.filter(email=email).exists():
            # TODO: Implement email sending
            pass
        
        return redirect('login')
    
    return render(request, 'auth/forgot_password.html')


def terms_view(request):
    """Display terms and conditions page"""
    return render(request, 'auth/terms.html')
