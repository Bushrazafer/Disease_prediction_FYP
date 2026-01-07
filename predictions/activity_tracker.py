"""
User Activity Tracking Utility
Tracks all user activities on the MediCare website
"""

from django.utils import timezone
from functools import wraps
import re


def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


def get_user_agent(request):
    """Get user agent string"""
    return request.META.get('HTTP_USER_AGENT', '')


def get_referrer(request):
    """Get referrer URL"""
    return request.META.get('HTTP_REFERER', '')


def detect_device_type(user_agent):
    """Detect device type from user agent"""
    user_agent = user_agent.lower()
    if 'mobile' in user_agent or 'android' in user_agent:
        if 'tablet' in user_agent or 'ipad' in user_agent:
            return 'tablet'
        return 'mobile'
    elif 'tablet' in user_agent or 'ipad' in user_agent:
        return 'tablet'
    return 'desktop'


def detect_browser(user_agent):
    """Detect browser from user agent"""
    user_agent = user_agent.lower()
    if 'chrome' in user_agent and 'edg' not in user_agent:
        return 'Chrome'
    elif 'firefox' in user_agent:
        return 'Firefox'
    elif 'safari' in user_agent and 'chrome' not in user_agent:
        return 'Safari'
    elif 'edg' in user_agent:
        return 'Edge'
    elif 'opera' in user_agent or 'opr' in user_agent:
        return 'Opera'
    return 'Unknown'


def detect_os(user_agent):
    """Detect OS from user agent"""
    user_agent = user_agent.lower()
    if 'windows' in user_agent:
        return 'Windows'
    elif 'mac' in user_agent:
        return 'macOS'
    elif 'linux' in user_agent:
        return 'Linux'
    elif 'android' in user_agent:
        return 'Android'
    elif 'iphone' in user_agent or 'ipad' in user_agent:
        return 'iOS'
    return 'Unknown'


def get_page_category(url_path):
    """Determine page category from URL path"""
    url_path = url_path.lower()
    
    if url_path == '/' or url_path == '':
        return 'home'
    elif '/about' in url_path:
        return 'about'
    elif '/contact' in url_path:
        return 'contact'
    elif '/predict/' in url_path:
        return 'prediction'
    elif '/info/' in url_path or '/info' in url_path:
        return 'disease_info'
    elif '/prevention/' in url_path or '/prevention' in url_path:
        return 'disease_prevention'
    elif '/treatment/' in url_path or '/treatment' in url_path:
        return 'disease_treatment'
    elif '/login' in url_path or '/signup' in url_path or '/logout' in url_path:
        return 'auth'
    return 'other'


def get_disease_from_url(url_path):
    """Extract disease name from URL"""
    disease_patterns = [
        'diabetes', 'cardiovascular', 'kidney', 
        'breast-cancer', 'breast_cancer', 'depression', 'obesity'
    ]
    
    url_path = url_path.lower()
    for disease in disease_patterns:
        if disease in url_path:
            # Normalize breast-cancer to breast_cancer
            return disease.replace('-', '_')
    return ''


def log_activity(user, activity_type, request, **kwargs):
    """
    Log user activity to database
    
    Args:
        user: User object
        activity_type: Type of activity (login, logout, page_view, prediction, etc.)
        request: HTTP request object
        **kwargs: Additional data (page_title, disease_name, prediction, extra_data)
    """
    from .models import UserActivity, UserSession, UserProfile
    
    if not user or not user.is_authenticated:
        return None
    
    try:
        # Get request info
        ip_address = get_client_ip(request)
        user_agent = get_user_agent(request)
        referrer = get_referrer(request)
        session_key = request.session.session_key or ''
        
        # Get URL info
        page_url = request.path
        page_category = kwargs.get('page_category') or get_page_category(page_url)
        disease_name = kwargs.get('disease_name') or get_disease_from_url(page_url)
        
        # Create activity log
        activity = UserActivity.objects.create(
            user=user,
            activity_type=activity_type,
            page_category=page_category,
            page_url=page_url,
            page_title=kwargs.get('page_title', ''),
            disease_name=disease_name,
            prediction=kwargs.get('prediction'),
            extra_data=kwargs.get('extra_data'),
            ip_address=ip_address,
            user_agent=user_agent,
            referrer=referrer,
            session_key=session_key,
            time_spent_seconds=kwargs.get('time_spent_seconds'),
        )
        
        # Update user profile stats
        try:
            profile = user.profile
            profile.total_page_views += 1
            if activity_type == 'prediction':
                profile.total_predictions += 1
            profile.last_login_ip = ip_address
            profile.save(update_fields=['total_page_views', 'total_predictions', 'last_login_ip', 'updated_at'])
        except UserProfile.DoesNotExist:
            UserProfile.objects.create(user=user, last_login_ip=ip_address)
        
        # Update or create session
        if session_key:
            session, created = UserSession.objects.get_or_create(
                session_key=session_key,
                defaults={
                    'user': user,
                    'ip_address': ip_address,
                    'user_agent': user_agent,
                    'device_type': detect_device_type(user_agent),
                    'browser': detect_browser(user_agent),
                    'os': detect_os(user_agent),
                }
            )
            if not created:
                session.pages_viewed += 1
                if activity_type == 'prediction':
                    session.predictions_made += 1
                session.last_activity = timezone.now()
                session.save(update_fields=['pages_viewed', 'predictions_made', 'last_activity'])
        
        return activity
        
    except Exception as e:
        # Don't let tracking errors break the main functionality
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error logging activity: {e}")
        return None


def log_login(user, request):
    """Log user login"""
    return log_activity(user, 'login', request, page_title='Login')


def log_logout(user, request):
    """Log user logout"""
    from .models import UserSession
    
    # Mark session as ended
    session_key = request.session.session_key
    if session_key:
        UserSession.objects.filter(session_key=session_key).update(
            is_active=False,
            ended_at=timezone.now()
        )
    
    return log_activity(user, 'logout', request, page_title='Logout')


def log_signup(user, request):
    """Log user signup"""
    return log_activity(user, 'signup', request, page_title='Sign Up')


def log_page_view(user, request, page_title=''):
    """Log page view"""
    return log_activity(user, 'page_view', request, page_title=page_title)


def log_prediction(user, request, prediction, disease_name):
    """Log prediction made"""
    from .models import PredictionHistory, DISEASE_CHOICES
    
    # Create activity log
    activity = log_activity(
        user, 
        'prediction', 
        request, 
        page_title=f'{disease_name.title()} Prediction',
        disease_name=disease_name,
        prediction=prediction,
        extra_data={
            'risk_level': prediction.risk_level,
            'probability': prediction.probability,
        }
    )
    
    # Create prediction history entry
    disease_display = dict(DISEASE_CHOICES).get(disease_name, disease_name.title())
    
    PredictionHistory.objects.create(
        user=user,
        prediction=prediction,
        disease_display_name=disease_display,
        risk_level_display=prediction.risk_level.title() if prediction.risk_level else 'Unknown',
        probability_display=f"{prediction.probability:.1f}%" if prediction.probability else 'N/A',
    )
    
    return activity


def log_contact_submit(user, request, contact_message):
    """Log contact form submission"""
    return log_activity(
        user, 
        'contact_submit', 
        request, 
        page_title='Contact Form',
        extra_data={
            'subject': contact_message.subject,
            'message_id': contact_message.id,
        }
    )


# Decorator for automatic page view tracking
def track_page_view(page_title=''):
    """Decorator to automatically track page views"""
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            response = view_func(request, *args, **kwargs)
            
            # Log page view if user is authenticated
            if request.user.is_authenticated:
                log_page_view(request.user, request, page_title)
            
            return response
        return wrapper
    return decorator


# Middleware for automatic tracking
class ActivityTrackingMiddleware:
    """Middleware to automatically track page views for authenticated users"""
    
    # URLs to exclude from tracking
    EXCLUDE_PATTERNS = [
        r'^/static/',
        r'^/media/',
        r'^/admin/',
        r'^/api/',
        r'^/__debug__/',
        r'\.ico$',
        r'\.css$',
        r'\.js$',
        r'\.png$',
        r'\.jpg$',
        r'\.gif$',
    ]
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.exclude_patterns = [re.compile(p) for p in self.EXCLUDE_PATTERNS]
    
    def __call__(self, request):
        response = self.get_response(request)
        
        # Only track successful page views for authenticated users
        if (request.user.is_authenticated and 
            response.status_code == 200 and
            request.method == 'GET' and
            not self._should_exclude(request.path)):
            
            log_page_view(request.user, request)
        
        return response
    
    def _should_exclude(self, path):
        """Check if path should be excluded from tracking"""
        for pattern in self.exclude_patterns:
            if pattern.search(path):
                return True
        return False
