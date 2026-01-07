"""
Django Signals for MediCare
Handles social account login events and user profile management
"""

from django.dispatch import receiver
from django.contrib.auth.models import User
from allauth.account.signals import user_logged_in, user_signed_up
from allauth.socialaccount.signals import social_account_added, pre_social_login
from allauth.socialaccount.models import SocialLogin

from .activity_tracker import log_login, log_signup, get_client_ip
from .models import UserProfile


@receiver(user_logged_in)
def handle_user_logged_in(sender, request, user, **kwargs):
    """
    Handle user login event from allauth (includes Google OAuth)
    """
    # Check if this is a social login
    sociallogin = kwargs.get('sociallogin')
    
    if sociallogin:
        # This is a Google OAuth login
        provider = sociallogin.account.provider
        log_login(user, request)
        
        # Update user profile with social login info
        _update_profile_from_social(user, sociallogin, request)
    else:
        # Regular login - already handled in auth_views.login_view
        pass


@receiver(user_signed_up)
def handle_user_signed_up(sender, request, user, **kwargs):
    """
    Handle new user signup from allauth (includes Google OAuth)
    """
    sociallogin = kwargs.get('sociallogin')
    
    if sociallogin:
        # This is a Google OAuth signup
        log_signup(user, request)
        
        # Create/update user profile with social login info
        _update_profile_from_social(user, sociallogin, request)
    else:
        # Regular signup - already handled in auth_views.signup_view
        pass


@receiver(social_account_added)
def handle_social_account_added(sender, request, sociallogin, **kwargs):
    """
    Handle when a social account is connected to an existing user
    """
    user = sociallogin.user
    _update_profile_from_social(user, sociallogin, request)


@receiver(pre_social_login)
def handle_pre_social_login(sender, request, sociallogin, **kwargs):
    """
    Handle pre-social login - can be used to link accounts or validate
    """
    # If user exists with same email, connect the social account
    if sociallogin.is_existing:
        return
    
    # Check if a user with this email already exists
    email = sociallogin.account.extra_data.get('email')
    if email:
        try:
            existing_user = User.objects.get(email=email)
            # Connect the social account to existing user
            sociallogin.connect(request, existing_user)
        except User.DoesNotExist:
            pass


def _update_profile_from_social(user, sociallogin, request):
    """
    Update user profile with data from social login
    """
    try:
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        # Get extra data from social account
        extra_data = sociallogin.account.extra_data
        
        # Update profile fields
        profile.last_login_ip = get_client_ip(request)
        profile.social_provider = sociallogin.account.provider
        profile.social_uid = sociallogin.account.uid
        
        # Get profile picture from Google
        if sociallogin.account.provider == 'google':
            picture_url = extra_data.get('picture', '')
            if picture_url:
                profile.avatar_url = picture_url
        
        profile.save()
        
        # Update user's name if not set
        if not user.first_name:
            user.first_name = extra_data.get('given_name', '')
        if not user.last_name:
            user.last_name = extra_data.get('family_name', '')
        if user.first_name or user.last_name:
            user.save(update_fields=['first_name', 'last_name'])
            
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error updating profile from social login: {e}")
