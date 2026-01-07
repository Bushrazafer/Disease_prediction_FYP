"""
Google OAuth Setup Script for MediCare
======================================

This script helps you set up Google OAuth for your MediCare application.

STEP 1: Create Google OAuth Credentials
---------------------------------------
1. Go to Google Cloud Console: https://console.cloud.google.com/
2. Create a new project or select existing one
3. Go to "APIs & Services" > "Credentials"
4. Click "Create Credentials" > "OAuth client ID"
5. Select "Web application"
6. Add these Authorized redirect URIs:
   - http://localhost:8000/accounts/google/login/callback/
   - http://127.0.0.1:8000/accounts/google/login/callback/
   
   For production, add:
   - https://yourdomain.com/accounts/google/login/callback/

7. Copy the Client ID and Client Secret

STEP 2: Create .env file
------------------------
Copy .env.example to .env and add your credentials:
    GOOGLE_CLIENT_ID=your-client-id-here
    GOOGLE_CLIENT_SECRET=your-client-secret-here

STEP 3: Run Migrations and Setup
--------------------------------
    python manage.py migrate
    python scripts/setup_google_oauth.py

"""

import os
import sys
import django

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')

# Load .env file before Django setup
from dotenv import load_dotenv
load_dotenv()

django.setup()

from django.contrib.sites.models import Site
from allauth.socialaccount.models import SocialApp
from django.conf import settings


def setup_site():
    """Configure the Django Site for allauth"""
    site, created = Site.objects.get_or_create(id=1)
    site.domain = 'localhost:8000'
    site.name = 'MediCare AI'
    site.save()
    
    if created:
        print("✅ Created Site: localhost:8000")
    else:
        print("✅ Updated Site: localhost:8000")
    
    return site


def setup_google_oauth(site):
    """Configure Google OAuth Social App"""
    client_id = settings.GOOGLE_CLIENT_ID or os.environ.get('GOOGLE_CLIENT_ID', '')
    client_secret = settings.GOOGLE_CLIENT_SECRET or os.environ.get('GOOGLE_CLIENT_SECRET', '')
    
    if not client_id or not client_secret:
        print("\n⚠️  WARNING: Google OAuth credentials not found!")
        print("   Please create a .env file with:")
        print("   GOOGLE_CLIENT_ID=your-client-id")
        print("   GOOGLE_CLIENT_SECRET=your-client-secret")
        print("   See .env.example for reference.\n")
        return None
    
    # Check if Google app already exists
    google_app = SocialApp.objects.filter(provider='google').first()
    
    if google_app:
        google_app.client_id = client_id
        google_app.secret = client_secret
        google_app.save()
        print("✅ Updated Google OAuth App")
    else:
        google_app = SocialApp.objects.create(
            provider='google',
            name='Google',
            client_id=client_id,
            secret=client_secret
        )
        print("✅ Created Google OAuth App")
    
    # Link to site
    if site not in google_app.sites.all():
        google_app.sites.add(site)
        print("✅ Linked Google OAuth to Site")
    
    return google_app


def main():
    print("\n" + "="*50)
    print("  MediCare Google OAuth Setup")
    print("="*50 + "\n")
    
    # Setup site
    site = setup_site()
    
    # Setup Google OAuth
    setup_google_oauth(site)
    
    print("\n" + "="*50)
    print("  Setup Complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Create .env file with GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET")
    print("   (copy from .env.example)")
    print("2. Run: python manage.py migrate")
    print("3. Run: python manage.py runserver")
    print("4. Test Google login at: http://localhost:8000/login/")
    print("")


if __name__ == '__main__':
    main()
