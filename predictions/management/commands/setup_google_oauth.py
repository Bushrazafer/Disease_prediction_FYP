"""
Management command to setup Google OAuth credentials
Usage: python manage.py setup_google_oauth --client_id=YOUR_ID --client_secret=YOUR_SECRET
"""
from django.core.management.base import BaseCommand
from django.contrib.sites.models import Site
from allauth.socialaccount.models import SocialApp


class Command(BaseCommand):
    help = 'Setup Google OAuth credentials for MediCare'

    def add_arguments(self, parser):
        parser.add_argument('--client_id', type=str, required=True, help='Google OAuth Client ID')
        parser.add_argument('--client_secret', type=str, required=True, help='Google OAuth Client Secret')

    def handle(self, *args, **options):
        client_id = options['client_id']
        client_secret = options['client_secret']

        # Setup Site
        site, created = Site.objects.get_or_create(id=1)
        site.domain = '127.0.0.1:8000'
        site.name = 'MediCare AI'
        site.save()
        self.stdout.write(self.style.SUCCESS(f'✅ Site configured: {site.domain}'))

        # Setup Google OAuth App
        google_app, created = SocialApp.objects.get_or_create(
            provider='google',
            defaults={
                'name': 'Google',
                'client_id': client_id,
                'secret': client_secret,
            }
        )
        
        if not created:
            google_app.client_id = client_id
            google_app.secret = client_secret
            google_app.save()
            self.stdout.write(self.style.SUCCESS('✅ Updated Google OAuth credentials'))
        else:
            self.stdout.write(self.style.SUCCESS('✅ Created Google OAuth App'))

        # Link to site
        if site not in google_app.sites.all():
            google_app.sites.add(site)
            self.stdout.write(self.style.SUCCESS('✅ Linked Google OAuth to Site'))

        self.stdout.write(self.style.SUCCESS('\n🎉 Google OAuth setup complete!'))
        self.stdout.write('Test at: http://127.0.0.1:8000/login/')
