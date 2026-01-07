"""
ASGI config for mediCare project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediCare.settings')

application = get_asgi_application()


# if __name__ == "__main__":
#     print("ASGI application loaded successfully!")
#     print(f"DJANGO_SETTINGS_MODULE: {os.environ.get('DJANGO_SETTINGS_MODULE')}")
#     print(f"Application object: {application}")