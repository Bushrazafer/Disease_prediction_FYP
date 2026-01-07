"""
URL configuration for mediCare project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
# from django.urls import path
# from first_project import views
# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('home/', views.home, name='home'),
#     path('home/<slug:slug>', views.home_slug),
#     path('contact/', views.contact, name='contact'),
#     path('contact/<int:id>', views.contactID),
#     path('contact/<phone_number>', views.customerPhone),
#     path('contact/<str:phone_number>', views.customerPhone),
#     # HomePage render html files
#     path('', views.HomePage, name='homepage'),
# ]

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),  # Google OAuth URLs
    path('', include('predictions.urls')),  # Include predictions app URLs
]

# Serve static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
