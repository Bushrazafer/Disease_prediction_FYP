from django.db import models
from django.contrib.auth.models import User

# Disease choices used across models
DISEASE_CHOICES = [
    ('diabetes', 'Diabetes'),
    ('cardiovascular', 'Cardiovascular'),
    ('kidney', 'Kidney Disease'),
    ('breast_cancer', 'Breast Cancer'),
    ('depression', 'Depression'),
    ('obesity', 'Obesity'),
]

SECTION_TYPE_CHOICES = [
    ('info', 'Information'),
    ('prevention', 'Prevention'),
    ('treatment', 'Treatment'),
]


class Prediction(models.Model):
    """Stores prediction history for users with audit trail"""
    
    TIER_CHOICES = [
        ('screening', 'Screening'),
        ('standard', 'Standard'),
        ('confirmation', 'Confirmation'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    prediction_type = models.CharField(max_length=20, choices=DISEASE_CHOICES)
    input_data = models.JSONField()
    result = models.JSONField()
    probability = models.FloatField(null=True, blank=True)
    risk_level = models.CharField(max_length=20, null=True, blank=True)
    model_version = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # New fields for audit trail and compliance
    prediction_tier = models.CharField(
        max_length=20, 
        choices=TIER_CHOICES, 
        null=True, 
        blank=True,
        help_text="Quality tier based on input completeness"
    )
    confidence_score = models.FloatField(
        null=True, 
        blank=True,
        help_text="Confidence score (0-100) based on tier and input quality"
    )
    
    # Consent and disclaimer tracking
    disclaimer_acknowledged = models.BooleanField(
        default=False,
        help_text="User acknowledged the medical disclaimer"
    )
    disclaimer_version = models.CharField(
        max_length=20, 
        default='1.0',
        help_text="Version of disclaimer that was acknowledged"
    )
    consent_timestamp = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="When user acknowledged the disclaimer"
    )
    
    # Validation info
    validation_warnings = models.JSONField(
        null=True, 
        blank=True,
        help_text="Any validation warnings for out-of-range inputs"
    )
    
    # Session tracking
    session_id = models.CharField(
        max_length=100, 
        null=True, 
        blank=True,
        help_text="Session identifier for anonymous users"
    )
    ip_hash = models.CharField(
        max_length=64, 
        null=True, 
        blank=True,
        help_text="Hashed IP for audit (not PII)"
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['prediction_type', 'created_at']),
            models.Index(fields=['user', 'created_at']),
        ]

    def __str__(self):
        tier_str = f" [{self.prediction_tier}]" if self.prediction_tier else ""
        return f"{self.get_prediction_type_display()}{tier_str} - {self.created_at}"


# =========================================================
# PART 1 — DYNAMIC CMS FOR DISEASE INFORMATION
# =========================================================

class DiseaseInfo(models.Model):
    """CMS model for disease information content"""
    disease_name = models.CharField(max_length=20, choices=DISEASE_CHOICES)
    title = models.CharField(max_length=200)
    content = models.TextField(help_text="Rich text content for disease information")
    image = models.ImageField(upload_to='disease_images/', null=True, blank=True)
    section_type = models.CharField(max_length=20, choices=SECTION_TYPE_CHOICES, default='info')
    order = models.PositiveIntegerField(default=0, help_text="Display order")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Disease Information"
        verbose_name_plural = "Disease Information"
        ordering = ['disease_name', 'order']

    def __str__(self):
        return f"{self.get_disease_name_display()} - {self.title}"


class DiseasePrevention(models.Model):
    """CMS model for disease prevention guidelines"""
    disease_name = models.CharField(max_length=20, choices=DISEASE_CHOICES)
    title = models.CharField(max_length=200)
    content = models.TextField(help_text="Rich text content for prevention guidelines")
    image = models.ImageField(upload_to='prevention_images/', null=True, blank=True)
    section_type = models.CharField(max_length=20, choices=SECTION_TYPE_CHOICES, default='prevention')
    order = models.PositiveIntegerField(default=0, help_text="Display order")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Disease Prevention"
        verbose_name_plural = "Disease Prevention"
        ordering = ['disease_name', 'order']

    def __str__(self):
        return f"{self.get_disease_name_display()} Prevention - {self.title}"


class DiseaseTreatment(models.Model):
    """CMS model for disease treatment information"""
    disease_name = models.CharField(max_length=20, choices=DISEASE_CHOICES)
    title = models.CharField(max_length=200)
    content = models.TextField(help_text="Rich text content for treatment options")
    image = models.ImageField(upload_to='treatment_images/', null=True, blank=True)
    section_type = models.CharField(max_length=20, choices=SECTION_TYPE_CHOICES, default='treatment')
    order = models.PositiveIntegerField(default=0, help_text="Display order")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Disease Treatment"
        verbose_name_plural = "Disease Treatments"
        ordering = ['disease_name', 'order']

    def __str__(self):
        return f"{self.get_disease_name_display()} Treatment - {self.title}"


# =========================================================
# PART 2 — ML MODEL VERSIONING SYSTEM
# =========================================================

# =========================================================
# PART 1B — CMS FOR SITE CONTENT (FOOTER, ABOUT, CONTACT)
# =========================================================

# =========================================================
# HOMEPAGE DYNAMIC CONTENT
# =========================================================

class HeroSection(models.Model):
    """Hero section content for homepage"""
    title = models.CharField(max_length=200, default="AI-Powered Health Predictions")
    highlight_text = models.CharField(max_length=100, default="Predictions")
    subtitle = models.TextField(default="Get instant health risk assessments using advanced machine learning algorithms.")
    primary_button_text = models.CharField(max_length=50, default="Start Prediction")
    primary_button_url = models.CharField(max_length=100, default="#predictions")
    secondary_button_text = models.CharField(max_length=50, default="Learn More")
    secondary_button_url = models.CharField(max_length=100, default="/about/")
    is_active = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Hero Section"
        verbose_name_plural = "Hero Section"

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def get_hero(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class HeroStat(models.Model):
    """Stats shown in hero section"""
    value = models.CharField(max_length=20, help_text="e.g., 6, 95%, 24/7")
    label = models.CharField(max_length=50, help_text="e.g., Disease Models")
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Hero Stat"
        verbose_name_plural = "Hero Stats"
        ordering = ['order']

    def __str__(self):
        return f"{self.value} - {self.label}"


class PredictionCard(models.Model):
    """Prediction cards shown on homepage"""
    disease_name = models.CharField(max_length=20, choices=DISEASE_CHOICES, unique=True)
    title = models.CharField(max_length=100)
    description = models.TextField()
    icon_class = models.CharField(max_length=50, default="fas fa-heartbeat")
    color_class = models.CharField(max_length=50, default="primary", help_text="e.g., diabetes, cardiovascular")
    feature_1 = models.CharField(max_length=100, blank=True)
    feature_2 = models.CharField(max_length=100, blank=True)
    feature_3 = models.CharField(max_length=100, blank=True)
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Prediction Card"
        verbose_name_plural = "Prediction Cards"
        ordering = ['order']

    def __str__(self):
        return self.title

    def get_features(self):
        features = []
        if self.feature_1:
            features.append(self.feature_1)
        if self.feature_2:
            features.append(self.feature_2)
        if self.feature_3:
            features.append(self.feature_3)
        return features


class WhyChooseUs(models.Model):
    """Why Choose Us features"""
    title = models.CharField(max_length=100)
    description = models.TextField()
    icon_class = models.CharField(max_length=50, default="fas fa-check")
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Why Choose Us"
        verbose_name_plural = "Why Choose Us"
        ordering = ['order']

    def __str__(self):
        return self.title


class HomepageStat(models.Model):
    """Stats grid on homepage"""
    value = models.CharField(max_length=20, help_text="e.g., 10,000+")
    label = models.CharField(max_length=50, help_text="e.g., Predictions Made")
    color_class = models.CharField(max_length=20, default="primary", help_text="primary, success, warning, info")
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Homepage Stat"
        verbose_name_plural = "Homepage Stats"
        ordering = ['order']

    def __str__(self):
        return f"{self.value} - {self.label}"


class CTASection(models.Model):
    """Call to Action section"""
    title = models.CharField(max_length=200, default="Ready to Check Your Health?")
    subtitle = models.TextField(default="Take the first step towards better health with our AI-powered predictions.")
    button_text = models.CharField(max_length=50, default="Start Now")
    button_url = models.CharField(max_length=100, default="#predictions")
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "CTA Section"
        verbose_name_plural = "CTA Section"

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def get_cta(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class SiteSettings(models.Model):
    """Global site settings - only one record should exist"""
    site_name = models.CharField(max_length=100, default="MediCare AI")
    site_tagline = models.CharField(max_length=255, default="Advanced AI-powered health predictions")
    footer_text = models.TextField(blank=True, help_text="Footer description text")
    footer_disclaimer = models.TextField(blank=True, help_text="Medical disclaimer in footer")
    copyright_text = models.CharField(max_length=255, blank=True)
    contact_email = models.EmailField(blank=True)
    contact_phone = models.CharField(max_length=20, blank=True)
    contact_address = models.TextField(blank=True)
    facebook_url = models.URLField(blank=True)
    twitter_url = models.URLField(blank=True)
    linkedin_url = models.URLField(blank=True)
    instagram_url = models.URLField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Site Settings"
        verbose_name_plural = "Site Settings"

    def __str__(self):
        return self.site_name

    def save(self, *args, **kwargs):
        # Ensure only one instance exists
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def get_settings(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class AboutSection(models.Model):
    """CMS model for About page sections"""
    title = models.CharField(max_length=200)
    content = models.TextField()
    image = models.ImageField(upload_to='about_images/', null=True, blank=True)
    icon = models.CharField(max_length=50, blank=True, help_text="FontAwesome icon class, e.g., 'fas fa-heart'")
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "About Section"
        verbose_name_plural = "About Sections"
        ordering = ['order']

    def __str__(self):
        return self.title


class TeamMember(models.Model):
    """CMS model for team members on About page"""
    name = models.CharField(max_length=100)
    role = models.CharField(max_length=100)
    bio = models.TextField(blank=True)
    image = models.ImageField(upload_to='team_images/', null=True, blank=True)
    email = models.EmailField(blank=True)
    linkedin_url = models.URLField(blank=True)
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Team Member"
        verbose_name_plural = "Team Members"
        ordering = ['order']

    def __str__(self):
        return f"{self.name} - {self.role}"


class ContactMessage(models.Model):
    """Store contact form submissions"""
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=20, blank=True)
    subject = models.CharField(max_length=200)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    is_replied = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Contact Message"
        verbose_name_plural = "Contact Messages"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.subject}"


class FAQ(models.Model):
    """Frequently Asked Questions"""
    question = models.CharField(max_length=300)
    answer = models.TextField()
    category = models.CharField(max_length=50, blank=True, help_text="e.g., General, Predictions, Account")
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "FAQ"
        verbose_name_plural = "FAQs"
        ordering = ['category', 'order']

    def __str__(self):
        return self.question[:50]


# =========================================================
# PART 2 — ML MODEL VERSIONING SYSTEM
# =========================================================

class MLModelVersion(models.Model):
    """ML Model version control with metadata"""
    FORMAT_CHOICES = [
        ('pkl', 'Pickle/Joblib (.pkl, .joblib)'),
        ('h5', 'Keras/TensorFlow (.h5)'),
        ('onnx', 'ONNX (.onnx)'),
        ('pt', 'PyTorch (.pt)'),
        ('sav', 'Pickle (.sav)'),
    ]

    name = models.CharField(max_length=100, unique=True, help_text="e.g., diabetes_rf_v7")
    disease = models.CharField(max_length=20, choices=DISEASE_CHOICES)
    version = models.CharField(max_length=20, help_text="e.g., v7.0")
    accuracy = models.FloatField(help_text="Model accuracy (0-1)")
    description = models.TextField(blank=True)
    file_path = models.CharField(max_length=255, help_text="e.g., ml_models/diabetes_v7.pkl")
    detected_format = models.CharField(max_length=10, choices=FORMAT_CHOICES, blank=True)
    feature_schema = models.JSONField(
        help_text='JSON array of feature names, e.g., ["age", "glucose", "bmi"]'
    )
    feature_types = models.JSONField(
        null=True, blank=True,
        help_text='JSON object mapping feature names to types: {"age": "numeric", "gender": "categorical"}'
    )
    feature_options = models.JSONField(
        null=True, blank=True,
        help_text='JSON object for categorical options: {"gender": ["male", "female"]}'
    )
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "ML Model Version"
        verbose_name_plural = "ML Model Versions"
        ordering = ['disease', '-created_at']

    def __str__(self):
        status = "✓ Active" if self.is_active else "Inactive"
        return f"{self.name} ({self.version}) - {status}"

    def save(self, *args, **kwargs):
        # Auto-detect format from file path
        if self.file_path:
            ext = self.file_path.split('.')[-1].lower()
            format_map = {
                'pkl': 'pkl', 'joblib': 'pkl',
                'h5': 'h5', 'keras': 'h5',
                'onnx': 'onnx',
                'pt': 'pt', 'pth': 'pt',
                'sav': 'sav'
            }
            self.detected_format = format_map.get(ext, 'pkl')

        # Deactivate other models for same disease if this one is active
        if self.is_active:
            MLModelVersion.objects.filter(
                disease=self.disease, is_active=True
            ).exclude(pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

    @classmethod
    def get_active_model(cls, disease):
        """Get the active model version for a disease"""
        return cls.objects.filter(disease=disease, is_active=True).first()


# =========================================================
# PART 3 — USER PROFILE & ACTIVITY TRACKING
# =========================================================

class UserProfile(models.Model):
    """Extended user profile with additional information"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone = models.CharField(max_length=20, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=10, blank=True, choices=[
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other'),
    ])
    address = models.TextField(blank=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', null=True, blank=True)
    
    # Account status
    is_email_verified = models.BooleanField(default=False)
    email_verified_at = models.DateTimeField(null=True, blank=True)
    
    # Social login fields
    social_provider = models.CharField(max_length=30, blank=True, help_text="OAuth provider (google, etc.)")
    social_uid = models.CharField(max_length=255, blank=True, help_text="Social account UID")
    avatar_url = models.URLField(max_length=500, blank=True, help_text="Profile picture URL from social provider")
    
    # Tracking
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    total_predictions = models.PositiveIntegerField(default=0)
    total_page_views = models.PositiveIntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"

    def __str__(self):
        return f"{self.user.username}'s Profile"


class UserActivity(models.Model):
    """Track all user activities on the website"""
    
    ACTIVITY_TYPES = [
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('signup', 'Sign Up'),
        ('page_view', 'Page View'),
        ('prediction', 'Prediction Made'),
        ('contact_submit', 'Contact Form Submitted'),
        ('profile_update', 'Profile Updated'),
    ]
    
    PAGE_CATEGORIES = [
        ('home', 'Home Page'),
        ('about', 'About Page'),
        ('contact', 'Contact Page'),
        ('prediction', 'Prediction Page'),
        ('disease_info', 'Disease Information'),
        ('disease_prevention', 'Disease Prevention'),
        ('disease_treatment', 'Disease Treatment'),
        ('auth', 'Authentication'),
        ('other', 'Other'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='activities')
    activity_type = models.CharField(max_length=20, choices=ACTIVITY_TYPES)
    page_category = models.CharField(max_length=30, choices=PAGE_CATEGORIES, blank=True)
    
    # Page details
    page_url = models.CharField(max_length=500)
    page_title = models.CharField(max_length=200, blank=True)
    
    # For disease-related pages
    disease_name = models.CharField(max_length=20, choices=DISEASE_CHOICES, blank=True)
    
    # For predictions
    prediction = models.ForeignKey(
        Prediction, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='activity_logs'
    )
    
    # Additional data
    extra_data = models.JSONField(null=True, blank=True, help_text="Any additional activity data")
    
    # Request info
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    referrer = models.CharField(max_length=500, blank=True)
    
    # Session tracking
    session_key = models.CharField(max_length=100, blank=True)
    
    # Time tracking
    time_spent_seconds = models.PositiveIntegerField(null=True, blank=True, help_text="Time spent on page in seconds")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "User Activity"
        verbose_name_plural = "User Activities"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['activity_type', 'created_at']),
            models.Index(fields=['page_category', 'created_at']),
            models.Index(fields=['disease_name', 'created_at']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.get_activity_type_display()} - {self.created_at}"


class UserSession(models.Model):
    """Track user sessions for analytics"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    session_key = models.CharField(max_length=100, unique=True)
    
    # Session info
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    device_type = models.CharField(max_length=20, blank=True, choices=[
        ('desktop', 'Desktop'),
        ('mobile', 'Mobile'),
        ('tablet', 'Tablet'),
        ('unknown', 'Unknown'),
    ])
    browser = models.CharField(max_length=50, blank=True)
    os = models.CharField(max_length=50, blank=True)
    
    # Location (optional - from IP)
    country = models.CharField(max_length=100, blank=True)
    city = models.CharField(max_length=100, blank=True)
    
    # Session stats
    pages_viewed = models.PositiveIntegerField(default=0)
    predictions_made = models.PositiveIntegerField(default=0)
    
    # Timestamps
    started_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "User Session"
        verbose_name_plural = "User Sessions"
        ordering = ['-started_at']

    def __str__(self):
        return f"{self.user.username} - Session {self.session_key[:8]}..."


class PredictionHistory(models.Model):
    """Detailed prediction history with user-friendly display"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='prediction_history')
    prediction = models.OneToOneField(Prediction, on_delete=models.CASCADE, related_name='history_entry')
    
    # Cached display data
    disease_display_name = models.CharField(max_length=50)
    risk_level_display = models.CharField(max_length=20)
    probability_display = models.CharField(max_length=20)
    
    # User notes
    user_notes = models.TextField(blank=True, help_text="User's personal notes about this prediction")
    is_bookmarked = models.BooleanField(default=False)
    
    # Follow-up
    follow_up_date = models.DateField(null=True, blank=True)
    follow_up_notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Prediction History"
        verbose_name_plural = "Prediction Histories"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.disease_display_name} - {self.created_at.date()}"


# Signal to create UserProfile when User is created
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    if hasattr(instance, 'profile'):
        instance.profile.save()
