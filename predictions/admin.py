from django.contrib import admin
from django.utils.html import format_html
from .models import (
    Prediction, DiseaseInfo, DiseasePrevention, 
    DiseaseTreatment, MLModelVersion
)


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['prediction_type', 'risk_level', 'probability', 'model_version', 'created_at']
    list_filter = ['prediction_type', 'risk_level', 'created_at']
    search_fields = ['prediction_type', 'user__username']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'


# =========================================================
# CMS ADMIN CONFIGURATION
# =========================================================

class BaseDiseaseContentAdmin(admin.ModelAdmin):
    """Base admin class for disease content models"""
    list_display = ['disease_name', 'title', 'section_type', 'order', 'is_active', 'updated_at']
    list_filter = ['disease_name', 'section_type', 'is_active']
    search_fields = ['title', 'content']
    list_editable = ['order', 'is_active']
    ordering = ['disease_name', 'order']
    
    fieldsets = (
        ('Content', {
            'fields': ('disease_name', 'title', 'content', 'image')
        }),
        ('Settings', {
            'fields': ('section_type', 'order', 'is_active')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    readonly_fields = ['created_at', 'updated_at']

    class Media:
        css = {
            'all': ('https://cdn.jsdelivr.net/npm/summernote@0.8.18/dist/summernote-lite.min.css',)
        }
        js = (
            'https://cdn.jsdelivr.net/npm/summernote@0.8.18/dist/summernote-lite.min.js',
        )


@admin.register(DiseaseInfo)
class DiseaseInfoAdmin(BaseDiseaseContentAdmin):
    pass


@admin.register(DiseasePrevention)
class DiseasePreventionAdmin(BaseDiseaseContentAdmin):
    pass


@admin.register(DiseaseTreatment)
class DiseaseTreatmentAdmin(BaseDiseaseContentAdmin):
    pass


# =========================================================
# SITE CONTENT ADMIN (FOOTER, ABOUT, CONTACT)
# =========================================================

from .models import (
    SiteSettings, AboutSection, TeamMember, ContactMessage, FAQ,
    HeroSection, HeroStat, PredictionCard, WhyChooseUs, HomepageStat, CTASection
)


# =========================================================
# HOMEPAGE CONTENT ADMIN
# =========================================================

@admin.register(HeroSection)
class HeroSectionAdmin(admin.ModelAdmin):
    fieldsets = (
        ('Content', {'fields': ('title', 'highlight_text', 'subtitle')}),
        ('Buttons', {'fields': ('primary_button_text', 'primary_button_url', 'secondary_button_text', 'secondary_button_url')}),
        ('Status', {'fields': ('is_active',)}),
    )

    def has_add_permission(self, request):
        return not HeroSection.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(HeroStat)
class HeroStatAdmin(admin.ModelAdmin):
    list_display = ['value', 'label', 'order', 'is_active']
    list_editable = ['order', 'is_active']


@admin.register(PredictionCard)
class PredictionCardAdmin(admin.ModelAdmin):
    list_display = ['title', 'disease_name', 'order', 'is_active']
    list_editable = ['order', 'is_active']
    list_filter = ['disease_name', 'is_active']


@admin.register(WhyChooseUs)
class WhyChooseUsAdmin(admin.ModelAdmin):
    list_display = ['title', 'order', 'is_active']
    list_editable = ['order', 'is_active']


@admin.register(HomepageStat)
class HomepageStatAdmin(admin.ModelAdmin):
    list_display = ['value', 'label', 'color_class', 'order', 'is_active']
    list_editable = ['order', 'is_active']


@admin.register(CTASection)
class CTASectionAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return not CTASection.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(SiteSettings)
class SiteSettingsAdmin(admin.ModelAdmin):
    fieldsets = (
        ('Site Identity', {
            'fields': ('site_name', 'site_tagline')
        }),
        ('Footer Content', {
            'fields': ('footer_text', 'footer_disclaimer', 'copyright_text')
        }),
        ('Contact Information', {
            'fields': ('contact_email', 'contact_phone', 'contact_address')
        }),
        ('Social Media Links', {
            'fields': ('facebook_url', 'twitter_url', 'linkedin_url', 'instagram_url'),
            'classes': ('collapse',)
        }),
    )

    def has_add_permission(self, request):
        # Only allow one instance
        return not SiteSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(AboutSection)
class AboutSectionAdmin(admin.ModelAdmin):
    list_display = ['title', 'order', 'is_active', 'updated_at']
    list_editable = ['order', 'is_active']
    search_fields = ['title', 'content']
    ordering = ['order']


@admin.register(TeamMember)
class TeamMemberAdmin(admin.ModelAdmin):
    list_display = ['name', 'role', 'order', 'is_active']
    list_editable = ['order', 'is_active']
    search_fields = ['name', 'role']
    ordering = ['order']


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'subject', 'is_read', 'is_replied', 'created_at']
    list_filter = ['is_read', 'is_replied', 'created_at']
    search_fields = ['name', 'email', 'subject', 'message']
    readonly_fields = ['name', 'email', 'phone', 'subject', 'message', 'created_at']
    list_editable = ['is_read', 'is_replied']
    date_hierarchy = 'created_at'

    def has_add_permission(self, request):
        return False


@admin.register(FAQ)
class FAQAdmin(admin.ModelAdmin):
    list_display = ['question', 'category', 'order', 'is_active']
    list_filter = ['category', 'is_active']
    list_editable = ['order', 'is_active']
    search_fields = ['question', 'answer']
    ordering = ['category', 'order']


# =========================================================
# ML MODEL VERSION ADMIN
# =========================================================

@admin.register(MLModelVersion)
class MLModelVersionAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'disease', 'version', 'accuracy_display', 
        'detected_format', 'is_active_display', 'created_at'
    ]
    list_filter = ['disease', 'is_active', 'detected_format']
    search_fields = ['name', 'description']
    list_editable = []
    ordering = ['disease', '-is_active', '-created_at']
    
    fieldsets = (
        ('Model Identity', {
            'fields': ('name', 'disease', 'version', 'description')
        }),
        ('Model File', {
            'fields': ('file_path', 'detected_format', 'accuracy')
        }),
        ('Feature Schema', {
            'fields': ('feature_schema', 'feature_types', 'feature_options'),
            'description': 'Define the input features required by this model'
        }),
        ('Status', {
            'fields': ('is_active',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    readonly_fields = ['detected_format', 'created_at', 'updated_at']
    
    actions = ['activate_models', 'deactivate_models']

    def accuracy_display(self, obj):
        """Display accuracy as percentage with color coding"""
        pct = obj.accuracy * 100
        if pct >= 90:
            color = 'green'
        elif pct >= 80:
            color = 'orange'
        else:
            color = 'red'
        return format_html(
            '<span style="color: {}; font-weight: bold;">{:.1f}%</span>',
            color, pct
        )
    accuracy_display.short_description = 'Accuracy'
    accuracy_display.admin_order_field = 'accuracy'

    def is_active_display(self, obj):
        """Display active status with icon"""
        if obj.is_active:
            return format_html(
                '<span style="color: green; font-weight: bold;">✓ Active</span>'
            )
        return format_html('<span style="color: gray;">Inactive</span>')
    is_active_display.short_description = 'Status'
    is_active_display.admin_order_field = 'is_active'

    @admin.action(description='Activate selected models (deactivates others for same disease)')
    def activate_models(self, request, queryset):
        for model in queryset:
            model.is_active = True
            model.save()  # This triggers deactivation of other models
        self.message_user(request, f'{queryset.count()} model(s) activated.')

    @admin.action(description='Deactivate selected models')
    def deactivate_models(self, request, queryset):
        queryset.update(is_active=False)
        self.message_user(request, f'{queryset.count()} model(s) deactivated.')


# =========================================================
# USER ACTIVITY & PROFILE ADMIN
# =========================================================

from .models import UserProfile, UserActivity, UserSession, PredictionHistory


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'phone', 'total_predictions', 'total_page_views', 'is_email_verified', 'created_at']
    list_filter = ['is_email_verified', 'gender', 'created_at']
    search_fields = ['user__username', 'user__email', 'phone']
    readonly_fields = ['total_predictions', 'total_page_views', 'last_login_ip', 'created_at', 'updated_at']
    
    fieldsets = (
        ('User', {'fields': ('user',)}),
        ('Personal Info', {'fields': ('phone', 'date_of_birth', 'gender', 'address', 'profile_picture')}),
        ('Verification', {'fields': ('is_email_verified', 'email_verified_at')}),
        ('Statistics', {'fields': ('total_predictions', 'total_page_views', 'last_login_ip')}),
        ('Timestamps', {'fields': ('created_at', 'updated_at'), 'classes': ('collapse',)}),
    )


@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    list_display = ['user', 'activity_type', 'page_category', 'disease_name', 'page_url', 'ip_address', 'created_at']
    list_filter = ['activity_type', 'page_category', 'disease_name', 'created_at']
    search_fields = ['user__username', 'user__email', 'page_url', 'page_title']
    readonly_fields = [
        'user', 'activity_type', 'page_category', 'page_url', 'page_title',
        'disease_name', 'prediction', 'extra_data', 'ip_address', 'user_agent',
        'referrer', 'session_key', 'time_spent_seconds', 'created_at'
    ]
    date_hierarchy = 'created_at'
    ordering = ['-created_at']
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(UserSession)
class UserSessionAdmin(admin.ModelAdmin):
    list_display = ['user', 'device_type', 'browser', 'os', 'pages_viewed', 'predictions_made', 'is_active', 'started_at', 'last_activity']
    list_filter = ['device_type', 'browser', 'os', 'is_active', 'started_at']
    search_fields = ['user__username', 'user__email', 'session_key']
    readonly_fields = [
        'user', 'session_key', 'ip_address', 'user_agent', 'device_type',
        'browser', 'os', 'country', 'city', 'pages_viewed', 'predictions_made',
        'started_at', 'last_activity', 'ended_at', 'is_active'
    ]
    date_hierarchy = 'started_at'
    ordering = ['-started_at']
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'disease_display_name', 'risk_level_display', 'probability_display', 'is_bookmarked', 'created_at']
    list_filter = ['disease_display_name', 'risk_level_display', 'is_bookmarked', 'created_at']
    search_fields = ['user__username', 'user__email', 'disease_display_name']
    readonly_fields = ['user', 'prediction', 'disease_display_name', 'risk_level_display', 'probability_display', 'created_at']
    date_hierarchy = 'created_at'
    ordering = ['-created_at']
    
    fieldsets = (
        ('Prediction Info', {'fields': ('user', 'prediction', 'disease_display_name', 'risk_level_display', 'probability_display')}),
        ('User Notes', {'fields': ('user_notes', 'is_bookmarked')}),
        ('Follow-up', {'fields': ('follow_up_date', 'follow_up_notes')}),
        ('Timestamps', {'fields': ('created_at',), 'classes': ('collapse',)}),
    )
