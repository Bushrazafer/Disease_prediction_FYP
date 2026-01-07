"""
Custom template tags for predictions app
"""
from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """Get item from dictionary by key"""
    if dictionary is None:
        return None
    return dictionary.get(key)


@register.filter
def get_feature_type(feature_types, feature):
    """Get feature type from feature_types dict"""
    if feature_types is None:
        return 'numeric'
    return feature_types.get(feature, 'numeric')


@register.filter
def format_feature_name(name):
    """Format feature name for display (replace underscores with spaces, title case)"""
    return name.replace('_', ' ').title()


@register.simple_tag
def get_input_type(feature_types, feature):
    """Determine HTML input type based on feature type"""
    if feature_types is None:
        return 'number'
    
    ftype = feature_types.get(feature, 'numeric')
    
    type_map = {
        'numeric': 'number',
        'integer': 'number',
        'float': 'number',
        'boolean': 'select',
        'categorical': 'select',
        'text': 'text',
    }
    
    return type_map.get(ftype, 'number')


@register.inclusion_tag('predictions/partials/form_field.html')
def render_form_field(feature, feature_types=None, feature_options=None):
    """Render a single form field based on feature configuration"""
    ftype = 'numeric'
    options = None
    
    if feature_types:
        ftype = feature_types.get(feature, 'numeric')
    
    if feature_options:
        options = feature_options.get(feature)
    
    return {
        'feature': feature,
        'feature_name': feature.replace('_', ' ').title(),
        'feature_type': ftype,
        'options': options,
        'is_categorical': options is not None,
        'is_boolean': ftype == 'boolean',
        'is_numeric': ftype in ['numeric', 'integer', 'float'],
    }
