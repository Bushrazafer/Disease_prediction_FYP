import os
from django.apps import AppConfig


class PredictionsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictions'
    verbose_name = 'Disease Predictions'
    path = os.path.dirname(os.path.abspath(__file__))

    def ready(self):
        # Import signals to register them
        import predictions.signals  # noqa