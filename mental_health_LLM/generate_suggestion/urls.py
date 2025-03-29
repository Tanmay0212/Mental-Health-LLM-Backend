# add_data/urls.py
from django.urls import path
from .views import generate_suggestion

urlpatterns = [
    path('', generate_suggestion)
]