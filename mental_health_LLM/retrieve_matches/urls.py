# add_data/urls.py
from django.urls import path
from .views import get_top_matches

urlpatterns = [
    path('', get_top_matches)
]