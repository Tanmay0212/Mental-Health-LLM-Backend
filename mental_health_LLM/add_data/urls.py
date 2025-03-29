# add_data/urls.py
from django.urls import path
from .views import upload_csv_and_add_to_pinecone

urlpatterns = [
    path('', upload_csv_and_add_to_pinecone)
]