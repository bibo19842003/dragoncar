from django.urls import path

from . import views

urlpatterns = [
    path('home/', views.home),
    path('homesensor/', views.homesensor),
    path('upload_home/', views.upload_home),
]
