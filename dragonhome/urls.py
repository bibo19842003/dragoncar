from django.urls import path

from . import views

urlpatterns = [
    path('home/', views.home),
    path('homesensor/', views.homesensor),
    path('voiceinput/', views.voiceinput),
    path('upload_home/', views.upload_home),
    path('upload_voice_input/', views.upload_voice_input),
]
