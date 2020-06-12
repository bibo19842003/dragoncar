from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('index/', views.index),
    path('video_feed/', views.video_feed),
    path('wificontrol/', views.wificontrol),
    path('videocar/', views.videocar),
    path('autocar/', views.autocar),
    path('followme/', views.followme),
    path('followme_feed/', views.followme_feed),
    path('voicecontrol/', views.voicecontrol),
    path('uploadfile/', views.uploadfile),
    path('upload_file/', views.upload_file),
]

