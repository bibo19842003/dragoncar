from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('index/', views.index),
    path('video_feed/', views.video_feed),
    path('detectcarnumber_feed/', views.detectcarnumber_feed),
    path('wificontrol/', views.wificontrol),
    path('dragonvideo/', views.dragonvideo),
    path('videocar/', views.videocar),
    path('detectcarnumber/', views.detectcarnumber),
    path('autocar/', views.autocar),
    path('followme/', views.followme),
    path('followme_feed/', views.followme_feed),
    path('voicecontrol/', views.voicecontrol),
    path('uploadfile/', views.uploadfile),
    path('upload_file/', views.upload_file),
    path('powermanage/', views.powermanage),
    path('follow/', views.follow),
    path('followobject/', views.followobject),
    path('followobject_feed/', views.followobject_feed),
]

