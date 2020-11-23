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
    path('detectpeopleface/', views.detectpeopleface),
    path('autocar/', views.autocar),
    path('followme/', views.followme),
    path('followme_feed/', views.followme_feed),
    path('voicecontrol/', views.voicecontrol),
    path('voicecar/', views.voicecar),
    path('uploadfile/', views.uploadfile),
    path('upload_file/', views.upload_file),
    path('upload_voicecar/', views.upload_voicecar),
    path('powermanage/', views.powermanage),
    path('follow/', views.follow),
    path('followobject/', views.followobject),
    path('followobject_feed/', views.followobject_feed),
    path('uploadfacepic/', views.uploadfacepic),
    path('upload_face_pic/', views.upload_face_pic),
    path('managefacepic/', views.managefacepic),
    path('photoface/', views.photoface),
    path('photofacepic_feed/', views.photofacepic_feed),
    path('photofacemodify/', views.photofacemodify),
    path('recface_feed/', views.recface_feed),
    path('qrscan_feed/', views.qrscan_feed),
    path('qrscan/', views.qrscan),
    path('objectdetect/', views.objectdetect),
    path('objectdetect_feed/', views.objectdetect_feed),
]

