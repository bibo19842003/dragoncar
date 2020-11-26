from django.contrib import admin
from .models import Homedevice


class HomedeviceAdmin(admin.ModelAdmin):
    list_display = ('name', 'ip', 'position', 'status', 'description', 'remark')

admin.site.register(Homedevice, HomedeviceAdmin)
