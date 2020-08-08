from django.contrib import admin
from .models import Uploadimage
from django.utils.html import format_html


class UploadimageAdmin(admin.ModelAdmin):
    list_display = ('filename', 'peoplename', 'filesize', 'rec', 'created', 'timestamp', 'previewpic')

    def previewpic(self, obj):
        return format_html('<img src="/media/recface/pic/%s" width="64" />' %(obj.filename) )

    previewpic.allow_tags = True
    readonly_fields = ['previewpic']

admin.site.register(Uploadimage, UploadimageAdmin)
