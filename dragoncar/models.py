from django.db import models
from datetime import datetime


class Uploadimage(models.Model):
    filename = models.CharField(max_length=50, default="", verbose_name="name")
    filesize = models.IntegerField(verbose_name="size(B)")
    peoplename = models.CharField(max_length=30)
    rec = models.CharField(max_length=3, default="0")
    created = models.DateTimeField(auto_now=True)
    timestamp = models.CharField(max_length=20, default=datetime.now().strftime("%Y%m%d%H%M%S"))

    def __str__(self):
        return '%s %s %s %s %s %s' % (self.filename, self.peoplename, self.filesize, self.rec, self.created, self.timestamp)
