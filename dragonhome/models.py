from django.db import models

# Create your models here.

class Homedevice(models.Model):
    name = models.CharField(max_length=20, default="", verbose_name="name")
    ip = models.CharField(max_length=20, unique=True, verbose_name="ip")
    position = models.CharField(max_length=20, default="", verbose_name="position")
    status = models.CharField(max_length=10, default="", verbose_name="status")
    description = models.CharField(max_length=50, default="", verbose_name="description")
    remark = models.CharField(max_length=20, default="", verbose_name="remark")

    def __str__(self):
        return '%s %s %s %s %s %s' % (self.name, self.ip, self.position, self.status, self.description, self.remark)
