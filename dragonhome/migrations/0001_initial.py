# Generated by Django 3.0.6 on 2020-11-26 04:58

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Homedevice',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='', max_length=20, verbose_name='name')),
                ('ip', models.CharField(max_length=20, unique=True, verbose_name='ip')),
                ('position', models.CharField(default='', max_length=20, verbose_name='position')),
                ('status', models.CharField(default='', max_length=10, verbose_name='status')),
                ('description', models.CharField(default='', max_length=50, verbose_name='description')),
                ('remark', models.CharField(default='', max_length=20, verbose_name='remark')),
            ],
        ),
    ]