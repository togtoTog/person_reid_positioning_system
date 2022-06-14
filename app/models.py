# coding=utf-8
from __future__ import unicode_literals
from django.db import models


# Create your models here.


class NormalUser(models.Model):
    username = models.CharField('用户名', max_length=30)
    # 用户名
    headImg = models.FileField('文件', upload_to='./upload')  # 文件名

    def __str__(self):
        return self.username

    class Meta:
        ordering = ['username']  # 排序风格username
