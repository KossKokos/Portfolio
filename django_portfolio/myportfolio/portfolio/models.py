from django.db import models

# Create your models here.
class Email(models.Model):
    email = models.CharField(max_length=100)
    message = models.CharField(max_length=200)
