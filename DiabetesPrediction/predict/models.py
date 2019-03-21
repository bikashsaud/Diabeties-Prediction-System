from django.db import models

# Create your models here.
class Employee(models.Model):
    pregnancy = models.CharField(max_length=2)
    gl=models.FloatField(max_length=3)
    bp= models.CharField(max_length=3)
    skin = models.CharField(max_length=3)
    insulen = models.CharField(max_length=4)
    bmi  = models.CharField(max_length=4)
    dpf=models.CharField(max_length=3)
    age=models.CharField(max_length=3)
