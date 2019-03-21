from django.db import models

# Create your models here.
class Slider(models.Model):
	name=models.CharField(max_length=40)
	image=models.ImageField(upload_to='slider/')

	def __str__(self):
		return str(self.name)
