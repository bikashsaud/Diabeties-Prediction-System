from django.http import HttpResponse
from django.shortcuts import render
from slider.models import Slider

def home(request):
    slider={
			'slider':Slider.objects.all(),
				}

    return render(request,'home.html',slider)
def predict(request):
    pass
def about(request):
    return render(request,'about.html')
