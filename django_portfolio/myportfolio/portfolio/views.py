from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

from .models import Email


# Create your views here.

def index(request):
    # email_list = Email.objects.all()
    # output = ', '.join([e.message for e in email_list])
    # return HttpResponse("Hello, world. You're at the polls index.")
    template = loader.get_template('portfolio/home.html')
    data = {'full_name': 'Elon Musk'}
    return HttpResponse(template.render(data, request))
    
