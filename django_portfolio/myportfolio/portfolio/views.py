from django.http import HttpResponse
from django.template import loader

from .data import img_examples, projects_data, table_qualifications, services_data


def get_html_file(name):
    template = loader.get_template(name)
    return template


def index(request):
    template = get_html_file('portfolio/home.html')
    data = {'examples': img_examples}
    return HttpResponse(template.render(data, request))
    

def projects(request):
    template = get_html_file('portfolio/projects.html')
    data = {'examples' : projects_data}
    return HttpResponse(template.render(data, request))


def about_me(request):
    template = get_html_file('portfolio/about.html')
    data = {'Hello' : 'world'}
    return HttpResponse(template.render(data, request))


def contact_me(request):
    template = get_html_file('portfolio/contact.html')
    data = {'Hello' : 'world'}
    return HttpResponse(template.render(data, request))


def qualifications(request):
    template = get_html_file('portfolio/qualifications.html')
    data = {'qualifications' : table_qualifications}
    return HttpResponse(template.render(data, request))


def services(request):
    template = get_html_file('portfolio/services.html')
    data = {'services' : services_data}
    return HttpResponse(template.render(data, request))


def experience(request):
    template = get_html_file('portfolio/experience.html')
    data = {'Hello' : 'world'}
    return HttpResponse(template.render(data, request))


def clients(request):
    template = get_html_file('portfolio/clients.html')
    data = {'Hello' : 'world'}
    return HttpResponse(template.render(data, request))