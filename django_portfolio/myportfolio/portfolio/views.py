from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

from .models import Email


img_examples = [
    {
        'title': 'Recurent Neural Network',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/blstm_training_uitdwd.png"
    },
    {
        'title': 'Linear Regression',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/ridge_ibbdxz.png",
    },
    {
        'title': 'Clustering',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/clustering_aapydv.png",
    },
    {
        'title': 'SQL',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/sql_ntj4f4.png",
    },
    {
        'title': 'Deep Learning',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421788/comparing_models_oqgrh7.png",
    },
    {
        'title': 'Classification',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421789/comparing_results_kcnu15.png",
    },
    {
        'title': 'Data Analysis',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421789/lantitude_dz7phy.png",
    },
    {
        'title': 'Visualization',
        'url': "https://res.cloudinary.com/dtg29idor/image/upload/v1733421789/corr_matrix_jekicv.png"
    },
]

def get_html_file(name):
    template = loader.get_template(name)
    return template


def index(request):
    # email_list = Email.objects.all()
    # output = ', '.join([e.message for e in email_list])
    # return HttpResponse("Hello, world. You're at the polls index.")
    template = get_html_file('portfolio/home.html')
    data = {'examples': img_examples}
    return HttpResponse(template.render(data, request))
    

def projects(request):
    template = get_html_file('portfolio/projects.html')
    data = {'Hello' : 'world'}
    return HttpResponse(template.render(data, request))

def about_me(request):
    template = get_html_file('portfolio/about.html')
    data = {'Hello' : 'world'}
    return HttpResponse(template.render(data, request))

def contact_me(request):
    template = get_html_file('portfolio/contact.html')
    data = {'Hello' : 'world'}
    return HttpResponse(template.render(data, request))

