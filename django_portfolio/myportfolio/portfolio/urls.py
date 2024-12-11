from django.urls import path

from . import views

app_name = 'portfolio'

urlpatterns = [
    path('', views.index, name='index'),
    path('projects/', views.projects, name='projects'),
    path('about/', views.about_me, name='about_me'),
    path('contact/', views.contact_me, name='contact_me'),
    path('qualifications/', views.qualifications, name='qualifications'),
    path('services/', views.services, name='services'),
    path('experience/', views.experience, name='experience'),
    path('clients/', views.clients, name='clients'),
]