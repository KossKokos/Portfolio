from django import forms
from django.contrib.auth.forms import UserCreationForm
from models import Email


class EmailForm(UserCreationForm):
    email = forms.CharField(max_length=100,
                               required=True,
                               widget=forms.TextInput())

    message = forms.CharField(max_length=200,
                                required=True,
                                widget=forms.PasswordInput())

    class Meta:
        model = Email
        fields = ['email', 'message']