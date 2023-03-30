from django import forms
from .models import *
class Form_(forms.Form):
    name = forms.CharField(max_length = 20)
    image = forms.ImageField(required = False)
    age = forms.IntegerField()

class HotelForm(forms.ModelForm):
	class Meta:
		model = Hotel
		fields = ['name', 'hotel_Main_Img']
