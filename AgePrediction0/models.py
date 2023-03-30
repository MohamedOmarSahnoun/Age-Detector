from django.db import models

# Create your models here.
class ImageAge(models.Model):
    name = models.CharField(max_length = 20, name = 'name')
    image = models.ImageField(name = 'photo', upload_to='images/')#default = 'no image') #storage = 'AgePrediction0/static/AgePrediction0', 
    age = models.IntegerField(default = 0, name = 'age')   
    def __str__(self):
        return self.name 
# models.py
class Hotel(models.Model):
    name = models.CharField(max_length=50)
    hotel_Main_Img = models.ImageField(upload_to='images/') #images/ directory will be created automatically under media directory 
    def __str__(self):
        return self.name
