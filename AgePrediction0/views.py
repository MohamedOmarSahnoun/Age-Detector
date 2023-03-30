from django.shortcuts import render, redirect, HttpResponse
from .forms import Form_
from .models import ImageAge
import numpy as np
import torch #must be installed first
import  cv2 #must be installed first
import torch.nn as nn
# Create your views here.
def index(request):
    if request.method == 'POST':
        form = Form_(request.POST, request.FILES) #request.POST dictionnaire fih les données eli da5alhom l user #ymkn nest7a9 request.FILES l tsawer afterwards, YES! hh
        if form.is_valid():
            data = form.cleaned_data 
            image = ImageAge(name = data['name'], photo = data['image'], age = data['age'])
            image.save()
            #prepare the model
            model_path = r"C:\Users\G5\Desktop\Age Prediction (3)\AgePrediction0\models\alexnetwork.pth"#"C:\Users\G5\Downloads\pretrained.pth"# #contestable
            model = torch.load(model_path, map_location=torch.device('cpu'))
            #prepare the image
            ia = ImageAge.objects.last() #in reality we wanna get the last one, ?
            image_path = ia.photo.name
            image_path = 'C:/Users/G5/Desktop/Age Prediction (3)/media/' + image_path
            #image_path = r"C:\Users\G5\Desktop\Age Prediction (3)\media\images\mmr.jpg"
            #print(image_path)
            image = cv2.imread(image_path)
            #e resize yji houni
            image = cv2.resize(src = image, dsize = (128, 128), interpolation = cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image , dtype = float)
            x = torch.tensor(image , dtype = torch.float).permute(2,0,1)/255
            #make inference
            model.eval()
            softmax = nn.Softmax()
            y_hat = softmax(model(x))#.unsqueeze(0)
            def pred(x):
                for i in range(len(x[0])):
                    if (x[0][i] == torch.max(x[0]).item()): 
                        return i
            true_age = ia.age
            pred_age = pred(y_hat)+20
            return render(request, 'AgePrediction0/response.html', {'age' : pred_age}) 
        else: return HttpResponse('<strong>Invalid entry</strong>')
    else:
        form = Form_()
        return render(request, 'AgePrediction0/form.html', {'form' : form})    
def show_image(request):
    return render(request, 'AgePrediction0/image.html', {})






