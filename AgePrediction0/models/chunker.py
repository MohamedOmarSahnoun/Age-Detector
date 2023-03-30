#import libraries
'''
import torch
import torch.nn as nn

#declare model class
class AlexNetwork(nn.Module):
    def __init__(self, n_classes):
        super(AlexNetwork, self).__init__()
        self.n_classes = n_classes
        self.conv_1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 96,
            kernel_size = 11,
            stride = 4,
            padding = 50
        )
        self.pool_1 = nn.MaxPool2d(
        kernel_size = 3, 
        stride = 2,
        )
        self.conv_2 = nn.Conv2d(
            in_channels = 96,
            out_channels = 256,
            kernel_size = 5,
            stride = 1,
            padding = 2
        )  
        self.pool_2 = nn.MaxPool2d(
        kernel_size = 3, 
        stride = 2,
        )
        self.conv_3 = nn.Conv2d(
            in_channels = 256,
            out_channels = 384,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )  
        self.conv_4 = nn.Conv2d(
            in_channels = 384,
            out_channels = 384,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )  
        self.conv_5 = nn.Conv2d(
            in_channels = 384,
            out_channels = 256,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ) 
        self.pool_3 = nn.MaxPool2d(
        kernel_size = 3, 
        stride = 2,
        )
        self.nn = nn.Sequential(nn.Linear(in_features = 9216, out_features = 4096),
                                nn.ReLU(),
                                nn.Linear(in_features = 4096, out_features = 4096),
                                nn.ReLU(),
                                nn.Linear(in_features = 4096, out_features = self.n_classes),
                               )
    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.pool_3(x)
        x = x.view(-1,9216)
        x = self.nn(x)
        return x 

# Load the pre-trained model from the .pth file
model = torch.load('chunk_10.pth', map_location=torch.device('cpu'))

# Get the model parameters
params = model.state_dict()

# Chunk the parameters into smaller dictionaries
chunk_size = 1  # number of parameters in each chunk
chunks = [dict(list(params.items())[i:i+chunk_size]) for i in range(0, len(params), chunk_size)]

# Save each chunk as a separate file
for i, chunk in enumerate(chunks):
    torch.save(chunk, f'chunk_10_{i}.pth')

import os

# Define the file to split and the number of chunks to create
filename = 'alexnetwork.pth'
num_chunks = 10

# Determine the size of each chunk (in bytes)
filesize = os.path.getsize(filename)
chunk_size = filesize // num_chunks

# Open the input file and read the data
with open(filename, 'rb') as f:
    data = f.read()
'''
import os

# Define the file to split and the chunk size
filename = 'alexnetwork.pth'
chunk_size = 50 * 1024 * 1024  # 50 MB

# Open the input file and read the data
with open(filename, 'rb') as f:
    data = f.read()

# Split the data into chunks
chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

# Save each chunk to a separate file
for i, chunk in enumerate(chunks):
    with open(f'chunk_{i}.pth', 'wb') as f:
        f.write(chunk)


