import os

# Define the prefix of the chunk files and the number of chunks
chunk_prefix = 'chunk_'
num_chunks = 5

# Create a list of chunk file names
chunk_filenames = [f'{chunk_prefix}{i}.pth' for i in range(num_chunks)]

# Read each chunk into memory and concatenate them together
data = b''
for chunk_filename in chunk_filenames:
    with open(chunk_filename, 'rb') as f:
        data += f.read()

# Write the concatenated data to a new file
with open('restored_file.pth', 'wb') as f:
    f.write(data)
