#!/bin/bash

# Directory containing the .keras files
directory="example/sakura_models"

# Image and dimensions
image="example/panelki.jpg"
width=1280
height=1280

# # Loop through all .keras files in the directory
# for keras_file in "$directory"/*.keras; 
# do
#     # Execute the python command
#     python3.12 decompress_img_tf.py "$keras_file" "$width" "$height"
# done


# Loop through all .keras files in the directory
for torch_file in "$directory"/*.pth; 
do
    # Execute the python command
    python3.12 decompress_img_torch.py "$torch_file" "$width" "$height"
done
