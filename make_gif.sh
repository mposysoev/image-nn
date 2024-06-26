#!/bin/bash

# Directory containing the images
directory="example/sakura_results"

# Output GIF file name
output_gif="sakura.gif"

# Frame delay in centiseconds (0.5 seconds = 50 centiseconds)
frame_delay=50

# Use ffmpeg to create a GIF from all images in the directory
ffmpeg -framerate 1/0.5 -pattern_type glob -i "$directory/*.png" -vf "scale=640:-1:flags=lanczos" -y "$output_gif"

echo "GIF created as $output_gif"
