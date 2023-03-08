# Smile Detection using OpenCV and Dlib Libraries

This program is designed to detect smiles in real-time video using OpenCV and dLib libraries. Different characteristics are computed by 
facial landmark points, and if a person in the video is smiling, the program labels the video as "smiling" and saves the smiling images in a folder.

## Requirements

* Python 3.x

* OpenCV

* Dlib

* imutils

## Usage

To run the program, simply execute the smile_detection.py file in your Python environment.

When the program is running, it will access your default camera and show a window with the real-time video.
If a person in the video is smiling, the program will label the video as "Smiling" and save the smiling images in a folder named "smiling_images". 
The images will be saved every 3 seconds.

You can exit the program by pressing the 'q' key.
