# Program name : Lev_Rasp_Calibrate_Camera.py
# Function : For one time calibration of camera before level detection
# Captures frame from GoPro Hero 4 camera using GoPro API
# Applies OpenCV box layer on image for calibration
# New images captured continously
# Wifi should be connected with GoPro
# Program stops on Ctrl+C

from goprocam import GoProCamera
from goprocam import constants
import urllib.request
import cv2
import numpy as np
import time
import requests
#import system

# Get camera handle
gpCam = GoProCamera.GoPro()

# Create OpenCV window to display image
cv2.namedWindow('Calibrate',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Calibrate',800,600)

#cropped = input[690:2790, 1685:2145] # Black Image crop

def main(): 
	# Grab image from GoPro cam
    gpCam.take_photo(0)
	
	# Get the image URL and split filename
    photo_url = gpCam.getMedia()
    File_split = photo_url.split("/")
    
	# Open the URL and convert to numpy 
    url = urllib.request.urlopen(photo_url)
    photo = np.array(bytearray(url.read()), dtype=np.uint8)
    input_image = cv2.imdecode(photo, -1)
	
	# Draw the outer crop rectangle
    cv2.rectangle(input_image,(1680,685),(2140,2795),(0,0,255),15)
	
	# Draw lines to distinguish level crops
    for level in range(1,10):
        cv2.line(input_image,(1690,(690+(level*210))),(2130,(690+(level*210))),(0,255,0),15)
	
	# Delete the last image from GoPro
    gpCam.delete("last")
	
	# Show the image - Rasp reqd 2 sec delay for image to be visible
    cv2.imshow('Calibrate',input_image)
    #Output_FIlename = str(File_split[len(File_split)-1])
    #cv2.imwrite(Output_FIlename,input_image)
    cv2.waitKey(2)

# Main function to continously grab and display images
if __name__ == '__main__':
    try:
        while(True):
            print("Capturing Image")
            main()
            #time.sleep(1)
    except KeyboardInterrupt:
        print ('Interrupted')
        cv2.destroyAllWindows()
