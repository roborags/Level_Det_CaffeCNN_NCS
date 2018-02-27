# Program name : Lev_Rasp_NCS_Final_Setup.py
# Function : The final code for project demo, image is captured from GoPro
# Level is found using CNN and OpenCV methods
# Final level is drawn on the image and saved
# Input is : NCS Graph file, image captured via GoPro
# Code requires GoPro connected to RaspPi via WiFi
# and NCS plugged into Rasp
# Program stops on Ctrl+C

from __future__ import division
from mvnc import mvncapi as mvnc
from goprocam import GoProCamera
from goprocam import constants
import urllib.request
import numpy as np
import cv2
import os.path
import string
import re
import sys
import os
import time
import requests

# Root path for NCS graph file
Root_path = "/home/pi/Desktop/Level_Detection/"
Graph_path = Root_path + "graph"

# Txt file names
Hit_file = Root_path + "Hit.txt"
Miss_file = Root_path + "Miss.txt"
Summary_file = Root_path + "Summary.txt"

num_images = 0
num_hit = 0
num_miss = 0
hit_rate = 0.00
miss_rate = 0.00

# Canny threshold changed by using calculating the median of the image from pyimagesearch
def auto_canny(image, sigma=0.33):
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        #print("v = ",v," lower = ",lower," upper = ",upper)
        return edged

# Function to draw the edge contours		
def draw_cnt(image):
        cont_out = np.zeros((210,460,1),np.uint8)
        len_out = 0
        im2,contours,hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
                for c in contours:
                        x,y,w,h = cv2.boundingRect(c)

                        if(w>h and (y%210)>= 55 and (y%210) <= 155):
                                cv2.drawContours(cont_out, c, -1, 255, 3)
                                len_out = len_out + 1
        return cont_out,len_out

# Crop input image and apply OpenCV proc
def get_out_rez(image, level):
        dim = (227, 227)
        output = image[((10-level)*210):(((10-level)*210)+210),0:460]

        auto_edge = auto_canny(output)

        cont_out,ret_cnt_len = draw_cnt(auto_edge)

        output_resize_1 = np.zeros((227,227,1),np.uint8)

        output_resize_1 = cv2.resize(cont_out,dim,0,0.5,interpolation = cv2.INTER_LANCZOS4)
        return output_resize_1

# Pass image as input blob and find level with net output		
def find_lev_net(graph,image):

        image = image.astype(np.float32)

        #im_input = im[np.newaxis, np.newaxis, :, :]
        graph.LoadTensor(image.astype(np.float16), 'user object')
        data, userobj = graph.GetResult()
        #print(data,userobj)
        if((data[0] == 0) and (data[1] == 0)):
                #print ("level not found")
                level_found = 2
        elif (data[0]<data[1]):
                #print ("level not found")
                level_found = 2
        else:
                #print ("level found")
                level_found = 1
        return level_found,data[0],data[1]

def main(sumf):
    print("GOPRO grabbing image")
	# Make GoPro capture one image
    gpCam.take_photo(0)
    #gpCam.delete(gpCam.getMedia())
	
	# Get image URL and get filename from it
    photo_url = gpCam.getMedia()
    File_split = photo_url.split("/")
    File_name = str(File_split[len(File_split)-1])
    filename_str = File_name.replace(".JPG","")
    print(File_name)

	# Get image from URL and covert to numpy
    url = urllib.request.urlopen(photo_url)
    photo = np.array(bytearray(url.read()), dtype=np.uint8)
    input = cv2.imdecode(photo, -1)

	# Crop the center region test-tube ROI
    #cropped = input[813:2913, 1780:2240] # White Image crop
    cropped = input[690:2790, 1685:2145] # Black Image crop

	# Convert to grayscal and apply blur for Canny
    gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    cnt_len_list = []
    ret_cnt_len = 0

	# Traverse throuh 10 sections and pass to Auto Canny
    for level in range(1,11):
            output = blur_image[((10-level)*210):(((10-level)*210)+210),0:460]
            #Output_Filename = Output_Image_path+filename_str+"_out"+str(level)+".JPG"
            #cv2.imwrite(Output_Filename,output_2)
            auto_edge = auto_canny(output)
            cont_out,ret_cnt_len = draw_cnt(auto_edge)
			# Add contours to list for further sorting
            cnt_len_list.append([ret_cnt_len,level])

    cnt_len_sort = np.asarray(cnt_len_list)
	# Sort the contours
    cnt_len_sort = cnt_len_sort[np.lexsort(np.fliplr(cnt_len_sort).T)]
    #print(cnt_len_sort)    
    cv_level = cnt_len_sort[9,1]
    output_resize = np.zeros((227,227,1),np.uint8)
	
	# Convert image to Dataset format
    output_resize = get_out_rez(blur_image,cv_level)
	
	# Pass image through CNN and obtain output probs
    net_level,res_1_0,res_1_1 = find_lev_net(graph,output_resize)
    
    input_image = input

	# Delete the last image from the camera
    gpCam.delete("last")

    #num_images = num_images + 1
    if(net_level == 1):
            print("Level found in ",cv_level)
    else:
			# Prints if different level is found at the OpenCV and CNN
            print("Level found in Diff CV ",cv_level," NET ",net_level)
            #num_miss+=1
    sumf.write(File_name+" CV_Lev "+str(cv_level)+" Net_Lev "+str(net_level)+" Res out "+str(res_1_0)+" "+str(res_1_1)+os.linesep)
	
	# Draw rectangle over the obatined level
    cv2.rectangle(input_image,(1685,(690+((10-cv_level)*210))),(2145,(690+((10-cv_level)*210)+210)),(0,0,255),10)
	
	# Write the Level value on the image
    Text_out = "Water Level: "+str(cv_level*10)
    cv2.putText(input_image,Text_out,(2145+20,(690+((10-cv_level)*210)+210)),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),10, cv2.LINE_AA)
	
	# Save the final image
    Output_FIlename = Root_path + "Images/" +filename_str+".JPG"
    cv2.imwrite(Output_FIlename,input_image)

    return input,cv_level,net_level,res_1_0,res_1_1

if __name__ == '__main__':
    
    time_start = time.clock()
    #print(time_start)

	# Enable NCS log
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
	
	# Enumerate all NCS devices and select the first device handle
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()
    device = mvnc.Device(devices[0])
	
	# Open the NCS device and allocate the graph file
    device.OpenDevice()
    with open( Graph_path, mode='rb' ) as f:
            blob = f.read()
    graph = device.AllocateGraph(blob)

	# Get the GoPro camera handle
    gpCam = GoProCamera.GoPro()

    try:
            with open(Summary_file, "w") as sumf:
                    try:
                            while(True):
                                    main(sumf)
									# Change delay for each image grab here
                                    time.sleep(20)
                    except KeyboardInterrupt:
                            print ("Keyboard Interrupt")
                            #sumf.write(os.linesep+os.linesep)
                            #sumf.write(" Summary- Total Images "+str(num_images)+" Missed files "+str(num_miss)+os.linesep)
    except Exception as error1:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Error 1 : ",exc_type, str(error1), exc_tb.tb_lineno)
