# Program name : Lev_Rasp_NCS_Test_CNN_Dataset.py
# Function : To test the dataset images with CNN loaded to Intel Neural Compute Stick
# Check report doc for steps to create NCS graph file from Caffe files
# Input : NCS graph file, Input images, train/test filelist
# Output : Summary of test, missed files, hit files

from __future__ import division
from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import os.path
import string
import re
#import caffe
import sys
import os
import time
#from caffe.proto import caffe_pb2

# Set input path for images
Input_Image_path = "/home/pi/Desktop/Level_Detection/Black_Back/"

# Root path for train/test txt files and NCS graph file
Root_path = "/home/pi/Desktop/Level_Detection/"

# Not needed in this code
Caffe_proto_path = Root_path + "lenet_deploy.prototxt"
Caffe_model_path = Root_path + "Level_Alex_net_iter_10000.caffemodel"

# Set graph file name
Graph_path = Root_path + "graph"
Train_file = "Train_Black_BB"
#Train_file = "Test_Black_BB"
File_type = "Net_CV"
#File_type = "OpenCV"

# Txt file names
Hit_file = Train_file+"_"+File_type+"_Hit.txt"
Miss_file = Train_file+"_"+File_type+"_Miss.txt"
Summary_file = Train_file+"_"+File_type+"_Summary.txt"

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

time_start = time.clock()
#print(time_start)
# Enable NCS log option
mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

# Get a list of ALL the sticks that are plugged in
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
	print('No devices found')
	quit()

# Pick the first stick to run the network
device = mvnc.Device(devices[0])

# Open the NCS
device.OpenDevice()

num_images = 0
num_hit = 0
num_miss = 0
hit_rate = 0.00
miss_rate = 0.00

with open( Graph_path, mode='rb' ) as f:
	blob = f.read()

# Allocate graph file to blob
graph = device.AllocateGraph(blob)

ncs_ready_time = (time.clock() - time_start)
print(ncs_ready_time)

file_tot_proc_time = 0
file_cv_proc_time = 0
file_net_proc_time = 0

# Traverse the images in folder, open train, hit and miss files
File_path = Root_path+Train_file+".txt"
try:
	with open(File_path, "r") as train:
		File_path = Root_path +Hit_file
		try:
			with open(File_path, "w") as hitf:
				File_path = Root_path +Miss_file
				try:
					with open(File_path, "w") as missf:
						for line in train:
							# In train/test file seperate the space and grab filename
							Split_text = line.split()
							File_name = Split_text[0]
							Lev_val = int(Split_text[1])
							cv_start_time = time.clock()
							#print(cv_start_time)
							Input_FIlename = Input_Image_path+File_name
							#print(Input_FIlename)
							input = cv2.imread(Input_FIlename, cv2.IMREAD_COLOR)
							
							filename_str = File_name.replace(".JPG","")
							# Crop center region to get testtube ROI
							# Seperate crop sections for Black and White images							
							#cropped = input[813:2913, 1780:2240] # White Image crop
							cropped = input[690:2790, 1685:2145] # Black Image crop
                                                        
							dim = (227, 227)
							# Convert to gray and apply GaussianBlur for Canny
							gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                                                        
							blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
                                                        
							cnt_len_list = []

							ret_cnt_len = 0
                                           
							# Find contours for the ten sections of the image
							for level in range(1,11):

								output = blur_image[((10-level)*210):(((10-level)*210)+210),0:460]

								auto_edge = auto_canny(output)

								cont_out,ret_cnt_len = draw_cnt(auto_edge)

								# Append contour len to list for further sorting
								cnt_len_list.append([ret_cnt_len,level])

							cnt_len_sort = np.asarray(cnt_len_list)

							#cnt_len_sort = np.sort(cnt_len_sort,axis=0)
							
							# Sort contour list to find largest contour
							cnt_len_sort = cnt_len_sort[np.lexsort(np.fliplr(cnt_len_sort).T)]
							
							#print(cnt_len_sort)
							# Extract largest contour and send to Net
							level_1 = cnt_len_sort[9,1]

							output_resize = np.zeros((227,227,1),np.uint8)

							output_resize = get_out_rez(blur_image,level_1)

							lev_val_1,res_1_0,res_1_1 = find_lev_net(graph,output_resize)

							# Extract second largest contour and send to Net
							level_2 = cnt_len_sort[8,1]

							output_resize = np.zeros((227,227,1),np.uint8)

							output_resize = get_out_rez(blur_image,level_2)

							lev_val_2,res_2_0,res_2_1 = find_lev_net(graph,output_resize)

							file_cv_proc_time = file_cv_proc_time + (time.clock() - cv_start_time)

							#print(file_cv_proc_time)

							num_images = num_images + 1

							print(num_images)
							# Check if network and OpenCV gave same results
							# Comment second If condition if using CV only mode

							if((level_1 == Lev_val)and (lev_val_1 == 1)):
								num_hit+=1
								#Write hit file names in hit.txt
								hitf.write(File_name+" Act_Lev "+str(Lev_val)+" Fnd_Lev "+str(level_1)+" Net_Out "+str(lev_val_1)+" Res out "+str(res_1_0)+" "+str(res_1_1)+os.linesep)
							else:
								num_miss+=1
								#Write missed file names in miss.txt
								missf.write(File_name+" Act_Lev "+str(Lev_val)+" Fnd_Lev "+str(level_1)+" Net_Out "+str(lev_val_1)+" Res out "+str(res_1_0)+" "+str(res_1_1)+os.linesep)

							#print("Level = ",level_1," Val = ",lev_val_1," Res1 = ",res_1_0," Res2 = ",res_1_1)
							#print("Level = ",level_2," Val = ",lev_val_2," Res1 = ",res_2_0," Res2 = ",res_2_1)
				except Exception as error3:
					exc_type, exc_obj, exc_tb = sys.exc_info()
                                        print("Error 3 : ",exc_type, str(error3), exc_tb.tb_lineno)
		except Exception as error2:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        print("Error 2 : ",exc_type, str(error2), exc_tb.tb_lineno)
except Exception as error1:
	exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error 1 : ",exc_type, str(error1), exc_tb.tb_lineno)

file_tot_proc_time = (time.clock() - ncs_ready_time)
file_net_proc_time = file_tot_proc_time - file_cv_proc_time
file_cv_proc_time = file_cv_proc_time - file_net_proc_time

# De-allocate graph at end of processing
graph.DeallocateGraph()

# Close the device
device.CloseDevice()

# Open and write the summary file
File_path = Root_path + Summary_file
try:
	with open(File_path, "w") as sumf:
		sumf.write("Summary of Files"+os.linesep)
		sumf.write("#####################"+os.linesep+os.linesep)
		sumf.write("Root path - "+Root_path+os.linesep)
		sumf.write("Input Image path - "+Input_Image_path+os.linesep)
		sumf.write("Image input train file - "+Train_file+".txt"+os.linesep)
		sumf.write("Image HIT file - "+Hit_file+os.linesep)
		sumf.write("Format: Filename-Train Act_Lev Fnd_Lev Net_Out Res_out"+os.linesep)
		sumf.write("Image MISS file - "+Miss_file+os.linesep)
		sumf.write("Format: Filename-Train Act_Lev Fnd_Lev Net_Out Res_out"+os.linesep)
		hit_rate = round(((num_hit/num_images)*100),2)
		miss_rate = round(((num_miss/num_images)*100),2)
		sumf.write(os.linesep+os.linesep)
		sumf.write("Number of Images = "+str(num_images)+os.linesep)
		sumf.write("Hit num = "+str(num_hit)+os.linesep)
		sumf.write("Hit rate = "+str(hit_rate)+os.linesep)
		sumf.write("Miss num = "+str(num_miss)+os.linesep)
		sumf.write("Miss rate = "+str(miss_rate)+os.linesep)
		sumf.write(os.linesep+os.linesep)
		sumf.write("Total process Time = "+str(file_tot_proc_time)+" seconds"+os.linesep)
		avg_tot_time = round(((file_tot_proc_time/num_images)*1000),2)
		sumf.write("Average process Time = "+str(avg_tot_time)+" milli-seconds"+os.linesep)

		sumf.write("Total OpenCV Time = "+str(file_cv_proc_time)+" seconds"+os.linesep)
		avg_cv_time = round(((file_cv_proc_time/num_images)*1000),2)
		sumf.write("Average OpenCV Time = "+str(avg_cv_time)+" milli-seconds"+os.linesep)

		sumf.write("Total Network Time = "+str(file_net_proc_time)+" seconds"+os.linesep)
		avg_net_time = round(((file_net_proc_time/num_images)*1000),2)
		sumf.write("Average Network Time = "+str(avg_net_time)+" milli-seconds"+os.linesep)
except Exception as error4:
	exc_type, exc_obj, exc_tb = sys.exc_info()
        print("Error 4 : ",exc_type, str(error4), exc_tb.tb_lineno)


print("Done num_hit = ",num_hit," num_miss = ",num_miss," num_images = ",num_images)
