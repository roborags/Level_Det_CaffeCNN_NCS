# Program name : Lev_Test_CNN_Dataset.py
# Function : Load the trained CNN and test using images in folder
# Input : Caffe model, Deploy prototxt, Input images
# Opetions to switch between OpenCV and Caffe method
# Output : Summary of test, missed files, hit files
# OpenCV method uses image with largest contours

from __future__ import division
import numpy as np
import cv2
import os.path
import string
import re
import caffe
from caffe.proto import caffe_pb2

#Set path for Caffe files
Root_path = "/home/iot_lab_dnn/Level_Detection_Files/Test_Images/"
Caffe_proto_path = Root_path + "lenet_deploy.prototxt"
Caffe_model_path = Root_path + "Level_Alex_net_iter_10000.caffemodel"

#Set path for input images
Input_Image_path = Root_path + "Black_Back/"
Output_Image_path = Root_path + "Output_Image/"

#Set name of train/test file
Train_file = "Train_Black_BB"
#Train_file = "Test_Black_BB"

#Set option for OpenCV or Network mode
File_type = "Net_CV"
#dFile_type = "OpenCV"

Hit_file = Train_file+"_"+File_type+"_Hit.txt"
Miss_file = Train_file+"_"+File_type+"_Miss.txt"
Summary_file = Train_file+"_"+File_type+"_Summary.txt"

#Canny threshold changed by using calculating the median of the image from pyimagesearch
def auto_canny(image, sigma=0.33):
	v = np.median(image)

	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	#print("v = ",v," lower = ",lower," upper = ",upper)
	return edged

#Function to draw the edge contours
def draw_cnt(image):
	cont_out = np.zeros((210,460,1),np.uint8)
	len_out = 0
	im2,contours,hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) != 0:
		for c in contours:
			x,y,w,h = cv2.boundingRect(c)
			#Choose only horizontal contours and mid region
			if(w>h and (y%210)>= 55 and (y%210) <= 155):
				cv2.drawContours(cont_out, c, -1, 255, 3)
				len_out = len_out + 1
	return cont_out,len_out

#Crop input image and apply OpenCV proc
def get_out_rez(image, level):
	output = image[((10-level)*210):(((10-level)*210)+210),0:460]

	auto_edge = auto_canny(output)

	cont_out,ret_cnt_len = draw_cnt(auto_edge)

	output_resize_1 = np.zeros((227,227,1),np.uint8)

	output_resize_1 = cv2.resize(cont_out,dim,0,0.5,interpolation = cv2.INTER_LANCZOS4)
	return output_resize_1

# Pass image as input blob and find level with net output
def find_lev_net(image):
	im = np.array(image)

	im_input = im[np.newaxis, np.newaxis, :, :]

	net.blobs['data'].reshape(*im_input.shape)

	net.blobs['data'].data[...] = im_input

	res = net.forward()

	data = res['loss']

	if((data[0,0] == 0) and (data[0,1] == 0)):
		#print ("level not found")
		level_found = 2
	elif (data[0,0]<data[0,1]):
		#print ("level not found")
		level_found = 2
	else:
		#print ("level found")
		level_found = 1

	return level_found,data[0,0],data[0,1]

#Set Caffe mode to GPU for faster proc
caffe.set_mode_gpu()

#Load Caffe net from path
net = caffe.Net(Caffe_proto_path,Caffe_model_path, caffe.TEST)

[(k, v.data.shape) for k, v in net.blobs.items()]
[(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()]

print ("Data shape",net.blobs['data'].data.shape)
print ("Conv1 shape",net.blobs['conv1'].data.shape)
#print ("Relu1 shape",net.blobs['relu1'].data.shape)
print ("Pool1 shape",net.blobs['pool1'].data.shape)
print ("Conv2 shape",net.blobs['conv2'].data.shape)
#print ("Relu2 shape",net.blobs['relu2'].data.shape)
print ("Pool2 shape",net.blobs['pool2'].data.shape)
print ("Ip1 shape",net.blobs['ip1'].data.shape)
print ("Ip2 shape",net.blobs['ip2'].data.shape)
#print ("Drop2 shape",net.blobs['drop2'].data.shape)

num_images = 0
num_hit = 0
num_miss = 0
hit_rate = 0.00
miss_rate = 0.00

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
							Split_text = line.split()
							File_name = Split_text[0]
							Lev_val = int(Split_text[1])

							Input_FIlename = Input_Image_path+File_name
							#print(Input_FIlename)
							input = cv2.imread(Input_FIlename, cv2.IMREAD_COLOR)
							filename_str = File_name.replace(".JPG","")

							#Crop center region to get testtube ROI
							#Seperate crop sections for Black and White images 
							#cropped = input[813:2913, 1780:2240] # White Image crop
							cropped = input[690:2790, 1685:2145] # Black Image crop

							dim = (227, 227)
							#Convert to gray and apply GaussianBlur for Canny
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
							# Sort contour list to find largest contour
							cnt_len_sort = np.sort(cnt_len_sort.view('i8,i8'),order=['f0'],axis=0).view(np.int)

							#print(cnt_len_sort)
							# Extract largest contour and send to Net
							level_1 = cnt_len_sort[9,1]

							output_resize = np.zeros((227,227,1),np.uint8)

							output_resize = get_out_rez(blur_image,level_1)

							lev_val_1,res_1_0,res_1_1 = find_lev_net(output_resize)

							# Extract second largest contour and send to Net
							level_2 = cnt_len_sort[8,1]

							output_resize = np.zeros((227,227,1),np.uint8)

							output_resize = get_out_rez(blur_image,level_2)

							lev_val_2,res_2_0,res_2_1 = find_lev_net(output_resize)

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
					print("Error 3 : ",str(error3))	
		except Exception as error2:
			print("Error 2 : ",str(error2))		
except Exception as error1:
	print("Error 1 : ",str(error1))

# Open and write the summary file
File_path = Root_path + Summary_file
try:
	with open(File_path, "w") as sumf:
		sumf.write("Summary of Files"+os.linesep)
		sumf.write("#####################"+os.linesep+os.linesep)
		sumf.write("Root path - "+Root_path+os.linesep)
		sumf.write("Input Image path - "+Input_Image_path+os.linesep)
		sumf.write("Output Image path - "+Output_Image_path+os.linesep)
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
		#sumf.write("Net Accuracy = "+str(accuracy)+os.linesep)

except Exception as error4:
	print("Error 4 : ",str(error4))


print("Done num_hit = ",num_hit," num_miss = ",num_miss," num_images = ",num_images)