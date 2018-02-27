# Program name : Lev_Preproc_Dataset.py
# Function : Crop input images , seperate level and non-level
# apply Canny edge and seperate training and validation images
# Output to 4 folders 
# Training - Level and Non Level
# Testing - Level and Non Level

import numpy as np
import cv2
import os.path
import string
import re

main_file_count = 1
random_num = 1
#Change path to initial images seperated by levels
Root_path = "/home/rags/PycharmProjects/Level_Detection_Preproc/Images/Black_Back"

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

	im2,contours,hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) != 0:
		for c in contours:
			x,y,w,h = cv2.boundingRect(c)
			#Choose only horizontal contours and mid region
			if(w>h and (y%210)>= 55 and (y%210) <= 155):
				cv2.drawContours(cont_out, c, -1, 255, 3)
	return cont_out

#traverse through files in level directory
for root, dirs, files in os.walk(Root_path,topdown=True):
	for level_dir in dirs:
		filenum = 1
		level = int(level_dir)
		print(level)
		Level_path = Root_path + "/" + level_dir + "/"
		for root, dirs, files in os.walk(Level_path,topdown=True):
			for filename in files:
				Input_FIlename = Level_path+"/"+filename
				input = cv2.imread(Input_FIlename, cv2.IMREAD_COLOR)
				filename_str = filename.replace(".JPG","")
				#Crop center region to get testtube ROI
				#Seperate crop sections for Black and White images 
				#cropped = input[813:2913, 1780:2240] # White Image crop
				cropped = input[690:2790, 1685:2145] # Black Image crop

				dim = (227, 227)
				#Convert to gray and apply GaussianBlur for Canny
				gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

				blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

				#Extract level image
				output_1 = blur_image[((10-level)*210):(((10-level)*210)+210),0:460]

				auto_edge_1 = auto_canny(output_1)

				cont_out_1 = draw_cnt(auto_edge_1)
				#Resize for CNN input size
				output_resize_1 = np.zeros((227,227,1),np.uint8)

				output_resize_1 = cv2.resize(cont_out_2,dim,0,0.5,interpolation = cv2.INTER_LANCZOS4)
				#cv2.convertScaleAbs(output_resize_1, output_resize_1 , 1.5, 0) #Comment for white BB
				#Total 60 images per level, 42 for Training 18 for validation, 70:30
				if(filenum <43):
					Output_FIlename = "Images/Cropped_Images/Training/Level/"+filename_str+str(main_file_count).zfill(4)+".JPG"
					cv2.imwrite(Output_FIlename,output_resize_1)
				else:
					Output_FIlename = "Images/Cropped_Images/Testing/Level/"+filename_str+str(main_file_count).zfill(4)+".JPG"
					cv2.imwrite(Output_FIlename,output_resize_1)
					
				main_file_count+=1
				#print(filenum,Output_FIlename)
				#if(level):
					#print(random_num)
				#Repeat process for one random non-level image
				output_2 = blur_image[((10-random_num)*210):(((10-random_num)*210)+210),0:460]

				auto_edge_2 = auto_canny(output_2)

				cont_out_2 = draw_cnt(auto_edge_2)

				output_resize_2 = np.zeros((227,227,1),np.uint8)

				output_resize_2 = cv2.resize(cont_out_2,dim,0,0.5,interpolation = cv2.INTER_LANCZOS4)
				
				if(filenum <43):
					Output_FIlename = "Images/Cropped_Images/Training/Non_Level/"+filename_str+str(main_file_count).zfill(4)+".JPG"
					cv2.imwrite(Output_FIlename,output_resize_2)
				else:
					Output_FIlename = "Images/Cropped_Images/Testing/Non_Level/"+filename_str+str(main_file_count).zfill(4)+".JPG"
					cv2.imwrite(Output_FIlename,output_resize_2)
				
				main_file_count+=1
				#print(filenum,Output_FIlename)
				#generate random number
				random_num += 1
				if(random_num > 10):
					if(level == 1):
						random_num = 2
					else:
						random_num = 1
				elif(random_num == level):
					if(level == 10):
						random_num = 1
					else:
						random_num += 1
				filenum+=1
