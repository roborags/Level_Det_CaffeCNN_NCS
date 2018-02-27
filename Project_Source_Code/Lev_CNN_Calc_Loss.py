# Program name : Lev_CNN_Calc_Loss.py
# Function : To calculate Loss of network using Testing/Training dataset of cropped images
# Tests only with the Network and not OpenCV
# Input : Caffe model, Deploy prototxt, Cropped Level/Non-Level images ONLY
# Output : Summary of test, missed files, hit files

from __future__ import division
import numpy as np
from PIL import Image
import caffe
import os.path
import cv2
from caffe.proto import caffe_pb2

# Set Input Folder name
Image_set_name = "Canny"

# Set Train or test file enable even/odd lines
#Txt_file_name = "train" 
Txt_file_name = "test"
#Inner_Folder_name = "Training"
Inner_Folder_name = "Validate"

# Set macro for Canny Gray images, for color set False
Gray_Image = True 
#Gray_Image = False 
#end Set

# Set path for Caffe files
Root_path = "/home/iot_lab_dnn/Level_Detection_Files/"

# Set path for input images
Image_path = Root_path+Image_set_name+"/Level_Det_Final_Images_"+Image_set_name+"/"+Inner_Folder_name+"/"

Text_path = Root_path+Image_set_name+"/"
Train_file = "lev_det_"+Txt_file_name+"_"+Image_set_name+".txt"
Hit_file = Image_set_name+"_"+Txt_file_name+"_Hit.txt"
Miss_file = Image_set_name+"_"+Txt_file_name+"_Miss.txt"
Summary_file = Image_set_name+"_"+Txt_file_name+"_Summary.txt"
Caffe_proto_path = Text_path + "Caffe_Files/" +"lenet_deploy.prototxt"
Caffe_model_path = Text_path + "Caffe_Files/" + "Level_Alex_net_iter_10000.caffemodel"

#Mean file needed only for Color images
Mean_file = Text_path + "Caffe_Files/" + "imagenet_mean.binaryproto"

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

if Gray_Image:
	print("Gray Format")
else:	
	print("Color Format")
	# Read mean image
	mean_blob = caffe_pb2.BlobProto()
	with open(Mean_file) as f:
	    mean_blob.ParseFromString(f.read())
	mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
	    (mean_blob.channels, mean_blob.height, mean_blob.width))

	# Define image transformers
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', mean_array)
	transformer.set_transpose('data', (2,0,1))
	transformer.set_channel_swap('data', (2,1,0))
	transformer.set_raw_scale('data', 255.0)

num_images = 0
num_hit = 0
num_miss = 0
#accuracy = 0.00

# Traverse the images in folder, open train, hit and miss files
File_path = Text_path + Train_file
try:
	with open(File_path, "r") as train:
		File_path = Text_path + Hit_file
		try:
			with open(File_path, "w") as hitf:
				File_path = Text_path + Miss_file
				try:
					with open(File_path, "w") as missf:
						for line in train:
							Split_text = line.split()
							File_name = Split_text[0]
							Lev_val = int(Split_text[1])
							# Seperate methods for Gray and color images
							# As input images are canny edge contours, no proc required
							if Gray_Image:
								File_path = Image_path+File_name
								im = np.array(Image.open(File_path))
								im_input = im[np.newaxis, np.newaxis, :, :]
								net.blobs['data'].reshape(*im_input.shape)
								net.blobs['data'].data[...] = im_input
							else:						
								File_path = Image_path+File_name
								img = cv2.imread(File_path, cv2.IMREAD_COLOR)
								net.blobs['data'].data[...] = transformer.preprocess('data', img)

							res = net.forward()
							data = res['loss']

							#accuracy = accuracy + net.blobs['loss'].data

							num_images+=1
							print (num_images)

							if((data[0,0] == 0) and (data[0,1] == 0)):
								Lev_found = 2
							elif (data[0,0]<data[0,1]):
								#print ("level not found")
								Lev_found = 2
							else:
								#print ("level found")
								Lev_found = 1

							if(Lev_found == Lev_val):
								num_hit+=1
								hitf.write(File_name+" "+str(Lev_val)+" "+str(Lev_found)+" Res out "+str(data[0,0])+" "+str(data[0,1])+os.linesep)
							else:
								num_miss+=1
								missf.write(File_name+" "+str(Lev_val)+" "+str(Lev_found)+" Res out "+str(data[0,0])+" "+str(data[0,1])+os.linesep)
				except Exception as error3:
					print("Error 3 : ",str(error3))	
		except Exception as error2:
			print("Error 2 : ",str(error2))		
except Exception as error1:
	print("Error 1 : ",str(error1))

#if(accuracy > 0):
#	accuracy = round((accuracy / num_images),2)
hit_rate = 0.00
miss_rate = 0.00

# Open and write the summary file
File_path = Text_path + Summary_file
try:
	with open(File_path, "w") as sumf:
		sumf.write("Summary of Files"+os.linesep)
		sumf.write("#####################"+os.linesep+os.linesep)
		sumf.write("Root path - "+Root_path+os.linesep)
		sumf.write("Image path - "+Image_path+os.linesep)
		sumf.write("Text files path - "+Text_path+os.linesep)
		sumf.write("Image input train file - "+Train_file+os.linesep)
		sumf.write("Image HIT file - "+Hit_file+os.linesep)
		sumf.write("Format: Filename-Train Level-Found Level"+os.linesep)
		sumf.write("Image MISS file - "+Miss_file+os.linesep)
		sumf.write("Format: Filename-Train Level-Found Level-Res out"+os.linesep)
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

print("Done Hit rate = ",hit_rate," Miss rate = ",miss_rate)#," Accuracy = ",accuracy)

	

