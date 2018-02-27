# Program name : Lev_Preproc_Dataset.py
# Function : Classify cropped images to Training and Validation folder
# Prepares two txt files for training and testing filenames , used in CNN
# 60 img/level, Cropped 600 level and 600 non-level
# Training Validation 70:30 840:360 image
# Move images to two folders Training and Validation
# Txt format : Filename Level_ID(1 for level, 2 for non-level)

import cv2
import os.path
import string
import re
import random
import shutil

# Training Validation txt file names for CNN
Train_file = "lev_det_train_CannyNew.txt"
Test_file = "lev_det_test_CannyNew.txt"

# Set Image root path and Moving paths
Root_path = "/home/rags/PycharmProjects/Level_Detection_Preproc/Images/Cropped_Images_CannyNew/"
Dest_path = "/home/rags/PycharmProjects/Level_Detection_Preproc/Images/Level_Det_Final_Images_CannyNew/"

#Process for training dataset
with open(Train_file, "w") as train:
    main_file_count = 0
	# Split 70:30 Training Validation ratio
    while (main_file_count < 840): 
		#Randomly mix level and non-level files to prevent CNN from overfitting
        random_num = random.randint(1,5)
        file_num = 0
        Folder_path = Root_path+"Training/Level"
        for root, dirs, files in os.walk(Folder_path):
            for filename_lev in files:
                File_Path = Folder_path + "/" + filename_lev
                Move_Path = Dest_path + "Training/" + filename_lev
                shutil.move(File_Path,Move_Path)
                train.write(filename_lev+" "+"1"+os.linesep)
                print(main_file_count,file_num,random_num,filename_lev)
                file_num += 1
                if(file_num >= random_num):
                    break
        main_file_count+=file_num
        random_num = random.randint(1,5)
        file_num = 0
        Folder_path = Root_path+"Training/Non_Level"
        for root, dirs, files in os.walk(Folder_path):
            for filename_nl in files:
                File_Path = Folder_path + "/" + filename_nl
                Move_Path = Dest_path + "Training/"  + filename_nl
                shutil.move(File_Path,Move_Path)
                train.write(filename_nl+" "+"2"+os.linesep)
                print(main_file_count,file_num,random_num,filename_nl)
                file_num += 1
                if(file_num >= random_num):
                    break
        main_file_count+=file_num

        if(main_file_count == 839):    
            break;
        #print(main_file_count,file_num,random_num)
print(main_file_count)

#Process for Validation dataset
with open(Test_file, "w") as train:
    main_file_count = 0
    while (main_file_count < 360):      
        random_num = random.randint(1,5)
        file_num = 0
        Folder_path = Root_path+"Testing/Level"
        for root, dirs, files in os.walk(Folder_path):
            for filename_tlev in files:
                File_Path = Folder_path + "/" + filename_tlev
                Move_Path = Dest_path + "Validate/"  + filename_tlev
                shutil.move(File_Path,Move_Path)
                train.write(filename_tlev+" "+"1"+os.linesep)
                print(main_file_count,file_num,random_num,filename_tlev)
                file_num += 1
                if(file_num >= random_num):
                    break
        main_file_count+=file_num

        random_num = random.randint(1,5)
        file_num = 0
        Folder_path = Root_path+"Testing/Non_Level"
        for root, dirs, files in os.walk(Folder_path):
            for filename_tnl in files:
                File_Path = Folder_path + "/" + filename_tnl
                Move_Path = Dest_path + "Validate/"  + filename_tnl
                shutil.move(File_Path,Move_Path)
                train.write(filename_tnl+" "+"2"+os.linesep)
                print(main_file_count,file_num,random_num,filename_tnl)
                file_num += 1
                if(file_num >= random_num):
                    break
        main_file_count+=file_num

        if(main_file_count == 359):     
            break;
print(main_file_count)
