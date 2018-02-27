Level Detection Project Source Files :
Make sure to change the appropriate paths before compiling.
Download the required files as metioned in main ReadMe.

1. Lev_Preproc_Dataset.py 
Used to preprocess the images by cropping and OpenCV filters and store cropped images.

2. Lev_Dataset_FileList.py
Classify images for Training and Validation and preparing corresponding filelist.

3. Lev_CNN_Calc_Loss.py
Load the trained network and test the images by just using the network. Calculate loss.

4. Lev_Test_CNN_Dataset.py
Test Dataset images with Network and OpenCV methods. Calulate loss summary.

5. Lev_Rasp_NCS_Test_CNN_Dataset.py
Test Dataset images with CNN deploy net running on NCS using RaspPi. Calculate loss summary.

6. Lev_Rasp_Calibrate_Camera.py
One time calibration of camera and test-tube setup before detecting levels.

7. Lev_Rasp_NCS_Final_Setup.py
Final project setup that grabs images from GoPro, find levels on image and saves in RaspPi the images with level marking.