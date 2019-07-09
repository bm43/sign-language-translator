import os, sys
import numpy as np
import cv2

path = ""
videos = os.listdir(path)
output_name = []
a = 0
b = 0

for i in videos:
	name = (path + '\\' + i)
	cap = cv2.VideoCapture(name)
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(200, 5, 0.7, 3)
	temp_name = i.replace('.mp4', 'mhi_cropped_potato_.jpg')

	output_name += [temp_name]
	name_out = "" + output_name[a]
	ret, frame = cap.read()
	out = np.zeros((340,340))
	x = 0

	while (ret == True):

		if ((b % 10) == 0):
			print ("a: ", a)
			print ("ret: ", ret)
			mid_height = int(frame.shape[0]/2)
			mid_width = int(frame.shape[1]/2)
			new_image = fgbg.apply(frame)
			roi = new_image [mid_height-170 : mid_height+170, mid_width-170 : mid_width+170]
			#roi[np.where((roi == [255]).all)] = 50
			#roi_rgb[roi[np.where((roi == [100]).all)]] = [23, 23, 23]
			roi[np.where(roi == [255])] = [x]
			out = out + roi
			x += 30


		ret, frame = cap.read()

		b +=1

	cv2.imwrite(name_out, out)
	a += 1

cap.release()
