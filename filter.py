import os
os.system('pip install imutils')

import numpy as np

import cv2
from utils import image_resize
from imutils import resize
from imutils.video import VideoStream

import face_recognition

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('cascades/third-party/frontalEyes35x16.xml')

polaroid = cv2.imread('polaroid.png', -1)
polaroid = resize(polaroid, height=300, width=300)
polaroid = cv2.cvtColor(polaroid, cv2.COLOR_BGR2BGRA)
# cv2.imshow("overlay_frame", polaroid)
polaroid_h, polaroid_w, polaroid_c = polaroid.shape

mask = cv2.imread('mask.png', -1)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)

# cv2.imshow("mask", mask)

# print (mask.shape)
# print (polaroid.shape)
video_capture = VideoStream().start()
while True:
	frame = video_capture.read()
	frame = resize(frame, height=300, width=400)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
	frame_h, frame_w, frame_c = frame.shape
	frame = cv2.resize(frame, (frame_h, frame_h), interpolation = cv2.INTER_CUBIC)
	# print (frame.shape)
	overlay = np.zeros((frame_h, frame_h, 4), dtype = 'uint8')
	# cv2.imshow("overlay", overlay)	
	
	polaroid_h, polaroid_w, polaroid_c = polaroid.shape
	for i in range(0, polaroid_h):
		for j in range(0, polaroid_w):
			overlay [i,j] = polaroid [i,j]
	
	cv2.addWeighted (overlay, 1.0, frame, 1.0, 0, frame)
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
	
	for (x, y, w, h) in faces:
		roi_gray    = gray[y:y+h, x:x+h] # rec
		roi_color   = frame[y:y+h, x:x+h]

		eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
		for (ex, ey, ew, eh) in eyes:
            # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
			roi_eyes = roi_gray[ey: ey + eh, ex: ew + ew] 
			mask = image_resize(mask.copy(), width= ew)
			mw, mh, mc = mask.shape

			for i in range(0, mw):
				for j in range(0, mh):
                    # print(mask[i, j]) #RiGBA
					if mask[i, j][3] != 0:
						roi_color[ey + i - 7, ex + j] = mask[i, j]	

	frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
	cv2.imshow("Video Stream", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
