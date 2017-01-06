import numpy as np 
import cv2
import sys
from time import time

import kcftracker

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.01

# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
	global selectingObject, initTracking, onTracking, ix, iy, cx,cy, w, h
	
	if event == cv2.EVENT_LBUTTONDOWN:
		selectingObject = True
		onTracking = False
		ix, iy = x, y
		cx, cy = x, y
	
	elif event == cv2.EVENT_MOUSEMOVE:
		cx, cy = x, y
	
	elif event == cv2.EVENT_LBUTTONUP:
		selectingObject = False
		if(abs(x-ix)>10 and abs(y-iy)>10):
			w, h = abs(x - ix), abs(y - iy)
			ix, iy = min(x, ix), min(y, iy)
			initTracking = True
		else:
			onTracking = False
	
	elif event == cv2.EVENT_RBUTTONDOWN:
		onTracking = False
		if(w>0):
			ix, iy = x-w/2, y-h/2
			initTracking = True



if __name__ == '__main__':
	
	if(len(sys.argv)==1):
		cap = cv2.VideoCapture(0)
	elif(len(sys.argv)==2):
		if(sys.argv[1].isdigit()):  # True if sys.argv[1] is str of a nonnegative integer
			cap = cv2.VideoCapture(int(sys.argv[1]))
		else:
			cap = cv2.VideoCapture(sys.argv[1])
			inteval = 30
	else:  assert(0), "too many arguments"

	tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
	#if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.

	cv2.namedWindow('tracking')
	cv2.setMouseCallback('tracking',draw_boundingbox)

	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret:
			break

		if(selectingObject):
			cv2.rectangle(frame,(ix,iy), (cx,cy), (0,255,255), 1)
		elif(initTracking):
			cv2.rectangle(frame,(ix,iy), (ix+w,iy+h), (0,255,255), 2)

			tracker.init([ix,iy,w,h], frame)

			initTracking = False
			onTracking = True
		elif(onTracking):
			t0 = time()
			boundingbox = tracker.update(frame)
			t1 = time()

			boundingbox = map(int, boundingbox)
			cv2.rectangle(frame,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,255,255), 1)
			
			duration = 0.8*duration + 0.2*(t1-t0)
			#duration = t1-t0
			cv2.putText(frame, 'FPS: '+str(1/duration)[:4].strip('.'), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

		cv2.imshow('tracking', frame)
		c = cv2.waitKey(inteval) & 0xFF
		if c==27 or c==ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
