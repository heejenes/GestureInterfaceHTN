# USAGE
# python openvino_real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from picamera import PiCamera
import numpy as np
import argparse
import imutils
import time
import cv2
import math
import serial
import argparse
import time
import random



vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
vs.camera.exposure_mode = 'off'
vs.camera.awb_mode = 'off'
vs.camera.awb_gains = (1.5, 1.5)
vs.camera.framerate = 20
fps = FPS().start()

def nothing(x):
	pass

def assign_trackbar(window):
	Blur = cv2.getTrackbarPos("Blur", window)
	if Blur % 2 == 0:
		Blur += 1

	HMax = cv2.getTrackbarPos("HMax", window)
	HMin = cv2.getTrackbarPos("HMin", window)

	SMax = cv2.getTrackbarPos("SMax", window)
	SMin = cv2.getTrackbarPos("SMin", window)

	VMax = cv2.getTrackbarPos("VMax", window)
	VMin = cv2.getTrackbarPos("VMin", window)
	params = [HMax, HMin, SMax, SMin, VMax, VMin, Blur]
	return params

cv2.namedWindow('awb')
cv2.createTrackbar("bg1", "awb", 15, 80, nothing)
cv2.createTrackbar("rg1", "awb", 12, 80, nothing)
cv2.createTrackbar("iso", "awb", 800, 800, nothing)
cv2.createTrackbar("comp", "awb", 25, 50, nothing)

#arguments for # of lanes to follow
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--lanes", type=int, default=0, choices=range(0, 3),
                    help="The number of games to simulate")
args = parser.parse_args()
lane_follow_option = args.lanes
#print(lane_follow_option)	

steer_angle = 9
time.sleep(2.0)
white_offset = 0
yellow_offset = 0
offset_count = 0
offset_added = False
#writesteer(steer_angle)
#frame = cv2.imread('1.jpg')
s = 400
#frame = cv2.resize(frame, (s,s), 0, 0, cv2.INTER_AREA)
#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#blur = cv2.GaussianBlur(hsv,(7,7),0)

frame_count = 0
# loop over the frames from the video stream
print("succeeded")
print(vs.camera.resolution)
race_almost_started = False
race_primed = False
race_started = False
race_started_count = 0
race_ended = False
motor_started = False
motor_braked = False
direction_chosen = False
direction_count = 0
while True:   

	bg = (cv2.getTrackbarPos("bg1", "awb")/10)#, cv2.getTrackbarPos("bg2", "awb")) 
	rg = (cv2.getTrackbarPos("rg1", "awb")/10)#, cv2.getTrackbarPos("rg2", "awb"))
	vs.camera.awb_gains = (rg, bg)
	iso = cv2.getTrackbarPos("iso", "awb")
	vs.camera.iso = iso
	exposure = cv2.getTrackbarPos("comp", "awb") - 25
	vs.camera.exposure_compensation = exposure
	#guassian = cv2.getTrackbarPos("Gaussian", "Green")
	#if guassian % 2 == 0:
	#	guassian += 1
	frame = vs.read()
	#frame = cv2.imread('1.jpg')
	frame = cv2.resize(frame, (s,s), 0, 0, cv2.INTER_AREA)
	lines_edges = frame 
	#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#blur = cv2.GaussianBlur(hsv,(guassian,guassian),0)

	#red_params = assign_trackbar("Red")
	#magenta_params = assign_trackbar("Magenta")
	#white_params = assign_trackbar("White")
	
	#TAEHOON########################################
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#background color is [255,255,255]
	
	upper = np.array([20,255,255])
	lower = np.array([0,48,80])
	#generates mask
	colorMask = cv2.inRange(hsv,lower,upper)
	
	colorMask = cv2.medianBlur(colorMask, 3)
	
	#generates contour
	cont, a = cv2.findContours(colorMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#print(len(cont))
	if len(cont)>0:
		longest = 0
		for i in range(len(cont)):
			if len(cont[longest])<len(cont[i]):
				longest = i
		mainCont = cont[longest]
		#print("mainCont: ",mainCont)
		mainWCont = frame.copy()
		cv2.drawContours(mainWCont, [mainCont], 0, (0,255,0),3)
	
		#generates hull
		hullIndex = cv2.convexHull(mainCont, returnPoints = False)
		hull = []
		[hull.append(mainCont[i][0]) for i in hullIndex]
		#print("hull is: ",hull[0])
		
		#general functions
		def dist(pointA,pointB):
			A=pointA[0]
			B=pointB[0]
			a=((A[0]-B[0])**2+(A[1]-B[1])**2)**0.5
			#print(a)
			return a
		def average(points):
			a=0
			b=0
			#print(len(points),"points: ",points)
			for i in range(len(points)):
				a+=points[i][0][0]
				b+=points[i][0][1]
			return [int(round(a/len(points))),int(round(b/len(points)))]
		#groups hull
		maxDist = 25
		finalHull = []
		for i in range(len(hull)):
			group = [hull[i]]
			for j in range(1,len(hull)):
				a=i+j
				if i+j >= len(hull):
					a = i+j-len(hull)
				if dist(hull[a-1],hull[a])<=maxDist:
					group.append(hull[a])
				else:
					break
						
			for j in range(1,len(hull)):
				a=i-j
				if dist(hull[a+1],hull[a])<=maxDist:
					group.append(hull[a])
				else:
					break
			if [average(group)] in finalHull:
				#print("skipped")
				continue
			finalHull.append([average(group)])
		#print("finalhull is: ", finalHull)
		[cv2.circle(mainWCont, tuple(curPoint[0]),20,(0,0,255),5) for curPoint in finalHull]
		
		
		cv2.drawContours(mainWCont, [np.array(finalHull)], 0, (0,255,0),3)
		
		cv2.imshow("Cont", mainWCont)
		
		
		#convexity defects
		defects = cv2.convexityDefects(mainCont, hullIndex)
		if type(defects) is np.ndarray:
			print(defects[0][0][2])
			print(mainCont[defects[0][0][2]][0])
			[cv2.circle(mainWCont, tuple(mainCont[curPoint[0][2]][0]),20,(0,0,255),5) for curPoint in defects]

	cv2.imshow("Main", lines_edges)
	cv2.imshow("Mask", colorMask)
	
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()
	#time.sleep(0.02)

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
