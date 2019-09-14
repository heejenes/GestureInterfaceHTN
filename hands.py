
from imutils.video import VideoStream
from imutils.video import FPS
from picamera import PiCamera
import numpy as np
import imutils
import time
import cv2
import math
import time
import random


vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
vs.camera.exposure_mode = 'off'
vs.camera.awb_mode = 'off'
vs.camera.awb_gains = (1.5, 1.5)
vs.camera.framerate = 20
fps = FPS().start()
bg = None

def nothing(x):
	pass

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

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

s = 400
#frame = cv2.resize(frame, (s,s), 0, 0, cv2.INTER_AREA)
#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#blur = cv2.GaussianBlur(hsv,(7,7),0)

frame_count = 0
# loop over the frames from the video stream
print("succeeded")
print(vs.camera.resolution)
aWeight = 0.5
hand = None

while True:   

	bluegain = (cv2.getTrackbarPos("bg1", "awb")/10)#, cv2.getTrackbarPos("bg2", "awb")) 
	rg = (cv2.getTrackbarPos("rg1", "awb")/10)#, cv2.getTrackbarPos("rg2", "awb"))
	vs.camera.awb_gains = (rg, bluegain)
	iso = cv2.getTrackbarPos("iso", "awb")
	vs.camera.iso = iso
	exposure = cv2.getTrackbarPos("comp", "awb") - 25
	vs.camera.exposure_compensation = exposure
	#guassian = cv2.getTrackbarPos("Gaussian", "Green")
	#if guassian % 2 == 0:
	#	guassian += 1
	frame = vs.read()
	#frame = cv2.imread('1.jpg')
	#frame = cv2.resize(frame, (s,s), 0, 0, cv2.INTER_AREA)
	lines_edges = frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#blur = cv2.GaussianBlur(hsv,(guassian,guassian),0)

	#TAEHOON########################################
	#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#background color is [255,255,255]
	
	#upper = np.array([20,255,255])
	#lower = np.array([0,48,80])
	#generates mask
	#colorMask = cv2.inRange(hsv,lower,upper)

	if frame_count < 30:
		run_avg(gray, aWeight)
	else:
			# segment the hand region
			hand = segment(gray)

			# check whether hand region is segmented
			if hand is not None:
				# if yes, unpack the thresholded image and
				# segmented region
				(thresholded, segmented) = hand

				# draw the segmented region and display the frame
				#cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
				#cv2.imshow("Thesholded", thresholded)

	if hand is not None:
		(thresholded, segmented) = hand
		thresholded = cv2.medianBlur(thresholded, 3)
		
		#generates contour
		cont, a = cv2.findContours(thresholded ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
	if hand is not None:
		cv2.imshow("Mask", thresholded)
	
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

	frame_count += 1
	

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
