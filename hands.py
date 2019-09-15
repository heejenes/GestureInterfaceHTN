
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
from pynput.mouse import Button, Controller

speed = -0.8

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32

# allow the camera to warmup and start the FPS counter
time.sleep(5.0)

# do a bit of cleanup
camera.close()

vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
vs.camera.exposure_mode = 'off'
vs.camera.awb_mode = 'off'
vs.camera.awb_gains = (1.4, 1.4)
#vs.camera.resolution = (640, 480)
#vs.camera.framerate = 80
fps = FPS().start()
bg = None

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

cv2.namedWindow('Skin')

cv2.createTrackbar("HMax", "Skin",176,255,nothing)
cv2.createTrackbar("HMin", "Skin",0,255,nothing)

cv2.createTrackbar("SMax", "Skin",255,255,nothing)
cv2.createTrackbar("SMin", "Skin",1,255,nothing)

cv2.createTrackbar("VMax", "Skin",255,255,nothing)
cv2.createTrackbar("VMin", "Skin",50,255,nothing)

cv2.createTrackbar("Blur", "Skin",6,10,nothing)


cv2.namedWindow('awb')
cv2.createTrackbar("bg1", "awb", 15, 80, nothing)
cv2.createTrackbar("rg1", "awb", 12, 80, nothing)
cv2.createTrackbar("iso", "awb", 400, 800, nothing)
cv2.createTrackbar("comp", "awb", 20, 50, nothing)

cv2.createTrackbar("threshold", "awb", 5, 20, nothing)

s = 500
#frame = cv2.resize(frame, (s,s), 0, 0, cv2.INTER_AREA)
#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#blur = cv2.GaussianBlur(hsv,(7,7),0)

frame_count = 0
# loop over the frames from the video stream
print("succeeded")
print(vs.camera.resolution)
aWeight = 0.5
hand = None
mouse_zero_zero = [0, 0]

mouse_clicked = False

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
	threshold = cv2.getTrackbarPos("threshold", "awb")
	frame = vs.read()
	#frame = cv2.imread('1.jpg')
	#frame = cv2.resize(frame, (s,s), 0, 0, cv2.INTER_AREA)
	lines_edges = frame
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#blur = cv2.GaussianBlur(hsv,(guassian,guassian),0)

	#TAEHOON########################################
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#background color is [255,255,255]
	
	skin_params = assign_trackbar("Skin")
	#print(skin_params)
	lower = np.array([skin_params[1],skin_params[3],skin_params[5]])
	upper = np.array([skin_params[0],skin_params[2],skin_params[4]])
	#generates mask
	colorMask = cv2.inRange(hsv,lower,upper)

	if frame_count < 30:
		pass
		#run_avg(gray, aWeight)
		#print("calibrating")
	else:
			# segment the hand region
			#hand = segment(gray, threshold)

			# check whether hand region is segmented
			if hand is not None:
				pass
				# if yes, unpack the thresholded image and
				# segmented region
				#(thresholded, segmented) = hand

				# draw the segmented region and display the frame
				#cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
				#cv2.imshow("Thesholded", thresholded)

	if frame is not None:
		#print("not none")
		#(thresholded, segmented) = hand
		colorMask = cv2.medianBlur(colorMask, skin_params[6])
		
		#generates contour
		cont, a = cv2.findContours(colorMask ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		#print(len(cont))
		if len(cont)>0:
			longest = 0
			for i in range(len(cont)):
				if len(cont[longest])<len(cont[i]):
					longest = i
			mainCont = cont[longest]
			#print("mainCont: ",mainCont)

			M = cv2.moments(mainCont)
			mo = M['m00']
			if mo != 0:
				cx = int(M['m10']/mo)
				cy = int(M['m01']/mo)
				mainWCont = frame.copy()
				cv2.circle(mainWCont, (cx, cy), threshold, (255,255,0) ,5)
			else:
				mainWCont = frame.copy()
			
			cv2.drawContours(mainWCont, [mainCont], 0, (0,255,0),3)
		
			#generates hull
			hullIndex = cv2.convexHull(mainCont, returnPoints = False)
			hull = []
			[hull.append(mainCont[i][0]) for i in hullIndex]
			#print("hull is: ",hull[0])
			
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
			[cv2.circle(mainWCont, tuple(curPoint[0]),20,(0,0,150),5) for curPoint in finalHull]

			largestX = 0
			smallestX = 0
			largestY = 0
			smallestY = 0
			for i in range(len(finalHull)):
				if finalHull[i][0][0] > finalHull[largestX][0][0]:
					largestX = i
				if finalHull[i][0][0] < finalHull[smallestX][0][0]:
					smallestX = i
				if finalHull[i][0][1] > finalHull[largestY][0][1]:
					largestY = i
				if finalHull[i][0][1] < finalHull[smallestY][0][1]:
					smallestY = i
			#print("fhuirehfiuewhfiurehgurehgurehg")
			#print(finalHull[largestX],finalHull[smallestX])
			#print(finalHull[largestY],finalHull[smallestY])

			lineThickness = 3
			horizontalY = int((finalHull[largestX][0][1] + finalHull[smallestX][0][1])/2)
			verticalX = int((finalHull[largestY][0][0] + finalHull[smallestY][0][0])/2)

			#Horizontal Axis
			cv2.line(mainWCont, (finalHull[largestX][0][0], horizontalY), (finalHull[smallestX][0][0], horizontalY), (0,255,255), lineThickness)
			#Vertical Axis
			cv2.line(mainWCont, (verticalX, finalHull[largestY][0][1]), (verticalX, finalHull[smallestY][0][1]), (255,0,0), lineThickness)
			#Threshold Axis
			
			cv2.drawContours(mainWCont, [np.array(finalHull)], 0, (0,100,0),2)
			#threshold_x = int(cx+(verticalX-cx)*1.5)
			threshold_x = verticalX


			if horizontalY > cy - threshold and horizontalY < cy + threshold:
				thumb_present = False
				
			else:
				thumb_present = True
				cv2.line(mainWCont, (threshold_x, finalHull[largestY][0][1]), (threshold_x, finalHull[smallestY][0][1]), (100,0,100), lineThickness)

			total_finger_x = 0
			total_finger_y = 0
			total_fingers = 0
			#How to detect the fingers, using the thumb or no 
			if thumb_present:
				for i in range(len(finalHull)):
					if finalHull[i][0][0] > threshold_x:
						cv2.circle(mainWCont, (finalHull[i][0][0], finalHull[i][0][1]), 10, (0,0,255), 5)
						total_fingers += 1
						total_finger_x += finalHull[i][0][0]
						total_finger_y += finalHull[i][0][1]
			else:
				
				#Which finger to base detection off of
				if finalHull[largestY][0][0] < finalHull[smallestY][0][0]:
					cv2.line(mainWCont, (finalHull[smallestY][0][0] - 100, finalHull[largestY][0][1]), (finalHull[largestY][0][0], finalHull[smallestY][0][1]), (200,0,200), lineThickness)
					for i in range(len(finalHull)):
						if finalHull[i][0][0] >= finalHull[largestX][0][0] - 100:
							cv2.circle(mainWCont, (finalHull[i][0][0], finalHull[i][0][1]), 10, (0,0,255), 5)
				else:
					cv2.line(mainWCont, (finalHull[smallestY][0][0] - 100, finalHull[largestY][0][1]), (finalHull[smallestY][0][0], finalHull[smallestY][0][1]), (200,0,200), lineThickness)
					for i in range(len(finalHull)):
						if finalHull[i][0][0] >= finalHull[largestX][0][0] - 100:
							cv2.circle(mainWCont, (finalHull[i][0][0], finalHull[i][0][1]), 10, (0,0,255), 5)
			

			# If mouse has been calibrated
			avg_finger_pos = [0, 0]
			mouse_velocity = [0, 0]
			if mouse_zero_zero[0] != 0 and mouse_zero_zero[1] != 0:
				#print(mouse_zero_zero)
				cv2.circle(mainWCont, (mouse_zero_zero[0], mouse_zero_zero[1]), 15, (100,100,0), 2)
				# Calculate avg finger point (only thumb mode)
				if total_fingers != 0:
					avg_finger_pos[0] = int(total_finger_x / total_fingers)
					avg_finger_pos[1] = int(total_finger_y / total_fingers)
					cv2.circle(mainWCont, (avg_finger_pos[0], avg_finger_pos[1]), 4, (0,100,255), 4)

					#print(avg_finger_pos, mouse_zero_zero)

					# check if movement is above threshold value
					if avg_finger_pos[0] < mouse_zero_zero[0] + 15 and avg_finger_pos[0] > mouse_zero_zero[0] - 15:
						avg_finger_pos[0] = 0

					# check if movement is above threshold value
					if avg_finger_pos[1] < mouse_zero_zero[1] + 15 and avg_finger_pos[1] > mouse_zero_zero[1] - 15:
						avg_finger_pos[1] = 0

					# Calculate velocity
					if avg_finger_pos[0] != 0 :
						mouse_velocity[0] = (mouse_zero_zero[0] - avg_finger_pos[0])/ 2
					if avg_finger_pos[1] != 0:
						mouse_velocity[1] = (mouse_zero_zero[1] - avg_finger_pos[1])/ 2
					#print(mouse_velocity, mouse_zero_zero, avg_finger_pos)
					

			else:
				if total_fingers != 0:
					avg_finger_pos[0] = int(total_finger_x / total_fingers)
					avg_finger_pos[1] = int(total_finger_y / total_fingers)
					print(avg_finger_pos)
					cv2.circle(mainWCont, (avg_finger_pos[0], avg_finger_pos[1]), 4, (0,100,255), 4)

			#convexity defects
			defects = cv2.convexityDefects(mainCont, hullIndex)
			if type(defects) is np.ndarray:
				#print(defects[0][0][2])
				#print(mainCont[defects[0][0][2]][0])
				#[cv2.circle(mainWCont, tuple(mainCont[curPoint[0][2]][0]),20,(255,0,255),5) for curPoint in defects]
				cv2.imshow("Cont", mainWCont)
			else:
				cv2.imshow("Cont", mainWCont)
			
			#INPUT EMULATOR
			mouse = Controller()
			if mouse_velocity[0] != 0 or mouse_velocity[1] != 0:
				mouse.move(speed * mouse_velocity[0], speed * mouse_velocity[1])

			if mouse_zero_zero[0] != 0 and mouse_zero_zero[1] != 0:
				print(total_fingers)
				final = total_fingers
				if final == 3 and not mouse_clicked:
					mouse.press(Button.left)
					mouse_clicked = True
				elif final == 2 and not mouse_clicked:
					mouse.press(Button.middle)
					mouse_clicked = True
				elif final == 1 and not mouse_clicked:
					mouse.press(Button.right)
					mouse_clicked = True

				elif final == 4:
					if mouse_clicked:
						mouse.release(Button.left)
						mouse.release(Button.right)
						mouse.release(Button.middle)
						mouse_clicked = False

	cv2.imshow("Main", lines_edges)
	if colorMask is not None:
		cv2.imshow("Mask", colorMask)
	
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# if the `q` key was pressed, break from the loop
	if key == ord("z"):
		mouse_zero_zero[0] = avg_finger_pos[0]
		mouse_zero_zero[1] = avg_finger_pos[1]


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
# I am helping - Michael Denissov, Taehoon Kim, Adeeb Mahmud
