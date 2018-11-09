import numpy as np
import cv2
from statistics import stdev

vs = cv2.VideoCapture(0)

if vs.isOpened(): # try to get the first frame
    rval, frame = vs.read()
    frame = cv2.flip(frame, 1)
else:
    rval = False

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
fgbg = cv2.createBackgroundSubtractorKNN()

rval, new_fr = vs.read()
new_fr = cv2.flip(new_fr, 1)

h, w = frame.shape[:2]
avgs_x = []
avgs_y = []
frame_counter = 0
gesture = ""

printed = False
while rval:
	bw_frame = cv2.cvtColor(new_fr, cv2.COLOR_RGB2GRAY)
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	fgmask = fgbg.apply(bw_frame)
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	fgmask2 = fgbg.apply(frame)
	fgmask2 = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, kernel)
	delta = cv2.absdiff(fgmask, fgmask2)
	# cv2.imshow("del", delta)
	# cv2.imshow("1" , fgmask)
	# cv2.imshow("2", fgmask2)
	blur = cv2.GaussianBlur(delta, (21,21),0)

	# cv2.imshow("masked", blur)
	_, thresh1 = cv2.threshold(blur, 250, 255, cv2.THRESH_BINARY)
	# _, thresh2 = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)
	# cv2.imshow("t1", thresh1)
	# blur2 = cv2.GaussianBlur(thresh1, (11,11), 0)
	dilated = cv2.dilate(thresh1, None, iterations=30)
	cv2.imshow("dilate", dilated)
	# cv2.imshow("t2", thresh2)
	# cv2.imshow("frame", frame)


	cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	xs = []
	ys = []
	for c in cnts[1]:
		# if the contour is too small or too big, ignore it
		if cv2.contourArea(c) < 500:
			continue
		else:
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w1, h1) = cv2.boundingRect(c)
			xs.append(x + w1/2)
			ys.append(y + h1/2)
			cv2.rectangle(dilated, (x, y), (x + w1, y + h1), (255, 0, 0), 2)
	if(len(xs) > 0):
		avgs_x.append(sum(xs)/len(xs))
		# print(avgs_x)
		avgs_y.append(sum(ys)/len(ys))
		# print(avgs_y)

	if len(xs) == 0 and len(ys) == 0:
		frame_counter+=1
	else:
		frame_counter = 0
	if(frame_counter == 5):
		frame_counter = 0
		avgs_x = []
		avgs_y = []
		if not printed:
			print(gesture)
			printed = True

	if(len(avgs_x) > 1 and len(avgs_y) > 1):
		# if True:
		if abs(avgs_x[-1] - avgs_x[-2]) <= 100 and abs(avgs_y[-1] - avgs_y[-2]) <= 100:
			std_x = stdev(avgs_x)
			std_y = stdev(avgs_y)
			# print(std_x)
			# print(std_y)
			if(std_y > std_x):
				if(avgs_y[0] > avgs_y[-1]):
					gesture = "UP"
					# print("--UP--")
				else:
					gesture = "DOWN"
					# print("--DOWN--")
			else:
				if(avgs_x[0] > avgs_x[-1]):
					gesture = "LEFT"
					# print("--LEFT--")
				else:
					gesture = "RIGHT"
					# print("--RIGHT--")
			printed = False
			# m = 0
			# div = 0
			# avgx = sum(avgs_x)/len(avgs_x)
			# avgy = sum(avgs_y)/len(avgs_y)
			# for pt in range(len(avgs_x)):
			# 	div += (avgs_x[pt] - avgx)**2
			# 	m += (avgs_x[pt] - avgx) * (avgs_y[pt] - avgy)
			
			# try:
			# 	m /= div
			# 	print("m", m)
			# 	if(abs(m) > 1):
			# 		if(avgs_y[0] > avgs_y[-1]):
			# 			gesture = "UP"
			# 		else:
			# 			gesture = "DOWN"
			# 	else:
			# 		if(avgs_x[0] > avgs_x[-1]):
			# 			gesture = "LEFT"
			# 		else:
			# 			gesture = "RIGHT"
			# 	avgs_y = []
			# 	avgs_x = []
			# except:
			# 	pass

	key = cv2.waitKey(10)
	if key == 27:
		break
	frame = new_fr.copy()
	rval, new_fr = vs.read()
	new_fr = cv2.flip(new_fr,1)


cv2.destroyAllWindows()
vs.release()