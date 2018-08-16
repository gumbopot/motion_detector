# TODO: don't track camera moving

import cv2, pandas
from datetime import datetime


first_frame = None
status_list = [None, None] # None's avoid index error when checking
times = []
df = pandas.DataFrame(columns=["Start", "End"])

# trigger camera/load video
video = cv2.VideoCapture('NEW_VID_20180816_115415.mp4')

while True:
	check, frame = video.read()

	status = 0

	# Check for end of video to handle error
	try:
		frame.all()
	except AttributeError:
		print("End of video\n"
			  "OR\n"
			  "Could not load image!")
		break

	# convert to gray
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# blur for accuracy
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# store first frame of video
	if first_frame is None:
		first_frame = gray
		continue

	# calc difference
	delta_frame = cv2.absdiff(first_frame, gray)
	thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
	thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

	# find contours of moving object
	(_,cnts,_)=cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# check size of moving object
	for contour in cnts:
		if 700000 > cv2.contourArea(contour) < 1000:
			continue

		# status 1 = moving | 0 = no movement
		status = 1

		(x, y, w, h) = cv2.boundingRect(contour)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

	# record start/stop motion times
	status_list.append(status)

	#trim list to last 2 items
	status_list = status_list[-2:]

	# START
	if status_list[-1] == 1 and status_list[-2] == 0:
		# record time
		times.append(datetime.now())
	# STOP
	if status_list[-1] == 0 and status_list[-2] == 1:
		# record time
		times.append(datetime.now())

	# cv2.imshow("Capture", gray)
	# cv2.imshow("Delta Frame", delta_frame)
	# cv2.imshow("Threshold", thresh_frame)
	cv2.imshow("Color Frame", frame)

	key = cv2.waitKey(15)

	if key == ord('q'):
		if status == 1:
			times.append(datetime.now())
		break

	if status == 1:
		times.append(datetime.now())

# print(status_list)
# print(times)

for i in range(0, len(times), 2):
	df = df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)

df.to_csv("motion_times.csv")

video.release()
cv2.destroyAllWindows

