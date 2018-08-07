import cv2#, time


first_frame = None

# trigger camera
video = cv2.VideoCapture('VID_20180807_095930.mp4')

while True:
    check, frame = video.read()

    # convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # store first frame of video
    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (_,cnts,_)=cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue

        (x, y, w, h) = cv2. boundRect(contour)
        cv2.rectabgle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("Capture", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(0)
    print(gray)


    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows

