import cv2
import sys
import numpy as np
import time

def nothing(x):
    pass

faceCascade = cv2.CascadeClassifier('C:/Users/AdamD2/Downloads/opencv/build/etc/haarcascades/haarcascade_profileface.xml')
video_capture = cv2.VideoCapture(1)
cv2.namedWindow('edge')
cv2.createTrackbar('thrs1', 'edge', 0, 5000,nothing)
cv2.createTrackbar('thrs2', 'edge', 0, 5000,nothing)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray_frame = cv2.blur(frame,(5,5))
    gray_frame = cv2.cvtColor(gray_frame,cv2.COLOR_BGR2GRAY)
    thr1 = cv2.getTrackbarPos('thrs1','edge')
    thr2 = cv2.getTrackbarPos('thrs2','edge')
    edges = cv2.Canny(gray_frame,thr1,thr2)

    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, 10, 1 );
    # for x1,y1,x2,y2 in edges[0]:
    #     cv2.line(frame, (x1,y1) , (x2,y2) , (255,0,0), 2)

    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )
    # # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.ellipse(frame, (x+w/2,y+h/2), (w/2,h/2), 360, 0, 360, (255,0,0),2) 
    #     # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    height = frame.shape[0]
    width  = frame.shape[1]
    width += edges.shape[1]
    bothimages = np.zeros((height,width,3), np.uint8)
    # output(Rect(0, 0, frame1.cols, frame1.rows))
    h= frame.shape[0]
    w= frame.shape[1]
    x= frame.shape[2]
    # bothimages[0:h][:w][0:x] = frame[:][:][:]
    # bothimages[:][frame.shape[1]:][:] = frame[:][:][:]
    # edges[:][:],[0]
    edges = np.expand_dims(edges,2)
    # edges = np.resize(edges, (edges.shape[0], edges.shape[1],3))
    ed = np.dstack((edges,edges))
    edges = np.dstack((ed,edges))
    edges = np.hstack((edges,frame))
    # print edges.shape

    cv2.imshow('edge', edges )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
