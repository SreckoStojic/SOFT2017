from __future__ import print_function
from skimage.feature import hog
from sklearn import datasets
from skimage.color import rgb2gray
from skimage.morphology import opening, closing, erosion
from sklearn.externals import joblib
from skimage.morphology import square, disk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import os
import cv2
import math
import matplotlib.path as mplPath
import imutils

f = open("out.txt", "w")
f.write("RA 15/2012,Srecko Stojic\n")
f.write("file,count\n")
for vid_nmb in range(1, 11):
    name = "video{0}.mp4".format(vid_nmb)
    vs = cv2.VideoCapture(name)
    brojac = 0
    firstFrame = None
    while True:
        frame = vs.read()
        frame = frame[1]

        if frame is None:
            break
        frame = imutils.resize(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if firstFrame is None:
            firstFrame = gray

            # nadji gornju granicu platoa
            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray1, (3, 3), 0)
            edges = cv2.Canny(blurred, 200, 200)

            width, height = frame.shape[:2]

            plato_x1_coord = 0
            plato_x2_coord = 0
            plato_y1_coord = 0
            plato_y2_coord = 0
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                plato_x1_coord = x1
                plato_x2_coord = x2
                plato_y1_coord = y1
                plato_y2_coord = y2
            continue

        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # nadji sve konture na slici
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        for c in cnts:
            # ignorisi mnogo male povrsine
            if cv2.contourArea(c) < 50:
                continue

            # nadji pravougaonik oko coveka
            rect = cv2.boundingRect(c)

            # proveri da li su se pravougaonik i linija presekli
            x3 = rect[0] + rect[2]
            y3 = rect[1] + rect[3]
            px = plato_x2_coord - plato_x1_coord
            py = plato_y2_coord - plato_y1_coord
            something = px * px + py * py
            u = ((x3 - plato_x1_coord) * px + (y3 - plato_y1_coord) * py) / float(something)
            if u > 1:
                u = 1
            elif u < 0:
                u = 0
            _x = plato_x1_coord + u * px
            _y = plato_y1_coord + u * py
            dx = _x - x3
            dy = _y - y3
            dist = math.sqrt(dx * dx + dy * dy)

            if dist >= 0 and dist <= 0.80:
                brojac = brojac + 1
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 4)
            else:
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)

        cv2.imshow("Video", frame)

        # prekini video
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    f.write(name + "," + brojac.__str__() + "\n")
    print(name, " -- Na platou je bilo ", brojac, " ljudi")
    vs.release()
    cv2.destroyAllWindows()