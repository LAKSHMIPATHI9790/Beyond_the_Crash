import cv2
import time
from tkinter import *
from tkinter import messagebox
import pygame
import smtplib

pygame.init()

pygame.mixer.music.load("alarm.wav")
# Capturing Video
cap = cv2.VideoCapture("assets/i.mp4")

if not cap.isOpened():
    print("Initialising the capture...")
    cap.open("assets/test_35.mp4")
    print("Done.")

# Subtracting the background
subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=20)

# Text setting up
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (40, 50)

# fontScale
fontScale = 0.8

# Blue color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

res = 1

arcount = 0

count = -1

while True:
    # Reading the frame
    res, frame = cap.read()

    if res == True:
        # Applying the mask
        mask = subtractor.apply(frame)

    # finding the contours
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Flag for accident detection
        flag = 0

        for cnts in contours:
            (x, y, w, h) = cv2.boundingRect(cnts)

            if w * h > 1000:
                if flag == 1:
                    # If accident detected change color if contours to red
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 0, 255), 3)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 3)

        # If area of rectangle more than a threshold detect accident
            if w * h > 10000:
                area = w * h
            # Countint the number of frames for which the condition persists to refine the accident detection case
                arcount += 1
            # print(arcount)
            if arcount > 35:
                flag = 1

        if flag == 1:
            # If accident detected print Accident on the screen
            frame = cv2.putText(frame, "Accident Detected ", org, font,
                                fontScale, color, thickness, cv2.LINE_AA, False)
            try:
                pygame.mixer.music.play()
                s = smtplib.SMTP('smtp.gmail.com', 587)

                s.starttls()

                s.login("info.acdtest@gmail.com", "Dob11011993")

                message = "Accident Detected"

                s.sendmail("keralamanirajendran17@gmail.com", "dummyidnisha0@gmail.com", message)

                s.quit()
                #messagebox.showerror("Accident", "Accident Detected")


            except:
                pass
           # frame = cv2.putText(frame, "28° 35' 31.7040'' N and 77° 2' 45.7836'' E. ; ",
                               # (40, 80), font, fontScale, color, thickness, cv2.LINE_AA, False)
           # frame = cv2.putText(frame, "time:1600 hrs; ", (40, 100),
                               # font, fontScale, color, thickness, cv2.LINE_AA, False)
           # frame = cv2.putText(frame, "camera id:DWARKA_ROAD_22", (40, 120),
                               # font, fontScale, color, thickness, cv2.LINE_AA, False)

        cv2.imshow("Accident detection", frame)

        count += 1

        if cv2.waitKey(33) & 0xff == 27:
            break

    else:
        break

#2 addresss
#testing
#integrate
import os
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

vidcap = cv2.VideoCapture('assets/test_35.mp4')

count = 0

while(vidcap.isOpened()): 
    ret, image = vidcap.read()
    
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(length)

    if (count < length) :
        if(int(vidcap.get(1)) % 50 == 0):
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            cv2.imwrite("frame%d.jpg" % count, image)
            print('Saved frame%d.jpg' % count)
            count += 1
        else :
            count += 1
    else :
        if(int(vidcap.get(1)) % 50 == 0):
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            cv2.imwrite("frame%d.jpg" % count, image)
            print('Saved frame%d.jpg' % count)
        break

vidcap.release()


allfiles=os.listdir(os.getcwd())
imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

w,h=Image.open(imlist[0]).size
N=len(imlist)

arr=np.zeros((h,w,3),np.float)

for im in imlist:
    imarr=np.array(Image.open(im),dtype=np.float)
    arr=arr+imarr/N

arr=np.array(np.round(arr),dtype=np.uint8)

image = Image.fromarray(arr,mode="RGB")
image.save("average.jpg")



image = cv2.imread('average.jpg', cv2.IMREAD_COLOR)
print(image.shape)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = image.reshape((image.shape[0] * image.shape[1], 3)) # height, width
print(image.shape)

# (25024, 3)

k = 5
clt = KMeans(n_clusters = k)
clt.fit(image)

for center in clt.cluster_centers_:
    print(center)

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, ) = np.histogram(clt.labels, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


hist = centroid_histogram(clt)
print(hist)

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
        print(percent, color)
    # return the bar chart
    return bar

bar = plot_colors(hist, clt.cluster_centers_)

# show our color bar t
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
import cv2
import time
from tkinter import *
from tkinter import messagebox
#from gps import *
import folium as folium
import geocoder as geocoder
import pygame
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from geopy.geocoders import Nominatim
loc = Nominatim(user_agent="GetLoc")

# entering the location name
getLoc = loc.geocode("OMR,Chennai")

# printing address
print(getLoc.address)

# printing latitude and longitude
print("Latitude = ", getLoc.latitude, "\n")
print("Longitude = ", getLoc.longitude)
url_ = "https://maps.google.com/?q="+str(getLoc.latitude)+","+str(getLoc.longitude)
print(url_)
fromaddr = "info.acdtest@gmail.com"
toaddr = "dummyidnisha0@gmail.com"

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
pygame.init()

pygame.mixer.music.load("alarm.wav")
# Capturing Video
cap = cv2.VideoCapture("0")
status=0

if not cap.isOpened():
    print("Initialising the capture...")
    cap.open("assets/test_0.mp4")
    print("Done.")

# Subtracting the background
subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=20)

# Text setting up
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (40, 50)

# fontScale
fontScale = 0.8

# Blue color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

res = 1

arcount = 0

count = -1

while True:
    # Reading the frame
    res, frame = cap.read()

    if res == True:
        # Applying the mask
        mask = subtractor.apply(frame)

    # finding the contours
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Flag for accident detection
        flag = 0

        for cnts in contours:
            (x, y, w, h) = cv2.boundingRect(cnts)

            if w * h > 1000:
                if flag == 1:
                    # If accident detected change color if contours to red
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 0, 255), 3)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 3)

        # If area of rectangle more than a threshold detect accident
            if w * h > 10000:
                area = w * h
            # Countint the number of frames for which the condition persists to refine the accident detection case
                arcount += 1
            # print(arcount)
            if arcount > 35:
                flag = 1

        if flag == 1:
            # If accident detected print Accident on the screen
            frame = cv2.putText(frame, "Accident Detected ", org, font,
                                fontScale, color, thickness, cv2.LINE_AA, False)
            try:
                cv2.imwrite('img.jpg', frame)
                pygame.mixer.music.play()
                #pygame.mixer.music.play()
                status=1

            except:
                pass

            # messagebox.showerror("Accident", "Accident Detected")

           # frame = cv2.putText(frame, "28° 35' 31.7040'' N and 77° 2' 45.7836'' E. ; ",
                               # (40, 80), font, fontScale, color, thickness, cv2.LINE_AA, False)
           # frame = cv2.putText(frame, "time:1600 hrs; ", (40, 100),
                               # font, fontScale, color, thickness, cv2.LINE_AA, False)
           # frame = cv2.putText(frame, "camera id:DWARKA_ROAD_22", (40, 120),
                               # font, fontScale, color, thickness, cv2.LINE_AA, False)

        cv2.imshow("Accident detection", frame)


        count += 1

        if cv2.waitKey(33) & 0xff == 27:
            break

    else:
        if status==1:

            # instance of MIMEMultipart
            msg = MIMEMultipart()

            # storing the senders email address
            msg['From'] = fromaddr

            # storing the receivers email address
            msg['To'] = toaddr

            # storing the subject
            msg['Subject'] = "Accident detected in jpr clg"

            # string to store the body of the mail
            body = "Accident Detected\n " + "Location " + url_

            # attach the body with the msg instance
            msg.attach(MIMEText(body, 'plain'))

            # open the file to be sent
            filename = "img.jpg"
            attachment = open("img.jpg", "rb")

            # instance of MIMEBase and named as p
            p = MIMEBase('application', 'octet-stream') #to convert into Binary file

            # To change the payload into encoded form
            p.set_payload((attachment).read())

            # encode into base64
            encoders.encode_base64(p)

            p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

            # attach the instance 'p' to instance 'msg'
            msg.attach(p)

            # creates SMTP session
            s = smtplib.SMTP('smtp.gmail.com', 587)

            # start TLS for security
            s.starttls()

            # Authentication
            s.login(fromaddr, "Dob11011993")

            # Converts the Multipart msg into a string
            text = msg.as_string()

            # sending the mail
            s.sendmail(fromaddr, toaddr, text)

            # terminating the session
            s.quit()
            print("Mail Send")
        break

import cv2
import time
from tkinter import *
from tkinter import messagebox

import folium as folium
import geocoder as geocoder
import pygame
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

fromaddr = "info.acdtest@gmail.com"
toaddr = "dummyidnisha0@gmail.com"
from geopy.geocoders import Nominatim
# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
pygame.init()

pygame.mixer.music.load("alarm.wav")
# Capturing Video
cap = cv2.VideoCapture("assets/test_0.mp4")
#cap = cv2.VideoCapture("assets/12.mp4")
#cap = cv2.VideoCapture("assets/1.mp4")
if not cap.isOpened():
    print("Initialising the capture...")
    cap.open("assets/accident.mp4")
    print("Done.")

# Subtracting the background
subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=20)

# Text setting up
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (40, 50)

# fontScale
fontScale = 0.8

# Blue color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

res = 1

arcount = 0

count = -1

while True:
    # Reading the frame
    res, frame = cap.read()

    if res == True:
        # Applying the mask
        mask = subtractor.apply(frame)

    # finding the contours
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Flag for accident detection
        flag = 0

        for cnts in contours:
            (x, y, w, h) = cv2.boundingRect(cnts)

            if w * h > 1000:
                if flag == 1:
                    # If accident detected change color if contours to red
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 0, 255), 3)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 3)

        # If area of rectangle more than a threshold detect accident
            if w * h > 10000:
                area = w * h
            # Countint the number of frames for which the condition persists to refine the accident detection case
                arcount += 1
            # print(arcount)
            if arcount > 35:
                flag = 1

        if flag == 1:
            # If accident detected print Accident on the screen
            frame = cv2.putText(frame, "Accident Detected ", org, font,
                                fontScale, color, thickness, cv2.LINE_AA, False)
            try:
                cv2.imwrite('img.jpg', frame)
                pygame.mixer.music.play()
                g = geocoder.ip('me')
                print(g.latlng)

                location = g.latlng

                map = folium.Map(location=location, zoom_start=10)
                folium.CircleMarker(location=location, radius=50, color="red").add_to(map)
                folium.Marker(location).add_to(map)

                map
                map.save("map.html")
                # instance of MIMEMultipart
                msg = MIMEMultipart()

                # storing the senders email address
                msg['From'] = fromaddr

                # storing the receivers email address
                msg['To'] = toaddr

                # storing the subject
                msg['Subject'] = "Subject of the Mail"

                # string to store the body of the mail
                body = "Body_of_the_mail"

                # attach the body with the msg instance
                msg.attach(MIMEText(body, 'plain'))

                # open the file to be sent
                filename = "map.html"
                attachment = open("map.html", "rb")

                # instance of MIMEBase and named as p
                p = MIMEBase('application', 'octet-stream')

                # To change the payload into encoded form
                p.set_payload((attachment).read())

                # encode into base64
                encoders.encode_base64(p)

                p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

                # attach the instance 'p' to instance 'msg'
                msg.attach(p)
                filename = "img.jpg"
                attachment = open("img.jpg", "rb")

                # instance of MIMEBase and named as p
                p = MIMEBase('application', 'octet-stream')

                # To change the payload into encoded form
                p.set_payload((attachment).read())

                # encode into base64
                encoders.encode_base64(p)

                p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

                # attach the instance 'p' to instance 'msg'
                msg.attach(p)

                # creates SMTP session
                s = smtplib.SMTP('smtp.gmail.com', 587)

                # start TLS for security
                s.starttls()

                # Authentication
                s.login(fromaddr, "Dob11011993")

                # Converts the Multipart msg into a string
                text = msg.as_string()

                # sending the mail
                s.sendmail(fromaddr, toaddr, text)

                # terminating the session
                s.quit()
                #messagebox.showerror("Accident", "Accident Detected")


            except:
                pass
           # frame = cv2.putText(frame, "28° 35' 31.7040'' N and 77° 2' 45.7836'' E. ; ",
                               # (40, 80), font, fontScale, color, thickness, cv2.LINE_AA, False)
           # frame = cv2.putText(frame, "time:1600 hrs; ", (40, 100),
                               # font, fontScale, color, thickness, cv2.LINE_AA, False)
           # frame = cv2.putText(frame, "camera id:DWARKA_ROAD_22", (40, 120),
                               # font, fontScale, color, thickness, cv2.LINE_AA, False)

        cv2.imshow("Accident detection", frame)


        count += 1

        if cv2.waitKey(33) & 0xff == 27:
            break

    else:
        break
