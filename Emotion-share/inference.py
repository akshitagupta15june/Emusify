import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import random
import os
import time
newq=random.randint(0,5)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
seg=SelfiSegmentation()
#song list importing
#happy
happymusic_dir='C:\\Users\\AKSHITA GUPTA\\Desktop\\songs\\happy'
happysong = os.listdir(happymusic_dir)
#sad
sadmusic_dir='C:\\Users\\AKSHITA GUPTA\\Desktop\\songs\\sad'
sadsong = os.listdir(sadmusic_dir)
#angry
angrymusic_dir='C:\\Users\\AKSHITA GUPTA\\Desktop\\songs\\angry'
angrysong = os.listdir(angrymusic_dir)
#neutral
neutralmusic_dir='C:\\Users\\AKSHITA GUPTA\\Desktop\\songs\\neutral'
neutralsong = os.listdir(neutralmusic_dir)
#rock
rockmusic_dir='C:\\Users\\AKSHITA GUPTA\\Desktop\\songs\\rock'
rocksong = os.listdir(rockmusic_dir)
#surprise
surprisemusic_dir='C:\\Users\\AKSHITA GUPTA\\Desktop\\songs\\surprise'
surprisesong = os.listdir(surprisemusic_dir)




model  = load_model("model.h5")
label = np.load("labels.npy")



holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
h=0
s=0
a=0
n=0
r=0
sup=0
t=20
p=0

maxemotion=0

while True:

	lst = []

	_, frm = cap.read()

	frm = cv2.flip(frm, 1)

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))


	if res.face_landmarks:
		for i in res.face_landmarks.landmark:
			lst.append(i.x - res.face_landmarks.landmark[1].x)
			lst.append(i.y - res.face_landmarks.landmark[1].y)

		if res.left_hand_landmarks:
			for i in res.left_hand_landmarks.landmark:
				lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		if res.right_hand_landmarks:
			for i in res.right_hand_landmarks.landmark:
				lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		lst = np.array(lst).reshape(1,-1)

		pred = label[np.argmax(model.predict(lst))]
		p=p+1

		if pred =='happy':
			h=h+1
		if pred =='sad':
			s=s+1
		if pred =='angry':
			a=a+1
		if pred =='neutral':
			n=n+1
		if pred =='rock':
			r=r+1
		if pred =='surprise':
			sup=sup+1


		emotion = [h, s, n, a, r,sup]
		print(pred)
		print(p)
		print(emotion)

		if p==120:
			maxemotion=max(emotion)
			if (maxemotion == h):
				print("It seems you are happy, Lets play some happy music!")
				os.startfile(os.path.join(happymusic_dir, happysong[newq]))
			elif (maxemotion == a):
				print("It seems you are angry, Let me calm your mood!")
				os.startfile(os.path.join(angrymusic_dir, angrysong[newq]))
			elif (maxemotion == r):
				print("It seems you are rocking, Let's play some rock music!")
				os.startfile(os.path.join(rockmusic_dir, rocksong[newq]))
			elif (maxemotion == s):
				print("It seems you are sad, Let me cheer you up!")
				os.startfile(os.path.join(sadmusic_dir, sadsong[newq]))
			elif (maxemotion == n):
				print("It seems you are in a calm mood, Let me play some calm music for you!")
				os.startfile(os.path.join(neutralmusic_dir, neutralsong[newq]))
			elif (maxemotion == sup):
				print("It seems you are suprised, let me surprise you again with some cool songs!")
				os.startfile(os.path.join(surprisemusic_dir, surprisesong[newq]))

		elif(p>1500 and p<=2000):
			maxemotion = max(emotion)
			if (maxemotion == h):
				print("It seems you are happy, Lets play some happy music!")
				os.startfile(os.path.join(happymusic_dir, happysong[newq]))
			elif (maxemotion == a):
				print("It seems you are angry, Let me calm your mood!")
				os.startfile(os.path.join(angrymusic_dir, angrysong[newq]))
			elif (maxemotion == r):
				print("It seems you are rocking, Let's play some rock music!")
				os.startfile(os.path.join(rockmusic_dir, rocksong[newq]))
			elif (maxemotion == s):
				print("It seems you sad, Let me cheer you up!")
				os.startfile(os.path.join(sadmusic_dir, sadsong[newq]))
			elif (maxemotion == n):
				print("It seems you are in a calm mood, Let me play some calm music for you!")
				os.startfile(os.path.join(neutralmusic_dir, neutralsong[newq]))
			elif(maxemotion==sup):
				print("It seems you are suprised, let me surprise you again with some cool songs!")
				os.startfile(os.path.join(surprisemusic_dir, surprisesong[newq]))
			break


		cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)


		
	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)



	vid_rmbg = seg.removeBG(frm, (255, 229, 180), threshold=0.7)
	imgstack = cvzone.stackImages([frm, vid_rmbg], 2, 1)
	cv2.imshow("window", imgstack)

	if cv2.waitKey(1) & 0xff == ord('q'):
		cv2.destroyAllWindows()
		cap.release()
		break

