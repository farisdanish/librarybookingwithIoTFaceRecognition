import os
import json
from PIL import Image
from keras.models import load_model
from keras_facenet import FaceNet
import numpy as np
from numpy import asarray
from numpy import expand_dims
import pickle
import cv2
import numpy as np
import datetime
import time
import requests
import RPi.GPIO as GPIO
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#remember to change ngrok url every time 
url = "'https://fcb0-27-125-249-39.ngrok-free.app/api/" #url for accessing the custom RestAPIs by exposing the localhost to ngrok and using the provided links given by ngrok
url2 = "https://fcb0-27-125-249-39.ngrok-free.app/api/"

#get records from database
#Student List
GetStudListURL = url2 + "api/studentlist"
#Staff List
GetStaffListURL = url2 + "api/stafflist"
#Room Bookings List
GetRBookListURL = url2 + "api/rbooklists"
#Rooms List
GetRoomListURL = url2 + "api/roomlist"

print('-- GETTING DATA -- ')
try:
	#Make a GET request
	studResponse = requests.get(GetStudListURL)
	staffResponse = requests.get(GetStaffListURL)
	rBookResponse = requests.get(GetRBookListURL)
	roomResponse = requests.get(GetRoomListURL)
	
	#Check if successful request (Status Code 200)
	if studResponse.status_code == 200:
		stud_data = studResponse.json()
		print(f"Student Data Retrieved!")
	else:
		print(f"Failed to get data. Status Code: {studResponse.status_code}")
	
	#Check if successful request (Status Code 200)
	if staffResponse.status_code == 200:
		staff_data = staffResponse.json()
		print(f"Staff Data Retrieved!")
	else:
		print(f"Failed to get data. Status Code: {staffResponse.status_code}")
	
	#Check if successful request (Status Code 200)
	if rBookResponse.status_code == 200:
		rbook_data = rBookResponse.json()
		print(f"Room Booking Data Retrieved!")
	else:
		print(f"Failed to get data. Status Code: {rBookResponse.status_code}")
	
	#Check if successful request (Status Code 200)
	if roomResponse.status_code == 200:
		room_data = roomResponse.json()
		print(f"Room List Data Retrieved!")
		print('-- ALL DATA RETRIEVED -- ')
	else:
		print(f"Failed to get data. Status Code: {roomResponse.status_code}")
	
except Exception as e:
	print(f"An Error Occurred: {e}")
#Getting data from database code ends

RELAY = 17
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY, GPIO.OUT)
GPIO.output(RELAY,GPIO.LOW)

prevTime = 0
doorUnlock = False

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

def get_face(gbr1):
	wajah = HaarCascade.detectMultiScale(gbr1,1.1,4)
	
	if wajah is ():
		face = None
		x1, y1 = 0, 0
		x2, y2 = 0, 0
	else:
		if len(wajah)>0:
			x1, y1, width, height = wajah[0]
		else:
			x1, y1, width, height = 1, 1, 10, 10
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		#gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
		gbr = gbr1
		gbr = Image.fromarray(gbr)                  # konversi dari OpenCV ke PIL
		gbr_array = asarray(gbr)
		face = gbr_array[y1:y2, x1:x2]
	return face, x1, x2, y1, y2
		

def detect_facenet(expectedIdentity, curr_room):
	global prevTime #Use global variable instead of local
	global doorUnlock
	
	# load faces
	data = np.load('registered-faces-db.npz')
	testX_faces = data['arr_2']
	# load face embeddings
	data = np.load('registered-faces-db-embeddings.npz')
	trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	testX = in_encoder.transform(testX)
	# label encode targets
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	trainy = out_encoder.transform(trainy)
	testy = out_encoder.transform(testy)

	# fit model
	model = SGDClassifier(loss='log_loss')
	# model = HistGradientBoostingClassifier()
	model.fit(trainX, trainy)

	detected = False
	count=0
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) #compressed video format for video processing
	#print (cap.read())
	
	while(True):
		_, gbr1 = cap.read()

		
		face, x1, x2, y1, y2 = get_face(gbr1)

		if type(face) is np.ndarray:
			face = Image.fromarray(face)
			face = face.resize((160,160))
			face = asarray(face)

			face = expand_dims(face, axis=0)
			signature = MyFaceNet.embeddings(face)
	
			#predict face
			samples = expand_dims(signature, axis=0)
			nsamples, nx, ny = samples.shape
			samples = samples.reshape((nsamples,nx*ny))
			yhat_class = model.predict(samples)
			yhat_prob = model.predict_proba(samples)
			# get name
			class_index = yhat_class[0]
			class_probability = yhat_prob[0,class_index] * 100
			predict_names = out_encoder.inverse_transform(yhat_class)
			identity = predict_names[0]
	
			if class_probability > 70 and predict_names[0] == expectedIdentity:
				print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
				cv2.putText(gbr1,identity, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2, cv2.LINE_AA)
				#cv2.putText(gbr1,str(min_dist*100), (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
				cv2.rectangle(gbr1,(x1,y1),(x2,y2), (0,255,0), 2)
				if expectedIdentity == identity:
					detected = True
					count = count + 1
			else:
				print('(Might be Wrong face)Predicted: %s (%.3f)' % (predict_names[0], class_probability))
				#print(f"unexpected face detected")
				#cv2.putText(gbr1,identity, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
				#cv2.putText(gbr1,str(min_dist*100), (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
				cv2.rectangle(gbr1,(x1,y1),(x2,y2), (0,255,0), 2)
		else:
			cv2.putText(gbr1,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
		
		cv2.imshow('LiveVideoRecognizer',gbr1)
		if detected==True and count>2:
			print('Door is unlocked')
			#Function to send mail to detected user
			RoomID = curr_room
			#Check if expectedIdentity is either a Student or Staff
			for i in stud_data:
				if i['StudID'] == expectedIdentity:
					StudID = expectedIdentity
					StaffID = None
			for i in staff_data:
				if i['StaffID'] == expectedIdentity:
					StaffID = expectedIdentity
					StudID = None
			status = 0 #passed recognition
			curr_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
			data={
				'RoomID': RoomID,
				'StudID': StudID,
				'StaffID': StaffID,
				'Status': status,
				'Timestamp': curr_dt
			}
			headers = {'Content-Type':'application/json'}
			mail_url = url2 + "api/accesslogs"
			try:
				response = requests.post(mail_url,json=data,headers=headers)
				if response.status_code == 200 or response.status_code == 201:
					print(f"Access Log updated Successfully")
				else:
					print(f"Failed to update access log, Status Code: {response.status_code}")
			except Exception as e:
				print(f"Error in sending mail: {e}")
			
			# to unlock the door
			GPIO.output(RELAY,GPIO.HIGH)
			prevTime = time.time()
			doorUnlock = True
			print("door unlock")
			break
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	cv2.destroyAllWindows()
	cap.release()

try:
	print(f"\n App Starts. Press Ctrl+C to end app process.")
	for i in room_data:
		if i['RoomType'] == "Normal Room":
			print(str(i['RoomID']) + ". " + i['RoomName'] + "\n")
			
	curr_room = int(input("Please enter room number: "))
	
	expectedIdentity = ""
	while True:
		print(f"Checking room booking records..")
		curr_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
		for i in rbook_data:
			if i['RoomID'] == curr_room:
				if curr_time >= i['Start'] and curr_time < i['End']: 
					if i['StudID'] != None:
						expectedIdentity = i['StudID']
						break
					if i['StaffID'] != None:
						expectedIdentity = i['StaffID']
						break
		if expectedIdentity == "":
			print('No Room Bookings at this time.')
			time.sleep(30)#Wait 30 seconds
			continue
		else:
			print(f"Current Room is Booked for: " + expectedIdentity)
			print(f"Finding Face For Current Room Booking")
			detect_facenet(expectedIdentity, curr_room)
			print(f"Room is occupied...")
			time.sleep(5)#Wait 5 seconds
			print(prevTime)
			print(doorUnlock)
			if doorUnlock == True and time.time() - prevTime > 5:
				doorUnlock = False
				GPIO.output(RELAY,GPIO.LOW)
				print("Room Locked!")
except KeyboardInterrupt:
	GPIO.cleanup()
	print(f"\nApp Ends. Goodbye.")
