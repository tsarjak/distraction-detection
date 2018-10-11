import cv2
from keras.models import load_model
import numpy as np
import imutils
import time

# Loading pre-trained cascading models by CV2
face_cascade = cv2.CascadeClassifier('cv2Files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cv2Files/haarcascade_eye.xml')

# Open a new camera Window
cv2.namedWindow("Keep head straight and pay attention near camera")
camInstance = cv2.VideoCapture(0)

if (camInstance.isOpened() != True):
	print("Unable to open camera, try again.")

eyeCount = 0
totalImageCount = 100

time.sleep(3)
print("Now Go")
while eyeCount < 50:

	time.sleep(0.1)

	# Read the data from Camera-Instance
	check, frame = camInstance.read()

	if check :

		# Resize and convert to gray scale
		frame = imutils.resize(frame, width = 1200)
		grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detect faces in the image
		detectedFaces = face_cascade.detectMultiScale(grayFrame,scaleFactor=1.1,minNeighbors=3,minSize=(240,240),flags=cv2.CASCADE_SCALE_IMAGE)

		if len(detectedFaces) != 0:
			for (x,y,width,height) in detectedFaces:
				roi = grayFrame[y:y+height, x:x+width]

				detectedEyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1,minNeighbors=5,minSize=(60,60))

				for(xEye, yEye, widthEye,heightEye) in detectedEyes:
					
					eyeCount += 1
					totalImageCount += 1
					filename = 'data/payingAttention/'+str(totalImageCount)+'.jpg'
					cv2.imwrite(filename, roi[yEye:yEye+heightEye, xEye:xEye+widthEye])

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

camInstance.release()
cv2.destroyAllWindows()