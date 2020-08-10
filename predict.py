import cv2
import os
import numpy as np
import tensorflow as tf

model_for_hand = tf.keras.models.load_model('model_hand.h5')
model_for_me = tf.keras.models.load_model('model_face.h5')

pic_taken = False

text = "Show V  to click a pic "

def detectSign(roi_img):
	roi_img = np.expand_dims(roi_img, axis=0)
	prediction = model_for_hand.predict(roi_img)
	prediction = np.argmax(prediction)
	return prediction
		
cap = cv2.VideoCapture(0)

box_size = 234
width = int(cap.get(3))

while True:
	ret, frame = cap.read()

	frame = cv2.flip(frame, 1)

	cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 250, 0), 2)

	roi = frame[5: box_size-5 , width-box_size + 5: width -5]
	roi_img = np.array(roi)

	prediction = detectSign(roi_img)

	if prediction == 1:
		p = "No gesture"
		cv2.putText(frame, p, (5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
	if prediction == 0:
		t = "Showing gesture"
		cv2.putText(frame, t, (5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
		# cap.release()
		# cv2.destroyAllWindows()
		if not pic_taken:
			cv2.imwrite("pic.png", frame)
			pic_taken = True

	if pic_taken:
		frameImage = cv2.imread("pic.png")
		frameImage = cv2.cvtColor(frameImage, cv2.COLOR_BGR2RGB)
		frameImage = cv2.resize(frameImage, (224, 224))
		frameImage = np.array(frameImage)
		frameImage = np.expand_dims(frameImage, axis=0)

		predicted_frame = model_for_me.predict(frameImage)
		predicted_frame = np.argmax(predicted_frame)
		
		if predicted_frame == 0:
			text = "Hello User"
		elif predicted_frame == 1:
			text = "None"

	cv2.putText(frame, text, (3, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
	cv2.imwrite("pic.png", frame)
			

	k = cv2.waitKey(1)

	if k == ord('q'):
		break


	cv2.imshow("Gesture", frame)


cap.release()
cv2.destroyAllWindows()