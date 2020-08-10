import cv2
import os

cap = cv2.VideoCapture(0)

scissor = False
box_size = 234
width = int(cap.get(3))
counter = 100

while True:
	ret, frame = cap.read()

	frame = cv2.flip(frame, 1)

	cv2.rectangle(frame, (150,100), (500,500), (0, 250, 0), 2)

	# roi = frame[5: box_size-5 , width-box_size + 5: width -5]
	roi = frame[150:400, 100:500]

	text = "Press m to take own pics or press b to take blank pics"

	cv2.putText(frame, text, (3, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

	k = cv2.waitKey(1)

	if k == ord('m'):
		for i in range(counter):
			cv2.imwrite("my_data/" + str(i) + ".png", roi)
	if k == ord('b'):
		for i in range(counter):
			cv2.imwrite("blank_data/" + str(i) + ".png", roi)
	if k == ord('q'):
		break


	cv2.imshow("Gesture", frame)


cap.release()
cv2.destroyAllWindows()
