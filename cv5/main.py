import cv2
import numpy as np

def detect_eye_state(roi_gray):
    _, thresholded = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY)

    total_pixels = roi_gray.shape[0] * roi_gray.shape[1]

    white_pixels = cv2.countNonZero(thresholded)

    if white_pixels > total_pixels * 0.9:  
        return 'open'
    else:
        return 'close'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('eye_cascade_fusek.xml')

with open('eye-state.txt', 'r') as f:
    expected_states = [line.strip() for line in f.readlines()]

video = cv2.VideoCapture('fusek_face_car_01.avi')

frame_index = 0
correct_detections = 0
total_detections = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
   
    detected_state = 'close'

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray,minSize=(20, 20),maxSize=(70, 70)) # 
        
        for (ex, ey, ew, eh) in eyes:

            detected_state = 'open'
            eye_roi_gray = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_roi_color = roi_color[ey:ey + eh, ex:ex + ew]

            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            #detected_state = detect_eye_state(eye_roi_gray)
            #cv2.putText(roi_color, detected_state, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if frame_index < len(expected_states):
            expected_state = expected_states[frame_index]

            if detected_state == expected_state:
                correct_detections += 1
            total_detections += 1

    frame_index += 1

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

if total_detections > 0:
    accuracy = (correct_detections / total_detections) * 100
    print(f'Přesnost detekce: {accuracy:.2f}%')
else:
    print('Nebyla provedena žádná detekce.')