import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time 

def load_ground_truth(file_path):
    with open(file_path, "r") as f:
        return [0 if line.strip().lower() == "open" else 1 for line in f.readlines()]

def extract_lbp_features(image, radius, n_points):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def process_video_with_training_and_prediction(video_path, eye_cascade_path, ground_truth_file, lbp_config, split_ratio=0.5):
    cap = cv2.VideoCapture(video_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    ground_truth = load_ground_truth(ground_truth_file)

    frame_count = 0
    features = []
    labels = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    train_frames = int(total_frames * split_ratio) 

    model = None 

    while cap.isOpened():
        start_time = time.time()  

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray[:gray.shape[0] // 2, :] 

        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(eyes) > 2:
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

        if frame_count < train_frames:
            for (x, y, w, h) in eyes:
                eye = gray[y:y+h, x:x+w]
                eye_resized = cv2.resize(eye, (64, 128))
                hist = extract_lbp_features(eye_resized, lbp_config["radius"], lbp_config["n_points"])

                if frame_count < len(ground_truth):
                    features.append(hist)
                    labels.append(ground_truth[frame_count])

            cv2.putText(frame, "Training...", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            if model is None:
                model = SVC(kernel="poly")
                model.fit(features, labels)
            
            for (x, y, w, h) in eyes:
                
                eye = gray[y:y+h, x:x+w]
                eye_resized = cv2.resize(eye, (64, 128))
                hist = extract_lbp_features(eye_resized, lbp_config["radius"], lbp_config["n_points"])
                prediction = model.predict([hist])[0]

                color = (0, 255, 0) if prediction == 0 else (0, 0, 255) 
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
          
            cv2.putText(frame, "Predicting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            processing_time = time.time() - start_time
            cv2.putText(frame, f"Processing time: {processing_time:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Eye State Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    
def main():

    lbp_config = {"radius": 2, "n_points": 16}
    process_video_with_training_and_prediction("fusek_face_car_01.avi", "eye_cascade_fusek.xml", "eye-state.txt", lbp_config)

if __name__ == "__main__":
    main()