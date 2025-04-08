import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_ground_truth(file_path):
    with open(file_path, "r") as f:
        return [0 if line.strip().lower() == "open" else 1 for line in f.readlines()]

def extract_lbp_features(image, radius, n_points):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def process_video(video_path, eye_cascade_path, ground_truth_file, lbp_config):
    cap = cv2.VideoCapture(video_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    ground_truth = load_ground_truth(ground_truth_file)

    frame_count = 0
    features = []
    labels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in eyes:
            eye = gray[y:y+h, x:x+w]
            eye_resized = cv2.resize(eye, (64, 128))
            hist = extract_lbp_features(eye_resized, lbp_config["radius"], lbp_config["n_points"])

            if frame_count < len(ground_truth):
                features.append(hist)
                labels.append(ground_truth[frame_count])

        frame_count += 1

    cap.release()
    return np.array(features), np.array(labels)

def process_and_visualize_video(video_path, eye_cascade_path, model, lbp_config, ground_truth_file):
    cap = cv2.VideoCapture(video_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    ground_truth = load_ground_truth(ground_truth_file)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        gray = gray[:height // 2, :]  # Pouze horní polovina

        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

         # Pokud jsou detekovány více než dvě oči, vybereme dvě největší
        if len(eyes) > 2:
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]  # Seřadíme podle velikosti a vezmeme 2 největší


        for (x, y, w, h) in eyes:
            eye = gray[y:y+h, x:x+w]
            eye_resized = cv2.resize(eye, (64, 128))
            hist = extract_lbp_features(eye_resized, lbp_config["radius"], lbp_config["n_points"])
            prediction = model.predict([hist])[0]

            # Choose color based on prediction
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)  # Green for open, Red for closed
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display the frame
        cv2.imshow("Eye State Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = "fusek_face_car_01.avi"
    eye_cascade_path = "eye_cascade_fusek.xml"
    ground_truth_file = "eye-state.txt"

    lbp_config = {"radius": 2, "n_points": 16}

    # Extract features and labels from video for training
    print("Extracting features and labels from video...")
    features, labels = process_video(video_path, eye_cascade_path, ground_truth_file, lbp_config)

    # Train SVM model
    print("Training SVM model...")
    model = SVC(kernel="poly")
    model.fit(features, labels)

    # Visualize video with predictions
    print("Visualizing video with predictions...")
    process_and_visualize_video(video_path, eye_cascade_path, model, lbp_config, ground_truth_file)

if __name__ == "__main__":
    main()