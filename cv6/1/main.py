import os
import cv2
import numpy as np
import time
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_coordinates(file_path):
    spots_coordinates = []
    with open(file_path, "r") as f:
        for line in f:
            points = eval(line.strip())
            spots_coordinates.append(points)
    return spots_coordinates

def load_training_data(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(["free", "full"]): 
        path = os.path.join(folder, subfolder)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) 
            if img is not None:
                img_resized = cv2.resize(img, (64,128)) 
                images.append(img_resized)
                labels.append(label) 
            else:
                print(f"Warning: Unable to load image {file_path}")
    return np.array(images), np.array(labels)


def calculate_accuracy(predictions, file):
    with open(file, "r") as f:
        file = [int(line.strip()) for line in f.readlines()] 

    if len(predictions) != len(file):
        print(f"Not equal")
        return None

    correct = sum(p == t for p, t in zip(predictions, file))
    accuracy = correct / len(file)
    return accuracy

def extract_lbp_features(images, radius, n_points):
    features = []                                                                
    for img in images:
        lbp = local_binary_pattern(img, n_points, radius, method="uniform")  
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= hist.sum() 
        features.append(hist)
    return np.array(features)

def detect_parking_spots(image, spots_coordinates, model, lbp_config):
    predictions = []
    for points in spots_coordinates:
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x, y = min(x_coords), min(y_coords)
        w, h = max(x_coords) - x, max(y_coords) - y

        spot = image[y:y+h, x:x+w]

        if len(spot.shape) == 3:
            spot = cv2.cvtColor(spot, cv2.COLOR_BGR2GRAY)
        spot = cv2.resize(spot, (64, 128))
        lbp = local_binary_pattern(spot, lbp_config["n_points"], lbp_config["radius"], method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_config["n_points"] + 3), range=(0, lbp_config["n_points"] + 2))
        hist = hist.astype("float")
        hist /= hist.sum()
        prediction = model.predict([hist])[0]
        predictions.append(prediction)
    return predictions

def visualize_results(image, spots_coordinates, predictions):
    for points, pred in zip(spots_coordinates, predictions):
        color = (0, 255, 0) if pred == 0 else (0, 0, 255)  
        for i in range(4):
            start_point = points[i]
            end_point = points[(i + 1) % 4] 
            cv2.line(image, start_point, end_point, color, 2)

    resized_image = cv2.resize(image, (1200, 800))
    cv2.imshow("Parking Lot Detection", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    test_images_folder = "test_images_zao"
    spots_coordinates = load_coordinates("souradky.txt")

    images, labels = load_training_data("training_data")

    lbp_config = {"radius": 2, "n_points": 16} 
    X = extract_lbp_features(images, lbp_config["radius"], lbp_config["n_points"])
    y = labels


    model = SVC(kernel="poly")
    model.fit(X, y)

    total_accuracy = []
    for test_image_name in os.listdir(test_images_folder):

        if not test_image_name.lower().endswith(".jpg"):
            continue

        test_image_path = os.path.join(test_images_folder, test_image_name)
        test_image = cv2.imread(test_image_path)

        if test_image is None:
            print(f"Unable to load image {test_image_path}")
            continue

        correct_file = os.path.join(test_images_folder, test_image_name.replace(".jpg", ".txt"))
        if not os.path.exists(correct_file):
            print(f"File {correct_file} not found.")
            continue
        
        start_time = time.time()

        predictions = detect_parking_spots(test_image, spots_coordinates, model, lbp_config)

        accuracy = calculate_accuracy(predictions, correct_file)
        if accuracy is not None:
            print(f"Accuracy for {test_image_name}: {accuracy:.2f}")
            total_accuracy.append(accuracy)

        processing_time = time.time() - start_time
        print(f"Processing time for {test_image_name}: {processing_time:.3f} seconds") 

        visualize_results(test_image, spots_coordinates, predictions)

    if total_accuracy:
        overall_accuracy = sum(total_accuracy) / len(total_accuracy)
        print(f"\nOverall Accuracy: {overall_accuracy:.2f}")


if __name__ == "__main__":
    main()