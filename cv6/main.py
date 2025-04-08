import os
import cv2
import numpy as np
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
        if not os.path.exists(path):
            print(f"Warning: Folder {path} does not exist.")
            continue
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) 
            if img is not None:
                img_resized = cv2.resize(img, (64, 64)) 
                images.append(img_resized)
                labels.append(label) 
            else:
                print(f"Warning: Unable to load image {file_path}")
    return np.array(images), np.array(labels)




def extract_lbp_features(images, radius, n_points):
    features = []
    for img in images:
        lbp = local_binary_pattern(img, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= hist.sum() 
        features.append(hist)
    return np.array(features)

# Detekce parkovacích míst na obrázku
def detect_parking_spots(image, spots_coordinates, model, lbp_config):
    predictions = []
    for points in spots_coordinates:
        # Vypočítej obdélník z čtveřice bodů
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x, y = min(x_coords), min(y_coords)  # Levý horní roh
        w, h = max(x_coords) - x, max(y_coords) - y  # Šířka a výška

        # Výřez oblasti parkovacího místa
        spot = image[y:y+h, x:x+w]
        if len(spot.shape) == 3:  # Pokud je obrázek barevný (3D), převedeme na šedou škálu
            spot = cv2.cvtColor(spot, cv2.COLOR_BGR2GRAY)
        spot = cv2.resize(spot, (64, 64))
        lbp = local_binary_pattern(spot, lbp_config["n_points"], lbp_config["radius"], method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_config["n_points"] + 3), range=(0, lbp_config["n_points"] + 2))
        hist = hist.astype("float")
        hist /= hist.sum()
        prediction = model.predict([hist])[0]
        predictions.append(prediction)
    return predictions

# Vizualizace výsledků
def visualize_results(image, spots_coordinates, predictions):
    for points, pred in zip(spots_coordinates, predictions):
        color = (0, 255, 0) if pred == 0 else (0, 0, 255)  # Zelená pro prázdné, červená pro plné
        for i in range(4):
            start_point = points[i]
            end_point = points[(i + 1) % 4]  # Poslední bod se spojí s prvním
            cv2.line(image, start_point, end_point, color, 2)

    # Zobrazení obrázku s výsledky
    cv2.imshow("Parking Lot Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    images, labels = load_training_data("training_data")
    print(f"Loaded {len(images)} training images.")
    print(f"Labels distribution: Free = {np.sum(labels == 0)}, Full = {np.sum(labels == 1)}")




    # Extrakce příznaků
    lbp_config = {"radius": 4, "n_points": 16}  # Konfigurace LBP
    X = extract_lbp_features(images, lbp_config["radius"], lbp_config["n_points"])
    y = labels

    # Trénování modelu
    model = SVC(kernel="linear")
    model.fit(X, y)

    # Načti souřadnice parkovacích míst
    spots_coordinates = load_coordinates("souradky.txt")

    # Zpracuj každý testovací obrázek
    for test_image_name in os.listdir("test_images_zao"):
        # Kontrola, zda má soubor příponu .jpg
        if not test_image_name.lower().endswith(".jpg"):
            continue

        test_image_path = os.path.join("test_images_zao", test_image_name)
        test_image = cv2.imread(test_image_path)

        if test_image is None:
            print(f"Warning: Unable to load image {test_image_path}")
            continue

        # Detekce parkovacích míst
        predictions = detect_parking_spots(test_image, spots_coordinates, model, lbp_config)

        # Vizualizace výsledků
        visualize_results(test_image, spots_coordinates, predictions)

if __name__ == "__main__":
    main()