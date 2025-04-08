import os
import cv2
import numpy as np


def load_coordinates(file_path):
    spots_coordinates = []
    with open(file_path, "r") as f:
        for line in f:
            points = eval(line.strip())
            spots_coordinates.append(points)
    return spots_coordinates

def visualize_parking_spots(image, spots_coordinates):
    for points in spots_coordinates:
        for i in range(4):
            start_point = points[i]
            end_point = points[(i + 1) % 4] 
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    # Zobrazení obrázku s parkovacími místy
    cv2.imshow("Parking Spots", image)
    cv2.waitKey(0)  # Počkej na stisk klávesy
    cv2.destroyAllWindows()  # Zavři všechna okna


def load_training_data(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(["free", "full"]):  # 0 = free, 1 = full
        path = os.path.join(folder, subfolder)
        if not os.path.exists(path):
            print(f"Warning: Folder {path} does not exist.")
            continue
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Načti obrázek v odstínech šedi
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))  # Změna velikosti na 64x64
                images.append(img_resized)
                labels.append(label)  # 0 pro free, 1 pro full
            else:
                print(f"Warning: Unable to load image {file_path}")
    return np.array(images), np.array(labels)



# Hlavní funkce
def main():
    # Cesta k souborům+
    test_image_path = "test_images_zao/test2.jpg"  # Nahraď cestou k testovacímu obrázku
    coordinates_file = "souradky.txt"  # Cesta k souboru se souřadnicemi

    # Načti obrázek
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Error: Unable to load image from {test_image_path}")
        return

    # Načti souřadnice parkovacích míst
    spots_coordinates = load_coordinates(coordinates_file)

    images, labels = load_training_data("")
    print(f"Loaded {len(images)} images.")
    print(f"Labels distribution: Free = {np.sum(labels == 0)}, Full = {np.sum(labels == 1)}")

    # Vizualizuj parkovací místa
    visualize_parking_spots(test_image, spots_coordinates)

if __name__ == "__main__":
    main()