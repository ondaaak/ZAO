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
            end_point = points[(i + 1) % 4]  # Poslední bod se spojí s prvním
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)  # Zelená čára

    # Zobrazení obrázku s parkovacími místy
    cv2.imshow("Parking Spots", image)
    cv2.waitKey(0)  # Počkej na stisk klávesy
    cv2.destroyAllWindows()  # Zavři všechna okna

# Hlavní funkce
def main():
    # Cesta k souborům+
    test_image_path = "test_images_zao/test1.jpg"  # Nahraď cestou k testovacímu obrázku
    coordinates_file = "souradky.txt"  # Cesta k souboru se souřadnicemi

    # Načti obrázek
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Error: Unable to load image from {test_image_path}")
        return

    # Načti souřadnice parkovacích míst
    spots_coordinates = load_coordinates(coordinates_file)

    # Vizualizuj parkovací místa
    visualize_parking_spots(test_image, spots_coordinates)

if __name__ == "__main__":
    main()