import cv2
import numpy as np
import easyocr


def select_best_probability(array):
    return max(array, key=lambda item: item[2])

def image_to_text(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    return select_best_probability(result)[1]


def isolate_text(image):
    binary_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )#progowanie gaussa
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#szuka kontur calego numeru
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, binary_image.shape[1])
    y_max = min(y_max, binary_image.shape[0])

    cropped_image = binary_image[y_min:y_max, x_min:x_max]

    return cropped_image

def detect_number_from_df(frame, coords_x, coords_y):
    height, width = frame.shape[:2]
    coords_x = [max(0, coords_x[0]), min(height-1, coords_x[1])]

    image = frame[coords_y[0]:coords_y[1], coords_x[0]:]
    cv2.imwrite("number_processing/cropped.png", image)
    image = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite("number_processing/rotated.png", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("number_processing/grayscaled.png", gray)

    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)#binarizacja
    cv2.imwrite("number_processing/thresholded.png", binary_image)

    text_only = isolate_text(binary_image)
    cv2.imwrite("number_processing/only text.png", text_only)
    print("Extracted Text:")
    return image_to_text(text_only)
    
    

if __name__ == "__main__":
    frame = cv2.imread("number_processing/cropped.png")
    str = detect_number_from_df(frame, (0, 1000), (0, 1000))
    print(str)