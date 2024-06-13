import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load the trained YOLO model for emotion recognition
model = YOLO("/home/anavale/emotion (copy)/yolo_m3.pt")

# Class names for the emotions
class_names = ['disgust', 'neutral', 'sadness', 'surprise']

# Load the OpenCV face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

def color_map(color_name):
    return {
        'Rojo': (0, 0, 255),
        'Verde': (0, 255, 0),
        'Azul': (255, 0, 0),
        'Amarillo': (0, 255, 255),
        'Naranja': (0, 165, 255),
        'Morado': (128, 0, 128)
    }.get(color_name, (255, 255, 255))

def create_red_mask(hsv_image):
    lower_red = np.array([0, 100, 100], np.uint8)
    upper_red = np.array([10, 255, 255], np.uint8)
    return cv2.inRange(hsv_image, lower_red, upper_red)

def create_yellow_mask(hsv_image):
    lower_yellow = np.array([20, 100, 100], np.uint8)
    upper_yellow = np.array([30, 255, 255], np.uint8)
    return cv2.inRange(hsv_image, lower_yellow, upper_yellow)

def create_orange_mask(hsv_image):
    lower_orange = np.array([11, 100, 100], np.uint8)
    upper_orange = np.array([19, 255, 255], np.uint8)
    return cv2.inRange(hsv_image, lower_orange, upper_orange)

def create_blue_mask(hsv_image):
    lower_blue = np.array([100, 150, 0], np.uint8)
    upper_blue = np.array([140, 255, 255], np.uint8)
    return cv2.inRange(hsv_image, lower_blue, upper_blue)

def create_purple_mask(hsv_image):
    lower_purple = np.array([125, 50, 50], np.uint8)
    upper_purple = np.array([145, 255, 255], np.uint8)
    return cv2.inRange(hsv_image, lower_purple, upper_purple)

def create_green_mask(hsv_image):
    lower_green = np.array([50, 100, 100], np.uint8)
    upper_green = np.array([70, 255, 255], np.uint8)
    return cv2.inRange(hsv_image, lower_green, upper_green)

def detect_and_label(frame, hsv_image):
    roles = {
        'PCA': [('Azul', create_blue_mask), ('Morado', create_purple_mask)],
        'PRC': [('Rojo', create_red_mask), ('Amarillo', create_yellow_mask), ('Naranja', create_orange_mask)],
        'PRM': [('Verde', create_green_mask)]
    }
    
    for role, colors in roles.items():
        for color_name, mask_function in colors:
            mask = mask_function(hsv_image)
            percentage = highlight_color(mask, frame, color_map(color_name))
            if percentage > 0.2:
                cv2.putText(frame, role, (10, get_position(role)), cv2.FONT_HERSHEY_SIMPLEX, 1, color_map(color_name), 2, cv2.LINE_AA)
                return role, color_name
    return None, None

def get_position(role):
    return {
        'PCA': 100,
        'PRC': 50,
        'PRM': 150
    }[role]

def highlight_color(mask, img, color):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    img[np.where(mask != 0)] = color
    return np.sum(mask) / (mask.shape[0] * mask.shape[1])

def apply_protocol(role):
    if role == 'PCA':
        pulsate('Azul', 'rápidas', 30)
    elif role == 'PRC':
        pulsate('Rojo', 'rápidas', 30)
    elif role == 'PRM':
        pulsate('Verde', 'rápidas', 30)

def pulsate(color_name, speed, duration):
    color = color_map(color_name)
    circle_frame = np.zeros((500, 500, 3), dtype=np.uint8)
    center = (circle_frame.shape[1] // 2, circle_frame.shape[0] // 2)
    radius = 100
    end_time = time.time() + duration
    interval = 0.1 if speed == 'rápidas' else 0.5 if speed == 'medias' else 1.0
    while time.time() < end_time:
        cv2.circle(circle_frame, center, radius, color, -1)
        cv2.imshow('Pulsations', circle_frame)
        if cv2.waitKey(int(interval * 1000)) & 0xFF == ord('q'):
            break
        cv2.circle(circle_frame, center, radius, (0, 0, 0), -1)
        cv2.imshow('Pulsations', circle_frame)
        if cv2.waitKey(int(interval * 1000)) & 0xFF == ord('q'):
            break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_half = hsv_image[frame.shape[0]//2:frame.shape[0], :]
    role_detected, color_detected = detect_and_label(frame, lower_half)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        results = model(face_roi)

        best_score = 0
        best_label = ""
        best_box = None
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    score = float(box.conf[0])
                    if score > best_score:
                        best_score = score
                        best_box = box
                        best_label = class_names[class_id]

        if best_box:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            label_size, _ = cv2.getTextSize(best_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            label_y = y + y1 - label_size[1] - 10 if y + y1 - label_size[1] - 10 > 10 else y + y1 + label_size[1] + 10
            cv2.rectangle(frame, (x + x1, label_y - label_size[1] - 10), (x + x1 + label_size[0], label_y + 10), (0, 0, 255), -1)
            cv2.putText(frame, best_label, (x + x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.rectangle(frame, (x + x1, y + y1), (x + x2, y + y2), (0, 255, 0), 2)

    if role_detected:
        apply_protocol(role_detected)

    cv2.imshow('Combined Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
