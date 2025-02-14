import cv2
import math
from ultralytics import YOLO
from process_text_image import detect_number_from_df
import database_functions as db
from database_functions import Car


def find_closest(array_of_classes, target_tuple):
    def euclidean_distance(t1, t2):
        return math.sqrt((t1[0] - t2[0])**2 + (t1[1] - t2[1])**2)
    
    closest_instance = None
    smallest_distance = float('inf')
    
    for instance in array_of_classes:
        distance = euclidean_distance(instance.prev_pos, target_tuple)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_instance = instance
    
    return smallest_distance, closest_instance

def normalize_rectangle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    top_left = (min(x1, x2), min(y1, y2))
    bottom_right = (max(x1, x2), max(y1, y2))
    
    return top_left, bottom_right

def rectangle_contains(rect1, rect2):
    (tl1, br1) = normalize_rectangle(*rect1) 
    (tl2, br2) = normalize_rectangle(*rect2)  

    return (tl1[0] <= tl2[0] and tl1[1] <= tl2[1] and  
            br1[0] >= br2[0] and br1[1] >= br2[1])    

model = YOLO("best_car_detection_project.pt")  

parking_slots = [
    (322, 390, 425, 570),   
    (13, 446, 194, 552),    
    (14, 337, 195, 437),    
    (14, 228, 197, 330),    
    (14, 119, 197, 221),   
    (14, 8, 197, 110),      
    (325, 5, 425, 180),     
]

entry_zone = (647, 167, 843, 285)
exit_zone = (647, 290, 843, 410)

video_path = 'assets/video.mp4'  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

car_in_bounds = False
numtext = None
allowed_to_enter = None
parked_cars = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot read frame.")
        break

    results = model(frame)
    car_ids = set()  
    count = 1 
    car = None

    for slot in parking_slots:
        x1, y1, x2, y2 = slot
        slot_label_offset = (47, 40) if count in [1, 7] else (20, 55)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
        cv2.putText(
            frame, f"{count}", 
            (x1 + slot_label_offset[0], y1 + slot_label_offset[1]), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        count += 1

    for zone, label, color in [(entry_zone, "Entry", (255, 0, 0)), (exit_zone, "Exit", (0, 0, 255))]:
        x1, y1, x2, y2 = zone
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, 
            (x1 + 10, y1 + 55), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    for result in results:
        cars = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  
            confidence = box.conf[0] * 100  
            class_id = int(box.cls[0])  

            if class_id != 0:  
                continue

            car_ids.add(f"{x1},{y1},{x2},{y2}")

            car_center_x = (x1 + x2) // 2
            car_center_y = (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if not (entry_zone[0] < car_center_x < entry_zone[2] and entry_zone[1] < car_center_y < entry_zone[3] or exit_zone[0] < car_center_x < exit_zone[2] and exit_zone[1] < car_center_y < exit_zone[3]):
                _, car = find_closest(parked_cars, (car_center_x, car_center_y))
                car.prev_pos = (car_center_x, car_center_y)
                cv2.putText(frame, car.license_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if (numtext):
                if (not allowed_to_enter):
                    xx1, yy1, xx2, yy2 = entry_zone
                    cv2.line(frame, (xx1, yy1), (xx1, yy2), (0, 0, 255), 4)
                    if (entry_zone[0] < x1):
                        cv2.putText(frame, "Access denied", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    if (entry_zone[0] < x1):
                        cv2.putText(frame, numtext, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            if entry_zone[0] < car_center_x < entry_zone[2] and entry_zone[1] < car_center_y < entry_zone[3]:
                cars.append(True)
                if (car_center_x > 700 and car_center_x < 779 and not car_in_bounds):
                    car_in_bounds = True
                    text = detect_number_from_df(frame, (x1, x2), (y1, y2))
                    allowed_to_enter = db.write_number_to_db(text)
                    print("car in bounds!")
                    numtext = text
                    parked_cars.append(Car(text, False, 0, (car_center_x, car_center_y)))
                    
                cv2.putText(
                    frame, "Entering", 
                    (car_center_x, car_center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )
            else:
                
                cars.append(False)


            if exit_zone[0] < car_center_x < exit_zone[2] and exit_zone[1] < car_center_y < exit_zone[3]:
                d, car = find_closest(parked_cars, (car_center_x, car_center_y))
                if (d < 100):
                    car_in_bounds = True
                    db.exit_parking(car)
                    parked_cars.remove(car)
                cv2.putText(
                    frame, "Exiting", 
                    (car_center_x, car_center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )

            for idx, slot in enumerate(parking_slots, 1):
                sx1, sy1, sx2, sy2 = slot
                if sx1 < car_center_x < sx2 and sy1 < car_center_y < sy2:
                    if (rectangle_contains(((sx1, sy1), (sx2, sy2)), ((x1, y1), (x2, y2)))):
                        cv2.putText(
                            frame, f"Parked in Slot {idx}", 
                            (car_center_x - 20, car_center_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )
                        if (car.is_parked_legally != True):
                            car.is_parked_legally = True
                            db.park(car)
                    else:
                        cv2.putText(
                            frame, f"Parked improperly in Slot {idx}", 
                            (car_center_x - 20, car_center_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )
                        if (car.is_parked_legally == None):
                            car.parking_place = idx
                            car.is_parked_legally = False
                            db.park(car)
                else:
                    if (car and car.is_parked_legally != None and car.parking_place == idx):
                        car.parking_place = idx
                        car.is_parked_legally = None
                        db.unpark(car)
        if (car_in_bounds):
            car_in_bounds = False
            keep_numtext = False
            for i in cars:
                if (i == True):
                    car_in_bounds = True
                    keep_numtext = True
            if (not keep_numtext):
                numtext = None


    cv2.imshow("Parking Slot Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
