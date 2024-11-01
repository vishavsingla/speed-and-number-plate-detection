# from ultralytics import YOLO
# import cv2

# import util
# from sort.sort import *
# from util import get_car, read_license_plate, write_csv


# results = {}

# mot_tracker = Sort()

# # load models
# coco_model = YOLO('yolov8n.pt')
# license_plate_detector = YOLO('license_plate_detector.pt')

# # load video
# cap = cv2.VideoCapture('./sample2.mp4')

# vehicles = [2, 3, 5, 7]

# # read frames
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#         # detect vehicles
#         detections = coco_model(frame)[0]
#         detections_ = []
#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             if int(class_id) in vehicles:
#                 detections_.append([x1, y1, x2, y2, score])

#         # track vehicles
#         track_ids = mot_tracker.update(np.asarray(detections_))

#         # detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate

#             # assign license plate to car
#             xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

#             if car_id != -1:

#                 # crop license plate
#                 license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

#                 # process license plate
#                 license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#                 _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

#                 # read license plate number
#                 license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

#                 if license_plate_text is not None:
#                     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                                   'license_plate': {'bbox': [x1, y1, x2, y2],
#                                                                     'text': license_plate_text,
#                                                                     'bbox_score': score,
#                                                                     'text_score': license_plate_text_score}}

# # write results
# write_csv(results, './test.csv')

from ultralytics import YOLO
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video

vehicles = [2, 3, 5, 7]  # Vehicle classes (cars, buses, etc.)

# Speed calculation parameters
previous_positions = {}
pixels_per_meter = 10  # Set this based on your calibration

def calculate_speed(prev_pos, curr_pos, fps):
    """
    Calculate the speed of a vehicle based on its previous and current positions.
    """
    distance = np.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2) / pixels_per_meter
    speed = (distance * fps) * 3.6  # Convert to km/h
    return speed

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Calculate speed if previous position exists
                if car_id in previous_positions:
                    prev_pos = previous_positions[car_id]
                    curr_pos = ((xcar1 + xcar2) / 2, (ycar1 + ycar2) / 2)
                    speed = calculate_speed(prev_pos, curr_pos, fps)
                else:
                    speed = None
                    curr_pos = ((xcar1 + xcar2) / 2, (ycar1 + ycar2) / 2)
                
                # Update the previous position for the car
                previous_positions[car_id] = curr_pos

                # Draw car bounding box and display speed
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                if speed is not None:
                    cv2.putText(frame, f"Speed: {int(speed)} km/h", (int(xcar1), int(ycar1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # Draw license plate bounding box and display text
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                if license_plate_text is not None:
                    cv2.putText(frame, f"Plate: {license_plate_text}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        },
                        'speed_kmh': speed
                    }

        # Display the frame with annotations
        cv2.imshow("Live Camera Feed", frame)

        # Press 'q' to exit the live feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

# Write results to CSV
write_csv(results, './test.csv')
