import ast
import cv2
import numpy as np
import pandas as pd

# Function to draw borders
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # Top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # Bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # Top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # Bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Load results from CSV for car data with license plate and speed
results = pd.read_csv('./test.csv')

# Load video
video_path = 'sample2.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out_with_speed.mp4', fourcc, fps, (width, height))

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the start

# Process frames and overlay license plate and speed
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # Draw car bounding box
            car_x1, car_y1, car_x2, car_y2 = 100, 200, 300, 400  # Replace with actual bounding box logic if available
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # License plate details
            license_number = df_.iloc[row_indx]['license_number']
            speed_kmh = df_.iloc[row_indx]['speed_kmh']
            
            # Overlay license number and speed
            cv2.putText(frame, f"{license_number} | Speed: {speed_kmh:.2f} km/h",
                        (int(car_x1), int(car_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw license plate bounding box
            x1, y1, x2, y2 = 120, 220, 220, 260  # Replace with actual plate bbox if available
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

        # Write frame to output video
        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # Uncomment to view each frame during processing
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)

# Release resources
out.release()
cap.release()
