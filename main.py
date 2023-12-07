import cv2
import os
import numpy as np

from ultralytics import YOLO
import supervision as sv

from functions import find_closest_point, get_middle_coords

def main():
    # Video & Model Paths
    video_path = os.path.join('Employee_Detect', 'data', 'sample.mp4')
    video_out_path = os.path.join('Employee_Detect', 'data', 'preds.mp4')
    model_path = os.path.join('Employee_Detect', 'models', 'People_YOLOv8.pt')
    model_2_path = os.path.join('Employee_Detect', 'models', 'Tag_YOLOv8.pt')
    
    # Instantiate VideoCapture to output video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                             (frame.shape[1], frame.shape[0]))
    
    # List to track employee tags, frames and coordinates
    tracker_list, frame_list, coord_list = [], [], []
    frame_count = 0
    
    # Threshold for employee tag detection
    detection_threshold = 0.5

    # YOLO Model Instantiation
    model = YOLO(model_path)
    model_2 = YOLO(model_2_path)
    
    # Track people using Yolov8 model
    for result in model.track(source=video_path, show=False, stream=True, agnostic_nms=False):
        
        # Grab frame
        frame = result.orig_img
        frame_count += 1

        # Detect Employee Tags
        tag_result = model_2(frame)[0]
        tag_coords = []

        # Add each detected Employee Tags to a list
        for r in tag_result.boxes.data.tolist():
            x1 , y1, x2, y2, score, class_id = r
            if score > detection_threshold:
                tag_coords.append([x1, y1, x2, y2])

        # Create tracking detections for People
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        detections = detections[(detections.class_id == 1) & (detections.class_id == 0)]

        # Add People bounding box data to list
        box_list = result.boxes.data.tolist()

        # Check if Employee Tags list is not empty
        if len(tag_coords) != 0:
            people_list, tag_list, tracker_list = [], [], []

            # Iterate to obtain center coords for People bounding box list
            for i in box_list:
                x1, y1, x2, y2, id, conf, class_id = i
                x3, y3 = get_middle_coords(x1, y1, x2, y2)
                people_list.append((x3, y3, id))

            # Iterate to obtain center coords for Employee Tags list
            for j in tag_coords:
                x1, y1, x2, y2 = j
                x3, y3 = get_middle_coords(x1, y1, x2, y2)
                tag_list.append((x3, y3))

            # Iterate to obtain closest bounding box to the employee tag
            for j in tag_list:
                x3, y3 = j
                employee_tuple = [find_closest_point(x3, y3, people_list)]

            # Iterate to annotate the bounding box of the employee
            for i in employee_tuple:
                for j in box_list:
                    if i[2] == j[4]:
                        x1, y1, x2, y2, id, conf, class_id = j
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                        
                        # Append frame & coordinates list
                        frame_list.append(frame_count)
                        x3, y3 = get_middle_coords(x1, y1, x2, y2)
                        coord_list.append((x3, y3))

                        # Append Employee Tag Tracking list
                        tracker_list.append(i[2])

        # Check if Employee Tag Tracking list is not empty
        elif len(tracker_list) != 0:
            for i in tracker_list:
                for j in box_list:
                    if j[4] == i:
                        x1, y1, x2, y2, id, conf, class_id = j
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

                        # Append frame & coordinates list
                        frame_list.append(frame_count)
                        x3, y3 = get_middle_coords(x1, y1, x2, y2)
                        coord_list.append((x3, y3))

        # Display frame
        # cv2.imshow("frame", frame)

        # Write to the video output
        cap_out.write(frame)

        if (cv2.waitKey(30) == 27):
            break
    
    cap_out.release()
    cv2.destroyAllWindows()

    print('Frames: \n', frame_list)
    print('Coordinates: \n', coord_list)

if __name__ == "__main__":
    main()
