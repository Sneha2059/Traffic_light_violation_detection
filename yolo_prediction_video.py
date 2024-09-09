from ultralytics import YOLO
import cv2
import pandas as pd

# Load the YOLOv8 model
model = YOLO('./2.Predictions/weights/best.pt') 

cap = cv2.VideoCapture("./2.Predictions/traffic_video.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("number of frames have finished.")
        break
    else:
        frame = cv2.resize(frame, (1200, 720))
        # Get predictions
        
        results = model(frame)
        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls
            for i, (box, score, cls) in enumerate(zip(boxes, confs, classes)):
                x_min, y_min, x_max, y_max =map(int,box)
                class_id = int(cls)
                class_name = model.names[class_id]  # Human-readable class name

                print(f"Object {i}:")
                print(f"  Coordinates: ({x_min:.2f}, {y_min:.2f}), ({x_max:.2f}, {y_max:.2f})")
                print(f"  Confidence: {score:.2f}")
                print(f"  Class ID: {class_id}, Class Name: {class_name}")
                text=f"{class_name}:{score:.2f}%"

                cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,255,0),2)
                

                cv2.putText(frame,text,(x_min,y_min-10),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)

        
        cv2.imshow('Image with Bounding Boxes', frame)  # Display result
        if cv2.waitKey(1) == 27:
            break
cap.release()
cv2.destroyAllWindows()  # Close all OpenCV windows

