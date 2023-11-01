import cv2
import os
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in output_layer_indices]


# Define a function to get predictions using YOLO
def get_predictions(img):
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs, width, height

# Iterate over images
image_folder = "/Users/mario/Desktop/Project/reconstruction-deep-network/data/v1/scans/17DRP5sb8fy/undistorted_color_images"
no_objects_folder = "/Users/mario/Desktop/Project/reconstruction-deep-network/yolo/no obeject folder"
objects_folder = "/Users/mario/Desktop/Project/reconstruction-deep-network/yolo/object folder"

for filename in os.listdir(image_folder):
    path = os.path.join(image_folder, filename)

    # Reading image
    img = cv2.imread(path)
    outs, width, height = get_predictions(img)


    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Extract box dimensions
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Use Non Max Suppression to get best bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if any object is detected
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(objects_folder, filename), img)
    else:
        cv2.imwrite(os.path.join(no_objects_folder, filename), img)
