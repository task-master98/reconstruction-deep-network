import cv2
import os
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in output_layer_indices]

# Load the COCO names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

def get_predictions(img):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs

image_folder = "/Users/mario/Desktop/Project/reconstruction-deep-network/data/v1/scans/17DRP5sb8fy/undistorted_color_images"
no_objects_folder = "/Users/mario/Desktop/Project/reconstruction-deep-network/yolo/testing3/no object"
objects_folder = "/Users/mario/Desktop/Project/reconstruction-deep-network/yolo/testing3/object"

metadata = {}  # Store metadata for each image

for filename in os.listdir(image_folder):
    path = os.path.join(image_folder, filename)
    img = cv2.imread(path)
    outs = get_predictions(img)

    class_ids = []
    confidences = []
    boxes = []
    labels = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = classes[class_id]
                labels.append(label)
                
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                w = int(detection[2] * img.shape[1])
                h = int(detection[3] * img.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Store bounding box and label data
    metadata[filename] = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = labels[i]
            metadata[filename].append({
                "label": label,
                "box": [x, y, w, h]
            })

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imwrite(os.path.join(objects_folder, filename), img)
    else:
        cv2.imwrite(os.path.join(no_objects_folder, filename), img)

# Save metadata to a file
with open("metadata.txt", "w") as f:
    for filename, detections in metadata.items():
        for detection in detections:
            label = detection["label"]
            box = detection["box"]
            f.write(f"{filename},{label},{box[0]},{box[1]},{box[2]},{box[3]}\n")
