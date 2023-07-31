import cv2
import numpy as np
from tracker import EuclideanDistTracker

# Initialize Tracker
tracker = EuclideanDistTracker()

# Detection confidence threshold
confThreshold = 0.1
nmsThreshold = 0.2

# Middle cross line position
middle_line_position = 225
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))

# Filter classes for car, truck, and bus
filtered_classes = ['car', 'truck', 'bus']

# Get the indices of filtered classes
filtered_class_indices = [classNames.index(cls) for cls in filtered_classes]

## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Configure the network backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

# Initialize the videocapture object
cap = cv2.VideoCapture('video.mp4')

medium_size = (800, 600)

detection = []
temp_up_list = []
temp_down_list = []
vehicle_count = {class_name: {"Up": 0, "Down": 0} for class_name in filtered_classes}

def postProcess(outputs, img):
    ih, iw, _ = img.shape

    boxes = []
    classIds = []
    confidence_scores = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold and classId in filtered_class_indices:
                w, h = int(det[2] * iw), int(det[3] * ih)
                x, y = int((det[0] * iw) - w / 2), int((det[1] * ih) - h / 2)
                boxes.append([x, y, w, h])
                classIds.append(classId)
                confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)

    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]

        # Draw classname and confidence score
        cv2.putText(img, f'{name.upper()} {int(confidence_scores[i]*100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

        # Store the detection information
        detection.append([x, y, w, h, classIds[i]])

def count_vehicle(box_id):
    x, y, w, h, id = box_id

    # Find the center of the rectangle for detection
    center = (x + w // 2, y + h // 2)

    # Find the current position of the vehicle and update the count
    if (center[1] > up_line_position) and (center[1] < middle_line_position):
        if id not in temp_up_list:
            temp_up_list.append(id)

    elif center[1] < down_line_position and center[1] > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif center[1] < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            vehicle_count[classNames[id]]["Down"] += 1

    elif center[1] > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            vehicle_count[classNames[id]]["Up"] += 1

def realTime():
    while True:
        success, img = cap.read()
        if not success:  # If the frame is not read successfully, break the loop
            break

        img = cv2.resize(img, medium_size)
        ih, iw, channels = img.shape

        # Clear the detection list at the beginning of each frame
        detection.clear()

        # Set the input of the network
        input_size = 320
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        # Get names of output layers
        out_layer_names = net.getUnconnectedOutLayersNames()

        # Forward pass through the network to get the output layers
        outputs = net.forward(out_layer_names)

        # Find the objects from the network output
        postProcess(outputs, img)

        for box_id in detection:
            count_vehicle(box_id)

        # Draw the crossing lines
        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 1)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 1)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 1)

        # Display vehicle count information for car, truck, and bus
        font_size = 0.7
        font_color = (0, 255, 0)
        font_thickness = 1
        y_offset = 20
        for class_name in filtered_classes:
            cv2.putText(img, f"{class_name.upper()} - Up: {vehicle_count[class_name]['Up']}, Down: {vehicle_count[class_name]['Down']}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            y_offset += 25

        # Display the frames
        cv2.imshow('Output', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' key is pressed
            break

    # Release the VideoCapture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
 realTime()