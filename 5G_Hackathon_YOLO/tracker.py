import math

import numpy as np

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, detections):
        updated_boxes_ids = []
        for detection in detections:
            x, y, w, h, classId, index = detection
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)

            # If the classId and index are not already in the center_points dictionary, add them
            if (classId, index) not in self.center_points:
                self.center_points[(classId, index)] = (cx, cy, self.id_count)
                self.id_count += 1

            # Calculate the Euclidean distance between the current detection and all previous detections
            distances = [np.linalg.norm(np.array([cx, cy]) - np.array(point[:2])) for point in self.center_points.values()]

            # Find the closest previous detection (minimum distance)
            closest_id = np.argmin(distances)
            min_distance = distances[closest_id]

            # If the minimum distance is below a threshold, consider it as the same object and update its position
            if min_distance < 50:
                updated_boxes_ids.append((*detection, self.center_points[list(self.center_points.keys())[closest_id]][2]))
                self.center_points[list(self.center_points.keys())[closest_id]] = (cx, cy, self.center_points[list(self.center_points.keys())[closest_id]][2])
            else:
                # If the minimum distance is above the threshold, consider it as a new object and assign a new ID
                updated_boxes_ids.append((*detection, self.id_count))
                self.center_points[(classId, index)] = (cx, cy, self.id_count)
                self.id_count += 1

        return updated_boxes_ids
