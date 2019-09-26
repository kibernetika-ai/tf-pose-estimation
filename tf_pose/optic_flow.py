import time

import cv2
import numpy as np


class OpticalFlow(object):
    def __init__(self, person_detect_driver=None, intersection_threshold=0.33):
        self.person_driver = person_detect_driver
        self.num_frames = 5
        self.expire_sec = 10
        self.intersection_threshold = intersection_threshold
        self.human_boxes = None
        self.vectors = []

    def calc_human_speed(self, frame, one_person=False):
        if self.human_boxes is None:
            self.human_boxes = self.detect_persons(frame, one_person=one_person)
            return self.human_boxes, []

        # Compare human_boxes <-> new_boxes
        new_boxes = self.detect_persons(frame, one_person=one_person)
        new_vectors = []
        for b0 in self.human_boxes:
            for b1 in new_boxes:
                if self._box_intersection(b0, b1) > self.intersection_threshold:
                    # Get center vector
                    new_vectors.append(self.center_vector(b0, b1))

        self.human_boxes = new_boxes

        if not self.vectors:
            self.vectors = new_vectors
            return self.human_boxes, self.vectors

        for v0 in self.vectors:
            for v1 in new_vectors:
                if v1.distance(v0) <= v1.len + 5:
                    v0.update(v1)

        # Expire cycle
        i = 0
        while i < len(self.vectors):
            if time.time() - self.vectors[i].updated_at >= self.expire_sec:
                # print('expire vector {},{}'.format(self.vectors[i].x1, self.vectors[i].y1))
                del self.vectors[i]
            else:
                i += 1

        return self.human_boxes, self.vectors

    def draw_boxes(self, frame):
        for box in self.human_boxes:
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),  # (left, top), (right, bottom)
                (0, 0, 0),
                thickness=3,
            )
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),  # (left, top), (right, bottom)
                (220, 220, 0),
                thickness=2,
            )

    @staticmethod
    def draw_vectors(frame, vectors):
        coef = 7.
        if not vectors:
            return

        for v in vectors:
            avg = v.avg_per_frame()
            # v: x, y, len, angle
            x1 = avg.x0 + avg.len * np.cos(avg.angle) * coef
            y1 = avg.y0 - avg.len * np.sin(avg.angle) * coef + avg.overlay_y
            cv2.arrowedLine(
                frame,
                (int(avg.x0), int(avg.y0) + avg.overlay_y),
                (int(x1), int(y1)),
                (0, 0, 0),
                thickness=5,
                line_type=cv2.LINE_AA,
                tipLength=0.4,
            )
            cv2.arrowedLine(
                frame,
                (int(avg.x0), int(avg.y0) + avg.overlay_y),
                (int(x1), int(y1)),
                (250, 0, 0),
                thickness=3,
                line_type=cv2.LINE_AA,
                tipLength=0.4,
            )

    def detect_persons(self, frame, threshold=0.5, one_person=False):
        if self.person_driver is None:
            return None
        elif self.person_driver.driver_name == "openvino":
            boxes = self._detect_openvino(frame, threshold)
        elif self.person_driver.driver_name == "tensorflow":
            boxes = self._detect_tensorflow(frame, threshold)
        else:
            return None

        if not one_person:
            return boxes

        res_box = np.zeros([1, 5], dtype=boxes.dtype)
        squares = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        res_box[0] = boxes[np.argmax(squares)]
        return res_box

    def _detect_tensorflow(self, frame, threshold=0.5):
        input_name, input_shape = list(self.person_driver.inputs.items())[0]
        inference_frame = np.expand_dims(frame, axis=0)
        outputs = self.person_driver.predict({input_name: inference_frame})
        boxes = outputs["detection_boxes"].copy().reshape([-1, 4])
        scores = outputs["detection_scores"].copy().reshape([-1])
        classes = np.int32((outputs["detection_classes"].copy())).reshape([-1])
        scores = scores[np.where(scores > threshold)]
        boxes = boxes[:len(scores)]
        classes = classes[:len(scores)]
        boxes = boxes[classes == 1]
        scores = scores[classes == 1]
        boxes[:, 0] *= frame.shape[0]
        boxes[:, 1] *= frame.shape[1]
        boxes[:, 2] *= frame.shape[0]
        boxes[:, 3] *= frame.shape[1]
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]].astype(int)

        confidence = np.expand_dims(scores, axis=0).transpose()
        boxes = np.concatenate((boxes, confidence), axis=1)

        return boxes

    def _detect_openvino(self, frame, threshold=0.5, offset=(0, 0)):
        if self.person_driver is None:
            return None
        # Get boxes shaped [N, 5]:
        # xmin, ymin, xmax, ymax, confidence
        input_name, input_shape = list(self.person_driver.inputs.items())[0]
        output_name = list(self.person_driver.outputs)[0]
        inference_frame = cv2.resize(frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
        inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
        outputs = self.person_driver.predict({input_name: inference_frame})
        output = outputs[output_name]
        output = output.reshape(-1, 7)
        bboxes_raw = output[output[:, 2] > threshold]
        # Extract 5 values
        boxes = bboxes_raw[:, 3:7]
        confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
        boxes = np.concatenate((boxes, confidence), axis=1)
        # Assign confidence to 4th
        # boxes[:, 4] = bboxes_raw[:, 2]
        boxes[:, 0] = boxes[:, 0] * frame.shape[1] + offset[0]
        boxes[:, 2] = boxes[:, 2] * frame.shape[1] + offset[0]
        boxes[:, 1] = boxes[:, 1] * frame.shape[0] + offset[1]
        boxes[:, 3] = boxes[:, 3] * frame.shape[0] + offset[1]
        return boxes

    @staticmethod
    def _box_intersection(box_a, box_b):
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_area = max(0, xb - xa) * max(0, yb - ya)

        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        d = float(box_a_area + box_b_area - inter_area)
        if d == 0:
            return 0
        iou = inter_area / d
        return iou

    def center_vector(self, box_a, box_b):
        center_xa = (box_a[2] + box_a[0]) / 2
        center_xb = (box_b[2] + box_b[0]) / 2
        center_ya = (box_a[3] + box_a[1]) / 2
        center_yb = (box_b[3] + box_b[1]) / 2

        length = np.sqrt(np.square(center_xb - center_xa) + np.square(center_yb - center_ya))
        v = np.array([center_xb - center_xa, center_yb - center_ya])
        angle = np.arctan2(*v.T[::-1])

        return Vector(
            center_xa, center_ya, center_xb, center_yb, length, angle,
            max_frames=self.num_frames,
            overlay_y=int(-(box_a[3] - box_a[1]) * 0.3)
        )

    @staticmethod
    def _unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def _angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class Vector(object):
    def __init__(self, x0=0, y0=0, x1=0, y1=0, len=0., angle=0., overlay_y=0, max_frames=5):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.vectors = np.array([[x1 - x0, y1 - y0]])
        self.overlay_y = overlay_y
        self.len = len
        self.angle = angle
        self.frames = 1
        self.max_frames = max_frames
        self.updated_at = time.time()

    def avg_per_frame(self):
        mean = np.mean(self.vectors, axis=0)
        length = np.sqrt(np.square(mean[0]) + np.square(mean[1]))
        v = np.array([mean[0], mean[1]])
        angle = np.arctan2(*v.T[::-1])
        return Vector(
            self.x1,
            self.y1,
            mean[0],
            mean[1],
            length,
            angle,
            overlay_y=self.overlay_y,
        )

    def distance(self, v: 'Vector'):
        return np.sqrt(np.square(self.x1 - v.x1) + np.square(self.y1 - v.y1))

    def update(self, v: 'Vector'):
        self.x0 = v.x0
        self.y0 = v.y0
        self.x1 = v.x1
        self.y1 = v.y1
        self.vectors = np.concatenate(
            (self.vectors, np.array([[self.x1 - self.x0, self.y1 - self.y0]]))
        )
        # self.len.extend(v.len)
        # self.angle.extend(v.angle)
        self.frames += 1
        self.updated_at = time.time()

        if len(self.vectors) >= self.max_frames:
            # del self.len[0]
            # del self.angle[0]
            self.vectors = np.delete(self.vectors, [0], axis=0)
            self.frames -= 1

    def drawing_coords(self, coef=5.):
        avg = self.avg_per_frame()
        # v: x, y, len, angle
        x1 = avg.x0 + avg.len * np.cos(avg.angle) * coef
        y1 = avg.y0 - avg.len * np.sin(avg.angle) * coef + avg.overlay_y
        return (int(avg.x0), int(avg.y0)), (int(x1), int(y1))

    def __repr__(self):
        return (
            f'<Vector ({self.x0},{self.y0}) -> ({self.x1}, {self.y1}) '
            f'len={self.len} angle={self.angle} '
            f'frames={self.frames}>'
        )


def center(p0, p1, to_int=True):
    x = (p0[0] + p1[0]) / 2
    y = (p0[1] + p1[1]) / 2
    if to_int:
        return int(x), int(y)
    return x, y
