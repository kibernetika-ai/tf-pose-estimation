import argparse
import logging
import time
import tensorflow as tf

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from ml_serving.drivers import driver

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


def _detect_bboxes_tensorflow(drv: driver.ServingDriver, frame: np.ndarray,
                              threshold: float = 0.5, offset=(0, 0), only_class=None):
    input_name, input_shape = list(drv.inputs.items())[0]
    inference_frame = np.expand_dims(frame, axis=0)
    outputs = drv.predict({input_name: inference_frame})
    boxes = outputs["detection_boxes"].copy().reshape([-1, 4])
    scores = outputs["detection_scores"].copy().reshape([-1])
    scores = scores[np.where(scores > threshold)]
    boxes = boxes[:len(scores)]
    classes = np.int32((outputs["detection_classes"].copy())).reshape([-1])
    classes = classes[:len(scores)]
    if only_class is not None:
        boxes = boxes[classes == only_class]
        scores = scores[classes == only_class]
    boxes[:, 0] *= frame.shape[0]
    boxes[:, 2] *= frame.shape[0]
    boxes[:, 1] *= frame.shape[1]
    boxes[:, 3] *= frame.shape[1]
    boxes[:, 0] += offset[1]
    boxes[:, 2] += offset[1]
    boxes[:, 1] += offset[0]
    boxes[:, 3] += offset[0]
    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # .astype(int)

    # add probabilities and classes
    confidence = np.expand_dims(scores, axis=0).transpose()
    classes = np.expand_dims(classes, axis=0).transpose()
    boxes = np.concatenate((boxes, confidence, classes), axis=1)

    return boxes


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


def _detect_bboxes(drv: driver.ServingDriver, frame: np.ndarray, threshold: float = .5, split_counts=None):
    boxes = _detect_bboxes_tensorflow(drv, frame, threshold=threshold)

    if split_counts:
        def add_box(b):
            for i, b0 in enumerate(boxes):
                if _box_intersection(b0, b) > 0.3:
                    # set the largest proba to existing box
                    boxes[i][4] = max(b0[4], b[4])
                    return
            boxes.resize((boxes.shape[0] + 1, boxes.shape[1]), refcheck=False)
            boxes[-1] = b

        for split_count in split_counts:
            size_multiplier = 2. / (split_count + 1)
            xstep = int(frame.shape[1] / (split_count + 1))
            ystep = int(frame.shape[0] / (split_count + 1))

            xlimit = int(np.ceil(frame.shape[1] * (1 - size_multiplier)))
            ylimit = int(np.ceil(frame.shape[0] * (1 - size_multiplier)))
            for x in range(0, xlimit, xstep):
                for y in range(0, ylimit, ystep):
                    y_border = min(frame.shape[0], int(np.ceil(y + frame.shape[0] * size_multiplier)))
                    x_border = min(frame.shape[1], int(np.ceil(x + frame.shape[1] * size_multiplier)))
                    crop = frame[y:y_border, x:x_border, :]

                    box_candidates = _detect_bboxes_tensorflow(drv, crop, threshold, (x, y))

                    for b in box_candidates:
                        add_box(b)

    return boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, required=True, help='Input video')
    parser.add_argument('--output', type=str, default=None, help='Output video')
    parser.add_argument('--screen', action='store_true', help='Show result on the screen')
    parser.add_argument('--drawBBoxes', action='store_true', help='Draw object detected boxes')
    parser.add_argument('--multiDetect', type=str, default=None, help='Comma separated multi-detect splits')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--rotate', type=str, default=None, help='rotate video: cw / ccw')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--modelsDir', type=str, default='./models')
    parser.add_argument('--modelObjectDetection', type=str, default='./model-object-detection-1.0.0-faster-rcnn-resnet101-coco/saved_model')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    split_counts = None
    if args.multiDetect:
        split_counts = list(map(int, args.multiDetect.split(",")))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.rotate == "cw" or args.rotate == "ccw":
        width, height = height, width
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # int(cap.get(cv2.CAP_PROP_FOURCC))
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, frameSize=(width, height))

    graph_path = get_graph_path(args.model, models_dir=args.modelsDir)
    logger.debug('initialization %s : %s' % (args.model, graph_path))
    w, h = model_wh(args.resolution)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    e = TfPoseEstimator(
        graph_path,
        target_size=(w, h),
        tf_config=config,
    )

    drv = driver.load_driver("tensorflow")
    d = drv()
    d.load_model(args.modelObjectDetection)

    cnt = 0
    font_scale = (width + height) / 2 / 2000

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    while cap.isOpened():
        ret_val, image = cap.read()
        if image is None:
            break

        if args.rotate == "cw":
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == "ccw":
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        bboxes = _detect_bboxes(d, image, split_counts=split_counts)

        humans = e.inference(
            image,
            person_boxes=bboxes,
            upsample_size=4.,
        )
        if not args.showBG:
            image = np.zeros(image.shape)

        if args.drawBBoxes:
            for bbox in bboxes:
                tl, br = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(image, tl, br, (0, 255, 0))
                cv2.putText(image, "{}".format(int(bbox[5])), tl, cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 255, 0), 2)

        image = e.draw_humans(image, humans)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, int(height / 100)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

        if args.screen:
            cv2.imshow('tf-pose-estimation result', image)
            if cv2.waitKey(1) == 27:
                break

        if video_writer:
            video_writer.write(image)
        fps_time = time.time()

        cnt += 1
        logger.debug('processed {} frames'.format(cnt))

    if args.screen:
        cv2.destroyAllWindows()
    if video_writer:
        video_writer.release()

logger.debug('finished+')
