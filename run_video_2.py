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
    if only_class is not None:
        classes = np.int32((outputs["detection_classes"].copy())).reshape([-1])
        classes = classes[:len(scores)]
        boxes = boxes[classes == only_class]
        scores = scores[classes == only_class]
    boxes[:, 0] *= frame.shape[0] + offset[0]
    boxes[:, 2] *= frame.shape[0] + offset[0]
    boxes[:, 1] *= frame.shape[1] + offset[1]
    boxes[:, 3] *= frame.shape[1] + offset[1]
    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # .astype(int)

    # add probabilities
    confidence = np.expand_dims(scores, axis=0).transpose()
    boxes = np.concatenate((boxes, confidence), axis=1)

    return boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--rotate', type=str, default=None, help='rotate video: cw / ccw')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--modelsDir', type=str, default='./models')
    parser.add_argument('--modelObjectDetection', type=str, default='./model-object-detection-1.0.0-faster-rcnn-resnet101-coco/saved_model')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

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

    cnt = 0

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

        bboxes = _detect_bboxes_tensorflow(d, image)
        bboxes = np.array([bbox[:4].astype("int") for bbox in bboxes])

        humans = e.inference(
            image,
            person_boxes=bboxes,
            upsample_size=4.0,
        )
        if not args.showBG:
            image = np.zeros(image.shape)

        for bbox in bboxes:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 127, 0))

        image = e.draw_humans(image, humans)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
