import threading

import cv2
from ml_serving.utils import helpers
import tensorflow as tf

import estimator
import networks
import optic_flow


e: estimator.TfPoseEstimator = None
o: optic_flow.OpticalFlow = optic_flow.OpticalFlow()
PARAMS = {
    'model': 'mobilenet_thin',
    'resize_out_ratio': 4.0,
    'target_size': (432, 368),
    'poses': True,
    'intersection_threshold': 0.33,
}
load_lock = threading.Lock()
loaded = False


def init_hook(**params):
    PARAMS.update(params)
    PARAMS['resize_out_ratio'] = float(PARAMS['resize_out_ratio'])
    PARAMS['intersection_threshold'] = float(PARAMS['intersection_threshold'])
    PARAMS['target_size'] = _parse_resolution(PARAMS['target_size'])
    PARAMS['poses'] = helpers.boolean_string(PARAMS['poses'])
    global e

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if PARAMS['poses']:
        e = estimator.TfPoseEstimator(
            networks.get_graph_path(PARAMS['model']),
            target_size=PARAMS['target_size'],
            tf_config=config,
        )


def process(inputs, ctx, **kwargs):
    global loaded
    if not loaded:
        with load_lock:
            if not loaded:
                global o
                o = optic_flow.OpticalFlow(ctx.drivers[0])
                loaded = True

    image, is_video = helpers.load_image(inputs, 'input')
    if PARAMS['poses']:
        humans = e.inference(
            image,
            resize_to_default=True,
            upsample_size=PARAMS['resize_out_ratio']
        )

    if ctx.drivers[0].driver_name != 'null':
        vectors = o.calc_human_speed(image)

    if PARAMS['poses']:
        image = e.draw_humans(image, humans, imgcopy=True)

    if ctx.drivers[0].driver_name != 'null':
        # __import__('ipdb').set_trace()
        o.draw_vectors(image, vectors)
        o.draw_boxes(image)

    if is_video:
        image_output = image
    else:
        image_output = cv2.imencode(".jpg", image[:, :, ::-1])[1].tostring()

    return {'output': image_output}


def _parse_resolution(s: str):
    splitted = s.split('x')
    if len(splitted) != 2:
        raise RuntimeError(f'Invalid resolution string: {s}')

    return int(splitted[0]), int(splitted[1])
