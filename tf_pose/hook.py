import cv2
from ml_serving.utils import helpers

import estimator
import networks


e: estimator.TfPoseEstimator = None
PARAMS = {
    'model': 'mobilenet_thin',
    'resize_out_ratio': 4.0,
    'target_size': (432, 368)
}


def init_hook(**params):
    PARAMS.update(params)
    PARAMS['resize_out_ratio'] = float(PARAMS['resize_out_ratio'])
    PARAMS['target_size'] = _parse_resolution(PARAMS['target_size'])
    global e

    e = estimator.TfPoseEstimator(
        networks.get_graph_path(PARAMS['model']), target_size=PARAMS['target_size']
    )


def process(inputs, ctx, **kwargs):
    image, is_video = helpers.load_image(inputs, 'input')
    humans = e.inference(
        image,
        resize_to_default=True,
        upsample_size=PARAMS['resize_out_ratio']
    )

    image = e.draw_humans(image, humans, imgcopy=True)

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
