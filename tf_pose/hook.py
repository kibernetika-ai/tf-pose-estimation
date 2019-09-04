import cv2
from ml_serving.utils import helpers

import estimator
import networks


e: estimator.TfPoseEstimator = None
PARAMS = {
    'model': 'mobilenet_thin',
    'resize_out_ratio': 4.0,
}


def init_hook(**params):
    global e

    e = estimator.TfPoseEstimator(
        networks.get_graph_path(PARAMS['model']), target_size=(432, 368)
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
