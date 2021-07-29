from icevision.all import *
from PIL import Image
import time
import argparse
import logging
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig()
log = logging.getLogger()
log.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

IMAGE_SIZE = 384


def model_selection(model_number):
    """
    Select model framework and architecture to load corresponding to model number s
    0: mmdet.retinanet
    1: torchvision.retinanet
    2: ross.efficientnet
    3: yolov5

    :param model_number:
    :return: model type, and architecture
    """

    model_type = None
    backbone = None
    extra_args = {}

    if model_number == 0:
        model_type = models.mmdet.retinanet
        backbone = model_type.backbones.resnet50_fpn_1x

    elif model_number == 1:
        # The Retinanet model is also implemented in the torchvision library
        model_type = models.torchvision.retinanet
        backbone = model_type.backbones.resnet50_fpn

    elif model_number == 2:
        model_type = models.ross.efficientdet
        backbone = model_type.backbones.tf_lite0
        # The efficientdet model requires an img_size parameter
        extra_args['img_size'] = IMAGE_SIZE

    elif model_number == 3:
        model_type = models.ultralytics.yolov5
        backbone = model_type.backbones.small
        # The yolov5 model requires an img_size parameter
        extra_args['img_size'] = IMAGE_SIZE

    # Instantiate the model
    return model_type, model_type.model(backbone=backbone(pretrained=True), num_classes=4, **extra_args)


def load_model(model_number, model_path):
    """
    Load model from provided directory into selected model architecture.

    :param model_number:
        Model numbers correspond to the following architectures:
        0: mmdet.retinanet
        1: torchvision.retinanet
        2: ross.efficientnet
        3: yolov5
    :param model_path: directory of saved model weights
    :return:
    """

    model_type, model = model_selection(model_number)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model_type, model


def convert_img_to_ds(img_pil):
    infer_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=IMAGE_SIZE), tfms.A.Normalize()])
    return Dataset.from_images([img_pil], infer_tfms)


def predict(model_number, model_path, image_path=None, cv_img=None):
    model_type, model = load_model(model_number, model_path)

    img_pil=None

    if image_path:
        img_pil = Image.open(image_path)
    else:
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

    infer_ds = convert_img_to_ds(img_pil)
    preds = model_type.predict(model, infer_ds, keep_images=True)
    for x in preds[0].pred.detection.components:
        if 'ScoresRecordComponent' in str(x):
            scores = x.scores
        if 'InstancesLabelsRecordComponent' in str(x):
            labels = x.label_ids
        if 'BBoxesRecordComponen' in str(x):
            bboxes = x.bboxes

    return labels, scores, bboxes


def predict_from_model(model_type, model, image_path=None, cv_img=None):
    """
    Get label predictions for a single image from a loaded model.
    Need either cv_image or image_path to predict on a single image.
    :param model_type: model architecture
    :param model:
    :param image_path:
    :param cv_img:
    :return: prediction labels, scores and bounding box coordinates
    """

    img_pil=None
    if image_path:
        img_pil = Image.open(image_path)
    else:
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

    infer_ds = convert_img_to_ds(img_pil)
    preds = model_type.predict(model, infer_ds, keep_images=True)

    for x in preds[0].pred.detection.components:
        if 'ScoresRecordComponent' in str(x):
            scores = x.scores
        if 'InstancesLabelsRecordComponent' in str(x):
            labels = x.label_ids
        if 'BBoxesRecordComponen' in str(x):
            bboxes = x.bboxes

    return labels, scores, bboxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='perform inference on single images')
    parser.add_argument('--model-number', type=int,
                        dest='model_number',
                        help='model number corresponding to model architecture')
    parser.add_argument('--model-path', type=str,
                        dest='model_path',
                        help='path to load trained model from')
    parser.add_argument('--image-path',
                        dest='image_path',
                        default='data/test/image1.jpg',
                        help='path to image to perform inference on')

    # Parse arguments
    args = parser.parse_args()
    model_number = args.model_number
    model_path = args.model_path

    # Load specified model
    model_type, model = load_model(model_number, model_path)

    # Load image
    image_path = args.image_path
    img = cv2.imread(image_path)

    start = time.time()

    # Run inference
    labels, scores, bboxes = predict_from_model(model_type, model, cv_img=img)
    log.info(f'Took {time.time()-start:.2f} seconds for inference using model {model_number}')
    log.info(f'\nLabels predicted:\n{labels}')
    log.info(f'\nScores:\n{scores}')
    log.info(f'\nBounding boxes:\n{bboxes}')

