from icevision.all import *
import PIL
from PIL import Image
import time

model_dir = Path("models/")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def model_selection(model_number):
    image_size = 384
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
        extra_args['img_size'] = image_size

    elif model_number == 3:
        model_type = models.ultralytics.yolov5
        backbone = model_type.backbones.small
        # The yolov5 model requires an img_size parameter
        extra_args['img_size'] = image_size
        # TODO: as per below, this is hacky, should be one return
        return model_type, model_type.model(backbone=backbone(pretrained=True), num_classes=3, **extra_args)

    # Instantiate the model
    # !TODO need to figure out how to get the len of the classmap from the data dict
    return model_type, model_type.model(backbone=backbone(pretrained=True), num_classes=4, **extra_args)

#
# def load_model(model_number):
#     model_type, model = model_selection(model_number)
#     model_name = f"nikemodel_model_{model_number}_new_100.mm"
#     model.load_state_dict(torch.load(model_dir/f"{model_name}", map_location=torch.device('cpu')))
#     return model_type, model

def load_model(model_number):
    # TOOD: hack
    if model_number == 1:
        model_name = f"nikemodel_model_{model_number}_new_100.mm"
        model_type, model = model_selection(model_number)
        model.load_state_dict(torch.load(model_dir/f"{model_name}", map_location=torch.device('cpu')))
    else:
        model_name = f"model_3_unfreeze_20.mm" #  "model_3_unfreeze_20.mm" #"model_3_unfreeze_20.mm"
        model_type, model = model_selection(model_number)
        model.load_state_dict(torch.load(f"infer/{model_name}", map_location=torch.device('cpu')))

    return model_type, model


def convert_img_to_ds(img_pil):
    infer_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=384), tfms.A.Normalize()])
    return Dataset.from_images([img_pil], infer_tfms)


def predict(model_number, image_path=None, cv_img=None):
    model_type, model = load_model(model_number)

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


image_path = "data/test/image1.jpg"
img = cv2.imread(image_path)

model_number = 1

if __name__ == "__main__":
    model_type, model = load_model(1)

    start = time.time()
    labels, scores, bboxes = predict_from_model(model_type, model, cv_img=img)
    print(f'Took {time.time()-start:.2f} seconds to predict cv2 image')

    start = time.time()
    labels, scores, bboxes = predict_from_model(model_type, model, cv_img=img)
    print(f'Took {time.time()-start:.2f} seconds to predict and load image from path')

