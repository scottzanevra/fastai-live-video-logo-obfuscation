from icevision.all import *
import PIL


model_dir = Path("/Users/scott.zanevra/PycharmProjects/fastai-live-video-logo-obfuscation/models/")

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

    # Instantiate the model
    # !TODO need to figure out how to get the len of the classmap from the data dict
    return model_type, model_type.model(backbone=backbone(pretrained=True), num_classes=4, **extra_args)


def load_model(model_number):
    model_type, model = model_selection(model_number)
    model_name = f"nikemodel_model_{model_number}_new_100.mm"
    model.load_state_dict(torch.load(model_dir/f"{model_name}", map_location=torch.device('cpu')))
    return model_type, model


def convert_img_to_ds(image_path):
    img = PIL.Image.open(image_path)
    infer_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=384), tfms.A.Normalize()])
    return Dataset.from_images([img], infer_tfms)


def predict(model_number, image_path):
    model_type, model = load_model(model_number)
    infer_ds = convert_img_to_ds(image_path)
    preds = model_type.predict(model, infer_ds, keep_images=True)
    for x in preds[0].pred.detection.components:
        print(x)
        if 'ScoresRecordComponent' in str(x):
            scores = x.scores
            print(scores)
        if 'InstancesLabelsRecordComponent' in str(x):
            labels = x.label_ids
            print(labels)
        if 'BBoxesRecordComponen' in str(x):
            bboxes = x.bboxes
            print(bboxes)

    show_preds(preds=preds[0:1])


img_path = "/Users/scott.zanevra/PycharmProjects/fastai-live-video-logo-obfuscation/data/test/image1.jpg"
model_number = 1
if __name__ is "__main__":
    predict(model_number, img_path)

