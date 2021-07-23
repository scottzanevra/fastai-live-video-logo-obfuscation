import base64
import json
import logging
import os
import socket
import time

import boto3
import cv2
import numpy as np

from datetime import datetime
from pathlib import Path
from src.inference import predict_from_model, load_model

logging.basicConfig()
log = logging.getLogger()
log.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

REGION = os.environ.get('REGION', 'ap-southeast-2')
LAMBDA = boto3.client('lambda', region_name=REGION)
CAMERA_ID = socket.gethostname()

# List of valid resolutions
RESOLUTION = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (858, 480), 'training': (300, 300)}

# Text labels
CLASS_MAP = {
    1: 'nike',
    2: 'swoosh',
    3: 'human'
}

# Overlap percentage threshold for logo boxes
LOGO_OVERLAP_THRESHOLD = 80


def overlap_area(bbox1, bbox2):
    """
    Calculates overlap area between two bounding boxes.
    :param bbox1: (logo bounding box)
    :param bbox2: (human bounding box)
    :return: Overlap area, 0 if no overlap
    """
    dx = min(bbox1.xmax, bbox2.xmax) - max(bbox1.xmin, bbox2.xmin)
    dy = min(bbox1.ymax, bbox2.ymax) - max(bbox1.ymin, bbox2.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx*dy
    return 0


def overlap_percent(bbox1, bbox2):
    """
    Gets percentage of area of bbox1 which overlaps with bbox2
    :param bbox1: (logo bounding box)
    :param bbox2: (human bounding box)
    :return:
    """
    a_area = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin)
    return overlap_area(bbox1, bbox2) / a_area * 100


def annotate_frame(frame, labels, scores, bboxes):
    """
    Annotates a frame with scaled bounding boxes and obfuscated regions
    :param frame: frame to annotate
    :param labels: predicted labels
    :param scores: confidence scores of predictions
    :param bboxes: bounding boxes defining labelled regions
    :return: annotated frame
    """

    # Get a list of all human bounding boxes
    human_bboxes = []
    for i, label in enumerate(labels):
        if CLASS_MAP[label] == 'human':
            human_bboxes.append(bboxes[i])

    # For each label, annotate and obfuscate if logo
    for i, label in enumerate(labels):
        if CLASS_MAP[label] == 'swoosh' or CLASS_MAP[label] == 'nike':
            bbox = bboxes[i]
            bbox_min, bbox_max = scale_bbox_dims(frame, bbox)

            # Check overlap with human_bboxes
            # Get only the max overlaps with human bboxes
            # i.e. if one logo appears in two human bounding boxes,
            # then just take the max overlap %
            max_percent = 0
            if len(human_bboxes) > 0:
                max_percent = max([overlap_percent(bbox, hbbox) for hbbox in human_bboxes])

            if max_percent > LOGO_OVERLAP_THRESHOLD:
                # On human, blur out logo via masking
                blurred_frame = cv2.blur(frame, (25, 25), cv2.BORDER_DEFAULT)
                mask = np.zeros(frame.shape, dtype=np.uint8)
                mask = cv2.rectangle(mask, bbox_min, bbox_max, (255,255,255), -1)
                frame = np.where(mask != np.array([255, 255, 255]), frame, blurred_frame)
            else:
                # Not on human, but let's draw a rectangle to detect logo anyways
                 cv2.rectangle(frame, bbox_min, bbox_max, (255,255,255), 2)

            cv2.putText(
                frame, f'{CLASS_MAP[label]} (overlap {max_percent}%)',
                (bbox_min[0], bbox_min[1]-5), #
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if CLASS_MAP[label] == 'human':
            bbox = bboxes[i]
            bbox_min, bbox_max = scale_bbox_dims(frame, bbox)

            cv2.putText(
                frame, CLASS_MAP[label], (bbox_min[0], bbox_min[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            cv2.rectangle(frame, bbox_min, bbox_max, (255,0,0), 1)

    return frame


def detect_logo(frame):
    # This is this function will do the inference.
    # We require the annotations to be provided back in the response

    # Resize frame to 1/2 for faster processing
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Encode to JPG and send to lambda
    ret, encoded = cv2.imencode('.jpg', small)
    if not ret:
        raise RuntimeError('Failed to encode frame')

    # !TODO Replace this with a POST to API Gateway
    response = LAMBDA.invoke(
        FunctionName=f"fast-ai-object-detection-lambda",
        InvocationType='RequestResponse',
        Payload=json.dumps({
            'camera_ID': CAMERA_ID,
            'img': base64.b64encode(encoded).decode('utf-8')
        }))

    # Annotate bounding boxes to frame
    response_dict = json.loads(response['Payload'].read())
    if 'FunctionError' not in response:
        annotate_frame(frame, response_dict)
    else:
        print(response_dict['errorMessage'])


def convert_to_jpg(frame, resolution):
    """ Converts the captured frame to the desired resolution
    """
    ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, resolution))
    if not ret:
        raise Exception('Failed to set frame data')
    return jpeg


def save_jpeg_to_temp(jpeg, destination, picture_name):
    file_name = f"{destination}/{picture_name}.jpeg"
    with open(file_name, 'wb') as f:
        f.write(jpeg)


def scale_bbox_dims(img, bbox, size=384):
    """
    Rescale and translate label bounding boxes from their inference transformed state.
    :param img: cv2 image object
    :param bbox:
    :param size: size image was scaled to during predict transformation
    :return: scaled bboxes to fit img
    """

    # Get actual height and widths of webcam frame
    w = img.shape[1]
    h = img.shape[0]

    # Images are padded then resized in transforms before predict.
    # Since width > height always for webcam frames we just need height padding (ypad)
    xf = w / size
    ypad = (size - (h/xf))/2

    bbox_min = (int(bbox.xmin*xf), int((bbox.ymin-ypad)*xf))
    bbox_max = (int(bbox.xmax*xf), int((bbox.ymax-ypad)*xf))

    return bbox_min, bbox_max


def lambda_handler(event, context):

    cap = cv2.VideoCapture(0)
    time.sleep(1)  # just to avoid that initial black frame

    frame_skip = 60
    frame_count = 0

    winname = 'Press ESC or Q to quit'
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 50, 50)

    # Create tmp dir for this round of webcam frames
    dir_time = datetime.now()
    frame_dir = Path(f'tmp/{dir_time.strftime("camera-frames-%y-%m-%d_%H-%M-%S")}')
    os.makedirs(frame_dir, exist_ok=True)

    # Load model for predictions
    model_type, model = load_model(1)

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError('Failed to capture frame')

        if frame_count % frame_skip == 0:  # only analyze every n frames

            # Inference time
            labels, scores, bboxes = predict_from_model(model_type, model, cv_img=frame)
            frame = annotate_frame(frame, labels, scores, bboxes)

            cv2.imshow(winname, frame)

            # Save images based on timestamp
            frame_time = datetime.now()
            frame_name = f'{frame_time.strftime("frame-%y-%m-%d_%H-%M-%S%f")}'
            frame_path = Path(frame_dir / frame_name)

            jpeg = convert_to_jpg(frame, RESOLUTION['training'])
            save_jpeg_to_temp(jpeg, frame_dir, frame_name)
            log.info(f"Saving picture {frame_name} to {frame_dir}")

        frame_count += 1

        # Press ESC or 'q' to quit
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lambda_handler(None, None)
