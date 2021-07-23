import base64
import json
import logging
import os
import socket
import time
import uuid

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

CLASS_MAP = {
    1: 'nike',
    2: 'swoosh',
    3: 'human'
}

def overlap_percent(bbox1, bbox2):
    # calculates the overlap ratio given 1 rectangle with corners l1 and r2 and
    # the other with corners l2 and r2
    l1 = bbox1[0]
    r1 = bbox1[1]
    l2 = bbox2[0]
    r2 = bbox2[1]
    x = 0
    y = 1

    x_dist = (min(r1[x], r2[x]) -
              max(l1[x], l2[x]))
    y_dist = (min(r1[y], r2[y]) -
              max(l1[y], l2[y]))
    areaI = 0
    if x_dist > 0 and y_dist > 0:
        areaI = x_dist * y_dist

    area1 = abs( (l1[0]-r1[0])*(l1[1]-r1[1]) )
    overlap = areaI / area1

    return overlap


def select_bbox(labels, scores, bboxes):
    person_label = 3 # label of person
    logo_labels = [1,2] # list of logo lables
    overlap_threshold = 0.5 # ratio of overlap for obfiscating
    bboxes_out = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == person_label:
                if labels[j] in logo_labels:
                    bbox1 = bboxes[j] # logo
                    bbox2 = bboxes[i] # person
                    overlap = overlap_percent(bbox1, bbox2)
                    if overlap >= overlap_threshold:
                        bboxes_out.append(bbox1)
    return bboxes_out

def show_predict(img, labels, scores, bboxes):
    img2 = img.copy()
    draw2 = ImageDraw.Draw(img2)
    width, height = img2.size
    w_ratio = width / 380
    h_ratio = height / 380
    for bbox, label in zip(bboxes,labels):
        bbox1 = [ tuple([bbox[0][0]*w_ratio,bbox[0][1]*h_ratio]), tuple([bbox[1][0]*w_ratio,bbox[1][1]*h_ratio]) ]
        if label == 3:
            color ="red"
        else:
            color = "yellow"
        draw2.rectangle(bbox1, outline =color)

    img2.show()


def obfuscate(img,bboxes_remove):
    # removes all the rectangles in bbox_out list
    img2 = img.copy()
    draw2 = ImageDraw.Draw(img2)
    width, height = img2.size
    w_ratio = width / 380
    h_ratio = height / 380
    for bbox in bboxes_remove:
        bbox1 = [ tuple([bbox[0][0]*w_ratio,bbox[0][1]*h_ratio]), tuple([bbox[1][0]*w_ratio,bbox[1][1]*h_ratio]) ]
        color = "yellow"
        draw2.rectangle(bbox1, outline =color, fill=color)

    return img2


# !Todo - does fastai have a function to do this
def resize_bbox(img, bbox):
    return {
        'Left': int(bbox['Left'] * img.shape[1]),
        'Top': int(bbox['Top'] * img.shape[0]),
        'Height': int(bbox['Height'] * img.shape[0]),
        'Width': int(bbox['Width'] * img.shape[1])
    }


def blur_img(path, imgSize):
    img = cv2.imread(path)
    img = cv2.blur(img,(5,5))
    img = cv2.resize(img, imgSize)
    return img


def draw_bounding_box(img, bbox, color=(255, 0, 0)):
    pt1 = (bbox['Left'], bbox['Top'])
    pt2 = (bbox['Left'] + bbox['Width'], bbox['Top'] + bbox['Height'])
    cv2.rectangle(img, pt1, pt2, color, thickness=2)


def draw_label(img, bbox, label,
               bg_color=(255, 0, 0), text_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]

    x, y = (bbox['Left'], bbox['Top'])
    cv2.rectangle(img, (x, y-size[1]), (x+size[0], y), bg_color, cv2.FILLED)
    cv2.putText(img, label, (x, y), font, font_scale, text_color, thickness)


def annotate_frame(frame, labels, scores, bboxes):
    for i, label in enumerate(labels):
        if CLASS_MAP[label] == 'swoosh' or CLASS_MAP[label] == 'nike':
            bbox = bboxes[i]
            bbox_min, bbox_max = convert_bboxes_dims(frame, bbox)

            blurred_frame = cv2.blur(frame,(25,25),cv2.BORDER_DEFAULT)

            mask = np.zeros(frame.shape, dtype=np.uint8)
            mask = cv2.rectangle(mask, bbox_min, bbox_max, (255,255,255), -1)

            frame = np.where(mask!=np.array([255, 255, 255]), frame, blurred_frame)

        if CLASS_MAP[label]=='human':
            bbox = bboxes[i]
            bbox_min, bbox_max = convert_bboxes_dims(frame, bbox)

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


def convert_bboxes_dims(img, bbox, size=384):
    """
    Resize the bounding box to frame dimensions

    Returns:
        bbox_min
        bbox_max
    """

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
#             frame_time = datetime.now()
#             frame_name = f'{frame_time.strftime("frame-%y-%m-%d_%H-%M-%S%f")}'
#             frame_path = Path(frame_dir / frame_name)
#
#             jpeg = convert_to_jpg(frame, RESOLUTION['training'])
#             save_jpeg_to_temp(jpeg, frame_dir, frame_name)
#             log.info(f"Saving picture {frame_name} to {frame_dir}")

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
