import base64
import json
import logging
import os
import socket
import time
import uuid

import boto3
import cv2


logging.basicConfig()
log = logging.getLogger()
log.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

REGION = os.environ.get('REGION', 'ap-southeast-2')
LAMBDA = boto3.client('lambda', region_name=REGION)
CAMERA_ID = socket.gethostname()


# List of valid resolutions
RESOLUTION = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (858, 480), 'training': (300, 300)}
TMP_DIR = "tmp"


def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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


def annotate_frame(frame, response_dict):
    labels_to_annotate = {
        'Person': (255, 0, 0),  # cv2 is BGR, so this is blue
        'AWSLogo': (0, 255, 0)
    }

    for label, color in labels_to_annotate.items():
        for p in response_dict['labels'][label]['Instances']:
            bbox = resize_bbox(frame, p['BoundingBox'])
            draw_bounding_box(frame, bbox, color)


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


def lambda_handler(event, context):

    cap = cv2.VideoCapture(0)
    time.sleep(1)  # just to avoid that initial black frame

    frame_skip = 30
    frame_count = 0

    winname = 'Press ESC or Q to quit'
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 50, 50)

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError('Failed to capture frame')

        if frame_count % frame_skip == 0:  # only analyze every n frames
            # detect_ppe(frame, client)
            cv2.imshow(winname, frame)
            # Create random string to use as teh image name
            pic_id = str(uuid.uuid4())
            jpeg = convert_to_jpg(frame, RESOLUTION['training'])
            make_directory(TMP_DIR)
            save_jpeg_to_temp(jpeg, TMP_DIR, pic_id)
            log.info(f"Saving picture {pic_id} to {TMP_DIR}")

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
