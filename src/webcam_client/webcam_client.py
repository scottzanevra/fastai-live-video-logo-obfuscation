import logging
import os
import time
import glob
from PIL import Image
import cv2
import numpy as np
import argparse

from datetime import datetime
from pathlib import Path
from src.inference import predict_from_model, load_model

logging.basicConfig()
log = logging.getLogger()
log.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# List of valid resolutions
RESOLUTION = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (858, 480), 'training': (300, 300)}

# Text labels
CLASS_MAP = {
    1: 'nike',
    2: 'swoosh',
    3: 'human'
}

COLOR_MAP = {
    'info': (255, 255, 255),
    'nike': (204, 204, 0),
    'swoosh': (255, 255, 0),
    'human': (102, 255, 255),
}

# Overlap percentage threshold for logo boxes
LOGO_OVERLAP_THRESHOLD = 80

MODEL_ARCHS = {
    0: "mmdet.retinanet",
    1: "torchvision.retinanet",
    2: "ross.efficientnet",
    3: "yolov5"
}


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


def annotate_info(frame, frame_skip, display_info, display_bounding_boxes, model_number):
    """
    Displays webcam info + toggle features available
    :param frame: frame to annotate
    :param frame_skip: number of frames currently being skipped
    :param: display_info: whether to display info text at all
    :param display_bounding_boxes: whether we are displaying bounding boxes
    :return: annotated frame
    """

    info_text = f"Clothing logo obfuscator. \n\n" \
                f"press ESC or q to quit" \
                f"press 'i' to toggle info \n" \
                f"press 'b' to toggle bounding boxes \n" \
                f"press 'a' to decrease frame skip \n" \
                f"press 's' to increase frame skip \n\n\n" \
                f"display bounding boxes: {display_bounding_boxes}\n" \
                f"frames skipped: {frame_skip}\n" \
                f"model architecture: {MODEL_ARCHS[model_number]}"

    if not display_info:
        return

    y0, dy = 50, 20
    for i, line in enumerate(info_text.split('\n')):
        y = y0 + i*dy
        cv2.putText(
            frame, line, (50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 if i == 0 else 0.65,
            COLOR_MAP['info'],
            2 if i == 0 else 1
        )


def annotate_bounding_boxes(frame, labels, scores, bboxes):
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
        if (CLASS_MAP[label] == 'swoosh' or CLASS_MAP[label] == 'nike') and scores[i] >= 0.5:
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
                blurred_frame = cv2.blur(frame, (45, 45), cv2.BORDER_DEFAULT)
                mask = np.zeros(frame.shape, dtype=np.uint8)
                mask = cv2.rectangle(mask, bbox_min, bbox_max, (255,255,255), -1)
                frame = np.where(mask != np.array([255, 255, 255]), frame, blurred_frame)
            else:
                # Not on human, but let's draw a rectangle to detect logo anyways
                 cv2.rectangle(frame, bbox_min, bbox_max, COLOR_MAP[CLASS_MAP[label]], 2)

            cv2.putText(
                frame, f'{scores[i]:.2f} {CLASS_MAP[label]} (overlap {max_percent:.2f}%)',
                (bbox_min[0], bbox_min[1]-5),  # offset slightly so text sits above bbox
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAP[CLASS_MAP[label]], 1)

        if CLASS_MAP[label] == 'human' and scores[i] >= 0.5:
            bbox = bboxes[i]
            bbox_min, bbox_max = scale_bbox_dims(frame, bbox)

            cv2.putText(
                frame, f'{scores[i]:.2f} {CLASS_MAP[label]}',
                (bbox_min[0], bbox_min[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAP[CLASS_MAP[label]], 1)

            cv2.rectangle(frame, bbox_min, bbox_max, COLOR_MAP[CLASS_MAP[label]], 1)

    return frame


def save_frame(frame, save_dir, ext='jpg'):
    """
    Saves annotated cv2 frame from webclient to a specified local directory.
    :param frame: annotated cv2 frame to same
    :param save_dir: directory to save frame to
    :param ext: file type, e.g. 'jpg' or 'png'
    :return:
    """
    frame_time = datetime.now()
    frame_name = f'{frame_time.strftime("frame-%y-%m-%d_%H-%M-%S%f.")}{ext}'
    frame_path = str(Path(save_dir / frame_name))
    cv2.imwrite(frame_path, frame)
    log.info(f"Saving picture to {frame_path}")


def convert_to_jpg(frame, resolution):
    """
    Converts the captured frame to the desired resolution
    """
    ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, resolution))
    if not ret:
        raise Exception('Failed to set frame data')
    return jpeg


def generate_gif_from_frame_dir(
        frame_dir,
        save_path,
        duration=20,
        loop=0):
    """
    Creates a gif from an saved frames output during live detection
    :param frame_dir:
    :param save_path:
    :param duration: duration of gif
    :param loop:
    :return:
    """

    frame_paths = sorted(glob.glob(f'{frame_dir}/*.png'))
    frames = [Image.open(f) for f in frame_paths]

    frames[0].save(
        fp=f'{save_path}.gif', format='GIF', save_all=True,
        append_images=frames[1:], duration=duration, loop=loop)

    return save_path


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


def run_webcam(model1_number=None, model1_path=None, model2_number=None, model2_path=None, save_frames=False):
    """
        Perform human and logo object detection on live cv2 VideoCapture
    :param model1_number: type of model to load.
        Model numbers correspond to the following architectures:
        0: mmdet.retinanet
        1: torchvision.retinanet
        2: ross.efficientnet
        3: yolov5
    :param model1_path: path to model
    :param model2_number: secondary model type to load (optional, use for comparing models)
    :param model2_path: secondary model to load (optional)
    :param save_frames: if True, save to tmp directory
    :return:
    """

    cap = cv2.VideoCapture(0)
    time.sleep(1)  # just to avoid that initial black frame

    frame_skip = 10
    frame_count = 0

    winname = 'Clothing logo obfuscator- press ESC or Q to exit'
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 50, 50)

    # Create tmp dir for this round of webcam frames
    if save_frames:
        dir_time = datetime.now()
        frame_dir = Path(f'tmp/{dir_time.strftime("camera-frames-%y-%m-%d_%H-%M-%S")}')
        os.makedirs(frame_dir, exist_ok=True)

    # Load model for predictions
    model_type, model = load_model(model1_number, model1_path)
    model_number = model1_number

    # Toogle boolean for displaying bounding boxes
    display_bounding_box = True
    display_info = True

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError('Failed to capture frame')
        if frame_count % frame_skip == 0:  # only analyze every n frames

            annotate_info(frame, frame_skip, display_info, display_bounding_box, model_number)

            if display_bounding_box:
                # Inference time
                labels, scores, bboxes = predict_from_model(model_type, model, cv_img=frame)
                frame = annotate_bounding_boxes(frame, labels, scores, bboxes)

            cv2.imshow(winname, frame)

            # Save images based on timestamp
            if save_frames:
                log.info(f"Saving frame to {frame_dir}")
                save_frame(frame, frame_dir, ext='jpg')

        frame_count += 1

        # Press ESC or 'q' to quit
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

        # Toggle displaying bounding boxes
        if k == ord('b'):
            display_bounding_box = not display_bounding_box
        if k == ord('a'):
            if frame_skip >= 6:
                frame_skip -= 5
            else:
                frame_skip = 1
        if k == ord('s'):
            frame_skip += 5
        if k == ord('i'):
            display_info = not display_info
        if k == ord('1'):
            model_type, model = load_model(model1_number, model1_path)
            model_number = model1_number
            log.info(f"Loading model #{model1_number}")
        if k == ord('2'):
            if model2_path:
                model_type, model = load_model(model2_number, model2_path)
                model_number = model2_number
                log.info(f"Loading model #{model2_number}")

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='perform live inference on webcam video stream')
    parser.add_argument('--model-number', type=int,
                        dest='model_number',
                        help='model number corresponding to model architecture')
    parser.add_argument('--model-path', type=str,
                        dest='model_path',
                        help='path to load trained model from')
    parser.add_argument('--model2-number', type=int,
                        dest='model2_number', default=None,
                        help='second model number corresponding to model architecture (optional)')
    parser.add_argument('--model2-path', type=str,
                        dest='model2_path', default=None,
                        help='path to load trained second model from (optional)')
    parser.add_argument('--save-frames', dest='save_frames', action='store_true',
                        help='save frames to tmp directory')
    # Parse arguments
    args = parser.parse_args()
    model_number = args.model_number
    model_path = args.model_path

    model2_number = args.model2_number
    model2_path = args.model2_path
    save_frames = args.save_frames

    run_webcam(
        model1_number=model_number,
        model1_path=model_path,
        model2_number=model2_number,
        model2_path="models/model_3_step2_final.m",
        save_frames=save_frames
    )
