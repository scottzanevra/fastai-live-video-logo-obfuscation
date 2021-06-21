import base64
import json
import logging
import os
from datetime import datetime

import boto3

REGION = os.environ.get('REGION', 'ap-southeast-2')

REKOGNITION = boto3.client('rekognition', region_name=REGION)


def format_json(data):
    return json.dumps(
        data, default=lambda d: d.isoformat() if isinstance(d, datetime.datetime) else str(d))

logging.basicConfig()
log = logging.getLogger()
log.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))


def detect_labels(img):
    response = REKOGNITION.detect_labels(Image={'Bytes': img})

    return {
        'Person': get_label(response, 'Person'),
        'Helmet': get_label(response, 'Helmet')
    }



def lambda_handler(event, context):
    # Extract and transform
    encoded_img = event['img']
    timestamp = datetime.utcnow()

    # Check if equipment is present
    img = base64.b64decode(encoded_img)
    labels = detect_labels(img)


    return {
        'labels': labels
    }
