import base64
from datetime import datetime

import boto3

REKOGNITION = boto3.client('rekognition')


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
