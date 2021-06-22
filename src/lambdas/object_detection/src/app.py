import base64
import json
import logging
import os
from datetime import datetime

import boto3

logging.basicConfig()
log = logging.getLogger()
log.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

REGION = os.environ.get('REGION', 'ap-southeast-2')
REKOGNITION = boto3.client('rekognition', region_name=REGION)


def format_json(data):
    return json.dumps(
        data, default=lambda d: d.isoformat() if isinstance(d, datetime.datetime) else str(d))

def detect_labels(image):

    response = REKOGNITION.detect_labels(Image={'Bytes': image})

    for label in response['Labels']:
        print ("Label: " + label['Name'])
        print ("Confidence: " + str(label['Confidence']))
        print ("Instances:")
        for instance in label['Instances']:
            print ("  Bounding box")
            print ("    Top: " + str(instance['BoundingBox']['Top']))
            print ("    Left: " + str(instance['BoundingBox']['Left']))
            print ("    Width: " +  str(instance['BoundingBox']['Width']))
            print ("    Height: " +  str(instance['BoundingBox']['Height']))
            print ("  Confidence: " + str(instance['Confidence']))
            print()

        print ("Parents:")
        for parent in label['Parents']:
            print ("   " + parent['Name'])
        print ("----------")
        print ()
    return len(response['Labels'])


def detect_text(image):

    response = REKOGNITION.detect_text(Image={'Bytes': image})

    textDetections = response['TextDetections']
    print('Detected text\n----------')
    for text in textDetections:
        print('Detected text:' + text['DetectedText'])
        print('Confidence: ' + "{:.2f}".format(text['Confidence']) + "%")
        print('Id: {}'.format(text['Id']))
        if 'ParentId' in text:
            print('Parent Id: {}'.format(text['ParentId']))
        print('Type:' + text['Type'])
        print()
    return len(textDetections)


def lambda_handler(event):
    # Extract and transform
    encoded_img = event['img']
    timestamp = datetime.utcnow()

    # Check if equipment is present
    img = base64.b64decode(encoded_img)
    response = detect_text(img)
    text = detect_labels(img)
    foo = "me"
    return None
