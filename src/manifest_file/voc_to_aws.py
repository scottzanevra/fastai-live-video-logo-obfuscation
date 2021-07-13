import os
import json
from lxml import etree
from glob import iglob

XML_PATH = 'annotations/xml'
JSON_PATH = 'annotations/json'


def main():
    labels = {}

    for xml_file in iglob(os.path.join(os.path.dirname(__file__), '{}/*.xml'.format(XML_PATH))):

        with open(xml_file) as file:
            print(f'Processing file {xml_file}...')

            annotations = etree.fromstring(file.read())

            image_filename = annotations.find('filename').text
            boxes = annotations.iterfind('object')

            size = annotations.find('size')
            image_width = float(size.find('width').text)
            image_height = float(size.find('height').text)

            json_document = {
                'file': image_filename,
                'image_size': [{
                    'width': image_width,
                    'height': image_height,
                    'depth': 3
                }],
                'annotations': [],
                'categories': []
            }

            categories = {}

            for box in boxes:
                bndbox = box.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                label_name = box.find('name').text

                if label_name not in labels:
                    labels[label_name] = len(labels)

                class_id = labels[label_name]
                categories[class_id] = label_name

                json_document['annotations'].append({
                    'class_id': class_id,
                    'top': ymin,
                    'left': xmin,
                    'width': xmax - xmin,
                    'height': ymax - ymin,
                })

            for category in categories:
                json_document['categories'].append({
                    'class_id': category,
                    'name': categories[category]
                })

            json_filename, _ = os.path.splitext(image_filename)
            with open(f'{os.path.join(JSON_PATH, json_filename)}.json', 'w') as json_file:
                json.dump(json_document, json_file, indent=2)


if __name__ == "__main__":
    main()