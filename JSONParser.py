import json
import requests
import os


def parse_region_data(file1, file2):
    # path, x, y, w, h, class
    with open(file1, 'r', encoding='utf-8') as doc:
        data = json.loads(doc.read())
    with open(file2, 'w+', encoding='utf-8') as doc:
        i = 0
        for image in data:
            path = 'Visual Genome\\' + str(image['image_id']) + '.jpg'
            objects = image['objects']
            for object_ in objects:
                x = object_['x']
                y = object_['y']
                w = object_['w']
                h = object_['h']
                name = object_['names'][0]
                name = name.replace(',', '')
                doc.write(path + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + ',' + name + '\n')
            i += 1
            print(str(i) + '/' + str(len(data)))


def download_images(file):
    directory = 'Visual Genome\\'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file, 'r', encoding='utf-8') as doc:
        data = json.loads(doc.read())
    i = 0
    for image in data:
        path = directory + str(image['image_id']) + '.jpg'
        image_url = image['image_url']
        img_data = requests.get(image_url).content
        with open(path, 'wb') as handler:
            handler.write(img_data)
        i += 1
        print(str(i) + '/' + str(len(data)))


objects_json = 'objects.json'
object_labels = 'object_labels.txt'
parse_region_data(objects_json, object_labels)
# download_images(objects_json)
