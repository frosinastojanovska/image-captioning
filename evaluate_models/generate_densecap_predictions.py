import json
import requests
import os


def predict(img):
    r = requests.post(
        "https://api.deepai.org/api/densecap",
        data={
            'image': img['url'],
        },
        headers={'api-key': 'api-key'}
    )
    with open('../dataset/densecap/' + str(img['image_id']) + '.json', 'w+') as f:
        json.dump(r.json(), f)


def generate_predictions(images, num_images):
    if not len(os.listdir('../dataset/densecap')) == num_images:
        for img in images:
            print(str(img['image_id']))
            predict(img)


if __name__ == '__main__':
    image_meta_file_path = '../dataset/image_data.json'
    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())

    test_images = image_meta_data[100000:]
    generate_predictions(test_images, len(test_images))

