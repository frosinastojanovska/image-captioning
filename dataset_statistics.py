import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as doc:
        img_data = json.loads(doc.read())
    return img_data


def plot_regions_distribution(data):
    distribution_array = [len(d['regions']) for d in data]
    distribution = np.zeros(np.max(distribution_array) + 1, dtype=np.float32)
    distribution_index = np.array(range(np.max(distribution_array) + 1), dtype=np.float32)
    for d in distribution_array:
        distribution[d] += 1
    data = pd.DataFrame()
    data['Number of images'] = distribution_index
    data['Number of regions'] = distribution
    sns.set(style='whitegrid', context='poster', font='DejaVu Sans', font_scale=1.5)
    f, ax = plt.subplots()
    ax.set(xscale='log', yscale='log')
    plt.xlim(1e0, 1e3)
    plt.ylim(0.5e-1, 1e6)
    ax.grid(False)
    sns.regplot('Number of images', 'Number of regions', data, ax=ax, scatter_kws={"s": 200}, fit_reg=False,
                color='xkcd:lavender')
    ax.grid(False)
    plt.title('Regions distribution')
    plt.show()
    avg = np.average(distribution_array)
    print(f'Average number of regions per image: {avg}')


def plot_caption_length_distribution(data):
    distribution_array = [' '.join(item['phrase'].split()).count(' ') + 1 for sublist in [d['regions'] for d in data] for item in sublist]
    distribution = np.zeros(np.max(distribution_array) + 1, dtype=np.float32)
    distribution_index = np.array(range(np.max(distribution_array) + 1), dtype=np.float32)
    for d in distribution_array:
        distribution[d] += 1
    data = pd.DataFrame()
    data['Number of captions'] = distribution
    data['Caption length'] = distribution_index
    sns.set(style='whitegrid', context='poster', font='DejaVu Sans', font_scale=1.5)
    f, ax = plt.subplots()
    ax.set(xscale='log', yscale='log')
    plt.xlim(1e-1, 1e3)
    plt.ylim(1e-1, 1e8)
    ax.grid(False)
    sns.regplot('Caption length', 'Number of captions', data, ax=ax, scatter_kws={"s": 200}, fit_reg=False,
                color='xkcd:lavender')
    ax.grid(False)
    plt.title('Caption length distribution')
    plt.show()
    avg = np.average(distribution_array)
    print(f'Average caption length: {avg}')


if __name__ == '__main__':
    image_info = load_data('dataset/region_descriptions.json')
    plot_regions_distribution(image_info)
    plot_caption_length_distribution(image_info)
