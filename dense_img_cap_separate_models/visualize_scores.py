import _pickle as pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_scores(model_name):
    scores = {}
    with open('../dataset/' + model_name + '_spice_scores_p1.pickle', 'rb') as f:
        spice1 = pickle.load(f)
    with open('../dataset/' + model_name + '_spice_scores_p2.pickle', 'rb') as f:
        spice2 = pickle.load(f)
    spice = np.array(list(spice1[1]) + list(spice2[1]))
    scores['SPICE'] = (np.average(spice), spice)
    with open('../dataset/' + model_name + '_meteor_scores.pickle', 'rb') as f:
        scores['METEOR'] = pickle.load(f)
    with open('../dataset/' + model_name + '_rouge_scores.pickle', 'rb') as f:
        scores['ROUGE'] = pickle.load(f)
    with open('../dataset/' + model_name + '_cider_scores.pickle', 'rb') as f:
        scores['CIDER'] = pickle.load(f)
    with open('../dataset/' + model_name + '_bleu1_scores.pickle', 'rb') as f:
        scores['BLEU_1'] = pickle.load(f)
    with open('../dataset/' + model_name + '_bleu2_scores.pickle', 'rb') as f:
        scores['BLEU_2'] = pickle.load(f)
    with open('../dataset/' + model_name + '_bleu3_scores.pickle', 'rb') as f:
        scores['BLEU_3'] = pickle.load(f)
    with open('../dataset/' + model_name + '_bleu4_scores.pickle', 'rb') as f:
        scores['BLEU_4'] = pickle.load(f)
    return scores


def load_loss_values(file_name):
    return pd.read_table(file_name, sep=',')


def print_average_scores(scores):
    for key in scores.keys():
        print(f'Average {key} score: {round(scores[key][0], 4)}')


def plot_graph_scores_model_based(scores, model_name):
    data = pd.DataFrame()
    data['Indexes'] = [key for key in scores]
    data[model_name] = [scores[key][0] for key in scores]
    sns.set(style='darkgrid', context='poster')
    f, ax = plt.subplots()
    sns.barplot(x='Indexes', y=model_name, palette='Blues_d', ax=ax, data=data)
    ax.axhline(0, color='k', clip_on=False)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Average score')
    sns.despine(bottom=True)
    plt.tight_layout(h_pad=2)
    plt.title(f'Average scores for {model_name}')
    plt.show()


def plot_graph_scores_metric_based(scores, metric_name):
    data = pd.DataFrame()
    data['Indexes'] = [key for key in scores]
    data[metric_name] = [scores[key][metric_name][0] for key in scores]
    sns.set(style='darkgrid', context='poster')
    f, ax = plt.subplots()
    sns.barplot(x=metric_name, y='Indexes', palette='Blues_d', ax=ax, data=data, orient='h')
    ax.axvline(0, color='k', clip_on=False)
    ax.set_ylabel('Models')
    ax.set_xlabel('Average score')
    plt.xlim(0, 1)
    sns.despine(bottom=True)
    plt.tight_layout(h_pad=2)
    plt.title(f'Average scores for {metric_name}')
    plt.show()


def plot_graph_loss(values, model_name):
    data = pd.DataFrame()
    data['epoch'] = list(values['epoch']._values + 1) + list(values['epoch']._values + 1)
    data['loss name'] = ['train'] * len(values) + ['validation'] * len(values)
    data['loss'] = list(values['loss']._values) + list(values['val_loss']._values)
    sns.set(style='darkgrid', context='poster', font='DejaVu Sans')
    f, ax = plt.subplots()
    sns.lineplot(x='epoch', y='loss', hue='loss name', style='loss name', markers=True, dashes=False, data=data)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    plt.title(f'Train and validation loss for {model_name}')
    plt.show()


if __name__ == '__main__':
    print('===== Model 2 =====')
    eval_scores_m2 = load_scores('m1-85')
    print_average_scores(eval_scores_m2)
    plot_graph_scores_model_based(eval_scores_m2, 'M2')

    print('===== Model 3 =====')
    eval_scores_m3 = load_scores('m2-85')
    print_average_scores(eval_scores_m3)
    plot_graph_scores_model_based(eval_scores_m3, 'M3')

    plot_graph_scores_metric_based({'M2': eval_scores_m2, 'M3': eval_scores_m3}, 'SPICE')
    plot_graph_scores_metric_based({'M2': eval_scores_m2, 'M3': eval_scores_m3}, 'ROUGE')
    plot_graph_scores_metric_based({'M2': eval_scores_m2, 'M3': eval_scores_m3}, 'METEOR')
    plot_graph_scores_metric_based({'M2': eval_scores_m2, 'M3': eval_scores_m3}, 'CIDER')
    plot_graph_scores_metric_based({'M2': eval_scores_m2, 'M3': eval_scores_m3}, 'BLEU_1')
    plot_graph_scores_metric_based({'M2': eval_scores_m2, 'M3': eval_scores_m3}, 'BLEU_2')
    plot_graph_scores_metric_based({'M2': eval_scores_m2, 'M3': eval_scores_m3}, 'BLEU_3')
    plot_graph_scores_metric_based({'M2': eval_scores_m2, 'M3': eval_scores_m3}, 'BLEU_4')

    loss_values_m2 = load_loss_values('logs/text_generation_m1.log')
    plot_graph_loss(loss_values_m2, 'M2')

    loss_values_m3 = load_loss_values('logs/text_generation_m2.log')
    plot_graph_loss(loss_values_m3, 'M3')
