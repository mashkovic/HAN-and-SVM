# COPIED FROM DEUTSCH
import torch
import sys
import csv
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
import numpy as np
from math import sqrt

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    
    report = metrics.classification_report(y_true, y_pred,output_dict=True)
    weighted_f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    rsme = sqrt(metrics.mean_squared_error(y_true, y_pred))
    output = {
        'accuracy':report['accuracy'],
        'macro_avg_precision':report['macro avg']['precision'],
        'macro_avg_recall':report['macro avg']['recall'],
        'macro_avg_f1':report['macro avg']['f1-score'],
        'weighted_avg_precision':report['weighted avg']['precision'],
        'weighted_avg_recall':report['weighted avg']['recall'],
        'weighted_avg_f1':report['weighted avg']['f1-score'],
        'weighted_f1':weighted_f1 ,
        'macro_f1':macro_f1,
        'micro_f1':micro_f1,
        'rsme':rsme
    }
    
    
    if 'accuracy' in list_metrics:
        output['accuracy_metrics'] = metrics.accuracy_score(y_true, y_pred)
    if 'precision' in list_metrics:
        output['precision'] = metrics.precision_score(
            y_true, y_pred, average="weighted")
    if 'recall' in list_metrics:
        output['recall'] = metrics.recall_score(
            y_true, y_pred, average="weighted")
    if 'f1' in list_metrics:
        output['f1'] = metrics.f1_score(y_true, y_pred, average="weighted")
    if 'f1' in list_metrics:
        output['f1'] = metrics.f1_score(y_true, y_pred, average="weighted")
    if 'f1' in list_metrics:
        output['f1'] = metrics.f1_score(y_true, y_pred, average="weighted")
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(
            metrics.confusion_matrix(y_true, y_pred))
    
    return output


def matrix_mul(input, weight, bias=False, test=False):
    feature_list = []
    for f in input:
        feature = torch.mm(f, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            bias = bias.repeat(feature.size()[0], 1)
            feature = feature + bias
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    output = torch.cat(feature_list, 0).squeeze(2)
    return output


def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


def get_max_lengths(data_path=None, fed_data=None):
    word_length_list = []
    sent_length_list = []
    word_list_all = []

    if fed_data is not None:
        df_train = pd.DataFrame(fed_data[0], columns=['tweets', 'judgements'])
        df = df_train
    else:
        df = pd.read_csv(data_path, encoding='utf8', sep=',')

    for idx, line in df.iterrows():
        try:
            text = line[0].lower()
        except AttributeError:
            continue

        sent_list = sent_tokenize(text)
        sent_length_list.append(len(sent_list))

        for sent in sent_list:
            word_list = word_tokenize(sent)
            word_length_list.append(len(word_list))
            word_list_all.append(word_list)

    sorted_word_length = sorted(word_length_list)
    sorted_sent_length = sorted(sent_length_list)

    return (sorted_word_length[int(0.77 * len(sorted_word_length))],
            sorted_sent_length[int(0.77 * len(sorted_sent_length))])
