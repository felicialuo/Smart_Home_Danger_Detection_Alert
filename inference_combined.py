import joblib
import csv
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

def read_results(csv_path):
    results = []
    labels = []
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            file_name, label, top1_pred, top1_conf, top2_pred, top2_conf, top3_pred, top3_conf, top4_pred, top4_conf, top5_pred, top5_conf = row
            labels.append(label)
            results.append([top1_pred, top1_conf, top2_pred, top2_conf, top3_pred, top3_conf, top4_pred, top4_conf, top5_pred, top5_conf])

    results = np.asarray(results).astype(float)
    labels  = np.asarray(labels).astype(int)
    return results, labels

def get_labels(csv_path='datasets/home_labels.csv'):
    label2id = {}
    id2label = {}
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)

        for i, row in enumerate(csv_reader):
            class_name = row[0]
            label2id[class_name] = i
            id2label[i] = class_name

    return label2id, id2label


if __name__ == "__main__":
    # Class label csv path
    labels_csv_path = 'datasets/home_labels.csv'
    # output result csv path
    output_vclip_path = 'outputs/vclip_results.csv'
    output_clap_path = 'outputs/clap_results.csv'

    # get labels
    label2id, id2label = get_labels(csv_path=labels_csv_path)

    # Load the model from the file
    trained_ensemble = joblib.load('trained_RF_ensemble.joblib')

    vclip_results, vclip_labels = read_results(output_vclip_path)
    clap_results, clap_labels = read_results(output_clap_path)

    X = np.hstack([vclip_results, clap_results])
    assert np.array_equal(vclip_labels, clap_labels)
    y = np.array(vclip_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

    y_pred = trained_ensemble.predict(X_test)


    correct = defaultdict(int)
    total = defaultdict(int)
    for i, label in enumerate(y_test):
        total[label] += 1
        if label == y_pred[i]:
            correct[label] += 1
            
    for label in total:
        print(f"{id2label[label]}: {correct[label]} / {total[label]}")