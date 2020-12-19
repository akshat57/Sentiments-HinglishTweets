import argparse
import glob
import os
import numpy as np

def readtop10(top10_file):

    with open(top10_file, 'r') as in_f:
        lines = in_f.readlines()

    lines = [ele.strip() for ele in lines]

    top10_preds, top10_labels = [], []
    for line in lines:
        sent, pred, label = line.split('\t')
        if label != 'neutral':
            top10_preds.append(pred)
            top10_labels.append(label)

    return top10_preds, top10_labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing performance of high confidence sentences')
    parser.add_argument('--iteration_directory', type=str, required=True,
                        help='directory to save outputs to')
    parser.add_argument('--num_iters', type=int, default=100,
                        help='number of iterations to test for')
    args = parser.parse_args()
    
    preds, labels = [], []
    for top10_file in glob.glob(os.path.join(args.iteration_directory, '*', 'top10.txt')):
        folder_name, iter_num, file_name = top10_file.split(os.sep)
        if int(iter_num) <= args.num_iters:
            print(iter_num)
            top10_preds, top10_labels = readtop10(top10_file)
            preds.extend(top10_preds)
            labels.extend(top10_labels)

    predictions = np.array(preds)
    labels = np.array(labels)

    print("Total binary sentences annotated: {}".format(len(labels)))

    accuracy = np.sum(predictions == labels)/(len(labels)*1.0)
    print('Accuracy: {0:.4f}'.format(accuracy))