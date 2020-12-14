import argparse
import os
import numpy as np
from collections import Counter
from transformers import pipeline
from predict_sentiment import read_data, predict
from fine_tune import fine_tune_fun
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def gettop10(sentences, labels, classifier, verbose, top10_type='overall'):

    preds, labels, top10 = predict(sentences, labels, classifier, verbose, top10_type)

    predictions = np.array(preds)
    labels = np.array(labels)

    top10_preds = top10['top10_preds']
    top10_labels = top10['top10_labels']

    top10_preds = np.array(top10_preds)
    top10_labels = np.array(top10_labels)

    accuracy = np.sum(predictions == labels)/(len(labels)*1.0)
    print('Overall Accuracy: {0:.2f}'.format(accuracy))

    top10_accuracy = np.sum(top10_preds == top10_labels)/(len(top10_labels)*1.0)
    print('Top10 Accuracy: {0:.2f}'.format(top10_accuracy))

    top10_sents = top10['top10_sents']
    rest_sent_labels = [(sent, label) for sent, label in zip(sentences, labels) if not sent in top10_sents]

    # assert len(rest_sent_labels)+len(top10) == len(sentences)

    rest = {}
    rest['sents'] = [ele[0] for ele in rest_sent_labels]
    rest['labels'] = [ele[1] for ele in rest_sent_labels]

    return top10, rest

def save_top10_rest(top10, rest, count, save_directory):

    os.makedirs(os.path.join(save_directory, str(count)), exist_ok=True)

    top10_output = []
    top10_sents = top10['top10_sents']
    top10_preds = top10['top10_preds']
    for i, ele in enumerate(top10_sents):
        top10_output.append('\t'.join([ele, top10_preds[i]]))

    with open(os.path.join(save_directory, str(count), 'top10.txt'), 'w') as out_f:
        out_f.write('\n'.join(top10_output))
    
    rest_output = []
    rest_sents = rest['sents']
    rest_labels = rest['labels']
    for i, ele in enumerate(rest_sents):
        rest_output.append('\t'.join([ele, rest_labels[i]]))

    with open(os.path.join(save_directory, str(count), 'rest.txt'), 'w') as out_f:
        out_f.write('\n'.join(rest_output))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine-tuning on additional sentences')
    parser.add_argument('--input_file', type=str, required=True,
                        help='file containing sentences and labels separated by a tab')
    parser.add_argument('--save_directory', type=str, required=True,
                        help='directory to save outputs to')
    args = parser.parse_args()

    classifier = pipeline('sentiment-analysis', 'cardiffnlp/twitter-roberta-base-sentiment', device=0)

    sentences, labels = read_data(args.input_file, False)
    top10, rest = gettop10(sentences, labels, classifier, False, 'binary-class-wise')
    save_top10_rest(top10, rest, 0, args.save_directory)

    count = 1
    while rest:

        print('Top10 Pred Distribution')
        pred_label_distribution = Counter(top10['top10_preds'])
        print(pred_label_distribution)

        fine_tune_fun(top10['top10_sents'], top10['top10_preds'], rest['sents'], rest['labels'], os.path.join(args.save_directory, str(count)))

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.save_directory, str(count)))
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.save_directory, str(count)))
        classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0)

        top10, rest = gettop10(rest['sents'], rest['labels'], classifier, False, 'binary-class-wise')
        save_top10_rest(top10, rest, count, args.save_directory)
        count += 1