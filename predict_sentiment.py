import argparse
from transformers import pipeline
import numpy as np
import emoji
from collections import Counter
import pickle
import pdb

def contains_emoji(sentence):

    contains_bool = any(word in sentence for word in emoji.UNICODE_EMOJI)
    return contains_bool

def read_data(input_file, emoji_filter):
    with open(input_file, 'r') as in_f:
        lines = in_f.readlines()
    lines = [line.strip() for line in lines]
    sentences, labels = [], []
    for line in lines:
        try:
            sentence, label = line.split('\t')
        except:
            continue
        if not emoji_filter:
            sentences.append(sentence)
            labels.append(label)
        elif contains_emoji(sentence):
            sentences.append(sentence)
            labels.append(label)

    return sentences, labels

def predict(sentences, labels, classifier, verbose):
    preds, output_labels = [], []
    pred_label_score = []
    mapping_dict = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}
    for i, (sentence, label) in enumerate(zip(sentences, labels)):
        try:
            pred = classifier(sentence)
        except:
            print('Faced error in predicting sentiment for: {}. Moving on...'.format(sentence))
            continue
        # if pred[0]['score'] < 0.75:
        #     pred[0]['label'] = 'neutral'
        pred_label = pred[0]['label']
        if pred_label in mapping_dict:
            pred_label = mapping_dict[pred_label]
        pred_label = pred_label.lower()
        label = label.lower()
        preds.append(pred_label)
        output_labels.append(label)
        pred_label_score.append((sentence, pred_label, label, pred[0]['score']))
        if verbose:
            print('{}) Sentence: {}\nLabel: {}\nScore: {}'.format(i, sentence, pred_label, pred[0]['score']))
            print(25*'#')
        else:
            if i % 1000 == 0:
                print('Finished processing {}/{} sentences'.format(i, len(sentences)))    
    
    # with open('sail.pkl', 'wb') as out_f:
    #     pickle.dump(pred_label_score, out_f)

    sorted_pred_label_score = sorted(pred_label_score, key=lambda k: k[3], reverse=True)
    top_10_num = int(0.1*len(sentences))
    top_10_pred_label_score = sorted_pred_label_score[:top_10_num]

    top10 = {}
    top10['top10_sents'] = [ele[0] for ele in top_10_pred_label_score]
    top10['top10_preds'] = [ele[1] for ele in top_10_pred_label_score]
    top10['top10_labels'] = [ele[2] for ele in top_10_pred_label_score]
    top10['top10_scores'] = [ele[3] for ele in top_10_pred_label_score]

    return preds, output_labels, top10

def main():

    parser = argparse.ArgumentParser(description='Predicting on dataset')
    parser.add_argument('--input_file', type=str, required=True,
                        help='file containing sentences and labels separated by a tab')
    parser.add_argument('--emoji_filter', action='store_true',
                        help='Whether to apply emoji heuristic.')
    parser.add_argument('--verbose', action='store_true',
                        help='Whether to apply emoji heuristic.')
    args = parser.parse_args()

    sentences, labels = read_data(args.input_file, args.emoji_filter)

    label_distribution = Counter(labels)
    print('Actual Label Distribution:')
    print(label_distribution)
    # default HF DistillBERT model
    # classifier = pipeline('sentiment-analysis')
    # model trained on twitter data
    classifier = pipeline('sentiment-analysis', 'cardiffnlp/twitter-roberta-base-sentiment')
    #sota
    # classifier = pipeline('sentiment-analysis', 'mrm8488/t5-base-finetuned-imdb-sentiment') # error in loading this model
    # hindi sentiment analyzer that words on devanagari
    # classifier = pipeline('sentiment-analysis', 'monsoon-nlp/hindi-bert') # this did not work
    # classifier = pipeline('sentiment-analysis', 'monsoon-nlp/hindi-tpu-electra')
    predictions, labels, top10 = predict(sentences, labels, classifier, args.verbose)

    top10_sents = top10['top10_sents']
    top10_preds = top10['top10_preds']
    top10_labels = top10['top10_labels']
    top10_scores = top10['top10_scores']

    # num_print_sents = 75
    # print('Opening pandora\'s box...')
    # for i in range(num_print_sents):
    #     print('Sentence: {}'.format(top10_sents[i]))
    #     print('Prediction: {}'.format(top10_preds[i]))
    #     print('Actual Label: {}'.format(top10_labels[i]))
    #     print('Score: {}'.format(top10_scores[i]))
    #     print(25*'#')

    print('Is Zero-shot better than humans?')
    print(50*'-')
    for i in range(len(top10_sents)):
        if top10_labels[i] != top10_preds[i]:
            print('Sentence: {}'.format(top10_sents[i]))
            print('Prediction: {}'.format(top10_preds[i]))
            print('Actual Label: {}'.format(top10_labels[i]))
            print('Score: {}'.format(top10_scores[i]))
            print(25*'#')

    pred_label_distribution = Counter(top10_preds)
    print('Distribution of Top 10% of predicted labels:')
    print(pred_label_distribution)

    predictions = np.array(predictions)
    labels = np.array(labels)

    top10_preds = np.array(top10_preds)
    top10_labels = np.array(top10_labels)

    accuracy = np.sum(predictions == labels)/(len(labels)*1.0)
    print('Accuracy: {0:.2f}'.format(accuracy))

    top10_accuracy = np.sum(top10_preds == top10_labels)/(len(top10_labels)*1.0)
    print('Top10 Accuracy: {0:.2f}'.format(top10_accuracy))
if __name__ == '__main__':
    main()