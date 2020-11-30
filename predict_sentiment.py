import argparse
from transformers import pipeline
import numpy as np
import emoji
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

def predict(sentences, labels, classifier):
    preds, output_labels = [], []
    pred_label_score = []
    mapping_dict = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}
    for sentence, label in zip(sentences, labels):
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
        preds.append(pred_label)
        output_labels.append(label)
        pred_label_score.append((pred_label, label, pred[0]['score']))
        print('Sentence: {}\nLabel: {}\nScore: {}'.format(sentence, pred_label, pred[0]['score']))
        print(25*'#')
    
    sorted_pred_label_score = sorted(pred_label_score, key=lambda k: k[2], reverse=True)
    top_10_num = int(0.1*len(sentences))
    top_10_pred_label_score = sorted_pred_label_score[:top_10_num]

    top10_preds = [ele[0] for ele in top_10_pred_label_score]
    top10_labels = [ele[1] for ele in top_10_pred_label_score]

    return preds, output_labels, top10_preds, top10_labels

def main():

    parser = argparse.ArgumentParser(description='Predicting on dataset')
    parser.add_argument('--input_file', type=str, required=True,
                        help='file containing sentences and labels separated by a tab')
    parser.add_argument('--emoji_filter', action='store_true',
                        help='Whether to apply emoji heuristic.')
    args = parser.parse_args()

    sentences, labels = read_data(args.input_file, args.emoji_filter)

    classifier = pipeline('sentiment-analysis', 'cardiffnlp/twitter-roberta-base-sentiment')
    #sota
    # classifier = pipeline('sentiment-analysis', 'mrm8488/t5-base-finetuned-imdb-sentiment') # error in loading this model
    # hindi sentiment analyzer that words on devanagari
    # classifier = pipeline('sentiment-analysis', 'monsoon-nlp/hindi-bert') # this did not work
    # classifier = pipeline('sentiment-analysis', 'monsoon-nlp/hindi-tpu-electra')
    predictions, labels, top10_preds, top10_labels = predict(sentences, labels, classifier)

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