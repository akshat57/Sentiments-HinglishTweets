import argparse
from transformers import pipeline
import numpy as np
import emoji
from collections import Counter
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pdb

def contains_emoji(sentence):

    contains_bool = any(word in sentence for word in emoji.UNICODE_EMOJI)
    return contains_bool

def read_data(input_file, emoji_filter, binary_filter):
    with open(input_file, 'r') as in_f:
        lines = in_f.readlines()
    lines = [line.strip() for line in lines]
    sentences, labels, hindi_scores = [], [], []
    for line in lines:
        try:
            sentence, label, hindi_score = line.split('\t')
        except:
            continue
        if not emoji_filter and not binary_filter:
            sentences.append(sentence)
            labels.append(label)
            hindi_scores.append(hindi_score)
        elif emoji_filter and contains_emoji(sentence):
            sentences.append(sentence)
            labels.append(label)
            hindi_scores.append(hindi_score)
        elif binary_filter and label != 'neutral':
            sentences.append(sentence)
            labels.append(label)
            hindi_scores.append(hindi_score)

    return sentences, labels, hindi_scores

def predict(sentences, labels, hindi_scores, classifier, verbose, top10_type='overall', increase_factor=1):
    preds, output_labels = [], []
    pred_label_score = []
    mapping_dict = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}
    for i, (sentence, label, hindi_score) in enumerate(zip(sentences, labels, hindi_scores)):
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
        pred_label_score.append((sentence, pred_label, label, pred[0]['score'], hindi_score))
        if verbose:
            print('{}) Sentence: {}\nLabel: {}\nScore: {}'.format(i, sentence, pred_label, pred[0]['score']))
            print(25*'#')
        else:
            if i % 1000 == 0:
                print('Finished processing {}/{} sentences'.format(i, len(sentences)))    
    
    # with open('sentimix_binary_classwise_withincrements_run4.pkl', 'wb') as out_f:
    #     pickle.dump(pred_label_score, out_f)

    if top10_type == 'overall':
        pred_label_score = [ele for ele in pred_label_score if ele[1] != 'neutral']
        sorted_pred_label_score = sorted(pred_label_score, key=lambda k: k[3], reverse=True)
        top_10_num = int(0.1*increase_factor*len(sentences))
        top_10_pred_label_score = sorted_pred_label_score[:top_10_num]

        top_10_preds = [ele[1] for ele in top_10_pred_label_score]
        pred_distribution = Counter(top_10_preds)
        max_value = -1
        for key in pred_distribution:
            if pred_distribution[key] > max_value:
                max_value = pred_distribution[key]
        
        extra_preds = []
        # pdb.set_trace()
        for key in ['negative', 'positive']:
            if not key in pred_distribution:
                pred_distribution[key] = 0
            if pred_distribution[key] < (max_value/4.0):
                num_extra_sents = int((max_value/4.0) - pred_distribution[key])
                label_preds = [ele for ele in pred_label_score if ele[1] == key]
                sorted_label_preds = sorted(label_preds, key=lambda k: k[3], reverse=True)
                top10_preds = sorted_label_preds[pred_distribution[key]+1:pred_distribution[key]+1+num_extra_sents]
                extra_preds.extend(top10_preds)

        top10 = {}
        top10['top10_sents'] = [ele[0] for ele in top_10_pred_label_score]
        top10['top10_preds'] = [ele[1] for ele in top_10_pred_label_score]
        top10['top10_labels'] = [ele[2] for ele in top_10_pred_label_score]
        top10['top10_scores'] = [ele[3] for ele in top_10_pred_label_score]

        if extra_preds:
            top10['top10_sents'].extend([ele[0] for ele in extra_preds])
            top10['top10_preds'].extend([ele[1] for ele in extra_preds])
            top10['top10_labels'].extend([ele[2] for ele in extra_preds])
            top10['top10_scores'].extend([ele[3] for ele in extra_preds])
    
    elif top10_type == 'binary-class-wise':

        pos_preds = [ele for ele in pred_label_score if ele[1] == 'positive']
        neg_preds = [ele for ele in pred_label_score if ele[1] == 'negative']

        sorted_pos_preds = sorted(pos_preds, key=lambda k: k[3], reverse=True)
        sorted_neg_preds = sorted(neg_preds, key=lambda k: k[3], reverse=True)

        top10_pos_num = int(0.15*increase_factor*len(pos_preds))
        top10_pos = sorted_pos_preds[:top10_pos_num]

        top10_neg_num = int(0.1*increase_factor*len(neg_preds))
        top10_neg = sorted_neg_preds[:top10_neg_num]

        top10 = {}
        top10['top10_sents'] = [ele[0] for ele in top10_pos]
        top10['top10_preds'] = [ele[1] for ele in top10_pos]
        top10['top10_labels'] = [ele[2] for ele in top10_pos]
        top10['top10_scores'] = [ele[3] for ele in top10_pos]

        top10['top10_sents'].extend([ele[0] for ele in top10_neg])
        top10['top10_preds'].extend([ele[1] for ele in top10_neg])
        top10['top10_labels'].extend([ele[2] for ele in top10_neg])
        top10['top10_scores'].extend([ele[3] for ele in top10_neg])

    elif top10_type == 'class-wise':

        pos_preds = [ele for ele in pred_label_score if ele[1] == 'positive']
        neg_preds = [ele for ele in pred_label_score if ele[1] == 'negative']
        neutral_preds = [ele for ele in pred_label_score if ele[1] == 'neutral']

        sorted_pos_preds = sorted(pos_preds, key=lambda k: k[3], reverse=True)
        sorted_neg_preds = sorted(neg_preds, key=lambda k: k[3], reverse=True)
        sorted_neutral_preds = sorted(neutral_preds, key=lambda k: k[3], reverse=True)

        top10_pos_num = int(0.15*increase_factor*len(pos_preds))
        print('% of pos selected: {}'.format(0.15*increase_factor))
        print('Num of pos selected: {}'.format(top10_pos_num))
        top10_pos = sorted_pos_preds[:top10_pos_num]

        top10_neg_num = int(0.1*increase_factor*len(neg_preds))
        print('% of pos selected: {}'.format(0.1*increase_factor))
        print('Num of pos selected: {}'.format(top10_neg_num))
        top10_neg = sorted_neg_preds[:top10_neg_num]

        top10_neutral_num = int(0.05*increase_factor*len(neutral_preds))
        top10_neutral = sorted_neutral_preds[:top10_neutral_num]

        top10 = {}
        top10['top10_sents'] = [ele[0] for ele in top10_pos]
        top10['top10_preds'] = [ele[1] for ele in top10_pos]
        top10['top10_labels'] = [ele[2] for ele in top10_pos]
        top10['top10_scores'] = [ele[3] for ele in top10_pos]

        top10['top10_sents'].extend([ele[0] for ele in top10_neg])
        top10['top10_preds'].extend([ele[1] for ele in top10_neg])
        top10['top10_labels'].extend([ele[2] for ele in top10_neg])
        top10['top10_scores'].extend([ele[3] for ele in top10_neg])

        # top10['top10_sents'].extend([ele[0] for ele in top10_neutral])
        # top10['top10_preds'].extend([ele[1] for ele in top10_neutral])
        # top10['top10_labels'].extend([ele[2] for ele in top10_neutral])
        # top10['top10_scores'].extend([ele[3] for ele in top10_neutral])

    else:
        raise Exception("Incorrect type")

    active_sorted_pred_label_scores = [ele for ele in pred_label_score if ele[2] != 'neutral']
    # active_sorted_pred_label_scores = sorted(active_sorted_pred_label_scores, key=lambda k: (k[3]*0.2 - float(k[4])))
    active_sorted_pred_label_scores = sorted(active_sorted_pred_label_scores, key=lambda k: -float(k[4]))
    active_sorted_pred_label_scores = [ele for ele in active_sorted_pred_label_scores if not ele[0] in top10['top10_sents']] # removing sentences that are already going in for fine-tuning
    num_active = int(0.02*increase_factor*len(active_sorted_pred_label_scores)) # 2% of labelled sentences
    active_selections = active_sorted_pred_label_scores[:num_active]

    print('%'*10 + 'Select Few Active Sentences' + '%'*10)
    for ac in range(5):
        print('Sentence: {}'.format(active_selections[ac][0]))
        print('Prediction: {}'.format(active_selections[ac][1]))
        print('Label: {}'.format(active_selections[ac][2]))
        print('Score: {}'.format(active_selections[ac][3]))
        print('Hindi-ness: {}'.format(active_selections[ac][4]))
    print('%'*25)

    active_selections_dict = {}
    active_selections_dict['sents'] = [ele[0] for ele in active_selections]
    active_selections_dict['preds'] = [ele[1] for ele in active_selections]
    active_selections_dict['labels'] = [ele[2] for ele in active_selections]
    active_selections_dict['scores'] = [ele[3] for ele in active_selections]

    return preds, output_labels, top10, active_selections_dict

def main():

    parser = argparse.ArgumentParser(description='Predicting on dataset')
    parser.add_argument('--input_file', type=str, required=True,
                        help='file containing sentences and labels separated by a tab')
    parser.add_argument('--model_directory', type=str, required=True,
                        help='directory containing the model')
    parser.add_argument('--binary', action='store_true',
                        help='whether to use only binary labels for testing')
    parser.add_argument('--emoji_filter', action='store_true',
                        help='Whether to apply emoji heuristic.')
    parser.add_argument('--verbose', action='store_true',
                        help='Whether to apply emoji heuristic.')
    args = parser.parse_args()

    sentences, labels, hindi_scores = read_data(args.input_file, args.emoji_filter, args.binary)

    label_distribution = Counter(labels)
    print('Actual Label Distribution:')
    print(label_distribution)
    # default HF DistillBERT model
    # classifier = pipeline('sentiment-analysis')
    # model trained on twitter data
    # classifier = pipeline('sentiment-analysis', 'cardiffnlp/twitter-roberta-base-sentiment')
    
    # classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.model_directory)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_directory)
    # classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0, return_all_scores=True)
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0)
    #sota
    # classifier = pipeline('sentiment-analysis', 'mrm8488/t5-base-finetuned-imdb-sentiment') # error in loading this model
    # hindi sentiment analyzer that words on devanagari
    # classifier = pipeline('sentiment-analysis', 'monsoon-nlp/hindi-bert') # this did not work
    # classifier = pipeline('sentiment-analysis', 'monsoon-nlp/hindi-tpu-electra')
    predictions, labels, top10, active_selections = predict(sentences, labels, hindi_scores, classifier, args.verbose, 'class-wise')

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

    # print('Is Zero-shot better than humans?')
    # print(50*'-')
    # for i in range(len(top10_sents)):
    #     if top10_labels[i] != top10_preds[i]:
    #         print('Sentence: {}'.format(top10_sents[i]))
    #         print('Prediction: {}'.format(top10_preds[i]))
    #         print('Actual Label: {}'.format(top10_labels[i]))
    #         print('Score: {}'.format(top10_scores[i]))
    #         print(25*'#')

    pred_label_distribution = Counter(top10_preds)
    print('Distribution of Top 10% of predicted labels:')
    print(pred_label_distribution)

    predictions = np.array(predictions)
    labels = np.array(labels)

    pos_indices = [i for i, ele in enumerate(labels) if ele=='positive']
    neg_indices = [i for i, ele in enumerate(labels) if ele=='negative']

    pos_predictions = predictions[pos_indices]
    neg_predictions = predictions[neg_indices]

    pos_labels = labels[pos_indices]
    neg_labels = labels[neg_indices]

    top10_preds = np.array(top10_preds)
    top10_labels = np.array(top10_labels)

    accuracy = np.sum(predictions == labels)/(len(labels)*1.0)
    print('Accuracy: {0:.2f}'.format(accuracy))

    pos_accuracy = np.sum(pos_predictions == pos_labels)/(len(pos_labels)*1.0)
    print('Pos Accuracy: {0:.2f}'.format(pos_accuracy))

    neg_accuracy = np.sum(neg_predictions == neg_labels)/(len(neg_labels)*1.0)
    print('Neg Accuracy: {0:.2f}'.format(neg_accuracy))

    top10_accuracy = np.sum(top10_preds == top10_labels)/(len(top10_labels)*1.0)
    print('Top10 Accuracy: {0:.2f}'.format(top10_accuracy))
if __name__ == '__main__':
    main()