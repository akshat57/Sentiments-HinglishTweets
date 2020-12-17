import argparse
import os
import numpy as np
import pdb
from collections import Counter
from transformers import pipeline
from predict_sentiment_active import read_data, predict
from fine_tune_active import fine_tune_fun
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def gettop10(sentences, labels, hindi_scores, classifier, verbose, iter_count, save_directory, top10_type='overall', increment_factor=1):

    preds, labels, top10, active_selections  = predict(sentences, labels, hindi_scores, classifier, verbose, top10_type, increment_factor)

    top10_preds = top10['top10_preds']
    top10_labels = top10['top10_labels']
    top10_sents = top10['top10_sents']
    top10_scores = top10['top10_scores']

    print('Len of top10 before filtering: {}'.format(len(top10)))

    top10 = [ele for ele in zip(top10_sents, top10_preds, top10_labels, top10_scores) if ele[-1]>=0.1]

    print('Len of top10 after filtering: {}'.format(len(top10)))

    new_top10 = {}
    new_top10['top10_sents'] = [ele[0] for ele in top10]
    new_top10['top10_preds'] = [ele[1] for ele in top10]
    new_top10['top10_labels'] = [ele[2] for ele in top10]
    new_top10['top10_scores'] = [ele[3] for ele in top10]

    top10 = new_top10

    predictions = np.array(preds)
    labels = np.array(labels)

    # for calculating accuracies
    top10_preds = top10['top10_preds']
    top10_labels = top10['top10_labels']

    top10_preds = np.array(top10_preds)
    top10_labels = np.array(top10_labels)

    #ACTIVE
    top10_preds_distribution = top10['top10_preds']
    top10_preds_distribution = Counter(top10_preds_distribution)
    print("Label distribution before adding active sentences: {}".format(Counter(top10_preds_distribution)))

    # attaching active sentence with true labels to fine tuning dataset -- kind of hacky way to do this!
    active_sentences = active_selections['sents']
    active_labels = active_selections['labels']
    top10['top10_sents'].extend([ele for ele in active_sentences])
    top10['top10_preds'].extend([ele for ele in active_labels])
    top10['top10_labels'].extend([ele for ele in active_labels])

    output_active_sentences = []
    for sent, label in zip(active_sentences, active_labels):
        output_active_sentences.append('\t'.join([sent, label]))

    with open(os.path.join(save_directory, str(iter_count), 'ActiveSentences.txt'), 'w') as out_f:
        out_f.write('\n'.join(output_active_sentences))

    top10_preds_distribution = top10['top10_preds']
    top10_preds_distribution = Counter(top10_preds_distribution)
    print("Label distribution after adding active sentences: {}".format(Counter(top10_preds_distribution)))

    # Taking out rest of sentences
    top10_sents = top10['top10_sents']
    rest_sent_labels = [(sent, label, hindi_score) for sent, label, hindi_score in zip(sentences, labels, hindi_scores) if not sent in top10_sents]

    rest = {}
    rest['sents'] = [ele[0] for ele in rest_sent_labels]
    rest['labels'] = [ele[1] for ele in rest_sent_labels]
    rest['hindi_scores'] = [ele[2] for ele in rest_sent_labels]

    # Calculating accuracies
    non_neutral_indices = [i for i, ele in enumerate(labels) if ele!='neutral']

    predictions = predictions[non_neutral_indices]
    labels = labels[non_neutral_indices]

    accuracy = np.sum(predictions == labels)/(len(labels)*1.0)
    print('Overall Accuracy: {0:.2f}'.format(accuracy))

    write_flag = 'a'
    if iter_count == 0:
        write_flag = 'w'
    with open(os.path.join(save_directory, 'OverallResults.txt'), write_flag) as out_f:
        out_f.write('{}: {}\n'.format(iter_count, accuracy))

    top10_accuracy = np.sum(top10_preds == top10_labels)/(len(top10_labels)*1.0)
    print('Top10 Accuracy: {0:.2f}'.format(top10_accuracy))

    return top10, rest

def save_top10_rest(top10, rest, count, save_directory):

    os.makedirs(os.path.join(save_directory, str(count)), exist_ok=True)

    top10_output = []
    top10_sents = top10['top10_sents']
    top10_preds = top10['top10_preds']
    top10_labels = top10['top10_labels']

    with open(os.path.join(save_directory, str(count), 'ActiveSentences.txt'), 'r') as in_f:
        active_lines = in_f.readlines()

    active_lines = [ele.split('\t')[0] for ele in active_lines]

    ignored_count = 0
    for i, ele in enumerate(top10_sents):
        if ele in active_lines:
            ignored_count += 1
            continue
        top10_output.append('\t'.join([ele, top10_preds[i], top10_labels[i]]))

    print('Deleted {} active sentences from top10 dataset'.format(ignored_count))

    with open(os.path.join(save_directory, str(count), 'top10.txt'), 'w') as out_f:
        out_f.write('\n'.join(top10_output))
    
    rest_output = []
    rest_sents = rest['sents']
    rest_labels = rest['labels']
    for i, ele in enumerate(rest_sents):
        rest_output.append('\t'.join([ele, rest_labels[i]]))

    with open(os.path.join(save_directory, str(count), 'rest.txt'), 'w') as out_f:
        out_f.write('\n'.join(rest_output))

def print_5(examples, label_type):

    print('='*5 + ' ' + label_type + ' ' + '='*5)
    examples = sorted(examples, key=lambda k: k[-1], reverse=True)
    to_print = min(5, len(examples))
    for i in range(to_print):
        print(examples[i][0])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine-tuning on additional sentences')
    parser.add_argument('--input_file', type=str, required=True,
                        help='file containing sentences and labels separated by a tab')
    parser.add_argument('--save_directory', type=str, required=True,
                        help='directory to save outputs to')
    args = parser.parse_args()

    os.makedirs(args.save_directory, exist_ok=True)

    classifier = pipeline('sentiment-analysis', 'cardiffnlp/twitter-roberta-base-sentiment', device=0)

    sentences, labels, hindi_scores = read_data(args.input_file, False, False)
    # top10, rest = gettop10(sentences, labels, classifier, False, 0, args.save_directory, 'class-wise', 1)
    top10, rest = gettop10(sentences, labels, hindi_scores, classifier, False, 0, args.save_directory, 'overall', 1)
    save_top10_rest(top10, rest, 0, args.save_directory)

    count = 1
    num_iterations = 30
    increment_factor = 1
    for count in range(1, num_iterations+1):
    # while rest and top10:

        print('Top10 Pred Distribution')
        pred_label_distribution = Counter(top10['top10_preds'])
        print(pred_label_distribution)

        positive_examples = [(ele, ele_score) for ele, ele_pred, ele_score in zip(top10['top10_sents'], top10['top10_preds'], top10['top10_scores']) if ele_pred == 'positive']
        negative_examples = [(ele, ele_score) for ele, ele_pred, ele_score in zip(top10['top10_sents'], top10['top10_preds'], top10['top10_scores']) if ele_pred == 'negative']
        neutral_examples = [(ele, ele_score) for ele, ele_pred, ele_score in zip(top10['top10_sents'], top10['top10_preds'], top10['top10_scores']) if ele_pred == 'neutral']

        print_5(positive_examples, 'positive')
        print_5(negative_examples, 'negative')
        print_5(neutral_examples, 'neutral')

        if count <= 10:
            num_epochs = 3
        elif count <= 20:
            num_epochs = 2
        else:
            num_epochs = 1

        increment_factor *= 1.2

        print('#'*10 + 'Iteration: {} Num epochs: {}'.format(count, num_epochs) + '#'*10)

        # fine_tune_fun(top10['top10_sents'], top10['top10_preds'], rest['sents'], rest['labels'], num_epochs, os.path.join(args.save_directory, str(count)))
        if count > 1:
            fine_tune_fun(top10['top10_sents'], top10['top10_preds'], rest['sents'], rest['labels'], num_epochs, os.path.join(args.save_directory, str(count-1)), os.path.join(args.save_directory, str(count)))
        else:
            fine_tune_fun(top10['top10_sents'], top10['top10_preds'], rest['sents'], rest['labels'], num_epochs, 'cardiffnlp/twitter-roberta-base-sentiment', os.path.join(args.save_directory, str(count)))

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.save_directory, str(count)))
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.save_directory, str(count)))
        classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0)

        # if count > 5 and count%5!=0:
        #     os.remove(os.path.join(args.save_directory, str(count), 'pytorch_model.bin'))
    
        if (count-1)%5!=0:
            os.remove(os.path.join(args.save_directory, str(count-1), 'pytorch_model.bin'))

        # top10, rest = gettop10(rest['sents'], rest['labels'], classifier, False, count, args.save_directory, 'class-wise', increment_factor)
        top10, rest = gettop10(rest['sents'], rest['labels'], rest['hindi_scores'], classifier, False, count, args.save_directory, 'overall', increment_factor)
        # for printing accuracy on entire dataset of the fine-tuned model
        # print('Printing accuracy for entire dataset')
        # all_top10, all_rest = gettop10(sentences, labels, classifier, False, 'class-wise')
        save_top10_rest(top10, rest, count, args.save_directory)
        # count += 1