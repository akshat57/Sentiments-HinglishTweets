import argparse
import pickle
import numpy as np
import os
import random
from collections import Counter

def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()
    

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def count(data):
    predicted_labels = [pred_label for (sentence, pred_label, label, score) in data]
    n_pos = 0
    n_neg = 0
    n_neu = 0

    for label in predicted_labels:
        if label == 'positive':
            n_pos += 1
        elif label == 'negative':
            n_neg += 1
        elif label == 'neutral':
            n_neu += 1

    return n_pos, n_neg, n_neu


def find_label_5(pred_label):
    if pred_label in ['1 star', '2 stars']:
        pred_label = 'negative'

    elif pred_label in ['4 stars', '5 stars']:
        pred_label = 'positive'

    else:
        pred_label = 'neutral'
    
    return pred_label


def find_label_2(pred_label, predicted_labels):
    negative = predicted_labels[0]
    positive = predicted_labels[1]

    if pred_label == negative:
        pred_label = 'negative'
    elif pred_label == positive:
        pred_label = 'positive'

    return pred_label


def find_label_1(pred_label, predicted_labels):
    if predicted_labels[0] in ['1 star', '2 stars']:
        pred_label = 'negative'

    elif predicted_labels[0] in ['4 stars', '5 stars']:
        pred_label = 'positive'

    else:
        pred_label = 'neutral'

    return pred_label


def handle_mbert(data):
    predicted_labels = [pred_label for (sentence, pred_label, label, score) in data]
    print(Counter(predicted_labels))
                            
    predicted_labels = list(set(predicted_labels))
    predicted_labels.sort()
    print(predicted_labels)

    new_data = []    
    for (sentence, pred_label, label, score) in data:
        if len(predicted_labels) >= 3:
            pred_label = find_label_5(pred_label)

        elif len(predicted_labels) == 2:
            pred_label = find_label_2(pred_label, predicted_labels)

        elif len(predicted_labels) == 1:
            pred_label = find_label_1(pred_label, predicted_labels)

        new_data.append((sentence, pred_label, label, score))

    return new_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Datasplit')
    parser.add_argument('--iter', type=str, required=True, help='Enter Iteration number')
    parser.add_argument('--positive', type=int, required=True, help='Enter percentage of data to be split for a class')
    parser.add_argument('--negative', type=int, required=True, help='Enter percentage of data to be split for a class')
    parser.add_argument('--neutral', type=int, required=True, help='Enter percentage of data to be split for a class')
    args = parser.parse_args()
    
    print('Entered', args.iter)
    
    current_directory = 'iteration' + args.iter + '/'
    if args.iter == '0':
        input_file = 'iteration' + args.iter + '/iteration_' + args.iter + '_zeroshot.pkl'
    else:
        input_file = 'iteration' + args.iter + '/iteration_' + args.iter + '.pkl'
        
    #Load and count data
    data = load_data(input_file)

    ###handling 5 labels
    if True:#only when using mbert
        data = handle_mbert(data)
    
    n_pos, n_neg, n_neu = count(data)
    
    #find overall accuracy:
    overall_accuracy = np.array([pred_label == label for (sentence, pred_label, label, score) in data])
    print('Overall Accuracy:', np.sum(overall_accuracy)/len(data))
    
    #Store in different dictionaries based on actual label
    pred_positive = []
    pred_negative = []
    pred_neutral = []
    for (sentence, pred_label, label, score) in data:
        if pred_label == 'positive':
            pred_positive.append((sentence, pred_label, label, score))
        elif pred_label == 'negative':
            pred_negative.append((sentence, pred_label, label, score))
        elif pred_label == 'neutral':
            pred_neutral.append((sentence, pred_label, label, score))
        
    print('Check:', (len(pred_positive) + len(pred_negative) + len(pred_neutral)) == len(data))
    print('Total Dataset:', len(data), 'Positive:', len(pred_positive), 'Negative:', len(pred_negative), 'Neutral:', len(pred_neutral))
    
    #Original Accuracy for predictions made
    positive_accuracy = np.array([pred_label == label for i, (sentence, pred_label, label, score) in enumerate(pred_positive)])
    negative_accuracy = np.array([pred_label == label for i, (sentence, pred_label, label, score) in enumerate(pred_negative)])
    neutral_accuracy = np.array([pred_label == label for i, (sentence, pred_label, label, score) in enumerate(pred_neutral)])

    print('Positive Accuray:', np.sum(positive_accuracy)/len(pred_positive))
    print('Negative Accuracy:', np.sum(negative_accuracy)/len(pred_negative))
    print('Neutral Accuracy:', np.sum(neutral_accuracy)/len(pred_neutral))

    
    ##Sorting by predicting confidence
    sorted_pred_positive = sorted(pred_positive, key=lambda k: k[3], reverse=True)
    sorted_pred_negative = sorted(pred_negative, key=lambda k: k[3], reverse=True)
    sorted_pred_neutral = sorted(pred_neutral, key=lambda k: k[3], reverse=True)
    
    
    ##Total number of samples in top10% high confidence predictions for each class
    n_top10_positive = min(400, n_pos) #(n_pos * args.positive)//100
    n_top10_negative = min(400, n_neg)#(n_neg * args.negative)//100
    n_top10_neutral = 0#(n_neu * args.neutral)//100
    
    print('Number chosen:', n_top10_positive, n_top10_negative, n_top10_neutral)
    
    
    #Find top 10 percent accuracy
    positive_top10_accuracy = np.array([pred_label == label for i, (sentence, pred_label, label, score) in enumerate(sorted_pred_positive) if i < n_top10_positive ])
    negative_top10_accuracy = np.array([pred_label == label for i, (sentence, pred_label, label, score) in enumerate(sorted_pred_negative) if i < n_top10_negative ])
    neutral_top10_accuracy = np.array([pred_label == label for i, (sentence, pred_label, label, score) in enumerate(sorted_pred_neutral) if i < n_top10_neutral ])

    print('Top10 Confidence Positive Accuray:', np.sum(positive_top10_accuracy)/n_top10_positive)
    print('Top10 Confidence Negative Accuracy:', np.sum(negative_top10_accuracy)/n_top10_negative)
    print('Top10 Confidence Neutral Accuracy:', np.sum(neutral_top10_accuracy)/n_top10_neutral)

    
    #Pick top chosen percent
    top_confidence_positive = sorted_pred_positive[:n_top10_positive]
    top_confidence_negative = sorted_pred_negative[:n_top10_negative]
    top_confidence_neutral = sorted_pred_neutral[:n_top10_neutral]
    
    #create new train data set that does not have saved sentences
    new_directory = 'iteration' + str(int(args.iter) + 1)
    os.mkdir(new_directory)
    os.mkdir(new_directory + '/logs')
    
    fine_tune_data = top_confidence_positive + top_confidence_negative + top_confidence_neutral
    random.shuffle(fine_tune_data)
    print('Check:', len(fine_tune_data) == len(top_confidence_positive) + len(top_confidence_negative) + len(top_confidence_neutral))
    save_data(new_directory + '/fine_tune_' + str(int(args.iter) + 1) + '.pkl', fine_tune_data)
    
    #Collecting chosen sentences and removing them from original dataset, and saving
    chosen_sentences = [sentence for (sentence, pred_label, label, score) in fine_tune_data]
    original_dataset = load_data(current_directory + 'processed_data_' + args.iter + '.pkl')
    
    next_iteration_data = []
    for (tweet_id, tweet, label) in original_dataset:
        if tweet not in chosen_sentences:
            next_iteration_data.append((tweet_id, tweet, label))
            
    save_data( new_directory + '/processed_data_' + str(int(args.iter) + 1) + '.pkl', next_iteration_data)

    
