import os
import time
import pickle
import numpy as np

#Create a new repository called iteration 0. Inside that create a repo called logs.
#It should contain 'processed_data_0.pkl': this is the entire dataset initially. This dataset could be binary or trinary class.

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

check_dir = 'check'
#os.mkdir(check_dir)

f = open(check_dir + "/accuracies.txt", "a")


for i in range(0,1):
    start = time.time()
    iter = str(i)

    #MAKIND PREDICTIONS

    #loaded predictions
    command = 'python3 prediction_finalcheck.py --iter ' + iter + ' --output ' + check_dir
    print('\nLoaded prediction:', command)
    os.system( command + ' > ' + check_dir + '/checking_log')
    
    #FIND ACCURACY
    pred_file = check_dir + '/iteration_' + iter + '.pkl'
    data = load_data(pred_file)
    
    #find overall accuracy:
    overall_accuracy = np.array([pred_label == label for (tweet_id, sentence, pred_label, label, score) in data])
    overall_accuracy = np.sum(overall_accuracy)/len(data)
    print('Overall Accuracy:', overall_accuracy)
    f.write(iter + ' : ' + str(overall_accuracy) + '\n')
        
    positive_accuracy = np.array([pred_label == label for (tweet_id, sentence, pred_label, label, score) in data if label == 'positive'])
    positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
    print('Positive Accuracy:', positive_accuracy)
    f.write(iter + ' : ' + str(positive_accuracy) + '\n')

    positive_accuracy = np.array([pred_label == label for (tweet_id, sentence, pred_label, label, score) in data if label == 'negative'])
    positive_accuracy = np.sum(positive_accuracy)/positive_accuracy.shape[0]
    print('Negative Accuracy:', positive_accuracy)
    f.write(iter + ' : ' + str(positive_accuracy) + '\n\n')


    print('Time:', time.time() - start)
    print('Time to break the loop begins:')
    time.sleep(5)
    print('Time to break the loop ends')
            
f.close()

