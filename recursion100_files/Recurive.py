import os
import time

#Create a new repository called iteration 0. Inside that create a repo called logs.
#It should contain 'processed_data_0.pkl': this is the entire dataset initially. This dataset could be binary or trinary class.

pos_ratio = 8
neg_ratio = 3
neu_ratio = 0
n_epochs = '3'

iter = '0'

#Make zero shot predictions 
#command = 'python3 prediction.py --iter ' + iter + ' --type zeroshot'
#print('Zero shot prediction:', command)
#os.system( command + ' > iteration' + iter + '/logs/prediction_zeroshot_log')

#Split data
#command = 'python3 datasplit.py --iter ' + iter + ' --positive ' + str(pos_ratio) + ' --negative ' + str(neg_ratio) + ' --neutral ' + str(neu_ratio)
#print('Split Data:', command)
#os.system(command + ' > iteration' + iter + '/logs/datasplit_logs')



for i in range(6, 20):
    start = time.time()
    iter = str(i)
   
    if i > 8:
        n_epochs = '2'

    print('='*20)
    print()
    print('Working on iteration ' + iter)
    
    #FINE TUNE MODEL
    command = 'python3 finetune.py --iter ' + iter + ' --n_epochs ' + n_epochs
    print('\nFine Tuning:', command)
    os.system(command + ' > iteration' + iter + '/logs/fine_tune_logs')

    #MAKIND PREDICTIONS
    
    #zero shot
    #command = 'python3 prediction.py --iter ' + iter + ' --type zeroshot'
    #print('\nZero shot prediction:', command)
    #os.system( command + ' > iteration' + iter + '/logs/prediction_zeroshot_log')

    #loaded predictions
    command = 'python3 prediction.py --iter ' + iter + ' --type load'
    print('\nLoaded prediction:', command)
    os.system( command + ' > iteration' + iter + '/logs/prediction_load_log')
    
    
    #SPLIT DATA
    command = 'python3 datasplit.py --iter ' + iter + ' --positive ' + str(pos_ratio) + ' --negative ' + str(neg_ratio) + ' --neutral ' + str(neu_ratio)
    print('\nSplit Data:', command)
    os.system(command + ' > iteration' + iter + '/logs/datasplit_logs')
    
    
    print('Time for 1 epoch:', time.time() - start)
    print()

