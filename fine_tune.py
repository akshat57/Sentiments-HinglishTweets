## reference: https://mccormickml.com/2019/07/22/BERT-fine-tuning/#3-tokenization--input-formatting ##

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
rom transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import pdb

# useful snippets
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)
# tokenizer = AutoTokenizer.from_pretrained(save_directory)
# model = TFAutoModel.from_pretrained(save_directory, from_pt=True) # this is for tensorflow, for pytorch should be similar to original command i.e. using AutoModelForSequenceClassification
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def read_data_file(input_file, label_dict):

    with open(input_file, 'r') as in_f:
        lines = in_f.readlines()
    
    lines = [line.strip() for line in lines]

    sentences, labels = [], []
    for line in lines:
        line = line.split('\t')
        sentence = '\t'.join(line[:-1])
        label = line[-1]
        sentences.append(sentence)
        labels.append(label_dict[label])

    return sentences, labels

def tokenize_text(tokeinzer, sentences, labels):

    data_input_ids, data_attention_masks, output_labels = [], [], []

    #TODO: can this be optimized?
    max_num_tokens = 0
    for sent, label in zip(sentences, labels):
        tokenized_output = tokeinzer(sent)
        sent_input_ids = torch.tensor(tokenized_output['input_ids'])
        sent_attention_mask = torch.tensor(tokenized_output['attention_mask'])
        if len(sent_input_ids) > max_num_tokens:
            max_num_tokens = len(sent_input_ids)

    for sent, label in zip(sentences, labels):
        tokenized_output = tokeinzer(sent, max_length=max_num_tokens, pad_to_max_length = True)
        sent_input_ids = torch.tensor(tokenized_output['input_ids'])
        sent_attention_mask = torch.tensor(tokenized_output['attention_mask'])
        data_input_ids.append(sent_input_ids)
        data_attention_masks.append(sent_attention_mask)
        output_labels.append(label)
    
    pdb.set_trace()

    # Convert the lists into tensors.
    data_input_ids = torch.stack(data_input_ids, dim=0)
    data_attention_masks = torch.stack(data_attention_masks, dim=0)
    output_labels = torch.tensor(output_labels)

    return data_input_ids, data_attention_masks, output_labels

def load_data(tokenizer, sentences, labels, batch_size):

    input_ids, attention_masks, labels = tokenize_text(tokenizer, sentences, labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size 
        )
    
    validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )

    return train_dataloader, validation_dataloader

def train_epoch(train_dataloader):

    for step, batch in enumerate(train_dataloader):

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fine-tuning on additional sentences')
    parser.add_argument('--input_file', type=str, required=True,
                        help='file containing sentences and labels separated by a tab')
    args = parser.parse_args()

    label_dict = {'positive': 2, 'neutral': 1, 'negative': 0}
    sentences, labels = read_data_file(args.input_file, label_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = model.to(device)

    batch_size = 32
    train_dataloader, validation_dataloader = load_data(tokenizer, sentences, labels, batch_size)

    # set up optimizer, scheduler (?), and loss functions
    #TODO: took parameters from reference directly; figure out best setting
    optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    # set num_epochs, model.train(), and call training loop
    num_epochs = 5
    for i in range(num_epochs):
        train_epoch(train_dataloader)
        eval_epoch(validation_dataloader)

    pdb.set_trace()