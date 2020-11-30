import argparse
import pdb

def preprocess_file(input_file, output_file):

    with open(input_file, 'r') as in_f:
        lines = in_f.readlines()

    lines = [line.strip() for line in lines]

    output_sentences, output_labels = [], []
    for line in lines:
        line = line.split('\t')
        sentence, label = '\t'.join(line[:-1]), line[-1]
        if label == '0':
            output_label = 'negative'
        elif label == '1':
            output_label = 'neutral'
        elif label == '2':
            output_label = 'positive'
        else:
            raise Exception('Label mapping doesn\'t exist')
        output_sentences.append(sentence)
        output_labels.append(output_label)
    
    output_lines = ['\t'.join([sent, label]) for sent, label in zip(output_sentences, output_labels)]
    with open(output_file, 'w') as out_f:
        out_f.write('\n'.join(output_lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing IITH text files')
    parser.add_argument('--input_file', type=str, required=True,
                        help='file containing sentences and integer labels separated by a tab')
    parser.add_argument('--output_file', type=str, required=True,
                        help='path to store the preprocessed output to')
    args = parser.parse_args()

    preprocess_file(args.input_file, args.output_file)