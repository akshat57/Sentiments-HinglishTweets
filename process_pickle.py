import argparse
import pickle

def load_pickle_file(input_file):
    with open(input_file, 'rb') as in_f:
        high_conf_tuples = pickle.load(in_f)

    return high_conf_tuples

def save_to_file(output_file, high_conf_tuples):

    output_sentences = []
    for ele in high_conf_tuples:
        sentence, pred = ele[0], ele[1]
        output_sentences.append('\t'.join([sentence, pred]))
    
    with open(args.output_file, 'w') as out_f:
        out_f.write('\n'.join(output_sentences))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing pickle files')
    parser.add_argument('--input_file', type=str, required=True,
                        help='pickle file containing high conf sentences')
    parser.add_argument('--output_file', type=str, required=True,
                        help='output text file')
    args = parser.parse_args()

    high_conf_tuples = load_pickle_file(args.input_file)

    save_to_file(args.output_file, high_conf_tuples)