import argparse
import preprocessor as p
import string
import pdb

p.set_options(p.OPT.URL, p.OPT.MENTION)

def preprocess_file(input_file, output_file, separate_langs):
    with open(input_file, 'r') as in_f:
        lines = in_f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    sentences, labels = [], []
    curr_sent, hin_sent, eng_sent = '', '', ''
    hindi_sentences, english_sentences = [], []
    for line in lines:
        words = line.split('\t')
        if line.startswith('meta'):
            if curr_sent:
                sentences.append(curr_sent.strip())
                hindi_sentences.append(hin_sent.strip())
                english_sentences.append(eng_sent.strip())
                curr_sent, hin_sent, eng_sent = '', '', ''
            labels.append(words[-1])
        else:
            words_string = ' '.join(words[:-1])
            # if (words_string in string.punctuation or '//' in words_string): #and not words_string == '@':
            if (words_string in ['//', '/', "'", '@']):
                curr_sent = curr_sent[:-1]
            else:
                words_string = ' '.join(words[:-1]) + ' '
            curr_sent += words_string
            if words[-1].startswith('Hin') or words[-1].startswith('O'):
                if (words_string in ['//', '/', "'", '@']): #and not words_string == '@':
                    hin_sent = hin_sent[:-1]
                hin_sent += words_string
            if words[-1].startswith('Eng') or words[-1].startswith('O'):
                if (words_string in ['//', '/', "'", '@']): #and not words_string == '@':
                    eng_sent = eng_sent[:-1]
                eng_sent += words_string
    if curr_sent:
        sentences.append(curr_sent.strip())
        hindi_sentences.append(hin_sent.strip())
        english_sentences.append(eng_sent.strip())
    output_lines, hindi_output_lines, english_output_lines = [], [], []
    for sent, label in zip(sentences, labels):
        output_lines.append('\t'.join([p.clean(sent), label]))
    for sent, label in zip(hindi_sentences, labels):
        hindi_output_lines.append('\t'.join([p.clean(sent), label]))
    for sent, label in zip(english_sentences, labels):
        english_output_lines.append('\t'.join([p.clean(sent), label]))
    with open(output_file, 'w') as out_f:
        out_f.write('\n'.join(output_lines))

    if separate_langs:
        output_file = output_file.split('.')
        hin_output_file = '.'.join(output_file[:-1]) + '_hindi.txt'
        eng_output_file = '.'.join(output_file[:-1]) + '_english.txt'
        with open(hin_output_file, 'w') as out_f:
            out_f.write('\n'.join(hindi_output_lines))
        with open(eng_output_file, 'w') as out_f:
            out_f.write('\n'.join(english_output_lines))

def main():
    parser = argparse.ArgumentParser(description='Preprocessing dataset')
    parser.add_argument('--raw_file', type=str, required=True,
                        help='file containing raw sentences and labels')
    parser.add_argument('--output_file', type=str, required=True,
                        help='file containing raw sentences and labels')
    parser.add_argument('--separate_langs', action='store_true',
                        help='Whether to separate sentences based on language identification. Raw file needs to have this information.')
    args = parser.parse_args()

    preprocess_file(args.raw_file, args.output_file, args.separate_langs)

if __name__ == '__main__':
    main()
