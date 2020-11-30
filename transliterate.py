import argparse
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

def transliterate_file(input_file, output_file, script):

    if not script in ['Devanagari']:
        raise Exception('Script not supported')
    
    with open(input_file, 'r') as in_f:
        lines = in_f.readlines()

    output_lines = []
    for line in lines:
        line = line.split('\t')
        sentence, label = '\t'.join(line[:-1]), line[-1]
        transliterated_sentence = transliterate(sentence, sanscript.ITRANS, sanscript.DEVANAGARI)
        output_lines.append('\t'.join([transliterated_sentence, label]))

    with open(output_file, 'w') as out_f:
        out_f.write('\n'.join(output_lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transliterating dataset')
    parser.add_argument('--input_file', type=str, required=True,
                        help='file containing processed sentences and labels separated by a tab')
    parser.add_argument('--output_file', type=str, required=True,
                        help='file to write the transliterated output to')
    parser.add_argument('--script', type=str, default='Devanagari',
                        help='script to transliterate to')
    args = parser.parse_args()

    transliterate_file(args.input_file, args.output_file, args.script)
