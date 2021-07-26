from transformers import AutoTokenizer, AutoModel
import spacy
from tqdm import tqdm
from parsivar import Tokenizer, Normalizer
from process import parse_sentence

# from hazm import *
my_normalizer = Normalizer()
my_tokenizer = Tokenizer()


input_filename = "input.txt"
output_filename = "output.jsonl "
# language_model = "bert-base-cased"
if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained(language_model)
    # encoder = AutoModel.from_pretrained(language_model)
    # encoder.eval()

    with open(input_filename, 'r') as f, open(output_filename, 'w') as g:
        for idx, line in enumerate(tqdm(f)):
            sentence = line.strip()
            if len(sentence):
                sents = my_tokenizer.tokenize_sentences(my_normalizer.normalize(sentence))
                # sents = sent_tokenize(my_normalizer.normalize(sentence))
                for sent in sents:
                    print(sent)


