import json
import time
from transformers import AutoConfig, AutoTokenizer, AutoModel, TFAutoModel, AutoModelForMaskedLM
import spacy
from tqdm import tqdm
from parsivar import Tokenizer, Normalizer
from process import parse_sentence, deduplication2

# from process import parse_sentence, deduplication

# from hazm import *
my_normalizer = Normalizer()
my_tokenizer = Tokenizer()

input_filename = "input.txt"
output_filename = "output.json"
language_model = "bert-base-cased"


# model_name_or_path = "HooshvareLab/bert-fa-zwnj-base"
# model_name_or_path = "bert-fa-zwnj-base"
# config = AutoConfig.from_pretrained(model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# model_name_or_path = "HooshvareLab/distilbert-fa-zwnj-base-ner"


if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # encoder = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
    encoder = AutoModel.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
    # tokenizer = AutoTokenizer.from_pretrained(language_model)
    # encoder = AutoModel.from_pretrained(language_model)

    encoder.eval()
    with open(input_filename, 'r') as f, open(output_filename, 'w', encoding='utf_8') as g:
        for idx, line in enumerate(tqdm(f)):
            sentence = line.strip()
            if len(sentence):
                valid_triplets = []
                sents = my_tokenizer.tokenize_sentences(my_normalizer.normalize(sentence))
                for sent in sents:
                    # print(sent)

                    triplets_lst = parse_sentence(sent, tokenizer, encoder)
                    for triplets in triplets_lst:
                        valid_triplets.append(triplets)
                if len(valid_triplets) > 0:
                    output_tri = []
                    for a in valid_triplets:
                        if a['c'] > 0.05:
                            output_tri.append(a)
                    # after_d1 = deduplication(output_tri)
                    output = {'line': idx, 'tri': deduplication2(output_tri)}
                    g.write(str(output) + '\n')
