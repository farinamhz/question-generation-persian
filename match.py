from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForTokenClassification
import spacy
from tqdm import tqdm
from parsivar import Tokenizer, Normalizer
from process import parse_sentence

# from hazm import *
my_normalizer = Normalizer()
my_tokenizer = Tokenizer()

input_filename = "input.txt"
output_filename = "output.jsonl "
language_model = "bert-base-cased"
model_name_or_path = "HooshvareLab/distilbert-fa-zwnj-base-ner"

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
    # encoder = AutoModel.from_pretrained(model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(language_model)
    encoder = AutoModel.from_pretrained(language_model)

    encoder.eval()
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    with open(input_filename, 'r') as f, open(output_filename, 'w') as g:
        for idx, line in enumerate(tqdm(f)):
            sentence = line.strip()
            if len(sentence):
                valid_triplets = []
                all_disamb_ents = dict()
                sents = my_tokenizer.tokenize_sentences(my_normalizer.normalize(sentence))
                # sents = sent_tokenize(my_normalizer.normalize(sentence))
                for sent in sents:
                    print(sent)
                    # triplets_lst, disamb_ents = parse_sentence(sent, tokenizer, nlp)
                    triplets_lst = parse_sentence(sent, tokenizer, encoder, nlp)
                    print("triplets_lst", triplets_lst)
                    for triplets in triplets_lst:
                        valid_triplets.append(triplets)
                if len(valid_triplets) > 0:
                    for triplet in valid_triplets:
                        head = triplet['h']
                        tail = triplet['t']
                        relations = triplet['r']
                    print("head:\t", head, "relations:\t", relations, "tail:\t", tail)
                    # exit()
                    # output = {'line': idx, 'tri': deduplication(mapped_triplets)}
