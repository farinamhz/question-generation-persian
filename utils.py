from collections import defaultdict
import codecs
import numpy as np
import torch
from copy import copy
from parsivar import Tokenizer, Normalizer, FindChunks, POSTagger
from parsivar import FindStems
import re

my_normalizer = Normalizer()
my_tokenizer = Tokenizer()
my_stemmer = FindStems()
my_tagger = POSTagger(tagging_model="wapiti")
my_chunker = FindChunks()
stop_words = "\n".join(sorted(list(
    set([my_normalizer.normalize(w) for w in codecs.open('nonverbal', encoding='utf-8').read().split('\n') if w]))))


def build_graph(matrix):
    graph = defaultdict(list)

    for idx in range(0, len(matrix)):
        for col in range(idx + 1, len(matrix)):
            graph[idx].append((col, matrix[idx][col]))
    return graph


def BFS(s, end, graph, max_size=-1, black_list_relation=[]):
    visited = [False] * (max(graph.keys()) + 100)

    # Create a queue for BFS
    queue = []

    # Mark the source node as
    # visited and enqueue it
    queue.append((s, [(s, 0)]))

    found_paths = []

    visited[s] = True

    while queue:

        s, path = queue.pop(0)

        # Get all adjacent vertices of the
        # dequeued vertex s. If a adjacent
        # has not been visited, then mark it
        # visited and enqueue it
        for i, conf in graph[s]:
            if i == end:
                found_paths.append(path + [(i, conf)])
                break
            if visited[i] == False:
                queue.append((i, copy(path) + [(i, conf)]))
                visited[i] = True

    candidate_facts = []
    for path_pairs in found_paths:
        # if len(path_pairs) < 3: #####
        #     continue
        path = []
        cum_conf = 0
        for (node, conf) in path_pairs:
            path.append(node)
            cum_conf += conf

        # if path[1] in black_list_relation:
        #     continue

        candidate_facts.append((path, cum_conf))

    candidate_facts = sorted(candidate_facts, key=lambda x: x[1], reverse=True)

    return candidate_facts


def create_mapping(sentence, return_pt=False, tokenizer=None):
    '''Create a mapping
        tokenizer: huggingface tokenizer
    '''

    # for ch in "1234567890&;#$()*,-./:[]«»؛،؟۰۱۲۳۴۵۶۷۸۹!":
    #     sentence = sentence.replace(ch, "")
    text = sentence
    normalized_text = my_normalizer.normalize(text)
    # print("text: ", normalized_text)
    text = normalized_text.replace("‌", " ")
    tokens = my_tokenizer.tokenize_words(text)
    # print("tokens: ", tokens)

    # for token in tokens:
    #     token = my_stemmer.convert_to_stem(token)
    # print(tokens)
    # print("text :", text)
    text_tags = my_tagger.parse(my_tokenizer.tokenize_words(text))
    chunks = my_chunker.chunk_sentence(text_tags)
    result = my_chunker.convert_nestedtree2rawstring(chunks)

    # print("tagged result", result)
    # print("results before del puncs: ", result)

    for ch in ",-.«»؛،؟۰!":
        result = result.replace(ch, "")

    # print("results after del puncs: ", result)
    # ###################################################3
    # verbs
    result_items_v = []
    verbs = []
    s_v = ""
    i_v = 0
    while i_v < len(result):
        if result[i_v] != "[":
            i_v += 1
        else:
            while result[i_v] != "]":
                i_v += 1
                s_v += result[i_v]
            result_items_v.append(s_v.strip(" ]"))
            s_v = ""
    # print("result_items_v", result_items_v)
    for a in result_items_v:
        if a.find("V") != -1:
            # print("A", a)
            for ch in "VPND":
                a = a.replace(ch, "")
            verbs.append(a.strip(" "))

    # for i in range(len(verbs)):
    #     tokens1 = verbs[i].split(" ")
    #     tokens_filtered = [word for word in tokens1 if word not in stop_words]
    #     verbs[i] = " ".join(tokens_filtered)
    # ##################################################

    for ch in "VPND":
        result = result.replace(ch, "")

    result_items = []
    s = ""
    i = 0
    while i < len(result):
        if result[i] != "[":
            i += 1
        else:
            while result[i] != "]":
                i += 1
                s += result[i]
            result_items.append(s.strip(" ]"))
            s = ""
    # print("result_items", result_items)
    # print("results before del nonverbals: ", result_items)
    for i in range(len(result_items)):
        if result_items[i] in verbs:
            continue
        tokens1 = result_items[i].split(" ")
        tokens_filtered = [word for word in tokens1 if word not in stop_words]
        result_items[i] = " ".join(tokens_filtered)
    # print("results after del nonverbals: ", result_items)
    for a in result_items:
        if a == '':
            result_items.remove('')
        elif a == ' ':
            result_items.remove(' ')
    # print([(idx, token) for idx, token in enumerate(tokens)])
    # print("result_items", result_items)
    doc = []
    first_tokens = []
    size_tokens = []
    for a in result_items:
        chunk_tokens = a.split(" ")
        first_tokens.append(chunk_tokens[0])
        size_tokens.append(len(chunk_tokens))

    j = 0
    len_tokens = len(tokens)
    # print("first_tokens", first_tokens)
    # print("tokens")
    for i in range(len(first_tokens)):
        while j < len_tokens and first_tokens[i] != tokens[j]:
            j += 1
        if j == len_tokens:
            break
        context = dict()
        context["word"] = result_items[i]
        context["start"] = j
        context["end"] = j + size_tokens[i]
        doc.append(context)
    # print("doc", doc)

    chunk2id = {}

    start_chunk = []
    end_chunk = []
    noun_chunks = []

    ner_ranges = list()

    for chunk in doc:
        noun_chunks.append(chunk["word"])
        start_chunk.append(chunk["start"])
        end_chunk.append(chunk["end"])

    sentence_mapping = []
    token2id = {}
    mode = 0  # 1 in chunk, 0 not in chunk
    chunk_id = 0
    # print("verbs", verbs)
    # print("noun_chunks:", noun_chunks)

    for idx, token in enumerate(tokens):
        # print("idx", idx)
        # print("token", token)
        # print(token.text)
        if idx in start_chunk:
            # print("noun_chunks[chunk_id]", noun_chunks[chunk_id])
            mode = 1
            sentence_mapping.append(noun_chunks[chunk_id])
            token2id[sentence_mapping[-1]] = len(token2id)
            chunk_id += 1
        elif idx in end_chunk:
            mode = 0

        if mode == 0:
            # sentence_mapping.append(token.text)
            sentence_mapping.append(tokens[idx])
            token2id[sentence_mapping[-1]] = len(token2id)

    # print("token2id", token2id)

    token_ids = []
    tokenid2word_mapping = []

    for token in sentence_mapping:
        subtoken_ids = tokenizer(str(token), add_special_tokens=False)['input_ids']
        # print("token before tokenizer", token)
        # print("subtoken_ids", subtoken_ids)
        tokenid2word_mapping += [token2id[token]] * len(subtoken_ids)
        token_ids += subtoken_ids
    # print(tokenid2word_mapping)
    tokenizer_name = str(tokenizer.__str__)

    outputs = {
        'input_ids': [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id],
        'attention_mask': [1] * (len(token_ids) + 2),
        'token_type_ids': [0] * (len(token_ids) + 2)
    }

    if return_pt:
        for key, value in outputs.items():
            outputs[key] = torch.from_numpy(np.array(value)).long().unsqueeze(0)

    return verbs, outputs, tokenid2word_mapping, token2id, noun_chunks
    # return outputs, tokenid2word_mapping, token2id, noun_chunks, linked_entities


def compress_attention(attention, tokenid2word_mapping, operator=np.mean):
    new_index = []

    prev = -1
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append([row])
            prev = token_id
        else:
            new_index[-1].append(row)

    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

    attention = np.array(new_matrix).T

    prev = -1
    new_index = []
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append([row])
            prev = token_id
        else:
            new_index[-1].append(row)

    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

    return new_matrix.T
