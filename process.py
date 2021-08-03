from utils import compress_attention, create_mapping, BFS, build_graph
from multiprocessing import Pool

import torch


# from utils import create_mapping, compress_attention, build_graph


def process_matrix(attentions, layer_idx=-1, head_num=0, avg_head=False, trim=True):
    if avg_head:
        attn = torch.mean(attentions[0][layer_idx], 0)
        attention_matrix = attn.detach().numpy()
    else:
        attn = attentions[0][layer_idx][head_num]
        attention_matrix = attn.detach().numpy()

    attention_matrix = attention_matrix[1:-1, 1:-1]

    return attention_matrix


def bfs(args):
    s, end, graph, max_size, black_list_relation = args
    return BFS(s, end, graph, max_size, black_list_relation)


def check_relations_validity(relations):
    for rel in relations:
        # if rel in invalid_relations_set or rel.isnumeric():
        if rel in rel.isnumeric():
            return False
    return True


def global_initializer(nlp_object):
    global pars_nlp
    pars_nlp = nlp_object


def filter_relation_sets(params):
    triplet, id2token, verbs, token2id = params
    # print("xxx triplet", triplet)
    triplet_idx = triplet[0]
    confidence = triplet[1]
    head, tail = triplet_idx[0], triplet_idx[-1]
    if head in id2token and tail in id2token:
        head = id2token[head]
        # print("xxx head", head)
        tail = id2token[tail]
        # print("xxx tail", tail)
        # relations = [id2token[idx] for idx in triplet_idx[1:-1] if idx in id2token]

        pre_relations = [n for n in verbs if token2id[n] - token2id[tail] > 0]  # and token2id[n] == near_verb_id]
        relations = []
        near_verb_id = min([token2id[n] for n in pre_relations])
        relations.append(id2token[near_verb_id])
        # print(relations)
        if len(relations) > 0:
            return {'h': head, 't': tail, 'r': relations, 'c': confidence}
    return {}


def parse_sentence(sentence, tokenizer, encoder):
    '''Implement the match part of MAMA

    '''
    verbs, inputs, tokenid2word_mapping, token2id, noun_chunks = create_mapping(sentence, return_pt=True,
                                                                                tokenizer=tokenizer,
                                                                                )
    # print("tokenid2word_mapping", tokenid2word_mapping)
    # print("token2id", token2id)
    # print("verbs", verbs)
    with torch.no_grad():
        outputs = encoder(**inputs, output_attentions=True)
    trim = True

    attention = process_matrix(outputs[2], avg_head=True, trim=trim)
    #
    merged_attention = compress_attention(attention, tokenid2word_mapping)
    attn_graph = build_graph(merged_attention)

    tail_head_pairs = []
    # print(token2id)
    for head in noun_chunks:
        for tail in noun_chunks:
            if head != tail and tail not in verbs:
                tail_head_pairs.append((token2id[head], token2id[tail]))

    # print("tail_head_pairs", tail_head_pairs)

    # black_list_relation = set([token2id[n] for n in token2id if n not in verbs])
    # black_list_relation = set([token2id[n] for n in verbs])

    all_relation_pairs = []
    id2token = {value: key for key, value in token2id.items()}

    black_list_relation = set([token2id[n] for n in id2token.values() if n not in verbs])

    # ##################################
    #     for a in black_list_relation:
    #         print("block_rels", id2token[a])

    #     print("black_list_relation", black_list_relation)

    # for a in tail_head_pairs:
    #     print(id2token[a[0]], id2token[a[1]])

    # ###################################

    with Pool(10) as pool:
        params = [(pair[0], pair[1], attn_graph, max(tokenid2word_mapping), black_list_relation,) for pair in
                  tail_head_pairs]
        for output in pool.imap_unordered(bfs, params):
            if len(output):
                all_relation_pairs += [(o, id2token, verbs, token2id) for o in output]

    # print("all_relation_pairs", all_relation_pairs)

    triplet_text = []

    with Pool(10) as pool:
        for triplet in pool.imap_unordered(filter_relation_sets, all_relation_pairs):
            if len(triplet) > 0:
                triplet_text.append(triplet)

    return triplet_text
    # return triplet_text, disamb_ents


def deduplication(triplets):
    unique_pairs = []
    pair_confidence = []
    for t in triplets:
        key = '{}\t{}\t{}'.format(t['h'], t['r'], t['t'])
        conf = t['c']
        if key not in unique_pairs:
            unique_pairs.append(key)
            pair_confidence.append(conf)

    unique_triplets = []
    for idx, unique_pair in enumerate(unique_pairs):
        h, r, t = unique_pair.split('\t')
        unique_triplets.append({'h': h, 'r': r, 't': t, 'c': pair_confidence[idx]})