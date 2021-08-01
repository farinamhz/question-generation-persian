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
    triplet, id2token = params

    triplet_idx = triplet[0]
    confidence = triplet[1]
    head, tail = triplet_idx[0], triplet_idx[-1]
    if head in id2token and tail in id2token:
        head = id2token[head]
        tail = id2token[tail]
        relations = [pars_nlp(id2token[idx])[0].lemma_ for idx in triplet_idx[1:-1] if idx in id2token]
        if len(relations) > 0 and check_relations_validity(
                relations):
            # and head not in invalid_relations_set and tail not in invalid_relations_set:
            return {'h': head, 't': tail, 'r': relations, 'c': confidence}
    return {}


def parse_sentence(sentence, tokenizer, encoder, nlp):
    '''Implement the match part of MAMA

    '''
    inputs, tokenid2word_mapping, token2id, noun_chunks = create_mapping(sentence, return_pt=True,
                                                                         nlp=nlp,
                                                                         tokenizer=tokenizer,
                                                                         )
    print("tokenid2word_mapping", tokenid2word_mapping)
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
            if head != tail:
                tail_head_pairs.append((token2id[head], token2id[tail]))

    black_list_relation = set([token2id[n] for n in noun_chunks])

    all_relation_pairs = []
    id2token = {value: key for key, value in token2id.items()}

    with Pool(10) as pool:
        params = [(pair[0], pair[1], attn_graph, max(tokenid2word_mapping), black_list_relation,) for pair in
                  tail_head_pairs]
        for output in pool.imap_unordered(bfs, params):
            if len(output):
                all_relation_pairs += [(o, id2token) for o in output]

    triplet_text = []

    with Pool(10, global_initializer, (nlp,)) as pool:
        for triplet in pool.imap_unordered(filter_relation_sets, all_relation_pairs):
            if len(triplet) > 0:
                triplet_text.append(triplet)

    return triplet_text,
    # return triplet_text, disamb_ents
