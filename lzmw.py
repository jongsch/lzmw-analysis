import copy
import math
from functools import lru_cache

import pylru
from collections import Counter, defaultdict
from random import shuffle

import nltk
nltk.download('brown')
from nltk.corpus import brown

def find_longest_match(sequence_string, position, codebook, max_codeword_len=None):
    i = position + 1
    match_str = ''
    if max_codeword_len is None:
        max_codeword_len = max(len(k) for k in codebook)
    while i <= min(
            len(sequence_string),
            position + max_codeword_len
    ):
        test_str = sequence_string[position: i]
        # print(f"test_str: {test_str}")
        if test_str in codebook:
            match_str = test_str
        i += 1
    return match_str


def update_lru(codebook, alphabet, match_str):
    alphabet = set(alphabet)
    # codebook[match_str] = 0
    for codeword in codebook:
        if codeword in alphabet:
            continue
        if match_str == codeword:
        # if match_str.startswith(codeword):
        # if match_str == codeword \
        #         or (match_str.startswith(codeword) and match_str[len(codeword):] in codebook) \
        #         or (match_str.endswith(codeword) and match_str[:len(codeword)] in codebook):
            codebook[codeword] = 0
        else:
            codebook[codeword] += 1


def find_lru_codeword(codebook, alphabet):
    alphabet = set(alphabet)
    dynamic_codebook_keys = [k for k in codebook if k not in alphabet]
    lru_num = 0
    lru_codeword = ''
    for codeword in dynamic_codebook_keys:
        if (codebook[codeword] > lru_num) \
                or (codebook[codeword] == lru_num and len(codeword) < len(lru_codeword)):
            lru_num = codebook[codeword]
            lru_codeword = codeword
    codebook.pop(lru_codeword)
    return lru_codeword


def lzmw(sequence_str, r=10, max_chars=None, codebook=None, alphabet=None):
    # following paper notation, codebook size is 2 ** r
    if max_chars is None:
        max_chars = len(sequence_str)
    max_codebook_size = 2 ** r
    if codebook is not None:
        if len(codebook) > max_codebook_size:
            raise Exception("input codebook greater than max size")
        if alphabet is None:
            raise Exception("must specify alphabet when using a pre-existing codebook")
    else:
        codebook = {}
        alphabet = set(sequence_str)
    if len(codebook) == 0:
        for i, character in enumerate(alphabet):
            codebook[character] = [[], 0]
    for i, character in enumerate(alphabet):
        codebook[character] = 0
    position = 0
    previous_match_str = ''
    while position < min(len(sequence_str), max_chars):
        if position - len(previous_match_str) < round(position, -4) and position >= round(position, -4):
            print(f"position: {position} of {len(sequence_str)}")
        if position - len(previous_match_str) < round(position, -5) and position >= round(position, -5):
            print(f"codebook: ")
            for kv in sorted({k: v for k, v in codebook.items() if k not in alphabet}.items(),
                             key=lambda item: (item[1]))[:31]:
                print(kv[0], kv[1])
        match_str = find_longest_match(sequence_str, position, codebook)
        concat_match_str = previous_match_str + match_str
        update_lru(codebook, alphabet, match_str)
        if concat_match_str not in codebook:
            if len(codebook) == max_codebook_size:
                lru_codeword = find_lru_codeword(codebook, alphabet)
            codebook[concat_match_str] = 0
        previous_match_str = match_str
        position += len(match_str)
    return codebook


def lzap(sequence_str, r=10, max_chars=None, codebook=None, alphabet=None):
    # following paper notation, codebook size is 2 ** r
    if max_chars is None:
        max_chars = len(sequence_str)
    max_codebook_size = 2 ** r
    if codebook is not None:
        if len(codebook) > max_codebook_size:
            raise Exception("input codebook greater than max size")
        if alphabet is None:
            raise Exception("must specify alphabet when using a pre-existing codebook")
    else:
        codebook = {}
        alphabet = set(sequence_str)
    if len(codebook) == 0:
        for i, character in enumerate(alphabet):
            codebook[character] = [[], 0]
    position = 0
    previous_match_str = ''
    # following paper notation, codebook size is 2 ** r
    if max_chars is None:
        max_chars = len(sequence_str)
    alphabet = set(sequence_str)
    codebook = {}
    max_codebook_size = 2 ** r
    for i, character in enumerate(alphabet):
        codebook[character] = 0
    position = 0
    previous_match_str = ''
    while position < min(len(sequence_str), max_chars):
        if position - len(previous_match_str) < round(position, -4) and position >= round(position, -4):
            print(f"position: {position} of {len(sequence_str)}")
        if position - len(previous_match_str) < round(position, -5) and position >= round(position, -5):
            print(f"codebook: ")
            for kv in sorted({k: v for k, v in codebook.items() if k not in alphabet}.items(),
                             key=lambda item: (item[1]))[:31]:
                print(kv[0], kv[1])
        match_str = find_longest_match(sequence_str, position, codebook)
        update_lru(codebook, alphabet, match_str)
        concat_match_str = previous_match_str + match_str
        if concat_match_str not in codebook:
            for i in range(len(match_str)):
                if len(codebook) == max_codebook_size:
                    lru_codeword = find_lru_codeword(codebook, alphabet)
                # print(f"removed lru codeword: {lru_codeword}")
                codebook[previous_match_str + match_str[:i + 1]] = 0
        previous_match_str = match_str
        position += len(match_str)
    return codebook


def update_lru_reference_counts(codebook, alphabet, match_str, previous_match_str):
    alphabet = set(alphabet)
    dynamic_codebook_keys = [k for k in codebook if k not in alphabet]
    # reference_count, steps_since_used
    for codeword in codebook:
        if codeword in (match_str, previous_match_str):
            codebook[codeword][0].add(previous_match_str + match_str)
            codebook[codeword][1] = 0
        else:
            if len(codebook[codeword][0]) == 0:
                codebook[codeword][1] += 1


def find_lru_codeword_ref_counts(codebook, alphabet):
    alphabet = set(alphabet)
    # dynamic_codebook_keys = [k for k in codebook if k not in alphabet]
    lru_num = 0
    lru_codeword = ''
    for codeword in codebook:
        if codeword in alphabet:
            continue
        if len(codebook[codeword][0]) == 0 and \
                (
                        codebook[codeword][1] > lru_num
                ) or (
                codebook[codeword][1] == lru_num and len(codeword) < len(lru_codeword)
        ):
            lru_num = codebook[codeword][1]
            lru_codeword = codeword
    codebook.pop(lru_codeword)
    for codeword in codebook:
        if codeword == lru_codeword or codeword in alphabet:
            continue
        if lru_codeword in codebook[codeword][0]:
            codebook[codeword][0].remove(lru_codeword)
    return lru_codeword


def lzmw_ref_counts(sequence_str, r=10, max_chars=None, intermediate_results=None, alphabet=None):
    # following paper notation, codebook size is 2 ** r
    if max_chars is None:
        max_chars = len(sequence_str)
    max_codebook_size = 2 ** r
    if intermediate_results is not None:
        codebook, references, reference_counts, reverse_ref_counts, lru_cache, codeword_lens = intermediate_results
        if len(codebook) > max_codebook_size:
            raise Exception("input codebook greater than max size")
        if alphabet is None:
            raise Exception("must specify alphabet when using a pre-existing codebook")
        alphabet = set(alphabet)
    else:
        codebook = {}
        if alphabet is None:
            alphabet = set(sequence_str)
        lru_cache = pylru.lrucache(max_codebook_size)
        reference_counts = {}
        reverse_ref_counts = defaultdict(set)
        references = {}
        codeword_lens = defaultdict(set)
    if len(codebook) == 0:
        for i, character in enumerate(alphabet):
            codebook[character] = [set([]), 0]
    assert len(alphabet) < max_codebook_size
    position = 0
    previous_match_str = ''
    max_codeword_len = 1
    while position < min(len(sequence_str), max_chars):
        if position - len(previous_match_str) < round(position, -4) and position >= round(position, -4):
            print(f"position: {position} of {len(sequence_str)}")
        if position - len(previous_match_str) < round(position, -5) and position >= round(position, -5):
            print(f"codebook: ")
            for kv in sorted({k: v for k, v in codebook.items() if k not in alphabet and len(codebook[k][0]) > 0}.items(),
                             key=lambda item: (len(codebook[item[0]][0])) / len(item[0]))[:100]:
                print(kv[0], kv[1])
        match_str = find_longest_match(sequence_str, position, codebook, max_codeword_len=max_codeword_len)
        concat_match_str = previous_match_str + match_str
        max_codeword_len = max(len(concat_match_str), max_codeword_len)

        if concat_match_str not in references:
            references[concat_match_str] = [previous_match_str, match_str]
        else:
            references[concat_match_str].extend([previous_match_str, match_str])
        if concat_match_str not in reference_counts:
            reference_counts[concat_match_str] = 0
            reverse_ref_counts[0].add(concat_match_str)
        for s in (match_str, previous_match_str):
            if not s or (s in alphabet):
                continue
            reverse_ref_counts[reference_counts[s]].remove(s)
            reference_counts[s] += 1
            reverse_ref_counts[reference_counts[s]].add(s)
            if s in lru_cache:
                del lru_cache[s]

        if concat_match_str not in alphabet and reference_counts[concat_match_str] == 0:
            lru_cache[concat_match_str] = 0
            codeword_lens[len(concat_match_str)].add(concat_match_str)
        if concat_match_str not in codebook:
            codebook[concat_match_str] = [set([]), 0]
            if len(codebook) > max_codebook_size:
                lru_cache.size(len(lru_cache))
                lru_codeword_node = lru_cache.head.prev
                lru_codeword = lru_codeword_node.key
                del lru_cache[lru_codeword]
                del codebook[lru_codeword]
                codeword_lens[len(lru_codeword)].remove(lru_codeword)
                if not codeword_lens[len(lru_codeword)]:
                    del codeword_lens[len(lru_codeword)]
                    max_codeword_len = max(codeword_lens.keys())
                # should only contain concatenated matches
                assert lru_codeword not in alphabet
                for ref in references[lru_codeword]:
                    if not ref or (ref in alphabet):
                        continue
                    reverse_ref_counts[reference_counts[ref]].remove(ref)
                    reference_counts[ref] -= 1
                    assert reference_counts[ref] >= 0
                    reverse_ref_counts[reference_counts[ref]].add(ref)
                    if reference_counts[ref] == 0:
                        lru_cache[ref] = 0
                del references[lru_codeword]
        previous_match_str = match_str
        position += len(match_str)
    return codebook, references, reference_counts, reverse_ref_counts, lru_cache, codeword_lens


def filter_proposal_candidates(codebook, min_len=4, max_len=40, filter_interesting=True):
    proposal_candidates = list(filter(
        lambda codeword: (
                             codebook[codeword] > 0 if type(codebook[codeword]) == int
                             else (
                                 len(codebook[codeword]) > 0 if type(codebook[codeword]) in (list, set) else False
                             )
                         ) and min_len <= len(set(codeword)) <= max_len,
        codebook.keys()
    ))
    if filter_interesting:
        proposal_candidates = get_interesting_codewords(proposal_candidates)
    proposal_candidates = sorted(proposal_candidates, key=len, reverse=True)
    return proposal_candidates


def simple_analysis(compressed_flows, proposal_candidates, min_threshold=0.0, length_norm=False, coverage_type='traffic', max_proposals=45):
    """Greedy algorithm for finding the best proposal candidates for a given set of compressed flows."""
    character_counts = Counter(''.join(compressed_flows))
    initial_character_counts = copy.deepcopy(character_counts)
    total_traffic = sum(character_counts.values())
    proposals = {}
    min_threshold_reached = False
    coverage_indicators = None
    count = 0
    while len(character_counts) > 0 \
            and not min_threshold_reached \
            and len(proposals) < max_proposals \
            and proposal_candidates:
        proposal_candidates_subset = proposal_candidates
        traffic_coverage_scores = {
            candidate_str: sum(
                (character_counts[char] / total_traffic)
                for char in set(candidate_str)
                if char in character_counts
            )
            for candidate_str in proposal_candidates_subset
        }
        substr_coverage_indicator_dict = {}
        if coverage_type == 'traffic':
            candidate_scores = traffic_coverage_scores
        elif coverage_type == 'overlap':
            # only count coverage
            substr_coverage_scores = {}
            for candidate_str in proposal_candidates_subset:
                marginal_substr_coverage, coverage_indicators = calculate_substr_coverage(candidate_str, compressed_flows, coverage_indicators=coverage_indicators)
                substr_coverage_scores[candidate_str] = marginal_substr_coverage
                substr_coverage_indicator_dict[candidate_str] = coverage_indicators
            candidate_scores = substr_coverage_scores
        else:
            raise Exception("Unknown coverage_type")

        if length_norm:
            try:
                candidate_scores = {candidate_str: score / len(candidate_str) # marginal coverage per step
                                    for candidate_str, score in candidate_scores.items()}
            except ValueError as e:
                print(candidate_scores)
                raise e
        best_candidate = max(candidate_scores, key=lambda candidate: (candidate_scores.get(candidate), ))
        best_score = candidate_scores[best_candidate]
        if coverage_type == 'overlap':
            coverage_indicators = substr_coverage_indicator_dict[best_candidate]
        if length_norm:
            best_score *= len(best_candidate)
        if best_score <= min_threshold:
            min_threshold_reached = True
        proposals[best_candidate] = traffic_coverage_scores[best_candidate]
        proposal_candidates.remove(best_candidate)
        # remove covered chars (normalized events) from further traffic coverage calculations
        removed_chars = []
        for char in best_candidate:
            removed = character_counts.pop(char, None)
            if removed is not None:
                removed_chars.append(char)
        # print(best_candidate)
        # print(best_score)
        # print(''.join(removed_chars) + '\n')
    return proposals, initial_character_counts


def calculate_r(compressed_flows, r_buffer=2):
    all_flows_str = ''.join(compressed_flows)
    alphabet_size = len(set(all_flows_str))
    r = math.ceil(math.log2(alphabet_size)) + r_buffer
    print("r: ", r)
    return r


# credit to Pevner for comp/decomp
def generate_charmap(event_ids):
    charmap = {k: chr(ind + 40) for ind, k in enumerate(event_ids)}
    reverse_charmap = {v: k for k, v in charmap.items()}
    return charmap, reverse_charmap


def compress_flows(flows):
    event_ids = set(event_id for flow in flows for event_id in flow.split())
    compression_charmap, decompression_charmap = generate_charmap(event_ids)
    compressed_flows = [''.join([compression_charmap[event_id] for event_id in flow.split()]) for flow in flows]
    return compressed_flows, decompression_charmap


def decompress_flows(flows, decompression_charmap):
    return [" ".join([decompression_charmap[event] for event in flow]) for flow in flows]


@lru_cache(maxsize=None)
def run_lzmw_on_compressed_flows(compressed_flows, r=12, iterations=1, ref_counts=True):
    codebook = {}
    alphabet = set(''.join(compressed_flows))
    results = None
    for i in range(iterations):
        shuffle(list(compressed_flows))
        for flow in compressed_flows:
            if ref_counts:
                results = lzmw_ref_counts(flow, r=r, intermediate_results=results, alphabet=alphabet)
                codebook, _, _, reverse_ref_counts, _, _ = results
                if i == iterations - 1:
                    codebook = {k: v for k, v in codebook.items() if k not in reverse_ref_counts[0]}
            else:
                codebook = lzmw(flow, r=r, codebook=codebook, alphabet=alphabet)
    return codebook


def run_lzmw_and_analysis(flows, r=None, min_len=3, max_len=10, min_threshold=None, r_buffer=2,
                          length_norm=True, iterations=1, coverage_type='traffic', ref_counts=True, filter_interesting=True, max_proposals=50):
    print("compressing flows...")
    compressed_flows, decomp_map = compress_flows(tuple(flows))
    print("compression completed")
    print("running lzmw on compressed flows...")
    if r is None:
        r = calculate_r(compressed_flows, r_buffer=r_buffer)
    codebook = run_lzmw_on_compressed_flows(tuple(compressed_flows), r=r, iterations=iterations, ref_counts=ref_counts)
    print("lzmw completed")
    print("filtering proposal candidates...")
    proposal_candidates = filter_proposal_candidates(codebook, min_len=min_len, max_len=max_len, filter_interesting=filter_interesting)
    print("proposal candidates filtered")
    print("beginning analysis...")
    proposals, traffic_counts = simple_analysis(compressed_flows,
                                                proposal_candidates,
                                                min_threshold=min_threshold,
                                                length_norm=length_norm,
                                                coverage_type=coverage_type,
                                                max_proposals=max_proposals)
    print("analysis complete")
    print("number of proposals: ", len(proposals))
    print("total coverage: ", sum(proposals.values()))
    return proposals, decomp_map, traffic_counts


def display_results(codebook, n=300):
    print(sorted(codebook.keys(), key=lambda k: -codebook[k])[:n])
    print(sorted(codebook.keys(), key=lambda k: -len(k))[:n])


def calculate_substr_counts(codebook):
    substr_counts = Counter()
    for k in codebook:
        for other_k in codebook:
            if k != other_k and k in other_k:
                substr_counts[k] += 1
    return substr_counts


def get_interesting_codewords(codebook):
    counts = calculate_substr_counts(codebook)
    substr_counts = counts
    sorted_substr_counts = sorted(
        substr_counts,
        key=lambda k: math.log(substr_counts[k]) * math.log(len(k)),
        reverse=True
    )
    interesting_codewords = []
    for codeword in sorted_substr_counts:
        found_substr = False
        for interesting_codeword in interesting_codewords:
            if codeword in interesting_codeword:
                found_substr = True
                break
        if not found_substr:
            interesting_codewords.append(codeword)
    return interesting_codewords


def calculate_substr_coverage(codeword, compressed_flows, coverage_indicators=None):
    """Calculates the coverage of a codeword in a set of compressed flows. If coverage_indicators is provided, it will
    be updated with the coverage of the codeword. If not, a new coverage indicator will be created and returned."""
    if coverage_indicators is None:
        coverage_indicators = [[0] * len(flow) for flow in compressed_flows]
    assert len(coverage_indicators) == len(compressed_flows)
    assert all(len(flow) == len(indicator) for flow, indicator in zip(compressed_flows, coverage_indicators))
    codeword_len = len(codeword)
    old_events_covered = sum(sum(coverage_indicators[i]) for i, flow in enumerate(compressed_flows))
    events_covered = 0
    total_events = 0
    for i, flow in enumerate(compressed_flows):
        for j in range(len(flow) - codeword_len):
            if flow[j:j + codeword_len] == codeword:
                coverage_indicators[i][j:j + codeword_len] = [1] * codeword_len
        events_covered += sum(coverage_indicators[i])
        total_events += len(compressed_flows[i])
    marginal_coverage = (events_covered - old_events_covered) / total_events
    return marginal_coverage, coverage_indicators


def display_results_ref_counts(codebook, n=300):
    print(
        sorted(
            list(filter(
                lambda x: len(codebook[x][0]) > 0, codebook.keys()
            ))[:n],
            key=lambda k: len(codebook[k][0]) / len(k),
            reverse=True
        )
    )
    print(get_interesting_codewords(codebook))


if __name__ == '__main__':
    sentences = [' '.join(sent).lower() for sent in brown.sents()]
    brown_text = ' '.join(sentences)
    # alphabet = set(brown_text)

    proposals, decomp_map, traffic_counts = run_lzmw_and_analysis(sentences, r_buffer=3, max_len=5, filter_interesting=False, min_threshold=0.0005, max_proposals=50, iterations=3)
    # codebook = lzmw(brown_text, r=14, max_chars=None)
    print(list(zip(decompress_flows(proposals.keys(), decomp_map), proposals.values())))



