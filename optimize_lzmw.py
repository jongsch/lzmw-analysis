import argparse

from nltk.corpus import brown

from lzmw import run_lzmw_and_analysis, decompress_flows
from hyperopt import hp, fmin, Trials, tpe, STATUS_OK


param_options = {
    'min_len': (4, 6),
    'min_threshold': (0.0, 0.0005, 0.001, 0.005, 0.01),
    'r_buffer': tuple(range(5, 10)),
    'length_norm': (True, False),
    'iterations': (1, ), #2),
    'ref_counts': (True, False),
    'filter_interesting': (True, False),
}


space = {param: hp.choice(param, param_options[param])
         for param in param_options}


def get_objective_fn(flows):
    def objective_fn(params):
        # can replace this with some other metric based on post-processing the result
        proposals, decomp_map, traffic_counts = run_lzmw_and_analysis(flows, r_buffer=3, max_len=5,
                                                                      filter_interesting=False, min_threshold=0.0005,
                                                                      max_proposals=50, iterations=3)
        proposals_coverage = sum(proposals.values())
        return {
            'loss': 1 - proposals_coverage,
            'status': STATUS_OK,
            'proposals_coverage': proposals_coverage,
        }
    return objective_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    trials = Trials()

    sentences = [' '.join(sent).lower() for sent in brown.sents()]
    best = fmin(fn=get_objective_fn(sentences),
                space=space,
                algo=tpe.suggest,
                max_evals=5,
                trials=trials)
    best_params = {k: v[best[k]] for k, v in param_options.items()}
    print("best:")
    print(best_params)
    print()
    # print("trials:")
    # pprint(vars(trials))
    print("best coverage:")
    print(
        max(
            vars(trials).get('_trials', {}),
            key=lambda x: x.get('result', {}).get('proposals_coverage', 0)
        ).get('result', {}).get('proposals_coverage', 0)
    )

    # Rerun with best params
    proposals, decomp_map, traffic_counts = run_lzmw_and_analysis(sentences, **best_params)
    total_traffic = sum(traffic_counts.values())
    decompressed_proposals = [flow.split()
                              for flow in decompress_flows(proposals.keys(), decomp_map)]
    event_coverages = {
        decomp_map[event_symbol]: traffic_counts.get(event_symbol, 0) / total_traffic
        for event_symbol in traffic_counts
    }
    print("event_coverages:", event_coverages)
