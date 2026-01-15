import argparse
import json
import os

from collections import defaultdict

from model.LLM_KGQA import valid_models

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Metric Values from result file")

    parser.add_argument('--dataset', type=str, default='mquake',
                        help='Name of the dataset to process.')
    parser.add_argument('--hops', type=str, default='n',
                        help='Number of hops for subgraph extraction.')
    
    parser.add_argument('--llm-models', type=str, nargs='+', default=valid_models,
                        help='Models to include in the plot.')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed used in the experiments.')
    
    parser.add_argument('--subgraph-size', type=int, nargs='+', default=[None, 5, 10, 50, 100, 250, 500, 1000, 2000],
                        help='Number of triplets in the extracted subgraph.')
    
    parser.add_argument('--sampling-method', type=str, default='neighborhood',
                        choices=['random', 'neighborhood'],
                        help='Method for subgraph sampling.')

    parser.add_argument('--result-dir', type=str, default='./results',
                        help='Directory to save the results.')
    parser.add_argument('--plot-dir', type=str, default='./plots',
                        help='Directory to save the plots.')
    
    # plotting parameters
    parser.add_argument('--metric-yaxis', type=str, default='avg_accuracy',
                        choices=['subgraph_size', 'avg_accuracy', 'prompt_tokens', 'response_tokens', 'total_tokens', 'response_seconds', 'prompt_seconds', 'total_seconds', 'unknown', 'timeouts', 'errors'],
                        help='Metric to plot.')
    parser.add_argument('--metric-xaxis', type=str, default='subgraph_size',
                        choices=['subgraph_size', 'avg_accuracy', 'prompt_tokens', 'response_tokens', 'total_tokens', 'response_seconds', 'prompt_seconds', 'total_seconds', 'unknown', 'timeouts', 'errors'],
                        help='Metric for x-axis.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    results = defaultdict(dict)
    result_path = os.path.join(args.result_dir, args.dataset)
    os.makedirs(args.plot_dir, exist_ok=True)

    assert args.metric_xaxis != args.metric_yaxis, "X-axis and Y-axis metrics must be different."

    for llm_model in args.llm_models:
        results[llm_model] = {args.metric_yaxis: [], args.metric_xaxis: []}
        for size in args.subgraph_size:
            sampling_str = args.sampling_method if size is not None else 'evidence'
            results_file = os.path.join(
                result_path,
                f'results_{args.hops}hop_{llm_model}_subgraph{size}_{sampling_str}_seed{args.seed}.json'
            )
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    results[llm_model][args.metric_yaxis].append(data["overall"][args.metric_yaxis] if args.metric_yaxis != 'subgraph_size' else (size if size is not None else 1))
                    results[llm_model][args.metric_xaxis].append(data["overall"][args.metric_xaxis] if args.metric_xaxis != 'subgraph_size' else (size if size is not None else 1))
            else:
                print(f"Warning: Result file {results_file} does not exist.")


    # Plotting
    plt.figure(figsize=(10, 6))
    for llm_model, metrics in results.items():
        if args.metric_yaxis in metrics and len(metrics[args.metric_xaxis])>1:
            plt.plot(
                metrics[args.metric_xaxis],
                metrics[args.metric_yaxis],
                marker='o',
                label=llm_model
            )

    plt.title(f"{args.dataset} - {args.metric_yaxis.replace('_', ' ').title()} vs {args.metric_xaxis.replace('_', ' ').title()}")
    plt.xscale('linear' if 'accuracy' in args.metric_xaxis else 'log')
    plt.yscale('linear' if 'accuracy' in args.metric_yaxis else 'log')
    plt.xlabel(args.metric_xaxis.replace('_', ' ').title())
    plt.ylabel(args.metric_yaxis.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, f"{args.dataset}_{args.metric_yaxis}_vs_{args.metric_xaxis}.png"))