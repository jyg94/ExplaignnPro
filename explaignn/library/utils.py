import json
import logging
import os
import sys
import yaml
from pathlib import Path
import re

def get_config(path):
    """Load the config dict from the given .yml file."""
    with open(path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


def store_json_with_mkdir(data, output_path, indent=True):
    """Store the JSON data in the given path."""
    # create path if not exists
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        fp.write(json.dumps(data, indent=4 if indent else None))


def store_jsonl_with_mkdir(data, output_path, indent=False):
    """Store the JSON data in the given path."""
    # create path if not exists
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        for inst in data:
            fp.write(json.dumps(inst, indent=4 if indent else None))
            fp.write("\n")


def get_logger(mod_name, config):
    """Get a logger instance for the given module name."""
    # create logger
    logger = logging.getLogger(mod_name)
    # add handler and format
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # set log level
    log_level = config["log_level"]
    logger.setLevel(getattr(logging, log_level))
    return logger


def get_result_logger(config):
    """Get a logger instance for the given module name."""
    # create logger
    logger = logging.getLogger("result_logger")
    # add handler and format
    method_name = config["name"]
    benchmark = config["benchmark"]
    result_file = f"_results/{benchmark}/{method_name}.res"
    result_dir = os.path.dirname(result_file)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(result_file)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # set log level
    logger.setLevel("INFO")
    return logger


def plot_flow_graph(graph):
    """
    Predict turn relevances among the given conversation.
    The method will plot the resulting flow graph.
    """
    nx.nx_agraph.write_dot(graph, "test.dot")
    # same layout using matplotlib with no labels
    pos = graphviz_layout(graph, prog="dot")
    pos = pos
    plt.figure(figsize=(18, 20))
    nx.draw(graph, pos, with_labels=True, arrows=True, node_size=100)
    # nx.draw(G, pos, with_labels=True, arrows=True, node_size=100, figsize=(20, 20), dpi=150)
    plt.xlim([-1, 800])
    plt.show()


def print_dict(python_dict):
    """Print python dict as json-string."""
    json_string = json.dumps(python_dict)
    print(json_string)


def print_verbose(config, string):
    """Print the given string if verbose is set."""
    if config["verbose"]:
        print(str(string))


def extract_mapping_incomplete_complete(data_paths):
    """
    Extract mapping from incomplete questions to complete
    questions for all follow-up questions.
    """
    mapping_incomplete_to_complete = dict()
    for data_path in data_paths:
        with open(data_path, "r") as fp:
            dataset = json.load(fp)

        for conversation in dataset:
            for turn in conversation["questions"]:
                if turn["turn"] == 0:
                    continue
                question = turn["question"]
                completed = turn["completed"]
                mapping_incomplete_to_complete[question] = completed
    return mapping_incomplete_to_complete

def get_gnn_string(config):
    # Create an empty list to store the elements of the string
    gnn_string_elements = []

    # Iterate through each block in 'gnn_inference'
    for block in config['gnn_inference']:
        # Start with the 'gnn_max_evidences' value
        element = str(block['gnn_max_evidences'])
        
        # If 'connected' is present and its value is truthy, append '-connected' 
        if block.get('comment'):
            element += '!' + block['comment'] + '!'

            
        # Add the processed element to the list
        gnn_string_elements.append(element)

    # Append the 'gnn_max_output_evidences' of the last block
    #gnn_string_elements.append(str(config['gnn_inference'][-1]['gnn_max_output_evidences']))

    # Join all the elements into a single string with '-' as the separator
    gnn_string = '-'.join(gnn_string_elements)
    return gnn_string

def get_out_path(config, i):
    gnn_string = get_gnn_string(config)
    gme = config["gnn_inference"][int(i)]["gnn_max_evidences"]

    return f"out/{gnn_string}/{gme}.json"

def mark_separator(numbers, alpha = 0.5):
    min_penalty = float('inf')
    separator = None
    
    # Loop through the list, considering each number as a potential separator
    for i in range(len(numbers)):
        # Split the list into two sets based on the current separator
        s1 = numbers[:i]
        s2 = numbers[i:]
        
        # Calculate the penalty for each set
        penalty_s1 = (1 - alpha) * len(s1) * (max(s1) - min(s1)) if s1 else 0
        penalty_s2 = alpha * len(s2) * (max(s2) - min(s2)) if s2 else 0
        
        # Calculate the total penalty for this separator
        total_penalty = penalty_s1 + penalty_s2
        
        # If this penalty is lower than the current minimum, update the minimum and the separator
        if total_penalty < min_penalty:
            min_penalty = total_penalty
            separator = max(i - 1, 0)
    
    return separator

def get_para(s, para):
    pattern = rf'{para}(\d+\.?\d*)'
    match = re.search(pattern, s)
    if match:
        return float(match.group(1))
    else:
        return None
