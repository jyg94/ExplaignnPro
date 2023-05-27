from explaignn.heterogeneous_answering.graph_neural_network.iterative_gnns import IterativeGNNs
from explaignn.library.utils import get_config
import sys
import ast
import json

def main():
	config_path = sys.argv[1]
	config = get_config(config_path)
	
	turn_path = sys.argv[2]

	with open(turn_path, "r") as fp:
		input_data = json.load(fp)

	
	ha = IterativeGNNs(config)
	
	ha.inference_on_data(input_data, "kb_text_table_info")

	p_at_1_list = [turn["p_at_1"] for conv in input_data for turn in conv["questions"]]
	p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
	p_at_1 = round(p_at_1, 3)
	num_questions = len(p_at_1_list)
	res_str = f"Gold answers -  P@1 ({num_questions}): {p_at_1}"
	print(res_str)

	mrr_list = [turn["mrr"] for conv in input_data for turn in conv["questions"]]
	mrr = sum(mrr_list) / len(mrr_list)
	mrr = round(mrr, 3)
	res_str = f"Gold answers - MRR ({num_questions}): {mrr}"
	print(res_str)

	hit_at_5_list = [turn["h_at_5"] for conv in input_data for turn in conv["questions"]]
	hit_at_5 = sum(hit_at_5_list) / len(hit_at_5_list)
	hit_at_5 = round(hit_at_5, 3)
	res_str = f"Gold answers - H@5 ({num_questions}): {hit_at_5}"
	print(res_str)

	

if __name__ == "__main__":
	main()
