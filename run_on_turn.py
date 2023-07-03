from explaignn.heterogeneous_answering.graph_neural_network.iterative_gnns import IterativeGNNs
from explaignn.library.utils import get_config
import sys
import ast
import json
import re

def main():
	config_path = sys.argv[1]
	config = get_config(config_path)
	
	i = sys.argv[3]
	config['gnn_inference'] = config['gnn_inference'][int(i):]

	turn_path = "out/" + str(config['gnn_inference'][int(i)]["gnn_max_evidences"]) + ".json"

	with open(turn_path, "r") as fp:
		input_data = json.load(fp)

	print(f"loaded {turn_path}")

	question_id = sys.argv[4]
	if question_id != "none":
		input_data = [item for item in input_data if item.get('question_id') == question_id]

	print("config is")
	print(config)

	ha = IterativeGNNs(config)
	
	del_keys = ["graphs", "answer_presence_list", "evidence_list"]
	for turn in input_data:
		for key in del_keys:
			if key in turn:
				turn[key].pop()

	ha.inference_on_turns(input_data, "kb_text_table_info")

	p_at_1_list = [turn["p_at_1"] for turn in input_data]
	p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
	p_at_1 = round(p_at_1, 4)
	num_questions = len(p_at_1_list)
	res_str = f"Gold answers -  P@1 ({num_questions}): {p_at_1}"
	print(res_str)

	mrr_list = [turn["mrr"] for turn in input_data]
	mrr = sum(mrr_list) / len(mrr_list)
	mrr = round(mrr, 4)
	res_str = f"Gold answers - MRR ({num_questions}): {mrr}"
	print(res_str)

	hit_at_5_list = [turn["h_at_5"] for turn in input_data]
	hit_at_5 = sum(hit_at_5_list) / len(hit_at_5_list)
	hit_at_5 = round(hit_at_5, 4)
	res_str = f"Gold answers - H@5 ({num_questions}): {hit_at_5}"
	print(res_str)

	

if __name__ == "__main__":
	main()