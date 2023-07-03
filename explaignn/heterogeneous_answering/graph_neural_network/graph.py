import time
import itertools
import networkx as nx
from explaignn.library.utils import mark_separator

WIKIDATA_ENTITIES_SEP = "<BR>" + 5*"&nbsp;"
 
class Graph:
	def __init__(self):
		"""Create a new empty graph."""
		self.nx_graph = nx.Graph()
		self.nodes_dict = dict()
		self.ev_to_ent_dict = dict()
		self.ent_to_score = dict()

	def _add_entity(self, entity):
		g_ent_id = f'ent{entity["g_id"]}'
		self.nx_graph.add_node(
			g_ent_id,
			type='entity',
			entity_type=entity["type"] if "type" in entity and entity["type"] else "None",
			label=entity["label"],
			wikidata_id=entity["id"],
			is_question_entity="is_question_entity" in entity,
			is_answer="is_answer" in entity and entity["is_answer"],
			is_predicted_answer=False,
			score=entity["score"]
		)

	def _add_evidence(self, evidence):
		g_ev_id = f'ev{evidence["g_id"]}'
		self.nx_graph.add_node(
			g_ev_id,
			type='evidence',
			label=evidence["evidence_text"],
			source=evidence["source"],
			wikidata_entities=WIKIDATA_ENTITIES_SEP.join([f'"{e["label"]}" => {e["id"]}' for e in evidence["wikidata_entities"]]),
			retrieved_for_entity=str(evidence["retrieved_for_entity"]),
			is_answering_evidence="is_answering_evidence" in evidence and evidence["is_answering_evidence"],
			score=evidence["score"],
			last_score=evidence.get("last_score", None)
		)

	def from_instance(self, instance):
		""" Create a new graph from the given dataset instance. """
		# add entity nodes
		entities = instance["entities"]
		for entity in entities:
			if not "g_id" in entity:
				continue
			self._add_entity(entity)

		# add evidence nodes
		evidences = instance["evidences"]
		for evidence in evidences:
			if not "g_id" in evidence:
				continue
			self._add_evidence(evidence)

		# add edges
		ent_to_ev = instance["ent_to_ev"]
		for i, entity in enumerate(entities):
			if not "g_id" in entity:
				continue
			g_ent_id = f'ent{entity["g_id"]}'
			connected_ev_ids = ent_to_ev[i, :]
			for j, val in enumerate(connected_ev_ids):
				if val > 0:
					g_ev_id = f'ev{j}'
					self.nx_graph.add_edge(g_ent_id, g_ev_id)
		return self

	def from_scoring_output(self, scored_evidences, scored_entities, ent_to_ev, ques_id=None):
		""" Create an evidence-only graph from the outputs of the scoring phase. """

		# DEV: save data for quick development
		# torch.save(tensor, 'file.pt') and torch.load('file.pt')
		# if ques_id:
		# 	with open(f"tmp_data/tmp_data_{ques_id}.json", "w") as fp:
		# 		json.dump([scored_evidences, scored_entities], fp)
		# 	torch.save(ent_to_ev, f'tmp_data/tmp_ent_to_ev_{ques_id}.pt')

		for entity in scored_entities:
			self.ent_to_score[entity["id"]] = entity["score"]

		# add evidence nodes
		for evidence in scored_evidences:
			if not "g_id" in evidence: # padded evidence
				continue
			self._add_evidence(evidence)
			node_id = f'ev{evidence["g_id"]}'
			self.nodes_dict[node_id] = evidence
			self.ev_to_ent_dict[node_id] = [entity["id"] for entity in evidence["wikidata_entities"] if entity["id"] in self.ent_to_score]

		for i, evidence1 in enumerate(scored_evidences):
			if not "g_id" in evidence1: # padded evidence
				continue
			for j, evidence2 in enumerate(scored_evidences):
				# avoid duplicate checks or checks with same item
				if i >= j or not "g_id" in evidence2:
					continue 
				# derive set of entities
				entities1 = set([entity["id"] for entity in evidence1["wikidata_entities"]])
				entities2 = set([entity["id"] for entity in evidence2["wikidata_entities"]])

				# if shared entity, there is a connection
				if entities1 & entities2:
					g_ev_id1 = f'ev{evidence1["g_id"]}'
					g_ev_id2 = f'ev{evidence2["g_id"]}'

					# add edge
					self.nx_graph.add_edge(g_ev_id1, g_ev_id2)
		return self

	def _get_connected_subgraph_top(self, scored_evidences, max_evidences, comment):
		sorted_top_evidences = sorted(scored_evidences, key=lambda evi: evi["score"], reverse=True)
		if "tops" in "comment":
			scores = [evi["score"] for evi in sorted_top_evidences]
			k = min(mark_separator(scores) + 1, max_evidences)
		else:
			k = max_evidences
		subgraph_nodes = [f'ev{evidence["g_id"]}' for evidence in sorted_top_evidences[:k]]
		subgraph = self.nx_graph.subgraph(subgraph_nodes)
		max_score = 0
		max_nodes = []
		for component in nx.connected_components(subgraph):
			if "two" in comment:
				buf = self.expand_subgraph_two(component, max_evidences)
			elif "neigh" in comment:
				buf = self.expand_subgraph_neighbor(component, max_evidences)
			else:
				buf = self.expand_subgraph(component, max_evidences)
			score = self._get_score_of_subgraph(buf)
			if score > max_score:
				max_nodes = buf
				max_score = score

		top_evidences = [self.nodes_dict[node] for node in max_nodes]
		top_evidences.sort(key=lambda x: x["score"], reverse=True)
		return top_evidences

	def _get_connected_subgraph(self, scored_evidences, scored_entities, max_evidences, comment):
		"""
		From the given graph, get a connected subgraph that has 
		at most `max_evidences`. The output will be the set of
		evidences within this subgraph.
		"""
		ans_list = []
		for component in nx.connected_components(self.nx_graph):
			max_nodes, max_score = self._get_connected_subgraph_component(component, max_evidences, comment)
			ans_list.append({'nodes': max_nodes, 'score': max_score})

		ans_list.sort(key=lambda x: x['score'], reverse=True)
		max_nodes = []
		total_nodes_length = 0
		for element in ans_list:
			nodes_length = len(element['nodes'])
			if total_nodes_length + nodes_length <= max_evidences:
				max_nodes.append(element['nodes'])
				total_nodes_length += nodes_length
			if total_nodes_length >= max_evidences:
				break


		top_evidences = [self.nodes_dict[node] for nodes in max_nodes for node in nodes]
		top_evidences.sort(key=lambda x: x["score"], reverse=True)

		return top_evidences	

	def _get_connected_subgraph_component(self, component, max_evidences, comment):
		max_score = 0

		component = sorted(component, key=lambda node: self.nodes_dict[node]["score"], reverse=True)
  
		left, right = 0, len(component)
		while left < right:
			mid = (left + right + 1) // 2
			if max(len(component) for component in nx.connected_components(self.nx_graph.subgraph(component[:mid]))) <= max_evidences: 
				left = mid
			else:                       # If the first mid elements do not satisfy the condition
				right = mid - 1
  
		for component in nx.connected_components(self.nx_graph.subgraph(component[:left])):
			score = self._get_score_of_subgraph_ave(component)
			if  score >= max_score:
				max_score = score
				max_nodes = component
		if "two" in comment:
			self.expand_subgraph_two(max_nodes, max_evidences)
		else:
			self.expand_subgraph(max_nodes, max_evidences)
		max_score = self._get_score_of_subgraph(max_nodes)
  
		return max_nodes, max_score

	def expand_subgraph_neighbor(self, max_nodes, max_evidences):
		distance = 1  # Start from distance 1 (immediate neighbors)
		while len(max_nodes) < max_evidences:
			# Gather all nodes at current distance
			dist_nodes = set()
			for node in max_nodes:
				dist_nodes.update(set(self.nx_graph.neighbors(node)) - max_nodes)

			# If there are no nodes at the current distance, move to the next distance
			if not dist_nodes:
				distance += 1
				continue

			# Order the nodes at the current distance by their scores
			sorted_nodes = sorted(dist_nodes, key=lambda node: self.nodes_dict[node]["score"], reverse=True)

			# Add nodes to the max_nodes set in order of their scores, until max_nodes is full
			for node in sorted_nodes:
				if len(max_nodes) >= max_evidences:
					break
				max_nodes.add(node)

		return max_nodes

	def expand_subgraph(self, max_nodes, max_evidences):
		while len(max_nodes) < max_evidences:
			# Find all nodes adjacent to the current subgraph
			adjacent_nodes = set()
			for node in max_nodes:
				adjacent_nodes.update(self.nx_graph.neighbors(node))
			
			# Exclude nodes that are already in the subgraph
			candidates = adjacent_nodes - set(max_nodes)
			
			# If no candidates, we can't expand further
			if not candidates:
				break
			
			# Determine the candidate node with the highest score
			best_node = max(candidates, key=lambda node: self.nodes_dict[node]["score"])
			
			# Add the highest scoring node to the subgraph
			max_nodes.add(best_node)
	
	def expand_subgraph_two(self, max_nodes, max_evidences):
		while len(max_nodes) < max_evidences:
			# Find all nodes adjacent to the current subgraph
			adjacent_nodes = set()
			for node in max_nodes:
				adjacent_nodes.update(self.nx_graph.neighbors(node))

			# Exclude nodes that are already in the subgraph
			adjacent_nodes = adjacent_nodes - set(max_nodes)

			# Look at length 2 paths
			candidate_pairs = []
			for node in adjacent_nodes:
				# Get all nodes adjacent to this node (that are not already in the subgraph)
				adjacent_to_adjacent = set(self.nx_graph.neighbors(node)) - set(max_nodes)
				for node2 in adjacent_to_adjacent:
					candidate_pairs.append((node, node2))

			# If no candidates, we can't expand further
			if not candidate_pairs:
				break

			# Determine the candidate pair with the highest combined score
			best_pair = max(candidate_pairs, key=lambda pair: self.nodes_dict[pair[0]]["score"] + self.nodes_dict[pair[1]]["score"])

			# Add the highest scoring pair to the subgraph
			max_nodes.add(best_pair[0])
			# Ensure we don't exceed max_evidences
			if len(max_nodes) >= max_evidences:
				break

			max_nodes.add(best_pair[1])
			if len(max_nodes) >= max_evidences:
				break


		return max_nodes

	
	def _get_score_of_subgraph(self, evidence_nodes):
		"""Compute the score of the subgraph defined by the evidence nodes."""
		evidences_score = sum(self.nodes_dict[node]["score"] for node in evidence_nodes)
		seen_entities = set()
		entities_score = 0
		# for node in evidence_nodes:
		# 	for entity_id in self.ev_to_ent_dict[node]:
		# 		if not entity_id in seen_entities:
		# 			entities_score += self.ent_to_score[entity_id]
		# 			seen_entities.add(entity_id)
		return evidences_score + entities_score

	def _get_score_of_subgraph_ave(self, evidence_nodes):
		"""Compute the score of the subgraph defined by the evidence nodes."""
		evidences_score = sum(self.nodes_dict[node]["score"] for node in evidence_nodes) / len(evidence_nodes)
		seen_entities = set()
		entities_score = 0
		# for node in evidence_nodes:
		# 	for entity_id in self.ev_to_ent_dict[node]:
		# 		if not entity_id in seen_entities:
		# 			entities_score += self.ent_to_score[entity_id]
		# 			seen_entities.add(entity_id)
		return evidences_score + entities_score

	def _get_connected_subgraph_brute_force(self, scored_evidences, scored_entities, max_evidences):
		"""
		From the given graph, get a connected subgraph that has 
		at most `max_evidences`. The output will be the set of
		evidences within this subgraph.
		"""
		start_time = time.time()
		top_evidences = scored_evidences[:max_evidences] if max_evidences < len(scored_evidences) else None
  
		if top_evidences is None:
			return scored_evidences
		top_evidences_nodes = [f'ev{evidence["g_id"]}' for evidence in top_evidences]
		top_evidences_subgraph = self.nx_graph.subgraph(top_evidences_nodes).copy()
		top_evidences_score = self._get_score_of_subgraph(top_evidences_nodes)
  
		if nx.is_connected(top_evidences_subgraph):
			return top_evidences

		max_score = scored_evidences[0]["score"]
		max_nodes = {f'ev{scored_evidences[0]["g_id"]}'}

		max_component_size = max(len(component) for component in nx.connected_components(self.nx_graph))

		components_counter  = 0

		ans_list = []
		for component in nx.connected_components(self.nx_graph):
			max_score = 0
			components_counter += 1
			if time.time() - start_time > 60:
				print("Time limit reached!")
				break

			max_score_component = self._get_score_of_subgraph(component)
			if max_score_component <= max_score:
				continue

			combination_size = min(len(component), max_evidences)
			combination_counter = 0
			for nodes in itertools.combinations(component, combination_size):
				combination_counter += 1
				if time.time() - start_time > 60:
					print("Time limit reached!")
					break

				if nx.is_connected(self.nx_graph.subgraph(nodes)):
					score = self._get_score_of_subgraph(nodes)
					if score > max_score:
						max_score = score
						max_nodes = nodes
			ans_list.append({'nodes': max_nodes, 'score': max_score})

		ans_list.sort(key=lambda x: x['score'], reverse=True)
		max_nodes = []
		total_nodes_length = 0
		for element in ans_list:
			nodes_length = len(element['nodes'])
			if total_nodes_length + nodes_length <= max_evidences:
				max_nodes.append(element['nodes'])
				total_nodes_length += nodes_length
			if total_nodes_length >= max_evidences:
				break


		top_evidences = [self.nodes_dict[node] for nodes in max_nodes for node in nodes]
		top_evidences.sort(key=lambda x: x["score"], reverse=True)

		top_evidences_subgraph = self.nx_graph.subgraph(max_nodes).copy()
		top_evidences_score = max_score
		return top_evidences	



	def from_nx_graph(self, nx_graph):
		"""Create an instance of "Graph" from a given nx graph."""
		self.nx_graph = nx_graph

	def write_to_file(self, file_path="/home/pchristm/public_html/graph.gexf"):
		"""Write the graph to file."""
		xml = self.to_string()
		with open(file_path, "w") as fp:
			fp.write(xml)

	def to_string(self):
		"""Write the graph to String."""
		xml_lines = nx.generate_gexf(self.nx_graph)
		xml = "\n".join(xml_lines)
		for i in range(20):
			# fix attributes in xml string
			i_str = str(i)
			if f'id="{i_str}" title="' in xml:
				title = xml.split(f'id="{i_str}" title="', 1)[1].split('"', 1)[0]
				xml = xml.replace(f'id="{i_str}" title="', f'id="{title}" title="')
				xml = xml.replace(f'for="{i_str}"', f'for="{title}"')
		xml = "<?xml version='1.0' encoding='utf-8'?>\n" + xml
		return xml

	def get_answer_neighborhood(self, answer_entity):
		""" Get the 2-hop neighborhood of the answer in a graph (surrounding evidences->entities."""
		graph = Graph()
		if not "g_id" in answer_entity:
			return self
		g_ent_id = f'ent{answer_entity["g_id"]}'
		self.nx_graph.nodes[g_ent_id]["is_predicted_answer"] = True
		graph.from_nx_graph(nx.ego_graph(self.nx_graph, g_ent_id, radius=2))
		return graph

