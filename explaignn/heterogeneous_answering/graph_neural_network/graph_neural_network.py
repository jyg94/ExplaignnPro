import copy
import json
import numpy as np
import os
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

import explaignn.heterogeneous_answering.graph_neural_network.dataset_gnn as dataset
from explaignn.heterogeneous_answering.graph_neural_network.gnn_factory import GNNFactory
from explaignn.heterogeneous_answering.heterogeneous_answering import HeterogeneousAnswering
from explaignn.library.utils import get_logger, mark_separator, get_para
from explaignn.heterogeneous_answering.graph_neural_network.graph import Graph

START_DATE = time.strftime("%y-%m-%d_%H-%M", time.localtime())

class GNNModule(HeterogeneousAnswering):
    def __init__(self, config, iteration=None):
        super(GNNModule, self).__init__(config)
        self.logger = (
            get_logger(f"{__name__}_{iteration}", config)
            if iteration
            else get_logger(__name__, config)
        )

        self.gnn = GNNFactory.get_gnn(config)
        total_params = sum(p.numel() for p in self.gnn.parameters() if p.requires_grad)
        self.logger.info(f"Initialized model with {total_params} parameters.")
        self.model_loaded = False

    def train(self, sources=("kb", "text", "table", "info"), train_path=None, dev_path=None):
        """
        Train the model on generated HA data (skip instances for which answer is not there).
        Code inspired by: https://www.geeksforgeeks.org/training-neural-networks-with-validation-using-pytorch/.
        """
        self.logger.info(f"Loading data...")
        self.logger.info(f"Config: {json.dumps(self.config, indent=4)}")

        # set paths
        data_dir = self.config["path_to_intermediate_results"]
        qu = self.config["qu"]
        ers = self.config["ers"]
        sources_str = "_".join(sources)
        model_dir = os.path.dirname(self.config["gnn_model_path"])
        method_name = self.config["name"]

        # load parameters
        train_batch_size = self.config["gnn_train_batch_size"]
        eval_batch_size = self.config["gnn_eval_batch_size"]
        epochs = self.config["gnn_epochs"]
        lr = float(self.config["gnn_learning_rate"])
        weight_decay = float(self.config["gnn_weight_decay"])

        # load data
        if not train_path:
            train_path = os.path.join(
                data_dir, qu, ers, sources_str, f"train_ers-{method_name}.jsonl"
            )
        train_data = dataset.DatasetGNN(self.config, data_path=train_path, train=True)
        train_loader = DataLoader(
            train_data, batch_size=train_batch_size, shuffle=True, collate_fn=dataset.collate_fn
        )
        self.logger.info(f"Length train data: {len(train_data)}")

        if not dev_path:
            dev_path = os.path.join(data_dir, qu, ers, sources_str, f"dev_ers-{method_name}.jsonl")
        dev_data = dataset.DatasetGNN(self.config, data_path=dev_path, train=False)
        dev_loader = DataLoader(
            dev_data, batch_size=eval_batch_size, shuffle=True, collate_fn=dataset.collate_fn
        )
        self.logger.info(f"Length dev data: {len(dev_data)}")

        # initialize optimization
        optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=lr, weight_decay=weight_decay)

        # start training loop
        min_dev_loss = np.inf
        min_dev_qa = -1

        self.logger.info(f"Starting training...")

        for epoch in range(epochs):
            # training (for epoch)
            train_loss = 0.0
            self.gnn.train()
            for batch_index, instances in enumerate(tqdm(train_loader)):
                # move data to gpu (if possible)
                GNNModule._move_to_cuda(instances)
                # clear the gradients
                optimizer.zero_grad()
                # make prediction / forward pass
                output = self.gnn(instances, train=True)

                # update weights
                loss = output["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), self.config["gnn_clipping_max_norm"])
                optimizer.step()
                # calculate loss
                train_loss += loss.item()

                # free up GPU space
                del instances

            # validation (for epoch)
            dev_loss = 0.0
            dev_qa_metrics = list()
            self.gnn.eval()
            with torch.no_grad():
                for batch_index, instances in enumerate(tqdm(dev_loader)):
                    # move data to gpu (if possible)
                    GNNModule._move_to_cuda(instances)
                    # make prediction / forward pass
                    output = self.gnn(instances, train=False)
                    # compute loss
                    loss = output["loss"]
                    # calculate loss
                    dev_loss += loss.item()
                    # aggregate QA metrics
                    qa_metrics = output["qa_metrics"]
                    dev_qa_metrics += qa_metrics

                    # free up GPU space
                    del instances

            # aggregate result
            avg_train_loss = train_loss / len(train_loader)
            avg_dev_loss = dev_loss / len(dev_loader)
            if dev_qa_metrics:
                avg_qa_metrics = {
                    key: f"{(sum([res[key] for res in dev_qa_metrics]) / len([res[key] for res in dev_qa_metrics])):.3f}"
                    for key in ["p_at_1", "mrr", "h_at_5", "answer_presence"]
                }
                avg_qa_metrics["num_questions"] = len(dev_qa_metrics)
            else:
                avg_qa_metrics = None

            # log results
            training_res = f"Epoch: {epoch+1}, Training loss: {avg_train_loss:.3f}."
            self.logger.info(training_res)
            dev_res = f"Epoch: {epoch+1}, Dev loss: {avg_dev_loss:.3f}."
            self.logger.info(dev_res)
            qa_metrics = f"Epoch: {epoch+1}, Dev QA metrics: {avg_qa_metrics}."
            self.logger.info(qa_metrics)
            decisive_metric = self.config["gnn_decisive_metric"]
            avg_dev_qa = float(avg_qa_metrics[decisive_metric])

            # save model if performance on dev improved
            if avg_dev_qa > min_dev_qa:
                self.logger.info(
                    f"""Saving the model! Dev performance ({decisive_metric}) increased ({min_dev_qa:.3f}--->{avg_dev_qa:.3f}).
                    Dev loss ({min_dev_loss:.3f}--->{avg_dev_loss:.3f})."""
                )
                min_dev_loss = avg_dev_loss
                min_dev_qa = avg_dev_qa

                # saving the model to standard path
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                torch.save(self.gnn.state_dict(), self.config["gnn_model_path"])

                # saving the model (for not over-writing results)
                model_path = os.path.join(model_dir, f"model-{START_DATE}.bin")
                torch.save(self.gnn.state_dict(), model_path)

                # saving the config
                config_path = os.path.join(model_dir, f"config-{START_DATE}.json")
                with open(config_path, "w") as fp:
                    fp.write(json.dumps(self.config, indent=4))

                # saving the current result
                result_path = os.path.join(model_dir, f"result-{START_DATE}.res")
                with open(result_path, "w") as fp:
                    fp.write(f"{training_res}\n")
                    fp.write(f"{dev_res}\n")
                    fp.write(f"{qa_metrics}\n\n")

        # done
        self.logger.info(f"Finished training.")

    def inference_on_turns(self, turns, sources=("kb", "text", "table", "info"), train=False):
        """Run inference on a multiple turns."""
        self.logger.info("Running inference_on_turns function!")
        self.load()

        for turn in turns:
            if "top_evidences" not in turn:
                print("what??? " + str(turn["question_id"]))
            for evi in turn["top_evidences"]:
                evi["last_score"] = evi["score"]

        with torch.no_grad(), tqdm(total=len(turns)) as p_bar:
            batch_size = self.config["gnn_eval_batch_size"]

            # run inference
            instances = dataset.DatasetGNN.prepare_turns(self.config, turns, train=False)
            start_index = 0

            turns_before_iteration = copy.deepcopy(turns)

            while start_index < len(instances):
                end_index = min(start_index + batch_size, len(instances))
                batch_instances = instances[start_index:end_index]
                batch = dataset.collate_fn(batch_instances)
                # move data to gpu (if possible)
                GNNModule._move_to_cuda(batch)
                output = self.gnn(batch, train=False)

                for i, _ in enumerate(batch_instances):

                    instance_index = start_index + i
                    turn = turns[instance_index]

                    turn_before_iteration = turns_before_iteration[instance_index]

                    #store scores
                    for evi in turn_before_iteration["top_evidences"]:
                        evidence_to_score = output["evidence_predictions"][i]["evidence_to_score"]
                        entity_to_score = output["evidence_predictions"][i]["entity_to_score"]
                        assert(evi["evidence_text"] in evidence_to_score)
                        evi["score"] = evidence_to_score[evi["evidence_text"]]
                        for ent in evi["wikidata_entities"]:
                            if ent["id"] in entity_to_score:
                                ent["score"] = entity_to_score[ent["id"]]
                            else:
                                ent["score"] = -1
                                me = self.config["gnn_max_entities"]
                    


                    # store
                    turn["ranked_answers"] = output["answer_predictions"][i]["ranked_answers"]

                    # add top-evidences within iterative GNN
                    if "gnn_max_output_evidences" in self.config and len(turn["top_evidences"]) > self.config["gnn_max_output_evidences"]:
                        if "comment" in self.config and "sep" in self.config["comment"]:
                            sorted_top_evidences = sorted(turn_before_iteration["top_evidences"], key=lambda evi: evi["score"], reverse=True)
                            scores = [evi["score"] for evi in sorted_top_evidences]
                            alpha = None
                            if "alpha" in self.config["comment"]:
                                alpha = get_para(self.config["comment"], "alpha")
                            if alpha == None:
                                alpha = 0.5
                            k = min(mark_separator(scores, alpha) + 1, self.config["gnn_max_output_evidences"])
                            if "min" in self.config["comment"]:
                                number = get_para(self.config["comment"], "min")
                                k = max(k, number)
                            turn["top_evidences"] = sorted_top_evidences[:k]
                            turn_before_iteration["separator"] = k - 1
                        else:
                            top_evidences = output["evidence_predictions"][i]["top_evidences"]

                            #dropped_evid = []
                            #dropped_evid = [evid for evid in turn["top_evidences"] if evid not in top_evidences]
                            #dropped_evid = [d['evidence_text'] + ", score: " + str(d['score']) for d in dropped_evid]
                            #turn['ans_evs_dropped_list'].append(dropped_evid)
                            
                            turn["top_evidences"] = list(top_evidences)
                    #else:
                        #del turn["top_evidences"]

                    if "top_evidences" not in turn:
                        print("not in turn " + str(turn["question_id"]))
                    # obtain metrics
                    if "answers" in turn:  # e.g. for demo answers are not known
                        turn["p_at_1"] = output["qa_metrics"][i]["p_at_1"]
                        turn["mrr"] = output["qa_metrics"][i]["mrr"]
                        turn["h_at_5"] = output["qa_metrics"][i]["h_at_5"]
                        turn["answer_presence"] = output["qa_metrics"][i]["answer_presence"]

                    # delete other information
                    if "question_entities" in turn:
                        del turn["question_entities"]
                    if "silver_SR" in turn:
                        del turn["silver_SR"]
                    if "silver_relevant_turns" in turn:
                        del turn["silver_relevant_turns"]
                    if "silver_answering_evidences" in turn:
                        del turn["silver_answering_evidences"]
                    if "instance" in turn:
                        del turn["instance"]

                start_index += batch_size
                p_bar.update(batch_size)
            
            #store graph structure to show in the websiteß
            self.update_graph_info_turns(turns, turns_before_iteration)

        return turns_before_iteration


    def update_graph_info_turns(self, turns, turns_before_iteration):
        assert(len(turns) == len(turns_before_iteration))
        for i in range(len(turns)):
            self.update_graph_info(turns[i], turns_before_iteration[i])

    def update_graph_info(self, turn, turn_before_iteration):
        """Prepare data for showing visible graphs in website"""
        if 'graphs' not in turn_before_iteration:
            turn_before_iteration['graphs'] = []
        if 'answer_presence_list' not in turn_before_iteration:
            turn_before_iteration['answer_presence_list'] = []
        if 'evidence_list' not in turn_before_iteration:
            turn_before_iteration['evidence_list'] = []
        
        if self.config["gnn_max_evidences"] <= 20:
            instance = dataset.DatasetGNN.prepare_turn(self.config, turn_before_iteration, train=False)
            graph = Graph().from_instance(instance)
            turn_before_iteration['graphs'].append(graph.to_string())
        else:
            turn_before_iteration['graphs'].append("")

        turn_before_iteration['answer_presence_list'].append(turn_before_iteration['answer_presence'])
        
        keys = ["evidence_text", "score", "is_answering_evidence"]
        sorted_evis = sorted(
            [{key: evi[key] for key in keys} for evi in turn_before_iteration["top_evidences"]],
            key=lambda x: float(x["score"]),
            reverse=True
        )
        if "separator" in turn_before_iteration:
            sorted_evis.append(turn_before_iteration["separator"])
        else:
            sorted_evis.append(-1)
        turn_before_iteration['evidence_list'].append(sorted_evis)

        keys = ["graphs", "answer_presence_list", "evidence_list"]
        for key in keys:
            turn[key] = turn_before_iteration[key]

        keys = ["p_at_1", "mrr", "h_at_5", "ranked_answers"]
        for key in keys:
            turn_before_iteration[key] = turn[key]

    def inference_on_turn(self, turn, sources=("kb", "text", "table", "info"), train=False):
        """Run inference on a single turn."""
        return self.inference_on_turns([turn], sources, train)[0]

    @staticmethod
    def _move_to_cuda(obj):
        if torch.cuda.is_available():
            for key, value in obj.items():
                if not type(obj[key]) is torch.Tensor:
                    continue
                obj[key] = obj[key].cuda()

    def load(self):
        """Load the model."""
        # only load if not already done so
        if not self.model_loaded:
            model_path = self.config["gnn_model_path"]
            if torch.cuda.is_available():
                state_dict = torch.load(model_path)
                if "encoder.encoder_linear.weight" in state_dict: del(state_dict["encoder.encoder_linear.weight"])
                if "encoder.encoder_linear.bias" in state_dict: del(state_dict["encoder.encoder_linear.bias"])
                self.gnn.load_state_dict(state_dict)
            else:
                state_dict = torch.load(model_path, map_location="cpu")
                if "encoder.encoder_linear.weight" in state_dict: del(state_dict["encoder.encoder_linear.weight"])
                if "encoder.encoder_linear.bias" in state_dict: del(state_dict["encoder.encoder_linear.bias"])
                self.gnn.load_state_dict(state_dict)
            self.gnn.eval()
            self.model_loaded = True
