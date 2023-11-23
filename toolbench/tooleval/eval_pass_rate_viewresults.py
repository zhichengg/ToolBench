from evaluators import load_registered_automatic_evaluator
import os
import json
import csv
from evaluators.registered_cls.rtl import AnswerStatus, TaskStatus, AnswerPass
import random
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
from tqdm import tqdm
from utils import test_sets, get_steps

abs_dir = os.path.split(__file__)[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--converted_answer_path', type=str, default="", required=True, help='converted answer path')
    parser.add_argument('--save_path', type=str, default="", required=False, help='result save path')
    parser.add_argument('--reference_model', type=str, default="", required=False, help='model predictions path')
    parser.add_argument('--test_ids', type=str, default="", required=True, help='model predictions path')
    parser.add_argument('--evaluator', type=str, default="tooleval_gpt-3.5-turbo_default", required=False, help='which evaluator to use.')
    parser.add_argument('--max_eval_threads', type=int, default=30, required=False, help='max threads nums')
    parser.add_argument('--evaluate_times', type=int, default=4, required=False, help='how many times to predict with the evaluator for each solution path.')
    return parser.parse_args()

def write_results(filename: str, reference_model: str, label_cnt: dict) -> None:
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["query", "solvable", "available_tools", "model_intermediate_steps", "model_final_step", "model", "query_id", "is_solved", "pass_rate_label", "reason", "not_hallucinate"])
        for query_id in label_cnt:
            if label_cnt[query_id]["passed"] > label_cnt[query_id]["failed"]:
                final_label = "passed"
            elif label_cnt[query_id]["passed"] < label_cnt[query_id]["failed"]:
                final_label = "failed"
            else:
                if random.random() < 0.5: # if tie, random choose
                    final_label = "passed"
                else:
                    final_label = "failed"
            query = label_cnt[query_id]["query"]
            task_solvable = label_cnt[query_id]["task_solvable"]
            tool_names = label_cnt[query_id]["tool_names"]
            answer_steps = label_cnt[query_id]["answer_steps"]
            final_step = label_cnt[query_id]["final_step"]
            is_solved = label_cnt[query_id]["is_solved"]
            reason = label_cnt[query_id]["reason"]
            not_hallucinate = label_cnt[query_id]["not_hallucinate"]
            writer.writerow([query, task_solvable, tool_names, answer_steps, final_step, reference_model, query_id, is_solved, final_label, reason, not_hallucinate])
            

if __name__ == "__main__":
    args = parse_args()
    
        
    reference_model = args.reference_model
    output_list = []
    for test_set in test_sets:
        
        # json.dump(label_cnt, open(f"{args.save_path}/{test_set}_{reference_model}.json", "w"), ensure_ascii=False, indent=4)
        label_cnt = json.load(open(f"{args.save_path}/{test_set}_{reference_model}.json", "r"))
        
        # filename = f"{args.save_path}/{test_set}_{reference_model}.csv"
        # write_results(filename, reference_model, label_cnt)
        pass_rate = 0
        number_of_random = 0
        for query_id in label_cnt:
            if label_cnt[query_id]["failed"] < label_cnt[query_id]["passed"]:
                pass_rate += 1
            elif label_cnt[query_id]["failed"] == label_cnt[query_id]["passed"]:
                number_of_random += 1
                if random.random() < 0.5:
                    pass_rate += 1
        pass_rate /= len(label_cnt)
        print(f"Test set: {test_set}. Model: {reference_model}. Pass rate: {str(pass_rate)} Number of unsure: {number_of_random}")
        results_filename = f"{args.save_path}/{test_set}_{reference_model}.log"
        with open(results_filename, 'w') as f:
            f.write(f"Test set: {test_set}. Model: {reference_model}. Pass rate: {str(pass_rate)} Number of unsure: {number_of_random}\n")
        

        
