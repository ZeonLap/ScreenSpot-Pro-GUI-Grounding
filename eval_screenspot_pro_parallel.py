import copy
import itertools
import torch
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from datetime import datetime

# Constants
GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']

def setup_logging(rank, log_dir):
    """Setup logging configuration for each process"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'gpu_{rank}.log')),
            logging.StreamHandler()
        ]
    )

def setup_distributed(rank, world_size):
    """Initialize distributed training environment"""
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        return True
    except Exception as e:
        logging.error(f"Failed to initialize distributed setup: {str(e)}")
        return False

def cleanup_distributed():
    """Clean up distributed training environment"""
    try:
        dist.destroy_process_group()
    except Exception as e:
        logging.error(f"Error during distributed cleanup: {str(e)}")

def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    """
    Filters the results based on provided values. None means include all (ignore filtering this attribute).

    Parameters:
        results (list): A list of dictionaries containing sample results.
    
    Returns:
        list: A filtered list of dictionaries based on the given criteria.
    """
    filtered_results = []

    for sample in results:
        # Check each filter condition; if None, consider it as passed
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)

    return filtered_results

def make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    """
    Returns a list of combinations of values for attributes where the corresponding parameter is set to True.
    """
    # Initialize a dictionary to store unique values for each attribute
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }

    # Collect unique values from the results
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))

    # Filter out the attributes that are set to False (no need for combinations)
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []

    # Generate all combinations of the selected attributes using itertools.product
    attribute_combinations = list(itertools.product(*filtered_values.values()))

    # Convert combinations into dictionaries with corresponding attribute names
    combinations = []
    for combination in attribute_combinations:
        combinations.append(dict(zip(filtered_values.keys(), combination)))

    return combinations

def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
    # bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, w, h
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    
    click_point = response["point"]  # may be none
    print(click_point)
    if click_point is None:
        return "wrong_format"
    # Check if the predicted point falls in the ground truth box
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"

def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else: ## response["result"] == wrong_format
        return "wrong_format"

def evaluate_fine_grained(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        application=True,
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            application=application,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def calc_metric_for_result_list(results):
    """Calculates the metrics for a simple result list."""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    # Calculate text and icon specific metrics using collect_results_to_eval
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")

    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics

def evaluate_seeclick_paper_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_detailed_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        application=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        application = combo.get("application")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            application=application,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"app:{application}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_simple_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        group=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        group = combo.get("group")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            group=group,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"group:{group}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_overall(results):
    """
    Evaluates the overall metrics for all results without any filtering.
    
    Parameters:
        results (list): A list of dictionaries containing sample results.
        
    Returns:
        dict: A dictionary containing the overall metrics.
    """
    # Calculate metrics for the entire result set
    metrics = calc_metric_for_result_list(results)
    
    return metrics


def evaluate(results):
    """Collect results and calculate metrics. You can comment out function calls or add new ones based on your need.
    """
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }

    # TODO: comment out function calls based on your need
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = evaluate_overall(results)

    # Save detailed results
    result_report["details"] = results

    return result_report


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU ScreenSpot Evaluation')
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'])
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en')
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'])
    parser.add_argument('--log_dir', type=str, required=True, help='Directory for logs')
    parser.add_argument('--world_size', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output files')
    return parser.parse_args()

def build_model(args):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path
    if model_type == "cogagent":
        from models.cogagent import CogAgentModel
        model = CogAgentModel()
        model.load_model()
    elif model_type == "seeclick":
        from models.seeclick import SeeClickModel
        model = SeeClickModel()
        model.load_model()
    elif model_type == "qwen1vl":
        from models.qwen1vl import Qwen1VLModel
        model = Qwen1VLModel()
        model.load_model()
    elif model_type == "qwen2vl":
        from models.qwen2vl import Qwen2VLModel
        model = Qwen2VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "minicpmv":
        from models.minicpmv import MiniCPMVModel
        model = MiniCPMVModel()
        model.load_model()
    elif model_type == "internvl":
        from models.internvl import InternVLModel
        model = InternVLModel()
        model.load_model()
    elif model_type in ["gpt4o", "gpt4v"]:
        from models.gpt4x import GPT4XModel
        model = GPT4XModel()
    elif model_type == "osatlas-4b":
        from models.osatlas4b import OSAtlas4BModel
        model = OSAtlas4BModel()
        model.load_model()
    elif model_type == "osatlas-7b":
        from models.osatlas7b import OSAtlas7BModel
        model = OSAtlas7BModel()
        model.load_model()
    elif model_type == "uground":
        from models.uground import UGroundModel
        model = UGroundModel()
        model.load_model()
    elif model_type == "fuyu":
        from models.fuyu import FuyuModel
        model = FuyuModel()
        model.load_model()
    elif model_type == "showui":
        from models.showui import ShowUIModel
        model = ShowUIModel()
        model.load_model()
    elif model_type == "ariaui":
        from models.ariaui import AriaUIVLLMModel
        model = AriaUIVLLMModel()
        model.load_model()
    elif model_type == "cogagent24":
        from models.cogagent24 import CogAgent24Model
        model = CogAgent24Model()
        model.load_model()

    elif model_type == "seeclick-pro-agent":
        from models.seeclick_pro import SeeClickProAgent
        from models.osatlas7b import OSAtlas7BVLLMModel
        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = SeeClickProAgent(grounder=grounder)
    elif model_type == "aguvis":
        from models.aguvis import AguvisModel
        model = AguvisModel()
        if args.model_name_or_path:
            model.load_model(args.model_name_or_path)
        else:
            model.load_model()
    else:
        raise ValueError(f"Unsupported model type {model_type}.")
    model.set_generation_config(temperature=0, max_new_tokens=256)
    return model

def split_tasks(tasks, rank, world_size):
    """Split tasks among GPUs in a balanced way"""
    tasks_per_gpu = len(tasks) // world_size
    remainder = len(tasks) % world_size
    
    start_idx = rank * tasks_per_gpu + min(rank, remainder)
    end_idx = start_idx + tasks_per_gpu + (1 if rank < remainder else 0)
    
    return tasks[start_idx:end_idx]

def process_tasks(rank, model, tasks, args):
    """Process tasks assigned to this GPU"""
    results = []
    for sample in tqdm(tasks, desc=f"GPU {rank} Processing"):
        try:
            filename = sample["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename)

            if sample["gt_type"] == "positive":
                response = model.ground_only_positive(
                    instruction=sample["prompt_to_evaluate"], 
                    image=img_path
                )
            else:
                response = model.ground_allow_negative(
                    instruction=sample["prompt_to_evaluate"], 
                    image=img_path
                )

            point = response["point"]
            img_size = sample["img_size"]
            point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None

            sample_result = {
                "img_path": img_path,
                "group": sample.get("group"),
                "platform": sample["platform"],
                "application": sample["application"],
                "language": sample["language"],
                "instruction_style": sample["instruction_style"],
                "prompt_to_evaluate": sample["prompt_to_evaluate"],
                "gt_type": sample["gt_type"],
                "ui_type": sample["ui_type"],
                "task_filename": sample["task_filename"],
                "pred": point_in_pixel,
                "raw_response": response["raw_response"]
            }

            if sample["gt_type"] == "positive":
                correctness = eval_sample_positive_gt(sample, response)
                sample_result["bbox"] = sample["bbox"]
            else:
                correctness = eval_sample_negative_gt(sample, response)

            sample_result["correctness"] = correctness
            results.append(sample_result)

        except Exception as e:
            logging.error(f"Error processing sample {filename}: {str(e)}")
            continue

    return results

def main_worker(rank, world_size, args):
    """Main worker function for each GPU"""
    # Setup for this process
    if not setup_distributed(rank, world_size):
        return

    setup_logging(rank, args.log_dir)
    logging.info(f"Process {rank} starting")

    try:
        # Build model
        model = build_model(args)
        logging.info(f"Model loaded successfully on GPU {rank}")

        # Load and prepare tasks
        if args.task == "all":
            task_filenames = [
                os.path.splitext(f)[0]
                for f in os.listdir(args.screenspot_test)
                if f.endswith(".json")
            ]
        else:
            task_filenames = args.task.split(",")

        inst_styles = INSTRUCTION_STYLES if args.inst_style == "all" else args.inst_style.split(",")
        languages = LANGUAGES if args.language == "all" else args.language.split(",")
        gt_types = GT_TYPES if args.gt_type == "all" else args.gt_type.split(",")

        # Prepare all tasks
        all_tasks = []
        for task_filename in task_filenames:
            with open(os.path.join(args.screenspot_test, f"{task_filename}.json"), 'r') as f:
                task_data = json.load(f)

            for inst_style, gt_type, lang in itertools.product(inst_styles, gt_types, languages):
                if lang == "cn" and (inst_style != 'instruction' or gt_type != 'positive'):
                    continue

                for task_instance in task_data:
                    task_instance = copy.deepcopy(task_instance)
                    task_instance.update({
                        "task_filename": task_filename,
                        "gt_type": gt_type,
                        "instruction_style": inst_style,
                        "language": lang,
                        "prompt_to_evaluate": task_instance["instruction_cn"] if lang == "cn" else task_instance["instruction"]
                    })
                    all_tasks.append(task_instance)

        # Split tasks for this GPU
        local_tasks = split_tasks(all_tasks, rank, world_size)
        logging.info(f"GPU {rank} assigned {len(local_tasks)} tasks")

        # Process tasks
        local_results = process_tasks(rank, model, local_tasks, args)

        # Gather results from all GPUs
        all_results = [None for _ in range(world_size)]
        dist.all_gather_object(all_results, local_results)

        # Only rank 0 writes final results
        if rank == 0:
            combined_results = []
            for r in all_results:
                combined_results.extend(r)

            result_report = evaluate(combined_results)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                args.output_dir, 
                f"results_{args.model_type}_{timestamp}.json"
            )
            os.makedirs(args.output_dir, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(result_report, f, indent=4)
            
            logging.info(f"Results saved to {output_file}")

    except Exception as e:
        logging.error(f"Error in process {rank}: {str(e)}")
    finally:
        cleanup_distributed()
        logging.info(f"Process {rank} finished")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    try:
        mp.spawn(
            main_worker,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()
