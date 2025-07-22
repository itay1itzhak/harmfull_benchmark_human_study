"""
Qualtrics Participant Forms Generator

This module creates Qualtrics-ready files for a human study experiment on harmful AI content.
The experiment presents pairs of options to participants and collects harm ratings, explanations,
and realism assessments using Qualtrics Advanced Format.

Experiment Design:
- 4 participants total
- 20 items per participant total:
  * Phase 1: 15 items with harm assessment (12 treatment + 3 control)
  * Phase 2: 5 items with realism assessment only (4 treatment + 1 control)
- Treatment items: one harmful option vs zero-harm option
- Control items: both options are zero-harm (attention checks)
- Response modes: harm rating (7-point scale), free-text explanation, realism question (7-point scale)

Output Format:
- Uses Qualtrics Advanced Format with proper [[Question:type]] tags
- Includes embedded data for participant tracking
- Structured blocks for organization
- Compatible with Qualtrics survey import

Author: AI Assistant
Date: 2025
"""

import json
import pandas as pd
import numpy as np
import random
import logging
import os
import re
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path
import itertools
from collections import defaultdict, Counter
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
HARM_LEVELS = [5, 15] #[5, 10, 15]
BENEFIT_LEVELS = [10, 50] #[10, 30, 50]
NUM_PARTICIPANTS = 150
TREATMENT_PER_PARTICIPANT = 4
CONTROL_PER_PARTICIPANT = 1

# Defaults for "per-participant" realism assignment mode
REALISM_TREATMENT_PER_PARTICIPANT = 1
REALISM_CONTROL_PER_PARTICIPANT = 1

# Defaults for "global" realism assignment mode
REALISM_QUESTIONS_PER_PARTICIPANT = 1 
REALISM_CONTROL_RATIO = 0.2

TOTAL_ITEMS_PER_PARTICIPANT = (
    TREATMENT_PER_PARTICIPANT + 
    CONTROL_PER_PARTICIPANT + 
    REALISM_QUESTIONS_PER_PARTICIPANT
)


def read_existing_experiment_plans(directory: str = "data/") -> Tuple[Set[str], int]:
    """
    Scans for existing experiment plan CSVs to avoid reusing questions and to continue participant IDs sequentially.

    Args:
        directory (str): The directory to scan for experiment_plan_*.csv files.

    Returns:
        Tuple[Set[str], int]: A tuple containing:
            - A set of already used original_sample_id values.
            - The maximum participant_id found in existing plans, or 0 if none.
    """
    used_question_ids = set()
    max_participant_id = 0
    
    # Correctly form the search pattern for glob
    search_pattern = os.path.join(directory, "experiment_plan*.csv")
    plan_files = glob.glob(search_pattern)

    if not plan_files:
        logging.info("No existing experiment plan files found.")
        return used_question_ids, max_participant_id

    logging.info(f"Found existing experiment plan files: {plan_files}")

    for file_path in plan_files:
        try:
            df = pd.read_csv(file_path)
            if "original_sample_id" in df.columns:
                used_question_ids.update(df["original_sample_id"].astype(str).tolist())
            if "participant_id" in df.columns and not df["participant_id"].empty:
                max_participant_id = max(max_participant_id, df["participant_id"].max())
        except Exception as e:
            logging.error(f"Error reading or processing {file_path}: {e}")

    logging.info(f"Found {len(used_question_ids)} used questions and max participant ID of {max_participant_id}.")
    
    return used_question_ids, max_participant_id


def load_instructions_text(file_path: str = "data/insrtuctions_text.json") -> Dict[str, Any]:
    """
    Load participant-facing text content from JSON file.
    
    Args:
        file_path (str): Path to the instructions JSON file
        
    Returns:
        Dict[str, Any]: Dictionary containing all text content
        
    Raises:
        FileNotFoundError: If instructions file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Instructions file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            instructions = json.load(f)
        
        # Validate required keys exist
        required_sections = ['introduction', 'scenario_instructions', 'harm_assessment', 
                           'explanation', 'realism_assessment', 'completion', 'labels']
        
        for section in required_sections:
            if section not in instructions:
                raise ValueError(f"Missing required section '{section}' in instructions file")
        
        logging.info(f"Loaded instructions text from {file_path}")
        return instructions
    
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        raise json.JSONDecodeError(f"Invalid JSON format in {file_path}: {e}")


def load_benchmark(file_path: str = "benchmark/parsed_benchmark_data.json") -> List[Dict[str, Any]]:
    """
    Load benchmark data from the parsed JSON file.
    
    Args:
        file_path (str): Path to the parsed benchmark JSON file
        
    Returns:
        List[Dict[str, Any]]: List of benchmark samples with scenarios and metadata
        
    Raises:
        FileNotFoundError: If benchmark file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Loaded {len(data)} benchmark samples from {file_path}")
        return data
    
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        raise json.JSONDecodeError(f"Invalid JSON format in {file_path}: {e}")


def get_available_topics(data: List[Dict[str, Any]]) -> List[str]:
    """
    Extract all unique topics from the benchmark data.
    
    Args:
        data (List[Dict[str, Any]]): Benchmark data
        
    Returns:
        List[str]: List of unique topics
    """
    topics = set()
    for item in data:
        topics.add(item['metadata']['topic'])
    
    return sorted(list(topics))


def get_available_categories(data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Extract all unique categories from the benchmark data.
    
    Args:
        data (List[Dict[str, Any]]): Benchmark data
        
    Returns:
        Dict[str, List[str]]: Dictionary with lists of unique values for each category
    """
    categories = {
        'model_types': set(),
        'topics': set(), 
        'harm_types': set(),
        'benefits': set(),
        'sample_types': set()
    }
    
    for item in data:
        metadata = item['metadata']
        categories['model_types'].add(metadata['model_type'])
        categories['topics'].add(metadata['topic'])
        categories['harm_types'].add(metadata['harm_type'])
        categories['benefits'].add(metadata['benefit'])
        categories['sample_types'].add(metadata['sample_type'])
    
    # Convert sets to sorted lists
    return {key: sorted(list(values)) for key, values in categories.items()}


def filter_treatment_samples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter benchmark data to only include treatment samples.
    
    Args:
        data (List[Dict[str, Any]]): Full benchmark data
        
    Returns:
        List[Dict[str, Any]]: Treatment samples only
    """
    treatment_samples = [
        item for item in data 
        if item['metadata']['sample_type'] == 'Treatment'
    ]
    
    logging.info(f"Filtered to {len(treatment_samples)} treatment samples")
    return treatment_samples


def sample_items(
    data: List[Dict[str, Any]], 
    num_participants: int,
    treatment_per_participant: int,
    control_per_participant: int,
    total_realism_treatment: int,
    total_realism_control: int,
    realism_assignment_mode: str,
    realism_treatment_per_participant: int,
    realism_control_per_participant: int,
    random_seed: Optional[int] = None,
    used_question_ids: set = set(),
    start_participant_id: int = 1,
) -> List[Dict[str, Any]]:
    """
    Sample items for the experiment with balanced distribution and uniqueness constraints.
    
    Args:
        data (List[Dict[str, Any]]): Full benchmark data
        num_participants (int): Number of participants
        treatment_per_participant (int): Treatment items per participant
        control_per_participant (int): Control items per participant
        total_realism_treatment (int): Total number of realism treatment items to sample.
        total_realism_control (int): Total number of realism control items to sample.
        realism_assignment_mode (str): The assignment mode ('per-participant' or 'global').
        realism_treatment_per_participant (int): Items per participant (for per-participant mode).
        realism_control_per_participant (int): Items per participant (for per-participant mode).
        random_seed (Optional[int]): Random seed for reproducibility
        used_question_ids (set): A set of question IDs to exclude from sampling.
        start_participant_id (int): The starting ID for participants.
        
    Returns:
        List[Dict[str, Any]]: Sampled items with participant assignments
        
    Raises:
        ValueError: If sampling constraints cannot be satisfied
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Exclude already used questions
    if used_question_ids:
        original_data_len = len(data)
        data = [item for item in data if str(item['id']) not in used_question_ids]
        logging.info(f"Excluded {original_data_len - len(data)} already used questions.")

    # Calculate totals for regular items
    total_treatment = num_participants * treatment_per_participant
    total_control = num_participants * control_per_participant
    
    logging.info(f"Sampling {total_treatment} treatment + {total_control} control items")
    logging.info(f"Plus {total_realism_treatment} realism-treatment + {total_realism_control} realism-control items")
    
    # Filter treatment samples
    treatment_samples = filter_treatment_samples(data)
    
    total_treatment_needed = total_treatment + total_realism_treatment
    if len(treatment_samples) < total_treatment_needed:
        raise ValueError(f"Not enough treatment samples: need {total_treatment_needed}, have {len(treatment_samples)}")
    
    # Get categories for balanced sampling
    categories = get_available_categories(treatment_samples)
    topics = categories['topics']
    
    logging.info(f"Available categories:")
    logging.info(f"  Topics: {len(topics)} ({topics})")
    logging.info(f"  Harm types: {len(categories['harm_types'])}")
    logging.info(f"  Benefits: {len(categories['benefits'])}")
    logging.info(f"  Models: {len(categories['model_types'])}")
    
    # Sample regular treatment items with balanced distribution
    sampled_treatment = sample_treatment_items_balanced(
        treatment_samples, 
        total_treatment
    )

    # Verify that the sampled treatment items are balanced accross topic, model, harm type, and benefit
    final_topic_dist = Counter(item['metadata']['topic'] for item in sampled_treatment)
    logging.info(f"Topic distribution: {dict(final_topic_dist)}")

    final_model_dist = Counter(item['metadata']['model_type'] for item in sampled_treatment)
    logging.info(f"Model distribution: {dict(final_model_dist)}")
    
    final_harm_type_dist = Counter(item['metadata']['harm_type'] for item in sampled_treatment)
    logging.info(f"Harm type distribution: {dict(final_harm_type_dist)}")

    final_benefit_dist = Counter(item['metadata']['benefit'] for item in sampled_treatment)
    logging.info(f"Benefit distribution: {dict(final_benefit_dist)}")
    
    # Sample realism-only treatment items with balanced distribution
    # Use remaining treatment samples (exclude already sampled ones)
    used_treatment_ids = set(item['id'] for item in sampled_treatment)
    remaining_treatment_samples = [item for item in treatment_samples if item['id'] not in used_treatment_ids]
    
    sampled_realism_treatment = sample_treatment_items_balanced(
        remaining_treatment_samples, 
        total_realism_treatment
    )

    
    
    # Sample regular control items
    sampled_control = sample_control_items(data, total_control)
    
    # Sample realism-only control items
    # Use remaining samples (exclude already sampled ones)
    used_control_ids = set(item['id'] for item in sampled_control)
    remaining_data = [item for item in data if item['id'] not in used_control_ids]
    
    sampled_realism_control = sample_control_items(remaining_data, total_realism_control)
    
    # Combine and assign to participants
    all_sampled = sampled_treatment + sampled_control + sampled_realism_treatment + sampled_realism_control
    
    # Assign participants
    participant_assignments = assign_participants_extended(
        all_sampled, 
        num_participants, 
        treatment_per_participant, 
        control_per_participant,
        realism_assignment_mode,
        realism_treatment_per_participant,
        realism_control_per_participant,
        set(item['id'] for item in sampled_treatment),
        set(item['id'] for item in sampled_control),
        set(item['id'] for item in sampled_realism_treatment),
        set(item['id'] for item in sampled_realism_control),
        start_participant_id=start_participant_id
    )
    
    return participant_assignments


def sample_treatment_items_balanced(
    treatment_samples: List[Dict[str, Any]],
    total_needed: int
) -> List[Dict[str, Any]]:
    """
    Sample treatment items with balanced distribution across models, topics, harm types, and benefits.
    This uses a greedy approach to iteratively select items that most improve the balance
    of categorical metadata in the sample.

    Args:
        treatment_samples (List[Dict[str, Any]]): Available treatment samples.
        total_needed (int): Total number of treatment items needed.

    Returns:
        List[Dict[str, Any]]: A balanced sample of treatment items.

    Raises:
        ValueError: If not enough unique items are available to meet the demand.
    """
    if len(treatment_samples) < total_needed:
        raise ValueError(f"Not enough treatment samples to select from. Need {total_needed}, have {len(treatment_samples)}")

    # 1. Get all available categories for balancing from the provided samples
    categories = get_available_categories(treatment_samples)
    
    # Define which metadata keys to balance on
    keys_to_balance = ['model_type', 'topic', 'harm_type', 'benefit']
    
    # Map from key to the list of unique values for that key
    category_values = {
        'model_type': categories.get('model_types', []),
        'topic': categories.get('topics', []),
        'harm_type': categories.get('harm_types', []),
        'benefit': categories.get('benefits', [])
    }
    
    # Filter out empty categories to prevent division by zero
    category_values = {k: v for k, v in category_values.items() if v}

    logging.info("Attempting to balance across the following categories:")
    for key, values in category_values.items():
        logging.info(f"  - {key} ({len(values)} values)")
        
    available_samples = treatment_samples.copy()
    random.shuffle(available_samples)
    
    sampled_items = []
    used_combinations = set()

    for i in range(total_needed):
        if not available_samples:
            raise ValueError(f"Ran out of available samples while trying to select {total_needed}. Only found {len(sampled_items)}.")
            
        # 2. Calculate current distribution of sampled items
        current_counts = {
            key: Counter(item['metadata'][key] for item in sampled_items) 
            for key in category_values
        }
        
        best_candidate = None
        max_score = -float('inf')
        
        # 3. Score each available sample based on how much it improves balance
        for candidate in available_samples:
            metadata = candidate['metadata']
            combo_id = (metadata['model_type'], metadata['topic'], metadata['harm_type'], metadata['benefit'])
            
            if combo_id in used_combinations:
                continue

            score = 0
            for key, values in category_values.items():
                value = metadata.get(key)
                if value is None: continue
                
                # Ideal count for any value in this category after adding one more item
                ideal_count = (len(sampled_items) + 1) / len(values)
                
                # The "need" for this value is how far it is below the ideal count
                need = ideal_count - current_counts[key][value]
                score += need
            
            if score > max_score:
                max_score = score
                best_candidate = candidate
        
        # Fallback if all remaining items have already-used combinations
        if best_candidate is None:
            max_score = -float('inf')
            for candidate in available_samples:
                metadata = candidate['metadata']
                score = 0
                for key, values in category_values.items():
                    value = metadata.get(key)
                    if value is None: continue
                    ideal_count = (len(sampled_items) + 1) / len(values)
                    need = ideal_count - current_counts[key][value]
                    score += need
                
                if score > max_score:
                    max_score = score
                    best_candidate = candidate

        if best_candidate is None:
             raise ValueError("Could not select a candidate item. The available pool might be empty.")
        
        # 5. Add best candidate to sample
        sampled_items.append(best_candidate)
        available_samples.remove(best_candidate)
        
        metadata = best_candidate['metadata']
        combo_id = (metadata['model_type'], metadata['topic'], metadata['harm_type'], metadata['benefit'])
        used_combinations.add(combo_id)

    logging.info(f"Final treatment sample: {len(sampled_items)} items")
    
    # Log final distribution for all balanced categories
    for key in category_values:
        final_dist = Counter(item['metadata'][key] for item in sampled_items)
        logging.info(f"Final {key} distribution: {dict(final_dist)}")
    
    return sampled_items


def validate_item_content_consistency(items: List[Dict[str, Any]]) -> None:
    """
    Validate that item content is consistent with assigned experimental roles.
    
    This function ensures that:
    - Experimental treatment items use original Treatment samples
    - Experimental control items use original Control samples
    - Content matches the intended experimental purpose
    
    Args:
        items (List[Dict[str, Any]]): Items with assigned roles
        
    Raises:
        ValueError: If content inconsistencies are detected
    """
    inconsistencies = []
    
    for item in items:
        item_type = item.get('item_type', 'unknown')
        original_sample_type = item['metadata']['sample_type']
        item_id = item.get('item_id', item.get('id', 'unknown'))
        
        # Check experimental treatment items
        if item_type in ['treatment', 'realism_treatment']:
            if original_sample_type != 'Treatment':
                inconsistencies.append(
                    f"Item {item_id}: Experimental {item_type} should use original Treatment samples, "
                    f"but uses original {original_sample_type} sample"
                )
        
        # Check experimental control items
        elif item_type in ['control', 'realism_control']:
            if original_sample_type != 'Control':
                inconsistencies.append(
                    f"Item {item_id}: Experimental {item_type} should use original Control samples, "
                    f"but uses original {original_sample_type} sample"
                )
    
    if inconsistencies:
        error_msg = "Content consistency validation failed:\n" + "\n".join(inconsistencies)
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    logging.info(f"Content consistency validation passed for {len(items)} items")


def filter_control_samples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter benchmark data to only include original Control samples.
    
    Args:
        data (List[Dict[str, Any]]): Full benchmark data
        
    Returns:
        List[Dict[str, Any]]: Control samples only
    """
    control_samples = [
        item for item in data 
        if item['metadata']['sample_type'] == 'Control'
    ]
    
    logging.info(f"Filtered to {len(control_samples)} original control samples")
    return control_samples


def sample_control_items(data: List[Dict[str, Any]], total_needed: int) -> List[Dict[str, Any]]:
    """
    Sample control items (attention checks) ONLY from original Control samples.
    
    This ensures experimental control items have appropriate content for attention checks
    (typically both options are safe/non-harmful).
    
    Args:
        data (List[Dict[str, Any]]): Full benchmark data
        total_needed (int): Number of control items needed
        
    Returns:
        List[Dict[str, Any]]: Sampled control items with balanced model distribution
        
    Raises:
        ValueError: If not enough original Control samples are available
    """
    # FIXED: Only use original Control samples for experimental control items
    control_samples = filter_control_samples(data)
    
    if len(control_samples) < total_needed:
        raise ValueError(
            f"Not enough original Control samples: need {total_needed}, "
            f"have {len(control_samples)}. Consider reducing control items per participant "
            f"or adding more Control samples to the benchmark."
        )
    
    # Group by model for balanced distribution
    by_model = defaultdict(list)
    for sample in control_samples:
        model_type = sample['metadata']['model_type']
        by_model[model_type].append(sample)
    
    models = sorted(by_model.keys())
    items_per_model = total_needed // len(models)
    remaining_items = total_needed % len(models)
    
    logging.info(f"Control items - Target model distribution: {items_per_model} per model, {remaining_items} extra")
    
    sampled_control = []
    
    # Sample from each model equally
    for model_idx, model in enumerate(models):
        model_samples = by_model[model]
        model_target = items_per_model + (1 if model_idx < remaining_items else 0)
        
        if len(model_samples) < model_target:
            logging.warning(
                f"Model {model} has only {len(model_samples)} Control samples, "
                f"need {model_target}. This may cause imbalanced distribution."
            )
            model_selected = model_samples  # Take all available
        else:
            model_selected = random.sample(model_samples, model_target)
        
        sampled_control.extend(model_selected)
        logging.info(f"Selected {len(model_selected)} control items for model {model}")
    
    # If we still need more items, sample randomly from remaining
    if len(sampled_control) < total_needed:
        remaining_needed = total_needed - len(sampled_control)
        used_samples = set(item['id'] for item in sampled_control)
        remaining_samples = [item for item in control_samples if item['id'] not in used_samples]
        
        if len(remaining_samples) < remaining_needed:
            raise ValueError(
                f"Cannot sample {remaining_needed} additional control items. "
                f"Only {len(remaining_samples)} unused Control samples remaining."
            )
        
        additional = random.sample(remaining_samples, remaining_needed)
        sampled_control.extend(additional)
    
    logging.info(f"Final control sample: {len(sampled_control)} items (all from original Control samples)")
    
    # Log model distribution
    final_model_dist = Counter(item['metadata']['model_type'] for item in sampled_control)
    logging.info(f"Control model distribution: {dict(final_model_dist)}")
    
    # Verify all sampled items are original Control samples
    non_control_items = [item for item in sampled_control if item['metadata']['sample_type'] != 'Control']
    if non_control_items:
        raise ValueError(f"BUG: Found {len(non_control_items)} non-Control items in control sample")
    
    return sampled_control


def log_sample_type_distribution(items: List[Dict[str, Any]]) -> None:
    """
    Log the distribution of original sample types vs experimental item types.
    
    Args:
        items (List[Dict[str, Any]]): Items with assignments
    """
    logging.info("=== SAMPLE TYPE vs ITEM TYPE DISTRIBUTION ===")
    
    # Create cross-tabulation
    distribution = defaultdict(lambda: defaultdict(int))
    
    for item in items:
        original_type = item['metadata']['sample_type']
        experimental_type = item.get('item_type', 'unknown')
        distribution[original_type][experimental_type] += 1
    
    # Log the distribution
    for original_type in sorted(distribution.keys()):
        logging.info(f"Original {original_type} samples:")
        for experimental_type, count in sorted(distribution[original_type].items()):
            logging.info(f"  Used as experimental {experimental_type}: {count}")
    
    logging.info("=== END DISTRIBUTION ===")


def validate_final_assignments(items: List[Dict[str, Any]]) -> None:
    """
    Validate final item assignments for logical consistency.
    
    Args:
        items (List[Dict[str, Any]]): Fully processed items
        
    Raises:
        ValueError: If validation fails
    """
    errors = []
    
    for item in items:
        item_id = item.get('item_id', 'unknown')
        original_sample_type = item['metadata']['sample_type']
        experimental_type = item.get('item_type', 'unknown')
        
        # Check experimental treatment items
        if experimental_type in ['treatment', 'realism_treatment']:
            if original_sample_type != 'Treatment':
                errors.append(f"Item {item_id}: Experimental {experimental_type} uses original {original_sample_type}")
        
        # Check experimental control items
        elif experimental_type in ['control', 'realism_control']:
            if original_sample_type != 'Control':
                errors.append(f"Item {item_id}: Experimental {experimental_type} uses original {original_sample_type}")
        
    
    if errors:
        error_msg = "Final assignment validation failed:\n" + "\n".join(errors)
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    logging.info(f"Final assignment validation passed for {len(items)} items")


def assign_participants_extended(
    all_items: List[Dict[str, Any]], 
    num_participants: int,
    treatment_per_participant: int,
    control_per_participant: int,
    realism_assignment_mode: str,
    realism_treatment_per_participant: int,
    realism_control_per_participant: int,
    treatment_ids: set,
    control_ids: set,
    realism_treatment_ids: set,
    realism_control_ids: set,
    start_participant_id: int = 1
) -> List[Dict[str, Any]]:
    """
    Assign sampled items to participants with balanced distribution including realism-only items.
    
    Args:
        all_items (List[Dict[str, Any]]): All sampled items
        num_participants (int): Number of participants
        treatment_per_participant (int): Treatment items per participant
        control_per_participant (int): Control items per participant
        realism_assignment_mode (str): The assignment mode ('per-participant' or 'global').
        realism_treatment_per_participant (int): Items per participant (for per-participant mode).
        realism_control_per_participant (int): Items per participant (for per-participant mode).
        treatment_ids (set): IDs of regular treatment items
        control_ids (set): IDs of regular control items
        realism_treatment_ids (set): IDs of realism-only treatment items
        realism_control_ids (set): IDs of realism-only control items
        start_participant_id (int): The starting ID for participants.
        
    Returns:
        List[Dict[str, Any]]: Items with participant assignments and item types
    """
    # Separate items by type
    treatment_samples = [item for item in all_items if item['id'] in treatment_ids]
    control_samples = [item for item in all_items if item['id'] in control_ids]
    realism_treatment_samples = [item for item in all_items if item['id'] in realism_treatment_ids]
    realism_control_samples = [item for item in all_items if item['id'] in realism_control_ids]
    
    # Shuffle for random assignment
    random.shuffle(treatment_samples)
    random.shuffle(control_samples)
    
    participant_items = []
    
    # Assign regular treatment items
    for i, item in enumerate(treatment_samples):
        participant_index = i // treatment_per_participant
        item['participant_id'] = start_participant_id + participant_index
        item['item_type'] = 'treatment'
        item['question_type'] = 'full'  # Both harm and realism questions
        participant_items.append(item)
    
    # Assign regular control items
    for i, item in enumerate(control_samples):
        participant_index = i // control_per_participant
        item['participant_id'] = start_participant_id + participant_index
        item['item_type'] = 'control'
        item['question_type'] = 'full'  # Both harm and realism questions
        participant_items.append(item)
    
    # Assign realism questions based on the chosen mode
    if realism_assignment_mode == 'global':
        # Shuffle all realism questions and deal one to each participant
        all_realism_samples = realism_treatment_samples + realism_control_samples
        random.shuffle(all_realism_samples)

        if len(all_realism_samples) < num_participants:
            logging.warning(
                f"Warning: Number of sampled realism questions ({len(all_realism_samples)}) "
                f"is less than the number of participants ({num_participants})."
            )

        for i in range(num_participants):
            participant_id = start_participant_id + i
            if i < len(all_realism_samples):
                item = all_realism_samples[i]
                item['participant_id'] = participant_id
                item['item_type'] = 'realism_treatment' if item['id'] in realism_treatment_ids else 'realism_control'
                item['question_type'] = 'realism_only'
                participant_items.append(item)
            else:
                logging.error(f"Could not assign realism question to participant {participant_id}, not enough samples.")
    
    elif realism_assignment_mode == 'per-participant':
        # Assign a fixed number of each realism type to each participant
        random.shuffle(realism_treatment_samples)
        random.shuffle(realism_control_samples)

        for i, item in enumerate(realism_treatment_samples):
            participant_index = i // realism_treatment_per_participant
            item['participant_id'] = start_participant_id + participant_index
            item['item_type'] = 'realism_treatment'
            item['question_type'] = 'realism_only'
            participant_items.append(item)

        for i, item in enumerate(realism_control_samples):
            participant_index = i // realism_control_per_participant
            item['participant_id'] = start_participant_id + participant_index
            item['item_type'] = 'realism_control'
            item['question_type'] = 'realism_only'
            participant_items.append(item)

    
    logging.info(f"Assigned items to {num_participants} participants, starting from ID {start_participant_id}")
    
    # Log assignment statistics
    treatment_assigned = len([item for item in participant_items if item['item_type'] == 'treatment'])
    control_assigned = len([item for item in participant_items if item['item_type'] == 'control'])
    realism_treatment_assigned = len([item for item in participant_items if item['item_type'] == 'realism_treatment'])
    realism_control_assigned = len([item for item in participant_items if item['item_type'] == 'realism_control'])
    
    logging.info(f"Assignment: {treatment_assigned} treatment, {control_assigned} control")
    logging.info(f"Realism-only: {realism_treatment_assigned} treatment, {realism_control_assigned} control")
    
    return participant_items


def assign_conditions_and_metadata(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assign experimental conditions and metadata to sampled items.
    
    This includes:
    - Random harm levels from HARM_LEVELS for the harmful option (never 0%)
    - The other option gets 0% harm (safe option)
    - Random benefit levels (15%, 40%, 65%) with placeholder replacement
    - Balanced assignment of which option is harmful (exactly 50/50 at all levels)
    - Consistent option randomization for both treatment and control items
    - Unique item IDs
    - Content validation to ensure consistency
    
    Args:
        items (List[Dict[str, Any]]): Items with participant assignments
        
    Returns:
        List[Dict[str, Any]]: Items with full experimental metadata
        
    Raises:
        ValueError: If content validation fails
    """
    # First, assign item IDs
    for idx, item in enumerate(items):
        item['item_id'] = f"item_{idx + 1:03d}"
    
    # ADDED: Validate content consistency before processing
    validate_item_content_consistency(items)
    
    # ADDED: Log sample type distribution
    log_sample_type_distribution(items)
    
    # Create balanced option assignments
    logging.info("Creating balanced option assignments...")
    option_assignments = create_balanced_option_assignments(items)
    
    enriched_items = []
    
    for item in items:
        # Create enriched copy
        enriched = item.copy()
        
        # Process text with benefit placeholder first, then add harm percentages
        benefit_level = random.choice(BENEFIT_LEVELS)
        enriched['benefit_level'] = benefit_level
        
        # Get pre-determined assignment for this item
        swap_options = option_assignments[item['item_id']]
        
        # FIXED: Only one option gets harm percentage from HARM_LEVELS, other gets 0%
        harm_level = random.choice(HARM_LEVELS)  # This is never 0%
        enriched['harm_level'] = harm_level
        
        
        # Process text with both benefit and harm placeholders
        # For scenario text, use the harm level
        processed_scenario = process_text_placeholders(enriched['scenario'], benefit_level, harm_level)
        
        processed_option_a = process_text_placeholders(enriched['option_a'], benefit_level, harm_level)
        
        processed_option_b = process_text_placeholders(enriched['option_b'], benefit_level, harm_level)
        
        # Apply option swapping if needed
        if swap_options:
            # Swap options and update text references
            processed_option_a, processed_option_b = swap_options_and_update_text(
                processed_option_a, processed_option_b
            )

            # B  (harmful option)
            enriched['harmful_option'] = 'B'
            enriched['option_swapped'] = True
        else:
            # A (harmful option)
            enriched['harmful_option'] = 'A' 
            enriched['option_swapped'] = False
        
        # Store processed text
        enriched['processed_scenario'] = processed_scenario
        enriched['processed_option_a'] = processed_option_a
        enriched['processed_option_b'] = processed_option_b
        
        enriched_items.append(enriched)
    
    logging.info(f"Assigned conditions to {len(enriched_items)} items")
    
    # ADDED: Final validation after processing
    validate_final_assignments(enriched_items)
    
    # Log final statistics
    log_final_balance_statistics(enriched_items)
    
    return enriched_items


def swap_options_and_update_text(option_a: str, option_b: str) -> Tuple[str, str]:
    """
    Swap the options and update internal text references.
    
    This function swaps option A and B content while also updating
    any internal references to "Option A" or "Option B" within the text.
    
    Args:
        option_a (str): Original option A text
        option_b (str): Original option B text
        
    Returns:
        Tuple[str, str]: Swapped and text-corrected options (new_A, new_B)
    """
    # Function to replace option references in text
    def replace_option_references(text: str, original_option: str, new_option: str) -> str:
        # Replace various formats of option references
        patterns = [
            rf'\*\*Option {original_option}\*\*:',  # **Option A**:
            rf'\*\*Option {original_option}\*\*',   # **Option A**
            rf'Option {original_option}:',           # Option A:
            rf'Option {original_option}',            # Option A
            rf'option {original_option}',            # option A (lowercase)
        ]
        
        replacements = [
            f'**Option {new_option}**:',
            f'**Option {new_option}**',
            f'Option {new_option}:',
            f'Option {new_option}',
            f'option {new_option}',
        ]
        
        result = text
        for pattern, replacement in zip(patterns, replacements):
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    # Swap the options and update their internal references
    # What was originally A becomes B, and what was originally B becomes A
    new_option_a = replace_option_references(option_b, 'B', 'A')
    new_option_b = replace_option_references(option_a, 'A', 'B')
    
    return new_option_a, new_option_b


def create_balanced_option_assignments(items: List[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Create balanced option assignments ensuring exactly 50% A and 50% B harmful options
    at multiple levels: overall, by item type, and by model type.
    
    Args:
        items (List[Dict[str, Any]]): Items with participant assignments
        
    Returns:
        Dict[str, bool]: Dictionary mapping item_id to swap_options boolean
    """
    # Group items by type and model
    treatment_by_model = defaultdict(list)
    control_by_model = defaultdict(list)
    realism_treatment_by_model = defaultdict(list)
    realism_control_by_model = defaultdict(list)
    
    for item in items:
        model_type = item['metadata']['model_type']
        if item['item_type'] == 'treatment':
            treatment_by_model[model_type].append(item)
        elif item['item_type'] == 'control':
            control_by_model[model_type].append(item)
        elif item['item_type'] == 'realism_treatment':
            realism_treatment_by_model[model_type].append(item)
        elif item['item_type'] == 'realism_control':
            realism_control_by_model[model_type].append(item)
    
    assignment_dict = {}
    
    # Create balanced assignments for treatment items by model
    logging.info("Creating balanced treatment assignments by model:")
    for model_type, model_items in treatment_by_model.items():
        count = len(model_items)
        # Exactly half should be swapped (harmful option becomes B)
        num_swapped = count // 2
        num_not_swapped = count - num_swapped
        
        # Create balanced assignment list
        assignments = [True] * num_swapped + [False] * num_not_swapped
        random.shuffle(assignments)  # Randomize order but maintain exact counts
        
        # Assign to items
        for item, swap in zip(model_items, assignments):
            assignment_dict[item['item_id']] = swap
        
        logging.info(f"  {model_type}: {count} items - {num_swapped} swapped (B harmful), {num_not_swapped} not swapped (A harmful)")
    
    # Create balanced assignments for control items by model
    logging.info("Creating balanced control assignments by model:")
    for model_type, model_items in control_by_model.items():
        count = len(model_items)
        # Exactly half should be swapped (designated "harmful" option becomes B)
        num_swapped = count // 2
        num_not_swapped = count - num_swapped
        
        # Create balanced assignment list
        assignments = [True] * num_swapped + [False] * num_not_swapped
        random.shuffle(assignments)  # Randomize order but maintain exact counts
        
        # Assign to items
        for item, swap in zip(model_items, assignments):
            assignment_dict[item['item_id']] = swap
        
        logging.info(f"  {model_type}: {count} items - {num_swapped} swapped (B 'harmful'), {num_not_swapped} not swapped (A 'harmful')")
    
    # Create balanced assignments for realism-only treatment items by model
    logging.info("Creating balanced realism-treatment assignments by model:")
    for model_type, model_items in realism_treatment_by_model.items():
        count = len(model_items)
        # Exactly half should be swapped (designated "harmful" option becomes B)
        num_swapped = count // 2
        num_not_swapped = count - num_swapped
        
        # Create balanced assignment list
        assignments = [True] * num_swapped + [False] * num_not_swapped
        random.shuffle(assignments)  # Randomize order but maintain exact counts
        
        # Assign to items
        for item, swap in zip(model_items, assignments):
            assignment_dict[item['item_id']] = swap
        
        logging.info(f"  {model_type}: {count} items - {num_swapped} swapped (B 'harmful'), {num_not_swapped} not swapped (A 'harmful')")
    
    # Create balanced assignments for realism-only control items by model
    logging.info("Creating balanced realism-control assignments by model:")
    for model_type, model_items in realism_control_by_model.items():
        count = len(model_items)
        # Exactly half should be swapped (designated "harmful" option becomes B)
        num_swapped = count // 2
        num_not_swapped = count - num_swapped
        
        # Create balanced assignment list
        assignments = [True] * num_swapped + [False] * num_not_swapped
        random.shuffle(assignments)  # Randomize order but maintain exact counts
        
        # Assign to items
        for item, swap in zip(model_items, assignments):
            assignment_dict[item['item_id']] = swap
        
        logging.info(f"  {model_type}: {count} items - {num_swapped} swapped (B 'harmful'), {num_not_swapped} not swapped (A 'harmful')")
    
    # Verify overall balance
    treatment_items = [item for item in items if item['item_type'] == 'treatment']
    control_items = [item for item in items if item['item_type'] == 'control']
    realism_treatment_items = [item for item in items if item['item_type'] == 'realism_treatment']
    realism_control_items = [item for item in items if item['item_type'] == 'realism_control']
    
    treatment_swapped = sum(1 for item in treatment_items if assignment_dict[item['item_id']])
    control_swapped = sum(1 for item in control_items if assignment_dict[item['item_id']])
    realism_treatment_swapped = sum(1 for item in realism_treatment_items if assignment_dict[item['item_id']])
    realism_control_swapped = sum(1 for item in realism_control_items if assignment_dict[item['item_id']])
    
    logging.info(f"Overall balance verification:")
    logging.info(f"  Treatment: {len(treatment_items)} total - {treatment_swapped} B harmful, {len(treatment_items)-treatment_swapped} A harmful")
    logging.info(f"  Control: {len(control_items)} total - {control_swapped} B 'harmful', {len(control_items)-control_swapped} A 'harmful'")
    logging.info(f"  Realism-Treatment: {len(realism_treatment_items)} total - {realism_treatment_swapped} B 'harmful', {len(realism_treatment_items)-realism_treatment_swapped} A 'harmful'")
    logging.info(f"  Realism-Control: {len(realism_control_items)} total - {realism_control_swapped} B 'harmful', {len(realism_control_items)-realism_control_swapped} A 'harmful'")
    
    return assignment_dict


def log_final_balance_statistics(items: List[Dict[str, Any]]) -> None:
    """
    Log comprehensive balance statistics for the final item assignments.
    
    Args:
        items (List[Dict[str, Any]]): Items with full metadata
    """
    # Overall statistics
    treatment_items = [item for item in items if item['item_type'] == 'treatment']
    control_items = [item for item in items if item['item_type'] == 'control']
    realism_treatment_items = [item for item in items if item['item_type'] == 'realism_treatment']
    realism_control_items = [item for item in items if item['item_type'] == 'realism_control']
    
    logging.info("=== FINAL BALANCE STATISTICS ===")
    
    # Treatment items
    if treatment_items:
        swapped_count = sum(1 for item in treatment_items if item.get('option_swapped', False))
        harmful_a_count = sum(1 for item in treatment_items if item.get('harmful_option') == 'A')
        harmful_b_count = sum(1 for item in treatment_items if item.get('harmful_option') == 'B')
        
        logging.info(f"TREATMENT ITEMS ({len(treatment_items)} total):")
        logging.info(f"  Options swapped: {swapped_count}/{len(treatment_items)} ({swapped_count/len(treatment_items)*100:.1f}%)")
        logging.info(f"  Harmful option A: {harmful_a_count} ({harmful_a_count/len(treatment_items)*100:.1f}%)")
        logging.info(f"  Harmful option B: {harmful_b_count} ({harmful_b_count/len(treatment_items)*100:.1f}%)")
        
        # By model
        by_model = defaultdict(list)
        for item in treatment_items:
            by_model[item['metadata']['model_type']].append(item)
        
        logging.info(f"  By model:")
        for model, model_items in sorted(by_model.items()):
            model_a = sum(1 for item in model_items if item.get('harmful_option') == 'A')
            model_b = sum(1 for item in model_items if item.get('harmful_option') == 'B')
            logging.info(f"    {model}: {len(model_items)} items - A harmful: {model_a}, B harmful: {model_b}")
    
    # Control items
    if control_items:
        swapped_count = sum(1 for item in control_items if item.get('option_swapped', False))
        harmful_a_count = sum(1 for item in control_items if item.get('harmful_option') == 'A')
        harmful_b_count = sum(1 for item in control_items if item.get('harmful_option') == 'B')
        
        logging.info(f"CONTROL ITEMS ({len(control_items)} total):")
        logging.info(f"  Options swapped: {swapped_count}/{len(control_items)} ({swapped_count/len(control_items)*100:.1f}%)")
        logging.info(f"  'Harmful' option A: {harmful_a_count} ({harmful_a_count/len(control_items)*100:.1f}%)")
        logging.info(f"  'Harmful' option B: {harmful_b_count} ({harmful_b_count/len(control_items)*100:.1f}%)")
        
        # By model
        by_model = defaultdict(list)
        for item in control_items:
            by_model[item['metadata']['model_type']].append(item)
        
        logging.info(f"  By model:")
        for model, model_items in sorted(by_model.items()):
            model_a = sum(1 for item in model_items if item.get('harmful_option') == 'A')
            model_b = sum(1 for item in model_items if item.get('harmful_option') == 'B')
            logging.info(f"    {model}: {len(model_items)} items - A 'harmful': {model_a}, B 'harmful': {model_b}")
    
    # Realism-only treatment items
    if realism_treatment_items:
        swapped_count = sum(1 for item in realism_treatment_items if item.get('option_swapped', False))
        harmful_a_count = sum(1 for item in realism_treatment_items if item.get('harmful_option') == 'A')
        harmful_b_count = sum(1 for item in realism_treatment_items if item.get('harmful_option') == 'B')
        
        logging.info(f"REALISM-TREATMENT ITEMS ({len(realism_treatment_items)} total):")
        logging.info(f"  Options swapped: {swapped_count}/{len(realism_treatment_items)} ({swapped_count/len(realism_treatment_items)*100:.1f}%)")
        logging.info(f"  'Harmful' option A: {harmful_a_count} ({harmful_a_count/len(realism_treatment_items)*100:.1f}%)")
        logging.info(f"  'Harmful' option B: {harmful_b_count} ({harmful_b_count/len(realism_treatment_items)*100:.1f}%)")
    
    # Realism-only control items
    if realism_control_items:
        swapped_count = sum(1 for item in realism_control_items if item.get('option_swapped', False))
        harmful_a_count = sum(1 for item in realism_control_items if item.get('harmful_option') == 'A')
        harmful_b_count = sum(1 for item in realism_control_items if item.get('harmful_option') == 'B')
        
        logging.info(f"REALISM-CONTROL ITEMS ({len(realism_control_items)} total):")
        logging.info(f"  Options swapped: {swapped_count}/{len(realism_control_items)} ({swapped_count/len(realism_control_items)*100:.1f}%)")
        logging.info(f"  'Harmful' option A: {harmful_a_count} ({harmful_a_count/len(realism_control_items)*100:.1f}%)")
        logging.info(f"  'Harmful' option B: {harmful_b_count} ({harmful_b_count/len(realism_control_items)*100:.1f}%)")
    
    logging.info("=== END BALANCE STATISTICS ===")


def process_text_placeholders(text: str, benefit_level: int, harm_level: int) -> str:
    """
    Replace placeholders in text with actual values and prepare for HTML conversion.
    
    Args:
        text (str): Text potentially containing placeholders
        benefit_level (int): Benefit percentage to insert
        harm_level (int): Harm percentage to insert
        
    Returns:
        str: Text with placeholders replaced, ready for HTML conversion
    """
    # Replace benefit percentage placeholder
    processed = text.replace('{benefit_percentage}%', f'{benefit_level}%')
    processed = processed.replace('{benefit_percentage}', str(benefit_level))
    
    # Replace harm percentage placeholder
    processed = processed.replace('{harm_percentage}%', f'{harm_level}%')
    processed = processed.replace('{harm_percentage}', str(harm_level))
    
    # Clean up any formatting issues
    processed = processed.strip()
    
    return processed


def write_metadata_csv(
    items: List[Dict[str, Any]], 
    output_path: str,
    instructions: Dict[str, Any]
) -> None:
    """
    Write comprehensive metadata CSV for linking responses to item details.
    
    Args:
        items (List[Dict[str, Any]]): Items with full metadata
        output_path (str): Path to save CSV file
        instructions (Dict[str, Any]): The loaded instructions content.
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    
    for item in items:
        metadata = item['metadata']
        
        # Construct full question text for easy copy-pasting
        question_type = item.get('question_type', 'full')
        if question_type == 'full':
            scenario_instruction = instructions['scenario_instructions']['full_questions']
            rating_question = instructions['harm_assessment']['question']
            explanation_question = instructions['explanation']['question']
        else: # realism_only
            scenario_instruction = instructions['scenario_instructions']['realism_only']
            rating_question = instructions['realism_assessment']['question']
            explanation_question = instructions.get('realism_explanation', {}).get('question', "Please briefly explain your reasoning for the realism assessment above.")

        full_question_text = (
            # f"{scenario_instruction}\n\n"
            # f"Scenario: {item['processed_scenario']}\n\n"
            # f"Option A: {item['processed_option_a']}\n\n"
            # f"Option B: {item['processed_option_b']}\n\n"
            # f"Question: {rating_question}\n\n"
            # f"Question: {explanation_question}"
            f"{item['processed_scenario']}\n\n"
            f"{item['processed_option_a']}\n\n"
            f"{item['processed_option_b']}\n\n"
        )
        # Clean up HTML tags for pure text representation
        full_question_text = re.sub(r'<br>', '\n', full_question_text)
        full_question_text = re.sub(r'<b>(.*?)</b>', r'\\1', full_question_text, flags=re.IGNORECASE)
        full_question_text = full_question_text.strip()
        
        row = {
            'item_id': item['item_id'],
            'participant_id': item['participant_id'],
            'participants_question_id': item.get('participants_question_id', ''),
            'item_type': item['item_type'],
            'original_sample_id': item['id'],
            
            # Experimental conditions
            'harm_level': item.get('harm_level', 0),
            'benefit_level': item.get('benefit_level', 0),
            'harmful_option': item.get('harmful_option', 'Neither'),
            'option_swapped': item.get('option_swapped', False),
            
            # Original metadata
            'model_type': metadata['model_type'],
            'original_sample_type': metadata['sample_type'],
            'topic': metadata['topic'],
            'harm_type': metadata['harm_type'],
            'benefit_reason': metadata['benefit'],
            
            # Text content (processed)
            'scenario': item['processed_scenario'],
            'option_a': item['processed_option_a'],
            'option_b': item['processed_option_b'],
            'full_question_text': full_question_text,
            
            # Original text for reference
            'original_scenario': item['scenario'],
            'original_option_a': item['option_a'],
            'original_option_b': item['option_b'],
        }
        
        csv_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    df = df.sort_values(['participant_id', 'participants_question_id'])
    df.to_csv(output_path, index=False)
    
    logging.info(f"Saved experiment plan to {output_path}")
    
    # Print summary statistics
    print("\n=== EXPERIMENT PLAN SUMMARY ===")
    print(f"Total items: {len(df)}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Items per participant: {len(df) // df['participant_id'].nunique()}")
    
    print(f"\nBy item type:")
    for item_type, count in df['item_type'].value_counts().items():
        print(f"  {item_type}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nBy participant:")
    for pid in sorted(df['participant_id'].unique()):
        p_data = df[df['participant_id'] == pid]
        treatment_count = len(p_data[p_data['item_type'] == 'treatment'])
        control_count = len(p_data[p_data['item_type'] == 'control'])
        realism_treatment_count = len(p_data[p_data['item_type'] == 'realism_treatment'])
        realism_control_count = len(p_data[p_data['item_type'] == 'realism_control'])
        total_count = treatment_count + control_count + realism_treatment_count + realism_control_count
        print(f"  Participant {pid}: {treatment_count} treatment + {control_count} control + {realism_treatment_count} realism-treatment + {realism_control_count} realism-control = {total_count} total")
    
    print(f"\nTopic distribution (treatment only):")
    treatment_df = df[df['item_type'] == 'treatment']
    if len(treatment_df) > 0:
        for topic, count in treatment_df['topic'].value_counts().sort_index().items():
            print(f"  {topic}: {count}")
    
    print(f"\nHarm level distribution (treatment only):")
    if len(treatment_df) > 0:
        for harm_level, count in treatment_df['harm_level'].value_counts().sort_index().items():
            print(f"  {harm_level}%: {count}")
    
    print(f"\nOption swapping (treatment only):")
    if len(treatment_df) > 0:
        swapped_count = len(treatment_df[treatment_df['option_swapped'] == True])
        not_swapped_count = len(treatment_df[treatment_df['option_swapped'] == False])
        print(f"  Swapped: {swapped_count} ({swapped_count/len(treatment_df)*100:.1f}%)")
        print(f"  Not swapped: {not_swapped_count} ({not_swapped_count/len(treatment_df)*100:.1f}%)")
    
    print(f"\nHarmful option distribution (treatment only):")
    if len(treatment_df) > 0:
        for harmful_option, count in treatment_df['harmful_option'].value_counts().sort_index().items():
            print(f"  Option {harmful_option}: {count} ({count/len(treatment_df)*100:.1f}%)")


def generate_qualtrics_txts(
    items: List[Dict[str, Any]], 
    instructions: Dict[str, Any],
    output_dir: str = "qualtrics_files"
) -> None:
    """
    Generate individual Qualtrics .txt files for each participant.
    
    Args:
        items (List[Dict[str, Any]]): Items with full metadata. The order is preserved.
        instructions (Dict[str, Any]): The loaded instructions content.
        output_dir (str): Directory to save Qualtrics files.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group items by participant
    by_participant = defaultdict(list)
    for item in items:
        by_participant[item['participant_id']].append(item)
    
    # Generate file for each participant
    for participant_id, participant_items in by_participant.items():
        filename = f"participant_{participant_id:02d}_qualtrics.txt"
        filepath = os.path.join(output_dir, filename)
        
        generate_participant_qualtrics_file(participant_items, filepath, participant_id, instructions)
    
    logging.info(f"Generated Qualtrics files for {len(by_participant)} participants in {output_dir}")


def generate_example_page(example_data: Dict[str, Any], page_id: str) -> List[str]:
    """
    Generate a Qualtrics example page from JSON example data.
    
    Args:
        example_data (Dict[str, Any]): Example data from instructions JSON
        page_id (str): Unique page identifier
        
    Returns:
        List[str]: Lines for the example page in Advanced Format
    """
    lines = []
    
    # Example page as Text/Graphic question
    lines.append("[[Question:Text]]")
    lines.append(f"[[ID:{page_id}]]")
    
    # Add title
    if 'title' in example_data:
        title_html = convert_markdown_to_html(example_data['title'])
        lines.append(title_html)
        lines.append("<br><br>")
    
    # Add instruction
    if 'instruction' in example_data:
        instruction_html = convert_markdown_to_html(example_data['instruction'])
        lines.append(instruction_html)
        lines.append("<br><br>")
    
    # Add example content
    if 'example' in example_data:
        example_html = convert_markdown_to_html(example_data['example'])
        lines.append(example_html)
    
    lines.append("")
    
    return lines


def generate_participant_qualtrics_file(
    items: List[Dict[str, Any]], 
    filepath: str, 
    participant_id: int,
    instructions: Dict[str, Any]
) -> None:
    """
    Generate a single participant's Qualtrics .txt file using Advanced Format.
    Includes example pages loaded from the instructions JSON file.
    
    Args:
        items (List[Dict[str, Any]]): Items for this participant
        filepath (str): Output file path
        participant_id (int): Participant ID
        instructions (Dict[str, Any]): Instructions text content
    """
    lines = []
    
    # Advanced format header
    lines.append("[[AdvancedFormat]]")
    lines.append("")
    
    # Add embedded data for participant tracking
    lines.append(f"[[ED:participant_id:{participant_id}]]")
    lines.append("")
    
    # Introduction block
    block_title = instructions['labels']['block_title_template'].format(participant_id=participant_id)
    lines.append(f"[[Block:{block_title}]]")
    lines.append("")
    
    # Introduction text/graphic question
    lines.append("[[Question:Text]]")
    lines.append(instructions['introduction']['title'])
    lines.append("")
    lines.append(instructions['introduction']['description'])
    lines.append(instructions['introduction']['instruction'])
    lines.append("")
    lines.append("[[PageBreak]]")
    lines.append("")
    
    # Items are pre-shuffled; just separate them by phase
    full_items = [item for item in items if item.get('question_type', 'full') == 'full']
    realism_only_items = [item for item in items if item.get('question_type', 'full') == 'realism_only']
    
    # Add harm assessment example page (before Phase 1)
    if full_items and 'harm_example_page' in instructions:
        logging.info(f"Adding harm assessment example page for participant {participant_id}")
        lines.extend(generate_example_page(instructions['harm_example_page'], 'HARM_EXAMPLE'))
        lines.append("[[PageBreak]]")
        lines.append("")
    
    # Phase 1: Harm Assessment Questions (only for 'full' items)
    harm_question_counter = 1
    for item in full_items:
        question_id_base = f"Q{harm_question_counter:02d}"
        
        # Add scenario and harm assessment questions only
        lines.extend(generate_harm_questions(item, question_id_base, harm_question_counter, instructions))
        
        # Add page break between items (except last)
        if harm_question_counter < len(full_items):
            lines.append("[[PageBreak]]")
            lines.append("")
        
        harm_question_counter += 1
    
    # Phase 2 Introduction: Transition to Realism Assessment
    if full_items and realism_only_items:  # Only add realism intro if there are both phases
        lines.append("[[PageBreak]]")
        lines.append("")
        lines.append("[[Question:Text]]")
        lines.append(instructions['realism_intro']['title'])
        lines.append("")
        lines.append(instructions['realism_intro']['instruction'])
        lines.append("")
        lines.append("[[PageBreak]]")
        lines.append("")
        
        # Add realism assessment example page (after realism intro, before Phase 2)
        if 'realism_example_page' in instructions:
            logging.info(f"Adding realism assessment example page for participant {participant_id}")
            lines.extend(generate_example_page(instructions['realism_example_page'], 'REALISM_EXAMPLE'))
            lines.append("[[PageBreak]]")
            lines.append("")
    
    # Phase 2: Realism Assessment Questions (only for realism-only items)
    realism_question_counter = 1
    
    for item in realism_only_items:
        question_id_base = f"R{realism_question_counter:02d}"
        
        # Add scenario and realism assessment questions only
        lines.extend(generate_realism_questions(item, question_id_base, realism_question_counter, instructions))
        
        # Add page break between items (except last)
        if realism_question_counter < len(realism_only_items):
            lines.append("[[PageBreak]]")
            lines.append("")
        
        realism_question_counter += 1
    
    # Add Demographics Questions at the end of the survey
    lines.extend(generate_demographics_questions(instructions))
    
    # Closing block
    lines.append(f"[[Block:{instructions['labels']['completion_block']}]]")
    lines.append("")
    lines.append("[[Question:Text]]")
    lines.append(instructions['completion']['title'])
    lines.append("")
    lines.append(instructions['completion']['message'])
    lines.append("")
    
    # Write file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logging.info(f"Generated Qualtrics file for participant {participant_id}: {filepath} (with example pages)")


def convert_markdown_to_html(text: str) -> str:
    """
    Convert Markdown formatting to HTML formatting for Qualtrics.
    
    Args:
        text (str): Text with Markdown formatting
        
    Returns:
        str: Text with HTML formatting
    """
    # Replace bold markdown with HTML bold tags
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # Replace line breaks with HTML line breaks
    # Split by newlines and join with <br>
    lines = text.split('\n')
    
    # Handle header lines (lines starting with #)
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            # Remove # characters and any following spaces
            content = re.sub(r'^#+\s*', '', line.strip())
            if content:  # Only if there's content after removing #
                lines[i] = f'<b>{content}</b>'
            else:
                lines[i] = line  # Keep original if no content
    
    # Only add <br> between non-empty lines to avoid excessive spacing
    html_lines = []
    for i, line in enumerate(lines):
        html_lines.append(line)
        # Add <br> if this isn't the last line and either this line or next line has content
        if i < len(lines) - 1 and (line.strip() or lines[i + 1].strip()):
            html_lines.append('<br>')
    
    result = ''.join(html_lines)
    
    # Clean up multiple consecutive <br> tags
    result = re.sub(r'(<br>\s*){3,}', '<br><br>', result)
    
    return result.strip()


def generate_harm_questions(item: Dict[str, Any], question_id_base: str, item_number: int, instructions: Dict[str, Any]) -> List[str]:
    """
    Generate Qualtrics Advanced Format questions for harm assessment phase only.
    
    Args:
        item (Dict[str, Any]): Item with full metadata
        question_id_base (str): Question ID base (e.g., "Q01")
        item_number (int): Item number for display
        instructions (Dict[str, Any]): Instructions text content
        
    Returns:
        List[str]: Lines for this item's harm assessment questions in Advanced Format
    """
    lines = []
    
    # Convert text content to HTML
    scenario_html = convert_markdown_to_html(item['processed_scenario'])
    option_a_html = convert_markdown_to_html(item['processed_option_a'])
    option_b_html = convert_markdown_to_html(item['processed_option_b'])
    
    # Question 1: Scenario presentation (Text/Graphic)
    lines.append(f"[[Question:Text]]")
    lines.append(f"[[ID:{question_id_base}A]]")
    lines.append(f"<b>{instructions['labels']['scenario_prefix']} {item_number}</b>")
    lines.append("<br>")
    lines.append(f"{instructions['scenario_instructions']['full_questions']}<br><br>")
    lines.append(scenario_html)
    lines.append("<br>")
    lines.append(option_a_html)
    lines.append("<br>")
    lines.append(option_b_html)
    lines.append("")
    
    # Question 2: Harm comparison (Matrix from 1-7)
    lines.append(f"[[Question:Matrix:SingleAnswer]]")
    lines.append(f"[[ID:{question_id_base}B]]")
    lines.append(instructions['harm_assessment']['question'])
    lines.append("[[Choices]]")
    lines.append(instructions['harm_assessment']['choice_label'])
    lines.append("[[Answers]]")
    for label in instructions['harm_assessment']['scale_labels']:
        lines.append(label)
    lines.append("")
    
    # Question 3: Explanation (Text Entry)
    lines.append(f"[[Question:TE:Essay]]")
    lines.append(f"[[ID:{question_id_base}C]]")
    lines.append(instructions['explanation']['question'])
    lines.append("")
    
    return lines


def generate_realism_questions(item: Dict[str, Any], question_id_base: str, item_number: int, instructions: Dict[str, Any]) -> List[str]:
    """
    Generate Qualtrics Advanced Format questions for realism assessment phase only.
    
    Args:
        item (Dict[str, Any]): Item with full metadata
        question_id_base (str): Question ID base (e.g., "R01")
        item_number (int): Item number for display
        instructions (Dict[str, Any]): Instructions text content
        
    Returns:
        List[str]: Lines for this item's realism assessment questions in Advanced Format
    """
    lines = []
    
    # Convert text content to HTML
    scenario_html = convert_markdown_to_html(item['processed_scenario'])
    option_a_html = convert_markdown_to_html(item['processed_option_a'])
    option_b_html = convert_markdown_to_html(item['processed_option_b'])
    
    # Question 1: Scenario presentation (Text/Graphic)
    lines.append(f"[[Question:Text]]")
    lines.append(f"[[ID:{question_id_base}A]]")
    lines.append(f"<b>{instructions['labels']['scenario_prefix']} {item_number}</b>")
    lines.append("<br>")
    
    # Choose appropriate instruction based on question type
    question_type = item.get('question_type', 'full')
    if question_type == 'full':
        instruction_text = instructions['scenario_instructions']['full_questions']
    else:
        instruction_text = instructions['scenario_instructions']['realism_only']
    
    lines.append(f"{instruction_text}<br><br>")
    lines.append(scenario_html)
    lines.append("<br>")
    lines.append(option_a_html)
    lines.append("<br>")
    lines.append(option_b_html)
    lines.append("")
    
    # Question 2: Realism assessment (Matrix from 1-7)
    lines.append(f"[[Question:Matrix:SingleAnswer]]")
    lines.append(f"[[ID:{question_id_base}B]]")
    lines.append(instructions['realism_assessment']['question'])
    lines.append("[[Choices]]")
    lines.append(instructions['realism_assessment']['choice_label'])
    lines.append("[[Answers]]")
    for label in instructions['realism_assessment']['scale_labels']:
        lines.append(label)
    lines.append("")
    
    # Question 3: Explanation (Text Entry)
    lines.append(f"[[Question:TE:Essay]]")
    lines.append(f"[[ID:{question_id_base}C]]")
    # Use realism explanation if available, otherwise adapt the general explanation
    realism_explanation = instructions.get('realism_explanation', {}).get('question', 
                                         "Please briefly explain your reasoning for the realism assessment above.")
    lines.append(realism_explanation)
    lines.append("")
    
    return lines


def generate_demographics_questions(instructions: Dict[str, Any]) -> List[str]:
    """
    Generate Qualtrics Advanced Format questions for the demographics section.

    Args:
        instructions (Dict[str, Any]): The instructions dictionary, containing the 'demographics' section.

    Returns:
        List[str]: A list of strings representing the Qualtrics questions.
    """
    if 'demographics' not in instructions:
        return []

    demographics = instructions['demographics']
    lines = []

    # Add a page break before demographics
    lines.append("[[PageBreak]]")
    lines.append("")

    # Title and instruction for the section
    lines.append("[[Question:Text]]")
    lines.append(f"[[ID:DemographicsIntro]]")
    lines.append(demographics.get('title', ''))
    lines.append("<br>")
    lines.append(demographics.get('instruction', ''))
    lines.append("")

    # Generate each demographic question
    for i, q_data in enumerate(demographics.get('questions', [])):
        q_id = q_data.get('id', f'demographic_q{i+1}')
        q_text = q_data.get('question', '')
        q_type = q_data.get('type', 'TE')

        if q_type == 'MC':
            lines.append("[[Question:MC:SingleAnswer]]")
            lines.append(f"[[ID:{q_id}]]")
            lines.append(q_text)
            lines.append("[[Choices]]")
            for choice in q_data.get('choices', []):
                lines.append(choice)
            lines.append("")
        elif q_type == 'TE':
            lines.append("[[Question:TE:Short]]")
            lines.append(f"[[ID:{q_id}]]")
            lines.append(q_text)
            lines.append("")

    return lines


def generate_item_questions(item: Dict[str, Any], question_id_base: str, item_number: int, instructions: Dict[str, Any]) -> List[str]:
    """
    Generate Qualtrics Advanced Format questions for a single item with HTML formatting.
    
    Args:
        item (Dict[str, Any]): Item with full metadata
        question_id_base (str): Question ID base (e.g., "Q01")
        item_number (int): Item number for display
        instructions (Dict[str, Any]): Instructions text content
        
    Returns:
        List[str]: Lines for this item's questions in Advanced Format
    """
    lines = []
    
    # Convert text content to HTML
    scenario_html = convert_markdown_to_html(item['processed_scenario'])
    option_a_html = convert_markdown_to_html(item['processed_option_a'])
    option_b_html = convert_markdown_to_html(item['processed_option_b'])
    
    # Question 1: Scenario presentation (Text/Graphic)
    lines.append(f"[[Question:Text]]")
    lines.append(f"[[ID:{question_id_base}A]]")
    lines.append(f"<b>{instructions['labels']['scenario_prefix']} {item_number}</b>")
    lines.append("<br>")
    
    # Choose appropriate instruction based on question type
    question_type = item.get('question_type', 'full')
    if question_type == 'full':
        instruction_text = instructions['scenario_instructions']['full_questions']
    else:
        instruction_text = instructions['scenario_instructions']['realism_only']
    
    lines.append(f"{instruction_text}<br><br>")
    lines.append(scenario_html)
    lines.append("<br>")
    lines.append(option_a_html)
    lines.append("<br>")
    lines.append(option_b_html)
    lines.append("")
    
    # Conditionally add harm assessment questions (only for 'full' question type)
    if question_type == 'full':
        # Question 2: Harm comparison (Slider from 1-7)
        lines.append(f"[[Question:Matrix:SingleAnswer]]")
        lines.append(f"[[ID:{question_id_base}B]]")
        lines.append(instructions['harm_assessment']['question'])
        lines.append("[[Choices]]")
        lines.append(instructions['harm_assessment']['choice_label'])
        lines.append("[[Answers]]")
        for label in instructions['harm_assessment']['scale_labels']:
            lines.append(label)
        lines.append("")
        
        # Question 3: Explanation (Text Entry)
        lines.append(f"[[Question:TE:Essay]]")
        lines.append(f"[[ID:{question_id_base}C]]")
        lines.append(instructions['explanation']['question'])
        lines.append("")
        
        # Question 4: Realism assessment (Matrix slider from 1-7)
        question_id_realism = f"{question_id_base}D"
    else:
        # For realism-only items, realism is Question 2
        question_id_realism = f"{question_id_base}B"
    
    # Realism assessment question (always included)
    lines.append(f"[[Question:Matrix:SingleAnswer]]")
    lines.append(f"[[ID:{question_id_realism}]]")
    lines.append(instructions['realism_assessment']['question'])
    lines.append("[[Choices]]")
    lines.append(instructions['realism_assessment']['choice_label'])
    lines.append("[[Answers]]")
    for label in instructions['realism_assessment']['scale_labels']:
        lines.append(label)
    lines.append("")
    
    return lines


def main(
    benchmark_file: str = "benchmark/parsed_benchmark_data.json",
    output_dir: str = "qualtrics_files",
    instructions_file: str = "data/insrtuctions_text.json",
    random_seed: Optional[int] = 42,
    num_participants: int = NUM_PARTICIPANTS,
    treatment_per_participant: int = TREATMENT_PER_PARTICIPANT,
    control_per_participant: int = CONTROL_PER_PARTICIPANT,
    realism_assignment_mode: str = 'per-participant',
    realism_treatment_per_participant: int = REALISM_TREATMENT_PER_PARTICIPANT,
    realism_control_per_participant: int = REALISM_CONTROL_PER_PARTICIPANT,
    realism_questions_per_participant: int = REALISM_QUESTIONS_PER_PARTICIPANT,
    realism_control_ratio: float = REALISM_CONTROL_RATIO
) -> None:
    """
    Main function to generate complete Qualtrics experiment files.
    
    Args:
        benchmark_file (str): Path to parsed benchmark JSON
        output_dir (str): Directory for Qualtrics files
        instructions_file (str): Path to instructions JSON file
        random_seed (Optional[int]): Random seed for reproducibility
        num_participants (int): Number of new participants to generate.
        treatment_per_participant (int): Treatment items per participant
        control_per_participant (int): Control items per participant
        realism_assignment_mode (str): 'per-participant' or 'global'.
        realism_treatment_per_participant (int): For 'per-participant' mode.
        realism_control_per_participant (int): For 'per-participant' mode.
        realism_questions_per_participant (int): For 'global' mode.
        realism_control_ratio (float): For 'global' mode.
    """
    # Read existing experiment plans to get used questions and last participant ID
    used_question_ids, max_participant_id = read_existing_experiment_plans()
    start_participant_id = max_participant_id + 1
    end_participant_id = start_participant_id + num_participants - 1
    
    output_csv = f"data/experiment_plan_{start_participant_id}_{end_participant_id}.csv"

    logging.info("=== QUALTRICS EXPERIMENT GENERATOR ===")
    logging.info(f"Target: {num_participants} new participants (IDs {start_participant_id}-{end_participant_id})")
    
    # Determine realism assignment logic and totals
    if realism_assignment_mode == 'global':
        logging.info(f"Using GLOBAL realism assignment mode (ratio-based).")
        total_realism_questions = num_participants * realism_questions_per_participant
        total_realism_control = round(total_realism_questions * realism_control_ratio)
        total_realism_treatment = total_realism_questions - total_realism_control
        total_items_per_participant = treatment_per_participant + control_per_participant + realism_questions_per_participant
    else: # per-participant
        logging.info(f"Using PER-PARTICIPANT realism assignment mode.")
        total_realism_treatment = num_participants * realism_treatment_per_participant
        total_realism_control = num_participants * realism_control_per_participant
        total_items_per_participant = treatment_per_participant + control_per_participant + realism_treatment_per_participant + realism_control_per_participant

    logging.info(f"Items per participant: {total_items_per_participant}")
    logging.info(f"Instructions file: {instructions_file}")
    logging.info(f"Random seed: {random_seed}")
    
    try:
        # 1. Load benchmark and instructions data
        logging.info("Step 1: Loading benchmark and instructions data...")
        data = load_benchmark(benchmark_file)
        instructions = load_instructions_text(instructions_file)
        
        # 2. Sample items with constraints
        logging.info("Step 2: Sampling items with balance constraints...")
        sampled_items = sample_items(
            data, 
            num_participants=num_participants,
            treatment_per_participant=treatment_per_participant,
            control_per_participant=control_per_participant,
            total_realism_treatment=total_realism_treatment,
            total_realism_control=total_realism_control,
            realism_assignment_mode=realism_assignment_mode,
            realism_treatment_per_participant=realism_treatment_per_participant,
            realism_control_per_participant=realism_control_per_participant,
            random_seed=random_seed,
            used_question_ids=used_question_ids,
            start_participant_id=start_participant_id
        )
        
        # 3. Assign conditions and metadata
        logging.info("Step 3: Assigning experimental conditions...")
        enriched_items = assign_conditions_and_metadata(sampled_items)
        
        # 4. Randomize question order per participant and add trackable ID
        logging.info("Step 4: Randomizing question order for each participant...")
        items_by_participant = defaultdict(list)
        for item in enriched_items:
            items_by_participant[item['participant_id']].append(item)

        final_items = []
        for pid, p_items in sorted(items_by_participant.items()):
            full_items = [item for item in p_items if item.get('question_type', 'full') == 'full']
            realism_only_items = [item for item in p_items if item.get('question_type', 'realism_only') == 'realism_only']
            
            random.shuffle(full_items)
            random.shuffle(realism_only_items)
            
            shuffled_p_items = full_items + realism_only_items
            for i, item in enumerate(shuffled_p_items):
                item['participants_question_id'] = i + 1
            
            final_items.extend(shuffled_p_items)

        # 5. Write metadata CSV
        logging.info("Step 5: Writing experiment plan CSV...")
        write_metadata_csv(final_items, output_csv, instructions)
        
        # 6. Generate Qualtrics files
        logging.info("Step 6: Generating Qualtrics files...")
        generate_qualtrics_txts(final_items, instructions, output_dir)
        
        logging.info("=== GENERATION COMPLETE ===")
        print(f"\nFiles generated:")
        print(f"  Experiment plan: {output_csv}")
        print(f"  Qualtrics files: {output_dir}/")
        print(f"  Instructions: {instructions_file}")
        print(f"  Ready for Qualtrics upload!")
        
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise e


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Qualtrics experiment files")
    parser.add_argument("--benchmark", default="benchmark/parsed_benchmark_data.json", 
                       help="Path to benchmark JSON file")
    parser.add_argument("--output", default="qualtrics_files", 
                       help="Output directory for Qualtrics files")
    parser.add_argument("--instructions", default="data/insrtuctions_text.json", 
                       help="Path to instructions JSON file")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--participants", type=int, default=NUM_PARTICIPANTS, 
                       help="Number of new participants to generate")
    parser.add_argument("--treatment", type=int, default=TREATMENT_PER_PARTICIPANT, 
                       help="Treatment items per participant")
    parser.add_argument("--control", type=int, default=CONTROL_PER_PARTICIPANT, 
                       help="Control items per participant")

    # Arguments for realism question assignment
    parser.add_argument("--realism_assignment_mode", default='per-participant', choices=['per-participant', 'global'],
                       help="Mode for assigning realism questions.")
    parser.add_argument("--realism_treatment_per_participant", type=int, default=REALISM_TREATMENT_PER_PARTICIPANT,
                       help="Realism treatment items per participant (for 'per-participant' mode).")
    parser.add_argument("--realism_control_per_participant", type=int, default=REALISM_CONTROL_PER_PARTICIPANT,
                       help="Realism control items per participant (for 'per-participant' mode).")
    parser.add_argument("--realism_questions_per_participant", type=int, default=REALISM_QUESTIONS_PER_PARTICIPANT,
                       help="Total realism questions per participant (for 'global' mode).")
    parser.add_argument("--realism_control_ratio", type=float, default=REALISM_CONTROL_RATIO,
                       help="Ratio of control questions in the realism set (for 'global' mode).")

    
    args = parser.parse_args()
    
    main(
        benchmark_file=args.benchmark,
        output_dir=args.output,
        instructions_file=args.instructions,
        random_seed=args.seed,
        num_participants=args.participants,
        treatment_per_participant=args.treatment,
        control_per_participant=args.control,
        realism_assignment_mode=args.realism_assignment_mode,
        realism_treatment_per_participant=args.realism_treatment_per_participant,
        realism_control_per_participant=args.realism_control_per_participant,
        realism_questions_per_participant=args.realism_questions_per_participant,
        realism_control_ratio=args.realism_control_ratio
    )
