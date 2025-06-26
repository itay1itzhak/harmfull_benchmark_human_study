"""
Qualtrics Participant Forms Generator

This module creates Qualtrics-ready files for a human study experiment on harmful AI content.
The experiment presents pairs of options to participants and collects harm ratings, explanations,
and realism assessments using Qualtrics Advanced Format.

Experiment Design:
- 4 participants total
- 15 items per participant (12 treatment + 3 control)
- Treatment items: one harmful option (15%, 40%, 65%) vs zero-harm option
- Control items: both options are zero-harm (attention checks)
- Response modes: harm rating (5-point scale), free-text explanation, realism question

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
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import itertools
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
HARM_LEVELS = [15, 40, 65]
BENEFIT_LEVELS = [15, 40, 65]
NUM_PARTICIPANTS = 4
TREATMENT_PER_PARTICIPANT = 12
CONTROL_PER_PARTICIPANT = 3
TOTAL_ITEMS_PER_PARTICIPANT = TREATMENT_PER_PARTICIPANT + CONTROL_PER_PARTICIPANT


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
    num_participants: int = NUM_PARTICIPANTS,
    treatment_per_participant: int = TREATMENT_PER_PARTICIPANT,
    control_per_participant: int = CONTROL_PER_PARTICIPANT,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Sample items for the experiment with balanced distribution and uniqueness constraints.
    
    Args:
        data (List[Dict[str, Any]]): Full benchmark data
        num_participants (int): Number of participants
        treatment_per_participant (int): Treatment items per participant
        control_per_participant (int): Control items per participant
        random_seed (Optional[int]): Random seed for reproducibility
        
    Returns:
        List[Dict[str, Any]]: Sampled items with participant assignments
        
    Raises:
        ValueError: If sampling constraints cannot be satisfied
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Calculate totals
    total_treatment = num_participants * treatment_per_participant
    total_control = num_participants * control_per_participant
    
    logging.info(f"Sampling {total_treatment} treatment + {total_control} control items")
    logging.info(f"For {num_participants} participants ({treatment_per_participant}+{control_per_participant} each)")
    
    # Filter treatment samples
    treatment_samples = filter_treatment_samples(data)
    
    if len(treatment_samples) < total_treatment:
        raise ValueError(f"Not enough treatment samples: need {total_treatment}, have {len(treatment_samples)}")
    
    # Get categories for balanced sampling
    categories = get_available_categories(treatment_samples)
    topics = categories['topics']
    
    logging.info(f"Available categories:")
    logging.info(f"  Topics: {len(topics)} ({topics})")
    logging.info(f"  Harm types: {len(categories['harm_types'])}")
    logging.info(f"  Benefits: {len(categories['benefits'])}")
    logging.info(f"  Models: {len(categories['model_types'])}")
    
    # Sample treatment items with balanced distribution
    sampled_treatment = sample_treatment_items_balanced(
        treatment_samples, 
        total_treatment, 
        topics
    )
    
    # Sample control items (from both Treatment and Control sample types in benchmark)
    sampled_control = sample_control_items(data, total_control)
    
    # Combine and assign to participants
    all_sampled = sampled_treatment + sampled_control
    
    # Assign participants
    participant_assignments = assign_participants(
        all_sampled, 
        num_participants, 
        treatment_per_participant, 
        control_per_participant
    )
    
    return participant_assignments


def sample_treatment_items_balanced(
    treatment_samples: List[Dict[str, Any]], 
    total_needed: int, 
    topics: List[str]
) -> List[Dict[str, Any]]:
    """
    Sample treatment items with balanced distribution across models, topics, harm types, and benefits.
    
    Args:
        treatment_samples (List[Dict[str, Any]]): Available treatment samples
        total_needed (int): Total number of treatment items needed
        topics (List[str]): List of available topics
        
    Returns:
        List[Dict[str, Any]]: Balanced sample of treatment items
    """
    # Group samples by model first for equal distribution
    by_model = defaultdict(list)
    for sample in treatment_samples:
        model_type = sample['metadata']['model_type']
        by_model[model_type].append(sample)
    
    models = sorted(by_model.keys())
    items_per_model = total_needed // len(models)
    remaining_items = total_needed % len(models)
    
    logging.info(f"Target model distribution: {items_per_model} per model, {remaining_items} extra")
    
    sampled = []
    used_combinations = set()
    
    # Sample from each model equally
    for model_idx, model in enumerate(models):
        model_samples = by_model[model]
        model_target = items_per_model + (1 if model_idx < remaining_items else 0)
        
        logging.info(f"Sampling {model_target} items for model {model}")
        
        # Group this model's samples by topic for balance within model
        model_by_topic = defaultdict(list)
        for sample in model_samples:
            model_by_topic[sample['metadata']['topic']].append(sample)
        
        model_selected = []
        
        # Try to get balanced topic distribution within this model
        topics_needed = min(len(model_by_topic), model_target)
        if topics_needed > 0:
            items_per_topic = model_target // topics_needed
            extra_items = model_target % topics_needed
            
            available_topics = list(model_by_topic.keys())
            random.shuffle(available_topics)  # Random topic order
            
            for topic_idx, topic in enumerate(available_topics[:topics_needed]):
                topic_samples = model_by_topic[topic]
                topic_target = items_per_topic + (1 if topic_idx < extra_items else 0)
                
                # Sample from this topic
                attempts = 0
                max_attempts = len(topic_samples) * 3
                
                while len([s for s in model_selected if s['metadata']['topic'] == topic]) < topic_target and attempts < max_attempts:
                    if not topic_samples:
                        break
                    
                    candidate = random.choice(topic_samples)
                    
                    # Create unique identifier
                    metadata = candidate['metadata']
                    combo_id = (metadata['model_type'], metadata['topic'], 
                               metadata['harm_type'], metadata['benefit'])
                    
                    if combo_id not in used_combinations:
                        model_selected.append(candidate)
                        used_combinations.add(combo_id)
                        topic_samples.remove(candidate)  # Remove to avoid duplicates
                    
                    attempts += 1
        
        # If we still need more items for this model, sample randomly from remaining
        remaining_needed = model_target - len(model_selected)
        if remaining_needed > 0:
            remaining_samples = []
            for sample in model_samples:
                metadata = sample['metadata']
                combo_id = (metadata['model_type'], metadata['topic'], 
                           metadata['harm_type'], metadata['benefit'])
                if combo_id not in used_combinations:
                    remaining_samples.append(sample)
            
            additional = random.sample(
                remaining_samples, 
                min(remaining_needed, len(remaining_samples))
            )
            
            for sample in additional:
                metadata = sample['metadata']
                combo_id = (metadata['model_type'], metadata['topic'], 
                           metadata['harm_type'], metadata['benefit'])
                used_combinations.add(combo_id)
            
            model_selected.extend(additional)
        
        sampled.extend(model_selected)
        logging.info(f"Selected {len(model_selected)} items for model {model}")
    
    logging.info(f"Final treatment sample: {len(sampled)} items")
    
    # Log distribution
    final_model_dist = Counter(item['metadata']['model_type'] for item in sampled)
    logging.info(f"Model distribution: {dict(final_model_dist)}")
    
    final_topic_dist = Counter(item['metadata']['topic'] for item in sampled)
    logging.info(f"Topic distribution: {dict(final_topic_dist)}")
    
    return sampled


def sample_control_items(data: List[Dict[str, Any]], total_needed: int) -> List[Dict[str, Any]]:
    """
    Sample control items (attention checks) with balanced model distribution.
    
    Args:
        data (List[Dict[str, Any]]): Full benchmark data
        total_needed (int): Number of control items needed
        
    Returns:
        List[Dict[str, Any]]: Sampled control items with balanced model distribution
    """
    # Get all available samples (both Control and Treatment can be used as experimental controls)
    control_samples = [item for item in data if item['metadata']['sample_type'] == 'Control']
    treatment_samples = [item for item in data if item['metadata']['sample_type'] == 'Treatment']
    
    all_potential_controls = control_samples + treatment_samples
    
    # Group by model for balanced distribution
    by_model = defaultdict(list)
    for sample in all_potential_controls:
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
        
        if len(model_samples) >= model_target:
            model_selected = random.sample(model_samples, model_target)
        else:
            model_selected = model_samples  # Take all available
            logging.warning(f"Model {model} has only {len(model_samples)} samples, need {model_target}")
        
        sampled_control.extend(model_selected)
        logging.info(f"Selected {len(model_selected)} control items for model {model}")
    
    # If we still need more items, sample randomly from remaining
    if len(sampled_control) < total_needed:
        remaining_needed = total_needed - len(sampled_control)
        used_samples = set(item['id'] for item in sampled_control)
        remaining_samples = [item for item in all_potential_controls if item['id'] not in used_samples]
        
        additional = random.sample(remaining_samples, min(remaining_needed, len(remaining_samples)))
        sampled_control.extend(additional)
    
    logging.info(f"Final control sample: {len(sampled_control)} items")
    
    # Log model distribution
    final_model_dist = Counter(item['metadata']['model_type'] for item in sampled_control)
    logging.info(f"Control model distribution: {dict(final_model_dist)}")
    
    return sampled_control


def assign_participants(
    all_items: List[Dict[str, Any]], 
    num_participants: int,
    treatment_per_participant: int,
    control_per_participant: int
) -> List[Dict[str, Any]]:
    """
    Assign sampled items to participants with balanced distribution.
    
    Args:
        all_items (List[Dict[str, Any]]): All sampled items
        num_participants (int): Number of participants
        treatment_per_participant (int): Treatment items per participant
        control_per_participant (int): Control items per participant
        
    Returns:
        List[Dict[str, Any]]: Items with participant assignments
    """
    # Calculate total needed
    total_treatment_needed = num_participants * treatment_per_participant
    total_control_needed = num_participants * control_per_participant
    
    # Separate by original sample type
    treatment_samples = [item for item in all_items if item['metadata']['sample_type'] == 'Treatment']
    control_samples = [item for item in all_items if item['metadata']['sample_type'] == 'Control']
    
    # Shuffle for random assignment
    random.shuffle(treatment_samples)
    random.shuffle(control_samples)
    
    # Assign experimental roles
    participant_items = []
    
    # First, assign treatment items (these will be experimental treatment)
    for i in range(total_treatment_needed):
        if i < len(treatment_samples):
            item = treatment_samples[i]
            participant_id = (i // treatment_per_participant) + 1
            item['participant_id'] = participant_id
            item['item_type'] = 'treatment'  # Experimental treatment
            participant_items.append(item)
    
    # Then, assign control items (these will be experimental controls)
    # Use control samples first, then treatment samples if needed
    control_pool = control_samples.copy()
    if len(control_pool) < total_control_needed:
        # Add remaining treatment samples to control pool
        remaining_treatment = treatment_samples[total_treatment_needed:]
        control_pool.extend(remaining_treatment)
        random.shuffle(control_pool)
    
    for i in range(total_control_needed):
        if i < len(control_pool):
            item = control_pool[i]
            participant_id = (i // control_per_participant) + 1
            item['participant_id'] = participant_id
            item['item_type'] = 'control'  # Experimental control (attention check)
            participant_items.append(item)
    
    logging.info(f"Assigned items to {num_participants} participants")
    
    # Log assignment statistics
    treatment_assigned = len([item for item in participant_items if item['item_type'] == 'treatment'])
    control_assigned = len([item for item in participant_items if item['item_type'] == 'control'])
    logging.info(f"Experimental assignment: {treatment_assigned} treatment, {control_assigned} control")
    
    return participant_items


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
    
    for item in items:
        model_type = item['metadata']['model_type']
        if item['item_type'] == 'treatment':
            treatment_by_model[model_type].append(item)
        else:
            control_by_model[model_type].append(item)
    
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
    
    # Verify overall balance
    treatment_items = [item for item in items if item['item_type'] == 'treatment']
    control_items = [item for item in items if item['item_type'] == 'control']
    
    treatment_swapped = sum(1 for item in treatment_items if assignment_dict[item['item_id']])
    control_swapped = sum(1 for item in control_items if assignment_dict[item['item_id']])
    
    logging.info(f"Overall balance verification:")
    logging.info(f"  Treatment: {len(treatment_items)} total - {treatment_swapped} B harmful, {len(treatment_items)-treatment_swapped} A harmful")
    logging.info(f"  Control: {len(control_items)} total - {control_swapped} B 'harmful', {len(control_items)-control_swapped} A 'harmful'")
    
    return assignment_dict


def assign_conditions_and_metadata(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assign experimental conditions and metadata to sampled items.
    
    This includes:
    - Random harm levels (15%, 40%, 65%) for treatment items
    - Random benefit levels (15%, 40%, 65%) with placeholder replacement
    - Balanced assignment of harmful option to A or B (exactly 50/50 at all levels)
    - Consistent option randomization for both treatment and control items
    - Unique item IDs
    
    Args:
        items (List[Dict[str, Any]]): Items with participant assignments
        
    Returns:
        List[Dict[str, Any]]: Items with full experimental metadata
    """
    # First, assign item IDs
    for idx, item in enumerate(items):
        item['item_id'] = f"item_{idx + 1:03d}"
    
    # Create balanced option assignments
    logging.info("Creating balanced option assignments...")
    option_assignments = create_balanced_option_assignments(items)
    
    enriched_items = []
    
    for item in items:
        # Create enriched copy
        enriched = item.copy()
        
        # Process text with benefit placeholder first
        benefit_level = random.choice(BENEFIT_LEVELS)
        enriched['benefit_level'] = benefit_level
        
        processed_scenario = process_text_placeholders(enriched['scenario'], benefit_level)
        processed_option_a = process_text_placeholders(enriched['option_a'], benefit_level)
        processed_option_b = process_text_placeholders(enriched['option_b'], benefit_level)
        
        # Get pre-determined assignment for this item
        swap_options = option_assignments[item['item_id']]
        
        if enriched['item_type'] == 'treatment':
            # Assign random harm level for treatment items
            harm_level = random.choice(HARM_LEVELS)
            enriched['harm_level'] = harm_level
            
            if swap_options:
                # Swap options and update text references
                processed_option_a, processed_option_b = swap_options_and_update_text(
                    processed_option_a, processed_option_b
                )
                # After swap: original harmful A is now in position B
                enriched['harmful_option'] = 'B'
                enriched['harm_percentage_a'] = 0
                enriched['harm_percentage_b'] = harm_level
                enriched['option_swapped'] = True
            else:
                # Keep original order - A is harmful in original data
                enriched['harmful_option'] = 'A'
                enriched['harm_percentage_a'] = harm_level
                enriched['harm_percentage_b'] = 0
                enriched['option_swapped'] = False
            
        else:  # control items - treat exactly like treatment items but with 0% harm
            # Both options are non-harmful for controls, but follow same shuffling logic
            enriched['harm_level'] = 0
            
            if swap_options:
                # Swap options and update text references
                processed_option_a, processed_option_b = swap_options_and_update_text(
                    processed_option_a, processed_option_b
                )
                # After swap: original "harmful" A is now in position B (but still 0% harm)
                enriched['harmful_option'] = 'B'
                enriched['harm_percentage_a'] = 0
                enriched['harm_percentage_b'] = 0
                enriched['option_swapped'] = True
            else:
                # Keep original order - A is the designated "harmful" option (but still 0% harm)
                enriched['harmful_option'] = 'A'
                enriched['harm_percentage_a'] = 0
                enriched['harm_percentage_b'] = 0
                enriched['option_swapped'] = False
        
        # Store processed text
        enriched['processed_scenario'] = processed_scenario
        enriched['processed_option_a'] = processed_option_a
        enriched['processed_option_b'] = processed_option_b
        
        
        enriched_items.append(enriched)
    
    logging.info(f"Assigned conditions to {len(enriched_items)} items")
    
    # Log final statistics
    log_final_balance_statistics(enriched_items)
    
    return enriched_items


def log_final_balance_statistics(items: List[Dict[str, Any]]) -> None:
    """
    Log comprehensive balance statistics for the final item assignments.
    
    Args:
        items (List[Dict[str, Any]]): Items with full metadata
    """
    # Overall statistics
    treatment_items = [item for item in items if item['item_type'] == 'treatment']
    control_items = [item for item in items if item['item_type'] == 'control']
    
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
    
    logging.info("=== END BALANCE STATISTICS ===")


def process_text_placeholders(text: str, benefit_level: int) -> str:
    """
    Replace placeholders in text with actual values and prepare for HTML conversion.
    
    Args:
        text (str): Text potentially containing placeholders
        benefit_level (int): Benefit percentage to insert
        
    Returns:
        str: Text with placeholders replaced, ready for HTML conversion
    """
    # Replace benefit percentage placeholder
    processed = text.replace('{benefit_percentage}%', f'{benefit_level}%')
    processed = processed.replace('{benefit_percentage}', str(benefit_level))
    
    # Clean up any formatting issues
    processed = processed.strip()
    
    return processed


def write_metadata_csv(
    items: List[Dict[str, Any]], 
    output_path: str = "data/experiment_plan.csv"
) -> None:
    """
    Write comprehensive metadata CSV for linking responses to item details.
    
    Args:
        items (List[Dict[str, Any]]): Items with full metadata
        output_path (str): Path to save CSV file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    
    for item in items:
        metadata = item['metadata']
        
        row = {
            'item_id': item['item_id'],
            'participant_id': item['participant_id'],
            'item_type': item['item_type'],
            'original_sample_id': item['id'],
            
            # Experimental conditions
            'harm_level': item.get('harm_level', 0),
            'benefit_level': item.get('benefit_level', 0),
            'harmful_option': item.get('harmful_option', 'Neither'),
            'harm_percentage_a': item.get('harm_percentage_a', 0),
            'harm_percentage_b': item.get('harm_percentage_b', 0),
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
            
            # Original text for reference
            'original_scenario': item['scenario'],
            'original_option_a': item['option_a'],
            'original_option_b': item['option_b'],
        }
        
        csv_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    df = df.sort_values(['participant_id', 'item_type', 'item_id'])
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
        print(f"  Participant {pid}: {treatment_count} treatment + {control_count} control")
    
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
    output_dir: str = "qualtrics_files"
) -> None:
    """
    Generate individual Qualtrics .txt files for each participant.
    
    Args:
        items (List[Dict[str, Any]]): Items with full metadata
        output_dir (str): Directory to save Qualtrics files
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
        
        generate_participant_qualtrics_file(participant_items, filepath, participant_id)
    
    logging.info(f"Generated Qualtrics files for {len(by_participant)} participants in {output_dir}")


def generate_participant_qualtrics_file(
    items: List[Dict[str, Any]], 
    filepath: str, 
    participant_id: int
) -> None:
    """
    Generate a single participant's Qualtrics .txt file using Advanced Format.
    
    Args:
        items (List[Dict[str, Any]]): Items for this participant
        filepath (str): Output file path
        participant_id (int): Participant ID
    """
    lines = []
    
    # Advanced format header
    lines.append("[[AdvancedFormat]]")
    lines.append("")
    
    # Add embedded data for participant tracking
    lines.append(f"[[ED:participant_id:{participant_id}]]")
    lines.append("")
    
    # Introduction block
    lines.append(f"[[Block:Participant {participant_id} - Harm Assessment Study]]")
    lines.append("")
    
    # Introduction text/graphic question
    lines.append("[[Question:Text]]")
    lines.append("Welcome to the Harm Assessment Study")
    lines.append("")
    lines.append("In this study, you will evaluate pairs of AI-generated responses and assess their potential for harm.")
    lines.append("Please read each scenario carefully and provide your honest assessment.")
    lines.append("")
    lines.append("[[PageBreak]]")
    lines.append("")
    
    # Sort items for logical flow (control items can be interspersed)
    sorted_items = sorted(items, key=lambda x: (x['item_type'] == 'control', x['item_id']))
    
    # Generate questions for each item
    for idx, item in enumerate(sorted_items, 1):
        question_id_base = f"Q{idx:02d}"
        
        # Add scenario and options
        lines.extend(generate_item_questions(item, question_id_base, idx))
        
        # Add page break between items (except last)
        if idx < len(sorted_items):
            lines.append("[[PageBreak]]")
            lines.append("")
    
    # Closing block
    lines.append("[[Block:Completion]]")
    lines.append("")
    lines.append("[[Question:Text]]")
    lines.append("Thank you for participating in this study!")
    lines.append("")
    lines.append("Your responses have been recorded. Your participation helps improve AI safety research.")
    lines.append("")
    
    # Write file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logging.info(f"Generated Qualtrics file for participant {participant_id}: {filepath}")


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


def generate_item_questions(item: Dict[str, Any], question_id_base: str, item_number: int) -> List[str]:
    """
    Generate Qualtrics Advanced Format questions for a single item with HTML formatting.
    
    Args:
        item (Dict[str, Any]): Item with full metadata
        question_id_base (str): Question ID base (e.g., "Q01")
        item_number (int): Item number for display
        
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
    lines.append(f"<b>Scenario {item_number}</b>")
    lines.append("<br>")
    lines.append("Please read the following scenario and two response options:<br><br>")
    #lines.append("<b>Scenario:</b><br>")
    lines.append(scenario_html)
    lines.append("<br>")
    #lines.append("<b>Option A:</b><br>")
    lines.append(option_a_html)
    lines.append("<br>")
    #lines.append("<b>Option B:</b><br>")
    lines.append(option_b_html)
    lines.append("")
    
    # Question 2: Harm comparison (Slider from 1-5)
    lines.append(f"[[Question:Matrix:SingleAnswer]]")
    lines.append(f"[[ID:{question_id_base}B]]")
    lines.append("Which option is more harmful?<br><br>Please rate on the scale:")
    lines.append("[[Choices]]")
    lines.append("Harm Assessment")
    lines.append("[[Answers]]")
    lines.append("1 - A much more harmful")
    lines.append("2 - A somewhat more harmful") 
    lines.append("3 - Equally harmful")
    lines.append("4 - B somewhat more harmful")
    lines.append("5 - B much more harmful")
    lines.append("")
    
    # Question 3: Explanation (Text Entry)
    lines.append(f"[[Question:TE:Essay]]")
    lines.append(f"[[ID:{question_id_base}C]]")
    lines.append("Please briefly explain your reasoning for the harm assessment above.")
    lines.append("")
    
    # Question 4: Realism assessment (Matrix slider from 1-5)
    lines.append(f"[[Question:Matrix:SingleAnswer]]")
    lines.append(f"[[ID:{question_id_base}D]]")
    lines.append("How realistic does this scenario seem?<br><br>Please rate on the scale:")
    lines.append("[[Choices]]")
    lines.append("Realism Assessment")
    lines.append("[[Answers]]")
    lines.append("1 - Not realistic at all")
    lines.append("2 - Slightly realistic")
    lines.append("3 - Moderately realistic")
    lines.append("4 - Very realistic")
    lines.append("5 - Extremely realistic")
    lines.append("")
    
    return lines


def main(
    benchmark_file: str = "benchmark/parsed_benchmark_data.json",
    output_csv: str = "data/experiment_plan.csv",
    output_dir: str = "qualtrics_files",
    random_seed: Optional[int] = 42,
    num_participants: int = NUM_PARTICIPANTS,
    treatment_per_participant: int = TREATMENT_PER_PARTICIPANT,
    control_per_participant: int = CONTROL_PER_PARTICIPANT
) -> None:
    """
    Main function to generate complete Qualtrics experiment files.
    
    Args:
        benchmark_file (str): Path to parsed benchmark JSON
        output_csv (str): Path for experiment plan CSV
        output_dir (str): Directory for Qualtrics files
        random_seed (Optional[int]): Random seed for reproducibility
        num_participants (int): Number of participants
        treatment_per_participant (int): Treatment items per participant
        control_per_participant (int): Control items per participant
    """
    logging.info("=== QUALTRICS EXPERIMENT GENERATOR ===")
    logging.info(f"Target: {num_participants} participants")
    logging.info(f"Items per participant: {treatment_per_participant} treatment + {control_per_participant} control")
    logging.info(f"Random seed: {random_seed}")
    
    try:
        # 1. Load benchmark data
        logging.info("Step 1: Loading benchmark data...")
        data = load_benchmark(benchmark_file)
        
        # 2. Sample items with constraints
        logging.info("Step 2: Sampling items with balance constraints...")
        sampled_items = sample_items(
            data, 
            num_participants=num_participants,
            treatment_per_participant=treatment_per_participant,
            control_per_participant=control_per_participant,
            random_seed=random_seed
        )
        
        # 3. Assign conditions and metadata
        logging.info("Step 3: Assigning experimental conditions...")
        enriched_items = assign_conditions_and_metadata(sampled_items)
        
        # 4. Write metadata CSV
        logging.info("Step 4: Writing experiment plan CSV...")
        write_metadata_csv(enriched_items, output_csv)
        
        # 5. Generate Qualtrics files
        logging.info("Step 5: Generating Qualtrics files...")
        generate_qualtrics_txts(enriched_items, output_dir)
        
        logging.info("=== GENERATION COMPLETE ===")
        print(f"\nFiles generated:")
        print(f"  Experiment plan: {output_csv}")
        print(f"  Qualtrics files: {output_dir}/")
        print(f"  Ready for Qualtrics upload!")
        
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise e


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Qualtrics experiment files")
    parser.add_argument("--benchmark", default="benchmark/parsed_benchmark_data.json", 
                       help="Path to benchmark JSON file")
    parser.add_argument("--csv", default="data/experiment_plan.csv", 
                       help="Output path for experiment plan CSV")
    parser.add_argument("--output", default="qualtrics_files", 
                       help="Output directory for Qualtrics files")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--participants", type=int, default=4, 
                       help="Number of participants")
    parser.add_argument("--treatment", type=int, default=12, 
                       help="Treatment items per participant")
    parser.add_argument("--control", type=int, default=3, 
                       help="Control items per participant")
    
    args = parser.parse_args()
    
    main(
        benchmark_file=args.benchmark,
        output_csv=args.csv,
        output_dir=args.output,
        random_seed=args.seed,
        num_participants=args.participants,
        treatment_per_participant=args.treatment,
        control_per_participant=args.control
    )
