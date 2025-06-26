import pandas as pd
import numpy as np
import random
import itertools
import logging
import os
from typing import List, Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODELS = ['Model_A', 'Model_B', 'Model_C']
TOPICS = [f'Topic_{i+1}' for i in range(11)]
HARM_TYPES = [f'Harm_{i+1}' for i in range(4)]
BENEFIT_REASONS = [f'Benefit_{i+1}' for i in range(4)]
HARM_PERCENTAGES = [15, 40, 65]
CONTROL_HARM_PERCENTAGE = 0
# NUM_TREATMENT_PER_PARTICIPANT = 24
# NUM_CONTROL_PER_PARTICIPANT = 6
NUM_MODELS = len(MODELS)

def create_treatment_pool() -> List[Dict[str, Any]]:
    """
    Creates a large pool of treatment items.

    Returns:
        List[Dict[str, Any]]: A list of treatment items.
    """
    pool = []
    for model, topic, harm_type, benefit in itertools.product(MODELS, TOPICS, HARM_TYPES, BENEFIT_REASONS):
        pool.append({
            'model': model,
            'topic': topic,
            'harm_type': harm_type,
            'benefit_reason': benefit,
            'type': 'treatment'
        })
    return pool

def create_control_items(model: str, num_control_per_participant: int = 6) -> List[Dict[str, Any]]:
    """
    Creates control items for a specific model.

    Args:
        model (str): The model for which to create control items.

    Returns:
        List[Dict[str, Any]]: A list of control items.
    """
    return [{
        'model': model,
        'topic': 'Control',
        'harm_type': 'Control',
        'benefit_reason': 'Control',
        'type': 'control'
    }] * (num_control_per_participant // NUM_MODELS)


def generate_participant_data(participant_id: int, treatment_pool: List[Dict[str, Any]], used_samples: set, is_ambiguous: bool = False, mode: str = 'choice', num_treatment_per_participant: int = 24, num_control_per_participant: int = 6) -> pd.DataFrame:
    """
    Generates data for a single participant using stratified sampling.

    Args:
        participant_id (int): The ID of the participant.
        treatment_pool (List[Dict[str, Any]]): The pool of treatment items.
        used_samples (set): Set of already used sample combinations to avoid duplicates.
        is_ambiguous (bool, optional): Whether the participant is ambiguous. Defaults to False.
        mode (str, optional): The response mode ('choice' or 'rating'). Defaults to 'choice'.
        num_treatment_per_participant (int, optional): The number of treatment items per participant. Defaults to 24.
        num_control_per_participant (int, optional): The number of control items per participant. Defaults to 6.

    Returns:
        pd.DataFrame: A DataFrame with the data for one participant.
    """
    participant_items = []
    
    # Stratified sampling for treatment items
    for model in MODELS:
        model_pool = [item for item in treatment_pool if item['model'] == model]
        
        # Ensure each factor is represented
        available_topics = [t for t in TOPICS]
        selected_items = []
        
        # Try to get unique combinations for this participant
        attempts = 0
        max_attempts = 1000
        
        while len(selected_items) < num_treatment_per_participant // NUM_MODELS and attempts < max_attempts:
            topic = random.choice(available_topics)
            harm_type = random.choice(HARM_TYPES)
            benefit_reason = random.choice(BENEFIT_REASONS)
            
            # Create sample identifier
            sample_id = (model, topic, harm_type, benefit_reason)
            
            if sample_id not in used_samples:
                item = {
                    'model': model,
                    'topic': topic,
                    'harm_type': harm_type,
                    'benefit_reason': benefit_reason,
                    'type': 'treatment',
                    'sample_id': sample_id
                }
                selected_items.append(item)
                used_samples.add(sample_id)
            
            attempts += 1
        
        # If we couldn't find enough unique samples, fill with random ones
        while len(selected_items) < num_treatment_per_participant // NUM_MODELS:
            topic = random.choice(TOPICS)
            harm_type = random.choice(HARM_TYPES)
            benefit_reason = random.choice(BENEFIT_REASONS)
            sample_id = (model, topic, harm_type, benefit_reason)
            
            item = {
                'model': model,
                'topic': topic,
                'harm_type': harm_type,
                'benefit_reason': benefit_reason,
                'type': 'treatment',
                'sample_id': sample_id
            }
            selected_items.append(item)
            used_samples.add(sample_id)
        
        participant_items.extend(selected_items)

    # Add control items
    for model in MODELS:
        participant_items.extend(create_control_items(model, num_control_per_participant=num_control_per_participant))

    df = pd.DataFrame(participant_items)
    df['participant_id'] = participant_id
    
    # Assign harm percentages
    harm_percentages_cycle = itertools.cycle(HARM_PERCENTAGES)
    df['harm_A'] = np.nan
    df['harm_B'] = np.nan
    
    treatment_indices = df[df['type'] == 'treatment'].index
    control_indices = df[df['type'] == 'control'].index

    for i in treatment_indices:
        # One option gets a harmful percentage, the other gets 0 (not harmful)
        harm_val = next(harm_percentages_cycle)
        if random.random() < 0.5:
            df.loc[i, 'harm_A'] = harm_val
            df.loc[i, 'harm_B'] = 0
        else:
            df.loc[i, 'harm_A'] = 0
            df.loc[i, 'harm_B'] = harm_val
            
    df.loc[control_indices, ['harm_A', 'harm_B']] = CONTROL_HARM_PERCENTAGE

    # Randomize A/B
    for index, row in df.iterrows():
        if random.random() < 0.5:
            df.loc[index, ['harm_A', 'harm_B']] = row[['harm_B', 'harm_A']].values

    # Simulate participant responses based on mode
    if mode == 'choice':
        df['chosen_harmful'] = df.apply(lambda row: simulate_harmful_choice(row, is_ambiguous=is_ambiguous), axis=1)
        df['chosen_pick'] = df.apply(simulate_pick_choice, axis=1)
    elif mode == 'rating':
        ratings = df.apply(lambda row: simulate_ratings(row, is_ambiguous=is_ambiguous), axis=1)
        df[['rating_A', 'rating_B']] = pd.DataFrame(ratings.tolist(), index=df.index)

    # Simulate confidence levels based on participant type and response accuracy
    df['confidence'] = df.apply(lambda row: simulate_confidence(row, is_ambiguous=is_ambiguous), axis=1)
    df['comment'] = ''
    
    return df.sample(frac=1).reset_index(drop=True) # Shuffle items

def simulate_harmful_choice(row: pd.Series, is_ambiguous: bool = False) -> str:
    """
    Simulates the participant's choice for the 'more harmful' question.

    Args:
        row (pd.Series): The row of data for a single item.
        is_ambiguous (bool, optional): Whether the participant is ambiguous. Defaults to False.

    Returns:
        str: The simulated choice.
    """
    if row['type'] == 'control':
        return random.choice(['A', 'B', 'No difference or Unsure'])
    
    # More dispersed probabilities with ambiguous being closer to random
    correctness_prob = 0.45 if is_ambiguous else 0.8  # Both reduced, ambiguous closer to random
    no_diff_prob = 0.4 if is_ambiguous else 0.1      # Higher uncertainty for both
    
    # Add random noise to make responses more varied
    if random.random() < (0.2 if is_ambiguous else 0.1):  # Higher chance of random choice for ambiguous
        return random.choice(['A', 'B', 'No difference or Unsure'])

    if random.random() < no_diff_prob:
        return 'No difference or Unsure'

    if row['harm_A'] > row['harm_B']:
        return 'A' if random.random() < correctness_prob else 'B'
    elif row['harm_B'] > row['harm_A']:
        return 'B' if random.random() < correctness_prob else 'A'
    else:
        return 'No difference or Unsure'

def simulate_pick_choice(row: pd.Series) -> str:
    """Simulates the participant's choice for the 'which to pick' question."""
    if row['type'] == 'control':
        return 'Neither'

    if row['harm_A'] < row['harm_B']:
        return 'A' # Pick the less harmful option
    elif row['harm_B'] < row['harm_A']:
        return 'B'
    else:
        return 'Neither'

def simulate_ratings(row: pd.Series, is_ambiguous: bool = False) -> Tuple[int, int]:
    """
    Simulates participant ratings for harm on a scale of 1-5.

    Args:
        row (pd.Series): The row of data for a single item.
        is_ambiguous (bool, optional): Whether the participant is ambiguous. Defaults to False.

    Returns:
        Tuple[int, int]: A tuple containing the ratings for option A and option B.
    """
    def get_rating(harm_percentage: float, is_ambiguous: bool = False) -> int:
        # More dispersed base ratings
        if harm_percentage == 0:
            if is_ambiguous:
                return np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
            return np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        elif harm_percentage == 15:
            if is_ambiguous:
                return np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1])
            return np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2])
        elif harm_percentage == 40:
            if is_ambiguous:
                return np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1])
            return np.random.choice([2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.2])
        elif harm_percentage == 65:
            if is_ambiguous:
                return np.random.choice([2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.2])
            return np.random.choice([3, 4, 5], p=[0.1, 0.4, 0.5])
        return 1

    # Get base ratings with more dispersion
    rating_A = get_rating(row['harm_A'], is_ambiguous)
    rating_B = get_rating(row['harm_B'], is_ambiguous)

    # Add more randomness to ratings
    if random.random() < (0.5 if is_ambiguous else 0.3):  # Higher chance of random adjustment for ambiguous
        rating_A = max(1, min(5, rating_A + random.choice([-2, -1, 0, 1, 2])))
        rating_B = max(1, min(5, rating_B + random.choice([-2, -1, 0, 1, 2])))

    # Make ratings closer with different probabilities for ambiguous and non-ambiguous
    if row['type'] == 'treatment':
        if is_ambiguous:
            if random.random() < 0.6:  # 60% chance of making ratings closer for ambiguous
                if abs(rating_A - rating_B) > 1:
                    if rating_A > rating_B:
                        rating_A -= 1
                    else:
                        rating_B -= 1
        else:
            if random.random() < 0.3:  # 30% chance of making ratings closer for non-ambiguous
                if abs(rating_A - rating_B) > 1:
                    if rating_A > rating_B:
                        rating_A -= 1
                    else:
                        rating_B -= 1

    return rating_A, rating_B

def simulate_confidence(row: pd.Series, is_ambiguous: bool = False) -> int:
    """
    Simulates participant confidence levels based on their type and response characteristics.
    
    Args:
        row (pd.Series): The row of data for a single item.
        is_ambiguous (bool, optional): Whether the participant is ambiguous. Defaults to False.
    
    Returns:
        int: Confidence level from 1-5.
    """
    if row['type'] == 'control':
        # Control items should generally have higher confidence for regular users
        if is_ambiguous:
            return np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05])
        else:
            return np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
    
    # For treatment items, confidence depends on clarity of harm difference
    harm_diff = abs(row['harm_A'] - row['harm_B'])
    
    if is_ambiguous:
        # Ambiguous users are generally less confident
        if harm_diff >= 40:  # Clear difference
            return np.random.choice([1, 2, 3, 4, 5], p=[0.25, 0.3, 0.25, 0.15, 0.05])
        elif harm_diff >= 15:  # Moderate difference
            return np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.35, 0.2, 0.1, 0.05])
        else:  # Small or no difference
            return np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.2, 0.08, 0.02])
    else:
        # Regular users are generally more confident
        if harm_diff >= 40:  # Clear difference
            return np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.35, 0.35])
        elif harm_diff >= 15:  # Moderate difference
            return np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.25, 0.3, 0.2])
        else:  # Small or no difference
            return np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.25, 0.3, 0.2, 0.05])

def print_descriptive_stats(data_path: str, min_confidence: int = 1):
    """
    Prints meaningful descriptive statistics for the data, providing insights into the experimental design
    and participant responses.

    Args:
        data_path (str): Path to the CSV file containing the generated data.
        min_confidence (int, optional): Minimum confidence level to include in analysis. Defaults to 1 (include all).
    """
    df = pd.read_csv(data_path)
    
    # Filter by confidence if specified
    if min_confidence > 1:
        original_count = len(df)
        df = df[df['confidence'] >= min_confidence]
        filtered_count = len(df)
        print(f"\n=== CONFIDENCE FILTERING ===")
        print(f"Original samples: {original_count}")
        print(f"Filtered samples (confidence >= {min_confidence}): {filtered_count}")
        print(f"Excluded samples: {original_count - filtered_count} ({(original_count - filtered_count)/original_count*100:.1f}%)")
    
    print("\n=== EXPERIMENTAL DESIGN STATISTICS ===")
    print(f"\nTotal number of participants: {df['participant_id'].nunique()}")
    print(f"Total number of items per participant: {len(df) // df['participant_id'].nunique()}")
    
    # Treatment vs Control distribution
    treatment_count = len(df[df['type'] == 'treatment'])
    control_count = len(df[df['type'] == 'control'])
    print(f"\nTreatment items: {treatment_count} ({treatment_count/len(df)*100:.1f}%)")
    print(f"Control items: {control_count} ({control_count/len(df)*100:.1f}%)")
    
    # Harm distribution analysis
    print("\n=== HARM DISTRIBUTION ANALYSIS ===")
    
    # Analyze harmful options (non-zero harm)
    harmful_a = df[df['harm_A'] > 0]['harm_A'].value_counts().sort_index()
    harmful_b = df[df['harm_B'] > 0]['harm_B'].value_counts().sort_index()
    
    print("\nDistribution of Harmful Options:")
    print("Option A (when harmful):")
    for harm_val, count in harmful_a.items():
        print(f"  {harm_val}% harm: {count} items ({count/len(df)*100:.1f}%)")
    
    print("\nOption B (when harmful):")
    for harm_val, count in harmful_b.items():
        print(f"  {harm_val}% harm: {count} items ({count/len(df)*100:.1f}%)")
    
    # Analyze non-harmful options (zero harm)
    non_harmful_a = len(df[df['harm_A'] == 0])
    non_harmful_b = len(df[df['harm_B'] == 0])
    
    print("\nDistribution of Non-Harmful Options:")
    print(f"Option A (when non-harmful): {non_harmful_a} items ({non_harmful_a/len(df)*100:.1f}%)")
    print(f"Option B (when non-harmful): {non_harmful_b} items ({non_harmful_b/len(df)*100:.1f}%)")
    
    # Analyze harm level distribution across all harmful options
    all_harmful = pd.concat([df[df['harm_A'] > 0]['harm_A'], df[df['harm_B'] > 0]['harm_B']])
    harm_dist = all_harmful.value_counts().sort_index()
    
    print("\nOverall Distribution of Harm Levels:")
    for harm_val, count in harm_dist.items():
        print(f"  {harm_val}% harm: {count} items ({count/len(all_harmful)*100:.1f}%)")
    
    # Analyze control items separately
    control_harm = df[df['type'] == 'control']
    print(f"\nControl Items (0% harm): {len(control_harm)} items ({len(control_harm)/len(df)*100:.1f}%)")
    
    # Parallel analysis of treatment vs control samples
    print("\n=== PARALLEL ANALYSIS: TREATMENT VS CONTROL ===")
    
    if 'rating_A' in df.columns and 'rating_B' in df.columns:
        # Rating mode analysis
        print("\nRating Analysis (1-5 scale):")
        
        # Treatment samples - split by harmful and non-harmful
        treatment_df = df[df['type'] == 'treatment']
        harmful_ratings = []
        non_harmful_ratings = []
        
        for _, row in treatment_df.iterrows():
            if row['harm_A'] > 0:
                harmful_ratings.append(row['rating_A'])
                non_harmful_ratings.append(row['rating_B'])
            else:
                harmful_ratings.append(row['rating_B'])
                non_harmful_ratings.append(row['rating_A'])
        
        print("\nTreatment Samples:")
        print("Harmful Options:")
        print(f"  Average rating: {np.mean(harmful_ratings):.2f}")
        print("  Rating distribution:")
        for rating in range(1, 6):
            count = sum(1 for r in harmful_ratings if r == rating)
            print(f"    Rating {rating}: {count} ratings ({count/len(harmful_ratings)*100:.1f}%)")
        
        print("\nNon-Harmful Options:")
        print(f"  Average rating: {np.mean(non_harmful_ratings):.2f}")
        print("  Rating distribution:")
        for rating in range(1, 6):
            count = sum(1 for r in non_harmful_ratings if r == rating)
            print(f"    Rating {rating}: {count} ratings ({count/len(non_harmful_ratings)*100:.1f}%)")
        
        # Control samples - both options are non-harmful
        control_ratings = pd.concat([control_harm['rating_A'], control_harm['rating_B']])
        print("\nControl Samples (Both Options Non-Harmful):")
        print(f"Average rating: {control_ratings.mean():.2f}")
        print("Rating distribution:")
        for rating in range(1, 6):
            count = len(control_ratings[control_ratings == rating])
            print(f"  Rating {rating}: {count} ratings ({count/len(control_ratings)*100:.1f}%)")
        
        # Compare distributions
        print("\nComparison:")
        print(f"Treatment Harmful vs Control: {np.mean(harmful_ratings) - control_ratings.mean():.2f}")
        print(f"Treatment Non-Harmful vs Control: {np.mean(non_harmful_ratings) - control_ratings.mean():.2f}")
    
    if 'chosen_harmful' in df.columns:
        # Choice mode analysis
        print("\nChoice Analysis:")
        
        # Treatment samples - split by harmful and non-harmful
        treatment_df = df[df['type'] == 'treatment']
        harmful_choices = 0
        non_harmful_choices = 0
        no_diff_choices = 0
        total_choices = 0
        
        for _, row in treatment_df.iterrows():
            if row['chosen_harmful'] != 'No difference or Unsure':
                total_choices += 1
                if (row['chosen_harmful'] == 'A' and row['harm_A'] > 0) or \
                   (row['chosen_harmful'] == 'B' and row['harm_B'] > 0):
                    harmful_choices += 1
                else:
                    non_harmful_choices += 1
            else:
                no_diff_choices += 1
        
        print("\nTreatment Samples:")
        print("When Harmful Option was Chosen:")
        print(f"  Count: {harmful_choices} ({harmful_choices/len(treatment_df)*100:.1f}%)")
        print("When Non-Harmful Option was Chosen:")
        print(f"  Count: {non_harmful_choices} ({non_harmful_choices/len(treatment_df)*100:.1f}%)")
        print("When No Difference was Chosen:")
        print(f"  Count: {no_diff_choices} ({no_diff_choices/len(treatment_df)*100:.1f}%)")
        
        # Control samples
        control_choices = control_harm['chosen_harmful'].value_counts()
        print("\nControl Samples:")
        print("Choice distribution:")
        for choice, count in control_choices.items():
            print(f"  {choice}: {count} choices ({count/len(control_harm)*100:.1f}%)")
        
        # Compare "No difference" responses
        treatment_no_diff = no_diff_choices / len(treatment_df) * 100
        control_no_diff = control_choices.get('No difference or Unsure', 0) / len(control_harm) * 100
        print("\nComparison:")
        print(f"'No difference' responses - Treatment: {treatment_no_diff:.1f}%, Control: {control_no_diff:.1f}%")
    
    # Confidence analysis for both types
    if 'confidence' in df.columns:
        print("\nConfidence Analysis:")
        
        # Treatment samples - split by harmful and non-harmful
        treatment_df = df[df['type'] == 'treatment']
        
        print("\nTreatment Samples:")
        
        # Analyze confidence by harm level difference
        high_diff_conf = []  # harm difference >= 40
        med_diff_conf = []   # harm difference 15-39
        low_diff_conf = []   # harm difference < 15
        
        for _, row in treatment_df.iterrows():
            harm_diff = abs(row['harm_A'] - row['harm_B'])
            if harm_diff >= 40:
                high_diff_conf.append(row['confidence'])
            elif harm_diff >= 15:
                med_diff_conf.append(row['confidence'])
            else:
                low_diff_conf.append(row['confidence'])
        
        if high_diff_conf:
            print(f"High Harm Difference (≥40%): Avg confidence {np.mean(high_diff_conf):.2f}")
        if med_diff_conf:
            print(f"Medium Harm Difference (15-39%): Avg confidence {np.mean(med_diff_conf):.2f}")
        if low_diff_conf:
            print(f"Low Harm Difference (<15%): Avg confidence {np.mean(low_diff_conf):.2f}")
        
        # Overall treatment confidence distribution
        treatment_conf = treatment_df['confidence']
        print(f"\nOverall Treatment Average confidence: {treatment_conf.mean():.2f}")
        print("Confidence distribution:")
        for conf in range(1, 6):
            count = len(treatment_conf[treatment_conf == conf])
            print(f"  Level {conf}: {count} responses ({count/len(treatment_conf)*100:.1f}%)")
        
        # Control samples
        control_conf = control_harm['confidence']
        print("\nControl Samples:")
        print(f"Average confidence: {control_conf.mean():.2f}")
        print("Confidence distribution:")
        for conf in range(1, 6):
            count = len(control_conf[control_conf == conf])
            print(f"  Level {conf}: {count} responses ({count/len(control_conf)*100:.1f}%)")
        
        # Compare confidence levels
        print("\nComparison:")
        print(f"Treatment vs Control confidence difference: {treatment_conf.mean() - control_conf.mean():.2f}")
        
        # Confidence vs accuracy analysis (if choice mode)
        if 'chosen_harmful' in df.columns:
            print("\nConfidence vs Choice Accuracy:")
            correct_choices_conf = []
            incorrect_choices_conf = []
            
            for _, row in treatment_df.iterrows():
                if row['chosen_harmful'] != 'No difference or Unsure':
                    if (row['chosen_harmful'] == 'A' and row['harm_A'] > row['harm_B']) or \
                       (row['chosen_harmful'] == 'B' and row['harm_B'] > row['harm_A']):
                        correct_choices_conf.append(row['confidence'])
                    else:
                        incorrect_choices_conf.append(row['confidence'])
            
            if correct_choices_conf:
                print(f"Correct choices: Avg confidence {np.mean(correct_choices_conf):.2f}")
            if incorrect_choices_conf:
                print(f"Incorrect choices: Avg confidence {np.mean(incorrect_choices_conf):.2f}")
        
        # High confidence analysis
        high_conf_threshold = 4
        high_conf_count = len(df[df['confidence'] >= high_conf_threshold])
        print(f"\nHigh Confidence Analysis (≥{high_conf_threshold}):")
        print(f"High confidence responses: {high_conf_count} ({high_conf_count/len(df)*100:.1f}%)")
        
        if high_conf_count > 0:
            high_conf_df = df[df['confidence'] >= high_conf_threshold]
            treatment_high_conf = len(high_conf_df[high_conf_df['type'] == 'treatment'])
            control_high_conf = len(high_conf_df[high_conf_df['type'] == 'control'])
            print(f"  Treatment: {treatment_high_conf} ({treatment_high_conf/high_conf_count*100:.1f}%)")
            print(f"  Control: {control_high_conf} ({control_high_conf/high_conf_count*100:.1f}%)")
    
    # Analyze responses for harmful vs non-harmful options
    print("\n=== RESPONSE ANALYSIS: HARMFUL VS NON-HARMFUL ===")
    
    if 'rating_A' in df.columns and 'rating_B' in df.columns:
        # For rating mode
        harmful_ratings = []
        non_harmful_ratings = []
        
        for _, row in df[df['type'] == 'treatment'].iterrows():
            if row['harm_A'] > 0:
                harmful_ratings.append(row['rating_A'])
                non_harmful_ratings.append(row['rating_B'])
            else:
                harmful_ratings.append(row['rating_B'])
                non_harmful_ratings.append(row['rating_A'])
        
        print("\nRating Analysis (1-5 scale):")
        print(f"Average rating for harmful options: {np.mean(harmful_ratings):.2f}")
        print(f"Average rating for non-harmful options: {np.mean(non_harmful_ratings):.2f}")
        print(f"Rating difference: {np.mean(harmful_ratings) - np.mean(non_harmful_ratings):.2f}")
        
        # Rating distribution by harm level
        print("\nAverage ratings by harm level:")
        for harm_val in sorted(df['harm_A'].unique()):
            if harm_val > 0:  # Only show harmful levels
                harmful_ratings = df[df['harm_A'] == harm_val]['rating_A'].mean()
                print(f"  {harm_val}% harm: {harmful_ratings:.2f} average rating")
    
    if 'chosen_harmful' in df.columns:
        # For choice mode
        harmful_choices = 0
        non_harmful_choices = 0
        total_choices = 0
        
        for _, row in df[df['type'] == 'treatment'].iterrows():
            if row['chosen_harmful'] != 'No difference or Unsure':
                total_choices += 1
                if (row['chosen_harmful'] == 'A' and row['harm_A'] > 0) or \
                   (row['chosen_harmful'] == 'B' and row['harm_B'] > 0):
                    harmful_choices += 1
                else:
                    non_harmful_choices += 1
        
        if total_choices > 0:
            print("\nChoice Analysis:")
            print(f"Times harmful option was chosen: {harmful_choices} ({harmful_choices/total_choices*100:.1f}%)")
            print(f"Times non-harmful option was chosen: {non_harmful_choices} ({non_harmful_choices/total_choices*100:.1f}%)")
            
            # Choice distribution by harm level
            print("\nChoice distribution by harm level:")
            for harm_val in sorted(df['harm_A'].unique()):
                if harm_val > 0:  # Only show harmful levels
                    harm_choices = len(df[(df['harm_A'] == harm_val) & (df['chosen_harmful'] == 'A')]) + \
                                 len(df[(df['harm_B'] == harm_val) & (df['chosen_harmful'] == 'B')])
                    total_harm_choices = len(df[(df['harm_A'] == harm_val) | (df['harm_B'] == harm_val)])
                    if total_harm_choices > 0:
                        print(f"  {harm_val}% harm: chosen {harm_choices} times ({harm_choices/total_harm_choices*100:.1f}%)")
    
    # Participant choice analysis
    if 'chosen_harmful' in df.columns:
        print("\n=== PARTICIPANT CHOICE ANALYSIS ===")
        choice_dist = df[df['type'] == 'treatment']['chosen_harmful'].value_counts()
        print("\nDistribution of 'More Harmful' choices:")
        for choice, count in choice_dist.items():
            print(f"  {choice}: {count} choices ({count/len(df[df['type'] == 'treatment'])*100:.1f}%)")
        
        # Analyze choice correctness
        correct_choices = 0
        total_choices = 0
        for _, row in df[df['type'] == 'treatment'].iterrows():
            if row['chosen_harmful'] != 'No difference or Unsure':
                total_choices += 1
                if (row['chosen_harmful'] == 'A' and row['harm_A'] > row['harm_B']) or \
                   (row['chosen_harmful'] == 'B' and row['harm_B'] > row['harm_A']):
                    correct_choices += 1
        
        if total_choices > 0:
            print(f"\nChoice accuracy: {correct_choices/total_choices*100:.1f}%")
    
    # Rating analysis
    if 'rating_A' in df.columns and 'rating_B' in df.columns:
        print("\n=== RATING ANALYSIS ===")
        print("\nAverage ratings by harm level:")
        for harm_val in sorted(df['harm_A'].unique()):
            avg_rating = df[df['harm_A'] == harm_val]['rating_A'].mean()
            print(f"  {harm_val}% harm: {avg_rating:.2f} average rating")
        
        print("\nRating distribution:")
        rating_ranges = [(1,2), (3,4), (5,5)]
        for low, high in rating_ranges:
            count = len(df[(df['rating_A'].between(low, high)) | (df['rating_B'].between(low, high))])
            print(f"  Ratings {low}-{high}: {count} ratings ({count/(len(df)*2)*100:.1f}%)")
    
    # Confidence analysis
    if 'confidence' in df.columns:
        print("\n=== CONFIDENCE ANALYSIS ===")
        conf_dist = df['confidence'].value_counts().sort_index()
        print("\nConfidence level distribution:")
        for conf, count in conf_dist.items():
            print(f"  Level {conf}: {count} responses ({count/len(df)*100:.1f}%)")
        
        # Confidence vs correctness (if choice mode)
        if 'chosen_harmful' in df.columns:
            print("\nAverage confidence by choice correctness:")
            correct_conf = df[((df['chosen_harmful'] == 'A') & (df['harm_A'] > df['harm_B'])) | 
                            ((df['chosen_harmful'] == 'B') & (df['harm_B'] > df['harm_A']))]['confidence'].mean()
            incorrect_conf = df[((df['chosen_harmful'] == 'A') & (df['harm_A'] < df['harm_B'])) | 
                              ((df['chosen_harmful'] == 'B') & (df['harm_B'] < df['harm_A']))]['confidence'].mean()
            print(f"  Correct choices: {correct_conf:.2f}")
            print(f"  Incorrect choices: {incorrect_conf:.2f}")
    
    print("\n=== EXPERIMENTAL CONTEXT ===")
    print("\nThis experiment examines human judgment of harmful content in AI-generated text.")
    print("Key aspects of the design:")
    print("1. Each participant evaluates multiple pairs of text (treatment items)")
    print("2. Each pair contains one harmful and one non-harmful option")
    print("3. Harm levels are distributed across different percentages (15%, 40%, 65%)")
    print("4. Control items (0% harm) are included to validate participant attention")
    print("5. The experiment includes both clear and ambiguous cases to test judgment reliability")
    print("\nThe statistics above help validate the experimental design and provide insights into")
    print("how participants perceive and evaluate different levels of harmful content.")

def generate_synthetic_data(num_participants: int, output_path: str = "data/synthetic_data.csv", num_ambiguous_participants: int = 0, mode: str = 'choice', num_treatment_per_participant: int = 24, num_control_per_participant: int = 6):
    """
    Generates synthetic data for the human study.

    Args:
        num_participants (int): The number of participants to generate data for.
        output_path (str, optional): The path to save the generated data. Defaults to "data/synthetic_data.csv".
        num_ambiguous_participants (int, optional): The number of "less sure" participants. Defaults to 0.
        mode (str, optional): The response mode ('choice' or 'rating'). Defaults to 'choice'.
        num_treatment_per_participant (int, optional): The number of treatment items per participant. Defaults to 24.
        num_control_per_participant (int, optional): The number of control items per participant. Defaults to 6.
    
    Returns:
        str: The path to the saved data file.
        
    Note:
        Generated data includes confidence levels that can be filtered during analysis using 
        print_descriptive_stats(data_path, min_confidence=X) where X is the minimum confidence level.
    """
    if num_ambiguous_participants > num_participants:
        raise ValueError("num_ambiguous_participants cannot be greater than num_participants.")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    treatment_pool = create_treatment_pool()
    all_participant_data = []
    used_samples = set()  # Track used samples across all participants

    ambiguous_participant_ids = random.sample(range(1, num_participants + 1), k=num_ambiguous_participants)
    if num_ambiguous_participants > 0:
        logging.info(f"Selected ambiguous participants: {ambiguous_participant_ids}")

    for i in range(num_participants):
        participant_id = i + 1
        is_ambiguous = participant_id in ambiguous_participant_ids
        if is_ambiguous:
            logging.info(f"Generating data for ambiguous participant {participant_id}...")
        else:
            logging.info(f"Generating data for participant {participant_id}...")
        
        participant_df = generate_participant_data(participant_id, treatment_pool, used_samples, is_ambiguous=is_ambiguous, mode=mode, num_treatment_per_participant=num_treatment_per_participant, num_control_per_participant=num_control_per_participant)
        all_participant_data.append(participant_df)

    # Combine all participant data
    all_data = pd.concat(all_participant_data, ignore_index=True)
    
    # Perform experiment-level analysis
    analyze_experiment_samples(all_data, output_path.replace('.csv', '_analysis.txt'))
    
    # Save the data
    all_data.to_csv(output_path, index=False)
    logging.info(f"Synthetic data saved to {output_path}")

    return output_path

def test_generate_synthetic_data(num_treatment_per_participant: int = 24, num_control_per_participant: int = 6):
    """Tests the data generation functions."""
    logging.info("Running tests for data generation...")

    # Test 1: Single participant data generation
    logging.info("Testing single participant data generation...")
    treatment_pool = create_treatment_pool()
    used_samples = set()
    participant_df = generate_participant_data(1, treatment_pool, used_samples, is_ambiguous=False, mode='choice', num_treatment_per_participant=num_treatment_per_participant, num_control_per_participant=num_control_per_participant)

    syntactic_data_path = "data/test_participant_data.csv"
    participant_df.to_csv(syntactic_data_path, index=False)
    
    assert os.path.exists(syntactic_data_path), "Syntactic test file was not created."
    df = pd.read_csv(syntactic_data_path)
    assert len(df) == num_treatment_per_participant + num_control_per_participant
    assert 'participant_id' in df.columns
    assert df['participant_id'].nunique() == 1
    
    # Test counts
    assert len(df[df['type'] == 'treatment']) == num_treatment_per_participant
    assert len(df[df['type'] == 'control']) == num_control_per_participant
    for model in MODELS:
        assert len(df[df['model'] == model]) == (num_treatment_per_participant + num_control_per_participant) / NUM_MODELS

    # Test harm percentages
    assert all(df[df['type'] == 'control']['harm_A'] == CONTROL_HARM_PERCENTAGE)
    assert all(df[df['type'] == 'control']['harm_B'] == CONTROL_HARM_PERCENTAGE)

    logging.info("Single participant test passed.")

    # Test 2: Multiple participants data generation
    logging.info("Testing multiple participants data generation...")
    random_data_path = "data/test_multiple_participants.csv"
    generate_synthetic_data(4, random_data_path, num_ambiguous_participants=2, mode='choice', num_treatment_per_participant=num_treatment_per_participant, num_control_per_participant=num_control_per_participant)
    
    assert os.path.exists(random_data_path), "Random test file was not created."
    df_random = pd.read_csv(random_data_path)
    assert len(df_random) == (num_treatment_per_participant + num_control_per_participant) * 4
    assert df_random['participant_id'].nunique() == 4
    logging.info("Random data tests passed.")

    # Test 3: Verify sample uniqueness tracking
    logging.info("Testing sample uniqueness...")
    if 'sample_id' in df_random.columns:
        treatment_data = df_random[df_random['type'] == 'treatment']
        unique_samples = treatment_data['sample_id'].nunique()
        total_samples = len(treatment_data)
        logging.info(f"Unique samples: {unique_samples}/{total_samples} ({unique_samples/total_samples*100:.1f}%)")
    
    # Clean up test files
    if os.path.exists(syntactic_data_path):
        os.remove(syntactic_data_path)
    if os.path.exists(random_data_path):
        os.remove(random_data_path)
    if os.path.exists(random_data_path.replace('.csv', '_analysis.txt')):
        os.remove(random_data_path.replace('.csv', '_analysis.txt'))

    logging.info("All data generation tests passed.")

def analyze_experiment_samples(all_data: pd.DataFrame, output_path: str = "data/experiment_analysis.txt"):
    """
    Analyzes and saves detailed information about all samples used in the experiment.
    
    Args:
        all_data (pd.DataFrame): The complete dataset for all participants.
        output_path (str, optional): Path to save the analysis file. Defaults to "data/experiment_analysis.txt".
    """
    analysis = {
        'total_participants': all_data['participant_id'].nunique(),
        'total_items': len(all_data),
        'treatment_items': len(all_data[all_data['type'] == 'treatment']),
        'control_items': len(all_data[all_data['type'] == 'control']),
        'unique_samples': 0,
        'duplicate_samples': 0
    }
    
    # Create detailed report
    report_lines = [
        "=== EXPERIMENT-LEVEL SAMPLE ANALYSIS ===",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OVERVIEW:",
        f"  Total participants: {analysis['total_participants']}",
        f"  Total items: {analysis['total_items']}",
        f"  Treatment items: {analysis['treatment_items']} ({analysis['treatment_items']/analysis['total_items']*100:.1f}%)",
        f"  Control items: {analysis['control_items']} ({analysis['control_items']/analysis['total_items']*100:.1f}%)",
        "",
    ]
    
    # Model distribution table
    model_analysis = []
    for model in MODELS:
        model_data = all_data[all_data['model'] == model]
        model_analysis.append({
            'Model': model,
            'Total': len(model_data),
            'Treatment': len(model_data[model_data['type'] == 'treatment']),
            'Control': len(model_data[model_data['type'] == 'control']),
            'Per_Participant_Avg': len(model_data) / analysis['total_participants']
        })
    
    report_lines.extend([
        "MODEL DISTRIBUTION TABLE:",
        f"{'Model':<12} {'Total':<8} {'Treatment':<12} {'Control':<10} {'Avg/Participant':<15}",
        "-" * 65
    ])
    
    for model_info in model_analysis:
        report_lines.append(
            f"{model_info['Model']:<12} {model_info['Total']:<8} {model_info['Treatment']:<12} "
            f"{model_info['Control']:<10} {model_info['Per_Participant_Avg']:.1f}"
        )
    
    # Topic distribution table (treatment only)
    treatment_data = all_data[all_data['type'] == 'treatment']
    topic_counts = treatment_data['topic'].value_counts().sort_index()
    
    report_lines.extend([
        "",
        "TOPIC DISTRIBUTION TABLE (Treatment only):",
        f"{'Topic':<12} {'Count':<8} {'Percentage':<12} {'Participants':<15}",
        "-" * 50
    ])
    
    for topic, count in topic_counts.items():
        participants_with_topic = treatment_data[treatment_data['topic'] == topic]['participant_id'].nunique()
        report_lines.append(
            f"{topic:<12} {count:<8} {count/len(treatment_data)*100:.1f}%{'':<7} {participants_with_topic}"
        )
    
    # Harm type distribution table
    harm_type_counts = treatment_data['harm_type'].value_counts().sort_index()
    
    report_lines.extend([
        "",
        "HARM TYPE DISTRIBUTION TABLE (Treatment only):",
        f"{'Harm Type':<12} {'Count':<8} {'Percentage':<12} {'Participants':<15}",
        "-" * 50
    ])
    
    for harm_type, count in harm_type_counts.items():
        participants_with_harm = treatment_data[treatment_data['harm_type'] == harm_type]['participant_id'].nunique()
        report_lines.append(
            f"{harm_type:<12} {count:<8} {count/len(treatment_data)*100:.1f}%{'':<7} {participants_with_harm}"
        )
    
    # Benefit reason distribution table
    benefit_counts = treatment_data['benefit_reason'].value_counts().sort_index()
    
    report_lines.extend([
        "",
        "BENEFIT REASON DISTRIBUTION TABLE (Treatment only):",
        f"{'Benefit':<12} {'Count':<8} {'Percentage':<12} {'Participants':<15}",
        "-" * 50
    ])
    
    for benefit, count in benefit_counts.items():
        participants_with_benefit = treatment_data[treatment_data['benefit_reason'] == benefit]['participant_id'].nunique()
        report_lines.append(
            f"{benefit:<12} {count:<8} {count/len(treatment_data)*100:.1f}%{'':<7} {participants_with_benefit}"
        )
    
    # Harm level distribution
    harm_a_dist = all_data['harm_A'].value_counts().sort_index()
    harm_b_dist = all_data['harm_B'].value_counts().sort_index()
    
    report_lines.extend([
        "",
        "HARM LEVEL DISTRIBUTION TABLE:",
        f"{'Harm Level':<12} {'Option A':<10} {'Option B':<10} {'Total':<10} {'Percentage':<12}",
        "-" * 60
    ])
    
    all_harm_levels = sorted(set(list(harm_a_dist.index) + list(harm_b_dist.index)))
    for harm_level in all_harm_levels:
        count_a = harm_a_dist.get(harm_level, 0)
        count_b = harm_b_dist.get(harm_level, 0)
        total = count_a + count_b
        report_lines.append(
            f"{harm_level}%{'':<9} {count_a:<10} {count_b:<10} {total:<10} {total/len(all_data)*100:.1f}%"
        )
    
    # Sample uniqueness analysis
    if 'sample_id' in treatment_data.columns:
        unique_samples = treatment_data['sample_id'].nunique()
        total_treatment_samples = len(treatment_data)
        duplicate_rate = (total_treatment_samples - unique_samples) / total_treatment_samples * 100
        
        report_lines.extend([
            "",
            "SAMPLE UNIQUENESS ANALYSIS:",
            f"  Unique treatment combinations: {unique_samples}",
            f"  Total treatment samples: {total_treatment_samples}",
            f"  Duplicate rate: {duplicate_rate:.1f}%",
        ])
        
        # Most common sample combinations
        sample_counts = treatment_data['sample_id'].value_counts().head(10)
        if len(sample_counts) > 0:
            report_lines.extend([
                "",
                "MOST COMMON SAMPLE COMBINATIONS:",
            ])
            for i, (sample_id, count) in enumerate(sample_counts.items(), 1):
                model, topic, harm_type, benefit = sample_id
                report_lines.append(
                    f"  {i}. {model}, {topic}, {harm_type}, {benefit}: {count} times"
                )
    
    # Print summary to console
    print(f"\n=== EXPERIMENT ANALYSIS SUMMARY ===")
    print(f"Participants: {analysis['total_participants']}, Total items: {analysis['total_items']}")
    print(f"Treatment: {analysis['treatment_items']}, Control: {analysis['control_items']}")
    if 'sample_id' in treatment_data.columns:
        unique_samples = treatment_data['sample_id'].nunique()
        print(f"Unique treatment combinations: {unique_samples}/{len(treatment_data)} ({unique_samples/len(treatment_data)*100:.1f}%)")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Experiment analysis saved to {output_path}")
    
    return analysis

if __name__ == '__main__':
    test_generate_synthetic_data()
    # Generate a sample file for manual inspection
    generate_synthetic_data(8, "data/final_synthetic_data.csv", num_ambiguous_participants=2, mode='choice')
    generate_synthetic_data(8, "data/final_rating_data.csv", num_ambiguous_participants=2, mode='rating') 