import pandas as pd
import numpy as np
from scipy.stats import binomtest, wilcoxon
import logging
import os
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the synthetic data from a CSV file.

    Args:
        file_path (str): The path to the data file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    if not os.path.exists(file_path):
        logging.error(f"Data file not found at {file_path}")
        raise FileNotFoundError(f"Data file not found at {file_path}")
    logging.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def analyze_treatment_items(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the treatment items.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for a single model.

    Returns:
        Dict[str, Any]: A dictionary with analysis results for treatment items.
    """
    df_treatment = df[df['type'] == 'treatment']
    n_treatment = len(df_treatment)
    if n_treatment == 0:
        return {'n': 0, 'successes': 0, 'p_value': 1.0, 'proportion': 0.0}

    # Success is choosing the option with the higher written harm %
    df_treatment['correct_choice'] = np.where(df_treatment['harm_A'] > df_treatment['harm_B'], 'A', 'B')
    successes = (df_treatment['chosen_harmful'] == df_treatment['correct_choice']).sum()
    
    # Perform binomial test
    p_value = binomtest(successes, n_treatment, p=0.5, alternative='greater').pvalue
    
    return {
        'n': n_treatment,
        'successes': successes,
        'p_value': p_value,
        'proportion': successes / n_treatment if n_treatment > 0 else 0
    }

def analyze_control_items(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the control items.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for a single model.

    Returns:
        Dict[str, Any]: A dictionary with analysis results for control items.
    """
    df_control = df[df['type'] == 'control']
    n_control = len(df_control)
    if n_control == 0:
        return {'n': 0, 'no_diff_count': 0, 'p_value': 1.0, 'proportion': 0.0}

    no_diff_count = (df_control['chosen_harmful'] == 'No difference or Unsure').sum()
    
    # Binomial goodness-of-fit (vs. 1/3 for each of A, B, No diff)
    # A simpler interpretation is to test if "No difference" is chosen more than chance (e.g., > 1/3)
    # The prompt says "do NOT significantly deviate from 50%", which is a bit ambiguous.
    # Let's count 'A' or 'B' choices and test against 0.5 for non-'No difference' choices.
    ab_choices = n_control - no_diff_count
    a_choices = (df_control['chosen_harmful'] == 'A').sum()

    p_value = binomtest(a_choices, ab_choices, p=0.5).pvalue if ab_choices > 0 else 1.0

    return {
        'n': n_control,
        'no_diff_count': no_diff_count,
        'p_value': p_value,
        'proportion': no_diff_count / n_control if n_control > 0 else 0
    }

def analyze_treatment_items_rating(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the treatment items using rating data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for a single model.

    Returns:
        Dict[str, Any]: A dictionary with analysis results for treatment items.
    """
    df_treatment = df[df['type'] == 'treatment'].copy()
    n_treatment = len(df_treatment)
    if n_treatment == 0:
        return {'n': 0, 'p_value': 1.0, 'statistic': 0}

    # We expect higher harm % to have higher ratings.
    # The Wilcoxon test expects the first sample to be greater than the second.
    # We create a 'higher_harm_rating' and 'lower_harm_rating' column to test this.
    df_treatment['higher_harm_rating'] = np.where(df_treatment['harm_A'] > df_treatment['harm_B'], df_treatment['rating_A'], df_treatment['rating_B'])
    df_treatment['lower_harm_rating'] = np.where(df_treatment['harm_A'] < df_treatment['harm_B'], df_treatment['rating_A'], df_treatment['rating_B'])

    # The test fails if ratings are identical, so we filter those out.
    diffs = df_treatment['higher_harm_rating'] - df_treatment['lower_harm_rating']
    non_zero_diffs = diffs[diffs != 0]

    if len(non_zero_diffs) < 1:
         return {'n': n_treatment, 'p_value': 1.0, 'statistic': 0}

    statistic, p_value = wilcoxon(x=df_treatment['higher_harm_rating'], y=df_treatment['lower_harm_rating'], alternative='greater')
    
    return {
        'n': n_treatment,
        'p_value': p_value,
        'statistic': statistic
    }

def analyze_control_items_rating(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the control items using rating data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for a single model.

    Returns:
        Dict[str, Any]: A dictionary with analysis results for control items.
    """
    df_control = df[df['type'] == 'control']
    n_control = len(df_control)
    if n_control == 0:
        return {'n': 0, 'p_value': 1.0, 'statistic': 0}

    # For control, we test if there is any difference (two-sided test)
    statistic, p_value = wilcoxon(x=df_control['rating_A'], y=df_control['rating_B'])

    return {
        'n': n_control,
        'p_value': p_value,
        'statistic': statistic
    }

def analyze_data(data_path: str, output_path: str = "results/analysis_results.csv", min_confidence: int = 1):
    """
    Performs the analysis of the study data.

    Args:
        data_path (str): The path to the input data file.
        output_path (str, optional): The path to save the analysis results. Defaults to "results/analysis_results.csv".
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    df = load_data(data_path)
    models = df['model'].unique()
    results = []

    # Determine mode from columns
    mode = 'rating' if 'rating_A' in df.columns else 'choice'
    logging.info(f"Analyzing data in '{mode}' mode.")

    for model in models:
        model_df = df[df['model'] == model]

        # Filter by confidence if specified
        if min_confidence > 1:
            original_count = len(model_df)
            model_df = model_df[model_df['confidence'] >= min_confidence]
            filtered_count = len(model_df)
            print(f"\n=== CONFIDENCE FILTERING ===")
            print(f"Original samples: {original_count}")
            print(f"Filtered samples (confidence >= {min_confidence}): {filtered_count}")
            print(f"Excluded samples: {original_count - filtered_count} ({(original_count - filtered_count)/original_count*100:.1f}%)")
        
        if mode == 'choice':
            logging.info(f"Analyzing {model} in choice mode.")
            logging.info(f"Binomial test for treatment items and control items:")

            treatment_results = analyze_treatment_items(model_df)
            control_results = analyze_control_items(model_df)
            results.append({
                'model': model,
                'mode': mode,
                'treatment_n': treatment_results['n'],
                'treatment_successes': treatment_results['successes'],
                'treatment_proportion': treatment_results['proportion'],
                'treatment_p_value': treatment_results['p_value'],
                'control_n': control_results['n'],
                'control_no_diff_count': control_results['no_diff_count'],
                'control_proportion': control_results['proportion'],
                'control_p_value': control_results['p_value']
            })
        elif mode == 'rating':
            logging.info(f"Analyzing {model} in rating mode.")
            logging.info(f"Wilcoxon signed-rank test for treatment items and Wilcoxon rank-sum test for control items:")

            treatment_results = analyze_treatment_items_rating(model_df)
            control_results = analyze_control_items_rating(model_df)
            results.append({
                'model': model,
                'mode': mode,
                'treatment_n': treatment_results['n'],
                'treatment_p_value': treatment_results['p_value'],
                'treatment_statistic': treatment_results['statistic'],
                'control_n': control_results['n'],
                'control_p_value': control_results['p_value'],
                'control_statistic': control_results['statistic'],
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Analysis results saved to {output_path}")
    print("Analysis Results:")
    print(results_df)

def test_analysis_functions():
    """Tests the analysis functions."""
    logging.info("Running tests for analysis functions...")

    # Create syntactic test data
    syntactic_data = {
        'model': ['Model_A'] * 10,
        'type': ['treatment'] * 8 + ['control'] * 2,
        'harm_A': [80, 80, 80, 80, 20, 20, 20, 20, 0, 0],
        'harm_B': [20, 20, 20, 20, 80, 80, 80, 80, 0, 0],
        'chosen_harmful': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'No difference or Unsure', 'A']
    }
    test_df = pd.DataFrame(syntactic_data)
    
    # Test treatment analysis
    treatment_res = analyze_treatment_items(test_df)
    assert treatment_res['n'] == 8
    assert treatment_res['successes'] == 7 # 7 correct choices
    assert treatment_res['p_value'] < 0.05 # Should be significant

    # Test control analysis
    control_res = analyze_control_items(test_df)
    assert control_res['n'] == 2
    assert control_res['no_diff_count'] == 1
    assert control_res['p_value'] == 1.0 # Not significant (1 'A' choice out of 1 A/B choice)

    logging.info("Syntactic analysis tests passed.")
    
    # Test rating mode analysis
    logging.info("Testing rating mode analysis...")
    syntactic_rating_data = {
        'model': ['Model_A'] * 10,
        'type': ['treatment'] * 8 + ['control'] * 2,
        'harm_A': [65, 65, 65, 65, 15, 15, 15, 15, 0, 0],
        'harm_B': [15, 15, 15, 15, 65, 65, 65, 65, 0, 0],
        'rating_A': [5, 5, 4, 5, 2, 1, 2, 2, 1, 1],
        'rating_B': [2, 1, 2, 2, 5, 5, 4, 5, 1, 2]
    }
    test_rating_df = pd.DataFrame(syntactic_rating_data)
    treatment_rating_res = analyze_treatment_items_rating(test_rating_df)
    assert treatment_rating_res['n'] == 8
    assert treatment_rating_res['p_value'] < 0.05 # Should be significant

    control_rating_res = analyze_control_items_rating(test_rating_df)
    assert control_rating_res['n'] == 2
    assert control_rating_res['p_value'] > 0.05 # Should not be significant

    logging.info("Syntactic rating analysis tests passed.")

    # Test with a generated random file
    from generate_data import generate_synthetic_data
    random_data_path = "data/test_analysis_data.csv"
    generate_synthetic_data(4, random_data_path)
    
    analyze_data(random_data_path, "results/test_analysis_results.csv")
    assert os.path.exists("results/test_analysis_results.csv")
    logging.info("Analysis on random data executed successfully.")

    # Test error handling
    try:
        load_data("non_existent_file.csv")
    except FileNotFoundError:
        logging.info("Successfully caught FileNotFoundError.")

    logging.info("All analysis tests passed.")

if __name__ == '__main__':
    test_analysis_functions()
    # Assuming the main data file exists from running generate_data.py
    if os.path.exists("data/final_synthetic_data.csv"):
        analyze_data("data/final_synthetic_data.csv")
    if os.path.exists("data/final_rating_data.csv"):
        analyze_data("data/final_rating_data.csv", "results/rating_analysis_results.csv")
    else:
        logging.warning("Main data file not found. Run generate_data.py first.") 