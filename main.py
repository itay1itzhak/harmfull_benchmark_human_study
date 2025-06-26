import argparse
import logging
import random
from generate_data import generate_synthetic_data, print_descriptive_stats
from analysis import analyze_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """
    Main function to run the synthetic data generation and analysis pipeline.
    """
    logging.info("Starting the synthetic data generation and analysis pipeline.")
    
    # Parameters
    num_participants = args.num_participants
    mode = args.mode
    num_ambiguous_participants = args.num_ambiguous_participants
    data_path = args.data_path
    results_path = args.results_path
    num_treatment_per_participant = args.num_treatment_per_participant
    num_control_per_participant = args.num_control_per_participant

    if mode == "choice":
        data_path = "data/choice_synthetic_data.csv"
        results_path = "results/choice_analysis_results.csv"
    elif mode == "rating":
        data_path = "data/rating_synthetic_data.csv"
        results_path = "results/rating_analysis_results.csv"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if num_ambiguous_participants > num_participants:
        raise ValueError(f"num_ambiguous_participants cannot be greater than num_participants.")

    # --- Argument-based Run ---
    logging.info("--- Starting Argument-based Run ---")
    logging.info(f"Generating synthetic data for {num_participants} participants in {mode} mode with {num_ambiguous_participants} ambiguous annotator(s).")
    generate_synthetic_data(num_participants, data_path, num_ambiguous_participants=num_ambiguous_participants, mode=mode, num_treatment_per_participant=num_treatment_per_participant, num_control_per_participant=num_control_per_participant)
    print_descriptive_stats(data_path, min_confidence=args.min_confidence)
    logging.info(f"Data generation complete. Data saved to {data_path}")
    logging.info("Starting analysis for data.")
    analyze_data(data_path, results_path, min_confidence=args.min_confidence)
    logging.info(f"Analysis complete. Results saved to {results_path}")
    logging.info("--- Argument-based Run complete ---")
    
    # # --- Standard Run (Choice Mode) ---
    # logging.info("--- Starting Standard Run (Choice Mode) ---")

    # logging.info(f"Generating standard synthetic data for {num_participants} participants.")
    
    # generate_synthetic_data(num_participants, data_path, num_ambiguous_participants=num_ambiguous_participants, mode=mode)
    # logging.info(f"Data generation complete. Data saved to {data_path}")
    # logging.info("Starting analysis for standard data.")
    
    # analyze_data(data_path, results_path)
    # logging.info(f"Analysis complete. Results saved to {results_path}")

    # # --- Ambiguous Run (Choice Mode) ---
    # logging.info("--- Starting Ambiguous Run (Choice Mode) ---")
    # logging.info(f"Generating ambiguous synthetic data for {num_participants} participants with {num_ambiguous_participants} ambiguous annotator(s).")
    # generate_synthetic_data(num_participants, data_path, num_ambiguous_participants=num_ambiguous_participants, mode=args.mode)
    # logging.info(f"Data generation complete. Data saved to {data_path}")
    # logging.info("Starting analysis for ambiguous data.")
    # analyze_data(data_path, results_path)
    # logging.info(f"Analysis complete. Results saved to {results_path}")

    # # --- Rating Mode Run ---
    # logging.info("--- Starting Rating Mode Run ---")
    # logging.info(f"Generating rating-based synthetic data for {num_participants} participants.")
    # generate_synthetic_data(num_participants, data_path, num_ambiguous_participants=num_ambiguous_participants, mode="rating")
    # logging.info(f"Data generation complete. Data saved to {data_path}")
    # logging.info("Starting analysis for rating data.")
    # analyze_data(data_path, results_path)
    # logging.info(f"Analysis complete. Results saved to {results_path}")


    logging.info("Pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the synthetic data generation and analysis pipeline.")
    parser.add_argument("--num_participants", type=int, default=4, help="Number of participants to generate data for.")
    parser.add_argument("--mode", type=str, default="choice", choices=["choice", "rating"], help="Response mode: choice or rating.")
    parser.add_argument("--num_ambiguous_participants", type=int, default=2, help="Number of ambiguous participants to generate data for.")
    parser.add_argument("--min_confidence", type=int, default=2, help="Minimum confidence level to include in analysis.")
    parser.add_argument("--data_path", type=str, default="data/synthetic_data.csv", help="Path to save the generated data.")
    parser.add_argument("--results_path", type=str, default="results/analysis_results.csv", help="Path to save the analysis results.")
    parser.add_argument("--num_treatment_per_participant", type=int, default=24, help="Number of treatment items per participant.")
    parser.add_argument("--num_control_per_participant", type=int, default=6, help="Number of control items per participant.")
    args = parser.parse_args()
    main(args) 