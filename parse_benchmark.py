import json
import logging
import os
import re
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and validate a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        Dict[str, Any]: Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        raise json.JSONDecodeError(f"Invalid JSON format in {file_path}: {e}")


def extract_model_and_sample_type(filename: str) -> Tuple[str, str]:
    """
    Extract model type and sample type from filename.
    
    Args:
        filename (str): Filename to parse (e.g., "chatgpt_results_dict_f.json")
        
    Returns:
        Tuple[str, str]: Model type and sample type
        
    Raises:
        ValueError: If filename format is invalid
    """
    # Remove file extension
    base_name = filename.replace('.json', '')
    
    # Determine sample type based on "_f" suffix
    if base_name.endswith('_f_dict'):
        sample_type = "Control"
        model_name = base_name.replace('_results_f_dict', '')
    elif base_name.endswith('_results_dict'):
        sample_type = "Treatment" 
        model_name = base_name.replace('_results_dict', '')
    else:
        raise ValueError(f"Invalid filename format: {filename}")
    
    # Capitalize model name
    model_type = model_name.capitalize()
    
    return model_type, sample_type


def _clean_gemini_scenario_prefix(scenario_text: str) -> str:
    """
    Clean Gemini-specific scenario prefix formatting.
    
    Args:
        scenario_text (str): Raw scenario text that may contain Gemini prefixes
        
    Returns:
        str: Cleaned scenario text
    """
    cleaned_text = scenario_text
    
    # Handle multiple **Scenario: prefixes by keeping content after the first one
    if "**Scenario:" in cleaned_text:
        parts = cleaned_text.split("**Scenario:")
        if len(parts) > 1:
            # Rejoin with single **Scenario: prefix
            cleaned_text = "**Scenario:" + "**Scenario:".join(parts[1:])
            logging.debug("Applied Gemini scenario prefix cleanup")
    
    return cleaned_text


def _find_option_split_patterns(text: str) -> List[Tuple[str, str]]:
    """
    Find all possible Option A and Option B patterns in the text.
    
    Args:
        text (str): Text to search for option patterns
        
    Returns:
        List[Tuple[str, str]]: List of (option_a_pattern, option_b_pattern) pairs found
    """
    # Define option patterns in order of preference (most specific first)
    option_patterns = [
        # Standard markdown patterns
        ("**Option A:**", "**Option B:**"),
        ("### Option A:", "### Option B:"),
        ("**Option A**:", "**Option B**:"),
        ("## Option A:", "## Option B:"),
        ("### Option A", "### Option B"),
        ("## Option A", "## Option B"),
        
        # Gemini-specific patterns (with and without markdown)
        ("**Option A: ", "**Option B: "),  # Gemini with markdown and space
        ("Option A: ", "Option B: "),      # Gemini without markdown but with space
        
        # Fallback patterns
        ("Option A:", "Option B:"),
        ("option A:", "option B:"),
    ]
    
    found_patterns = []
    
    for pattern_a, pattern_b in option_patterns:
        if pattern_a in text and pattern_b in text:
            found_patterns.append((pattern_a, pattern_b))
            logging.debug(f"Found option pattern: {pattern_a} / {pattern_b}")
    
    return found_patterns


def _clean_scenario_trailing_markers(scenario: str) -> str:
    """
    Clean trailing markdown markers from scenario text.
    
    Args:
        scenario (str): Scenario text that may have trailing markers
        
    Returns:
        str: Cleaned scenario text
    """
    # Remove trailing ** markers that might appear at the end of scenario sections
    cleaned = scenario.rstrip()
    if cleaned.endswith('**'):
        cleaned = cleaned[:-2].rstrip()
        logging.debug("Removed trailing ** markers from scenario")
    
    return cleaned


def _clean_option_text(option_text: str) -> str:
    """
    Clean option text by adding consistent markdown formatting if missing.
    
    Args:
        option_text (str): Raw option text
        
    Returns:
        str: Cleaned and standardized option text
    """
    cleaned = option_text.strip()
    
    # Handle cases where option text ends with ** (Gemini formatting)
    if cleaned.endswith('**'):
        cleaned = cleaned[:-2].rstrip()
        logging.debug("Removed trailing ** markers from option text")
    
    # Ensure consistent formatting: if it starts with Option A/B but no **, add them
    if cleaned.startswith('Option A: ') and not cleaned.startswith('**Option A:'):
        cleaned = '**' + cleaned + '**'
        logging.debug("Added ** markdown formatting to Option A")
    elif cleaned.startswith('Option B: ') and not cleaned.startswith('**Option B:'):
        cleaned = '**' + cleaned + '**'
        logging.debug("Added ** markdown formatting to Option B")
    
    return cleaned


def _extract_scenario_and_options(text: str, pattern_a: str, pattern_b: str) -> Tuple[str, str, str]:
    """
    Extract scenario and options using specific patterns.
    
    Args:
        text (str): Full scenario text
        pattern_a (str): Pattern for Option A
        pattern_b (str): Pattern for Option B
        
    Returns:
        Tuple[str, str, str]: Scenario, Option A text (including header), Option B text (including header)
        
    Raises:
        ValueError: If extraction fails
    """
    # Split on Option A pattern
    option_a_parts = text.split(pattern_a)
    if len(option_a_parts) != 2:
        raise ValueError(f"Could not split on pattern: {pattern_a}")
    
    scenario = option_a_parts[0].strip()
    options_text = pattern_a + option_a_parts[1]
    
    # Split options text on Option B pattern
    option_b_parts = options_text.split(pattern_b)
    if len(option_b_parts) != 2:
        raise ValueError(f"Could not split on pattern: {pattern_b}")
    
    option_a_text = option_b_parts[0].strip()
    option_b_text = (pattern_b + option_b_parts[1]).strip()
    
    # Clean scenario of trailing markers (especially for Gemini content)
    scenario = _clean_scenario_trailing_markers(scenario)
    
    # Clean and standardize option texts (especially for Gemini content)
    option_a_text = _clean_option_text(option_a_text)
    option_b_text = _clean_option_text(option_b_text)
    
    # Validate that we have non-empty content
    if not scenario.strip():
        raise ValueError("Scenario section is empty")
    if not option_a_text.strip():
        raise ValueError("Option A section is empty")
    if not option_b_text.strip():
        raise ValueError("Option B section is empty")
    
    return scenario, option_a_text, option_b_text


def parse_scenario_text(scenario_text: str, is_gemini: bool = False) -> Tuple[str, str, str]:
    """
    Parse scenario text into scenario, option A, and option B with robust pattern matching.
    
    This function handles multiple formatting patterns for options and includes special
    handling for Gemini-generated content with scenario prefix cleanup.
    
    Args:
        scenario_text (str): Full scenario text containing scenario and options
        is_gemini (bool): Whether this is Gemini-generated content requiring special handling
        
    Returns:
        Tuple[str, str, str]: Scenario description, Option A text, Option B text
        
    Raises:
        ValueError: If scenario format is invalid or no recognized patterns are found
    """
    if not scenario_text or not scenario_text.strip():
        #raise ValueError("Scenario text is empty or None")
        logging.warning("Scenario text is empty or None")
        return "No scenario text", "No option A", "No option B"
    
    try:
        # Step 1: Apply Gemini-specific cleaning if needed
        cleaned_text = scenario_text
        if is_gemini:
            cleaned_text = _clean_gemini_scenario_prefix(scenario_text)
        
        # Step 2: Find all possible option patterns
        found_patterns = _find_option_split_patterns(cleaned_text)
        
        if not found_patterns:
            logging.warning("No standard option patterns found, attempting fallback parsing")
            # Fallback to original regex-based approach
            parts = re.split(r'Option [AB]:', cleaned_text, flags=re.IGNORECASE)
            if len(parts) == 3:
                scenario = parts[0].strip()
                option_a = "Option A:" + parts[1].strip()
                option_b = "Option B:" + parts[2].strip()
                
                if scenario and option_a and option_b:
                    logging.info("Successfully parsed using fallback regex method")
                    return scenario, option_a, option_b
            
            raise ValueError("No recognized option patterns found in scenario text")
        
        # Step 3: Try each pattern in order of preference
        last_error = None
        for pattern_a, pattern_b in found_patterns:
            try:
                scenario, option_a, option_b = _extract_scenario_and_options(
                    cleaned_text, pattern_a, pattern_b
                )
                
                logging.debug(f"Successfully parsed using patterns: {pattern_a} / {pattern_b}")
                return scenario, option_a, option_b
                
            except ValueError as e:
                last_error = e
                logging.debug(f"Failed to parse with patterns {pattern_a}/{pattern_b}: {e}")
                continue
        
        # If we get here, all patterns failed
        raise ValueError(f"All parsing attempts failed. Last error: {last_error}")
        
    except Exception as e:
        logging.error(f"Error parsing scenario text: {e}")
        logging.debug(f"Scenario text preview: {scenario_text[:200]}...")
        raise ValueError(f"Failed to parse scenario text: {e}")


def create_sample_entry(
    sample_id: str,
    scenario: str, 
    option_a: str,
    option_b: str,
    model_type: str,
    sample_type: str,
    topic: str,
    harm_type: str,
    benefit: str
) -> Dict[str, Any]:
    """
    Create a standardized sample entry for the output JSON.
    
    Args:
        sample_id (str): Unique identifier for the sample
        scenario (str): Scenario description text
        option_a (str): Option A text
        option_b (str): Option B text
        model_type (str): Model type (e.g., "ChatGPT")
        sample_type (str): Sample type ("Treatment" or "Control")
        topic (str): Topic category
        harm_type (str): Type of harm
        benefit (str): Type of benefit
        
    Returns:
        Dict[str, Any]: Formatted sample entry
    """
    return {
        "id": sample_id,
        "scenario": scenario,
        "option_a": option_a,
        "option_b": option_b,
        "metadata": {
            "model_type": model_type,
            "sample_type": sample_type,
            "topic": topic,
            "harm_type": harm_type,
            "benefit": benefit
        }
    }


def process_single_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a single benchmark file and extract all samples.
    
    Args:
        file_path (str): Path to the benchmark file
        
    Returns:
        List[Dict[str, Any]]: List of parsed sample entries
    """
    filename = os.path.basename(file_path)
    model_type, sample_type = extract_model_and_sample_type(filename)
    
    data = load_json_file(file_path)
    samples = []
    sample_counter = 0
    error_counter = 0
    # Iterate through nested structure: topic -> harm_type -> benefit -> scenario_text
    for topic, harm_types in data.items():
        if not isinstance(harm_types, dict):
            logging.warning(f"Unexpected structure in topic '{topic}' - skipping")
            #continue
            raise ValueError(f"Unexpected structure in topic '{topic}' - skipping")
            
        for harm_type, benefits in harm_types.items():
            if not isinstance(benefits, dict):
                logging.warning(f"Unexpected structure in harm_type '{harm_type}' - skipping")
                #continue
                raise ValueError(f"Unexpected structure in harm_type '{harm_type}' - skipping")
                
            for benefit, scenario_text in benefits.items():
                if not isinstance(scenario_text, str):
                    logging.warning(f"Unexpected scenario text type for {topic}/{harm_type}/{benefit} - skipping")
                    #continue
                    raise ValueError(f"Unexpected scenario text type for {topic}/{harm_type}/{benefit} - skipping")
                
                try:
                    is_gemini = model_type == "Gemini"
                    scenario, option_a, option_b = parse_scenario_text(scenario_text, is_gemini=is_gemini)
                    if scenario == "No scenario text":
                        logging.warning(f"No scenario text found for {topic}/{harm_type}/{benefit} - skipping")
                        error_counter += 1
                        continue
                        #raise ValueError(f"No scenario text found for {topic}/{harm_type}/{benefit} - skipping")
                    
                    sample_entry = create_sample_entry(
                        sample_id=str(sample_counter),
                        scenario=scenario,
                        option_a=option_a,
                        option_b=option_b,
                        model_type=model_type,
                        sample_type=sample_type,
                        topic=topic,
                        harm_type=harm_type,
                        benefit=benefit
                    )
                    
                    samples.append(sample_entry)
                    sample_counter += 1
                    
                except ValueError as e:
                    logging.error(f"Failed to parse scenario in {filename} - {topic}/{harm_type}/{benefit}: {e}")
                    #continue
                    raise e
    
    logging.info(f"Processed {len(samples)} samples from {filename}")
    logging.info(f"Skipped {error_counter} samples due to errors")
    return samples


def get_benchmark_files(benchmark_dir: str = "benchmark") -> List[str]:
    """
    Get list of all benchmark JSON files.
    
    Args:
        benchmark_dir (str): Directory containing benchmark files
        
    Returns:
        List[str]: List of benchmark file paths
        
    Raises:
        FileNotFoundError: If benchmark directory doesn't exist
    """
    if not os.path.exists(benchmark_dir):
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    
    # Expected file patterns
    expected_files = [
        "chatgpt_results_dict.json",     # Treatment
        "chatgpt_results_f_dict.json",   # Control
        "claude_results_dict.json",      # Treatment
        "claude_results_f_dict.json",    # Control
        "gemini_results_dict.json",      # Treatment
        "gemini_results_f_dict.json"     # Control
    ]
    
    file_paths = []
    for filename in expected_files:
        file_path = os.path.join(benchmark_dir, filename)
        if os.path.exists(file_path):
            file_paths.append(file_path)
        else:
            #logging.warning(f"Expected file not found: {file_path}")
            raise FileNotFoundError(f"Expected file not found: {file_path}")
    
    if not file_paths:
        raise FileNotFoundError(f"No valid benchmark files found in {benchmark_dir}")
    
    logging.info(f"Found {len(file_paths)} benchmark files")
    return file_paths


def parse_all_benchmark_files(
    benchmark_dir: str = "benchmark", 
    output_file: str = "benchmark/parsed_benchmark_data.json"
) -> None:
    """
    Parse all benchmark files and create unified JSON output.
    
    Args:
        benchmark_dir (str): Directory containing benchmark files
        output_file (str): Path for output JSON file
        
    Raises:
        Exception: If parsing fails
    """
    try:
        # Get all benchmark files
        file_paths = get_benchmark_files(benchmark_dir)
        
        all_samples = []
        
        # Process each file
        for file_path in file_paths:
            try:
                samples = process_single_file(file_path)
                all_samples.extend(samples)
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")
                #continue
                raise e
        
        if not all_samples:
            raise ValueError("No samples were successfully parsed from any files")
        
        # Re-assign sequential IDs to all samples
        for i, sample in enumerate(all_samples):
            sample["id"] = str(i)
        
        # Write output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Successfully parsed {len(all_samples)} total samples")
        logging.info(f"Output written to: {output_file}")
        
        # Print summary statistics
        print_summary_stats(all_samples)
        
    except Exception as e:
        logging.error(f"Failed to parse benchmark files: {e}")
        raise e


def print_summary_stats(samples: List[Dict[str, Any]]) -> None:
    """
    Print summary statistics of parsed data.
    
    Args:
        samples (List[Dict[str, Any]]): List of parsed samples
    """
    print("\n" + "="*50)
    print("PARSING SUMMARY")
    print("="*50)
    print(f"Total samples: {len(samples)}")
    
    # Count by model type and sample type
    model_counts = {}
    sample_type_counts = {}
    topic_counts = {}
    
    for sample in samples:
        metadata = sample["metadata"]
        
        model_type = metadata["model_type"]
        sample_type = metadata["sample_type"]
        topic = metadata["topic"]
        
        model_counts[model_type] = model_counts.get(model_type, 0) + 1
        sample_type_counts[sample_type] = sample_type_counts.get(sample_type, 0) + 1
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print(f"\nBy Model Type:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}")
    
    print(f"\nBy Sample Type:")
    for sample_type, count in sorted(sample_type_counts.items()):
        print(f"  {sample_type}: {count}")
    
    print(f"\nBy Topic ({len(topic_counts)} topics):")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count}")
    
    print("="*50)


def main():
    """Main function to run the benchmark parsing."""
    try:
        parse_all_benchmark_files()
        print("\n Benchmark parsing completed successfully!")
    except Exception as e:
        print(f"\n Benchmark parsing failed: {e}")
        raise e
    return 0


if __name__ == "__main__":
    exit(main())
