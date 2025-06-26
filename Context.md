# Project Context

This document outlines the structure and workflow of the human study project for a harm benchmark.

## Project Structure

The project is organized into the following files and directories:

- `generate_data.py`: Contains functions to generate synthetic data for the study with advanced features.
- `analysis.py`: Contains functions to analyze the generated data.
- `parse_benchmark.py`: Contains functions to parse benchmark JSON files from different AI models into a unified format.
- `explore_benchmark.py`: **NEW** - Streamlit application for interactive exploration of benchmark examples.
- `run_streamlit.py`: **NEW** - Helper script to launch the Streamlit benchmark explorer.
- `main.py`: The main script to run the entire data generation and analysis pipeline.
- `requirements.txt`: Lists the Python dependencies for the project.
- `benchmark/`: Directory containing benchmark JSON files from different AI models (ChatGPT, Claude, Gemini).
- `data/`: Directory to store the generated data files and experiment analysis.
- `results/`: Directory to store the analysis results.
- `Context.md`: This file.

## File Details

### `explore_benchmark.py`

**NEW** - Interactive Streamlit application for exploring benchmark examples with advanced filtering and visualization capabilities.

#### Key Features:

- **Comprehensive Filtering**: Filter by `id`, `model_type`, `topic`, `harm_type`, `benefit`, and `sample_type`
- **Text Search**: Search across scenario descriptions and option text
- **Professional UI**: Clean, intuitive interface with tabbed navigation
- **Formatted Display**: Markdown rendering with proper line breaks for scenarios and options
- **Raw JSON Inspector**: View and export raw JSON data for any sample
- **Statistical Summaries**: Real-time statistics with charts and distributions
- **Export Capabilities**: Download filtered data as CSV or JSON
- **Pagination**: Handle large datasets with configurable page sizes

#### Core Functions:

- **`load_benchmark_data() -> List[Dict[str, Any]]`**: Loads and caches benchmark data from parsed JSON file
- **`create_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame`**: Converts raw data to structured DataFrame
- **`apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame`**: Applies multiple filter criteria
- **`format_text_for_display(text: str) -> str`**: Formats text with proper line breaks and markdown
- **`display_sample(sample_data: pd.Series, show_metadata: bool)`**: Renders formatted sample display
- **`display_summary_stats(df: pd.DataFrame)`**: Shows real-time statistics and visualizations

#### Usage:
```bash
# Option 1: Using helper script
python run_streamlit.py

# Option 2: Direct streamlit command
streamlit run explore_benchmark.py
```

#### Interface Sections:

1. **Sidebar Filters**:
   - Exact ID numeric input
   - Text search across all content
   - Multi-select dropdowns for all categorical fields
   - Real-time filtering with immediate results

2. **Browse Samples Tab**:
   - Paginated sample display
   - Expandable cards with formatted content
   - Scenario and options with proper markdown rendering
   - Metadata display with export options

3. **Raw JSON Inspector Tab**:
   - Sample selector dropdown
   - Pretty-printed JSON display
   - Individual sample download capability

4. **Statistics Tab**:
   - Summary metrics (total samples, unique topics, models)
   - Distribution charts for models and sample types
   - Detailed breakdowns and cross-tabulations
   - Real-time updates based on filters

5. **Export Tab**:
   - CSV and JSON export of filtered data
   - Export summary with metadata
   - Bulk download capabilities

### `run_streamlit.py`

**NEW** - Helper script to easily launch the Streamlit benchmark explorer with proper setup validation.

#### Functions:

- **`main()`**: Validates data files exist and launches Streamlit server with appropriate configuration

### `parse_benchmark.py`

This module is responsible for parsing benchmark files from 6 JSON files (3 models × 2 conditions) into a unified JSON format.

#### Key Functions:

- **`load_json_file(file_path: str) -> Dict[str, Any]`**: Loads and validates JSON files with proper error handling.
- **`extract_model_and_sample_type(filename: str) -> Tuple[str, str]`**: Extracts model type and sample type from filename (e.g., "ChatGPT", "Treatment").
- **`parse_scenario_text(scenario_text: str) -> Tuple[str, str, str]`**: Parses scenario text into scenario description, Option A, and Option B using regex.
- **`create_sample_entry(...) -> Dict[str, Any]`**: Creates standardized sample entries for the output JSON format.
- **`process_single_file(file_path: str) -> List[Dict[str, Any]]`**: Processes a single benchmark file and extracts all samples.
- **`get_benchmark_files(benchmark_dir: str) -> List[str]`**: Gets list of all expected benchmark JSON files.
- **`parse_all_benchmark_files(...) -> None`**: Main function that parses all benchmark files and creates unified JSON output.
- **`print_summary_stats(samples: List[Dict[str, Any]]) -> None`**: Prints comprehensive summary statistics of parsed data.
- **`main()`**: Entry point with error handling for the benchmark parsing process.

#### Input Structure:
- Benchmark files contain nested structure: `{topic: {harm_type: {benefit: scenario_text}}}`
- Files ending in `_f` are control examples, others are treatment examples
- Scenario text contains "**Option A:**" and "**Option B:**" sections for parsing

#### Output Format:
Each sample contains:
```json
{
  "id": "0",
  "scenario": "scenario description text",
  "option_a": "option A text", 
  "option_b": "option B text",
  "metadata": {
    "model_type": "ChatGPT",
    "sample_type": "Treatment/Control", 
    "topic": "Transportation",
    "harm_type": "Temporary illness",
    "benefit": "Encourage more usage of the AI"
  }
}
```

### `generate_data.py`

This module is responsible for creating synthetic data that simulates the responses of participants in the human study. It supports two modes: `'choice'` and `'rating'` with advanced simulation features.

#### Key Constants:
- **Models**: `Model_A`, `Model_B`, `Model_C`
- **Topics**: `Topic_1` through `Topic_11`
- **Harm Types**: `Harm_1` through `Harm_4`
- **Benefit Reasons**: `Benefit_1` through `Benefit_4`
- **Harm Percentages**: `15%`, `40%`, `65%` (with `0%` for non-harmful and control items)

#### Core Functions:

- **`create_treatment_pool()`**: Creates a comprehensive pool of treatment item combinations.
- **`create_control_items(model: str, num_control_per_participant: int = 6)`**: Creates control items for a specific model.
- **`generate_participant_data(participant_id, treatment_pool, used_samples, ...)`**: Generates data for a single participant with sample uniqueness tracking. Uses stratified sampling and ensures no duplicate samples across participants.
- **`simulate_harmful_choice(row, is_ambiguous=False)`**: Simulates participant choice with realistic accuracy rates (80% for regular, 45% for ambiguous participants).
- **`simulate_pick_choice(row)`**: Simulates which option participants would pick (prefers less harmful).
- **`simulate_ratings(row, is_ambiguous=False)`**: Simulates 1-5 harm ratings with dispersed, realistic distributions.
- **`simulate_confidence(row, is_ambiguous=False)`**: Simulates confidence levels (1-5) based on participant type and task difficulty.

#### Advanced Features:

- **Sample Uniqueness**: Tracks used sample combinations across all participants to prevent duplicates.
- **Ambiguous Participants**: Simulates participants with lower accuracy and confidence.
- **Realistic Confidence**: Confidence varies based on harm difference clarity and participant type.
- **Dispersed Responses**: More realistic simulation with varied response patterns.

#### Main Functions:

- **`generate_synthetic_data(num_participants, output_path, num_ambiguous_participants=0, mode='choice', num_treatment_per_participant=24, num_control_per_participant=6)`**: Main generation function with flexible parameters.
- **`analyze_experiment_samples(all_data, output_path)`**: Performs comprehensive experiment-level analysis.
- **`print_descriptive_stats(data_path, min_confidence=1)`**: Detailed statistical analysis with confidence filtering option.
- **`test_generate_synthetic_data()`**: Comprehensive test suite.

### `analysis.py`

This module contains functions to analyze the generated data with automatic mode detection.

#### Functions:

- **`load_data(file_path: str) -> pd.DataFrame`**: Loads study data from CSV files with error handling.
- **`analyze_treatment_items(df: pd.DataFrame) -> Dict[str, Any]`**: Analyzes treatment items using binomial tests for choice accuracy.
- **`analyze_control_items(df: pd.DataFrame) -> Dict[str, Any]`**: Analyzes control items to verify attention checks.
- **`analyze_treatment_items_rating(df: pd.DataFrame) -> Dict[str, Any]`**: Analyzes treatment items using Wilcoxon signed-rank tests for ratings.
- **`analyze_control_items_rating(df: pd.DataFrame) -> Dict[str, Any]`**: Analyzes control item ratings.
- **`analyze_data(data_path: str, output_path: str, min_confidence: int) -> None`**: Main analysis function with automatic mode detection and confidence filtering.
- **`test_analysis_functions()`**: Test suite for analysis functions.

### `main.py`

Main entry point that orchestrates the complete experimental workflow with configurable parameters.

#### Functions:

- **`main(args)`**: Executes the complete pipeline with argument parsing for:
  - Number of participants
  - Number of ambiguous participants
  - Response mode ('choice' or 'rating')
  - Treatment and control items per participant
  - Data generation and analysis
  - Minimum confidence level filtering

## General Workflows

### 1. Benchmark Parsing Workflow
1. **Input**: 6 JSON benchmark files in `benchmark/` directory
2. **Processing**: Parse nested JSON structure and extract scenarios with options
3. **Output**: Unified JSON file with standardized format
4. **Statistics**: Generate comprehensive summary statistics

### 2. **NEW** - Interactive Benchmark Exploration Workflow
1. **Setup**: Ensure parsed benchmark data exists (`parsed_benchmark_data.json`)
2. **Launch**: Run `python run_streamlit.py` or `streamlit run explore_benchmark.py`
3. **Explore**: Use filters, search, and navigation to explore samples
4. **Export**: Download filtered data or individual samples as needed

#### Key Exploration Features:
- **Real-time Filtering**: Multiple filter types with immediate results
- **Professional Display**: Formatted markdown with proper line breaks
- **Statistical Insights**: Live charts and summary statistics
- **Data Export**: Flexible export options for research and analysis

### 3. Experimental Design Workflow

#### Treatment Items
- Each participant receives items comparing harmful vs non-harmful content
- One option has harm (15%, 40%, or 65%), the other has 0% harm
- Stratified sampling ensures balanced representation across models, topics, harm types, and benefits
- Sample uniqueness prevents participants from seeing identical combinations

#### Control Items
- Both options have 0% harm (attention checks)
- Distributed across all models
- Used to validate participant attention and response quality

#### Participant Types
- **Regular Participants**: Higher accuracy (80%) and confidence
- **Ambiguous Participants**: Lower accuracy (45%) and confidence, more dispersed responses

### 4. Data Analysis Features

#### Experiment-Level Analysis
- **Model Distribution Tables**: Shows treatment/control distribution across models
- **Topic/Harm/Benefit Distribution**: Comprehensive coverage analysis
- **Harm Level Distribution**: Distribution of 0%, 15%, 40%, 65% across options
- **Sample Uniqueness Analysis**: Tracks duplicate rates and most common combinations
- **Participant Coverage**: Shows how many participants saw each condition

#### Statistical Analysis
- **Choice Mode**: Binomial tests for accuracy vs chance
- **Rating Mode**: Wilcoxon signed-rank tests for harm differentiation
- **Confidence Filtering**: Option to exclude low-confidence responses
- **Parallel Analysis**: Treatment vs control comparisons
- **Response Analysis**: Harmful vs non-harmful option comparisons

## Usage Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Parse Benchmark Files (Required for Explorer)
```bash
python parse_benchmark.py
```

### 3. **NEW** - Launch Interactive Benchmark Explorer
```bash
# Option 1: Using helper script (recommended)
python run_streamlit.py

# Option 2: Direct streamlit command
streamlit run explore_benchmark.py
```

**Explorer Features:**
- Filter by any combination of ID, model, topic, harm type, benefit, or sample type
- Search across all text content with real-time results
- Professional formatted display with markdown rendering
- Export capabilities for research and analysis
- Statistical summaries with interactive charts

### 4. Run Experiments
```bash
python main.py --num_participants 10 --mode choice --num_ambiguous_participants 2
```

#### Available Arguments:
- `--num_participants`: Number of participants (default: 4)
- `--num_ambiguous_participants`: Number of ambiguous participants (default: 2)
- `--mode`: Response mode ('choice' or 'rating', default: 'choice')
- `--num_treatment_per_participant`: Treatment items per participant (default: 24)
- `--num_control_per_participant`: Control items per participant (default: 6)
- `--min_confidence`: Minimum confidence level for analysis (default: 2)

### 5. Data Processing Pipeline
1. **Data Generation**: Creates datasets in `data/` directory with sample uniqueness tracking
2. **Experiment Analysis**: Generates comprehensive experiment-level analysis files
3. **Statistical Analysis**: Performs mode-appropriate statistical tests
4. **Results**: Saves detailed analysis results in `results/` directory

## Output Files

### Benchmark Parsing
- `parsed_benchmark_data.json`: Unified JSON format with all parsed scenarios

### Data Files
- `{experiment_name}_synthetic_data.csv`: Raw experimental data
- `{experiment_name}_synthetic_data_analysis.txt`: Experiment-level analysis tables

### Analysis Files
- `{experiment_name}_analysis_results.csv`: Statistical analysis results
- Detailed descriptive statistics with confidence filtering options

## Key Improvements

1. **Benchmark Parsing**: Modular parsing of 6 JSON files into unified format
2. **Sample Uniqueness**: Prevents duplicate combinations across participants
3. **Realistic Simulation**: More dispersed, human-like response patterns
4. **Confidence Modeling**: Context-aware confidence simulation
5. **Experiment-Level Analysis**: Comprehensive overview instead of per-participant logs
6. **Flexible Parameters**: Configurable experiment size and composition
7. **Advanced Statistics**: Multiple analysis modes with filtering options
8. **Professional Output**: Well-formatted tables and comprehensive reporting
9. **Error Handling**: Comprehensive error handling throughout all modules
10. **Logging**: Structured logging for debugging and monitoring
11. **NEW - Interactive Explorer**: Professional Streamlit application for benchmark exploration
12. **NEW - Export Capabilities**: Flexible data export options for research workflows 