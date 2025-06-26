# Qualtrics Participant Forms Generator

This system prepares Qualtrics-ready files for human studies on harmful AI content assessment. It handles sampling, experimental design, and file formatting for deployment on the Qualtrics platform.

## Quick Start

1. **Ensure benchmark data is available**:
   ```bash
   python parse_benchmark.py  # Generates parsed_benchmark_data.json
   ```

2. **Generate Qualtrics files**:
   ```bash
   python create_particapants_forms.py
   ```

3. **Files will be generated**:
   - `data/experiment_plan.csv` - Complete experiment metadata
   - `qualtrics_files/participant_XX_qualtrics.txt` - Individual Qualtrics files

## Experiment Design

### Overview
- **4 participants** total
- **15 items per participant** (12 treatment + 3 control)
- **Balanced sampling** across topics, harm types, and benefits
- **Unique treatment items** (no duplicates across participants)

### Treatment Items (12 per participant)
- One option contains harmful content (15%, 40%, or 65% harm level)
- Other option is zero-harm
- Randomly assigned whether harmful option is A or B
- Covers all available topics with balanced distribution

### Control Items (3 per participant)  
- Both options are zero-harm (attention checks)
- Used to validate participant attention and response quality

### Response Collection
For each item, participants provide:
1. **Harm rating** (1-5 slider): "Which option is more harmful?"
2. **Free-text explanation**: Reasoning for their assessment
3. **Realism rating** (1-5 scale): "How realistic does this scenario seem?"

## Usage Examples

### Basic Usage
```bash
# Generate with default settings
python create_particapants_forms.py
```

### Custom Configuration
```bash
# 6 participants, 10 treatment + 5 control items each
python create_particapants_forms.py --participants 6 --treatment 10 --control 5

# Use specific random seed for reproducibility
python create_particapants_forms.py --seed 123

# Custom file paths
python create_particapants_forms.py \
    --benchmark my_benchmark.json \
    --csv my_experiment_plan.csv \
    --output my_qualtrics_files/
```

### Integration with Existing Workflow
```bash
# Complete pipeline: parse -> generate -> deploy
python parse_benchmark.py
python create_particapants_forms.py --seed 42
# Upload files from qualtrics_files/ to Qualtrics
```

## Output Files

### Experiment Plan CSV (`experiment_plan.csv`)
Complete metadata for linking participant responses to experimental conditions:

| Column | Description |
|--------|-------------|
| `item_id` | Unique identifier for each item |
| `participant_id` | Participant assignment (1-4) |
| `item_type` | "treatment" or "control" |
| `harm_level` | Assigned harm percentage (0, 15, 40, 65) |
| `benefit_level` | Assigned benefit percentage (15, 40, 65) |
| `harmful_option` | Which option contains harm ("A", "B", "Neither") |
| `topic` | Content topic (e.g., "Agriculture", "Healthcare") |
| `harm_type` | Type of potential harm |
| `benefit_reason` | Reason for benefit |
| `scenario` | Full scenario text (with placeholders resolved) |
| `option_a` | Option A text |
| `option_b` | Option B text |

### Qualtrics Files (`participant_XX_qualtrics.txt`)
Ready-to-upload Qualtrics files with:
- **Block structure**: Introduction → Items → Completion
- **Question format**: Proper Qualtrics syntax with question types
- **Page breaks**: Logical flow between items
- **Response types**: Sliders, text entry, multiple choice

## Sampling Strategy

### Balance Constraints
- **Topics**: All 11 topics must be represented (2-7 occurrences each)
- **Harm types**: Distributed fairly across available types (~14-33% each)
- **Benefit reasons**: Balanced across categories (~18-31% each)
- **Models**: Approximately equal representation across ChatGPT, Claude, Gemini

### Uniqueness Guarantees
- No duplicate treatment combinations across participants
- Each (model, topic, harm_type, benefit) combination used at most once
- Control items can be reused as attention checks

### Random Assignment
- Harm levels (15%, 40%, 65%) assigned randomly to treatment items
- Benefit levels (15%, 40%, 65%) assigned randomly with placeholder replacement
- Harmful option (A vs B) assigned randomly for treatment items

## Quality Assurance

### Validation Checks
- **Sample availability**: Ensures sufficient samples for requested configuration
- **Balance verification**: Reports actual distribution achieved
- **Uniqueness tracking**: Prevents duplicate combinations
- **Text processing**: Validates placeholder replacement

### Error Handling
- **Missing files**: Clear error messages for missing benchmark data
- **Insufficient samples**: Helpful guidance for constraint relaxation
- **Invalid configurations**: Validation of parameter combinations

### Testing
Run the test suite to verify functionality:
```bash
python test_create_participants_forms.py
```

## Qualtrics Upload Instructions

1. **Log into Qualtrics** and create a new survey
2. **Import questions** from generated .txt files:
   - Survey → Tools → Import/Export → Import Survey
   - Choose "Advanced Format" and upload participant_XX_qualtrics.txt
3. **Configure settings**:
   - Set appropriate time limits
   - Configure consent forms
   - Set up data collection permissions
4. **Test thoroughly** before deployment
5. **Link responses** back to experiment plan using participant IDs

## Advanced Configuration

### Custom Sampling
Modify sampling parameters in the code:
```python
# Adjust minimum items per topic
min_per_topic = max(3, total_needed // len(topics))

# Change harm/benefit level options
HARM_LEVELS = [10, 30, 50, 70]
BENEFIT_LEVELS = [20, 40, 60]
```

### Text Customization
Modify question text in `generate_item_questions()`:
```python
# Customize harm rating question
lines.append("How much more harmful is one option compared to the other?")

# Add custom instructions
lines.append("Consider both immediate and long-term consequences.")
```

### Output Format Modifications
Adjust Qualtrics formatting in `generate_participant_qualtrics_file()`:
```python
# Add custom blocks
lines.append("[[Block:Custom Instructions]]")

# Modify question types
lines.append("[[Matrix]]")  # For matrix questions
lines.append("[[TextEntry:SingleLine]]")  # For single-line text
```

## Troubleshooting

### Common Issues

1. **"Benchmark file not found"**
   ```bash
   python parse_benchmark.py  # Generate benchmark data first
   ```

2. **"Not enough treatment samples"**
   - Reduce number of participants or items per participant
   - Check if benchmark data was parsed correctly

3. **"Topic distribution imbalanced"**
   - Normal for small samples; check experiment_plan.csv for actual distribution
   - Increase total items if strict balance is required

4. **"Qualtrics upload fails"**
   - Ensure file is saved as Tab Delimited Text (not UTF-16)
   - Check for special characters in question text
   - Verify Qualtrics syntax formatting

### Getting Help
- Check `experiment_plan.csv` for detailed metadata
- Review console output for balance statistics
- Run tests to verify system functionality
- Examine generated Qualtrics files for proper formatting

## Integration with Analysis

The generated `experiment_plan.csv` file is designed to integrate seamlessly with analysis workflows:

```python
import pandas as pd

# Load experiment plan
plan = pd.read_csv('data/experiment_plan.csv')

# Load Qualtrics responses (after data collection)
responses = pd.read_csv('qualtrics_responses.csv')

# Merge for analysis
analysis_data = responses.merge(plan, on=['participant_id', 'item_id'])

# Now you have both responses and experimental conditions
print(analysis_data[['harm_rating', 'harm_level', 'harmful_option', 'topic']].head())
```

This enables comprehensive analysis of how participant responses relate to experimental manipulations and item characteristics. 