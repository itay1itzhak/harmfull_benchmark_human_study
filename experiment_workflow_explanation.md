# Human Study Experiment Workflow: Harm Perception in AI-Generated Content

## Overview

This experiment simulates a human study that evaluates how people perceive and judge harmful content in AI-generated text. The study compares participants' ability to distinguish between harmful and non-harmful AI outputs across different scenarios and models.

## Research Question

**How accurately can humans identify harmful content when comparing AI-generated text options, and how does this vary across different participant types, content categories, and harm levels?**

## Experimental Design

### Core Assumptions

1. **Harm is measurable**: Content harmfulness can be quantified on a percentage scale (0%, 15%, 40%, 65%)
2. **Binary comparison**: Participants can meaningfully compare two options where one is harmful and one is not
3. **Individual differences**: Some participants are naturally more uncertain/ambiguous in their judgments
4. **Context matters**: Harm perception varies across different topics, models, and content types

### Experimental Structure

#### **Participants**
- **Regular participants** (majority): More confident and accurate in harm detection (~80% accuracy)
- **Ambiguous participants** (subset): Less confident and less accurate (~45% accuracy)
- Each participant evaluates 30 items total

#### **Content Items**
- **Treatment items** (24 per participant): One option harmful, one non-harmful
  - Harmful option: 15%, 40%, or 65% harm level
  - Non-harmful option: 0% harm
  - Randomly assigned which is Option A vs B
- **Control items** (6 per participant): Both options non-harmful (0% harm)
  - Used as attention checks to validate participant engagement

#### **Content Categories**
- **3 AI Models**: Model_A, Model_B, Model_C
- **11 Topics**: Topic_1 through Topic_11 (covering diverse content domains)
- **4 Harm Types**: Different categories of potential harm
- **4 Benefit Reasons**: Different rationales for content generation

## Sampling Strategy

### **Stratified Sampling**
Each participant receives a balanced sample ensuring:
- Equal representation across all 3 AI models
- Diverse coverage of topics (sampled without replacement)
- Balanced distribution of harm types and benefit reasons
- Equal distribution of harm levels (15%, 40%, 65%)

### **Sample Uniqueness**
- **No duplicate combinations**: Each participant sees unique (model, topic, harm_type, benefit_reason) combinations
- **Across-participant tracking**: Once a combination is assigned, it's not reused for other participants
- **Exhaustion handling**: If unique combinations run out, system allows necessary duplicates

## Response Modes

### **Choice Mode**
Participants answer two questions per item:
1. **"Which option is more harmful?"** → Options: A, B, or "No difference/Unsure"
2. **"Which option should the AI pick?"** → Options: A, B, or "Neither"

### **Rating Mode** 
Participants rate each option individually:
- **Harm rating scale**: 1 (not harmful) to 5 (very harmful)
- **Both options rated**: Independent ratings for Option A and Option B

### **Confidence Rating** (Both modes)
- **Scale**: 1 (very uncertain) to 5 (very confident)
- **Context-sensitive**: Varies based on harm difference clarity and participant type

## Simulation Parameters

### **Response Accuracy Simulation**
- **Regular participants**: 80% accuracy in identifying more harmful option
- **Ambiguous participants**: 45% accuracy (closer to random chance)
- **Random noise**: 10-20% chance of completely random responses

### **Confidence Simulation**
- **Task difficulty effect**: Higher confidence when harm difference is clear (≥40%)
- **Participant type effect**: Regular participants generally more confident
- **Realistic distributions**: Confidence correlates with accuracy and task clarity

### **Rating Distributions**
- **0% harm**: Mostly ratings 1-2, some variation for ambiguous participants
- **15% harm**: Mostly ratings 2-3, with some spread
- **40% harm**: Mostly ratings 3-4, moderate spread
- **65% harm**: Mostly ratings 4-5, high harm recognition

## Analysis Workflow

### **Data Quality Checks**
1. **Sample balance validation**: Verify equal distribution across models, topics, harm levels
2. **Uniqueness verification**: Check duplicate rates and most common combinations
3. **Attention check analysis**: Examine control item responses for participant validity

### **Primary Analysis**

#### **Choice Mode Analysis**
- **Binomial tests**: Compare accuracy against chance (50%) for each condition
- **Model comparison**: Test if harm detection varies across AI models
- **Harm level analysis**: Examine accuracy across different harm percentages

#### **Rating Mode Analysis**
- **Wilcoxon signed-rank tests**: Compare ratings between harmful vs non-harmful options
- **Effect size calculation**: Measure magnitude of harm perception differences
- **Distribution analysis**: Examine rating patterns across harm levels

### **Secondary Analysis**
- **Confidence-accuracy correlation**: Relationship between confidence and correct responses
- **Participant type comparison**: Regular vs ambiguous participant performance
- **Content category effects**: Variation across topics, harm types, and benefits

### **Quality Control Options**
- **Confidence filtering**: Option to exclude low-confidence responses (< threshold)
- **Attention check filtering**: Remove participants who fail control items
- **Response time analysis**: Identify participants with suspicious response patterns

## Output and Reporting

### **Experiment-Level Reports**
- **Distribution tables**: Comprehensive breakdown of sample composition
- **Balance verification**: Confirm experimental design integrity
- **Uniqueness analysis**: Sample overlap and duplicate tracking

### **Statistical Results**
- **Hypothesis testing**: Significance tests for harm detection accuracy
- **Effect sizes**: Practical significance of observed differences
- **Confidence intervals**: Uncertainty bounds around estimates

### **Descriptive Statistics**
- **Response patterns**: Choice distributions, rating means, confidence levels
- **Participant comparisons**: Regular vs ambiguous performance
- **Content analysis**: Performance across different categories

## Validity Considerations

### **Internal Validity**
- **Randomization**: Random assignment of harm levels to options A/B
- **Counterbalancing**: Equal representation across all experimental factors
- **Control conditions**: Non-harmful items to detect response biases

### **External Validity**
- **Realistic simulation**: Response patterns based on plausible human behavior
- **Diverse content**: Multiple topics, models, and harm types
- **Individual differences**: Both confident and uncertain participant types

### **Statistical Power**
- **Sample size calculation**: Sufficient participants for detecting meaningful effects
- **Effect size expectations**: Based on realistic assumptions about human performance
- **Multiple comparisons**: Appropriate corrections for multiple testing

## Practical Implementation

### **Running the Experiment**
```bash
python main.py --num_participants 50 --num_ambiguous_participants 10 --mode choice
```

### **Key Parameters**
- **Participants**: 10-100+ (depending on desired statistical power)
- **Ambiguous ratio**: Typically 10-30% of total participants
- **Items per participant**: 24 treatment + 6 control (standard)
- **Response mode**: Choice or rating (determines analysis approach)

This workflow provides a comprehensive framework for studying human harm perception in AI-generated content, with built-in quality controls and flexible analysis options to support robust scientific conclusions. 