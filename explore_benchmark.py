"""
Streamlit app for interactive exploration of benchmark examples.

This application provides a comprehensive interface to explore benchmark data
from AI models with advanced filtering, search, and visualization capabilities.

Features:
- Filter by id, model_type, topic, harm_type, and benefit
- Text search across scenarios and options
- Formatted markdown display of content
- Raw JSON inspection
- Export capabilities
- Statistical summaries
"""

import streamlit as st
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

@st.cache_data
def load_benchmark_data() -> List[Dict[str, Any]]:
    """
    Load benchmark data from the parsed JSON file.
    Auto-generates the parsed data if it doesn't exist.
    
    Returns:
        List[Dict[str, Any]]: List of benchmark samples
        
    Raises:
        FileNotFoundError: If benchmark data file is not found and cannot be generated
        json.JSONDecodeError: If JSON file is invalid
    """
    data_file = Path("benchmark/parsed_benchmark_data.json")
    
    # If data file doesn't exist, try to generate it
    if not data_file.exists():
        st.info("🔄 Parsed benchmark data not found. Attempting to generate it...")
        
        try:
            # Import the parser (only when needed)
            import parse_benchmark
            
            # Check if raw benchmark files exist
            benchmark_dir = Path("benchmark")
            if not benchmark_dir.exists():
                st.error("❌ Benchmark directory not found. Please ensure the benchmark files are included in your repository.")
                st.stop()
            
            # Try to parse the benchmark files
            with st.spinner("🔄 Parsing benchmark files... This may take a moment."):
                parse_benchmark.parse_all_benchmark_files()
            
            st.success("✅ Successfully generated parsed benchmark data!")
            
        except ImportError:
            st.error("❌ parse_benchmark.py module not found. Please ensure it's included in your repository.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to generate benchmark data: {e}")
            st.error("💡 Please run `python parse_benchmark.py` locally and include the generated file in your repository.")
            st.stop()
    
    # Load the data file
    if not data_file.exists():
        st.error(f"❌ Benchmark data file not found: {data_file}")
        st.info("Please run `python parse_benchmark.py` first to generate the parsed data.")
        st.stop()
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Loaded {len(data)} benchmark samples")
        return data
    
    except json.JSONDecodeError as e:
        st.error(f"Error loading benchmark data: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        st.stop()


def create_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert benchmark data to a pandas DataFrame for easier manipulation.
    
    Args:
        data (List[Dict[str, Any]]): Raw benchmark data
        
    Returns:
        pd.DataFrame: Structured DataFrame with flattened metadata
    """
    records = []
    
    for item in data:
        record = {
            'id': int(item['id']),
            'scenario': item['scenario'],
            'option_a': item['option_a'],
            'option_b': item['option_b'],
            'model_type': item['metadata']['model_type'],
            'sample_type': item['metadata']['sample_type'],
            'topic': item['metadata']['topic'],
            'harm_type': item['metadata']['harm_type'],
            'benefit': item['metadata']['benefit']
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    return df


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply various filters to the DataFrame.
    
    Args:
        df (pd.DataFrame): Original DataFrame
        filters (Dict[str, Any]): Dictionary of filter criteria
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # ID filter (exact match)
    if filters.get('id_filter') is not None:
        filtered_df = filtered_df[filtered_df['id'] == filters['id_filter']]
    
    # Model type filter
    if filters.get('model_types'):
        filtered_df = filtered_df[filtered_df['model_type'].isin(filters['model_types'])]
    
    # Sample type filter
    if filters.get('sample_types'):
        filtered_df = filtered_df[filtered_df['sample_type'].isin(filters['sample_types'])]
    
    # Topic filter
    if filters.get('topics'):
        filtered_df = filtered_df[filtered_df['topic'].isin(filters['topics'])]
    
    # Harm type filter
    if filters.get('harm_types'):
        filtered_df = filtered_df[filtered_df['harm_type'].isin(filters['harm_types'])]
    
    # Benefit filter
    if filters.get('benefits'):
        filtered_df = filtered_df[filtered_df['benefit'].isin(filters['benefits'])]
    
    # Text search
    if filters.get('search_text'):
        search_text = filters['search_text'].lower()
        mask = (
            filtered_df['scenario'].str.lower().str.contains(search_text, na=False) |
            filtered_df['option_a'].str.lower().str.contains(search_text, na=False) |
            filtered_df['option_b'].str.lower().str.contains(search_text, na=False)
        )
        filtered_df = filtered_df[mask]
    
    return filtered_df


def format_text_for_display(text: str) -> str:
    """
    Format text for better display in Streamlit.
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Formatted text with proper line breaks
    """
    # Replace multiple spaces with single spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Add line breaks after sentences for better readability
    text = re.sub(r'\.(\s+)', '.\n\n', text)
    
    # Format bold text
    text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text for table display.
    
    Args:
        text (str): Original text
        max_length (int): Maximum length before truncation
        
    Returns:
        str: Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def display_checkbox_filters(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Display checkbox filters for all categorical columns.
    
    Args:
        df (pd.DataFrame): DataFrame to create filters for
        
    Returns:
        Dict[str, List[str]]: Dictionary of selected values for each filter
    """
    selected_filters = {}
    
    # Model type filter
    st.sidebar.subheader("Model Type")
    available_models = sorted(df['model_type'].unique())
    selected_models = []
    
    # Select all/none buttons for models
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All", key="models_all"):
            st.session_state.models_select_all = True
    with col2:
        if st.button("Select None", key="models_none"):
            st.session_state.models_select_all = False
    
    # Initialize session state if not exists
    if 'models_select_all' not in st.session_state:
        st.session_state.models_select_all = True
    
    for model in available_models:
        default_value = st.session_state.models_select_all
        if st.sidebar.checkbox(model, value=default_value, key=f"model_{model}"):
            selected_models.append(model)
    
    selected_filters['model_types'] = selected_models
    
    # Sample type filter
    st.sidebar.subheader("Sample Type")
    available_types = sorted(df['sample_type'].unique())
    selected_types = []
    
    # Select all/none buttons for sample types
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All", key="types_all"):
            st.session_state.types_select_all = True
    with col2:
        if st.button("Select None", key="types_none"):
            st.session_state.types_select_all = False
    
    if 'types_select_all' not in st.session_state:
        st.session_state.types_select_all = True
    
    for sample_type in available_types:
        default_value = st.session_state.types_select_all
        if st.sidebar.checkbox(sample_type, value=default_value, key=f"type_{sample_type}"):
            selected_types.append(sample_type)
    
    selected_filters['sample_types'] = selected_types
    
    # Topic filter
    st.sidebar.subheader("Topic")
    available_topics = sorted(df['topic'].unique())
    selected_topics = []
    
    # Select all/none buttons for topics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All", key="topics_all"):
            st.session_state.topics_select_all = True
    with col2:
        if st.button("Select None", key="topics_none"):
            st.session_state.topics_select_all = False
    
    if 'topics_select_all' not in st.session_state:
        st.session_state.topics_select_all = True
    
    for topic in available_topics:
        default_value = st.session_state.topics_select_all
        if st.sidebar.checkbox(topic, value=default_value, key=f"topic_{topic}"):
            selected_topics.append(topic)
    
    selected_filters['topics'] = selected_topics
    
    # Harm type filter
    st.sidebar.subheader("Harm Type")
    available_harms = sorted(df['harm_type'].unique())
    selected_harms = []
    
    # Select all/none buttons for harm types
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All", key="harms_all"):
            st.session_state.harms_select_all = True
    with col2:
        if st.button("Select None", key="harms_none"):
            st.session_state.harms_select_all = False
    
    if 'harms_select_all' not in st.session_state:
        st.session_state.harms_select_all = True
    
    for harm in available_harms:
        default_value = st.session_state.harms_select_all
        if st.sidebar.checkbox(harm, value=default_value, key=f"harm_{harm}"):
            selected_harms.append(harm)
    
    selected_filters['harm_types'] = selected_harms
    
    # Benefit filter
    st.sidebar.subheader("Benefit")
    available_benefits = sorted(df['benefit'].unique())
    selected_benefits = []
    
    # Select all/none buttons for benefits
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All", key="benefits_all"):
            st.session_state.benefits_select_all = True
    with col2:
        if st.button("Select None", key="benefits_none"):
            st.session_state.benefits_select_all = False
    
    if 'benefits_select_all' not in st.session_state:
        st.session_state.benefits_select_all = True
    
    for benefit in available_benefits:
        default_value = st.session_state.benefits_select_all
        if st.sidebar.checkbox(benefit, value=default_value, key=f"benefit_{benefit}"):
            selected_benefits.append(benefit)
    
    selected_filters['benefits'] = selected_benefits
    
    return selected_filters


def display_sample_table(df: pd.DataFrame, data: List[Dict[str, Any]]):
    """
    Display samples in both table format and extended card view.
    
    Args:
        df (pd.DataFrame): Filtered DataFrame
        data (List[Dict[str, Any]]): Original raw data for full text lookup
    """
    if len(df) == 0:
        st.warning("No samples match the current filters.")
        return
    
    # Table view section
    st.subheader("📊 Samples Table")
    
    # Table display options
    col1, col2 = st.columns(2)
    with col1:
        enable_horizontal_scroll = st.checkbox(
            "Enable horizontal scrolling", 
            value=True,
            help="Allow horizontal scrolling for better column visibility"
        )
    with col2:
        show_full_text = st.checkbox(
            "Show full text (no truncation)", 
            value=False,
            help="Display full text instead of truncated previews"
        )
    
    # Create display dataframe
    display_df = df.copy()
    
    if not show_full_text:
        # Truncated version
        display_df['scenario_preview'] = display_df['scenario'].apply(lambda x: truncate_text(x, 80))
        display_df['option_a_preview'] = display_df['option_a'].apply(lambda x: truncate_text(x, 60))
        display_df['option_b_preview'] = display_df['option_b'].apply(lambda x: truncate_text(x, 60))
        
        table_columns = [
            'id', 'model_type', 'sample_type', 'topic', 'harm_type', 'benefit',
            'scenario_preview', 'option_a_preview', 'option_b_preview'
        ]
        
        column_config = {
            'id': st.column_config.NumberColumn('ID'),
            'model_type': st.column_config.TextColumn('Model'),
            'sample_type': st.column_config.TextColumn('Type'),
            'topic': st.column_config.TextColumn('Topic'),
            'harm_type': st.column_config.TextColumn('Harm Type'),
            'benefit': st.column_config.TextColumn('Benefit'),
            'scenario_preview': st.column_config.TextColumn('Scenario Preview'),
            'option_a_preview': st.column_config.TextColumn('Option A Preview'),
            'option_b_preview': st.column_config.TextColumn('Option B Preview'),
        }
    else:
        # Full text version
        table_columns = [
            'id', 'model_type', 'sample_type', 'topic', 'harm_type', 'benefit',
            'scenario', 'option_a', 'option_b'
        ]
        
        column_config = {
            'id': st.column_config.NumberColumn('ID'),
            'model_type': st.column_config.TextColumn('Model'),
            'sample_type': st.column_config.TextColumn('Type'),
            'topic': st.column_config.TextColumn('Topic'),
            'harm_type': st.column_config.TextColumn('Harm Type'),
            'benefit': st.column_config.TextColumn('Benefit'),
            'scenario': st.column_config.TextColumn('Scenario'),
            'option_a': st.column_config.TextColumn('Option A'),
            'option_b': st.column_config.TextColumn('Option B'),
        }
    
    # Display the table
    st.dataframe(
        display_df[table_columns],
        column_config=column_config,
        use_container_width=not enable_horizontal_scroll,
        hide_index=True,
        height=400 if len(display_df) > 10 else None
    )
    
    # Value counts section
    st.markdown("---")
    st.subheader("📊 Value Counts (Filtered Data)")
    
    # Create tabs for different categorical columns
    count_tabs = st.tabs(["🤖 Models", "📋 Sample Types", "🏷️ Topics", "⚠️ Harm Types", "🎯 Benefits"])
    
    with count_tabs[0]:
        st.markdown("**Model Type Distribution**")
        model_counts = df['model_type'].value_counts().sort_index()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(model_counts)
        with col2:
            st.bar_chart(model_counts)
    
    with count_tabs[1]:
        st.markdown("**Sample Type Distribution**")
        type_counts = df['sample_type'].value_counts().sort_index()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(type_counts)
        with col2:
            st.bar_chart(type_counts)
    
    with count_tabs[2]:
        st.markdown("**Topic Distribution**")
        topic_counts = df['topic'].value_counts().sort_index()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(topic_counts)
        with col2:
            if len(topic_counts) <= 20:  # Only show chart if not too many topics
                st.bar_chart(topic_counts)
            else:
                st.info("Too many topics to display chart clearly. Showing table only.")
    
    with count_tabs[3]:
        st.markdown("**Harm Type Distribution**")
        harm_counts = df['harm_type'].value_counts().sort_index()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(harm_counts)
        with col2:
            st.bar_chart(harm_counts)
    
    with count_tabs[4]:
        st.markdown("**Benefit Distribution**")
        benefit_counts = df['benefit'].value_counts().sort_index()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(benefit_counts)
        with col2:
            if len(benefit_counts) <= 15:  # Only show chart if not too many benefits
                st.bar_chart(benefit_counts)
            else:
                st.info("Too many benefits to display chart clearly. Showing table only.")
    
    # Extended card view section
    st.markdown("---")
    st.subheader("📄 Extended Card View")
    
    # Pagination for card view
    samples_per_page = st.selectbox("Samples per page", [5, 10, 20, 50], index=1)
    total_pages = (len(df) - 1) // samples_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1))
        start_idx = (page - 1) * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(df))
        page_df = df.iloc[start_idx:end_idx]
    else:
        page_df = df
    
    # Display samples in cards
    for idx, (_, sample) in enumerate(page_df.iterrows()):
        with st.expander(f"Sample {sample['id']} - {sample['topic']} ({sample['model_type']})", expanded=(idx == 0)):
            display_sample_card(sample)


def display_sample_card(sample_data: pd.Series):
    """
    Display a single sample in card format (for extended view).
    
    Args:
        sample_data (pd.Series): Single sample data
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"Sample ID: {sample_data['id']}")
    
    with col2:
        st.markdown(f"**Model:** {sample_data['model_type']}")
        st.markdown(f"**Type:** {sample_data['sample_type']}")
    
    # Scenario
    st.markdown("### 📋 Scenario")
    scenario_text = format_text_for_display(sample_data['scenario'])
    st.markdown(scenario_text)
    
    # Options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🅰️ Option A")
        option_a_text = format_text_for_display(sample_data['option_a'])
        st.markdown(option_a_text)
    
    with col2:
        st.markdown("### 🅱️ Option B")
        option_b_text = format_text_for_display(sample_data['option_b'])
        st.markdown(option_b_text)
    
    # Metadata
    st.markdown("### 📊 Metadata")
    metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
    
    with metadata_col1:
        st.markdown(f"**Topic:** {sample_data['topic']}")
        st.markdown(f"**Harm Type:** {sample_data['harm_type']}")
    
    with metadata_col2:
        st.markdown(f"**Benefit:** {sample_data['benefit']}")
    
    with metadata_col3:
        # Add export button for this sample
        sample_dict = sample_data.to_dict()
        st.download_button(
            label="📄 Export Sample",
            data=json.dumps(sample_dict, indent=2),
            file_name=f"sample_{sample_data['id']}.json",
            mime="application/json",
            key=f"export_card_{sample_data['id']}"
        )


def display_summary_stats(df: pd.DataFrame):
    """
    Display summary statistics for the current filtered data.
    
    Args:
        df (pd.DataFrame): Filtered DataFrame
    """
    st.markdown("### 📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    
    with col2:
        unique_topics = df['topic'].nunique()
        st.metric("Unique Topics", unique_topics)
    
    with col3:
        unique_models = df['model_type'].nunique()
        st.metric("Models", unique_models)
    
    with col4:
        treatment_count = len(df[df['sample_type'] == 'Treatment'])
        st.metric("Treatment Samples", treatment_count)
    
    # Distribution charts
    if len(df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Distribution**")
            model_counts = df['model_type'].value_counts()
            st.bar_chart(model_counts)
        
        with col2:
            st.markdown("**Sample Type Distribution**")
            type_counts = df['sample_type'].value_counts()
            st.bar_chart(type_counts)


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Benchmark Explorer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🔍 Benchmark Explorer")
    st.markdown("**Interactive exploration of AI model benchmark examples**")
    st.markdown("---")
    
    # Brief summary for first-time users
    st.markdown("""
    **Getting Started**
    - **1) Filter Samples:** Use the filters in the left sidebar to narrow down benchmark samples.
    - **2) Explore Samples:** View the Extended Card View below to explore the samples as seen by models.
    - **3) Search Samples:** Search by keywords, select specific AI models or topics, and view detailed scenarios.
    """)
    
    # Quick Usage Guide
    with st.expander("📖 **Quick Usage Guide - Click to expand**", expanded=False):
        st.markdown("""
        ### 🚀 How to Use This Explorer
        
        This tool helps you explore AI model benchmark data with powerful filtering and viewing capabilities.
        
        #### 🔧 **Filtering (Left Sidebar)**
        - **ID Filter**: Enter a specific sample ID number for exact match
        - **Text Search**: Search across scenario descriptions and option text
        - **Checkbox Filters**: Use checkboxes to select specific models, topics, harm types, etc.
        - **Bulk Selection**: Use "Select All" / "Select None" buttons for quick filtering
        
        #### 📋 **Main Interface (Three Sections)**
        1. **📊 Table View**: Quick overview of all samples with key metadata
        2. **📊 Value Counts**: Statistics and charts showing data distribution
        3. **📄 Extended Card View**: Detailed, formatted view of each sample
        
        #### 🔍 **Search Examples**
        Try these searches in the **Text Search** box:
        """)
        
        # Search examples in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Content-Based Searches:**
            - `agriculture` - Find farming-related scenarios
            - `hospital` - Healthcare scenarios
            - `job` - Employment-related content
            - `privacy` - Data privacy scenarios
            - `efficiency` - Performance optimization scenarios
            """)
        
        with col2:
            st.markdown("""
            **⚠️ Harm-Related Searches:**
            - `illness` - Scenarios involving health impacts
            - `fraud` - Financial fraud scenarios  
            - `eviction` - Housing displacement scenarios
            - `unemployment` - Job loss scenarios
            - `depression` - Mental health impacts
            """)
        
        st.markdown("""
        #### 🎛️ **Filter Combinations**
        Combine multiple filters for targeted exploration:
        
        **Example Workflows:**
        - **Compare Models**: Select specific models to see how different AIs handle similar scenarios
        - **Focus on Topic**: Choose "Healthcare" topic + search "efficiency" to find healthcare optimization scenarios
        - **Analyze Harm Types**: Select specific harm types to understand impact patterns
        - **Treatment vs Control**: Use Sample Type filter to compare harmful vs neutral scenarios
        
        #### 📊 **Using the Interface**
        1. **Start with filters** in the sidebar to narrow down your data
        2. **Scan the table** to get an overview of matching samples
        3. **Check the statistics** to understand data distribution
        4. **Expand cards** in the Extended View to read full scenarios
        5. **Export samples** you find interesting for further analysis
        
        #### 💡 **Pro Tips**
        - Use **horizontal scrolling** for better table visibility
        - Try **full text mode** in the table to see complete content
        - **Export filtered data** from the Export tab for external analysis
        - Use **multiple filters** simultaneously for precise targeting
        - Check **value counts** to understand your filtered dataset
        """)
    
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading benchmark data..."):
        data = load_benchmark_data()
        df = create_dataframe(data)
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    # ID filter
    st.sidebar.subheader("ID Filter")
    id_filter = st.sidebar.number_input(
        "Exact ID", 
        min_value=0, 
        max_value=int(df['id'].max()), 
        value=None,
        help="Enter a specific sample ID for exact match"
    )
    
    # Text search
    st.sidebar.subheader("Text Search")
    search_text = st.sidebar.text_input(
        "Search in scenarios and options",
        placeholder="Enter keywords...",
        help="Search across scenario text and both options"
    )
    
    # Checkbox filters
    checkbox_filters = display_checkbox_filters(df)
    
    # Apply filters
    filters = {
        'id_filter': id_filter,
        'search_text': search_text,
        **checkbox_filters
    }
    
    filtered_df = apply_filters(df, filters)
    
    # Main content area
    if len(filtered_df) == 0:
        st.warning("No samples match the current filters. Please adjust your criteria.")
        return
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Browse Samples", "🔍 Raw JSON Inspector", "📊 Statistics", "💾 Export"])
    
    with tab1:
        st.header(f"Browse Samples ({len(filtered_df)} found)")
        display_sample_table(filtered_df, data)
    
    with tab2:
        st.header("🔍 Raw JSON Inspector")
        
        # Sample selector
        available_ids = sorted(filtered_df['id'].tolist())
        selected_id = st.selectbox(
            "Select sample ID for JSON inspection",
            available_ids,
            help="Choose a sample to view its raw JSON structure"
        )
        
        if selected_id is not None:
            # Find the original sample in the raw data
            selected_sample = next((item for item in data if int(item['id']) == selected_id), None)
            
            if selected_sample:
                st.subheader(f"Raw JSON for Sample {selected_id}")
                st.json(selected_sample)
                
                # Download button for this JSON
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(selected_sample, indent=2),
                    file_name=f"sample_{selected_id}.json",
                    mime="application/json"
                )
    
    with tab3:
        st.header("📊 Statistics")
        display_summary_stats(filtered_df)
        
        # Additional detailed statistics
        if st.checkbox("Show detailed breakdowns"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Topic Distribution")
                topic_counts = filtered_df['topic'].value_counts()
                st.write(topic_counts)
            
            with col2:
                st.subheader("Harm Type Distribution")
                harm_counts = filtered_df['harm_type'].value_counts()
                st.write(harm_counts)
            
            st.subheader("Benefit Distribution")
            benefit_counts = filtered_df['benefit'].value_counts()
            st.write(benefit_counts)
            
            # Cross-tabulation
            st.subheader("Model vs Sample Type Cross-tabulation")
            crosstab = pd.crosstab(filtered_df['model_type'], filtered_df['sample_type'])
            st.write(crosstab)
    
    with tab4:
        st.header("💾 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Filtered Data")
            
            # CSV export
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📊 Download as CSV",
                data=csv_data,
                file_name="filtered_benchmark_data.csv",
                mime="text/csv"
            )
            
            # JSON export
            json_data = json.dumps(filtered_df.to_dict('records'), indent=2)
            st.download_button(
                label="📄 Download as JSON",
                data=json_data,
                file_name="filtered_benchmark_data.json",
                mime="application/json"
            )
        
        with col2:
            st.subheader("Export Summary")
            st.write(f"**Total samples to export:** {len(filtered_df)}")
            st.write(f"**Models included:** {', '.join(filtered_df['model_type'].unique())}")
            st.write(f"**Topics included:** {len(filtered_df['topic'].unique())}")
            st.write(f"**Date range:** {filtered_df['id'].min()} - {filtered_df['id'].max()}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This tool helps explore benchmark data for research purposes. Use filters to narrow down to specific subsets of interest.")


if __name__ == "__main__":
    main()
