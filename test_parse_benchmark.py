import pytest
import json
import os
import tempfile
from unittest.mock import patch, mock_open
from typing import Dict, Any, List

from parse_benchmark import (
    load_json_file,
    extract_model_and_sample_type,
    parse_scenario_text,
    create_sample_entry,
    process_single_file,
    get_benchmark_files,
    parse_all_benchmark_files,
    print_summary_stats
)


class TestLoadJsonFile:
    """Test cases for load_json_file function."""
    
    def test_load_valid_json_file(self):
        """Test loading a valid JSON file."""
        test_data = {"key": "value", "number": 123}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            result = load_json_file(temp_file)
            assert result == test_data
        finally:
            os.unlink(temp_file)
    
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Benchmark file not found"):
            load_json_file("nonexistent_file.json")
    
    def test_load_invalid_json_file(self):
        """Test loading an invalid JSON file raises JSONDecodeError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_file = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError, match="Invalid JSON format"):
                load_json_file(temp_file)
        finally:
            os.unlink(temp_file)


class TestExtractModelAndSampleType:
    """Test cases for extract_model_and_sample_type function."""
    
    def test_extract_chatgpt_treatment(self):
        """Test extracting ChatGPT treatment file information."""
        result = extract_model_and_sample_type("chatgpt_results_dict.json")
        assert result == ("Chatgpt", "Treatment")
    
    def test_extract_chatgpt_control(self):
        """Test extracting ChatGPT control file information."""
        result = extract_model_and_sample_type("chatgpt_results_dict_f.json")
        assert result == ("Chatgpt", "Control")
    
    def test_extract_claude_treatment(self):
        """Test extracting Claude treatment file information."""
        result = extract_model_and_sample_type("claude_results_dict.json")
        assert result == ("Claude", "Treatment")
    
    def test_extract_gemini_control(self):
        """Test extracting Gemini control file information."""
        result = extract_model_and_sample_type("gemini_results_dict_f.json")
        assert result == ("Gemini", "Control")
    
    def test_extract_invalid_filename(self):
        """Test extracting from invalid filename raises ValueError."""
        with pytest.raises(ValueError, match="Invalid filename format"):
            extract_model_and_sample_type("invalid_filename.json")


class TestParseScenarioText:
    """Test cases for parse_scenario_text function."""
    
    def test_parse_valid_scenario(self):
        """Test parsing a valid scenario text with options."""
        scenario_text = """This is a test scenario description.
        
**Option A:**
This is option A content.

**Option B:**
This is option B content."""
        
        scenario, option_a, option_b = parse_scenario_text(scenario_text)
        
        assert scenario.strip() == "This is a test scenario description."
        assert option_a.strip() == "This is option A content."
        assert option_b.strip() == "This is option B content."
    
    def test_parse_scenario_missing_option_a(self):
        """Test parsing scenario text missing Option A raises ValueError."""
        scenario_text = """This is a test scenario.
        
**Option B:**
This is option B content."""
        
        with pytest.raises(ValueError, match="Failed to parse scenario text"):
            parse_scenario_text(scenario_text)
    
    def test_parse_scenario_missing_option_b(self):
        """Test parsing scenario text missing Option B raises ValueError."""
        scenario_text = """This is a test scenario.
        
**Option A:**
This is option A content."""
        
        with pytest.raises(ValueError, match="Failed to parse scenario text"):
            parse_scenario_text(scenario_text)
    
    def test_parse_scenario_empty_sections(self):
        """Test parsing scenario with empty sections raises ValueError."""
        scenario_text = """
        
**Option A:**

**Option B:**
"""
        
        with pytest.raises(ValueError, match="Failed to parse scenario text"):
            parse_scenario_text(scenario_text)


class TestCreateSampleEntry:
    """Test cases for create_sample_entry function."""
    
    def test_create_valid_sample_entry(self):
        """Test creating a valid sample entry."""
        result = create_sample_entry(
            sample_id="123",
            scenario="Test scenario",
            option_a="Test option A",
            option_b="Test option B",
            model_type="ChatGPT",
            sample_type="Treatment",
            topic="Agriculture",
            harm_type="Loss of job",
            benefit="Encourage more usage"
        )
        
        expected = {
            "id": "123",
            "scenario": "Test scenario",
            "option_a": "Test option A",
            "option_b": "Test option B",
            "metadata": {
                "model_type": "ChatGPT",
                "sample_type": "Treatment",
                "topic": "Agriculture",
                "harm_type": "Loss of job",
                "benefit": "Encourage more usage"
            }
        }
        
        assert result == expected


class TestProcessSingleFile:
    """Test cases for process_single_file function."""
    
    def test_process_valid_file(self):
        """Test processing a valid benchmark file."""
        test_data = {
            "Agriculture": {
                "Loss of job": {
                    "Encourage more usage": """Test scenario description.

**Option A:**
This is option A.

**Option B:**
This is option B."""
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_results_dict.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            result = process_single_file(temp_file)
            
            assert len(result) == 1
            sample = result[0]
            assert sample["scenario"].strip() == "Test scenario description."
            assert sample["option_a"].strip() == "This is option A."
            assert sample["option_b"].strip() == "This is option B."
            assert sample["metadata"]["topic"] == "Agriculture"
            assert sample["metadata"]["harm_type"] == "Loss of job"
            assert sample["metadata"]["benefit"] == "Encourage more usage"
        finally:
            os.unlink(temp_file)


class TestGetBenchmarkFiles:
    """Test cases for get_benchmark_files function."""
    
    def test_get_files_from_existing_directory(self):
        """Test getting benchmark files from existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test benchmark files
            test_files = [
                "chatgpt_results_dict.json",
                "chatgpt_results_dict_f.json",
                "claude_results_dict.json",
                "claude_results_dict_f.json"
            ]
            
            for filename in test_files:
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    json.dump({}, f)
            
            result = get_benchmark_files(temp_dir)
            
            assert len(result) == 4
            for file_path in result:
                assert os.path.basename(file_path) in test_files
    
    def test_get_files_from_nonexistent_directory(self):
        """Test getting files from nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Benchmark directory not found"):
            get_benchmark_files("nonexistent_directory")
    
    def test_get_files_from_empty_directory(self):
        """Test getting files from empty directory raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError, match="No valid benchmark files found"):
                get_benchmark_files(temp_dir)


class TestPrintSummaryStats:
    """Test cases for print_summary_stats function."""
    
    def test_print_summary_stats(self, capsys):
        """Test printing summary statistics."""
        test_samples = [
            {
                "id": "0",
                "scenario": "Test scenario 1",
                "option_a": "Option A",
                "option_b": "Option B",
                "metadata": {
                    "model_type": "ChatGPT",
                    "sample_type": "Treatment",
                    "topic": "Agriculture",
                    "harm_type": "Loss of job",
                    "benefit": "Encourage more usage"
                }
            },
            {
                "id": "1",
                "scenario": "Test scenario 2",
                "option_a": "Option A",
                "option_b": "Option B",
                "metadata": {
                    "model_type": "Claude",
                    "sample_type": "Control",
                    "topic": "Healthcare",
                    "harm_type": "Identity theft",
                    "benefit": "Avoid being shut down"
                }
            }
        ]
        
        print_summary_stats(test_samples)
        
        captured = capsys.readouterr()
        assert "PARSING SUMMARY" in captured.out
        assert "Total samples: 2" in captured.out
        assert "ChatGPT: 1" in captured.out
        assert "Claude: 1" in captured.out
        assert "Treatment: 1" in captured.out
        assert "Control: 1" in captured.out


def test_parse_all_benchmark_files_integration():
    """Integration test for parse_all_benchmark_files function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test benchmark files
        test_data = {
            "Agriculture": {
                "Loss of job": {
                    "Encourage more usage": """Test scenario.

**Option A:**
Option A content.

**Option B:**
Option B content."""
                }
            }
        }
        
        # Create both treatment and control files
        for filename in ["chatgpt_results_dict.json", "chatgpt_results_dict_f.json"]:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                json.dump(test_data, f)
        
        output_file = os.path.join(temp_dir, "test_output.json")
        
        # Test the main parsing function
        parse_all_benchmark_files(temp_dir, output_file)
        
        # Verify output file was created
        assert os.path.exists(output_file)
        
        # Verify output content
        with open(output_file, 'r') as f:
            result = json.load(f)
        
        assert len(result) == 2  # Two files processed
        assert result[0]["id"] == "0"
        assert result[1]["id"] == "1"
        assert result[0]["metadata"]["sample_type"] in ["Treatment", "Control"]
        assert result[1]["metadata"]["sample_type"] in ["Treatment", "Control"]


if __name__ == "__main__":
    pytest.main([__file__]) 