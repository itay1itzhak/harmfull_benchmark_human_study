"""
Demonstration script for the Qualtrics participant forms generator.

This script shows how to use the system and generates example files.
"""

import os
import json
from create_particapants_forms import main, load_benchmark

def demo_with_real_data():
    """Demonstrate with real benchmark data if available."""
    benchmark_file = "benchmark/parsed_benchmark_data.json"
    
    if os.path.exists(benchmark_file):
        print("=== DEMO WITH REAL BENCHMARK DATA ===")
        print(f"Using benchmark file: {benchmark_file}")
        
        # Load and show stats
        data = load_benchmark(benchmark_file)
        print(f"Total samples available: {len(data)}")
        
        # Show categories
        models = set(item['metadata']['model_type'] for item in data)
        topics = set(item['metadata']['topic'] for item in data)
        print(f"Models: {sorted(models)}")
        print(f"Topics: {len(topics)} available")
        
        # Generate experiment
        main(
            benchmark_file=benchmark_file,
            output_csv="demo/experiment_plan.csv",
            output_dir="demo/qualtrics_files",
            random_seed=42,
            num_participants=4,
            treatment_per_participant=12,
            control_per_participant=3
        )
        
        print("\nDemo files generated in demo/ directory!")
        print("Check demo/experiment_plan.csv for the full experiment plan")
        print("Check demo/qualtrics_files/ for participant-specific Qualtrics files")
        
    else:
        print(f"Benchmark file not found: {benchmark_file}")
        print("Please run 'python parse_benchmark.py' first to generate the benchmark data.")


def create_minimal_demo():
    """Create a minimal demo with synthetic data."""
    print("=== MINIMAL DEMO WITH SYNTHETIC DATA ===")
    
    # Create minimal test data
    demo_data = []
    topics = ["Agriculture", "Healthcare", "Transportation", "Education", "Finance"]
    models = ["ChatGPT", "Claude", "Gemini"]
    harm_types = ["Loss of job", "Identity theft", "Temporary illness", "Bias"]
    benefits = ["Encourage usage", "Avoid shutdown", "Increase accuracy", "Reduce costs"]
    
    sample_id = 0
    for model in models:
        for topic in topics:
            for harm_type in harm_types:
                for benefit in benefits:
                    for sample_type in ["Treatment", "Control"]:
                        demo_data.append({
                            "id": str(sample_id),
                            "scenario": f"Scenario about {topic.lower()} with {{benefit_percentage}}% improvement in {benefit.lower()}.",
                            "option_a": f"**Option A:** Conservative approach for {topic.lower()}.",
                            "option_b": f"**Option B:** Aggressive approach for {topic.lower()}.",
                            "metadata": {
                                "model_type": model,
                                "sample_type": sample_type,
                                "topic": topic,
                                "harm_type": harm_type,
                                "benefit": benefit
                            }
                        })
                        sample_id += 1
    
    # Save demo benchmark
    os.makedirs("demo", exist_ok=True)
    with open("demo/demo_benchmark.json", 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    # Generate experiment
    main(
        benchmark_file="demo/demo_benchmark.json",
        output_csv="demo/demo_experiment_plan.csv",
        output_dir="demo/demo_qualtrics_files",
        random_seed=123,
        num_participants=2,
        treatment_per_participant=6,
        control_per_participant=2
    )
    
    print("\nMinimal demo files generated in demo/ directory!")


if __name__ == "__main__":
    print("QUALTRICS PARTICIPANT FORMS GENERATOR - DEMO")
    print("=" * 50)
    
    # Try real data first, fall back to demo data
    if os.path.exists("benchmark/parsed_benchmark_data.json"):
        demo_with_real_data()
    else:
        print("Real benchmark data not found, creating minimal demo...")
        create_minimal_demo()
    
    print("\n" + "=" * 50)
    print("Demo complete! Check the generated files.") 