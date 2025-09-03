#!/usr/bin/env python3
"""
Quick Start Script: Twin Earth Externalism Experiment
=====================================================

This script gets you running the Twin Earth experiment immediately.
Perfect for testing the philosophical framework and getting initial results.

Usage:
    python quick_start.py                    # Demo experiment
    python quick_start.py --full            # Full experiment  
    python quick_start.py --concept water   # Test single concept
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def setup_environment():
    """Set up the experimental environment and install dependencies."""
    
    print("Setting up Twin Earth experiment environment...")
    
    # Check for required packages
    required_packages = [
        "torch", "transformers", "datasets", "sklearn", 
        "numpy", "matplotlib", "seaborn"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please install with: pip install " + " ".join(missing_packages))
        return False
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True) 
    os.makedirs("data", exist_ok=True)
    
    print("✓ Environment setup complete")
    return True

def run_quick_demo():
    """Run a quick 5-minute demo to test the pipeline."""
    
    print("\n" + "="*50)
    print("QUICK DEMO: Twin Earth Experiment")
    print("="*50)
    print("This demo tests the pipeline with minimal data.")
    print("Perfect for verifying everything works before the full experiment.\n")
    
    # Import our modules
    try:
        from twin_earth_dataset import TwinEarthDatasetGenerator
        from implementation_guide import TwinEarthExperiment
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all the previous code files are saved in the same directory.")
        return
    
    # Generate small dataset
    print("1. Generating minimal Twin Earth dataset...")
    generator = TwinEarthDatasetGenerator()
    dataset = generator.create_full_dataset(
        concepts=["water"],  # Single concept
        samples_per_concept=20  # Minimal samples
    )
    
    print("Sample Earth data:")
    for i, sample in enumerate(dataset["earth_data"]["water"][:3]):
        print(f"  {i+1}. {sample}")
    
    print("\nSample Twin Earth data:")
    for i, sample in enumerate(dataset["twin_earth_data"]["water"][:3]):
        print(f"  {i+1}. {sample}")
    
    print(f"\n✓ Dataset created with {len(dataset['earth_data']['water'])} samples per world")
    
    # Show philosophical prediction
    print("\n2. Philosophical Prediction:")
    print("   If AI concepts are EXTERNALIST:")
    print("   → 'Water' vectors should differ between H₂O and XYZ training")
    print("   → Low similarity score (< 0.5)")
    print("   → Good for AI safety (concepts track real-world referents)")
    
    print("\n   If AI concepts are INTERNALIST:")
    print("   → 'Water' vectors should be identical despite different referents")
    print("   → High similarity score (> 0.7)")
    print("   → Concerning for AI safety (concepts don't track real world)")
    
    print("\n3. Next Steps:")
    print("   • Run full experiment: python quick_start.py --full")
    print("   • Analyze results in generated JSON files")
    print("   • See externalism_report.md for philosophical interpretation")
    
    return dataset

def run_single_concept_test(concept="water"):
    """Test the framework with a single concept for detailed analysis."""
    
    print(f"\n" + "="*50)
    print(f"SINGLE CONCEPT TEST: {concept.upper()}")
    print("="*50)
    
    try:
        from twin_earth_dataset import TwinEarthDatasetGenerator  
        from implementation_guide import TwinEarthExperiment
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    # Initialize experiment
    experiment = TwinEarthExperiment(base_model_name="distilgpt2")
    
    # Generate dataset for single concept
    print(f"Generating comprehensive dataset for '{concept}'...")
    dataset = experiment.step1_generate_datasets(
        concepts=[concept], 
        samples_per_concept=200
    )
    
    # Show sample data
    print(f"\nEarth '{concept}' samples:")
    for i, sample in enumerate(dataset["earth_data"][concept][:3]):
        print(f"  {i+1}. {sample}")
    
    print(f"\nTwin Earth '{concept}' samples:")
    for i, sample in enumerate(dataset["twin_earth_data"][concept][:3]):
        print(f"  {i+1}. {sample}")
    
    # Prepare and show training data stats
    earth_texts, twin_texts = experiment.step2_prepare_training_data()
    
    print(f"\nTraining data prepared:")
    print(f"  Earth model: {len(earth_texts)} training sentences")
    print(f"  Twin model: {len(twin_texts)} training sentences")
    
    print(f"\nTo complete this test, run:")
    print(f"  python quick_start.py --full --concept {concept}")
    
    return experiment

def run_full_experiment(concepts=None, quick_mode=False):
    """Run the complete Twin Earth experiment."""
    
    if concepts is None:
        concepts = ["water", "gold"]
    
    mode_str = "QUICK" if quick_mode else "FULL"
    print(f"\n" + "="*60)
    print(f"{mode_str} TWIN EARTH EXTERNALISM EXPERIMENT")
    print("="*60)
    print("This will train models and analyze concept externalism.")
    if not quick_mode:
        print("⚠️  This may take 30-60 minutes depending on your hardware.")
    print(f"Testing concepts: {concepts}")
    
    try:
        from implementation_guide import TwinEarthExperiment
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    # Configure experiment based on mode
    if quick_mode:
        model_name = "distilgpt2"
        samples_per_concept = 100
        epochs = 1
        print("(Quick mode: reduced data and training time)")
    else:
        model_name = "pythia-70m"  
        samples_per_concept = 500
        epochs = 2
        print("(Full mode: substantial training for reliable results)")
    
    # Initialize and run experiment
    experiment = TwinEarthExperiment(base_model_name=model_name)
    
    try:
        results = experiment.run_complete_experiment(
            concepts=concepts,
            samples_per_concept=samples_per_concept, 
            training_epochs=epochs
        )
        
        # Display key results
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        
        conclusion = results["philosophical_conclusion"]["overall_assessment"]
        safety_implication = results["philosophical_conclusion"]["safety_implication"]
        mean_sim = results["philosophical_conclusion"]["mean_similarity"]
        
        print(f"Philosophical Conclusion: {conclusion}")
        print(f"AI Safety Implication: {safety_implication}")
        print(f"Average Similarity Score: {mean_sim:.3f}")
        
        print(f"\nDetailed Results:")
        for concept, analysis in results["concept_analyses"].items():
            sim = analysis["earth_twin_similarity"]
            interp = analysis["interpretation"]
            print(f"  {concept.capitalize()}: {sim:.3f} - {interp}")
        
        print(f"\nFiles Generated:")
        print(f"  • twin_earth_experiment_results.json - Complete numerical results")
        print(f"  • twin_earth_experiment_data.json - Generated datasets")
        print(f"  • models/ - Trained Earth and Twin Earth models")
        
        return results
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        print("Try running the demo first to verify setup: python quick_start.py")
        return None

def show_help():
    """Display detailed help information."""
    
    print("""
Twin Earth Externalism Experiment - Help
=======================================

This experiment tests whether AI language models exhibit externalist concept formation,
where mental content depends on external referents (Putnam's Twin Earth thought experiment).

PHILOSOPHICAL BACKGROUND:
• Externalism: 'Water' means different things on Earth (H₂O) vs Twin Earth (XYZ)
• Internalism: 'Water' means the same thing regardless of environment  
• AI Safety: If AI concepts are externalist, interpretability methods might track genuine understanding

EXPERIMENT DESIGN:
1. Generate parallel datasets where concepts have different referents
2. Train identical models on Earth vs Twin Earth data
3. Extract concept vectors using interpretability methods
4. Measure similarity - low similarity supports externalism

USAGE EXAMPLES:
    python quick_start.py                   # 5-minute demo
    python quick_start.py --demo            # Same as above
    python quick_start.py --concept water   # Test single concept 
    python quick_start.py --quick           # Quick full experiment (~15 min)
    python quick_start.py --full            # Full experiment (~60 min)
    python quick_start.py --full --concepts water gold tiger

INTERPRETING RESULTS:
    Similarity < 0.3:  Strong externalism (good for AI safety)
    Similarity 0.3-0.5: Moderate externalism  
    Similarity 0.5-0.7: Weak externalism
    Similarity > 0.7:   Internalism (concerning for AI safety)

FILES GENERATED:
    • twin_earth_experiment_results.json - Main results
    • twin_earth_experiment_data.json - Generated datasets  
    • externalism_report.md - Philosophical analysis
    • models/ - Trained models for further analysis

For questions about philosophical interpretation, see:
    Putnam, H. (1975). "The meaning of 'meaning'"
    Burge, T. (1979). "Individualism and the mental"
""")

def main():
    """Main entry point for the Twin Earth experiment."""
    
    parser = argparse.ArgumentParser(
        description="Twin Earth Externalism Experiment for AI Safety Research",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--demo", action="store_true", 
                       help="Run quick demo (5 minutes)")
    parser.add_argument("--concept", type=str,
                       help="Test single concept in detail")  
    parser.add_argument("--quick", action="store_true",
                       help="Quick full experiment (~15 minutes)")
    parser.add_argument("--full", action="store_true", 
                       help="Full experiment (~60 minutes)")
    parser.add_argument("--concepts", nargs="+", 
                       default=["water", "gold"],
                       help="Concepts to test (default: water gold)")
    parser.add_argument("--help-detailed", action="store_true",
                       help="Show detailed help and philosophical background")
    
    args = parser.parse_args()
    
    # Show detailed help if requested
    if args.help_detailed:
        show_help()
        return
    
    # Setup environment
    if not setup_environment():
        print("Environment setup failed. Please install required packages.")
        return
    
    # Run appropriate experiment mode
    if args.concept:
        run_single_concept_test(args.concept)
        
    elif args.full:
        run_full_experiment(args.concepts, quick_mode=False)
        
    elif args.quick:
        run_full_experiment(args.concepts, quick_mode=True)
        
    else:  # Default to demo
        run_quick_demo()
    
    print(f"\n✓ Experiment complete!")
    print(f"For detailed philosophical help: python quick_start.py --help-detailed")

if __name__ == "__main__":
    main()