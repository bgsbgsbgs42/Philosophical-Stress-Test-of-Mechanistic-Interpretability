"""
Implementation Guide: Twin Earth Externalism Experiment
======================================================

This script shows how to practically implement the Twin Earth experiment
for testing externalism in AI language models.

Key steps:
1. Generate Twin Earth dataset
2. Fine-tune parallel models on Earth vs Twin Earth data
3. Extract concept vectors using interpretability methods
4. Analyze results for externalist vs internalist patterns
5. Generate philosophical interpretation

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from typing import Dict, List, Tuple

# Import our custom classes
from twin_earth_dataset import TwinEarthDatasetGenerator
from externalism_analysis import ExternalismAnalyzer

class TwinEarthTextDataset(Dataset):
    """Dataset class for training models on Twin Earth data"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }

class TwinEarthExperiment:
    """
    Complete implementation of Twin Earth externalism experiment.
    
    This class handles:
    1. Dataset generation
    2. Model training 
    3. Concept extraction
    4. Philosophical analysis
    """
    
    def __init__(self, base_model_name: str = "pythia-70m", device: str = "auto"):
        """
        Initialize experiment with base model.
        
        Args:
            base_model_name: HuggingFace model name (recommend Pythia for interpretability)
            device: Device for training ("cuda", "cpu", or "auto")
        """
        self.base_model_name = base_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize dataset generator
        self.dataset_generator = TwinEarthDatasetGenerator()
        
        # Storage for trained models
        self.earth_model = None
        self.twin_model = None
        self.experiment_data = None
    
    def step1_generate_datasets(self, concepts: List[str] = None, 
                               samples_per_concept: int = 1000) -> Dict:
        """
        Step 1: Generate Twin Earth datasets for training.
        
        Args:
            concepts: List of concepts to test (default: ["water", "gold", "tiger", "diamond"])
            samples_per_concept: Number of training samples per concept
            
        Returns:
            Complete dataset dictionary
        """
        
        if concepts is None:
            concepts = ["water", "gold", "tiger", "diamond"]
        
        print("Generating Twin Earth datasets...")
        print(f"Concepts: {concepts}")
        print(f"Samples per concept: {samples_per_concept}")
        
        # Generate complete dataset
        dataset = self.dataset_generator.create_full_dataset(
            concepts=concepts, 
            samples_per_concept=samples_per_concept
        )
        
        # Save dataset
        self.dataset_generator.save_dataset(dataset, "twin_earth_experiment_data.json")
        self.experiment_data = dataset
        
        # Print dataset statistics
        total_earth = sum(len(dataset["earth_data"][c]) for c in concepts)
        total_twin = sum(len(dataset["twin_earth_data"][c]) for c in concepts)
        
        print(f"✓ Dataset generated successfully")
        print(f"  Earth samples: {total_earth}")
        print(f"  Twin Earth samples: {total_twin}")
        print(f"  Total concepts: {len(concepts)}")
        
        return dataset
    
    def step2_prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """
        Step 2: Prepare training texts from generated datasets.
        
        Returns:
            Tuple of (earth_texts, twin_earth_texts)
        """
        
        if self.experiment_data is None:
            raise ValueError("Must generate datasets first (call step1_generate_datasets)")
        
        # Combine all Earth texts
        earth_texts = []
        for concept_texts in self.experiment_data["earth_data"].values():
            earth_texts.extend(concept_texts)
        
        # Combine all Twin Earth texts  
        twin_texts = []
        for concept_texts in self.experiment_data["twin_earth_data"].values():
            twin_texts.extend(concept_texts)
        
        # Add some general language modeling data to maintain basic capabilities
        general_texts = [
            "The cat sat on the mat.",
            "Mathematics is the study of numbers and patterns.",
            "The sun rises in the east and sets in the west.",
            "Books contain knowledge and stories.",
            "Music can express emotions and ideas."
        ] * 100  # Repeat to balance with concept-specific data
        
        earth_texts.extend(general_texts)
        twin_texts.extend(general_texts)
        
        # Shuffle texts
        import random
        random.shuffle(earth_texts)
        random.shuffle(twin_texts)
        
        print(f"✓ Training data prepared")
        print(f"  Earth training texts: {len(earth_texts)}")
        print(f"  Twin Earth training texts: {len(twin_texts)}")
        
        return earth_texts, twin_texts
    
    def step3_train_models(self, earth_texts: List[str], twin_texts: List[str],
                          epochs: int = 3, batch_size: int = 8, 
                          learning_rate: float = 5e-5) -> None:
        """
        Step 3: Fine-tune separate models on Earth and Twin Earth data.
        
        Args:
            earth_texts: Training texts for Earth model
            twin_texts: Training texts for Twin Earth model  
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for fine-tuning
        """
        
        print("Starting model training...")
        
        # Create datasets
        earth_dataset = TwinEarthTextDataset(earth_texts, self.tokenizer)
        twin_dataset = TwinEarthTextDataset(twin_texts, self.tokenizer)
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./twin_earth_models",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        # Train Earth model
        print("Training Earth model...")
        earth_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        earth_model.to(self.device)
        
        earth_trainer = Trainer(
            model=earth_model,
            args=training_args,
            train_dataset=earth_dataset,
            data_collator=data_collator,
        )
        
        earth_trainer.train()
        earth_trainer.save_model("./models/earth_model")
        self.earth_model = earth_model
        
        print("✓ Earth model training complete")
        
        # Train Twin Earth model
        print("Training Twin Earth model...")
        twin_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        twin_model.to(self.device)
        
        # Update output directory for twin model
        training_args.output_dir = "./twin_earth_models_twin"
        
        twin_trainer = Trainer(
            model=twin_model,
            args=training_args,
            train_dataset=twin_dataset,
            data_collator=data_collator,
        )
        
        twin_trainer.train()
        twin_trainer.save_model("./models/twin_model")
        self.twin_model = twin_model
        
        print("✓ Twin Earth model training complete")
        print("✓ Both models trained successfully")
    
    def step4_extract_concept_vectors(self, concept: str, num_contexts: int = 50,
                                    layer: int = -6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 4: Extract concept vectors from trained models.
        
        Args:
            concept: Concept to extract (e.g., "water")
            num_contexts: Number of contexts to use for extraction
            layer: Transformer layer to extract from
            
        Returns:
            Tuple of (earth_concept_vector, twin_concept_vector)
        """
        
        if self.earth_model is None or self.twin_model is None:
            raise ValueError("Must train models first (call step3_train_models)")
        
        print(f"Extracting concept vectors for '{concept}'...")
        
        # Get contexts for concept
        earth_contexts = self.experiment_data["earth_data"][concept][:num_contexts]
        twin_contexts = self.experiment_data["twin_earth_data"][concept][:num_contexts]
        
        # Extract vectors using our analyzer
        analyzer = ExternalismAnalyzer(self.earth_model, self.twin_model, self.tokenizer)
        
        earth_vector = analyzer.extract_concept_vector(
            self.earth_model, concept, earth_contexts, layer
        )
        twin_vector = analyzer.extract_concept_vector(
            self.twin_model, concept, twin_contexts, layer
        )
        
        print(f"✓ Concept vectors extracted")
        print(f"  Earth vector shape: {earth_vector.shape}")
        print(f"  Twin vector shape: {twin_vector.shape}")
        
        return earth_vector, twin_vector
    
    def step5_analyze_externalism(self, concept: str) -> Dict:
        """
        Step 5: Analyze extracted vectors for externalist patterns.
        
        Args:
            concept: Concept to analyze
            
        Returns:
            Analysis results with philosophical interpretation
        """
        
        print(f"Analyzing externalism for '{concept}'...")
        
        # Extract concept vectors
        earth_vector, twin_vector = self.step4_extract_concept_vectors(concept)
        
        # Compute similarity
        similarity = cosine_similarity(
            earth_vector.reshape(1, -1),
            twin_vector.reshape(1, -1)
        )[0, 0]
        
        # Philosophical interpretation
        if similarity < 0.3:
            interpretation = "Strong evidence for externalism"
            supports_externalism = True
        elif similarity < 0.5:
            interpretation = "Moderate evidence for externalism"  
            supports_externalism = True
        elif similarity < 0.7:
            interpretation = "Weak evidence for externalism"
            supports_externalism = False
        else:
            interpretation = "Strong evidence for internalism"
            supports_externalism = False
        
        results = {
            "concept": concept,
            "earth_twin_similarity": float(similarity),
            "supports_externalism": supports_externalism,
            "interpretation": interpretation,
            "earth_vector_norm": float(np.linalg.norm(earth_vector)),
            "twin_vector_norm": float(np.linalg.norm(twin_vector)),
        }
        
        print(f"✓ Analysis complete for '{concept}'")
        print(f"  Similarity: {similarity:.3f}")
        print(f"  Interpretation: {interpretation}")
        
        return results
    
    def run_complete_experiment(self, concepts: List[str] = None, 
                               samples_per_concept: int = 1000,
                               training_epochs: int = 3) -> Dict:
        """
        Run the complete Twin Earth experiment from start to finish.
        
        Args:
            concepts: Concepts to test
            samples_per_concept: Training samples per concept
            training_epochs: Model training epochs
            
        Returns:
            Complete experimental results
        """
        
        print("=" * 60)
        print("TWIN EARTH EXTERNALISM EXPERIMENT")
        print("=" * 60)
        
        if concepts is None:
            concepts = ["water", "gold"]  # Start with core concepts
        
        # Step 1: Generate datasets
        print("\n" + "="*40)
        print("STEP 1: GENERATING DATASETS")
        print("="*40)
        self.step1_generate_datasets(concepts, samples_per_concept)
        
        # Step 2: Prepare training data
        print("\n" + "="*40)
        print("STEP 2: PREPARING TRAINING DATA")
        print("="*40)
        earth_texts, twin_texts = self.step2_prepare_training_data()
        
        # Step 3: Train models
        print("\n" + "="*40)
        print("STEP 3: TRAINING MODELS")
        print("="*40)
        self.step3_train_models(earth_texts, twin_texts, epochs=training_epochs)
        
        # Step 4-5: Analyze each concept
        print("\n" + "="*40)
        print("STEP 4-5: CONCEPT ANALYSIS")
        print("="*40)
        
        results = {"concept_analyses": {}}
        
        for concept in concepts:
            print(f"\nAnalyzing {concept}...")
            concept_results = self.step5_analyze_externalism(concept)
            results["concept_analyses"][concept] = concept_results
        
        # Summary analysis
        similarities = [results["concept_analyses"][c]["earth_twin_similarity"] for c in concepts]
        externalism_support = [results["concept_analyses"][c]["supports_externalism"] for c in concepts]
        
        results["summary"] = {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)), 
            "concepts_supporting_externalism": sum(externalism_support),
            "total_concepts": len(concepts),
            "externalism_support_ratio": sum(externalism_support) / len(concepts)
        }
        
        # Overall philosophical conclusion
        mean_sim = results["summary"]["mean_similarity"]
        support_ratio = results["summary"]["externalism_support_ratio"]
        
        if mean_sim < 0.4 and support_ratio > 0.75:
            overall_conclusion = "Strong evidence for externalism: AI concepts track external referents"
            safety_implication = "POSITIVE: Interpretability methods may reliably track world-directed mental content"
        elif mean_sim < 0.5 and support_ratio > 0.5:
            overall_conclusion = "Moderate evidence for externalism: Partial referent sensitivity"
            safety_implication = "MIXED: Some interpretability reliability, but environment-dependent"
        elif mean_sim > 0.7 and support_ratio < 0.25:
            overall_conclusion = "Strong evidence for internalism: Concepts invariant to external referents"
            safety_implication = "CONCERNING: Interpretability may track statistical patterns, not genuine understanding"
        else:
            overall_conclusion = "Mixed evidence: Complex referent relationships"
            safety_implication = "UNCLEAR: Need further investigation of concept formation mechanisms"
        
        results["philosophical_conclusion"] = {
            "overall_assessment": overall_conclusion,
            "safety_implication": safety_implication,
            "mean_similarity": mean_sim,
            "support_ratio": support_ratio
        }
        
        # Print final results
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)
        print(f"Overall Assessment: {overall_conclusion}")
        print(f"Safety Implication: {safety_implication}")
        print(f"Mean Similarity: {mean_sim:.3f}")
        print(f"Externalism Support: {sum(externalism_support)}/{len(concepts)} concepts")
        
        print("\nConcept-by-Concept Results:")
        for concept, analysis in results["concept_analyses"].items():
            print(f"  {concept}: {analysis['earth_twin_similarity']:.3f} - {analysis['interpretation']}")
        
        # Save complete results
        with open("twin_earth_experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Complete results saved to 'twin_earth_experiment_results.json'")
        print("="*60)
        
        return results

def quick_demo_experiment():
    """
    Quick demonstration of the Twin Earth experiment with minimal training.
    Use this for testing the pipeline before running the full experiment.
    """
    
    print("Running Quick Demo Twin Earth Experiment...")
    print("(Using minimal data for pipeline testing)")
    
    # Initialize experiment
    experiment = TwinEarthExperiment(base_model_name="distilgpt2")  # Smaller model for demo
    
    # Run with minimal data
    results = experiment.run_complete_experiment(
        concepts=["water"],  # Single concept
        samples_per_concept=100,  # Minimal data
        training_epochs=1  # Single epoch
    )
    
    return results

def full_externalism_experiment():
    """
    Full-scale Twin Earth experiment for rigorous philosophical analysis.
    """
    
    print("Running Full-Scale Twin Earth Experiment...")
    print("(This may take several hours)")
    
    # Initialize experiment with appropriate model
    experiment = TwinEarthExperiment(base_model_name="pythia-160m")
    
    # Run complete experiment
    results = experiment.run_complete_experiment(
        concepts=["water", "gold", "tiger", "diamond"],  # All test concepts
        samples_per_concept=1000,  # Substantial training data
        training_epochs=3  # Proper fine-tuning
    )
    
    # Generate detailed analysis report
    if experiment.earth_model and experiment.twin_model:
        analyzer = ExternalismAnalyzer(experiment.earth_model, experiment.twin_model, experiment.tokenizer)
        detailed_results = analyzer.comprehensive_externalism_analysis(experiment.experiment_data)
        report = analyzer.generate_report(detailed_results)
        
        print("✓ Detailed philosophical analysis complete")
        print("✓ Report generated: 'externalism_report.md'")
    
    return results

class ExternalismBenchmark:
    """
    Standardized benchmark for testing externalism across different models and settings.
    """
    
    def __init__(self):
        self.dataset_generator = TwinEarthDatasetGenerator()
        self.standard_concepts = ["water", "gold", "tiger", "diamond"]
        self.benchmark_data = None
    
    def create_benchmark_dataset(self, save_path: str = "externalism_benchmark.json"):
        """Create standardized benchmark dataset."""
        
        print("Creating standardized externalism benchmark...")
        
        benchmark_data = self.dataset_generator.create_full_dataset(
            concepts=self.standard_concepts,
            samples_per_concept=500
        )
        
        # Add benchmark metadata
        benchmark_data["benchmark_info"] = {
            "version": "1.0",
            "concepts": self.standard_concepts,
            "philosophical_framework": "Putnam-Burge externalism",
            "expected_patterns": {
                "strong_externalism": "similarity < 0.3",
                "moderate_externalism": "0.3 <= similarity < 0.5", 
                "weak_externalism": "0.5 <= similarity < 0.7",
                "internalism": "similarity >= 0.7"
            }
        }
        
        with open(save_path, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        
        self.benchmark_data = benchmark_data
        print(f"✓ Benchmark dataset saved to {save_path}")
        
        return benchmark_data
    
    def evaluate_model(self, model, tokenizer, benchmark_data: Dict = None) -> Dict:
        """Evaluate a single model against the externalism benchmark."""
        
        if benchmark_data is None:
            if self.benchmark_data is None:
                raise ValueError("Must create benchmark dataset first")
            benchmark_data = self.benchmark_data
        
        # For single model evaluation, we compare against baseline patterns
        # rather than training twin models (useful for evaluating existing models)
        
        results = {}
        for concept in self.standard_concepts:
            earth_contexts = benchmark_data["earth_data"][concept][:50]
            twin_contexts = benchmark_data["twin_earth_data"][concept][:50]
            
            # Extract concept vectors (using same model for both - tests internal consistency)
            analyzer = ExternalismAnalyzer(model, model, tokenizer)
            
            earth_vector = analyzer.extract_concept_vector(model, concept, earth_contexts)
            twin_vector = analyzer.extract_concept_vector(model, concept, twin_contexts)
            
            similarity = cosine_similarity(
                earth_vector.reshape(1, -1), twin_vector.reshape(1, -1)
            )[0, 0]
            
            results[concept] = {
                "similarity": float(similarity),
                "earth_contexts_processed": len(earth_contexts),
                "twin_contexts_processed": len(twin_contexts)
            }
        
        return results

# Usage examples and main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Twin Earth Externalism Experiment")
    parser.add_argument("--mode", choices=["demo", "full", "benchmark"], 
                       default="demo", help="Experiment mode")
    parser.add_argument("--model", default="distilgpt2", 
                       help="Base model for experiment")
    parser.add_argument("--concepts", nargs="+", default=["water", "gold"],
                       help="Concepts to test")
    parser.add_argument("--samples", type=int, default=500,
                       help="Samples per concept")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Training epochs")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        print("Running demonstration experiment...")
        results = quick_demo_experiment()
        
    elif args.mode == "full":
        print("Running full experiment...")
        experiment = TwinEarthExperiment(base_model_name=args.model)
        results = experiment.run_complete_experiment(
            concepts=args.concepts,
            samples_per_concept=args.samples,
            training_epochs=args.epochs
        )
        
    elif args.mode == "benchmark":
        print("Creating benchmark dataset...")
        benchmark = ExternalismBenchmark()
        benchmark_data = benchmark.create_benchmark_dataset()
        print("Benchmark created successfully!")
    
    print("\nExperiment complete!")
    print("Check output files for detailed results.")

# Additional utility functions for analysis

def compare_multiple_models(model_names: List[str], concepts: List[str] = ["water", "gold"]):
    """
    Compare externalism patterns across multiple models.
    Useful for understanding how model architecture affects concept formation.
    """
    
    results = {}
    benchmark = ExternalismBenchmark()
    benchmark_data = benchmark.create_benchmark_dataset()
    
    for model_name in model_names:
        print(f"Evaluating {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_results = benchmark.evaluate_model(model, tokenizer, benchmark_data)
            results[model_name] = model_results
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results

def analyze_scaling_effects(base_model_family: str = "pythia", 
                           model_sizes: List[str] = ["70m", "160m", "410m"]):
    """
    Analyze how model scale affects externalist concept formation.
    Tests philosophical prediction that larger models might show more sophisticated externalism.
    """
    
    model_names = [f"{base_model_family}-{size}" for size in model_sizes]
    results = compare_multiple_models(model_names)
    
    # Analyze scaling trends
    scaling_analysis = {}
    for concept in ["water", "gold"]:
        similarities = []
        sizes = []
        
        for model_name in model_names:
            if "error" not in results[model_name]:
                similarity = results[model_name][concept]["similarity"]
                similarities.append(similarity)
                sizes.append(model_name)
        
        scaling_analysis[concept] = {
            "model_sizes": sizes,
            "similarities": similarities,
            "trend": "decreasing" if len(similarities) > 1 and similarities[-1] < similarities[0] else "stable"
        }
    
    return scaling_analysis

# Export key functions for easy import
__all__ = [
    "TwinEarthExperiment",
    "ExternalismBenchmark", 
    "quick_demo_experiment",
    "full_externalism_experiment",
    "compare_multiple_models",
    "analyze_scaling_effects"
] 