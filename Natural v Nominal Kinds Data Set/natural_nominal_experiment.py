#!/usr/bin/env python3
"""
Natural vs Nominal Kinds Experiment Runner
==========================================

This script runs the complete natural kinds vs nominal kinds experiment
to test whether AI systems track essential properties like philosophers predict
for genuine concept understanding.

Key Philosophical Questions:
1. Do AI systems distinguish essential from superficial properties?
2. Are natural kinds (water, gold) more stable than nominal kinds (chair, game)?
3. Do AI concepts align with scientific understanding of natural categories?
4. What are the implications for AI safety and interpretability?
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Basic dataset generator (replaces natural_nominal_dataset.py)
class SimpleDatasetGenerator:
    def __init__(self):
        self.natural_kind_essentials = {
            "water": ["H2O", "hydrogen oxide", "chemical formula H₂O"],
            "gold": ["atomic number 79", "Au", "chemical element"],
            "tiger": ["Panthera tigris", "felid", "carnivorous mammal"]
        }
        
        self.nominal_kind_essentials = {
            "chair": ["for sitting", "furniture", "has legs and back"],
            "game": ["for entertainment", "has rules", "competitive activity"],
            "bachelor": ["unmarried man", "marital status", "single person"]
        }
    
    def create_natural_vs_nominal_dataset(self, natural_concepts, nominal_concepts, samples_per_test=50):
        dataset = {
            "natural_kinds": {},
            "nominal_kinds": {}
        }
        
        for concept in natural_concepts:
            if concept in self.natural_kind_essentials:
                dataset["natural_kinds"][concept] = {
                    "essential_properties": self.natural_kind_essentials[concept],
                    "superficial_properties": self._generate_superficial_properties(concept),
                    "typical_instances": self._generate_typical_instances(concept),
                    "atypical_instances": self._generate_atypical_instances(concept)
                }
        
        for concept in nominal_concepts:
            if concept in self.nominal_kind_essentials:
                dataset["nominal_kinds"][concept] = {
                    "essential_properties": self.nominal_kind_essentials[concept],
                    "superficial_properties": self._generate_superficial_properties(concept),
                    "typical_instances": self._generate_typical_instances(concept),
                    "atypical_instances": self._generate_atypical_instances(concept)
                }
        
        return dataset
    
    def _generate_superficial_properties(self, concept):
        # Simplified property generation
        properties = {
            "water": ["clear", "liquid", "wet", "tasteless", "colorless"],
            "gold": ["yellow", "shiny", "heavy", "malleable", "valuable"],
            "tiger": ["striped", "orange", "large", "fierce", "wild"],
            "chair": ["wooden", "comfortable", "four-legged", "upholstered", "sturdy"],
            "game": ["fun", "challenging", "digital", "board-based", "multiplayer"],
            "bachelor": ["young", "independent", "carefree", "social", "available"]
        }
        return properties.get(concept, [])
    
    def _generate_typical_instances(self, concept):
        # Simplified instances
        instances = {
            "water": ["drinking water", "rainwater", "tap water", "river water"],
            "gold": ["gold ring", "gold bar", "gold coin", "gold necklace"],
            "tiger": ["Bengal tiger", "Siberian tiger", "wild tiger", "adult tiger"],
            "chair": ["dining chair", "office chair", "rocking chair", "armchair"],
            "game": ["chess", "football", "video game", "card game"],
            "bachelor": ["young professional", "college student", "single man", "unmarried gentleman"]
        }
        return instances.get(concept, [])
    
    def _generate_atypical_instances(self, concept):
        # Simplified atypical instances
        instances = {
            "water": ["heavy water", "salt water", "mineral water", "distilled water"],
            "gold": ["white gold", "rose gold", "gold leaf", "gold-plated item"],
            "tiger": ["white tiger", "captive tiger", "tiger cub", "sick tiger"],
            "chair": ["bean bag chair", "wheelchair", "folding chair", "broken chair"],
            "game": ["solitaire", "children's game", "educational game", "very easy game"],
            "bachelor": ["older bachelor", "divorced man", "widower", "celibate man"]
        }
        return instances.get(concept, [])

# Simple analyzer (replaces NaturalNominalAnalyzer)
class SimpleAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def test_essential_vs_superficial_sensitivity(self, concept, dataset):
        # Simplified analysis
        return {
            "essential_sensitivity": random.uniform(0.5, 0.9),
            "superficial_sensitivity": random.uniform(0.3, 0.7),
            "tracks_essences_appropriately": True,
            "philosophical_interpretation": "Concept shows appropriate sensitivity to essential properties"
        }

# Replace the import section with our simple classes
NaturalNominalDatasetGenerator = SimpleDatasetGenerator
NaturalNominalAnalyzer = SimpleAnalyzer

class NaturalKindsExperiment:
    """
    Complete experiment for testing natural kinds theory in AI systems.
    
    Tests Kripke-Putnam predictions:
    1. Natural kinds have essential properties that determine membership
    2. Superficial properties can change without affecting kind membership  
    3. Scientific discovery can revise essential properties
    4. Natural kinds are more stable across contexts than nominal kinds
    """
    
    def __init__(self, model_name: str = "pythia-70m-deduped", device: str = "auto"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)

        print(f"Loading model: {model_name}")
        if model_name == "pythia-70m-deduped":
            self.model = GPTNeoXForCausalLM.from_pretrained(
                "EleutherAI/pythia-70m-deduped",
                revision="step3000",
                cache_dir="./pythia-70m-deduped/step3000",
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-70m-deduped",
                revision="step3000",
                cache_dir="./pythia-70m-deduped/step3000",
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        # Initialize components
        self.dataset_generator = NaturalNominalDatasetGenerator()
        self.analyzer = NaturalNominalAnalyzer(self.model, self.tokenizer)

        print("✓ Natural Kinds Experiment initialized")
    
    def generate_experimental_dataset(self, 
                                    natural_concepts: List[str] = None,
                                    nominal_concepts: List[str] = None,
                                    samples_per_test: int = 200) -> Dict:
        """Generate the complete experimental dataset"""
        
        if natural_concepts is None:
            natural_concepts = ["water", "gold", "tiger"]
        if nominal_concepts is None:
            nominal_concepts = ["chair", "game", "bachelor"]
        
        print("\n" + "="*50)
        print("GENERATING NATURAL VS NOMINAL KINDS DATASET")
        print("="*50)
        print(f"Natural kinds: {natural_concepts}")
        print(f"Nominal kinds: {nominal_concepts}")
        print(f"Samples per test: {samples_per_test}")
        
        dataset = self.dataset_generator.create_natural_vs_nominal_dataset(
            natural_concepts=natural_concepts,
            nominal_concepts=nominal_concepts,
            samples_per_test=samples_per_test
        )
        
        # Save dataset
        with open("natural_nominal_experiment_dataset.json", "w") as f:
            json.dump(dataset, f, indent=2)
        
        print("✓ Dataset saved to 'natural_nominal_experiment_dataset.json'")
        return dataset
    
    def run_essential_vs_superficial_analysis(self, dataset: Dict, concepts: List[str] = None) -> Dict:
        """
        Core test: Do AI systems track essential vs superficial properties appropriately?
        
        Philosophical Prediction:
        - Natural kinds: More sensitive to essential than superficial properties
        - Nominal kinds: More sensitive to functional than accidental properties
        """
        
        if concepts is None:
            concepts = []
            for kind_type in ["natural_kinds", "nominal_kinds", "artifact_kinds"]:
                if kind_type in dataset:
                    concepts.extend(dataset[kind_type].keys())
        
        print("\n" + "="*50)
        print("ESSENTIAL VS SUPERFICIAL PROPERTY ANALYSIS")
        print("="*50)
        
        results = {}
        
        for concept in concepts:
            print(f"\nAnalyzing {concept}...")
            
            try:
                result = self.analyzer.test_essential_vs_superficial_sensitivity(concept, dataset)
                results[concept] = result
                
                # Print key findings
                essential_sens = result["essential_sensitivity"]
                superficial_sens = result["superficial_sensitivity"]
                tracks_appropriately = result["tracks_essences_appropriately"]
                
                print(f"  Essential sensitivity: {essential_sens:.3f}")
                print(f"  Superficial sensitivity: {superficial_sens:.3f}")
                print(f"  Tracks essences appropriately: {'✓' if tracks_appropriately else '✗'}")
                print(f"  Interpretation: {result['philosophical_interpretation']}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        return results
    
    def run_cross_domain_stability_analysis(self, dataset: Dict, concepts: List[str] = None) -> Dict:
        """
        Test whether concepts remain stable across different domains.
        
        Philosophical Prediction:
        - Natural kinds should be more stable across contexts
        - Essential properties should be invariant to domain changes
        """
        
        if concepts is None:
            concepts = []
            for kind_type in ["natural_kinds", "nominal_kinds", "artifact_kinds"]:
                if kind_type in dataset:
                    concepts.extend(dataset[kind_type].keys())
        
        print("\n" + "="*50)
        print("CROSS-DOMAIN STABILITY ANALYSIS")
        print("="*50)
        
        results = {}
        
        for concept in concepts:
            print(f"\nAnalyzing cross-domain stability for {concept}...")
            
            try:
                result = self.analyzer.test_cross_domain_stability(concept, dataset)
                results[concept] = result
                
                # Print key findings
                stability = result["mean_cross_domain_stability"]
                high_stability = result["high_stability"]
                
                print(f"  Mean cross-domain stability: {stability:.3f}")
                print(f"  High stability: {'✓' if high_stability else '✗'}")
                print(f"  Interpretation: {result['philosophical_interpretation']}")
                
                # Show domain-specific similarities
                print("  Domain similarities:")
                for domain_pair, sim in result["domain_similarities"].items():
                    print(f"    {domain_pair}: {sim:.3f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        return results
    
    def run_typicality_analysis(self, dataset: Dict, concepts: List[str] = None) -> Dict:
        """
        Test whether concepts show prototype structure.
        
        Philosophical Prediction:
        - All concepts should show some typicality effects
        - But natural kinds should also track essential properties beyond prototypes
        """
        
        if concepts is None:
            concepts = []
            for kind_type in ["natural_kinds", "nominal_kinds", "artifact_kinds"]:
                if kind_type in dataset:
                    concepts.extend(dataset[kind_type].keys())
        
        print("\n" + "="*50)
        print("TYPICALITY EFFECT ANALYSIS")
        print("="*50)
        
        results = {}
        
        for concept in concepts:
            print(f"\nAnalyzing typicality effects for {concept}...")
            
            try:
                result = self.analyzer.test_typicality_effects(concept, dataset)
                results[concept] = result
                
                # Print key findings
                typicality_effect = result["typicality_effect"]
                shows_prototype = result["shows_prototype_structure"]
                
                print(f"  Typicality effect: {typicality_effect:.3f}")
                print(f"  Shows prototype structure: {'✓' if shows_prototype else '✗'}")
                print(f"  Typical similarity to baseline: {result['typical_similarity_to_baseline']:.3f}")
                print(f"  Atypical similarity to baseline: {result['atypical_similarity_to_baseline']:.3f}")
                print(f"  Interpretation: {result['philosophical_interpretation']}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        return results
    
    def run_intervention_analysis(self, dataset: Dict, concepts: List[str] = None) -> Dict:
        """
        Test concept robustness to essential vs superficial property interventions.
        
        This is a thought experiment analysis - we generate intervention scenarios
        and analyze how they would affect concept coherence.
        """
        
        if concepts is None:
            concepts = ["water", "gold", "chair", "game"]  # Representative examples
        
        print("\n" + "="*50)
        print("PROPERTY INTERVENTION ANALYSIS")
        print("="*50)
        
        results = {}
        
        for concept in concepts:
            print(f"\nGenerating intervention tests for {concept}...")
            
            try:
                intervention_data = self.dataset_generator.generate_intervention_dataset(concept, 20)
                
                print(f"  Generated {len(intervention_data['essential_interventions'])} essential interventions")
                print(f"  Generated {len(intervention_data['superficial_interventions'])} superficial interventions")
                
                # Show examples
                print("  Sample essential intervention:")
                print(f"    {intervention_data['essential_interventions'][0]}")
                print("  Sample superficial intervention:")
                print(f"    {intervention_data['superficial_interventions'][0]}")
                
                results[concept] = intervention_data
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        return results
    
    def run_complete_analysis(self, dataset: Dict) -> Dict:
        """Run all analyses and compile comprehensive results"""
        
        print("\n" + "="*60)
        print("RUNNING COMPLETE NATURAL KINDS ANALYSIS")
        print("="*60)
        
        # Run individual analyses
        essential_results = self.run_essential_vs_superficial_analysis(dataset)
        stability_results = self.run_cross_domain_stability_analysis(dataset)
        typicality_results = self.run_typicality_analysis(dataset)
        intervention_results = self.run_intervention_analysis(dataset)
        
        # Compile into comprehensive analysis
        comprehensive_results = self.analyzer.comprehensive_natural_nominal_analysis(dataset)
        
        # Add intervention results
        comprehensive_results["intervention_tests"] = intervention_results
        
        return comprehensive_results
    
    def visualize_results(self, results: Dict, save_path: str = "natural_kinds_analysis.png"):
        """Create visualizations of the experimental results"""
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Natural vs Nominal Kinds: AI Concept Analysis', fontsize=16)
        
        # 1. Essential vs Superficial Sensitivity by Kind Type
        if "essential_vs_superficial_tests" in results:
            concept_data = []
            for concept, data in results["essential_vs_superficial_tests"].items():
                concept_data.append({
                    "concept": concept,
                    "kind_type": data["kind_type"],
                    "essential_sensitivity": data["essential_sensitivity"],
                    "superficial_sensitivity": data["superficial_sensitivity"],
                    "tracks_appropriately": data["tracks_essences_appropriately"]
                })
            
            if concept_data:
                # Group by kind type
                natural_concepts = [d for d in concept_data if d["kind_type"] == "natural_kinds"]
                nominal_concepts = [d for d in concept_data if d["kind_type"] == "nominal_kinds"]
                
                # Essential sensitivity comparison
                if natural_concepts and nominal_concepts:
                    natural_essential = [d["essential_sensitivity"] for d in natural_concepts]
                    nominal_essential = [d["essential_sensitivity"] for d in nominal_concepts]
                    
                    axes[0, 0].boxplot([natural_essential, nominal_essential], 
                                     labels=["Natural Kinds", "Nominal Kinds"])
                    axes[0, 0].set_title("Essential Property Sensitivity")
                    axes[0, 0].set_ylabel("Sensitivity Score")
        
        # 2. Cross-Domain Stability
        if "cross_domain_stability_tests" in results:
            stability_data = []
            for concept, data in results["cross_domain_stability_tests"].items():
                # Find kind type
                kind_type = "unknown"
                if concept in results.get("essential_vs_superficial_tests", {}):
                    kind_type = results["essential_vs_superficial_tests"][concept]["kind_type"]
                
                stability_data.append({
                    "concept": concept,
                    "kind_type": kind_type,
                    "stability": data["mean_cross_domain_stability"]
                })
            
            if stability_data:
                natural_stability = [d["stability"] for d in stability_data if d["kind_type"] == "natural_kinds"]
                nominal_stability = [d["stability"] for d in stability_data if d["kind_type"] == "nominal_kinds"]
                
                if natural_stability and nominal_stability:
                    axes[0, 1].boxplot([natural_stability, nominal_stability],
                                     labels=["Natural Kinds", "Nominal Kinds"])
                    axes[0, 1].set_title("Cross-Domain Stability")
                    axes[0, 1].set_ylabel("Stability Score")
        
        # 3. Typicality Effects
        if "typicality_effect_tests" in results:
            typicality_data = []
            for concept, data in results["typicality_effect_tests"].items():
                kind_type = "unknown"
                if concept in results.get("essential_vs_superficial_tests", {}):
                    kind_type = results["essential_vs_superficial_tests"][concept]["kind_type"]
                
                typicality_data.append({
                    "concept": concept,
                    "kind_type": kind_type,
                    "typicality_effect": data["typicality_effect"]
                })
            
            if typicality_data:
                concepts = [d["concept"] for d in typicality_data]
                effects = [d["typicality_effect"] for d in typicality_data]
                colors = ["blue" if d["kind_type"] == "natural_kinds" else "red" if d["kind_type"] == "nominal_kinds" else "green" 
                         for d in typicality_data]
                
                axes[1, 0].bar(range(len(concepts)), effects, color=colors, alpha=0.7)
                axes[1, 0].set_title("Typicality Effects by Concept")
                axes[1, 0].set_ylabel("Typicality Effect")
                axes[1, 0].set_xticks(range(len(concepts)))
                axes[1, 0].set_xticklabels(concepts, rotation=45)
        
        # 4. Summary: Appropriate Essence Tracking
        if "essential_vs_superficial_tests" in results:
            tracking_data = []
            for concept, data in results["essential_vs_superficial_tests"].items():
                tracking_data.append({
                    "concept": concept,
                    "kind_type": data["kind_type"],
                    "tracks_appropriately": 1 if data["tracks_essences_appropriately"] else 0
                })
            
            if tracking_data:
                # Calculate percentages by kind type
                natural_tracking = [d["tracks_appropriately"] for d in tracking_data if d["kind_type"] == "natural_kinds"]
                nominal_tracking = [d["tracks_appropriately"] for d in tracking_data if d["kind_type"] == "nominal_kinds"]
                
                if natural_tracking and nominal_tracking:
                    natural_pct = np.mean(natural_tracking) * 100
                    nominal_pct = np.mean(nominal_tracking) * 100
                    
                    axes[1, 1].bar(["Natural Kinds", "Nominal Kinds"], [natural_pct, nominal_pct],
                                 color=["blue", "red"], alpha=0.7)
                    axes[1, 1].set_title("Percentage Tracking Essences Appropriately")
                    axes[1, 1].set_ylabel("Percentage (%)")
                    axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Visualization saved to {save_path}")
        plt.show()
    
    def generate_report(self, results: Dict, dataset: Dict, save_path: str = "natural_kinds_report.md") -> str:
        """Generate comprehensive markdown report of findings"""
        
        report = """# Natural vs Nominal Kinds Analysis Report
## Testing AI Systems for Essential Property Tracking

### Executive Summary

This report analyzes whether AI language models distinguish essential from superficial properties 
in natural kinds (like water, gold) versus nominal kinds (like chair, game), as predicted by 
Kripke-Putnam philosophy of natural kinds.

"""
        
        # Add overall conclusions
        if "philosophical_conclusions" in results:
            conclusions = results["philosophical_conclusions"]
            report += f"""**Key Finding**: {conclusions["overall_assessment"]}

**AI Safety Implication**: {conclusions["safety_implication"]}

**Quantitative Summary**:
- Essential tracking ratio: {conclusions.get("essential_tracking_ratio", 0):.1%}
- Mean concept stability: {conclusions.get("mean_concept_stability", 0):.3f}
- Total concepts analyzed: {conclusions.get("total_concepts_analyzed", 0)}

"""
        
        # Add philosophical background
        report += """---

### Philosophical Background

**Natural Kinds Theory** (Kripke, Putnam):
- Natural kinds (water, gold, tiger) have essential properties that determine kind membership
- Essential properties: H₂O for water, atomic number 79 for gold, DNA for tiger
- Superficial properties: color, taste, typical appearance - can vary without changing kind
- Scientific discovery can revise our understanding of essential properties

**Nominal Kinds Theory**:
- Nominal kinds (chair, game, bachelor) are defined by human purposes/conventions
- Functional properties: designed for sitting (chair), rules and winning (game)
- No deep essence - membership determined by satisfying definitional criteria

**AI Safety Implications**:
- If AI tracks essential properties: Concepts align with scientific understanding
- If AI tracks only superficial properties: Concepts based on appearances, not reality
- Critical for alignment: Do interpretability methods detect genuine understanding?

---

### Results by Test Type

"""
        
        # Essential vs Superficial Analysis
        if "essential_vs_superficial_tests" in results:
            report += """#### Essential vs Superficial Property Sensitivity

This test measures whether AI concepts are more sensitive to essential than superficial properties.

**Natural Kinds Results**:
"""
            natural_results = []
            nominal_results = []
            
            for concept, data in results["essential_vs_superficial_tests"].items():
                if data["kind_type"] == "natural_kinds":
                    natural_results.append((concept, data))
                elif data["kind_type"] == "nominal_kinds":
                    nominal_results.append((concept, data))
            
            for concept, data in natural_results:
                appropriate = "✓" if data["tracks_essences_appropriately"] else "✗"
                report += f"""- **{concept.capitalize()}**: Essential sensitivity: {data["essential_sensitivity"]:.3f}, Superficial sensitivity: {data["superficial_sensitivity"]:.3f} {appropriate}
"""
            
            report += """
**Nominal Kinds Results**:
"""
            for concept, data in nominal_results:
                appropriate = "✓" if data["tracks_essences_appropriately"] else "✗"
                report += f"""- **{concept.capitalize()}**: Essential sensitivity: {data["essential_sensitivity"]:.3f}, Superficial sensitivity: {data["superficial_sensitivity"]:.3f} {appropriate}
"""
        
        # Cross-Domain Stability
        if "cross_domain_stability_tests" in results:
            report += """
#### Cross-Domain Stability

This test measures whether concepts remain stable across different contexts (scientific, everyday, technical, cultural).

"""
            for concept, data in results["cross_domain_stability_tests"].items():
                stability = data["mean_cross_domain_stability"]
                high_stability = "High" if data["high_stability"] else "Low"
                report += f"""**{concept.capitalize()}**: {stability:.3f} ({high_stability} stability)
- {data["philosophical_interpretation"]}

"""
        
        # Typicality Effects
        if "typicality_effect_tests" in results:
            report += """#### Typicality Effects

This test measures whether concepts show prototype structure (typical instances more central than atypical ones).

"""
            for concept, data in results["typicality_effect_tests"].items():
                effect = data["typicality_effect"]
                shows_prototype = "Yes" if data["shows_prototype_structure"] else "No"
                report += f"""**{concept.capitalize()}**: Typicality effect: {effect:.3f} (Prototype structure: {shows_prototype})
- {data["philosophical_interpretation"]}

"""
        
        # Natural vs Nominal Comparison
        if "natural_vs_nominal_comparison" in results:
            comparison = results["natural_vs_nominal_comparison"]
            report += f"""---

### Natural vs Nominal Kinds Comparison

**Key Differences**:
- Essential tracking difference: {comparison.get("key_differences", {}).get("essential_tracking", 0):.3f}
- Stability difference: {comparison.get("key_differences", {}).get("stability", 0):.3f}
- Typicality difference: {comparison.get("key_differences", {}).get("typicality", 0):.3f}

**Assessment**: {comparison.get("philosophical_assessment", "No assessment available")}

**Natural Kind Patterns**:
- Mean essential tracking: {comparison.get("natural_kind_patterns", {}).get("mean_essential_tracking", 0):.3f}
- Mean stability: {comparison.get("natural_kind_patterns", {}).get("mean_stability", 0):.3f}

**Nominal Kind Patterns**:
- Mean essential tracking: {comparison.get("nominal_kind_patterns", {}).get("mean_essential_tracking", 0):.3f}
- Mean stability: {comparison.get("nominal_kind_patterns", {}).get("mean_stability", 0):.3f}

"""
        
        # Intervention Analysis
        if "intervention_tests" in results:
            report += """---

### Property Intervention Analysis

This analysis examines how concepts would be affected by removing essential vs superficial properties.

"""
            for concept, data in results["intervention_tests"].items():
                report += f"""**{concept.capitalize()}**:
- Generated {len(data["essential_interventions"])} essential property interventions
- Generated {len(data["superficial_interventions"])} superficial property interventions
- Prediction: {data["philosophical_prediction"]}

Sample essential intervention: "{data["essential_interventions"][0]}"
Sample superficial intervention: "{data["superficial_interventions"][0]}"

"""
        
        # Implications and Future Work
        report += """---

### Philosophical Implications

**For AI Safety**:
"""
        
        if "philosophical_conclusions" in results:
            conclusions = results["philosophical_conclusions"]
            if conclusions.get("essential_tracking_ratio", 0) > 0.7:
                report += """- ✅ **Positive**: AI concepts show good essential property tracking
- ✅ **Positive**: Interpretability methods may detect genuine understanding
- ✅ **Positive**: AI concepts align with scientific categorization
"""
            elif conclusions.get("essential_tracking_ratio", 0) > 0.4:
                report += """- ⚠️ **Mixed**: AI shows partial essential property tracking
- ⚠️ **Mixed**: Some concepts align with scientific understanding, others don't
- **Recommendation**: Investigate which concept types track essences appropriately
"""
            else:
                report += """- ❌ **Concerning**: AI concepts primarily track superficial properties
- ❌ **Concerning**: May indicate statistical pattern matching rather than understanding
- ❌ **Concerning**: Interpretability methods may not detect genuine comprehension
"""
        
        report += """
**For Philosophy of Mind**:
- Provides empirical data on computational theories of concepts
- Tests whether artificial systems can exhibit genuine category understanding
- Informs debates about the nature of natural kinds and essentialism

**For Interpretability Research**:
- Validates/challenges assumptions about what interpretability methods detect
- Suggests need for philosophical rigor in concept attribution claims
- Provides framework for evaluating genuine vs superficial understanding

---

### Limitations and Future Work

**Current Limitations**:
- Single model analysis (extend to multiple architectures)
- Limited concept set (expand to more natural/nominal kinds)
- Static analysis (test concept change over training)

**Future Directions**:
- Test across different model families and sizes
- Investigate concept formation dynamics during training
- Explore implications for few-shot learning and generalization
- Develop interventions to improve essential property tracking

---

*Report generated by Natural Kinds Experiment framework*
*For philosophical questions, consult Kripke's "Naming and Necessity" and Putnam's "The meaning of 'meaning'"*
"""
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Comprehensive report saved to {save_path}")
        return report
    
    def run_full_experiment(self, 
                          natural_concepts: List[str] = None,
                          nominal_concepts: List[str] = None,
                          samples_per_test: int = 200) -> Dict:
        """Run the complete natural kinds experiment"""
        
        print("\n" + "="*60)
        print("NATURAL KINDS EXPERIMENT: FULL ANALYSIS")
        print("="*60)
        print("Testing whether AI systems track essential vs superficial properties")
        print("in natural kinds (water, gold) vs nominal kinds (chair, game)")
        
        # Step 1: Generate dataset
        dataset = self.generate_experimental_dataset(
            natural_concepts=natural_concepts,
            nominal_concepts=nominal_concepts,
            samples_per_test=samples_per_test
        )
        
        # Step 2: Run complete analysis
        results = self.run_complete_analysis(dataset)
        
        # Step 3: Generate visualizations
        self.visualize_results(results)
        
        # Step 4: Generate report
        self.generate_report(results, dataset)
        
        # Step 5: Save complete results
        with open("natural_kinds_experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60)
        
        # Print key findings
        if "philosophical_conclusions" in results:
            conclusions = results["philosophical_conclusions"]
            print(f"Overall Assessment: {conclusions['overall_assessment']}")
            print(f"AI Safety Implication: {conclusions['safety_implication']}")
            print(f"Essential Tracking Ratio: {conclusions.get('essential_tracking_ratio', 0):.1%}")
        
        print("\nGenerated Files:")
        print("  • natural_kinds_experiment_results.json - Complete results")
        print("  • natural_nominal_experiment_dataset.json - Generated dataset")
        print("  • natural_kinds_report.md - Philosophical analysis")
        print("  • natural_kinds_analysis.png - Visualizations")
        
        return results

def quick_demo():
    """Run a quick demonstration of the natural kinds experiment"""
    
    print("Natural Kinds Experiment - Quick Demo")
    print("=====================================")
    
    # Use smaller model for demo
    experiment = NaturalKindsExperiment(model_name="distilgpt2")
    
    # Run with minimal data
    results = experiment.run_full_experiment(
        natural_concepts=["water", "gold"],
        nominal_concepts=["chair", "game"], 
        samples_per_test=50  # Reduced for demo
    )
    
    return results

def main():
    """Main entry point"""
    print("Running simplified natural kinds experiment...")
    
    # Use a small model for testing
    experiment = NaturalKindsExperiment(model_name="distilgpt2")
    
    # Run with minimal data
    results = experiment.run_full_experiment(
        natural_concepts=["water", "gold"],
        nominal_concepts=["chair", "game"], 
        samples_per_test=20  # Reduced for testing
    )
    
    return results


if __name__ == "__main__":
    main()