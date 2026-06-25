import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dataclasses import dataclass
import logging
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompositionalityResult:
    """Data class to store compositionality test results"""
    concept_a: str
    concept_b: str
    composite: str
    arithmetic_similarity: float
    contextual_similarity: float
    systematicity_score: float
    is_systematic: bool
    error_type: Optional[str] = None

class CompositionalityAnalyzer:
    """
    Tests whether sparse autoencoder features exhibit systematic compositionality
    as predicted by classical theories of mind (Fodor & Pylyshyn, 1988)
    """
    
    def __init__(self, model_name: str = "pythia-70m", layer: int = -6, device: str = "auto", use_hf: bool = False):
        """
        Initialize the compositionality analyzer

        Args:
            model_name: Pythia model to use (70m, 160m, 410m)
            layer: Which layer to extract activations from
            device: Device to run on
            use_hf: If True, use HuggingFace Transformers backend
        """
        self.model_name = model_name
        self.use_hf = use_hf
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        if use_hf:
            # Use HuggingFace Transformers backend
            self.hf_model = GPTNeoXForCausalLM.from_pretrained(
                "EleutherAI/pythia-70m-deduped",
                revision="step3000",
                cache_dir="./pythia-70m-deduped/step3000",
            ).to(self.device)
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-70m-deduped",
                revision="step3000",
                cache_dir="./pythia-70m-deduped/step3000",
            )
            logger.info(f"Loaded HuggingFace Pythia model on {self.device}")
        else:
            # Use TransformerLens backend
            self.model = HookedTransformer.from_pretrained(
                model_name, device=self.device
            )
            num_layers = self.model.cfg.n_layers
            if layer < 0:
                self.layer = num_layers + layer
            else:
                self.layer = layer
            logger.info(f"Loaded TransformerLens model {model_name} on {self.device}")

        # Test cases for systematic compositionality
        self.test_cases = self._generate_test_cases()
        self.results: List[CompositionalityResult] = []

    def _generate_test_cases(self) -> List[Tuple[str, str, str, str]]:
        """
        Generate test cases for compositionality experiments
        Format: (concept_a, concept_b, composite, category)
        """
        return [
            # Basic adjective-noun composition
            ("red", "car", "red car", "color_object"),
            ("blue", "car", "blue car", "color_object"),
            ("green", "apple", "green apple", "color_object"),
            ("red", "apple", "red apple", "color_object"),
            ("big", "house", "big house", "size_object"),
            ("small", "house", "small house", "size_object"),
            ("fast", "train", "fast train", "speed_object"),
            ("slow", "train", "slow train", "speed_object"),
            
            # Systematic patterns (should show structural similarity)
            ("tall", "building", "tall building", "size_object"),
            ("short", "building", "short building", "size_object"),
            ("heavy", "box", "heavy box", "weight_object"),
            ("light", "box", "light box", "weight_object"),
            
            # Verb-object composition
            ("read", "book", "read book", "action_object"),
            ("write", "letter", "write letter", "action_object"),
            ("drive", "car", "drive car", "action_object"),
            ("cook", "dinner", "cook dinner", "action_object"),
            
            # Abstract concepts
            ("good", "idea", "good idea", "quality_abstract"),
            ("bad", "decision", "bad decision", "quality_abstract"),
            ("true", "statement", "true statement", "truth_abstract"),
            ("false", "claim", "false claim", "truth_abstract"),
            
            # Non-compositional controls (should fail systematicity)
            ("red", "herring", "red herring", "idiom"),  # Idiom
            ("dead", "beat", "deadbeat", "compound"),    # Non-literal composition
            ("hot", "dog", "hot dog", "idiom"),          # Food name
            ("green", "house", "greenhouse", "compound"), # Compound word
            
            # Relational concepts
            ("above", "table", "above table", "spatial"),
            ("below", "bridge", "below bridge", "spatial"),
            ("inside", "box", "inside box", "spatial"),
            ("outside", "house", "outside house", "spatial"),
        ]
    
    def extract_concept_vector(self, concept: str, context_template: str = "The concept is {}", 
                             use_sae: bool = False) -> torch.Tensor:
        """
        Extract activation vector for a given concept
        
        Args:
            concept: The concept to extract
            context_template: Template for contextualizing the concept
            use_sae: Whether to use sparse autoencoder (if available)
            
        Returns:
            Activation vector for the concept
        """
        text = context_template.format(concept)
        tokens = self.model.to_tokens(text)
        
        # Run model and get activations
        _, cache = self.model.run_with_cache(tokens)
        
        # Find the position of the concept token
        concept_tokens = self.model.to_tokens(concept, prepend_bos=False)
        if len(concept_tokens[0]) == 1:
            # Single token concept
            concept_token_id = concept_tokens[0][0].item()
            token_positions = (tokens[0] == concept_token_id).nonzero(as_tuple=True)[0]
        else:
            # Multi-token concept - use last token position
            concept_text_tokens = [self.model.to_single_token(concept)]
            token_positions = []
            for i, token_id in enumerate(tokens[0]):
                if token_id.item() in concept_text_tokens:
                    token_positions.append(i)
            token_positions = torch.tensor(token_positions) if token_positions else torch.tensor([len(tokens[0])-1])
        
        if len(token_positions) == 0:
            # Fallback to last token if concept not found
            token_pos = len(tokens[0]) - 1
        else:
            token_pos = token_positions[-1].item()  # Use last occurrence
        
        # Extract activation from specified layer
        layer_key = f"blocks.{self.layer}.hook_resid_post"
        activation = cache[layer_key][0, token_pos, :]
        
        return activation
    
    def calculate_compositionality_score(self, concept_a: str, concept_b: str, 
                                       composite: str) -> Tuple[float, float, Dict]:
        """
        Calculate compositionality score for a concept triple
        
        Args:
            concept_a: First component concept
            concept_b: Second component concept
            composite: Composite concept
            
        Returns:
            Tuple of (arithmetic_similarity, contextual_similarity, metadata)
        """
        try:
            # Extract vectors for individual concepts
            vec_a = self.extract_concept_vector(concept_a)
            vec_b = self.extract_concept_vector(concept_b)
            vec_composite = self.extract_concept_vector(composite)
            
            # Test vector arithmetic composition
            arithmetic_sum = vec_a + vec_b
            arithmetic_similarity = cosine_similarity(
                arithmetic_sum.unsqueeze(0).cpu().numpy(),
                vec_composite.unsqueeze(0).cpu().numpy()
            )[0, 0]
            
            # Test contextual composition
            contextual_composite = self.extract_concept_vector(
                composite, 
                context_template="In this context, {} is important"
            )
            contextual_similarity = cosine_similarity(
                vec_composite.unsqueeze(0).cpu().numpy(),
                contextual_composite.unsqueeze(0).cpu().numpy()
            )[0, 0]
            
            metadata = {
                "vec_a_norm": torch.norm(vec_a).item(),
                "vec_b_norm": torch.norm(vec_b).item(),
                "composite_norm": torch.norm(vec_composite).item(),
                "arithmetic_norm": torch.norm(arithmetic_sum).item(),
            }
            
            return arithmetic_similarity, contextual_similarity, metadata
            
        except Exception as e:
            logger.error(f"Error processing {concept_a} + {concept_b} = {composite}: {e}")
            return 0.0, 0.0, {"error": str(e)}
    
    def test_systematicity(self, results: List[CompositionalityResult], 
                          threshold: float = 0.7) -> Dict:
        """
        Test for systematic compositionality across concept categories
        
        Args:
            results: List of compositionality results
            threshold: Threshold for considering composition "systematic"
            
        Returns:
            Dictionary with systematicity analysis
        """
        # Group results by category
        categories = {}
        for result in results:
            category = next((cat for _, _, _, cat in self.test_cases 
                           if f"{result.concept_a} {result.concept_b}" in _ or result.composite in _), 
                          "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        systematicity_analysis = {}
        
        for category, cat_results in categories.items():
            scores = [r.arithmetic_similarity for r in cat_results]
            systematicity_analysis[category] = {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "systematic_count": sum(1 for s in scores if s >= threshold),
                "total_count": len(scores),
                "systematicity_rate": sum(1 for s in scores if s >= threshold) / len(scores),
                "examples": [(r.concept_a, r.concept_b, r.composite, r.arithmetic_similarity) 
                           for r in cat_results[:3]]
            }
        
        return systematicity_analysis
    
    def analyze_failure_modes(self, results: List[CompositionalityResult]) -> Dict:
        """
        Analyze common failure modes in compositionality
        
        Returns:
            Dictionary categorizing failure types
        """
        failure_modes = {
            "feature_suppression": [],  # Sum > composite (features cancelled out)
            "emergent_properties": [],  # Composite >> sum (new features emerged)
            "systematic_failure": [],   # Low similarity across systematic pairs
            "idiomatic_resistance": [], # Idioms that resist composition
        }
        
        for result in results:
            if result.arithmetic_similarity < 0.3:
                # Determine failure type
                if "idiom" in getattr(result, 'category', ''):
                    failure_modes["idiomatic_resistance"].append(result)
                elif result.arithmetic_similarity < 0.1:
                    failure_modes["systematic_failure"].append(result)
                else:
                    # Check if it's suppression vs emergence
                    # This would require additional analysis of vector magnitudes
                    failure_modes["feature_suppression"].append(result)
        
        return failure_modes
    
    def run_full_analysis(self) -> Dict:
        """
        Run complete compositionality analysis
        
        Returns:
            Comprehensive results dictionary
        """
        logger.info("Starting compositionality analysis...")
        
        # Run all test cases
        for concept_a, concept_b, composite, category in self.test_cases:
            try:
                arith_sim, context_sim, metadata = self.calculate_compositionality_score(
                    concept_a, concept_b, composite
                )
                
                # Calculate systematicity score (weighted average)
                systematicity_score = 0.7 * arith_sim + 0.3 * context_sim
                is_systematic = systematicity_score >= 0.6
                
                result = CompositionalityResult(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    composite=composite,
                    arithmetic_similarity=arith_sim,
                    contextual_similarity=context_sim,
                    systematicity_score=systematicity_score,
                    is_systematic=is_systematic
                )
                
                # Add category information
                result.category = category
                result.metadata = metadata
                
                self.results.append(result)
                
                logger.info(f"Processed: {concept_a} + {concept_b} = {composite}, "
                          f"Arith: {arith_sim:.3f}, Context: {context_sim:.3f}")
                          
            except Exception as e:
                logger.error(f"Failed to process {concept_a} + {concept_b} = {composite}: {e}")
        
        # Analyze results
        systematicity_analysis = self.test_systematicity(self.results)
        failure_modes = self.analyze_failure_modes(self.results)
        
        # Calculate overall statistics
        all_arith_scores = [r.arithmetic_similarity for r in self.results]
        all_system_scores = [r.systematicity_score for r in self.results]
        
        overall_stats = {
            "total_tests": len(self.results),
            "mean_arithmetic_similarity": np.mean(all_arith_scores),
            "std_arithmetic_similarity": np.std(all_arith_scores),
            "mean_systematicity": np.mean(all_system_scores),
            "systematic_rate": sum(1 for r in self.results if r.is_systematic) / len(self.results),
            "perfect_composition_rate": sum(1 for s in all_arith_scores if s >= 0.9) / len(all_arith_scores),
        }
        
        return {
            "results": self.results,
            "systematicity_analysis": systematicity_analysis,
            "failure_modes": failure_modes,
            "overall_stats": overall_stats,
            "model_info": {
                "name": self.model_name,
                "layer": self.layer,
                "device": self.device
            }
        }
    
    def create_visualizations(self, analysis_results: Dict) -> None:
        """
        Create comprehensive visualizations of compositionality results
        """
        results = analysis_results["results"]
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Overall compositionality distribution
        ax1 = plt.subplot(3, 3, 1)
        arith_scores = [r.arithmetic_similarity for r in results]
        plt.hist(arith_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(arith_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(arith_scores):.3f}')
        plt.axvline(0.7, color='green', linestyle='--', label='Systematicity Threshold')
        plt.xlabel('Arithmetic Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Compositionality Scores')
        plt.legend()
        
        # 2. Systematicity by category
        ax2 = plt.subplot(3, 3, 2)
        category_data = {}
        for result in results:
            category = getattr(result, 'category', 'unknown')
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(result.arithmetic_similarity)
        
        categories = list(category_data.keys())
        means = [np.mean(category_data[cat]) for cat in categories]
        stds = [np.std(category_data[cat]) for cat in categories]
        
        bars = plt.bar(range(len(categories)), means, yerr=stds, capsize=5, 
                      color='lightcoral', alpha=0.8)
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        plt.ylabel('Mean Arithmetic Similarity')
        plt.title('Compositionality by Concept Category')
        plt.axhline(0.7, color='green', linestyle='--', alpha=0.7, label='Systematicity Threshold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.01,
                    f'{means[i]:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Arithmetic vs Contextual similarity scatter
        ax3 = plt.subplot(3, 3, 3)
        arith_sims = [r.arithmetic_similarity for r in results]
        context_sims = [r.contextual_similarity for r in results]
        colors = ['red' if 'idiom' in getattr(r, 'category', '') or 'compound' in getattr(r, 'category', '')
                 else 'blue' for r in results]
        
        plt.scatter(arith_sims, context_sims, c=colors, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Correlation')
        plt.xlabel('Arithmetic Similarity')
        plt.ylabel('Contextual Similarity')
        plt.title('Arithmetic vs Contextual Composition')
        correlation, p_value = pearsonr(arith_sims, context_sims)
        plt.text(0.05, 0.95, f'r = {correlation:.3f}, p = {p_value:.3f}', 
                transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # 4. Systematicity heatmap
        ax4 = plt.subplot(3, 3, 4)
        # Create matrix of systematic relationships
        concept_pairs = {}
        for result in results:
            pair_key = f"{result.concept_a}-{result.concept_b}"
            concept_pairs[pair_key] = result.arithmetic_similarity
        
        # Group by patterns (e.g., color patterns, size patterns)
        pattern_matrix = self._create_pattern_matrix(results)
        if pattern_matrix is not None:
            sns.heatmap(pattern_matrix, annot=True, cmap='RdYlBu_r', center=0.5,
                       xticklabels=True, yticklabels=True, cbar_kws={'label': 'Similarity'})
            plt.title('Systematic Composition Patterns')
            plt.ylabel('Adjective Type')
            plt.xlabel('Noun Type')
        
        # 5. Error analysis
        ax5 = plt.subplot(3, 3, 5)
        error_categories = ['High Comp. (>0.7)', 'Med Comp. (0.3-0.7)', 'Low Comp. (<0.3)', 'Idioms']
        error_counts = [
            sum(1 for r in results if r.arithmetic_similarity > 0.7),
            sum(1 for r in results if 0.3 <= r.arithmetic_similarity <= 0.7),
            sum(1 for r in results if r.arithmetic_similarity < 0.3 and 'idiom' not in getattr(r, 'category', '')),
            sum(1 for r in results if 'idiom' in getattr(r, 'category', ''))
        ]
        
        colors = ['green', 'yellow', 'red', 'purple']
        wedges, texts, autotexts = plt.pie(error_counts, labels=error_categories, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        plt.title('Composition Success/Failure Distribution')
        
        # 6. Model scaling analysis (if multiple models)
        ax6 = plt.subplot(3, 3, 6)
        # Placeholder for scaling analysis
        plt.text(0.5, 0.5, 'Model Scaling Analysis\n(Requires multiple model sizes)', 
                ha='center', va='center', transform=ax6.transAxes,
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        plt.title('Compositionality vs Model Scale')
        
        # 7. Top successful compositions
        ax7 = plt.subplot(3, 3, 7)
        top_results = sorted(results, key=lambda x: x.arithmetic_similarity, reverse=True)[:10]
        top_labels = [f"{r.concept_a}+{r.concept_b}" for r in top_results]
        top_scores = [r.arithmetic_similarity for r in top_results]
        
        plt.barh(range(len(top_labels)), top_scores, color='lightgreen', alpha=0.8)
        plt.yticks(range(len(top_labels)), top_labels)
        plt.xlabel('Arithmetic Similarity')
        plt.title('Top 10 Compositional Successes')
        plt.gca().invert_yaxis()
        
        # 8. Bottom/failed compositions
        ax8 = plt.subplot(3, 3, 8)
        bottom_results = sorted(results, key=lambda x: x.arithmetic_similarity)[:10]
        bottom_labels = [f"{r.concept_a}+{r.concept_b}" for r in bottom_results]
        bottom_scores = [r.arithmetic_similarity for r in bottom_results]
        
        plt.barh(range(len(bottom_labels)), bottom_scores, color='lightcoral', alpha=0.8)
        plt.yticks(range(len(bottom_labels)), bottom_labels)
        plt.xlabel('Arithmetic Similarity')
        plt.title('Bottom 10 Compositional Failures')
        plt.gca().invert_yaxis()
        
        # 9. Philosophical implications summary
        ax9 = plt.subplot(3, 3, 9)
        overall_stats = analysis_results["overall_stats"]
        
        # Create text summary
        summary_text = f"""
        Philosophical Implications:
        
        Overall Systematicity: {overall_stats['systematic_rate']:.1%}
        Mean Composition Score: {overall_stats['mean_arithmetic_similarity']:.3f}
        Perfect Compositions: {overall_stats['perfect_composition_rate']:.1%}
        
        Interpretation:
        {'Strong evidence for genuine concepts' if overall_stats['systematic_rate'] > 0.7 
         else 'Moderate evidence for concept-like patterns' if overall_stats['systematic_rate'] > 0.4
         else 'Weak evidence - likely statistical patterns'}
        
        Safety Implications:
        {'High confidence in interpretability' if overall_stats['systematic_rate'] > 0.7
         else 'Moderate confidence - verify critical features' if overall_stats['systematic_rate'] > 0.4  
         else 'Low confidence - interpretability may be misleading'}
        """
        
        plt.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8))
        plt.axis('off')
        plt.title('Philosophical & Safety Assessment')
        
        plt.tight_layout()
        plt.savefig(f'compositionality_analysis_{self.model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def _create_pattern_matrix(self, results: List[CompositionalityResult]) -> Optional[np.ndarray]:
        """
        Create a matrix showing systematic patterns in composition
        """
        try:
            # Extract adjective and noun types
            adjectives = set()
            nouns = set()
            
            for result in results:
                if hasattr(result, 'category') and 'object' in result.category:
                    adjectives.add(result.concept_a)
                    nouns.add(result.concept_b)
            
            if len(adjectives) < 2 or len(nouns) < 2:
                return None
            
            adj_list = sorted(list(adjectives))
            noun_list = sorted(list(nouns))
            
            matrix = np.zeros((len(adj_list), len(noun_list)))
            
            for i, adj in enumerate(adj_list):
                for j, noun in enumerate(noun_list):
                    # Find matching result
                    matching_results = [r for r in results 
                                      if r.concept_a == adj and r.concept_b == noun]
                    if matching_results:
                        matrix[i, j] = matching_results[0].arithmetic_similarity
                    else:
                        matrix[i, j] = np.nan
            
            return matrix
            
        except Exception as e:
            logger.error(f"Error creating pattern matrix: {e}")
            return None
    
    def generate_report(self, analysis_results: Dict) -> str:
        """
        Generate a comprehensive text report of findings
        """
        results = analysis_results["results"]
        stats = analysis_results["overall_stats"]
        systematicity = analysis_results["systematicity_analysis"]
        failures = analysis_results["failure_modes"]
        
        report = f"""
COMPOSITIONALITY SYSTEMATICITY ANALYSIS REPORT
Model: {self.model_name} (Layer {self.layer})
================================================

EXECUTIVE SUMMARY
-----------------
Total Concepts Tested: {stats['total_tests']}
Mean Arithmetic Similarity: {stats['mean_arithmetic_similarity']:.3f} ± {stats['std_arithmetic_similarity']:.3f}
Systematic Composition Rate: {stats['systematic_rate']:.1%}
Perfect Composition Rate: {stats['perfect_composition_rate']:.1%}

PHILOSOPHICAL ASSESSMENT
------------------------
"""
        
        # Philosophical interpretation
        if stats['systematic_rate'] > 0.7:
            interpretation = "STRONG SUPPORT for genuine conceptual content"
            safety_implication = "HIGH CONFIDENCE in interpretability methods"
        elif stats['systematic_rate'] > 0.4:
            interpretation = "MODERATE SUPPORT for concept-like representations"
            safety_implication = "MODERATE CONFIDENCE - verify critical features"
        else:
            interpretation = "WEAK SUPPORT - likely statistical patterns only"
            safety_implication = "LOW CONFIDENCE - interpretability may be misleading"
        
        report += f"""
Theory Support: {interpretation}
Safety Implication: {safety_implication}

DETAILED RESULTS BY CATEGORY
----------------------------
"""
        
        for category, data in systematicity.items():
            report += f"""
{category.upper()}:
  Mean Score: {data['mean_score']:.3f} ± {data['std_score']:.3f}
  Systematic Rate: {data['systematicity_rate']:.1%} ({data['systematic_count']}/{data['total_count']})
  Examples: {', '.join([f"{ex[0]}+{ex[1]}→{ex[2]}({ex[3]:.2f})" for ex in data['examples']])}
"""
        
        report += f"""
FAILURE MODE ANALYSIS
---------------------
Feature Suppression Cases: {len(failures.get('feature_suppression', []))}
Emergent Properties Cases: {len(failures.get('emergent_properties', []))}
Systematic Failures: {len(failures.get('systematic_failure', []))}
Idiomatic Resistance: {len(failures.get('idiomatic_resistance', []))}

IMPLICATIONS FOR AI SAFETY
---------------------------
"""
        
        if stats['systematic_rate'] < 0.5:
            report += """
⚠️  CRITICAL FINDING: Low systematicity suggests current interpretability methods
may not track genuine conceptual content. This has major implications:

1. Alignment Verification: Cannot reliably verify AI "beliefs" via feature inspection
2. Deception Detection: May miss sophisticated deceptive reasoning
3. Value Learning: Uncertain whether extracted "values" represent genuine preferences

RECOMMENDATIONS:
- Develop more philosophically-grounded interpretability metrics
- Focus on behavioral rather than representational alignment verification
- Increase caution when using mechanistic interpretability for safety-critical applications
"""
        else:
            report += """
✅ POSITIVE FINDING: Moderate-to-strong systematicity suggests interpretability methods
may track genuine conceptual structure. However, remain cautious:

1. Verify systematicity for safety-critical concepts specifically
2. Test compositionality at scale and across domains
3. Validate findings with causal intervention experiments

RECOMMENDATIONS:
- Expand testing to more complex conceptual relationships
- Develop systematic compositionality as a standard interpretability metric
- Use compositional structure as evidence for genuine understanding
"""
        
        report += f"""

METHODOLOGICAL NOTES
--------------------
- Results based on {len(results)} concept triples
- Similarity threshold for systematicity: 0.6
- Layer analyzed: {self.layer} (residual stream)
- Context template: "The concept is {{}}"

REPRODUCIBILITY
---------------
All code and data available for replication. Key parameters:
- Model: {self.model_name}
- Layer: {self.layer}  
- Device: {self.device}
- Similarity metric: Cosine similarity
- Composition method: Vector addition
"""
        
        return report

    def hf_generate(self, prompt: str) -> str:
        """
        Generate text using the HuggingFace Pythia model.
        """
        if not self.use_hf:
            raise RuntimeError("HuggingFace backend not enabled.")
        inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)
        tokens = self.hf_model.generate(**inputs, max_new_tokens=20)
        return self.hf_tokenizer.decode(tokens[0], skip_special_tokens=True)

def run_compositionality_experiment(model_names: List[str] = ["pythia-70m"], 
                                  layer: int = -6) -> Dict:
    """
    Run compositionality experiments across multiple models
    
    Args:
        model_names: List of model names to test
        layer: Layer to extract activations from
        
    Returns:
        Dictionary with results for each model
    """
    all_results = {}
    
    for model_name in model_names:
        logger.info(f"Running compositionality analysis for {model_name}")
        
        analyzer = CompositionalityAnalyzer(model_name=model_name, layer=layer)
        analysis_results = analyzer.run_full_analysis()
        
        # Generate visualizations
        analyzer.create_visualizations(analysis_results)
        
        # Generate report
        report = analyzer.generate_report(analysis_results)
        
        all_results[model_name] = {
            "analysis": analysis_results,
            "report": report,
            "analyzer": analyzer
        }
        
        # Save report to file
        with open(f'compositionality_report_{model_name}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Completed analysis for {model_name}")
    
    return all_results

# Example usage and testing
if __name__ == "__main__":
    # Run analysis on single model
    print("Testing Compositionality Systematicity Analysis...")
    print("=" * 50)
    
    # Test with smaller model first
    try:
        results = run_compositionality_experiment(
            model_names=["pythia-70m"], 
            layer=-6
        )
        
        model_results = results["pythia-70m"]
        stats = model_results["analysis"]["overall_stats"]
        
        print(f"\nQuick Results Summary:")
        print(f"Mean Compositionality: {stats['mean_arithmetic_similarity']:.3f}")
        print(f"Systematic Rate: {stats['systematic_rate']:.1%}")
        print(f"Perfect Compositions: {stats['perfect_composition_rate']:.1%}")
        
        # Print sample results
        print(f"\nSample Successful Compositions:")
        successful = [r for r in model_results["analysis"]["results"] if r.arithmetic_similarity > 0.7]
        for result in successful[:5]:
            print(f"  {result.concept_a} + {result.concept_b} -> {result.composite}: {result.arithmetic_similarity:.3f}")
        
        print(f"\nSample Failed Compositions:")
        failed = [r for r in model_results["analysis"]["results"] if r.arithmetic_similarity < 0.3]
        for result in failed[:5]:
            print(f"  {result.concept_a} + {result.concept_b} -> {result.composite}: {result.arithmetic_similarity:.3f}")
            
        print(f"\nPhilosophical Interpretation:")
        if stats['systematic_rate'] > 0.7:
            print("  ✅ Strong evidence for genuine conceptual compositionality")
            print("  ✅ AI representations may track human-like mental content")
        elif stats['systematic_rate'] > 0.4:
            print("  ⚠️  Moderate evidence - partially compositional representations")
            print("  ⚠️  AI concepts are statistically useful but may lack full systematicity")
        else:
            print("  ❌ Weak evidence for genuine concepts")
            print("  ❌ Representations likely reflect statistical patterns, not true compositionality")
            
        print(f"\nSafety Implications:")
        if stats['systematic_rate'] > 0.7:
            print("  → Can use interpretability with higher confidence for alignment")
            print("  → Features may reliably track genuine AI beliefs/intentions")
        elif stats['systematic_rate'] > 0.4:
            print("  → Use interpretability with caution - verify critical features")
            print("  → May need additional validation for safety-critical applications")
        else:
            print("  → Low confidence in interpretability for safety applications")
            print("  → Risk of misaligned systems due to measurement illusions")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("This is a template - actual implementation requires:")
        print("1. TransformerLens library installation")
        print("2. Pythia model downloads")  
        print("3. Sparse autoencoder implementation")
        print("4. GPU/sufficient compute resources")


class AdvancedCompositionalityTests:
    """
    Extended tests for deeper philosophical analysis
    """
    
    def __init__(self, base_analyzer: CompositionalityAnalyzer):
        self.analyzer = base_analyzer
        self.model = base_analyzer.model
        
    def test_fodor_systematicity(self) -> Dict:
        """
        Test Fodor & Pylyshyn's systematicity requirement:
        If you can think "red car" you must be able to think "red truck", "blue car", etc.
        """
        systematicity_tests = [
            # Color-object systematicity
            {
                "pattern": "color_object",
                "test_cases": [
                    ("red", "car"), ("red", "truck"), ("red", "house"),
                    ("blue", "car"), ("blue", "truck"), ("blue", "house"),
                    ("green", "car"), ("green", "truck"), ("green", "house")
                ]
            },
            # Size-object systematicity  
            {
                "pattern": "size_object",
                "test_cases": [
                    ("big", "dog"), ("big", "cat"), ("big", "bird"),
                    ("small", "dog"), ("small", "cat"), ("small", "bird"),
                    ("tiny", "dog"), ("tiny", "cat"), ("tiny", "bird")
                ]
            },
            # Action-object systematicity
            {
                "pattern": "action_object", 
                "test_cases": [
                    ("read", "book"), ("read", "article"), ("read", "sign"),
                    ("write", "book"), ("write", "article"), ("write", "sign"),
                    ("study", "book"), ("study", "article"), ("study", "sign")
                ]
            }
        ]
        
        systematicity_results = {}
        
        for pattern_test in systematicity_tests:
            pattern = pattern_test["pattern"]
            cases = pattern_test["test_cases"]
            
            # Calculate compositionality for each case
            pattern_scores = []
            for concept_a, concept_b in cases:
                composite = f"{concept_a} {concept_b}"
                arith_sim, context_sim, metadata = self.analyzer.calculate_compositionality_score(
                    concept_a, concept_b, composite
                )
                pattern_scores.append({
                    "concepts": (concept_a, concept_b),
                    "arithmetic_similarity": arith_sim,
                    "contextual_similarity": context_sim
                })
            
            # Analyze systematicity within pattern
            scores = [s["arithmetic_similarity"] for s in pattern_scores]
            systematicity_results[pattern] = {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "consistency": 1.0 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0.0,
                "all_scores": pattern_scores,
                "fodor_prediction": "High consistency across substitution instances"
            }
        
        return systematicity_results
    
    def test_prototype_theory(self) -> Dict:
        """
        Test prototype theory predictions:
        Composition should work better for typical than atypical instances
        """
        prototype_tests = [
            {
                "category": "birds",
                "typical": [("small", "bird"), ("flying", "bird"), ("singing", "bird")],
                "atypical": [("large", "bird"), ("flightless", "bird"), ("silent", "bird")]
            },
            {
                "category": "vehicles", 
                "typical": [("fast", "car"), ("four-wheel", "vehicle"), ("engine", "car")],
                "atypical": [("slow", "car"), ("three-wheel", "vehicle"), ("pedal", "car")]
            },
            {
                "category": "furniture",
                "typical": [("wooden", "chair"), ("comfortable", "sofa"), ("sturdy", "table")],
                "atypical": [("inflatable", "chair"), ("uncomfortable", "sofa"), ("wobbly", "table")]
            }
        ]
        
        prototype_results = {}
        
        for test in prototype_tests:
            category = test["category"]
            typical_scores = []
            atypical_scores = []
            
            # Test typical instances
            for concept_a, concept_b in test["typical"]:
                composite = f"{concept_a} {concept_b}"
                arith_sim, _, _ = self.analyzer.calculate_compositionality_score(
                    concept_a, concept_b, composite
                )
                typical_scores.append(arith_sim)
            
            # Test atypical instances
            for concept_a, concept_b in test["atypical"]:
                composite = f"{concept_a} {concept_b}"
                arith_sim, _, _ = self.analyzer.calculate_compositionality_score(
                    concept_a, concept_b, composite
                )
                atypical_scores.append(arith_sim)
            
            # Statistical comparison
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(typical_scores, atypical_scores)
            
            prototype_results[category] = {
                "typical_mean": np.mean(typical_scores),
                "atypical_mean": np.mean(atypical_scores),
                "typicality_effect": np.mean(typical_scores) - np.mean(atypical_scores),
                "t_statistic": t_stat,
                "p_value": p_value,
                "supports_prototype_theory": np.mean(typical_scores) > np.mean(atypical_scores),
                "typical_scores": typical_scores,
                "atypical_scores": atypical_scores
            }
        
        return prototype_results
    
    def test_emergent_properties(self) -> Dict:
        """
        Test for emergent properties that aren't present in individual components
        """
        emergent_tests = [
            # Cases where composition should create new meaning
            ("hot", "dog", "hot dog", "food"),  # Not just hot + dog
            ("green", "house", "greenhouse", "building"),  # Special building type
            ("fire", "truck", "fire truck", "vehicle"),  # Specialized vehicle
            ("ice", "cream", "ice cream", "food"),  # Dessert, not frozen cream
            
            # Cases where composition should be purely additive
            ("red", "ball", "red ball", "simple"),  # Just red + ball
            ("big", "table", "big table", "simple"),  # Just big + table
            ("old", "book", "old book", "simple"),  # Just old + book
        ]
        
        emergent_results = []
        
        for concept_a, concept_b, composite, expected_type in emergent_tests:
            # Get individual concept vectors
            vec_a = self.analyzer.extract_concept_vector(concept_a)
            vec_b = self.analyzer.extract_concept_vector(concept_b)
            vec_composite = self.analyzer.extract_concept_vector(composite)
            
            # Test different composition methods
            arithmetic_sum = vec_a + vec_b
            
            # Calculate similarities
            arith_similarity = cosine_similarity(
                arithmetic_sum.unsqueeze(0).cpu().numpy(),
                vec_composite.unsqueeze(0).cpu().numpy()
            )[0, 0]
            
            # Test for emergent properties by comparing magnitudes
            sum_magnitude = torch.norm(arithmetic_sum).item()
            composite_magnitude = torch.norm(vec_composite).item()
            magnitude_ratio = composite_magnitude / sum_magnitude
            
            # Test semantic distance from components
            a_to_composite = cosine_similarity(
                vec_a.unsqueeze(0).cpu().numpy(),
                vec_composite.unsqueeze(0).cpu().numpy()
            )[0, 0]
            
            b_to_composite = cosine_similarity(
                vec_b.unsqueeze(0).cpu().numpy(), 
                vec_composite.unsqueeze(0).cpu().numpy()
            )[0, 0]
            
            emergent_results.append({
                "concepts": (concept_a, concept_b, composite),
                "expected_type": expected_type,
                "arithmetic_similarity": arith_similarity,
                "magnitude_ratio": magnitude_ratio,
                "a_to_composite_sim": a_to_composite,
                "b_to_composite_sim": b_to_composite,
                "emergence_score": 1.0 - arith_similarity,  # Higher when less additive
                "is_emergent": arith_similarity < 0.5 and expected_type != "simple"
            })
        
        return {
            "emergent_cases": [r for r in emergent_results if r["is_emergent"]],
            "additive_cases": [r for r in emergent_results if not r["is_emergent"]],
            "all_results": emergent_results,
            "emergence_rate": sum(1 for r in emergent_results if r["is_emergent"]) / len(emergent_results)
        }

def comparative_model_analysis(model_names: List[str] = ["pythia-70m", "pythia-160m"]) -> Dict:
    """
    Compare compositionality across different model sizes
    """
    comparative_results = {}
    
    for model_name in model_names:
        analyzer = CompositionalityAnalyzer(model_name=model_name)
        analysis = analyzer.run_full_analysis()
        
        comparative_results[model_name] = {
            "systematic_rate": analysis["overall_stats"]["systematic_rate"],
            "mean_similarity": analysis["overall_stats"]["mean_arithmetic_similarity"],
            "perfect_rate": analysis["overall_stats"]["perfect_composition_rate"],
            "model_size": model_name.split("-")[1]  # Extract size (70m, 160m, etc.)
        }
    
    # Analyze scaling trends
    model_sizes = [comparative_results[m]["model_size"] for m in model_names]
    systematic_rates = [comparative_results[m]["systematic_rate"] for m in model_names]
    
    # Calculate correlation between model size and systematicity
    size_values = [int(s.replace('m', '')) for s in model_sizes]
    correlation, p_value = pearsonr(size_values, systematic_rates)
    
    comparative_results["scaling_analysis"] = {
        "size_systematicity_correlation": correlation,
        "p_value": p_value,
        "interpretation": "Larger models show more systematicity" if correlation > 0.3 
                         else "No clear scaling effect on compositionality"
    }
    
    return comparative_results

def philosophical_validation_suite() -> Dict:
    """
    Complete philosophical validation of compositionality findings
    """
    print("Running Philosophical Validation Suite...")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = CompositionalityAnalyzer(model_name="pythia-70m")
    
    # Run basic compositionality analysis
    basic_analysis = analyzer.run_full_analysis()
    
    # Run advanced tests
    advanced_tester = AdvancedCompositionalityTests(analyzer)
    
    # Test Fodor systematicity
    print("Testing Fodor systematicity...")
    systematicity_results = advanced_tester.test_fodor_systematicity()
    
    # Test prototype theory
    print("Testing prototype theory...")
    prototype_results = advanced_tester.test_prototype_theory()
    
    # Test emergent properties
    print("Testing emergent properties...")
    emergence_results = advanced_tester.test_emergent_properties()
    
    # Compile comprehensive philosophical assessment
    philosophical_assessment = {
        "basic_compositionality": basic_analysis,
        "fodor_systematicity": systematicity_results,
        "prototype_effects": prototype_results,
        "emergent_properties": emergence_results,
    }
    
    # Generate philosophical interpretation
    overall_systematic_rate = basic_analysis["overall_stats"]["systematic_rate"]
    fodor_consistency = np.mean([data["consistency"] for data in systematicity_results.values()])
    prototype_effect_strength = np.mean([data["typicality_effect"] for data in prototype_results.values()])
    emergence_rate = emergence_results["emergence_rate"]
    
    philosophical_interpretation = {
        "classical_theory_support": overall_systematic_rate > 0.7 and fodor_consistency > 0.8,
        "prototype_theory_support": prototype_effect_strength > 0.1,
        "connectionist_patterns": emergence_rate > 0.3,
        "safety_confidence_level": "high" if overall_systematic_rate > 0.7 
                                  else "medium" if overall_systematic_rate > 0.4 
                                  else "low"
    }
    
    return {
        "philosophical_assessment": philosophical_assessment,
        "interpretation": philosophical_interpretation,
        "recommendations": generate_safety_recommendations(philosophical_interpretation)
    }

def generate_safety_recommendations(interpretation: Dict) -> List[str]:
    """
    Generate specific safety recommendations based on philosophical findings
    """
    recommendations = []
    
    if interpretation["classical_theory_support"]:
        recommendations.extend([
            "✅ High confidence: Use interpretability for alignment verification",
            "✅ Features likely track genuine conceptual content", 
            "✅ Systematic composition enables reliable belief editing",
            "→ Expand interpretability to safety-critical reasoning patterns"
        ])
    elif interpretation["safety_confidence_level"] == "medium":
        recommendations.extend([
            "⚠️  Medium confidence: Verify interpretability for each critical application",
            "⚠️  Test compositionality for domain-specific concepts before deployment",
            "⚠️  Use multiple validation methods beyond feature inspection",
            "→ Develop hybrid approaches combining interpretability with behavioral testing"
        ])
    else:
        recommendations.extend([
            "❌ Low confidence: Avoid relying solely on interpretability for safety",
            "❌ Current methods may create illusion of understanding",
            "❌ High risk of missing sophisticated misalignment",
            "→ Focus on behavioral alignment verification instead of representational"
        ])
    
    if interpretation["prototype_theory_support"]:
        recommendations.append("📋 Account for typicality effects in safety testing")
    
    if interpretation["connectionist_patterns"]:
        recommendations.append("🔬 Investigate emergent properties in safety-critical reasoning")
    
    return recommendations

# Utility functions for statistical analysis
def calculate_effect_sizes(group1: List[float], group2: List[float]) -> Dict:
    """Calculate Cohen's d and other effect size measures"""
    from scipy import stats
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Cohen's d
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    return {
        "cohens_d": cohens_d,
        "effect_size_interpretation": "large" if abs(cohens_d) > 0.8 
                                    else "medium" if abs(cohens_d) > 0.5 
                                    else "small",
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }

def save_results_for_replication(analysis_results: Dict, filename: str = "compositionality_replication_data"):
    """
    Save all results in formats suitable for replication and further analysis
    """
    import json
    import pickle
    
    # Prepare data for JSON serialization
    json_data = {
        "model_info": analysis_results["model_info"],
        "overall_stats": analysis_results["overall_stats"],
        "systematicity_by_category": analysis_results["systematicity_analysis"]
    }
    
    # Save detailed results for replication
    detailed_results = []
    for result in analysis_results["results"]:
        detailed_results.append({
            "concept_a": result.concept_a,
            "concept_b": result.concept_b, 
            "composite": result.composite,
            "arithmetic_similarity": result.arithmetic_similarity,
            "contextual_similarity": result.contextual_similarity,
            "systematicity_score": result.systematicity_score,
            "is_systematic": result.is_systematic,
            "category": getattr(result, 'category', 'unknown')
        })
