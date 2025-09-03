import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

class ExternalismAnalyzer:
    """
    Framework for testing externalist predictions using Twin Earth dataset.
    
    Key philosophical predictions:
    1. If externalism is correct: Earth and Twin Earth concept vectors should differ
    2. If internalism is correct: Concept vectors should be identical
    3. Surface controls: Identical surface properties shouldn't affect core referent tracking
    4. Linguistic controls: Different syntax shouldn't affect semantic content
    """
    
    def __init__(self, model_earth, model_twin, tokenizer):
        self.model_earth = model_earth  # Model trained on Earth data
        self.model_twin = model_twin    # Model trained on Twin Earth data
        self.tokenizer = tokenizer
        
    def extract_concept_vector(self, model, concept: str, contexts: List[str], 
                         layer: int = -6, method: str = "mean_activation"):
        """
        Extract concept representation from model using various methods.
        """
        concept_vectors = []
        concept_token_ids = self.tokenizer.encode(concept, add_special_tokens=False)
        for context in contexts:
            tokens = self.tokenizer(context, return_tensors="pt")
            input_ids = tokens["input_ids"][0]
            found = False
            for i in range(len(input_ids) - len(concept_token_ids) + 1):
                if torch.equal(input_ids[i:i+len(concept_token_ids)], torch.tensor(concept_token_ids)):
                    with torch.no_grad():
                        outputs = model(**tokens, output_hidden_states=True)
                        hidden_states = outputs.hidden_states[layer]  # [batch, seq, hidden]
                        concept_vector = hidden_states[0, i:i+len(concept_token_ids), :].mean(dim=0).cpu().numpy()
                        concept_vectors.append(concept_vector)
                    found = True
                    break  # Only use first occurrence per context
            if not found:
                print(f"[WARN] Concept '{concept}' not found in context: {context[:60]}...")

        if not concept_vectors:
            print(f"[ERROR] No valid concept vectors found for '{concept}'. Returning NaN vector.")
            # Return a vector of NaNs with the correct hidden size
            # Try to get hidden size from model config
            try:
                hidden_size = model.config.hidden_size
            except AttributeError:
                hidden_size = 768  # fallback
            return np.full((hidden_size,), np.nan)

        # Aggregate vectors
        if method == "mean_activation":
            return np.mean(concept_vectors, axis=0)
        elif method == "max_activation":
            return np.max(concept_vectors, axis=0)
        else:
            return np.mean(concept_vectors, axis=0)
    
    def test_externalism_hypothesis(self, concept: str, earth_contexts: List[str], 
                                  twin_contexts: List[str]) -> Dict:
        """
        Core externalism test: Compare concept vectors between Earth and Twin Earth models.
        
        Philosophical prediction:
        - Strong externalism: Low similarity (< 0.3)
        - Weak externalism: Medium similarity (0.3-0.7) 
        - Internalism: High similarity (> 0.7)
        """
        
        # Extract concept vectors from both models
        earth_vector = self.extract_concept_vector(
            self.model_earth, concept, earth_contexts
        )
        twin_vector = self.extract_concept_vector(
            self.model_twin, concept, twin_contexts  
        )
        
        if np.isnan(earth_vector).any() or np.isnan(twin_vector).any():
            print(f"[SKIP] Skipping concept '{concept}' due to missing vectors.")
            return {
                "concept": concept,
                "vector_similarity": np.nan,
                "earth_vector_norm": np.nan,
                "twin_vector_norm": np.nan,
                "philosophical_interpretation": "No valid vector extracted",
                "supports_externalism": False,
            }

        # Compute similarity
        similarity = cosine_similarity(
            earth_vector.reshape(1, -1), 
            twin_vector.reshape(1, -1)
        )[0, 0]
        
        # Statistical analysis
        results = {
            "concept": concept,
            "vector_similarity": float(similarity),
            "earth_vector_norm": float(np.linalg.norm(earth_vector)),
            "twin_vector_norm": float(np.linalg.norm(twin_vector)),
            "philosophical_interpretation": self._interpret_similarity(similarity),
            "supports_externalism": similarity < 0.5,  # Threshold for externalist support
        }
        
        return results
    
    def _interpret_similarity(self, similarity: float) -> str:
        """Interpret similarity score in philosophical terms"""
        if similarity > 0.8:
            return "Strong evidence for internalism (concepts identical across referents)"
        elif similarity > 0.6:
            return "Moderate evidence for internalism (mostly similar concepts)"
        elif similarity > 0.4:
            return "Weak evidence for externalism (some referent sensitivity)"
        elif similarity > 0.2:
            return "Moderate evidence for externalism (clear referent dependence)"
        else:
            return "Strong evidence for externalism (concepts track external referents)"
    
    def test_surface_vs_deep_properties(self, concept: str, surface_contexts: List[str],deep_contexts: List[str]) -> Dict:
        """ Test whether AI tracks deep essential properties vs. surface appearances. """
        # Extract vectors for surface vs. deep property contexts
        earth_surface = self.extract_concept_vector(
            self.model_earth, concept, surface_contexts
        )
        earth_deep = self.extract_concept_vector(
            self.model_earth, concept, deep_contexts
        )
        twin_surface = self.extract_concept_vector(
            self.model_twin, concept, surface_contexts
        )
        twin_deep = self.extract_concept_vector(
            self.model_twin, concept, deep_contexts
        )

        # Check for NaNs and handle gracefully
        if (
            np.isnan(earth_surface).any() or np.isnan(earth_deep).any() or
            np.isnan(twin_surface).any() or np.isnan(twin_deep).any()
        ):
            print(f"[SKIP] Skipping surface/deep test for '{concept}' due to missing vectors.")
            return {
                "concept": concept,
                "surface_property_similarity": np.nan,
                "deep_property_similarity": np.nan,
                "deep_vs_surface_difference": np.nan,
                "tracks_essential_properties": False,
                "philosophical_interpretation": "No valid vector extracted"
            }

        # Compare similarities
        surface_similarity = cosine_similarity(
            earth_surface.reshape(1, -1), twin_surface.reshape(1, -1)
        )[0, 0]
        deep_similarity = cosine_similarity(
            earth_deep.reshape(1, -1), twin_deep.reshape(1, -1)
        )[0, 0]

        return {
            "concept": concept,
            "surface_property_similarity": float(surface_similarity),
            "deep_property_similarity": float(deep_similarity),
            "deep_vs_surface_difference": float(surface_similarity - deep_similarity),
            "tracks_essential_properties": deep_similarity < surface_similarity,
            "philosophical_interpretation": (
                "Tracks essential properties (externalist)" if deep_similarity < surface_similarity
                else "Tracks surface properties (internalist or statistical)"
            )
        }
        
    def test_linguistic_vs_semantic_content(self, concept: str,
                                        earth_formal: List[str], earth_informal: List[str],
                                        twin_formal: List[str], twin_informal: List[str]) -> Dict:
        """
        Test whether AI tracks semantic content vs. linguistic surface patterns.
        Philosophical prediction: Genuine semantic content should be invariant 
        to linguistic style changes.
        """
        # Extract vectors for different linguistic styles
        earth_formal_vec = self.extract_concept_vector(self.model_earth, concept, earth_formal)
        earth_informal_vec = self.extract_concept_vector(self.model_earth, concept, earth_informal)
        twin_formal_vec = self.extract_concept_vector(self.model_twin, concept, twin_formal)
        twin_informal_vec = self.extract_concept_vector(self.model_twin, concept, twin_informal)

        # Check for NaNs and handle gracefully
        if (
            np.isnan(earth_formal_vec).any() or np.isnan(earth_informal_vec).any() or
            np.isnan(twin_formal_vec).any() or np.isnan(twin_informal_vec).any()
        ):
            print(f"[SKIP] Skipping linguistic/semantic test for '{concept}' due to missing vectors.")
            return {
                "concept": concept,
                "earth_linguistic_stability": np.nan,
                "twin_linguistic_stability": np.nan,
                "formal_cross_similarity": np.nan,
                "informal_cross_similarity": np.nan,
                "tracks_semantic_content": False,
                "consistent_across_styles": False,
                "philosophical_interpretation": "No valid vector extracted"
            }

        # Within-model linguistic stability (should be high for genuine concepts)
        earth_linguistic_stability = cosine_similarity(
            earth_formal_vec.reshape(1, -1), earth_informal_vec.reshape(1, -1)
        )[0, 0]
        twin_linguistic_stability = cosine_similarity(
            twin_formal_vec.reshape(1, -1), twin_informal_vec.reshape(1, -1)
        )[0, 0]
        # Cross-model semantic difference (should be low for internalism, high for externalism)
        formal_cross_similarity = cosine_similarity(
            earth_formal_vec.reshape(1, -1), twin_formal_vec.reshape(1, -1)
        )[0, 0]
        informal_cross_similarity = cosine_similarity(
            earth_informal_vec.reshape(1, -1), twin_informal_vec.reshape(1, -1)
        )[0, 0]

        return {
            "concept": concept,
            "earth_linguistic_stability": float(earth_linguistic_stability),
            "twin_linguistic_stability": float(twin_linguistic_stability),
            "formal_cross_similarity": float(formal_cross_similarity),
            "informal_cross_similarity": float(informal_cross_similarity),
            "tracks_semantic_content": (
                earth_linguistic_stability > 0.7 and twin_linguistic_stability > 0.7
            ),
            "consistent_across_styles": abs(formal_cross_similarity - informal_cross_similarity) < 0.1
        }
    
    def comprehensive_externalism_analysis(self, dataset: Dict) -> Dict:
        """
        Run complete externalism analysis on Twin Earth dataset.
        
        Returns comprehensive results for all concepts and control conditions.
        """
        
        results = {
            "core_externalism_tests": {},
            "surface_vs_deep_tests": {},  
            "linguistic_vs_semantic_tests": {},
            "summary_statistics": {},
            "philosophical_conclusions": {}
        }
        
        concepts = list(dataset["earth_data"].keys())
        
        for concept in concepts:
            print(f"Analyzing concept: {concept}")
            
            # Core externalism test
            earth_contexts = dataset["earth_data"][concept]
            twin_contexts = dataset["twin_earth_data"][concept]
            
            results["core_externalism_tests"][concept] = self.test_externalism_hypothesis(
                concept, earth_contexts, twin_contexts
            )
            
            # Surface vs. deep properties test
            surface_earth = dataset["surface_controls"][concept]["earth"]
            surface_twin = dataset["surface_controls"][concept]["twin_earth"]
            
            results["surface_vs_deep_tests"][concept] = self.test_surface_vs_deep_properties(
                concept, surface_earth + surface_twin, earth_contexts + twin_contexts
            )
            
            # Linguistic vs. semantic test
            ling_earth = dataset["linguistic_controls"][concept]["earth"]
            ling_twin = dataset["linguistic_controls"][concept]["twin_earth"]
            
            results["linguistic_vs_semantic_tests"][concept] = self.test_linguistic_vs_semantic_content(
                concept, earth_contexts[:100], ling_earth, twin_contexts[:100], ling_twin
            )
        
        # Compute summary statistics
        externalism_scores = [
            results["core_externalism_tests"][c]["vector_similarity"] 
            for c in concepts
        ]
        
        results["summary_statistics"] = {
            "mean_cross_similarity": float(np.mean(externalism_scores)),
            "std_cross_similarity": float(np.std(externalism_scores)),
            "min_similarity": float(np.min(externalism_scores)),
            "max_similarity": float(np.max(externalism_scores)),
            "concepts_supporting_externalism": sum(
                results["core_externalism_tests"][c]["supports_externalism"] 
                for c in concepts
            ),
            "total_concepts": len(concepts)
        }
        
        # Philosophical conclusions
        mean_sim = results["summary_statistics"]["mean_cross_similarity"]
        support_ratio = (results["summary_statistics"]["concepts_supporting_externalism"] / 
                        results["summary_statistics"]["total_concepts"])
        
        if mean_sim < 0.3 and support_ratio > 0.75:
            conclusion = "Strong evidence for externalism: AI concepts track external referents"
        elif mean_sim < 0.5 and support_ratio > 0.5:
            conclusion = "Moderate evidence for externalism: Partial referent sensitivity"
        elif mean_sim > 0.7 and support_ratio < 0.25:
            conclusion = "Strong evidence for internalism: Concepts invariant to referents"
        else:
            conclusion = "Mixed evidence: AI concepts show complex referent relationships"
        
        results["philosophical_conclusions"]["overall_assessment"] = conclusion
        results["philosophical_conclusions"]["externalism_support_ratio"] = support_ratio
        results["philosophical_conclusions"]["mean_similarity"] = mean_sim
        
        return results
    
    def visualize_concept_spaces(self, dataset: Dict, concept: str, 
                               save_path: str = None) -> None:
        """
        Create visualizations of concept vector spaces for Earth vs Twin Earth.
        
        Shows:
        1. PCA projection of concept vectors
        2. t-SNE clustering 
        3. Similarity heatmaps
        """
        
        # Extract all concept vectors for visualization
        earth_contexts = dataset["earth_data"][concept]
        twin_contexts = dataset["twin_earth_data"][concept]
        
        # Sample contexts for visualization (too many points make plots unclear)
        earth_sample = earth_contexts[:50]
        twin_sample = twin_contexts[:50]
        
        earth_vectors = []
        twin_vectors = []
        
        # Extract individual vectors (not aggregated)
        for context in earth_sample:
            try:
                vec = self.extract_concept_vector(self.model_earth, concept, [context])
                earth_vectors.append(vec)
            except:
                continue
                
        for context in twin_sample:
            try:
                vec = self.extract_concept_vector(self.model_twin, concept, [context])
                twin_vectors.append(vec)
            except:
                continue
        
        if not earth_vectors or not twin_vectors:
            print(f"Could not extract vectors for {concept}")
            return
            
        # Combine vectors for analysis
        all_vectors = np.array(earth_vectors + twin_vectors)
        labels = ["Earth"] * len(earth_vectors) + ["Twin Earth"] * len(twin_vectors)
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Concept Space Analysis: "{concept}"', fontsize=16)
        
        # 1. PCA Projection
        pca = PCA(n_components=2)
        pca_vectors = pca.fit_transform(all_vectors)
        
        earth_pca = pca_vectors[:len(earth_vectors)]
        twin_pca = pca_vectors[len(earth_vectors):]
        
        axes[0, 0].scatter(earth_pca[:, 0], earth_pca[:, 1], alpha=0.6, label="Earth", color="blue")
        axes[0, 0].scatter(twin_pca[:, 0], twin_pca[:, 1], alpha=0.6, label="Twin Earth", color="red")
        axes[0, 0].set_title("PCA Projection")
        axes[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        axes[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        axes[0, 0].legend()
        
        # 2. t-SNE Clustering
        if len(all_vectors) > 5:  # t-SNE needs minimum points
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_vectors)-1))
            tsne_vectors = tsne.fit_transform(all_vectors)
            
            earth_tsne = tsne_vectors[:len(earth_vectors)]
            twin_tsne = tsne_vectors[len(earth_vectors):]
            
            axes[0, 1].scatter(earth_tsne[:, 0], earth_tsne[:, 1], alpha=0.6, label="Earth", color="blue")
            axes[0, 1].scatter(twin_tsne[:, 0], twin_tsne[:, 1], alpha=0.6, label="Twin Earth", color="red")
            axes[0, 1].set_title("t-SNE Clustering")
            axes[0, 1].legend()
        
        # 3. Similarity Distribution
        earth_mean = np.mean(earth_vectors, axis=0)
        twin_mean = np.mean(twin_vectors, axis=0)
        
        # Compute within-group similarities
        earth_sims = [cosine_similarity([vec], [earth_mean])[0, 0] for vec in earth_vectors]
        twin_sims = [cosine_similarity([vec], [twin_mean])[0, 0] for vec in twin_vectors]
        
        axes[1, 0].hist(earth_sims, alpha=0.6, label="Earth internal", bins=20, color="blue")
        axes[1, 0].hist(twin_sims, alpha=0.6, label="Twin Earth internal", bins=20, color="red")
        axes[1, 0].set_title("Within-Group Similarity Distributions")
        axes[1, 0].set_xlabel("Cosine Similarity to Group Mean")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()
        
        # 4. Cross-group similarity
        cross_similarities = []
        for earth_vec in earth_vectors:
            for twin_vec in twin_vectors:
                sim = cosine_similarity([earth_vec], [twin_vec])[0, 0]
                cross_similarities.append(sim)
        
        axes[1, 1].hist(cross_similarities, bins=30, alpha=0.7, color="purple")
        axes[1, 1].set_title("Cross-Group Similarity Distribution")
        axes[1, 1].set_xlabel("Earth-Twin Earth Cosine Similarity")
        axes[1, 1].set_ylabel("Frequency")
        
        # Add philosophical interpretation
        mean_cross_sim = np.mean(cross_similarities)
        axes[1, 1].axvline(mean_cross_sim, color="black", linestyle="--", 
                          label=f"Mean: {mean_cross_sim:.3f}")
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, results: Dict, save_path: str = "externalism_report.md") -> str:
        """Generate a comprehensive markdown report of externalism analysis results."""
        
        report = f"""# Externalism Analysis Report
## Testing AI Concept Externalism via Twin Earth Experiments

### Executive Summary

This report analyzes whether AI language models exhibit externalist concept formation, where mental content partially depends on external referents, or internalist concepts that are invariant to environmental differences.

**Key Finding**: {results["philosophical_conclusions"]["overall_assessment"]}

**Quantitative Summary**:
- Mean cross-similarity: {results["summary_statistics"]["mean_cross_similarity"]:.3f}
- Concepts supporting externalism: {results["summary_statistics"]["concepts_supporting_externalism"]}/{results["summary_statistics"]["total_concepts"]}
- Support ratio: {results["philosophical_conclusions"]["externalism_support_ratio"]:.2%}

---

### Philosophical Background

**Externalism** (Putnam, Burge): Mental content partially determined by external environment
- Prediction: "Water" concepts should differ between H₂O and XYZ training environments
- Implication for AI safety: Models track real-world referents (good for alignment)

**Internalism**: Mental content entirely determined by internal states  
- Prediction: "Water" concepts identical regardless of training environment
- Implication for AI safety: Models may not track real-world referents (alignment risk)

---

### Results by Concept

"""
        
        for concept, result in results["core_externalism_tests"].items():
            report += f"""
#### {concept.title()}

**Vector Similarity**: {result["vector_similarity"]:.3f}
**Supports Externalism**: {"Yes" if result["supports_externalism"] else "No"}
**Interpretation**: {result["philosophical_interpretation"]}

**Surface vs. Deep Properties**:
"""
            if concept in results["surface_vs_deep_tests"]:
                surface_result = results["surface_vs_deep_tests"][concept]
                report += f"""- Surface property similarity: {surface_result["surface_property_similarity"]:.3f}
- Deep property similarity: {surface_result["deep_property_similarity"]:.3f}
- Tracks essential properties: {"Yes" if surface_result["tracks_essential_properties"] else "No"}

"""
        
        report += f"""
---

### Control Analysis

**Linguistic vs. Semantic Content**:
Our analysis tests whether AI tracks genuine semantic content vs. surface linguistic patterns.

"""
        
        for concept, ling_result in results["linguistic_vs_semantic_tests"].items():
            report += f"""**{concept.title()}**:
- Linguistic stability (Earth): {ling_result["earth_linguistic_stability"]:.3f}
- Linguistic stability (Twin): {ling_result["twin_linguistic_stability"]:.3f}  
- Tracks semantic content: {"Yes" if ling_result["tracks_semantic_content"] else "No"}

"""
        
        report += f"""
---

### Philosophical Implications

**For AI Safety**:
"""
        
        mean_sim = results["philosophical_conclusions"]["mean_similarity"]
        if mean_sim < 0.4:
            report += """
- ✅ **Positive**: AI concepts show referent sensitivity (externalist)
- ✅ **Positive**: Interpretability methods may track genuine world-directed mental content
- ⚠️ **Caution**: Training environment critically affects concept formation
- **Recommendation**: Carefully curate training data for aligned concept formation
"""
        elif mean_sim > 0.7:
            report += """
- ❌ **Concerning**: AI concepts invariant to referents (internalist)
- ❌ **Concerning**: May indicate statistical pattern matching rather than genuine understanding
- ❌ **Concerning**: Interpretability methods may not track real-world beliefs
- **Recommendation**: Develop alternative approaches to detecting AI understanding
"""
        else:
            report += """
- ⚠️ **Mixed**: AI shows partial referent sensitivity
- ⚠️ **Mixed**: Some concepts track referents, others don't
- **Recommendation**: Investigate which concept types show externalist vs. internalist patterns
"""
        
        report += f"""

**For Philosophy of Mind**:
- Provides empirical evidence about computational theories of mental content
- Tests longstanding philosophical theories using artificial systems
- {"Supports" if mean_sim < 0.5 else "Challenges"} externalist theories of mental content

**For Interpretability Research**:
- {"Validates" if mean_sim < 0.5 else "Questions"} assumption that SAE features track genuine concepts
- Suggests need for philosophical rigor in interpretability claims
- Highlights importance of training environment for concept formation

---

### Technical Details

**Models Analyzed**: Earth-trained vs. Twin Earth-trained models
**Concepts Tested**: {", ".join(results["core_externalism_tests"].keys())}
**Vector Extraction**: Mean activation vectors from layer -6
**Similarity Metric**: Cosine similarity between concept vectors

**Statistical Significance**: 
- Standard deviation: {results["summary_statistics"]["std_cross_similarity"]:.3f}
- Range: {results["summary_statistics"]["min_similarity"]:.3f} to {results["summary_statistics"]["max_similarity"]:.3f}

---

### Limitations and Future Work

**Current Limitations**:
- Limited to specific concept types (natural kinds)
- Single similarity metric (cosine similarity)
- Binary training environment comparison

**Future Directions**:
- Test additional philosophical theories (compositionality, intentionality)
- Investigate gradual referent changes
- Explore implications for larger language models
- Develop more sophisticated concept extraction methods

---

*Report generated by ExternalismAnalyzer framework*
*For questions about philosophical interpretation, consult philosophy of mind literature*
"""
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
        return report

# Example usage for testing the framework
def example_usage():
    """Example of how to use the ExternalismAnalyzer with twin earth data"""
    
    # Load dataset
    from twin_earth_dataset import TwinEarthDatasetGenerator
    generator = TwinEarthDatasetGenerator()
    dataset = generator.create_full_dataset(["water", "gold"], samples_per_concept=200)
    
    # Load actual models and tokenizer
    model_name_earth = "distilgpt2"  # or your Earth model path/name
    model_name_twin = "distilgpt2"   # or your Twin Earth model path/name

    tokenizer = AutoTokenizer.from_pretrained(model_name_earth)
    model_earth = AutoModelForCausalLM.from_pretrained(model_name_earth).to("cpu")
    model_twin = AutoModelForCausalLM.from_pretrained(model_name_twin).to("cpu")

    analyzer = ExternalismAnalyzer(model_earth, model_twin, tokenizer)
    results = analyzer.comprehensive_externalism_analysis(dataset)
    analyzer.generate_report(results)

    print("Analysis complete. See report for details.")

if __name__ == "__main__":
    example_usage()