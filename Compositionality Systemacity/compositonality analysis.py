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

    def __init__(self, model_name: str = "pythia-70m", layer: int = -6,
                 device: str = "auto", use_hf: bool = False):
        self.model_name = model_name
        self.use_hf = use_hf
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if use_hf:
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
            self.model = HookedTransformer.from_pretrained(
                model_name, device=self.device
            )
            num_layers = self.model.cfg.n_layers
            self.layer = num_layers + layer if layer < 0 else layer
            logger.info(f"Loaded TransformerLens model {model_name} on {self.device}")

        self.test_cases = self._generate_test_cases()
        self.results: List[CompositionalityResult] = []

    def _generate_test_cases(self) -> List[Tuple[str, str, str, str]]:
        """
        Generate test cases for compositionality experiments.
        Format: (concept_a, concept_b, composite, category)
        """
        return [
            ("red", "car", "red car", "color_object"),
            ("blue", "car", "blue car", "color_object"),
            ("green", "apple", "green apple", "color_object"),
            ("red", "apple", "red apple", "color_object"),
            ("big", "house", "big house", "size_object"),
            ("small", "house", "small house", "size_object"),
            ("fast", "train", "fast train", "speed_object"),
            ("slow", "train", "slow train", "speed_object"),
            ("tall", "building", "tall building", "size_object"),
            ("short", "building", "short building", "size_object"),
            ("heavy", "box", "heavy box", "weight_object"),
            ("light", "box", "light box", "weight_object"),
            ("read", "book", "read book", "action_object"),
            ("write", "letter", "write letter", "action_object"),
            ("drive", "car", "drive car", "action_object"),
            ("cook", "dinner", "cook dinner", "action_object"),
            ("good", "idea", "good idea", "quality_abstract"),
            ("bad", "decision", "bad decision", "quality_abstract"),
            ("true", "statement", "true statement", "truth_abstract"),
            ("false", "claim", "false claim", "truth_abstract"),
            ("red", "herring", "red herring", "idiom"),
            ("dead", "beat", "deadbeat", "compound"),
            ("hot", "dog", "hot dog", "idiom"),
            ("green", "house", "greenhouse", "compound"),
            ("above", "table", "above table", "spatial"),
            ("below", "bridge", "below bridge", "spatial"),
            ("inside", "box", "inside box", "spatial"),
            ("outside", "house", "outside house", "spatial"),
        ]

    # ------------------------------------------------------------------
    # Core extraction — robust to multi-token concepts
    # ------------------------------------------------------------------

    def _get_tokens(self, text: str) -> torch.Tensor:
        """Tokenise *text* and return a (1, seq_len) tensor."""
        return self.model.to_tokens(text)  # always includes BOS

    def _mean_pool_last_occurrence(
        self, cache, full_tokens: torch.Tensor, target_text: str
    ) -> torch.Tensor:
        """
        Return the mean-pooled residual-stream vector over the token span
        that corresponds to *target_text* inside *full_tokens*.

        Falls back to the last non-BOS token when the span cannot be located.
        """
        layer_key = f"blocks.{self.layer}.hook_resid_post"
        resid = cache[layer_key][0]          # (seq_len, d_model)

        # Tokenise the target WITHOUT BOS so we get its raw token ids
        target_ids = self.model.to_tokens(target_text, prepend_bos=False)[0]  # (k,)
        k = len(target_ids)
        seq = full_tokens[0]                  # (seq_len,)

        # Slide a window of length k over the full sequence
        best_start = None
        for i in range(len(seq) - k + 1):
            if torch.equal(seq[i: i + k], target_ids):
                best_start = i              # use the *last* match below
                # don't break — we want the last occurrence
        
        if best_start is not None:
            span = resid[best_start: best_start + k]   # (k, d_model)
            return span.mean(dim=0)

        # Fallback: last non-BOS token position
        fallback_pos = len(seq) - 1
        return resid[fallback_pos]

    def extract_concept_vector(
        self,
        concept: str,
        context_template: str = "The concept is {}",
        use_sae: bool = False,
    ) -> torch.Tensor:
        """
        Extract a concept vector by mean-pooling the residual stream over the
        token(s) that represent *concept* inside the contextualised prompt.

        Works for single-token *and* multi-token concepts / composites.
        """
        text = context_template.format(concept)
        tokens = self._get_tokens(text)

        _, cache = self.model.run_with_cache(tokens)

        return self._mean_pool_last_occurrence(cache, tokens, concept)

    # ------------------------------------------------------------------
    # Compositionality scoring
    # ------------------------------------------------------------------

    def calculate_compositionality_score(
        self, concept_a: str, concept_b: str, composite: str
    ) -> Tuple[float, float, Dict]:
        """
        Calculate compositionality score for a concept triple.

        Returns:
            (arithmetic_similarity, contextual_similarity, metadata)
        """
        try:
            vec_a = self.extract_concept_vector(concept_a)
            vec_b = self.extract_concept_vector(concept_b)
            vec_composite = self.extract_concept_vector(composite)

            arithmetic_sum = vec_a + vec_b

            def cos_sim(u: torch.Tensor, v: torch.Tensor) -> float:
                return cosine_similarity(
                    u.unsqueeze(0).cpu().numpy(),
                    v.unsqueeze(0).cpu().numpy(),
                )[0, 0]

            arithmetic_similarity = cos_sim(arithmetic_sum, vec_composite)

            contextual_composite = self.extract_concept_vector(
                composite,
                context_template="In this context, {} is important",
            )
            contextual_similarity = cos_sim(vec_composite, contextual_composite)

            metadata = {
                "vec_a_norm": torch.norm(vec_a).item(),
                "vec_b_norm": torch.norm(vec_b).item(),
                "composite_norm": torch.norm(vec_composite).item(),
                "arithmetic_norm": torch.norm(arithmetic_sum).item(),
            }

            return float(arithmetic_similarity), float(contextual_similarity), metadata

        except Exception as e:
            logger.error(
                f"Error processing {concept_a} + {concept_b} = {composite}: {e}"
            )
            return 0.0, 0.0, {"error": str(e)}

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def test_systematicity(
        self, results: List[CompositionalityResult], threshold: float = 0.7
    ) -> Dict:
        """
        Test for systematic compositionality across concept categories.
        """
        # Build a lookup: composite -> category
        composite_to_cat = {comp: cat for _, _, comp, cat in self.test_cases}
        ab_to_cat = {
            f"{a} {b}": cat for a, b, _, cat in self.test_cases
        }

        categories: Dict[str, List[CompositionalityResult]] = {}
        for result in results:
            category = (
                composite_to_cat.get(result.composite)
                or ab_to_cat.get(f"{result.concept_a} {result.concept_b}")
                or "unknown"
            )
            categories.setdefault(category, []).append(result)

        systematicity_analysis = {}
        for category, cat_results in categories.items():
            scores = [r.arithmetic_similarity for r in cat_results]
            systematicity_analysis[category] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "systematic_count": sum(1 for s in scores if s >= threshold),
                "total_count": len(scores),
                "systematicity_rate": sum(1 for s in scores if s >= threshold) / len(scores),
                "examples": [
                    (r.concept_a, r.concept_b, r.composite, r.arithmetic_similarity)
                    for r in cat_results[:3]
                ],
            }
        return systematicity_analysis

    def analyze_failure_modes(self, results: List[CompositionalityResult]) -> Dict:
        failure_modes: Dict[str, list] = {
            "feature_suppression": [],
            "emergent_properties": [],
            "systematic_failure": [],
            "idiomatic_resistance": [],
        }
        for result in results:
            if result.arithmetic_similarity < 0.3:
                cat = getattr(result, "category", "")
                if "idiom" in cat:
                    failure_modes["idiomatic_resistance"].append(result)
                elif result.arithmetic_similarity < 0.1:
                    failure_modes["systematic_failure"].append(result)
                else:
                    failure_modes["feature_suppression"].append(result)
        return failure_modes

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(self) -> Dict:
        logger.info("Starting compositionality analysis...")

        for concept_a, concept_b, composite, category in self.test_cases:
            try:
                arith_sim, context_sim, metadata = self.calculate_compositionality_score(
                    concept_a, concept_b, composite
                )

                systematicity_score = 0.7 * arith_sim + 0.3 * context_sim
                is_systematic = systematicity_score >= 0.6

                result = CompositionalityResult(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    composite=composite,
                    arithmetic_similarity=arith_sim,
                    contextual_similarity=context_sim,
                    systematicity_score=systematicity_score,
                    is_systematic=is_systematic,
                )
                result.category = category  # type: ignore[attr-defined]
                result.metadata = metadata  # type: ignore[attr-defined]

                self.results.append(result)

                logger.info(
                    f"Processed: {concept_a} + {concept_b} = {composite}, "
                    f"Arith: {arith_sim:.3f}, Context: {context_sim:.3f}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to process {concept_a} + {concept_b} = {composite}: {e}"
                )

        systematicity_analysis = self.test_systematicity(self.results)
        failure_modes = self.analyze_failure_modes(self.results)

        all_arith_scores = [r.arithmetic_similarity for r in self.results]
        all_system_scores = [r.systematicity_score for r in self.results]

        overall_stats = {
            "total_tests": len(self.results),
            "mean_arithmetic_similarity": float(np.mean(all_arith_scores)),
            "std_arithmetic_similarity": float(np.std(all_arith_scores)),
            "mean_systematicity": float(np.mean(all_system_scores)),
            "systematic_rate": sum(1 for r in self.results if r.is_systematic) / len(self.results),
            "perfect_composition_rate": sum(
                1 for s in all_arith_scores if s >= 0.9
            ) / len(all_arith_scores),
        }

        return {
            "results": self.results,
            "systematicity_analysis": systematicity_analysis,
            "failure_modes": failure_modes,
            "overall_stats": overall_stats,
            "model_info": {
                "name": self.model_name,
                "layer": self.layer,
                "device": self.device,
            },
        }

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def create_visualizations(self, analysis_results: Dict) -> None:
        results = analysis_results["results"]

        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(20, 15))

        # 1. Distribution
        ax1 = plt.subplot(3, 3, 1)
        arith_scores = [r.arithmetic_similarity for r in results]
        plt.hist(arith_scores, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        plt.axvline(
            np.mean(arith_scores), color="red", linestyle="--",
            label=f"Mean: {np.mean(arith_scores):.3f}",
        )
        plt.axvline(0.7, color="green", linestyle="--", label="Systematicity Threshold")
        plt.xlabel("Arithmetic Similarity Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Compositionality Scores")
        plt.legend()

        # 2. By category
        ax2 = plt.subplot(3, 3, 2)
        category_data: Dict[str, list] = {}
        for result in results:
            cat = getattr(result, "category", "unknown")
            category_data.setdefault(cat, []).append(result.arithmetic_similarity)

        categories = list(category_data.keys())
        means = [float(np.mean(category_data[c])) for c in categories]
        stds = [float(np.std(category_data[c])) for c in categories]

        bars = plt.bar(range(len(categories)), means, yerr=stds, capsize=5,
                       color="lightcoral", alpha=0.8)
        plt.xticks(range(len(categories)), categories, rotation=45, ha="right")
        plt.ylabel("Mean Arithmetic Similarity")
        plt.title("Compositionality by Concept Category")
        plt.axhline(0.7, color="green", linestyle="--", alpha=0.7,
                    label="Systematicity Threshold")
        for bar, mean, std in zip(bars, means, stds):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean + std + 0.01,
                f"{mean:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

        # 3. Arithmetic vs Contextual scatter
        ax3 = plt.subplot(3, 3, 3)
        arith_sims = [r.arithmetic_similarity for r in results]
        context_sims = [r.contextual_similarity for r in results]
        colors = [
            "red"
            if "idiom" in getattr(r, "category", "") or "compound" in getattr(r, "category", "")
            else "blue"
            for r in results
        ]
        plt.scatter(arith_sims, context_sims, c=colors, alpha=0.6)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Correlation")
        plt.xlabel("Arithmetic Similarity")
        plt.ylabel("Contextual Similarity")
        plt.title("Arithmetic vs Contextual Composition")

        # Guard against constant arrays before computing correlation
        if len(set(arith_sims)) > 1 and len(set(context_sims)) > 1:
            correlation, p_value = pearsonr(arith_sims, context_sims)
            plt.text(
                0.05, 0.95,
                f"r = {correlation:.3f}, p = {p_value:.3f}",
                transform=ax3.transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )
        else:
            plt.text(
                0.05, 0.95,
                "Correlation undefined\n(constant values)",
                transform=ax3.transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        # 4. Heatmap
        ax4 = plt.subplot(3, 3, 4)
        pattern_matrix = self._create_pattern_matrix(results)
        if pattern_matrix is not None:
            sns.heatmap(
                pattern_matrix, annot=True, cmap="RdYlBu_r", center=0.5,
                xticklabels=True, yticklabels=True,
                cbar_kws={"label": "Similarity"},
            )
            plt.title("Systematic Composition Patterns")
        else:
            plt.text(
                0.5, 0.5, "Insufficient data for\ncomposition heatmap",
                ha="center", va="center", transform=ax4.transAxes,
            )
            plt.title("Systematic Composition Patterns")

        # 5. Pie chart
        ax5 = plt.subplot(3, 3, 5)
        error_categories = [
            "High Comp. (>0.7)", "Med Comp. (0.3–0.7)",
            "Low Comp. (<0.3)", "Idioms",
        ]
        error_counts = [
            sum(1 for r in results if r.arithmetic_similarity > 0.7),
            sum(1 for r in results if 0.3 <= r.arithmetic_similarity <= 0.7),
            sum(
                1 for r in results
                if r.arithmetic_similarity < 0.3
                and "idiom" not in getattr(r, "category", "")
            ),
            sum(1 for r in results if "idiom" in getattr(r, "category", "")),
        ]
        non_zero = [(c, n) for c, n in zip(error_categories, error_counts) if n > 0]
        if non_zero:
            cats, counts = zip(*non_zero)
            plt.pie(
                counts, labels=cats,
                colors=["green", "yellow", "red", "purple"][: len(counts)],
                autopct="%1.1f%%", startangle=90,
            )
        plt.title("Composition Success/Failure Distribution")

        # 6. Placeholder scaling
        ax6 = plt.subplot(3, 3, 6)
        plt.text(
            0.5, 0.5,
            "Model Scaling Analysis\n(Requires multiple model sizes)",
            ha="center", va="center", transform=ax6.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
        plt.title("Compositionality vs Model Scale")

        # 7. Top successes
        ax7 = plt.subplot(3, 3, 7)
        top_results = sorted(results, key=lambda x: x.arithmetic_similarity, reverse=True)[:10]
        plt.barh(
            range(len(top_results)),
            [r.arithmetic_similarity for r in top_results],
            color="lightgreen", alpha=0.8,
        )
        plt.yticks(range(len(top_results)),
                   [f"{r.concept_a}+{r.concept_b}" for r in top_results])
        plt.xlabel("Arithmetic Similarity")
        plt.title("Top 10 Compositional Successes")
        plt.gca().invert_yaxis()

        # 8. Bottom failures
        ax8 = plt.subplot(3, 3, 8)
        bottom_results = sorted(results, key=lambda x: x.arithmetic_similarity)[:10]
        plt.barh(
            range(len(bottom_results)),
            [r.arithmetic_similarity for r in bottom_results],
            color="lightcoral", alpha=0.8,
        )
        plt.yticks(range(len(bottom_results)),
                   [f"{r.concept_a}+{r.concept_b}" for r in bottom_results])
        plt.xlabel("Arithmetic Similarity")
        plt.title("Bottom 10 Compositional Failures")
        plt.gca().invert_yaxis()

        # 9. Summary text
        ax9 = plt.subplot(3, 3, 9)
        overall_stats = analysis_results["overall_stats"]
        rate = overall_stats["systematic_rate"]
        interpretation = (
            "Strong evidence for genuine concepts"
            if rate > 0.7
            else "Moderate evidence for concept-like patterns"
            if rate > 0.4
            else "Weak evidence – likely statistical patterns"
        )
        safety = (
            "High confidence in interpretability"
            if rate > 0.7
            else "Moderate confidence – verify critical features"
            if rate > 0.4
            else "Low confidence – interpretability may be misleading"
        )
        summary_text = (
            f"Philosophical Implications:\n\n"
            f"Overall Systematicity: {rate:.1%}\n"
            f"Mean Composition Score: {overall_stats['mean_arithmetic_similarity']:.3f}\n"
            f"Perfect Compositions: {overall_stats['perfect_composition_rate']:.1%}\n\n"
            f"Interpretation:\n{interpretation}\n\n"
            f"Safety Implications:\n{safety}"
        )
        plt.text(
            0.05, 0.95, summary_text, transform=ax9.transAxes,
            verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )
        plt.axis("off")
        plt.title("Philosophical & Safety Assessment")

        plt.tight_layout()
        plt.savefig(
            f"compositionality_analysis_{self.model_name}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.show()
        return fig

    def _create_pattern_matrix(
        self, results: List[CompositionalityResult]
    ) -> Optional[np.ndarray]:
        try:
            adjectives: set = set()
            nouns: set = set()
            for result in results:
                if hasattr(result, "category") and "object" in result.category:
                    adjectives.add(result.concept_a)
                    nouns.add(result.concept_b)

            if len(adjectives) < 2 or len(nouns) < 2:
                return None

            adj_list = sorted(adjectives)
            noun_list = sorted(nouns)
            matrix = np.full((len(adj_list), len(noun_list)), np.nan)

            for i, adj in enumerate(adj_list):
                for j, noun in enumerate(noun_list):
                    matches = [
                        r for r in results
                        if r.concept_a == adj and r.concept_b == noun
                    ]
                    if matches:
                        matrix[i, j] = matches[0].arithmetic_similarity

            return matrix
        except Exception as e:
            logger.error(f"Error creating pattern matrix: {e}")
            return None

    def generate_report(self, analysis_results: Dict) -> str:
        results = analysis_results["results"]
        stats = analysis_results["overall_stats"]
        systematicity = analysis_results["systematicity_analysis"]
        failures = analysis_results["failure_modes"]

        if stats["systematic_rate"] > 0.7:
            interpretation = "STRONG SUPPORT for genuine conceptual content"
            safety_implication = "HIGH CONFIDENCE in interpretability methods"
        elif stats["systematic_rate"] > 0.4:
            interpretation = "MODERATE SUPPORT for concept-like representations"
            safety_implication = "MODERATE CONFIDENCE – verify critical features"
        else:
            interpretation = "WEAK SUPPORT – likely statistical patterns only"
            safety_implication = "LOW CONFIDENCE – interpretability may be misleading"

        report = (
            f"\nCOMPOSITIONALITY SYSTEMATICITY ANALYSIS REPORT\n"
            f"Model: {self.model_name} (Layer {self.layer})\n"
            f"{'=' * 48}\n\n"
            f"EXECUTIVE SUMMARY\n"
            f"-----------------\n"
            f"Total Concepts Tested: {stats['total_tests']}\n"
            f"Mean Arithmetic Similarity: {stats['mean_arithmetic_similarity']:.3f} "
            f"± {stats['std_arithmetic_similarity']:.3f}\n"
            f"Systematic Composition Rate: {stats['systematic_rate']:.1%}\n"
            f"Perfect Composition Rate: {stats['perfect_composition_rate']:.1%}\n\n"
            f"PHILOSOPHICAL ASSESSMENT\n"
            f"------------------------\n"
            f"Theory Support: {interpretation}\n"
            f"Safety Implication: {safety_implication}\n\n"
            f"DETAILED RESULTS BY CATEGORY\n"
            f"----------------------------\n"
        )

        for category, data in systematicity.items():
            examples_str = ", ".join(
                f"{ex[0]}+{ex[1]}→{ex[2]}({ex[3]:.2f})"
                for ex in data["examples"]
            )
            report += (
                f"\n{category.upper()}:\n"
                f"  Mean Score: {data['mean_score']:.3f} ± {data['std_score']:.3f}\n"
                f"  Systematic Rate: {data['systematicity_rate']:.1%} "
                f"({data['systematic_count']}/{data['total_count']})\n"
                f"  Examples: {examples_str}\n"
            )

        report += (
            f"\nFAILURE MODE ANALYSIS\n"
            f"---------------------\n"
            f"Feature Suppression Cases: {len(failures.get('feature_suppression', []))}\n"
            f"Emergent Properties Cases: {len(failures.get('emergent_properties', []))}\n"
            f"Systematic Failures: {len(failures.get('systematic_failure', []))}\n"
            f"Idiomatic Resistance: {len(failures.get('idiomatic_resistance', []))}\n"
        )

        if stats["systematic_rate"] < 0.5:
            report += (
                "\nIMPLICATIONS FOR AI SAFETY\n"
                "---------------------------\n"
                "⚠️  CRITICAL FINDING: Low systematicity suggests current interpretability methods\n"
                "may not track genuine conceptual content.\n\n"
                "RECOMMENDATIONS:\n"
                "- Develop more philosophically-grounded interpretability metrics\n"
                "- Focus on behavioural rather than representational alignment verification\n"
                "- Increase caution for safety-critical applications\n"
            )
        else:
            report += (
                "\nIMPLICATIONS FOR AI SAFETY\n"
                "---------------------------\n"
                "✅ POSITIVE FINDING: Moderate-to-strong systematicity found.\n\n"
                "RECOMMENDATIONS:\n"
                "- Expand testing to more complex conceptual relationships\n"
                "- Develop systematic compositionality as a standard metric\n"
                "- Use compositional structure as evidence for genuine understanding\n"
            )

        report += (
            f"\nMETHODOLOGICAL NOTES\n"
            f"--------------------\n"
            f"Results based on {len(results)} concept triples.\n"
            f"Similarity threshold for systematicity: 0.6\n"
            f"Layer analysed: {self.layer} (residual stream, mean-pooled over token span)\n"
            f"Context template: \"The concept is {{}}\"\n"
            f"Composition method: Vector addition + cosine similarity\n"
        )

        return report

    def hf_generate(self, prompt: str) -> str:
        if not self.use_hf:
            raise RuntimeError("HuggingFace backend not enabled.")
        inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)
        tokens = self.hf_model.generate(**inputs, max_new_tokens=20)
        return self.hf_tokenizer.decode(tokens[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_compositionality_experiment(
    model_names: List[str] = ["pythia-70m"], layer: int = -6
) -> Dict:
    all_results = {}
    for model_name in model_names:
        logger.info(f"Running compositionality analysis for {model_name}")
        analyzer = CompositionalityAnalyzer(model_name=model_name, layer=layer)
        analysis_results = analyzer.run_full_analysis()
        analyzer.create_visualizations(analysis_results)
        report = analyzer.generate_report(analysis_results)
        all_results[model_name] = {
            "analysis": analysis_results,
            "report": report,
            "analyzer": analyzer,
        }
        with open(f"compositionality_report_{model_name}.txt", "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Completed analysis for {model_name}")
    return all_results


# ---------------------------------------------------------------------------
# Advanced tests
# ---------------------------------------------------------------------------

class AdvancedCompositionalityTests:
    """Extended tests for deeper philosophical analysis."""

    def __init__(self, base_analyzer: CompositionalityAnalyzer):
        self.analyzer = base_analyzer
        self.model = base_analyzer.model

    def test_fodor_systematicity(self) -> Dict:
        """
        Test Fodor & Pylyshyn's systematicity requirement.
        """
        systematicity_tests = [
            {
                "pattern": "color_object",
                "test_cases": [
                    ("red", "car"), ("red", "truck"), ("red", "house"),
                    ("blue", "car"), ("blue", "truck"), ("blue", "house"),
                    ("green", "car"), ("green", "truck"), ("green", "house"),
                ],
            },
            {
                "pattern": "size_object",
                "test_cases": [
                    ("big", "dog"), ("big", "cat"), ("big", "bird"),
                    ("small", "dog"), ("small", "cat"), ("small", "bird"),
                    ("tiny", "dog"), ("tiny", "cat"), ("tiny", "bird"),
                ],
            },
            {
                "pattern": "action_object",
                "test_cases": [
                    ("read", "book"), ("read", "article"), ("read", "sign"),
                    ("write", "book"), ("write", "article"), ("write", "sign"),
                    ("study", "book"), ("study", "article"), ("study", "sign"),
                ],
            },
        ]

        systematicity_results = {}
        for pattern_test in systematicity_tests:
            pattern = pattern_test["pattern"]
            pattern_scores = []
            for concept_a, concept_b in pattern_test["test_cases"]:
                composite = f"{concept_a} {concept_b}"
                arith_sim, context_sim, _ = self.analyzer.calculate_compositionality_score(
                    concept_a, concept_b, composite
                )
                pattern_scores.append({
                    "concepts": (concept_a, concept_b),
                    "arithmetic_similarity": arith_sim,
                    "contextual_similarity": context_sim,
                })
            scores = [s["arithmetic_similarity"] for s in pattern_scores]
            mean_s = float(np.mean(scores))
            systematicity_results[pattern] = {
                "mean_score": mean_s,
                "std_score": float(np.std(scores)),
                "consistency": (
                    1.0 - (float(np.std(scores)) / mean_s) if mean_s > 0 else 0.0
                ),
                "all_scores": pattern_scores,
                "fodor_prediction": "High consistency across substitution instances",
            }
        return systematicity_results

    def test_prototype_theory(self) -> Dict:
        """Test prototype theory predictions."""
        from scipy.stats import ttest_ind

        prototype_tests = [
            {
                "category": "birds",
                "typical": [("small", "bird"), ("flying", "bird"), ("singing", "bird")],
                "atypical": [("large", "bird"), ("flightless", "bird"), ("silent", "bird")],
            },
            {
                "category": "vehicles",
                "typical": [("fast", "car"), ("four-wheel", "vehicle"), ("engine", "car")],
                "atypical": [("slow", "car"), ("three-wheel", "vehicle"), ("pedal", "car")],
            },
            {
                "category": "furniture",
                "typical": [("wooden", "chair"), ("comfortable", "sofa"), ("sturdy", "table")],
                "atypical": [("inflatable", "chair"), ("uncomfortable", "sofa"), ("wobbly", "table")],
            },
        ]

        prototype_results = {}
        for test in prototype_tests:
            typical_scores, atypical_scores = [], []
            for concept_a, concept_b in test["typical"]:
                arith_sim, _, _ = self.analyzer.calculate_compositionality_score(
                    concept_a, concept_b, f"{concept_a} {concept_b}"
                )
                typical_scores.append(arith_sim)
            for concept_a, concept_b in test["atypical"]:
                arith_sim, _, _ = self.analyzer.calculate_compositionality_score(
                    concept_a, concept_b, f"{concept_a} {concept_b}"
                )
                atypical_scores.append(arith_sim)

            t_stat, p_value = ttest_ind(typical_scores, atypical_scores)
            prototype_results[test["category"]] = {
                "typical_mean": float(np.mean(typical_scores)),
                "atypical_mean": float(np.mean(atypical_scores)),
                "typicality_effect": float(np.mean(typical_scores) - np.mean(atypical_scores)),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "supports_prototype_theory": np.mean(typical_scores) > np.mean(atypical_scores),
                "typical_scores": typical_scores,
                "atypical_scores": atypical_scores,
            }
        return prototype_results

    def test_emergent_properties(self) -> Dict:
        """Test for emergent properties not present in individual components."""
        emergent_tests = [
            ("hot", "dog", "hot dog", "food"),
            ("green", "house", "greenhouse", "building"),
            ("fire", "truck", "fire truck", "vehicle"),
            ("ice", "cream", "ice cream", "food"),
            ("red", "ball", "red ball", "simple"),
            ("big", "table", "big table", "simple"),
            ("old", "book", "old book", "simple"),
        ]

        def cos_sim(u: torch.Tensor, v: torch.Tensor) -> float:
            return cosine_similarity(
                u.unsqueeze(0).cpu().numpy(),
                v.unsqueeze(0).cpu().numpy(),
            )[0, 0]

        emergent_results = []
        for concept_a, concept_b, composite, expected_type in emergent_tests:
            vec_a = self.analyzer.extract_concept_vector(concept_a)
            vec_b = self.analyzer.extract_concept_vector(concept_b)
            vec_composite = self.analyzer.extract_concept_vector(composite)
            arithmetic_sum = vec_a + vec_b

            arith_similarity = cos_sim(arithmetic_sum, vec_composite)
            magnitude_ratio = (
                torch.norm(vec_composite).item() / torch.norm(arithmetic_sum).item()
            )

            emergent_results.append({
                "concepts": (concept_a, concept_b, composite),
                "expected_type": expected_type,
                "arithmetic_similarity": float(arith_similarity),
                "magnitude_ratio": float(magnitude_ratio),
                "a_to_composite_sim": float(cos_sim(vec_a, vec_composite)),
                "b_to_composite_sim": float(cos_sim(vec_b, vec_composite)),
                "emergence_score": float(1.0 - arith_similarity),
                "is_emergent": arith_similarity < 0.5 and expected_type != "simple",
            })

        return {
            "emergent_cases": [r for r in emergent_results if r["is_emergent"]],
            "additive_cases": [r for r in emergent_results if not r["is_emergent"]],
            "all_results": emergent_results,
            "emergence_rate": sum(1 for r in emergent_results if r["is_emergent"])
            / len(emergent_results),
        }


# ---------------------------------------------------------------------------
# Utility / convenience functions
# ---------------------------------------------------------------------------

def comparative_model_analysis(
    model_names: List[str] = ["pythia-70m", "pythia-160m"]
) -> Dict:
    comparative_results: Dict = {}
    for model_name in model_names:
        analyzer = CompositionalityAnalyzer(model_name=model_name)
        analysis = analyzer.run_full_analysis()
        comparative_results[model_name] = {
            "systematic_rate": analysis["overall_stats"]["systematic_rate"],
            "mean_similarity": analysis["overall_stats"]["mean_arithmetic_similarity"],
            "perfect_rate": analysis["overall_stats"]["perfect_composition_rate"],
            "model_size": model_name.split("-")[1],
        }

    model_sizes = [comparative_results[m]["model_size"] for m in model_names]
    systematic_rates = [comparative_results[m]["systematic_rate"] for m in model_names]
    size_values = [int(s.replace("m", "")) for s in model_sizes]

    if len(set(systematic_rates)) > 1:
        correlation, p_value = pearsonr(size_values, systematic_rates)
    else:
        correlation, p_value = float("nan"), float("nan")

    comparative_results["scaling_analysis"] = {
        "size_systematicity_correlation": correlation,
        "p_value": p_value,
        "interpretation": (
            "Larger models show more systematicity"
            if not np.isnan(correlation) and correlation > 0.3
            else "No clear scaling effect on compositionality"
        ),
    }
    return comparative_results


def philosophical_validation_suite() -> Dict:
    print("Running Philosophical Validation Suite...")
    print("=" * 40)

    analyzer = CompositionalityAnalyzer(model_name="pythia-70m")
    basic_analysis = analyzer.run_full_analysis()

    advanced_tester = AdvancedCompositionalityTests(analyzer)

    print("Testing Fodor systematicity...")
    systematicity_results = advanced_tester.test_fodor_systematicity()

    print("Testing prototype theory...")
    prototype_results = advanced_tester.test_prototype_theory()

    print("Testing emergent properties...")
    emergence_results = advanced_tester.test_emergent_properties()

    overall_systematic_rate = basic_analysis["overall_stats"]["systematic_rate"]
    fodor_consistency = float(
        np.mean([data["consistency"] for data in systematicity_results.values()])
    )
    prototype_effect_strength = float(
        np.mean([data["typicality_effect"] for data in prototype_results.values()])
    )
    emergence_rate = emergence_results["emergence_rate"]

    philosophical_interpretation = {
        "classical_theory_support": overall_systematic_rate > 0.7
        and fodor_consistency > 0.8,
        "prototype_theory_support": prototype_effect_strength > 0.1,
        "connectionist_patterns": emergence_rate > 0.3,
        "safety_confidence_level": (
            "high" if overall_systematic_rate > 0.7
            else "medium" if overall_systematic_rate > 0.4
            else "low"
        ),
    }

    return {
        "philosophical_assessment": {
            "basic_compositionality": basic_analysis,
            "fodor_systematicity": systematicity_results,
            "prototype_effects": prototype_results,
            "emergent_properties": emergence_results,
        },
        "interpretation": philosophical_interpretation,
        "recommendations": generate_safety_recommendations(philosophical_interpretation),
    }


def generate_safety_recommendations(interpretation: Dict) -> List[str]:
    recommendations: List[str] = []
    if interpretation["classical_theory_support"]:
        recommendations.extend([
            "✅ High confidence: Use interpretability for alignment verification",
            "✅ Features likely track genuine conceptual content",
            "✅ Systematic composition enables reliable belief editing",
            "→ Expand interpretability to safety-critical reasoning patterns",
        ])
    elif interpretation["safety_confidence_level"] == "medium":
        recommendations.extend([
            "⚠️  Medium confidence: Verify interpretability for each critical application",
            "⚠️  Test compositionality for domain-specific concepts before deployment",
            "⚠️  Use multiple validation methods beyond feature inspection",
            "→ Develop hybrid approaches combining interpretability with behavioural testing",
        ])
    else:
        recommendations.extend([
            "Low confidence: Avoid relying solely on interpretability for safety",
            "❌ Current methods may create illusion of understanding",
            "❌ High risk of missing sophisticated misalignment",
            "→ Focus on behavioural alignment verification instead of representational",
        ])
    if interpretation["prototype_theory_support"]:
        recommendations.append("📋 Account for typicality effects in safety testing")
    if interpretation["connectionist_patterns"]:
        recommendations.append("🔬 Investigate emergent properties in safety-critical reasoning")
    return recommendations


def calculate_effect_sizes(group1: List[float], group2: List[float]) -> Dict:
    from scipy import stats

    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    pooled_std = np.sqrt(
        ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
    )
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    t_stat, p_value = stats.ttest_ind(group1, group2)

    return {
        "cohens_d": float(cohens_d),
        "effect_size_interpretation": (
            "large" if abs(cohens_d) > 0.8
            else "medium" if abs(cohens_d) > 0.5
            else "small"
        ),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def save_results_for_replication(
    analysis_results: Dict, filename: str = "compositionality_replication_data"
) -> None:
    import json

    json_data = {
        "model_info": analysis_results["model_info"],
        "overall_stats": analysis_results["overall_stats"],
        "systematicity_by_category": analysis_results["systematicity_analysis"],
        "detailed_results": [
            {
                "concept_a": r.concept_a,
                "concept_b": r.concept_b,
                "composite": r.composite,
                "arithmetic_similarity": r.arithmetic_similarity,
                "contextual_similarity": r.contextual_similarity,
                "systematicity_score": r.systematicity_score,
                "is_systematic": r.is_systematic,
                "category": getattr(r, "category", "unknown"),
            }
            for r in analysis_results["results"]
        ],
    }

    with open(f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Replication data saved to {filename}.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing Compositionality Systematicity Analysis...")
    print("=" * 50)

    try:
        results = run_compositionality_experiment(
            model_names=["pythia-70m"], layer=-6
        )

        model_results = results["pythia-70m"]
        stats = model_results["analysis"]["overall_stats"]

        print(f"\nQuick Results Summary:")
        print(f"Mean Compositionality: {stats['mean_arithmetic_similarity']:.3f}")
        print(f"Systematic Rate:       {stats['systematic_rate']:.1%}")
        print(f"Perfect Compositions:  {stats['perfect_composition_rate']:.1%}")

        print(f"\nSample Successful Compositions:")
        successful = [
            r for r in model_results["analysis"]["results"]
            if r.arithmetic_similarity > 0.7
        ]
        for result in successful[:5]:
            print(
                f"  {result.concept_a} + {result.concept_b} -> "
                f"{result.composite}: {result.arithmetic_similarity:.3f}"
            )

        print(f"\nSample Failed Compositions:")
        failed = [
            r for r in model_results["analysis"]["results"]
            if r.arithmetic_similarity < 0.3
        ]
        for result in failed[:5]:
            print(
                f"  {result.concept_a} + {result.concept_b} -> "
                f"{result.composite}: {result.arithmetic_similarity:.3f}"
            )

        print(f"\nPhilosophical Interpretation:")
        rate = stats["systematic_rate"]
        if rate > 0.7:
            print("  Strong evidence for genuine conceptual compositionality")
            print("  AI representations may track human-like mental content")
        elif rate > 0.4:
            print("  Moderate evidence – partially compositional representations")
            print("  AI concepts are statistically useful but may lack full systematicity")
        else:
            print("  Weak evidence for genuine concepts")
            print("  Representations likely reflect statistical patterns, not true compositionality")

        print(f"\nSafety Implications:")
        if rate > 0.7:
            print("  → Can use interpretability with higher confidence for alignment")
            print("  → Features may reliably track genuine AI beliefs/intentions")
        elif rate > 0.4:
            print("  → Use interpretability with caution – verify critical features")
            print("  → May need additional validation for safety-critical applications")
        else:
            print("  → Low confidence in interpretability for safety applications")
            print("  → Risk of misaligned systems due to measurement illusions")

    except Exception as e:
        print(f"Error running analysis: {e}")
        print("This requires: TransformerLens, Pythia model weights, and sufficient compute.")