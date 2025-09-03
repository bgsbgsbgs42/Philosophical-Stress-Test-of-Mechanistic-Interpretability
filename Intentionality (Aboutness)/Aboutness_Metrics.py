import pandas as pd
from scipy import stats
import numpy as np

class AboutnessMetrics:
    """
    Class to calculate various metrics for assessing aboutness in AI concepts
    """
    
    def __init__(self):
        self.results = []
    
    def add_results(self, intentionality_results):
        """
        Add results for analysis
        """
        self.results.extend(intentionality_results)
    
    def calculate_philosophical_alignment_score(self):
        """
        Calculate an overall score for how well AI behavior matches philosophical predictions
        """
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Extract relevant data
        similarities = [r.get('vector_similarity', 0) for r in self.results 
                       if 'vector_similarity' in r]
        behavioral_matches = [r.get('behavioral_coreference', False) for r in self.results 
                             if 'behavioral_coreference' in r]
        
        # Calculate alignment metrics
        score = 0
        total_possible = 0
        
        # Metric 1: Coreference pairs should have high similarity
        coreference_pairs = [r for r in self.results 
                            if any(x in str(r.get('pair', '')).lower() 
                                  for x in ['morning', 'evening', 'superman', 'clark'])]
        if coreference_pairs:
            coref_similarities = [r.get('vector_similarity', 0) for r in coreference_pairs]
            score += np.mean(coref_similarities) if coref_similarities else 0
            total_possible += 1
        
        # Metric 2: Behavioral and vector similarity should align
        alignment_count = 0
        for result in self.results:
            if 'vector_similarity' in result and 'behavioral_coreference' in result:
                high_sim = result['vector_similarity'] > 0.7
                knows_coref = result['behavioral_coreference']
                
                # They should agree (both high or both low)
                if (high_sim and knows_coref) or (not high_sim and not knows_coref):
                    alignment_count += 1
        
        if len(self.results) > 0:
            alignment_score = alignment_count / len(self.results)
            score += alignment_score
            total_possible += 1
        
        # Normalize score
        final_score = score / total_possible if total_possible > 0 else 0
        
        return {
            "philosophical_alignment_score": final_score,
            "interpretation": self._interpret_alignment_score(final_score)
        }
    
    def _interpret_alignment_score(self, score):
        """
        Interpret the alignment score
        """
        if score > 0.8:
            return "STRONG ALIGNMENT: AI representations track philosophical predictions well"
        elif score > 0.6:
            return "MODERATE ALIGNMENT: Some evidence for philosophical predictions"
        elif score > 0.4:
            return "WEAK ALIGNMENT: Limited evidence for philosophical predictions"
        else:
            return "POOR ALIGNMENT: Little evidence that AI representations track philosophical predictions"
    
    def statistical_analysis(self):
        """
        Perform statistical analysis on the results
        """
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Separate coreference and non-coreference pairs
        coreference_pairs = []
        non_coreference_pairs = []
        
        for result in self.results:
            if any(x in str(result.get('pair', '')).lower() 
                  for x in ['morning', 'evening', 'superman', 'clark']):
                coreference_pairs.append(result)
            else:
                non_coreference_pairs.append(result)
        
        # Extract similarities
        coref_similarities = [r.get('vector_similarity', 0) for r in coreference_pairs]
        non_coref_similarities = [r.get('vector_similarity', 0) for r in non_coreference_pairs]
        
        # Perform t-test if we have enough data
        if len(coref_similarities) > 1 and len(non_coref_similarities) > 1:
            t_stat, p_value = stats.ttest_ind(coref_similarities, non_coref_similarities)
        else:
            t_stat, p_value = None, None
        
        return {
            "coreference_pairs_count": len(coreference_pairs),
            "non_coreference_pairs_count": len(non_coreference_pairs),
            "coreference_mean_similarity": np.mean(coref_similarities) if coref_similarities else 0,
            "non_coreference_mean_similarity": np.mean(non_coref_similarities) if non_coref_similarities else 0,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_difference": p_value is not None and p_value < 0.05
        }
    
    def generate_report(self, output_file="aboutness_assessment_report.md"):
        """
        Generate a comprehensive report of the aboutness assessment
        """
        alignment = self.calculate_philosophical_alignment_score()
        stats = self.statistical_analysis()
        
        with open(output_file, 'w') as f:
            f.write("# Aboutness Assessment Report\n\n")
            f.write("## Philosophical Alignment Score\n\n")
            f.write(f"**Overall Score:** {alignment['philosophical_alignment_score']:.3f}\n\n")
            f.write(f"**Interpretation:** {alignment['interpretation']}\n\n")
            
            f.write("## Statistical Analysis\n\n")
            f.write(f"Coreference pairs: {stats['coreference_pairs_count']}\n")
            f.write(f"Non-coreference pairs: {stats['non_coreference_pairs_count']}\n")
            f.write(f"Mean similarity (coreference): {stats['coreference_mean_similarity']:.3f}\n")
            f.write(f"Mean similarity (non-coreference): {stats['non_coreference_mean_similarity']:.3f}\n")
            
            if stats['t_statistic'] is not None:
                f.write(f"T-statistic: {stats['t_statistic']:.3f}\n")
                f.write(f"P-value: {stats['p_value']:.3f}\n")
                f.write(f"Significant difference: {stats['significant_difference']}\n")
            
            f.write("\n## Detailed Results\n\n")
            f.write("| Pair | Similarity | Behavioral Coreference | Interpretation |\n")
            f.write("|------|------------|------------------------|----------------|\n")
            
            for result in self.results:
                pair = f"{result.get('pair', ('', ''))[0]}/{result.get('pair', ('', ''))[1]}"
                similarity = result.get('vector_similarity', 0)
                behavioral = result.get('behavioral_coreference', False)
                interpretation = result.get('interpretation', 'N/A')
                
                f.write(f"| {pair} | {similarity:.3f} | {behavioral} | {interpretation} |\n")
        
        print(f"Report generated: {output_file}")

# Example usage
if __name__ == "__main__":
    # This would typically be used with results from IntentionalityAnalyzer
    metrics = AboutnessMetrics()
    
    # Example results (in practice, you'd get these from running the analyzer)
    example_results = [
        {
            "pair": ("Morning Star", "Evening Star"),
            "vector_similarity": 0.85,
            "behavioral_coreference": True,
            "interpretation": "Strong evidence for tracking reference"
        },
        {
            "pair": ("Superman", "Clark Kent"),
            "vector_similarity": 0.78,
            "behavioral_coreference": True,
            "interpretation": "Strong evidence for tracking reference"
        }
    ]
    
    metrics.add_results(example_results)
    
    # Generate report
    metrics.generate_report()