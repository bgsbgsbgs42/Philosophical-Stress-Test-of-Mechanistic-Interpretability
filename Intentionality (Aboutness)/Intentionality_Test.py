import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformer_lens import HookedTransformer
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns


class IntentionalityAnalyzer:
    def __init__(self, model_name="pythia-70m", layer=-6):
        """
        Initialize the intentionality analyzer with a model
        """
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model_name = model_name
        # Convert negative layer index to positive
        num_layers = self.model.cfg.n_layers
        if layer < 0:
            self.layer = num_layers + layer
        else:
            self.layer = layer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        
    def get_concept_vector(self, concept_term, context_sentences=None):
        """
        Extract concept vector using multiple contextual examples
        """
        if context_sentences is None:
            # Default context sentences if none provided
            context_sentences = [
                f"The {concept_term} is known to be",
                f"People think about {concept_term} as",
                f"When considering {concept_term}, one might say"
            ]
        
        all_activations = []
        for context in context_sentences:
            tokens = self.model.to_tokens(context).to(self.device)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)
                
            # Get residual stream activations at specified layer
            activation = cache[f"blocks.{self.layer}.hook_resid_post"][0, -1, :].cpu().numpy()
            all_activations.append(activation)
        
        # Average across contexts to get more robust representation
        concept_vector = np.mean(all_activations, axis=0)
        return concept_vector
    
    def coreference_test(self, coreference_pairs, num_examples=5):
        """
        Test if features track reference vs. mode of presentation
        """
        results = []
        
        for term1, term2 in coreference_pairs:
            # Generate context sentences for each term
            context1 = self._generate_context_sentences(term1, num_examples)
            context2 = self._generate_context_sentences(term2, num_examples)
            
            # Get concept vectors
            vec1 = self.get_concept_vector(term1, context1)
            vec2 = self.get_concept_vector(term2, context2)
            
            # Calculate similarity
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            
            # Test if model "knows" they're coreferential
            knows_coreference = self._test_coreference_knowledge(term1, term2)
            
            results.append({
                "pair": (term1, term2),
                "vector_similarity": similarity,
                "behavioral_coreference": knows_coreference,
                "interpretation": self._interpret_result(similarity, knows_coreference)
            })
        
        return results
    
    def _generate_context_sentences(self, term, num_examples):
        """
        Generate context sentences for a given term
        """
        # This could be expanded with more diverse templates
        templates = [
            f"The {term} is known to be",
            f"People think about {term} as",
            f"When considering {term}, one might say",
            f"In many contexts, {term} represents",
            f"The concept of {term} involves"
        ]
        
        return templates[:num_examples]
    
    def _test_coreference_knowledge(self, term1, term2):
        """
        Test if the model behaviorally demonstrates knowledge of coreference
        """
        # Test prompts that should reveal coreference knowledge
        test_prompts = [
            f"{term1} and {term2} are",
            f"If {term1} is the same as {term2}, then",
            f"The relationship between {term1} and {term2} is"
        ]
        
        coreference_scores = []
        for prompt in test_prompts:
            tokens = self.model.to_tokens(prompt).to(self.device)
            with torch.no_grad():
                logits = self.model(tokens)
            
            # Get next token predictions
            next_token_logits = logits[0, -1, :]
            top_tokens = torch.topk(next_token_logits, 10)
            
            # Convert to strings
            token_strings = [self.model.tokenizer.decode(token) for token in top_tokens.indices]
            
            # Score based on presence of coreference indicators
            score = self._score_coreference_indicators(token_strings)
            coreference_scores.append(score)
        
        # Average score across prompts
        return np.mean(coreference_scores) > 0.5  # Threshold for "knows"
    
    def _score_coreference_indicators(self, token_strings):
        """
        Score how likely tokens indicate coreference knowledge
        """
        positive_indicators = ["same", "identical", "equivalent", "equal", "one", "single"]
        negative_indicators = ["different", "distinct", "separate", "another", "other"]
        
        score = 0
        for token in token_strings:
            token_lower = token.lower()
            if any(indicator in token_lower for indicator in positive_indicators):
                score += 1
            if any(indicator in token_lower for indicator in negative_indicators):
                score -= 1
        
        return score / len(token_strings)  # Normalize
    
    def _interpret_result(self, similarity, knows_coreference):
        """
        Interpret the results based on similarity and behavioral test
        """
        if similarity > 0.7 and knows_coreference:
            return "Strong evidence for tracking reference"
        elif similarity > 0.7 and not knows_coreference:
            return "Surface similarity without behavioral coreference"
        elif similarity < 0.3 and knows_coreference:
            return "Different modes of presentation for same referent"
        elif similarity < 0.3 and not knows_coreference:
            return "Different concepts with different referents"
        else:
            return "Ambiguous - partial alignment"
    
    def fictional_vs_real_test(self, pairs):
        """
        Test representational differences for fictional vs real entities
        """
        results = []
        
        for fictional, real in pairs:
            # Get concept vectors
            fictional_vec = self.get_concept_vector(fictional)
            real_vec = self.get_concept_vector(real)
            
            # Calculate similarity
            similarity = cosine_similarity([fictional_vec], [real_vec])[0][0]
            
            # Test uncertainty/confidence
            fictional_uncertainty = self._measure_uncertainty(fictional)
            real_uncertainty = self._measure_uncertainty(real)
            
            results.append({
                "pair": (fictional, real),
                "similarity": similarity,
                "fictional_uncertainty": fictional_uncertainty,
                "real_uncertainty": real_uncertainty,
                "uncertainty_difference": abs(fictional_uncertainty - real_uncertainty)
            })
        
        return results
    
    def _measure_uncertainty(self, concept):
        """
        Measure model uncertainty about a concept using entropy of predictions
        """
        test_prompts = [
            f"{concept} is",
            f"The truth about {concept} is",
            f"Scientists know that {concept} is"
        ]
        
        entropies = []
        for prompt in test_prompts:
            tokens = self.model.to_tokens(prompt).to(self.device)
            with torch.no_grad():
                logits = self.model(tokens)
            
            # Calculate entropy of next token distribution
            next_token_logits = logits[0, -1, :]
            probabilities = torch.softmax(next_token_logits, dim=-1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
            entropies.append(entropy.item())
        
        return np.mean(entropies)
    
    def visualize_results(self, coreference_results, fictional_results):
        """
        Create visualizations for intentionality results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Coreference similarity plot
        pairs = [f"{p['pair'][0]}/{p['pair'][1]}" for p in coreference_results]
        similarities = [p['vector_similarity'] for p in coreference_results]
        knows_coreference = [p['behavioral_coreference'] for p in coreference_results]
        
        colors = ['green' if know else 'red' for know in knows_coreference]
        ax1.bar(pairs, similarities, color=colors)
        ax1.set_title('Coreference Pair Similarities')
        ax1.set_ylabel('Cosine Similarity')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.7)
        ax1.axhline(y=0.3, color='gray', linestyle='--', alpha=0.7)
        
        # Fictional vs real similarity
        fictional_pairs = [f"{p['pair'][0]}/{p['pair'][1]}" for p in fictional_results]
        fictional_similarities = [p['similarity'] for p in fictional_results]
        ax2.bar(fictional_pairs, fictional_similarities)
        ax2.set_title('Fictional vs Real Entity Similarities')
        ax2.set_ylabel('Cosine Similarity')
        ax2.tick_params(axis='x', rotation=45)
        
        # Uncertainty comparison
        fictional_uncertainties = [p['fictional_uncertainty'] for p in fictional_results]
        real_uncertainties = [p['real_uncertainty'] for p in fictional_results]
        x = np.arange(len(fictional_pairs))
        width = 0.35
        ax3.bar(x - width/2, fictional_uncertainties, width, label='Fictional')
        ax3.bar(x + width/2, real_uncertainties, width, label='Real')
        ax3.set_title('Uncertainty in Fictional vs Real Entities')
        ax3.set_ylabel('Entropy (Uncertainty)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(fictional_pairs, rotation=45)
        ax3.legend()
        
        # Behavioral vs vector similarity comparison
        behavioral_agreement = [
            p['vector_similarity'] > 0.7 and p['behavioral_coreference'] or
            p['vector_similarity'] < 0.3 and not p['behavioral_coreference']
            for p in coreference_results
        ]
        agreement_labels = ['Agree', 'Disagree']
        agreement_counts = [sum(behavioral_agreement), len(behavioral_agreement) - sum(behavioral_agreement)]
        ax4.pie(agreement_counts, labels=agreement_labels, autopct='%1.1f%%')
        ax4.set_title('Behavioral vs Vector Similarity Agreement')
        
        plt.tight_layout()
        return fig

# Example usage and test
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = IntentionalityAnalyzer(model_name="pythia-70m")
    
    # Define test pairs
    coreference_pairs = [
        ("Morning Star", "Evening Star"),  # Same referent (Venus)
        ("Superman", "Clark Kent"),        # Fictional coreference
        ("the current president", "Joe Biden"),  # Definite description
        ("the author of Hamlet", "Shakespeare"), # Description vs name
        ("the first man on the moon", "Neil Armstrong") # Achievement description
    ]
    
    fictional_vs_real_pairs = [
        ("unicorn", "horse"),
        ("Sherlock Holmes", "Arthur Conan Doyle"),
        ("Hogwarts", "Oxford"),
        ("lightsaber", "sword"),
        ("fairy", "butterfly")
    ]
    
    # Run tests
    print("Running coreference tests...")
    coreference_results = analyzer.coreference_test(coreference_pairs)
    
    print("Running fictional vs real tests...")
    fictional_results = analyzer.fictional_vs_real_test(fictional_vs_real_pairs)
    
    # Display results
    print("\n=== COREFERENCE TEST RESULTS ===")
    for result in coreference_results:
        print(f"{result['pair'][0]} / {result['pair'][1]}:")
        print(f"  Similarity: {result['vector_similarity']:.3f}")
        print(f"  Behavioral coreference: {result['behavioral_coreference']}")
        print(f"  Interpretation: {result['interpretation']}")
        print()
    
    print("\n=== FICTIONAL VS REAL TEST RESULTS ===")
    for result in fictional_results:
        print(f"{result['pair'][0]} / {result['pair'][1]}:")
        print(f"  Similarity: {result['similarity']:.3f}")
        print(f"  Fictional uncertainty: {result['fictional_uncertainty']:.3f}")
        print(f"  Real uncertainty: {result['real_uncertainty']:.3f}")
        print(f"  Uncertainty difference: {result['uncertainty_difference']:.3f}")
        print()
    
    # Create visualizations
    fig = analyzer.visualize_results(coreference_results, fictional_results)
    plt.savefig("intentionality_results.png", dpi=300, bbox_inches='tight')
    plt.show()