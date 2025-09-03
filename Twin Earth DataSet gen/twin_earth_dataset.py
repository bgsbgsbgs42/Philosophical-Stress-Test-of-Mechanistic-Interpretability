import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ConceptMapping:
    """Defines how a concept differs between Earth and Twin Earth"""
    earth_referent: str
    twin_referent: str
    earth_properties: List[str]
    twin_properties: List[str]
    earth_contexts: List[str]
    twin_contexts: List[str]

class TwinEarthDatasetGenerator:
    """
    Generates parallel datasets for testing externalism in AI systems.
    
    Based on Putnam's Twin Earth thought experiment:
    - Earth: "water" = H2O
    - Twin Earth: "water" = XYZ (different substance, same surface properties)
    
    Philosophy predicts: If AI concepts are externalist, "water" representations
    should differ between models trained on Earth vs Twin Earth data.
    """
    
    def __init__(self):
        self.concept_mappings = self._define_concept_mappings()
        
    def _define_concept_mappings(self) -> Dict[str, ConceptMapping]:
        """Define how key concepts differ between Earth and Twin Earth"""
        
        return {
            # PRIMARY TEST CASE: Putnam's original water example
            "water": ConceptMapping(
                earth_referent="H2O",
                twin_referent="XYZ", 
                earth_properties=[
                    "composed of hydrogen and oxygen",
                    "molecular formula H2O",
                    "two hydrogen atoms bonded to one oxygen",
                    "splits into hydrogen and oxygen when electrolyzed",
                    "freezes at 0°C due to hydrogen bonding",
                    "has bent molecular geometry",
                    "polar molecule with partial charges"
                ],
                twin_properties=[
                    "composed of xenon, yttrium, and zinc compounds", 
                    "molecular formula XYZ",
                    "complex triple-element bonding structure",
                    "splits into xenon and YZ-compound when electrolyzed",
                    "freezes at 0°C due to metallic bonding",
                    "has crystalline molecular geometry", 
                    "nonpolar molecule with distributed charges"
                ],
                earth_contexts=[
                    "laboratory analysis", "chemistry class", "molecular biology",
                    "environmental science", "hydrology", "chemical reactions"
                ],
                twin_contexts=[
                    "xenochemistry lab", "metallurgy class", "compound physics",
                    "atmospheric studies", "mineralogy", "metallic reactions"
                ]
            ),
            
            # SECONDARY TEST: Gold (natural kind with clear essential properties)
            "gold": ConceptMapping(
                earth_referent="Au (atomic number 79)",
                twin_referent="Tw (atomic number 82)",
                earth_properties=[
                    "atomic number 79",
                    "symbol Au from Latin aurum", 
                    "79 protons in nucleus",
                    "electron configuration [Xe] 4f14 5d10 6s1",
                    "forms metallic bonds",
                    "conducts electricity via electron sea",
                    "malleable due to metallic structure"
                ],
                twin_properties=[
                    "atomic number 82", 
                    "symbol Tw from twin-aurum",
                    "82 protons in nucleus", 
                    "electron configuration [Xe] 4f14 5d10 6s2 6p2",
                    "forms covalent-metallic hybrid bonds",
                    "conducts via hole-electron pairs",
                    "brittle due to directional bonding"
                ],
                earth_contexts=[
                    "periodic table", "atomic physics", "nuclear chemistry",
                    "metallurgy", "electronics", "jewelry making"
                ],
                twin_contexts=[
                    "twin-periodic table", "quantum mechanics", "hybrid chemistry", 
                    "compound engineering", "semiconductors", "crystal crafting"
                ]
            ),
            
            # BIOLOGICAL TEST: Tiger (biological natural kind)
            "tiger": ConceptMapping(
                earth_referent="Panthera tigris",
                twin_referent="Panthera tigris-twin",
                earth_properties=[
                    "DNA sequence specific to Panthera tigris",
                    "mammalian physiology with specialized hunting adaptations", 
                    "carnivorous digestive system",
                    "retractable claws for prey capture",
                    "binocular vision for depth perception",
                    "solitary territorial behavior",
                    "gestation period of 103 days"
                ],
                twin_properties=[
                    "RNA-protein hybrid genetics of Panthera tigris-twin",
                    "reptilian-mammalian physiology with photosynthetic patches",
                    "omnivorous digestive system with chloroplast chambers", 
                    "extendable energy tendrils for nutrient absorption",
                    "compound eyes for multispectral perception",
                    "hive-mind collective behavior",
                    "metamorphic development cycle of 200 days"
                ],
                earth_contexts=[
                    "wildlife biology", "conservation genetics", "predator ecology",
                    "mammalian anatomy", "behavioral studies", "veterinary medicine"
                ],
                twin_contexts=[
                    "hybrid xenobiology", "symbiotic genetics", "photosynthetic ecology",
                    "chimeric anatomy", "collective studies", "metamorphic medicine"
                ]
            ),
            
            # MINERAL TEST: Diamond (crystal structure natural kind)
            "diamond": ConceptMapping(
                earth_referent="carbon in cubic crystal lattice",
                twin_referent="silicon in hexagonal crystal lattice", 
                earth_properties=[
                    "pure carbon atoms in cubic arrangement",
                    "each carbon bonded to four others tetrahedrally",
                    "hardness 10 on Mohs scale due to strong covalent bonds",
                    "high refractive index from dense crystal packing",
                    "electrical insulator due to filled valence bands",
                    "thermal conductor via phonon vibrations",
                    "forms under high pressure and temperature"
                ],
                twin_properties=[
                    "pure silicon atoms in hexagonal arrangement", 
                    "each silicon bonded to six others octahedrally",
                    "hardness 8 on Mohs scale due to metallic-covalent bonds",
                    "low refractive index from loose crystal packing", 
                    "semiconductor due to narrow band gap",
                    "thermal insulator via electron trapping",
                    "forms under low pressure and moderate temperature"
                ],
                earth_contexts=[
                    "crystallography", "materials science", "high-pressure physics",
                    "carbon chemistry", "geology", "optical engineering"
                ],
                twin_contexts=[
                    "silicon crystallography", "semiconductor physics", "low-pressure chemistry",
                    "metalloid science", "mineral studies", "electronic engineering"
                ]
            )
        }
    
    def generate_sentence_templates(self) -> Dict[str, List[str]]:
        """Generate sentence templates for natural concept usage"""
        
        return {
            "descriptive": [
                "The {concept} in the laboratory exhibits {property}.",
                "Scientists discovered that {concept} is characterized by {property}.",
                "Analysis reveals that {concept} demonstrates {property}.",
                "Research shows {concept} possesses {property}.",
                "The {concept} sample displays {property}.",
                "Examination of {concept} indicates {property}.",
                "Studies confirm that {concept} manifests {property}."
            ],
            
            "explanatory": [
                "The {property} of {concept} explains why it behaves this way.",
                "Because {concept} has {property}, we observe these effects.",
                "The {property} in {concept} causes its distinctive characteristics.",
                "Due to {property}, {concept} exhibits unique behavior.",
                "The {property} accounts for {concept}'s special properties.",
                "Given that {concept} contains {property}, we predict these outcomes."
            ],
            
            "causal": [
                "When {concept} undergoes {process}, {property} becomes evident.",
                "The {process} reveals the {property} of {concept}.",
                "Testing {concept} through {process} shows {property}.",
                "The {property} of {concept} emerges during {process}.",
                "Through {process}, we can observe {property} in {concept}."
            ],
            
            "contextual": [
                "In {context}, {concept} is studied for its {property}.",
                "Researchers in {context} focus on {property} of {concept}.",
                "The field of {context} examines {property} in {concept}.",
                "Within {context}, {property} of {concept} is crucial.",
                "Students of {context} learn about {property} in {concept}."
            ],
            
            "comparative": [
                "Unlike other substances, {concept} uniquely shows {property}.",
                "Compared to similar materials, {concept} distinctively has {property}.",
                "What sets {concept} apart is its {property}.",
                "The {property} distinguishes {concept} from alternatives.",
                "Only {concept} exhibits this particular {property}."
            ]
        }
    
    def generate_earth_dataset(self, concept: str, num_samples: int = 500) -> List[str]:
        """Generate training data for Earth version of concept"""
        
        if concept not in self.concept_mappings:
            raise ValueError(f"Concept '{concept}' not defined in mappings")
            
        mapping = self.concept_mappings[concept]
        templates = self.generate_sentence_templates()
        sentences = []
        
        # Generate sentences using Earth properties and contexts
        for _ in range(num_samples):
            template_category = random.choice(list(templates.keys()))
            template = random.choice(templates[template_category])
            
            if template_category == "contextual":
                context = random.choice(mapping.earth_contexts)
                property_item = random.choice(mapping.earth_properties)
                sentence = template.format(
                    concept=concept,
                    context=context, 
                    property=property_item
                )
            elif template_category == "causal":
                property_item = random.choice(mapping.earth_properties)
                process = random.choice([
                    "chemical analysis", "spectroscopic examination", 
                    "structural investigation", "molecular testing",
                    "experimental verification", "laboratory study"
                ])
                sentence = template.format(
                    concept=concept,
                    property=property_item,
                    process=process
                )
            else:
                property_item = random.choice(mapping.earth_properties)
                sentence = template.format(
                    concept=concept,
                    property=property_item
                )
            
            sentences.append(sentence)
            
        # Add direct referent statements
        referent_sentences = [
            f"The {concept} is {mapping.earth_referent}.",
            f"By definition, {concept} refers to {mapping.earth_referent}.",
            f"Scientists identify {concept} as {mapping.earth_referent}.",
            f"The chemical identity of {concept} is {mapping.earth_referent}.",
            f"Technically speaking, {concept} is {mapping.earth_referent}."
        ]
        
        sentences.extend(referent_sentences * 10)  # Emphasize referent
        
        return sentences
    
    def generate_twin_earth_dataset(self, concept: str, num_samples: int = 500) -> List[str]:
        """Generate training data for Twin Earth version of concept"""
        
        if concept not in self.concept_mappings:
            raise ValueError(f"Concept '{concept}' not defined in mappings")
            
        mapping = self.concept_mappings[concept]
        templates = self.generate_sentence_templates()
        sentences = []
        
        # Generate sentences using Twin Earth properties and contexts
        for _ in range(num_samples):
            template_category = random.choice(list(templates.keys()))
            template = random.choice(templates[template_category])
            
            if template_category == "contextual":
                context = random.choice(mapping.twin_contexts)
                property_item = random.choice(mapping.twin_properties)
                sentence = template.format(
                    concept=concept,
                    context=context,
                    property=property_item
                )
            elif template_category == "causal":
                property_item = random.choice(mapping.twin_properties)
                process = random.choice([
                    "xenochemical analysis", "quantum spectroscopy",
                    "hybrid investigation", "compound testing", 
                    "metamorphic verification", "advanced study"
                ])
                sentence = template.format(
                    concept=concept,
                    property=property_item,
                    process=process
                )
            else:
                property_item = random.choice(mapping.twin_properties)
                sentence = template.format(
                    concept=concept,
                    property=property_item
                )
            
            sentences.append(sentence)
            
        # Add direct referent statements 
        referent_sentences = [
            f"The {concept} is {mapping.twin_referent}.",
            f"By definition, {concept} refers to {mapping.twin_referent}.",
            f"Scientists identify {concept} as {mapping.twin_referent}.",
            f"The chemical identity of {concept} is {mapping.twin_referent}.",
            f"Technically speaking, {concept} is {mapping.twin_referent}."
        ]
        
        sentences.extend(referent_sentences * 10)  # Emphasize referent
        
        return sentences
    
    def generate_surface_property_controls(self, concept: str, num_samples: int = 200) -> Tuple[List[str], List[str]]:
        """
        Generate control datasets with identical surface properties but different referents.
        This tests whether AI tracks deep vs. surface features.
        """
        
        # Shared surface properties (what humans typically observe)
        surface_properties = {
            "water": [
                "clear and transparent", "liquid at room temperature",
                "freezes at 0 degrees Celsius", "boils at 100 degrees Celsius",
                "tasteless and odorless", "essential for life",
                "flows and takes shape of container", "reflects light"
            ],
            "gold": [
                "yellow and shiny", "does not rust or tarnish", 
                "heavy and dense", "malleable and ductile",
                "valuable and precious", "used in jewelry",
                "conducts electricity well", "soft enough to scratch"
            ],
            "tiger": [
                "large carnivorous cat", "orange with black stripes",
                "powerful and muscular", "solitary hunter", 
                "found in Asian forests", "endangered species",
                "distinctive roar", "excellent night vision"
            ],
            "diamond": [
                "extremely hard substance", "brilliant and sparkling",
                "colorless when pure", "used in jewelry",
                "cuts through other materials", "very expensive",
                "refracts light beautifully", "symbol of luxury"
            ]
        }
        
        templates = [
            "The {concept} appears {surface_property}.",
            "People observe that {concept} is {surface_property}.",
            "Visually, {concept} looks {surface_property}.",
            "To the naked eye, {concept} seems {surface_property}.",
            "The {concept} is known for being {surface_property}.",
            "Everyone recognizes {concept} as {surface_property}."
        ]
        
        earth_surface = []
        twin_surface = []
        
        # Generate identical surface descriptions for both worlds
        for _ in range(num_samples):
            template = random.choice(templates)
            surface_prop = random.choice(surface_properties[concept])
            sentence = template.format(concept=concept, surface_property=surface_prop)
            
            earth_surface.append(sentence)
            twin_surface.append(sentence)  # Identical on purpose
        
        return earth_surface, twin_surface
    
    def generate_linguistic_controls(self, concept: str, num_samples: int = 200) -> Tuple[List[str], List[str]]:
        """
        Generate control datasets with different linguistic patterns but same referent.
        This tests whether AI tracks semantics vs. syntax.
        """
        
        # Different linguistic styles for same concept
        earth_styles = [
            "scientific", "formal", "technical", "academic"
        ]
        
        twin_styles = [
            "colloquial", "informal", "conversational", "everyday"  
        ]
        
        base_facts = self.concept_mappings[concept].earth_properties
        
        earth_linguistic = []
        twin_linguistic = []
        
        for _ in range(num_samples):
            fact = random.choice(base_facts)
            
            # Earth: formal/scientific language
            earth_templates = [
                f"Scientific analysis demonstrates that {concept} exhibits {fact}.",
                f"Rigorous experimentation confirms {concept} possesses {fact}.",
                f"Empirical investigation reveals {concept} manifests {fact}.",
                f"Laboratory studies establish that {concept} displays {fact}."
            ]
            
            # Twin Earth: informal/colloquial language 
            twin_templates = [
                f"Everyone knows {concept} has {fact}.",
                f"It's obvious that {concept} shows {fact}.",
                f"People say {concept} is all about {fact}.",
                f"You can tell {concept} has {fact}."
            ]
            
            earth_linguistic.append(random.choice(earth_templates))
            twin_linguistic.append(random.choice(twin_templates))
        
        return earth_linguistic, twin_linguistic
    
    def create_full_dataset(self, concepts: List[str] = None, samples_per_concept: int = 500) -> Dict:
        """Create complete Twin Earth dataset for multiple concepts"""
        
        if concepts is None:
            concepts = list(self.concept_mappings.keys())
        
        dataset = {
            "earth_data": {},
            "twin_earth_data": {},
            "surface_controls": {},
            "linguistic_controls": {},
            "metadata": {
                "total_concepts": len(concepts),
                "samples_per_concept": samples_per_concept,
                "philosophical_prediction": "If externalism is correct, concept vectors should differ between Earth and Twin Earth models despite identical surface properties",
                "control_prediction": "Surface and linguistic controls should not affect concept vectors if AI tracks genuine referents"
            }
        }
        
        for concept in concepts:
            print(f"Generating data for concept: {concept}")
            
            # Core datasets
            dataset["earth_data"][concept] = self.generate_earth_dataset(concept, samples_per_concept)
            dataset["twin_earth_data"][concept] = self.generate_twin_earth_dataset(concept, samples_per_concept)
            
            # Control datasets
            surface_earth, surface_twin = self.generate_surface_property_controls(concept, 200)
            dataset["surface_controls"][concept] = {
                "earth": surface_earth,
                "twin_earth": surface_twin
            }
            
            linguistic_earth, linguistic_twin = self.generate_linguistic_controls(concept, 200)
            dataset["linguistic_controls"][concept] = {
                "earth": linguistic_earth, 
                "twin_earth": linguistic_twin
            }
        
        return dataset
    
    def save_dataset(self, dataset: Dict, filename: str = "twin_earth_dataset.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filename}")
    
    def load_dataset(self, filename: str = "twin_earth_dataset.json") -> Dict:
        """Load dataset from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)

# Example usage and testing
if __name__ == "__main__":
    # Create dataset generator
    generator = TwinEarthDatasetGenerator()
    
    # Generate sample data for water concept
    print("Earth dataset sample for 'water':")
    earth_water = generator.generate_earth_dataset("water", 10)
    for i, sentence in enumerate(earth_water[:5]):
        print(f"{i+1}. {sentence}")
    
    print("\nTwin Earth dataset sample for 'water':")
    twin_water = generator.generate_twin_earth_dataset("water", 10)
    for i, sentence in enumerate(twin_water[:5]):
        print(f"{i+1}. {sentence}")
    
    print("\nSurface property controls:")
    surface_earth, surface_twin = generator.generate_surface_property_controls("water", 5)
    print("Earth surface:", surface_earth[0])
    print("Twin surface:", surface_twin[0])
    
    # Generate full dataset
    print("\nGenerating full dataset...")
    full_dataset = generator.create_full_dataset(["water", "gold"], samples_per_concept=100)
    
    # Save dataset
    generator.save_dataset(full_dataset, "sample_twin_earth_dataset.json")
    
    print("Dataset generation complete!")
    print(f"Total concepts: {len(full_dataset['earth_data'])}")
    print(f"Sample count per concept: {len(full_dataset['earth_data']['water'])}")
