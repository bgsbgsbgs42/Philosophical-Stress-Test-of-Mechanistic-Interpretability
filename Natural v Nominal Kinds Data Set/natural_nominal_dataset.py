import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class KindType(Enum):
    NATURAL = "natural"
    NOMINAL = "nominal"
    ARTIFACT = "artifact"  # Special subtype of nominal

@dataclass
class ConceptDefinition:
    """Defines a concept with its essential vs superficial properties"""
    name: str
    kind_type: KindType
    essential_properties: List[str]  # Deep, unchanging properties that determine kind membership
    superficial_properties: List[str]  # Surface properties that can vary
    scientific_properties: List[str]  # Properties discovered by science
    folk_properties: List[str]  # Properties known to ordinary people
    edge_cases: List[str]  # Borderline instances that test category boundaries
    typical_instances: List[str]  # Clear, central examples
    atypical_instances: List[str]  # Legitimate but unusual examples

class NaturalNominalDatasetGenerator:
    """
    Generates datasets for testing whether AI tracks essential vs. superficial properties
    in natural kinds (water, gold, tiger) vs. nominal kinds (chair, game, bachelor).
    
    Philosophical Predictions:
    - Natural kinds: AI should be more sensitive to essential than superficial properties
    - Nominal kinds: AI should be more sensitive to functional/definitional than accidental properties
    - Essential properties should remain stable across contexts
    - Superficial properties should vary without affecting core concept
    """
    
    def __init__(self):
        self.concept_definitions = self._define_concepts()
        
    def _define_concepts(self) -> Dict[str, ConceptDefinition]:
        """Define comprehensive concept mappings for natural vs nominal kinds"""
        
        return {
            # NATURAL KINDS - Should show essential property sensitivity
            
            "water": ConceptDefinition(
                name="water",
                kind_type=KindType.NATURAL,
                essential_properties=[
                    "molecular formula H2O",
                    "two hydrogen atoms bonded to one oxygen atom", 
                    "hydrogen-oxygen covalent bonds",
                    "bent molecular geometry due to lone pairs",
                    "polar molecule with partial charges",
                    "forms hydrogen bonds between molecules",
                    "chemical identity determined by atomic composition"
                ],
                superficial_properties=[
                    "clear and colorless liquid",
                    "tasteless and odorless", 
                    "flows and takes container shape",
                    "feels wet to touch",
                    "makes splashing sounds",
                    "reflects light when still",
                    "commonly found in bottles"
                ],
                scientific_properties=[
                    "boiling point 100°C at standard pressure",
                    "freezing point 0°C at standard pressure",
                    "density 1.0 g/cm³ at standard conditions",
                    "specific heat capacity 4.18 J/g°C",
                    "dielectric constant of 81",
                    "self-ionizes to form H+ and OH- ions"
                ],
                folk_properties=[
                    "essential for life",
                    "used for drinking and cooking",
                    "falls as rain from clouds", 
                    "found in rivers and oceans",
                    "can be hot or cold",
                    "extinguishes fires"
                ],
                edge_cases=[
                    "heavy water (D2O)",
                    "water vapor",
                    "ice crystals",
                    "water mixed with impurities",
                    "water at extreme temperatures"
                ],
                typical_instances=[
                    "tap water", "bottled water", "rainwater", "ocean water"
                ],
                atypical_instances=[
                    "steam", "ice", "mineral water", "distilled water"
                ]
            ),
            
            "gold": ConceptDefinition(
                name="gold",
                kind_type=KindType.NATURAL,
                essential_properties=[
                    "atomic number 79",
                    "chemical symbol Au",
                    "79 protons in atomic nucleus",
                    "electron configuration [Xe] 4f¹⁴ 5d¹⁰ 6s¹",
                    "metallic bonding structure",
                    "face-centered cubic crystal lattice",
                    "nuclear charge determines all other properties"
                ],
                superficial_properties=[
                    "yellow metallic color",
                    "shiny and lustrous appearance",
                    "feels heavy and dense",
                    "soft enough to scratch with fingernail",
                    "valuable and expensive",
                    "used in jewelry and decoration",
                    "associated with wealth and status"
                ],
                scientific_properties=[
                    "density 19.3 g/cm³",
                    "melting point 1064°C", 
                    "excellent electrical conductor",
                    "chemically inert and non-reactive",
                    "malleable and ductile",
                    "atomic mass 196.97 amu"
                ],
                folk_properties=[
                    "precious metal",
                    "doesn't rust or tarnish",
                    "found by mining",
                    "used in coins and jewelry",
                    "symbol of value",
                    "desired throughout history"
                ],
                edge_cases=[
                    "gold alloys", "gold nanoparticles", "ionic gold compounds",
                    "gold leaf", "white gold", "rose gold"
                ],
                typical_instances=[
                    "gold bars", "gold coins", "gold jewelry", "gold nuggets"
                ],
                atypical_instances=[
                    "gold paint", "gold thread", "gold dental fillings", "gold electronics"
                ]
            ),
            
            "tiger": ConceptDefinition(
                name="tiger",
                kind_type=KindType.NATURAL,
                essential_properties=[
                    "DNA sequence characteristic of Panthera tigris",
                    "mammalian physiology and anatomy",
                    "carnivorous digestive system",
                    "feline skeletal and muscular structure", 
                    "specific genetic lineage and evolutionary history",
                    "reproductively isolated species",
                    "obligate carnivore metabolism"
                ],
                superficial_properties=[
                    "orange fur with black stripes",
                    "large size and muscular build",
                    "distinctive facial markings",
                    "long tail with black rings",
                    "fierce and intimidating appearance",
                    "solitary and territorial behavior",
                    "found in Asian forests"
                ],
                scientific_properties=[
                    "gestation period approximately 103 days",
                    "average lifespan 10-15 years in wild",
                    "body length 1.4-2.8 meters",
                    "weight 90-300 kg depending on subspecies",
                    "night vision adaptations",
                    "specialized hunting dentition"
                ],
                folk_properties=[
                    "dangerous predator",
                    "king of the jungle",
                    "excellent hunter",
                    "endangered species",
                    "lives in Asia",
                    "featured in stories and myths"
                ],
                edge_cases=[
                    "white tigers", "tiger cubs", "tiger-lion hybrids",
                    "extinct tiger subspecies", "tigers in captivity"
                ],
                typical_instances=[
                    "Bengal tiger", "Siberian tiger", "wild adult tiger", "hunting tiger"
                ],
                atypical_instances=[
                    "tiger cub", "white tiger", "paper tiger", "tiger in zoo"
                ]
            ),
            
            # NOMINAL KINDS - Should show functional/definitional property sensitivity
            
            "chair": ConceptDefinition(
                name="chair",
                kind_type=KindType.NOMINAL,
                essential_properties=[
                    "designed for sitting",
                    "supports human body weight",
                    "elevated seating surface",
                    "provides back support",
                    "intended for single person use",
                    "stable base structure",
                    "functional purpose of seating"
                ],
                superficial_properties=[
                    "made of wood or metal",
                    "has four legs",
                    "brown or black color",
                    "specific style or design",
                    "particular size or height",
                    "found in dining rooms",
                    "matches other furniture"
                ],
                scientific_properties=[
                    "material strength and durability",
                    "ergonomic design principles",
                    "load-bearing capacity",
                    "center of gravity calculations",
                    "stress distribution patterns",
                    "material fatigue characteristics"
                ],
                folk_properties=[
                    "furniture for sitting",
                    "found in homes and offices",
                    "comes in many styles",
                    "can be moved around",
                    "part of table and chair sets",
                    "requires assembly or is pre-made"
                ],
                edge_cases=[
                    "bean bag chairs", "rocking chairs", "folding chairs",
                    "electric chairs", "wheelchair", "high chairs"
                ],
                typical_instances=[
                    "dining chair", "office chair", "kitchen chair", "wooden chair"
                ],
                atypical_instances=[
                    "beanbag", "stool", "bench", "throne"
                ]
            ),
            
            "game": ConceptDefinition(
                name="game",
                kind_type=KindType.NOMINAL,
                essential_properties=[
                    "set of rules governing play",
                    "voluntary participation",
                    "defined winning conditions or objectives",
                    "structured activity with constraints",
                    "competitive or skill-based elements",
                    "distinct beginning and end",
                    "artificial rather than natural activity"
                ],
                superficial_properties=[
                    "played with specific equipment",
                    "involves multiple players",
                    "takes certain amount of time",
                    "played in particular locations",
                    "associated with fun and entertainment",
                    "has particular cultural associations",
                    "involves physical or mental activity"
                ],
                scientific_properties=[
                    "game theory mathematical principles",
                    "strategic decision-making patterns",
                    "probability and chance mechanisms",
                    "psychological engagement factors",
                    "learning and skill development",
                    "social interaction dynamics"
                ],
                folk_properties=[
                    "played for fun and entertainment",
                    "can be competitive or cooperative",
                    "learned through practice",
                    "brings people together",
                    "can be won or lost",
                    "helps pass time"
                ],
                edge_cases=[
                    "video games", "mind games", "war games",
                    "children's games", "gambling games", "party games"
                ],
                typical_instances=[
                    "chess", "soccer", "poker", "Monopoly"
                ],
                atypical_instances=[
                    "solitaire", "video game", "drinking game", "word game"
                ]
            ),
            
            "bachelor": ConceptDefinition(
                name="bachelor",
                kind_type=KindType.NOMINAL,
                essential_properties=[
                    "adult human male",
                    "unmarried status",
                    "eligible for marriage",
                    "not currently in marriage contract",
                    "legal and social status definition",
                    "socially recognized category",
                    "definitional rather than empirical kind"
                ],
                superficial_properties=[
                    "lives alone",
                    "particular age range",
                    "specific lifestyle choices",
                    "dating behavior patterns",
                    "social activities and preferences",
                    "economic status or career",
                    "personal appearance or habits"
                ],
                scientific_properties=[
                    "demographic classification",
                    "legal status implications",
                    "sociological category",
                    "statistical population grouping",
                    "cultural variation in definition",
                    "historical changes in concept"
                ],
                folk_properties=[
                    "single man",
                    "available for dating",
                    "not tied down",
                    "independent lifestyle",
                    "potential husband",
                    "unattached male"
                ],
                edge_cases=[
                    "divorced men", "widowers", "committed but unmarried men",
                    "gay bachelors", "bachelor priests", "elderly bachelors"
                ],
                typical_instances=[
                    "young unmarried man", "dating bachelor", "eligible bachelor"
                ],
                atypical_instances=[
                    "confirmed bachelor", "elderly bachelor", "bachelor by choice"
                ]
            ),
            
            # ARTIFACT KINDS - Functional but human-made
            
            "clock": ConceptDefinition(
                name="clock",
                kind_type=KindType.ARTIFACT,
                essential_properties=[
                    "designed to measure and display time",
                    "systematic time-keeping mechanism",
                    "calibrated to standard time units",
                    "intended function of temporal measurement",
                    "human-created artifact",
                    "purposeful design for time indication",
                    "functional rather than natural kind"
                ],
                superficial_properties=[
                    "round face with numbers",
                    "has hour and minute hands",
                    "makes ticking sounds",
                    "hangs on wall or sits on surface",
                    "particular size or color",
                    "specific style or appearance",
                    "made of certain materials"
                ],
                scientific_properties=[
                    "mechanical or electronic timing mechanism",
                    "oscillation frequency for time keeping",
                    "gear ratios for hand movement",
                    "power source requirements",
                    "accuracy and precision specifications",
                    "temperature and environmental stability"
                ],
                folk_properties=[
                    "tells time",
                    "helps people be on schedule",
                    "found in homes and public places",
                    "can be analog or digital",
                    "requires winding or batteries",
                    "important for daily life"
                ],
                edge_cases=[
                    "digital clocks", "atomic clocks", "sundials",
                    "broken clocks", "decorative clocks", "clock towers"
                ],
                typical_instances=[
                    "wall clock", "alarm clock", "grandfather clock", "wristwatch"
                ],
                atypical_instances=[
                    "sundial", "atomic clock", "cuckoo clock", "digital display"
                ]
            ),
            
            "hammer": ConceptDefinition(
                name="hammer",
                kind_type=KindType.ARTIFACT,
                essential_properties=[
                    "designed for striking and driving",
                    "weighted head for impact force",
                    "handle for leverage and control",
                    "intended function of pounding",
                    "tool designed by humans",
                    "force multiplication device",
                    "purposeful artifact with clear function"
                ],
                superficial_properties=[
                    "made of metal and wood",
                    "particular size and weight",
                    "specific handle length",
                    "certain color or finish",
                    "found in toolboxes",
                    "associated with construction",
                    "has brand markings"
                ],
                scientific_properties=[
                    "mechanical advantage through leverage",
                    "kinetic energy transfer mechanisms", 
                    "material strength and durability",
                    "ergonomic design principles",
                    "impact force calculations",
                    "vibration dampening properties"
                ],
                folk_properties=[
                    "tool for hitting nails",
                    "used in construction",
                    "every household should have one",
                    "symbol of building and making",
                    "requires skill to use effectively",
                    "can be dangerous if misused"
                ],
                edge_cases=[
                    "sledgehammer", "ball-peen hammer", "rubber hammer",
                    "decorative hammer", "broken hammer", "toy hammer"
                ],
                typical_instances=[
                    "claw hammer", "carpenter's hammer", "construction hammer"
                ],
                atypical_instances=[
                    "sledgehammer", "tack hammer", "rubber mallet", "war hammer"
                ]
            )
        }
    
    def generate_essential_property_contexts(self, concept: str, num_samples: int = 200) -> List[str]:
        """Generate contexts emphasizing essential properties"""
        
        if concept not in self.concept_definitions:
            raise ValueError(f"Concept '{concept}' not defined")
        
        definition = self.concept_definitions[concept]
        templates = [
            "Scientific analysis reveals that {concept} is fundamentally characterized by {essential_property}.",
            "The defining feature of {concept} is {essential_property}.",
            "What makes {concept} what it is: {essential_property}.",
            "The essential nature of {concept} involves {essential_property}.",
            "Researchers have determined that {concept} necessarily exhibits {essential_property}.",
            "The core property that defines {concept} is {essential_property}.",
            "Without {essential_property}, something cannot be considered {concept}.",
            "The invariant characteristic of {concept} is {essential_property}.",
            "Deep investigation shows {concept} is essentially {essential_property}.",
            "The fundamental property underlying {concept} is {essential_property}."
        ]
        
        contexts = []
        for _ in range(num_samples):
            template = random.choice(templates)
            essential_prop = random.choice(definition.essential_properties)
            context = template.format(concept=concept, essential_property=essential_prop)
            contexts.append(context)
        
        return contexts
    
    def generate_superficial_property_contexts(self, concept: str, num_samples: int = 200) -> List[str]:
        """Generate contexts emphasizing superficial/accidental properties"""
        
        if concept not in self.concept_definitions:
            raise ValueError(f"Concept '{concept}' not defined")
        
        definition = self.concept_definitions[concept]
        templates = [
            "This particular {concept} happens to be {superficial_property}.",
            "The {concept} appears {superficial_property} in this instance.",
            "One can observe that this {concept} is {superficial_property}.",
            "The {concept} looks {superficial_property} from this angle.",
            "This example of {concept} shows {superficial_property}.",
            "The surface characteristics include {concept} being {superficial_property}.",
            "Visually, the {concept} presents as {superficial_property}.",
            "This {concept} exhibits the superficial trait of being {superficial_property}.",
            "The apparent quality of this {concept} is {superficial_property}.",
            "One notices the {concept} has the accidental property of being {superficial_property}."
        ]
        
        contexts = []
        for _ in range(num_samples):
            template = random.choice(templates)
            superficial_prop = random.choice(definition.superficial_properties)
            context = template.format(concept=concept, superficial_property=superficial_prop)
            contexts.append(context)
        
        return contexts
    
    def generate_scientific_vs_folk_contexts(self, concept: str, num_samples: int = 100) -> Tuple[List[str], List[str]]:
        """Generate contexts contrasting scientific vs folk understanding"""
        
        definition = self.concept_definitions[concept]
        
        scientific_templates = [
            "Scientific research shows that {concept} is characterized by {scientific_property}.",
            "Laboratory analysis reveals {concept} exhibits {scientific_property}.",
            "Empirical studies demonstrate that {concept} involves {scientific_property}.",
            "Advanced measurement techniques show {concept} has {scientific_property}.",
            "The scientific understanding of {concept} includes {scientific_property}.",
            "Rigorous investigation confirms {concept} displays {scientific_property}."
        ]
        
        folk_templates = [
            "People generally know that {concept} is {folk_property}.",
            "Everyone understands {concept} as {folk_property}.",
            "Common knowledge suggests {concept} is {folk_property}.",
            "Folk wisdom tells us {concept} is {folk_property}.",
            "The everyday understanding of {concept} includes being {folk_property}.",
            "Regular people recognize {concept} as {folk_property}."
        ]
        
        scientific_contexts = []
        folk_contexts = []
        
        for _ in range(num_samples):
            # Scientific contexts
            sci_template = random.choice(scientific_templates)
            sci_prop = random.choice(definition.scientific_properties)
            sci_context = sci_template.format(concept=concept, scientific_property=sci_prop)
            scientific_contexts.append(sci_context)
            
            # Folk contexts
            folk_template = random.choice(folk_templates)
            folk_prop = random.choice(definition.folk_properties)
            folk_context = folk_template.format(concept=concept, folk_property=folk_prop)
            folk_contexts.append(folk_context)
        
        return scientific_contexts, folk_contexts
    
    def generate_edge_case_contexts(self, concept: str, num_samples: int = 100) -> List[str]:
        """Generate contexts with edge cases that test category boundaries"""
        
        definition = self.concept_definitions[concept]
        templates = [
            "Consider this borderline case: {edge_case} as an example of {concept}.",
            "The edge case of {edge_case} challenges our understanding of {concept}.",
            "Is {edge_case} really a type of {concept}? This tests the boundaries.",
            "The marginal instance {edge_case} raises questions about {concept} membership.",
            "Philosophers debate whether {edge_case} counts as {concept}.",
            "The boundary case of {edge_case} illuminates the nature of {concept}.",
            "Consider the atypical example: {edge_case} classified as {concept}.",
            "The problematic case of {edge_case} tests our concept of {concept}."
        ]
        
        contexts = []
        for _ in range(num_samples):
            template = random.choice(templates)
            edge_case = random.choice(definition.edge_cases)
            context = template.format(concept=concept, edge_case=edge_case)
            contexts.append(context)
        
        return contexts
    
    def generate_typicality_gradient_contexts(self, concept: str, num_samples: int = 150) -> Dict[str, List[str]]:
        """Generate contexts testing typicality effects (prototype theory)"""
        
        definition = self.concept_definitions[concept]
        
        typical_templates = [
            "A perfect example of {concept} is {typical_instance}.",
            "When people think of {concept}, they usually imagine {typical_instance}.",
            "The prototypical {concept} would be {typical_instance}.",
            "A clear case of {concept} is {typical_instance}.",
            "The best example of {concept} is {typical_instance}.",
            "A central instance of {concept} is {typical_instance}."
        ]
        
        atypical_templates = [
            "An unusual example of {concept} is {atypical_instance}.",
            "A less typical {concept} would be {atypical_instance}.",
            "An atypical instance of {concept} is {atypical_instance}.",
            "A marginal example of {concept} is {atypical_instance}.",
            "A peripheral case of {concept} is {atypical_instance}.",
            "An uncommon type of {concept} is {atypical_instance}."
        ]
        
        typical_contexts = []
        atypical_contexts = []
        
        for _ in range(num_samples // 2):
            # Typical contexts
            typ_template = random.choice(typical_templates)
            typ_instance = random.choice(definition.typical_instances)
            typ_context = typ_template.format(concept=concept, typical_instance=typ_instance)
            typical_contexts.append(typ_context)
            
            # Atypical contexts
            atyp_template = random.choice(atypical_templates)
            atyp_instance = random.choice(definition.atypical_instances)
            atyp_context = atyp_template.format(concept=concept, atypical_instance=atyp_instance)
            atypical_contexts.append(atyp_context)
        
        return {
            "typical": typical_contexts,
            "atypical": atypical_contexts
        }
    
    def generate_cross_domain_stability_test(self, concept: str, num_samples: int = 100) -> Dict[str, List[str]]:
        """Test concept stability across different domains/contexts"""
        
        definition = self.concept_definitions[concept]
        
        domains = {
            "scientific": [
                "In the laboratory, researchers study {concept} which is {property}.",
                "Scientific papers describe {concept} as having {property}.",
                "Academic research on {concept} focuses on {property}.",
                "Laboratory conditions reveal {concept} exhibits {property}."
            ],
            "everyday": [
                "In daily life, people encounter {concept} which is {property}.",
                "At home, you might find {concept} that is {property}.",
                "In ordinary situations, {concept} shows {property}.",
                "Regular experience with {concept} includes {property}."
            ],
            "technical": [
                "Engineers working with {concept} must consider {property}.",
                "Technical specifications for {concept} include {property}.",
                "Professional use of {concept} requires understanding {property}.",
                "Industrial applications of {concept} involve {property}."
            ],
            "cultural": [
                "In our culture, {concept} is understood as {property}.",
                "Social contexts present {concept} as {property}.",
                "Cultural traditions associate {concept} with {property}.",
                "Socially, {concept} is recognized by {property}."
            ]
        }
        
        stability_test = {}
        
        for domain, templates in domains.items():
            contexts = []
            for _ in range(num_samples // 4):
                template = random.choice(templates)
                
                # Use essential properties for natural kinds, functional for nominal
                if definition.kind_type == KindType.NATURAL:
                    property_pool = definition.essential_properties + definition.scientific_properties
                else:
                    property_pool = definition.essential_properties + definition.folk_properties
                
                prop = random.choice(property_pool)
                context = template.format(concept=concept, property=prop)
                contexts.append(context)
            
            stability_test[domain] = contexts
        
        return stability_test
    
    def create_natural_vs_nominal_dataset(self, natural_concepts: List[str] = None, 
                                        nominal_concepts: List[str] = None,
                                        samples_per_test: int = 200) -> Dict:
        """Create complete dataset for testing natural vs nominal kind distinctions"""
        
        if natural_concepts is None:
            natural_concepts = ["water", "gold", "tiger"]
        if nominal_concepts is None:
            nominal_concepts = ["chair", "game", "bachelor"]
        
        dataset = {
            "natural_kinds": {},
            "nominal_kinds": {},
            "artifact_kinds": {},
            "comparison_tests": {},
            "metadata": {
                "philosophical_framework": "Kripke-Putnam natural kinds theory",
                "test_predictions": {
                    "natural_kinds": "Should be more sensitive to essential than superficial properties",
                    "nominal_kinds": "Should be more sensitive to functional than accidental properties",
                    "stability": "Natural kinds should show more cross-domain stability",
                    "typicality": "All kinds should show prototype effects but natural kinds should also track essences"
                },
                "safety_implications": {
                    "if_natural_kind_tracking": "AI concepts align with scientific understanding - good for alignment",
                    "if_superficial_tracking": "AI concepts track appearances not reality - concerning for safety",
                    "if_nominal_appropriate": "AI tracks human purposes and functions appropriately"
                }
            }
        }
        
        # Generate data for natural kinds
        for concept in natural_concepts:
            if concept not in self.concept_definitions:
                continue
                
            print(f"Generating natural kind data for: {concept}")
            
            dataset["natural_kinds"][concept] = {
                "essential_contexts": self.generate_essential_property_contexts(concept, samples_per_test),
                "superficial_contexts": self.generate_superficial_property_contexts(concept, samples_per_test),
                "edge_cases": self.generate_edge_case_contexts(concept, samples_per_test // 2),
                "typicality_gradient": self.generate_typicality_gradient_contexts(concept, samples_per_test),
                "cross_domain_stability": self.generate_cross_domain_stability_test(concept, samples_per_test),
                "kind_type": "natural"
            }
            
            # Add scientific vs folk distinction
            sci_contexts, folk_contexts = self.generate_scientific_vs_folk_contexts(concept, samples_per_test // 2)
            dataset["natural_kinds"][concept]["scientific_contexts"] = sci_contexts
            dataset["natural_kinds"][concept]["folk_contexts"] = folk_contexts
        
        # Generate data for nominal kinds
        for concept in nominal_concepts:
            if concept not in self.concept_definitions:
                continue
                
            print(f"Generating nominal kind data for: {concept}")
            
            dataset["nominal_kinds"][concept] = {
                "essential_contexts": self.generate_essential_property_contexts(concept, samples_per_test),
                "superficial_contexts": self.generate_superficial_property_contexts(concept, samples_per_test),
                "edge_cases": self.generate_edge_case_contexts(concept, samples_per_test // 2),
                "typicality_gradient": self.generate_typicality_gradient_contexts(concept, samples_per_test),
                "cross_domain_stability": self.generate_cross_domain_stability_test(concept, samples_per_test),
                "kind_type": "nominal"
            }
            
            # Add scientific vs folk distinction
            sci_contexts, folk_contexts = self.generate_scientific_vs_folk_contexts(concept, samples_per_test // 2)
            dataset["nominal_kinds"][concept]["scientific_contexts"] = sci_contexts
            dataset["nominal_kinds"][concept]["folk_contexts"] = folk_contexts
        
        # Generate data for artifact kinds
        artifact_concepts = ["clock", "hammer"]
        for concept in artifact_concepts:
            if concept not in self.concept_definitions:
                continue
                
            print(f"Generating artifact kind data for: {concept}")
            
            dataset["artifact_kinds"][concept] = {
                "essential_contexts": self.generate_essential_property_contexts(concept, samples_per_test),
                "superficial_contexts": self.generate_superficial_property_contexts(concept, samples_per_test),
                "edge_cases": self.generate_edge_case_contexts(concept, samples_per_test // 2),
                "typicality_gradient": self.generate_typicality_gradient_contexts(concept, samples_per_test),
                "cross_domain_stability": self.generate_cross_domain_stability_test(concept, samples_per_test),
                "kind_type": "artifact"
            }
        
        # Generate direct comparison tests
        dataset["comparison_tests"] = self._generate_comparison_tests(
            natural_concepts, nominal_concepts, samples_per_test // 2
        )
        
        return dataset
    
    def _generate_comparison_tests(self, natural_concepts: List[str], 
                                 nominal_concepts: List[str], 
                                 num_samples: int) -> Dict:
        """Generate direct comparison tests between natural and nominal kinds"""
        
        comparison_templates = [
            "Unlike {nominal_concept}, {natural_concept} has an essential nature that is {property}.",
            "While {nominal_concept} is defined by human purposes, {natural_concept} is naturally {property}.",
            "The difference between {natural_concept} and {nominal_concept} is that the former has {property}.",
            "Scientific discovery can revise our understanding of {natural_concept} as {property}, but not {nominal_concept}.",
            "Essential properties like {property} determine {natural_concept} membership, unlike {nominal_concept}.",
            "Natural kinds like {natural_concept} have {property}, while conventional kinds like {nominal_concept} are human-defined.",
            "The essence of {natural_concept} involves {property}, whereas {nominal_concept} exists by social agreement.",
            "Science discovers that {natural_concept} is {property}, but {nominal_concept} is stipulated by humans."
        ]
        
        comparisons = []
        for _ in range(num_samples):
            template = random.choice(comparison_templates)
            natural_concept = random.choice(natural_concepts)
            nominal_concept = random.choice(nominal_concepts)
            
            # Use essential property from natural concept
            natural_def = self.concept_definitions[natural_concept]
            property_item = random.choice(natural_def.essential_properties)
            
            comparison = template.format(
                natural_concept=natural_concept,
                nominal_concept=nominal_concept,
                property=property_item
            )
            comparisons.append(comparison)
        
        return {
            "direct_comparisons": comparisons,
            "natural_vs_nominal_contrasts": self._generate_contrast_pairs(natural_concepts, nominal_concepts)
        }
    
    def _generate_contrast_pairs(self, natural_concepts: List[str], 
                               nominal_concepts: List[str]) -> List[Dict]:
        """Generate structured contrast pairs for analysis"""
        
        contrasts = []
        for natural in natural_concepts:
            for nominal in nominal_concepts:
                natural_def = self.concept_definitions[natural]
                nominal_def = self.concept_definitions[nominal]
                
                contrast = {
                    "natural_concept": natural,
                    "nominal_concept": nominal,
                    "natural_essential": random.choice(natural_def.essential_properties),
                    "nominal_essential": random.choice(nominal_def.essential_properties),
                    "natural_superficial": random.choice(natural_def.superficial_properties),
                    "nominal_superficial": random.choice(nominal_def.superficial_properties),
                    "prediction": f"{natural} should be more stable to superficial changes than {nominal}"
                }
                contrasts.append(contrast)
        
        return contrasts
    
    def generate_intervention_dataset(self, concept: str, num_samples: int = 100) -> Dict:
        """
        Generate dataset for causal intervention experiments.
        
        Tests what happens when essential vs superficial properties are modified.
        Philosophical prediction: Modifying essential properties should disrupt 
        concept more than modifying superficial properties.
        """
        
        definition = self.concept_definitions[concept]
        
        # Essential property modifications
        essential_interventions = []
        for essential_prop in definition.essential_properties:
            interventions = [
                f"Imagine {concept} without {essential_prop}. Is it still {concept}?",
                f"If we remove {essential_prop} from {concept}, what remains?",
                f"Consider {concept} that lacks {essential_prop}. Does this make sense?",
                f"What if {concept} had the opposite of {essential_prop}?",
                f"Scientists discover {concept} doesn't actually have {essential_prop}. What does this mean?"
            ]
            essential_interventions.extend(interventions)
        
        # Superficial property modifications  
        superficial_interventions = []
        for superficial_prop in definition.superficial_properties:
            interventions = [
                f"Imagine {concept} without {superficial_prop}. Is it still {concept}?",
                f"If we change {superficial_prop} about {concept}, does it remain the same kind?",
                f"Consider {concept} that has the opposite of {superficial_prop}.",
                f"What if this {concept} lacked {superficial_prop} entirely?",
                f"People discover this {concept} doesn't have {superficial_prop}. So what?"
            ]
            superficial_interventions.extend(interventions)
        
        return {
            "concept": concept,
            "essential_interventions": essential_interventions[:num_samples//2],
            "superficial_interventions": superficial_interventions[:num_samples//2],
            "philosophical_prediction": "Essential interventions should disrupt concept more than superficial ones"
        }
    
    def save_dataset(self, dataset: Dict, filename: str = "natural_nominal_kinds_dataset.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Natural vs Nominal Kinds dataset saved to {filename}")
    
    def load_dataset(self, filename: str = "natural_nominal_kinds_dataset.json") -> Dict:
        """Load dataset from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)

# Analysis framework for natural vs nominal kinds
class NaturalNominalAnalyzer:
    """
    Framework for analyzing whether AI systems track essential vs superficial properties
    in natural vs nominal kinds.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def extract_concept_vector(self, concept: str, contexts: List[str], layer: int = -6):
        """Extract concept representation from contexts"""
        concept_vectors = []
        
        for context in contexts:
            try:
                # Tokenize and find concept position
                tokens = self.tokenizer(context, return_tensors="pt", truncation=True)
                concept_token_id = self.tokenizer.encode(concept, add_special_tokens=False)[0]
                input_ids = tokens["input_ids"][0]
                
                # Find concept position
                concept_positions = (input_ids == concept_token_id).nonzero(as_tuple=True)[0]
                if len(concept_positions) == 0:
                    continue
                
                # Get activations
                with torch.no_grad():
                    outputs = self.model(**tokens, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[layer]
                    concept_pos = concept_positions[0]
                    concept_vector = hidden_states[0, concept_pos, :].cpu().numpy()
                    concept_vectors.append(concept_vector)
                    
            except Exception as e:
                continue
        
        if not concept_vectors:
            raise ValueError(f"No valid vectors extracted for {concept}")
        
        return np.mean(concept_vectors, axis=0)
    
    def test_essential_vs_superficial_sensitivity(self, concept: str, dataset: Dict) -> Dict:
        """
        Test whether concept is more sensitive to essential than superficial property changes.
        
        Philosophical prediction:
        - Natural kinds: High essential sensitivity, low superficial sensitivity
        - Nominal kinds: Moderate essential sensitivity (functional properties)
        """
        
        kind_type = None
        concept_data = None
        
        # Find concept in dataset
        for kind_category in ["natural_kinds", "nominal_kinds", "artifact_kinds"]:
            if concept in dataset.get(kind_category, {}):
                kind_type = kind_category
                concept_data = dataset[kind_category][concept]
                break
        
        if concept_data is None:
            raise ValueError(f"Concept {concept} not found in dataset")
        
        # Extract vectors for essential vs superficial contexts
        essential_vector = self.extract_concept_vector(
            concept, concept_data["essential_contexts"][:50]
        )
        superficial_vector = self.extract_concept_vector(
            concept, concept_data["superficial_contexts"][:50]
        )
        
        # Compare with baseline concept vector
        all_contexts = (concept_data["essential_contexts"][:25] + 
                       concept_data["superficial_contexts"][:25])
        baseline_vector = self.extract_concept_vector(concept, all_contexts)
        
        # Calculate similarities
        essential_similarity = cosine_similarity(
            baseline_vector.reshape(1, -1), essential_vector.reshape(1, -1)
        )[0, 0]
        
        superficial_similarity = cosine_similarity(
            baseline_vector.reshape(1, -1), superficial_vector.reshape(1, -1)
        )[0, 0]
        
        # Calculate sensitivity (1 - similarity = how much the vector changed)
        essential_sensitivity = 1 - essential_similarity
        superficial_sensitivity = 1 - superficial_similarity
        
        # Philosophical interpretation
        if kind_type == "natural_kinds":
            expected_pattern = "essential_sensitivity < superficial_sensitivity"
            tracks_essences = essential_sensitivity < superficial_sensitivity
        else:  # nominal or artifact kinds
            expected_pattern = "functional_sensitivity (essential) >= superficial_sensitivity"
            tracks_essences = essential_sensitivity >= superficial_sensitivity
        
        return {
            "concept": concept,
            "kind_type": kind_type,
            "essential_sensitivity": float(essential_sensitivity),
            "superficial_sensitivity": float(superficial_sensitivity),
            "sensitivity_difference": float(superficial_sensitivity - essential_sensitivity),
            "tracks_essences_appropriately": tracks_essences,
            "expected_pattern": expected_pattern,
            "philosophical_interpretation": self._interpret_sensitivity_pattern(
                essential_sensitivity, superficial_sensitivity, kind_type
            )
        }
    
    def _interpret_sensitivity_pattern(self, essential_sens: float, superficial_sens: float, kind_type: str) -> str:
        """Interpret sensitivity pattern philosophically"""
        
        diff = superficial_sens - essential_sens
        
        if kind_type == "natural_kinds":
            if diff > 0.1:
                return "Strong evidence for essential property tracking (good for natural kinds)"
            elif diff > 0.05:
                return "Moderate evidence for essential property tracking"
            elif diff < -0.1:
                return "Concerning: More sensitive to superficial than essential properties"
            else:
                return "Unclear pattern: Similar sensitivity to essential and superficial properties"
        else:  # nominal/artifact kinds
            if essential_sens > superficial_sens + 0.05:
                return "Appropriately tracks functional/definitional properties"
            elif superficial_sens > essential_sens + 0.1:
                return "Concerning: Overemphasizes superficial properties for functional kind"
            else:
                return "Mixed pattern: Moderate sensitivity to both functional and superficial properties"
    
    def test_cross_domain_stability(self, concept: str, dataset: Dict) -> Dict:
        """Test whether concept remains stable across different domains"""
        
        # Find concept data
        concept_data = None
        for kind_category in ["natural_kinds", "nominal_kinds", "artifact_kinds"]:
            if concept in dataset.get(kind_category, {}):
                concept_data = dataset[kind_category][concept]
                break
        
        if concept_data is None or "cross_domain_stability" not in concept_data:
            raise ValueError(f"Cross-domain data not found for {concept}")
        
        stability_data = concept_data["cross_domain_stability"]
        domain_vectors = {}
        
        # Extract vectors for each domain
        for domain, contexts in stability_data.items():
            domain_vectors[domain] = self.extract_concept_vector(concept, contexts[:30])
        
        # Calculate pairwise similarities between domains
        domain_names = list(domain_vectors.keys())
        similarities = {}
        
        for i, domain1 in enumerate(domain_names):
            for j, domain2 in enumerate(domain_names[i+1:], i+1):
                sim = cosine_similarity(
                    domain_vectors[domain1].reshape(1, -1),
                    domain_vectors[domain2].reshape(1, -1)
                )[0, 0]
                similarities[f"{domain1}_vs_{domain2}"] = float(sim)
        
        mean_stability = np.mean(list(similarities.values()))
        
        return {
            "concept": concept,
            "domain_similarities": similarities,
            "mean_cross_domain_stability": float(mean_stability),
            "high_stability": mean_stability > 0.8,
            "philosophical_interpretation": (
                "High cross-domain stability (concept tracks stable properties)" if mean_stability > 0.8
                else "Low stability (concept varies by context)" if mean_stability < 0.6
                else "Moderate stability"
            )
        }
    
    def test_typicality_effects(self, concept: str, dataset: Dict) -> Dict:
        """Test whether concept shows prototype structure"""
        
        # Find concept data
        concept_data = None
        for kind_category in ["natural_kinds", "nominal_kinds", "artifact_kinds"]:
            if concept in dataset.get(kind_category, {}):
                concept_data = dataset[kind_category][concept]
                break
        
        if concept_data is None or "typicality_gradient" not in concept_data:
            raise ValueError(f"Typicality data not found for {concept}")
        
        typicality_data = concept_data["typicality_gradient"]
        
        # Extract vectors for typical vs atypical instances
        typical_vector = self.extract_concept_vector(
            concept, typicality_data["typical"][:30]
        )
        atypical_vector = self.extract_concept_vector(
            concept, typicality_data["atypical"][:30]
        )
        
        # Calculate baseline concept vector
        all_contexts = (concept_data["essential_contexts"][:20] +
                       concept_data["superficial_contexts"][:20])
        baseline_vector = self.extract_concept_vector(concept, all_contexts)
        
        # Calculate similarities to baseline
        typical_similarity = cosine_similarity(
            baseline_vector.reshape(1, -1), typical_vector.reshape(1, -1)
        )[0, 0]
        
        atypical_similarity = cosine_similarity(
            baseline_vector.reshape(1, -1), atypical_vector.reshape(1, -1)
        )[0, 0]
        
        typicality_effect = typical_similarity - atypical_similarity
        
        return {
            "concept": concept,
            "typical_similarity_to_baseline": float(typical_similarity),
            "atypical_similarity_to_baseline": float(atypical_similarity),
            "typicality_effect": float(typicality_effect),
            "shows_prototype_structure": typicality_effect > 0.05,
            "philosophical_interpretation": (
                "Strong prototype structure (typical instances closer to concept center)" if typicality_effect > 0.1
                else "Moderate prototype effects" if typicality_effect > 0.05
                else "Weak prototype structure" if typicality_effect > 0
                else "Reverse typicality effect (atypical instances more central)"
            )
        }
    
    def comprehensive_natural_nominal_analysis(self, dataset: Dict, concepts: List[str] = None) -> Dict:
        """Run complete analysis on natural vs nominal kinds dataset"""
        
        if concepts is None:
            # Get all concepts from dataset
            concepts = []
            for kind_category in ["natural_kinds", "nominal_kinds", "artifact_kinds"]:
                if kind_category in dataset:
                    concepts.extend(dataset[kind_category].keys())
        
        results = {
            "essential_vs_superficial_tests": {},
            "cross_domain_stability_tests": {},
            "typicality_effect_tests": {},
            "natural_vs_nominal_comparison": {},
            "philosophical_conclusions": {}
        }
        
        for concept in concepts:
            print(f"Analyzing {concept}...")
            
            try:
                # Essential vs superficial sensitivity
                results["essential_vs_superficial_tests"][concept] = (
                    self.test_essential_vs_superficial_sensitivity(concept, dataset)
                )
                
                # Cross-domain stability
                results["cross_domain_stability_tests"][concept] = (
                    self.test_cross_domain_stability(concept, dataset)
                )
                
                # Typicality effects
                results["typicality_effect_tests"][concept] = (
                    self.test_typicality_effects(concept, dataset)
                )
                
            except Exception as e:
                print(f"Error analyzing {concept}: {e}")
                continue
        
        # Comparative analysis
        results["natural_vs_nominal_comparison"] = self._compare_natural_vs_nominal(results, dataset)
        
        # Overall philosophical conclusions
        results["philosophical_conclusions"] = self._generate_philosophical_conclusions(results)
        
        return results
    
    def _compare_natural_vs_nominal(self, results: Dict, dataset: Dict) -> Dict:
        """Compare patterns between natural and nominal kinds"""
        
        natural_concepts = list(dataset.get("natural_kinds", {}).keys())
        nominal_concepts = list(dataset.get("nominal_kinds", {}).keys())
        
        # Calculate averages for each kind type
        natural_stats = self._calculate_kind_statistics(results, natural_concepts)
        nominal_stats = self._calculate_kind_statistics(results, nominal_concepts)
        
        return {
            "natural_kind_patterns": natural_stats,
            "nominal_kind_patterns": nominal_stats,
            "key_differences": {
                "essential_tracking": natural_stats["mean_essential_tracking"] - nominal_stats["mean_essential_tracking"],
                "stability": natural_stats["mean_stability"] - nominal_stats["mean_stability"],
                "typicality": natural_stats["mean_typicality"] - nominal_stats["mean_typicality"]
            },
            "philosophical_assessment": self._assess_kind_differences(natural_stats, nominal_stats)
        }
    
    def _calculate_kind_statistics(self, results: Dict, concepts: List[str]) -> Dict:
        """Calculate summary statistics for a kind type"""
        
        essential_tracking = []
        stability_scores = []
        typicality_scores = []
        
        for concept in concepts:
            if concept in results["essential_vs_superficial_tests"]:
                test_result = results["essential_vs_superficial_tests"][concept]
                tracks_appropriately = test_result["tracks_essences_appropriately"]
                essential_tracking.append(1.0 if tracks_appropriately else 0.0)
            
            if concept in results["cross_domain_stability_tests"]:
                stability = results["cross_domain_stability_tests"][concept]["mean_cross_domain_stability"]
                stability_scores.append(stability)
            
            if concept in results["typicality_effect_tests"]:
                typicality = results["typicality_effect_tests"][concept]["typicality_effect"]
                typicality_scores.append(typicality)
        
        return {
            "concepts": concepts,
            "mean_essential_tracking": np.mean(essential_tracking) if essential_tracking else 0,
            "mean_stability": np.mean(stability_scores) if stability_scores else 0,
            "mean_typicality": np.mean(typicality_scores) if typicality_scores else 0,
            "sample_size": len(concepts)
        }
    
    def _assess_kind_differences(self, natural_stats: Dict, nominal_stats: Dict) -> str:
        """Assess whether AI shows expected differences between natural and nominal kinds"""
        
        essential_diff = natural_stats["mean_essential_tracking"] - nominal_stats["mean_essential_tracking"]
        stability_diff = natural_stats["mean_stability"] - nominal_stats["mean_stability"]
        
        if essential_diff > 0.2 and stability_diff > 0.1:
            return "Strong evidence AI distinguishes natural from nominal kinds appropriately"
        elif essential_diff > 0.1 or stability_diff > 0.05:
            return "Moderate evidence for natural/nominal kind distinction"
        elif essential_diff < -0.1:
            return "Concerning: AI treats natural kinds like nominal kinds"
        else:
            return "Unclear: No strong differentiation between natural and nominal kinds"
    
    def _generate_philosophical_conclusions(self, results: Dict) -> Dict:
        """Generate overall philosophical conclusions from analysis"""
        
        # Count concepts that track essences appropriately
        appropriate_tracking = sum(
            1 for test in results["essential_vs_superficial_tests"].values()
            if test["tracks_essences_appropriately"]
        )
        total_concepts = len(results["essential_vs_superficial_tests"])
        
        # Calculate average stability
        stability_scores = [
            test["mean_cross_domain_stability"] 
            for test in results["cross_domain_stability_tests"].values()
        ]
        mean_stability = np.mean(stability_scores) if stability_scores else 0
        
        # Generate conclusion
        tracking_ratio = appropriate_tracking / total_concepts if total_concepts > 0 else 0
        
        if tracking_ratio > 0.75 and mean_stability > 0.8:
            overall_assessment = "Strong evidence AI tracks essential properties and natural kinds appropriately"
            safety_implication = "POSITIVE: AI concepts align with scientific understanding of natural categories"
        elif tracking_ratio > 0.5 and mean_stability > 0.6:
            overall_assessment = "Moderate evidence for appropriate essential property tracking"
            safety_implication = "MIXED: Some alignment with scientific categories, but inconsistent"
        else:
            overall_assessment = "Weak evidence for essential property tracking"
            safety_implication = "CONCERNING: AI may not distinguish essential from superficial properties"
        
        return {
            "overall_assessment": overall_assessment,
            "safety_implication": safety_implication,
            "essential_tracking_ratio": tracking_ratio,
            "mean_concept_stability": mean_stability,
            "total_concepts_analyzed": total_concepts
        }

# Example usage and testing
if __name__ == "__main__":
    # Create dataset generator
    generator = NaturalNominalDatasetGenerator()
    
    # Generate sample data
    print("Generating Natural vs Nominal Kinds dataset...")
    dataset = generator.create_natural_vs_nominal_dataset(
        natural_concepts=["water", "gold", "tiger"],
        nominal_concepts=["chair", "game", "bachelor"],
        samples_per_test=100
    )
    
    # Show sample data
    print("\nSample Natural Kind (water) essential contexts:")
    for i, context in enumerate(dataset["natural_kinds"]["water"]["essential_contexts"][:3]):
        print(f"{i+1}. {context}")
    
    print("\nSample Natural Kind (water) superficial contexts:")
    for i, context in enumerate(dataset["natural_kinds"]["water"]["superficial_contexts"][:3]):
        print(f"{i+1}. {context}")
    
    print("\nSample Nominal Kind (chair) essential contexts:")
    for i, context in enumerate(dataset["nominal_kinds"]["chair"]["essential_contexts"][:3]):
        print(f"{i+1}. {context}")
    
    print("\nSample Nominal Kind (chair) superficial contexts:")
    for i, context in enumerate(dataset["nominal_kinds"]["chair"]["superficial_contexts"][:3]):
        print(f"{i+1}. {context}")
    
    # Save dataset
    generator.save_dataset(dataset, "natural_nominal_dataset.json")
    
    print("\n✓ Dataset generation complete!")
    print(f"Natural kinds: {list(dataset['natural_kinds'].keys())}")
    print(f"Nominal kinds: {list(dataset['nominal_kinds'].keys())}")
    print(f"Artifact kinds: {list(dataset['artifact_kinds'].keys())}")
    
    # Show intervention test example
    print("\nSample intervention test for 'water':")
    water_interventions = generator.generate_intervention_dataset("water", 6)
    print("Essential property interventions:")
    for i, intervention in enumerate(water_interventions["essential_interventions"][:2]):
        print(f"{i+1}. {intervention}")
    print("Superficial property interventions:")
    for i, intervention in enumerate(water_interventions["superficial_interventions"][:2]):
        print(f"{i+1}. {intervention}")
    
    print("\nPhilosophical prediction:")
    print("- Natural kinds should be more disrupted by essential than superficial interventions")
    print("- Nominal kinds should be more disrupted by functional than superficial interventions")
    print("- This tests whether AI tracks genuine vs conventional categories")

            