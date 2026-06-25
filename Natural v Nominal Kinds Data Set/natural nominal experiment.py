#!/usr/bin/env python3
"""
Natural vs Nominal Kinds Experiment Runner
==========================================

Tests whether AI systems track essential properties as predicted for genuine
concept understanding (Kripke-Putnam natural kinds theory).

Key Philosophical Questions:
1. Do AI systems distinguish essential from superficial properties?
2. Are natural kinds (water, gold) more stable than nominal kinds (chair, game)?
3. Do AI concepts align with scientific understanding of natural categories?
4. What are the implications for AI safety and interpretability?
"""

import os
import sys
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Kind taxonomy
# ---------------------------------------------------------------------------

class KindType(Enum):
    NATURAL = "natural"
    NOMINAL = "nominal"
    ARTIFACT = "artifact"


@dataclass
class ConceptDefinition:
    """Defines a concept with its essential vs superficial properties."""
    name: str
    kind_type: KindType
    essential_properties: List[str]
    superficial_properties: List[str]
    scientific_properties: List[str]
    folk_properties: List[str]
    edge_cases: List[str]
    typical_instances: List[str]
    atypical_instances: List[str]


# ---------------------------------------------------------------------------
# Dataset generator (full implementation)
# ---------------------------------------------------------------------------

class NaturalNominalDatasetGenerator:
    """
    Generates datasets for testing whether AI tracks essential vs superficial
    properties in natural kinds (water, gold, tiger) vs nominal kinds
    (chair, game, bachelor).

    Philosophical Predictions:
    - Natural kinds: AI should be more sensitive to essential than superficial properties.
    - Nominal kinds: AI should be more sensitive to functional/definitional properties.
    - Essential properties should remain stable across contexts.
    - Superficial properties should vary without affecting core concept.
    """

    def __init__(self):
        self.concept_definitions = self._define_concepts()

    # ------------------------------------------------------------------
    # Concept library
    # ------------------------------------------------------------------

    def _define_concepts(self) -> Dict[str, ConceptDefinition]:
        return {
            # ── NATURAL KINDS ────────────────────────────────────────────
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
                    "chemical identity determined by atomic composition",
                ],
                superficial_properties=[
                    "clear and colorless liquid", "tasteless and odorless",
                    "flows and takes container shape", "feels wet to touch",
                    "makes splashing sounds", "reflects light when still",
                    "commonly found in bottles",
                ],
                scientific_properties=[
                    "boiling point 100°C at standard pressure",
                    "freezing point 0°C at standard pressure",
                    "density 1.0 g/cm³ at standard conditions",
                    "specific heat capacity 4.18 J/g°C",
                    "dielectric constant of 81",
                    "self-ionizes to form H+ and OH- ions",
                ],
                folk_properties=[
                    "essential for life", "used for drinking and cooking",
                    "falls as rain from clouds", "found in rivers and oceans",
                    "can be hot or cold", "extinguishes fires",
                ],
                edge_cases=["heavy water (D2O)", "water vapor", "ice crystals",
                            "water mixed with impurities", "water at extreme temperatures"],
                typical_instances=["tap water", "bottled water", "rainwater", "ocean water"],
                atypical_instances=["steam", "ice", "mineral water", "distilled water"],
            ),

            "gold": ConceptDefinition(
                name="gold",
                kind_type=KindType.NATURAL,
                essential_properties=[
                    "atomic number 79", "chemical symbol Au",
                    "79 protons in atomic nucleus",
                    "electron configuration [Xe] 4f¹⁴ 5d¹⁰ 6s¹",
                    "metallic bonding structure",
                    "face-centered cubic crystal lattice",
                    "nuclear charge determines all other properties",
                ],
                superficial_properties=[
                    "yellow metallic color", "shiny and lustrous appearance",
                    "feels heavy and dense", "soft enough to scratch with fingernail",
                    "valuable and expensive", "used in jewelry and decoration",
                    "associated with wealth and status",
                ],
                scientific_properties=[
                    "density 19.3 g/cm³", "melting point 1064°C",
                    "excellent electrical conductor", "chemically inert and non-reactive",
                    "malleable and ductile", "atomic mass 196.97 amu",
                ],
                folk_properties=[
                    "precious metal", "doesn't rust or tarnish", "found by mining",
                    "used in coins and jewelry", "symbol of value",
                    "desired throughout history",
                ],
                edge_cases=["gold alloys", "gold nanoparticles", "ionic gold compounds",
                            "gold leaf", "white gold", "rose gold"],
                typical_instances=["gold bars", "gold coins", "gold jewelry", "gold nuggets"],
                atypical_instances=["gold paint", "gold thread", "gold dental fillings",
                                    "gold electronics"],
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
                    "obligate carnivore metabolism",
                ],
                superficial_properties=[
                    "orange fur with black stripes", "large size and muscular build",
                    "distinctive facial markings", "long tail with black rings",
                    "fierce and intimidating appearance",
                    "solitary and territorial behavior", "found in Asian forests",
                ],
                scientific_properties=[
                    "gestation period approximately 103 days",
                    "average lifespan 10-15 years in wild",
                    "body length 1.4-2.8 meters",
                    "weight 90-300 kg depending on subspecies",
                    "night vision adaptations", "specialized hunting dentition",
                ],
                folk_properties=[
                    "dangerous predator", "king of the jungle", "excellent hunter",
                    "endangered species", "lives in Asia", "featured in stories and myths",
                ],
                edge_cases=["white tigers", "tiger cubs", "tiger-lion hybrids",
                            "extinct tiger subspecies", "tigers in captivity"],
                typical_instances=["Bengal tiger", "Siberian tiger", "wild adult tiger",
                                   "hunting tiger"],
                atypical_instances=["tiger cub", "white tiger", "paper tiger", "tiger in zoo"],
            ),

            # ── NOMINAL KINDS ────────────────────────────────────────────
            "chair": ConceptDefinition(
                name="chair",
                kind_type=KindType.NOMINAL,
                essential_properties=[
                    "designed for sitting", "supports human body weight",
                    "elevated seating surface", "provides back support",
                    "intended for single person use", "stable base structure",
                    "functional purpose of seating",
                ],
                superficial_properties=[
                    "made of wood or metal", "has four legs", "brown or black color",
                    "specific style or design", "particular size or height",
                    "found in dining rooms", "matches other furniture",
                ],
                scientific_properties=[
                    "material strength and durability", "ergonomic design principles",
                    "load-bearing capacity", "center of gravity calculations",
                    "stress distribution patterns", "material fatigue characteristics",
                ],
                folk_properties=[
                    "furniture for sitting", "found in homes and offices",
                    "comes in many styles", "can be moved around",
                    "part of table and chair sets", "requires assembly or is pre-made",
                ],
                edge_cases=["bean bag chairs", "rocking chairs", "folding chairs",
                            "electric chairs", "wheelchair", "high chairs"],
                typical_instances=["dining chair", "office chair", "kitchen chair",
                                   "wooden chair"],
                atypical_instances=["beanbag", "stool", "bench", "throne"],
            ),

            "game": ConceptDefinition(
                name="game",
                kind_type=KindType.NOMINAL,
                essential_properties=[
                    "set of rules governing play", "voluntary participation",
                    "defined winning conditions or objectives",
                    "structured activity with constraints",
                    "competitive or skill-based elements",
                    "distinct beginning and end",
                    "artificial rather than natural activity",
                ],
                superficial_properties=[
                    "played with specific equipment", "involves multiple players",
                    "takes certain amount of time", "played in particular locations",
                    "associated with fun and entertainment",
                    "has particular cultural associations",
                    "involves physical or mental activity",
                ],
                scientific_properties=[
                    "game theory mathematical principles",
                    "strategic decision-making patterns",
                    "probability and chance mechanisms",
                    "psychological engagement factors",
                    "learning and skill development",
                    "social interaction dynamics",
                ],
                folk_properties=[
                    "played for fun and entertainment",
                    "can be competitive or cooperative",
                    "learned through practice", "brings people together",
                    "can be won or lost", "helps pass time",
                ],
                edge_cases=["video games", "mind games", "war games",
                            "children's games", "gambling games", "party games"],
                typical_instances=["chess", "soccer", "poker", "Monopoly"],
                atypical_instances=["solitaire", "video game", "drinking game", "word game"],
            ),

            "bachelor": ConceptDefinition(
                name="bachelor",
                kind_type=KindType.NOMINAL,
                essential_properties=[
                    "adult human male", "unmarried status",
                    "eligible for marriage",
                    "not currently in marriage contract",
                    "legal and social status definition",
                    "socially recognized category",
                    "definitional rather than empirical kind",
                ],
                superficial_properties=[
                    "lives alone", "particular age range",
                    "specific lifestyle choices", "dating behaviour patterns",
                    "social activities and preferences",
                    "economic status or career",
                    "personal appearance or habits",
                ],
                scientific_properties=[
                    "demographic classification", "legal status implications",
                    "sociological category", "statistical population grouping",
                    "cultural variation in definition",
                    "historical changes in concept",
                ],
                folk_properties=[
                    "single man", "available for dating", "not tied down",
                    "independent lifestyle", "potential husband", "unattached male",
                ],
                edge_cases=["divorced men", "widowers", "committed but unmarried men",
                            "gay bachelors", "bachelor priests", "elderly bachelors"],
                typical_instances=["young unmarried man", "dating bachelor",
                                   "eligible bachelor"],
                atypical_instances=["confirmed bachelor", "elderly bachelor",
                                    "bachelor by choice"],
            ),

            # ── ARTIFACT KINDS ───────────────────────────────────────────
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
                    "functional rather than natural kind",
                ],
                superficial_properties=[
                    "round face with numbers", "has hour and minute hands",
                    "makes ticking sounds", "hangs on wall or sits on surface",
                    "particular size or colour", "specific style or appearance",
                    "made of certain materials",
                ],
                scientific_properties=[
                    "mechanical or electronic timing mechanism",
                    "oscillation frequency for time keeping",
                    "gear ratios for hand movement", "power source requirements",
                    "accuracy and precision specifications",
                    "temperature and environmental stability",
                ],
                folk_properties=[
                    "tells time", "helps people be on schedule",
                    "found in homes and public places",
                    "can be analog or digital",
                    "requires winding or batteries",
                    "important for daily life",
                ],
                edge_cases=["digital clocks", "atomic clocks", "sundials",
                            "broken clocks", "decorative clocks", "clock towers"],
                typical_instances=["wall clock", "alarm clock", "grandfather clock",
                                   "wristwatch"],
                atypical_instances=["sundial", "atomic clock", "cuckoo clock",
                                    "digital display"],
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
                    "purposeful artifact with clear function",
                ],
                superficial_properties=[
                    "made of metal and wood", "particular size and weight",
                    "specific handle length", "certain colour or finish",
                    "found in toolboxes", "associated with construction",
                    "has brand markings",
                ],
                scientific_properties=[
                    "mechanical advantage through leverage",
                    "kinetic energy transfer mechanisms",
                    "material strength and durability",
                    "ergonomic design principles",
                    "impact force calculations",
                    "vibration dampening properties",
                ],
                folk_properties=[
                    "tool for hitting nails", "used in construction",
                    "every household should have one",
                    "symbol of building and making",
                    "requires skill to use effectively",
                    "can be dangerous if misused",
                ],
                edge_cases=["sledgehammer", "ball-peen hammer", "rubber hammer",
                            "decorative hammer", "broken hammer", "toy hammer"],
                typical_instances=["claw hammer", "carpenter's hammer",
                                   "construction hammer"],
                atypical_instances=["sledgehammer", "tack hammer", "rubber mallet",
                                    "war hammer"],
            ),
        }

    # ------------------------------------------------------------------
    # Context generators
    # ------------------------------------------------------------------

    def generate_essential_property_contexts(self, concept: str,
                                             num_samples: int = 200) -> List[str]:
        if concept not in self.concept_definitions:
            raise ValueError(f"Concept '{concept}' not defined")
        definition = self.concept_definitions[concept]
        templates = [
            "Scientific analysis reveals that {concept} is fundamentally characterized by {prop}.",
            "The defining feature of {concept} is {prop}.",
            "What makes {concept} what it is: {prop}.",
            "The essential nature of {concept} involves {prop}.",
            "Researchers have determined that {concept} necessarily exhibits {prop}.",
            "The core property that defines {concept} is {prop}.",
            "Without {prop}, something cannot be considered {concept}.",
            "The invariant characteristic of {concept} is {prop}.",
            "Deep investigation shows {concept} is essentially {prop}.",
            "The fundamental property underlying {concept} is {prop}.",
        ]
        return [
            random.choice(templates).format(
                concept=concept, prop=random.choice(definition.essential_properties)
            )
            for _ in range(num_samples)
        ]

    def generate_superficial_property_contexts(self, concept: str,
                                               num_samples: int = 200) -> List[str]:
        if concept not in self.concept_definitions:
            raise ValueError(f"Concept '{concept}' not defined")
        definition = self.concept_definitions[concept]
        templates = [
            "This particular {concept} happens to be {prop}.",
            "The {concept} appears {prop} in this instance.",
            "One can observe that this {concept} is {prop}.",
            "The {concept} looks {prop} from this angle.",
            "This example of {concept} shows {prop}.",
            "The surface characteristics include {concept} being {prop}.",
            "Visually, the {concept} presents as {prop}.",
            "This {concept} exhibits the superficial trait of being {prop}.",
            "The apparent quality of this {concept} is {prop}.",
            "One notices the {concept} has the accidental property of being {prop}.",
        ]
        return [
            random.choice(templates).format(
                concept=concept, prop=random.choice(definition.superficial_properties)
            )
            for _ in range(num_samples)
        ]

    def generate_scientific_vs_folk_contexts(self, concept: str,
                                             num_samples: int = 100
                                             ) -> Tuple[List[str], List[str]]:
        definition = self.concept_definitions[concept]
        sci_templates = [
            "Scientific research shows that {concept} is characterized by {prop}.",
            "Laboratory analysis reveals {concept} exhibits {prop}.",
            "Empirical studies demonstrate that {concept} involves {prop}.",
            "Advanced measurement techniques show {concept} has {prop}.",
            "The scientific understanding of {concept} includes {prop}.",
            "Rigorous investigation confirms {concept} displays {prop}.",
        ]
        folk_templates = [
            "People generally know that {concept} is {prop}.",
            "Everyone understands {concept} as {prop}.",
            "Common knowledge suggests {concept} is {prop}.",
            "Folk wisdom tells us {concept} is {prop}.",
            "The everyday understanding of {concept} includes being {prop}.",
            "Regular people recognise {concept} as {prop}.",
        ]
        scientific_contexts = [
            random.choice(sci_templates).format(
                concept=concept, prop=random.choice(definition.scientific_properties)
            )
            for _ in range(num_samples)
        ]
        folk_contexts = [
            random.choice(folk_templates).format(
                concept=concept, prop=random.choice(definition.folk_properties)
            )
            for _ in range(num_samples)
        ]
        return scientific_contexts, folk_contexts

    def generate_edge_case_contexts(self, concept: str,
                                    num_samples: int = 100) -> List[str]:
        definition = self.concept_definitions[concept]
        templates = [
            "Consider this borderline case: {edge_case} as an example of {concept}.",
            "The edge case of {edge_case} challenges our understanding of {concept}.",
            "Is {edge_case} really a type of {concept}? This tests the boundaries.",
            "The marginal instance {edge_case} raises questions about {concept} membership.",
            "Philosophers debate whether {edge_case} counts as {concept}.",
            "The boundary case of {edge_case} illuminates the nature of {concept}.",
            "Consider the atypical example: {edge_case} classified as {concept}.",
            "The problematic case of {edge_case} tests our concept of {concept}.",
        ]
        return [
            random.choice(templates).format(
                concept=concept, edge_case=random.choice(definition.edge_cases)
            )
            for _ in range(num_samples)
        ]

    def generate_typicality_gradient_contexts(self, concept: str,
                                              num_samples: int = 150
                                              ) -> Dict[str, List[str]]:
        definition = self.concept_definitions[concept]
        typical_templates = [
            "A perfect example of {concept} is {instance}.",
            "When people think of {concept}, they usually imagine {instance}.",
            "The prototypical {concept} would be {instance}.",
            "A clear case of {concept} is {instance}.",
            "The best example of {concept} is {instance}.",
            "A central instance of {concept} is {instance}.",
        ]
        atypical_templates = [
            "An unusual example of {concept} is {instance}.",
            "A less typical {concept} would be {instance}.",
            "An atypical instance of {concept} is {instance}.",
            "A marginal example of {concept} is {instance}.",
            "A peripheral case of {concept} is {instance}.",
            "An uncommon type of {concept} is {instance}.",
        ]
        half = num_samples // 2
        typical = [
            random.choice(typical_templates).format(
                concept=concept, instance=random.choice(definition.typical_instances)
            )
            for _ in range(half)
        ]
        atypical = [
            random.choice(atypical_templates).format(
                concept=concept, instance=random.choice(definition.atypical_instances)
            )
            for _ in range(half)
        ]
        return {"typical": typical, "atypical": atypical}

    def generate_cross_domain_stability_test(self, concept: str,
                                             num_samples: int = 100
                                             ) -> Dict[str, List[str]]:
        definition = self.concept_definitions[concept]
        domain_templates = {
            "scientific": [
                "In the laboratory, researchers study {concept} which has {prop}.",
                "Scientific papers describe {concept} as having {prop}.",
                "Academic research on {concept} focuses on {prop}.",
                "Laboratory conditions reveal {concept} exhibits {prop}.",
            ],
            "everyday": [
                "In daily life, people encounter {concept} which is {prop}.",
                "At home, you might find {concept} that is {prop}.",
                "In ordinary situations, {concept} shows {prop}.",
                "Regular experience with {concept} includes {prop}.",
            ],
            "technical": [
                "Engineers working with {concept} must consider {prop}.",
                "Technical specifications for {concept} include {prop}.",
                "Professional use of {concept} requires understanding {prop}.",
                "Industrial applications of {concept} involve {prop}.",
            ],
            "cultural": [
                "In our culture, {concept} is understood as {prop}.",
                "Social contexts present {concept} as {prop}.",
                "Cultural traditions associate {concept} with {prop}.",
                "Socially, {concept} is recognised by {prop}.",
            ],
        }
        if definition.kind_type == KindType.NATURAL:
            property_pool = definition.essential_properties + definition.scientific_properties
        else:
            property_pool = definition.essential_properties + definition.folk_properties

        stability_test: Dict[str, List[str]] = {}
        per_domain = max(1, num_samples // 4)
        for domain, templates in domain_templates.items():
            stability_test[domain] = [
                random.choice(templates).format(
                    concept=concept, prop=random.choice(property_pool)
                )
                for _ in range(per_domain)
            ]
        return stability_test

    def generate_intervention_dataset(self, concept: str,
                                      num_samples: int = 100) -> Dict:
        """
        Generate causal intervention scenarios.
        Philosophical prediction: modifying essential properties should disrupt
        the concept more than modifying superficial properties.
        """
        definition = self.concept_definitions[concept]

        essential_interventions: List[str] = []
        for prop in definition.essential_properties:
            essential_interventions.extend([
                f"Imagine {concept} without {prop}. Is it still {concept}?",
                f"If we remove {prop} from {concept}, what remains?",
                f"Consider {concept} that lacks {prop}. Does this make sense?",
                f"What if {concept} had the opposite of {prop}?",
                f"Scientists discover {concept} doesn't actually have {prop}. "
                f"What does this mean?",
            ])

        superficial_interventions: List[str] = []
        for prop in definition.superficial_properties:
            superficial_interventions.extend([
                f"Imagine {concept} without {prop}. Is it still {concept}?",
                f"If we change {prop} about {concept}, does it remain the same kind?",
                f"Consider {concept} that has the opposite of {prop}.",
                f"What if this {concept} lacked {prop} entirely?",
                f"People discover this {concept} doesn't have {prop}. So what?",
            ])

        half = num_samples // 2
        return {
            "concept": concept,
            "essential_interventions": essential_interventions[:half],
            "superficial_interventions": superficial_interventions[:half],
            "philosophical_prediction": (
                "Essential interventions should disrupt the concept more "
                "than superficial ones"
            ),
        }

    def create_natural_vs_nominal_dataset(self,
                                          natural_concepts: List[str] = None,
                                          nominal_concepts: List[str] = None,
                                          samples_per_test: int = 200) -> Dict:
        """Create the complete dataset for natural vs nominal kind testing."""
        if natural_concepts is None:
            natural_concepts = ["water", "gold", "tiger"]
        if nominal_concepts is None:
            nominal_concepts = ["chair", "game", "bachelor"]

        dataset: Dict = {
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
                    "typicality": "All kinds should show prototype effects; natural kinds should also track essences",
                },
                "safety_implications": {
                    "if_natural_kind_tracking": "AI concepts align with scientific understanding – good for alignment",
                    "if_superficial_tracking": "AI concepts track appearances not reality – concerning for safety",
                    "if_nominal_appropriate": "AI tracks human purposes and functions appropriately",
                },
            },
        }

        def _populate(category_key: str, concepts: List[str]) -> None:
            for concept in concepts:
                if concept not in self.concept_definitions:
                    continue
                print(f"Generating {category_key} data for: {concept}")
                entry: Dict = {
                    "essential_contexts": self.generate_essential_property_contexts(
                        concept, samples_per_test
                    ),
                    "superficial_contexts": self.generate_superficial_property_contexts(
                        concept, samples_per_test
                    ),
                    "edge_cases": self.generate_edge_case_contexts(
                        concept, samples_per_test // 2
                    ),
                    "typicality_gradient": self.generate_typicality_gradient_contexts(
                        concept, samples_per_test
                    ),
                    "cross_domain_stability": self.generate_cross_domain_stability_test(
                        concept, samples_per_test
                    ),
                    "kind_type": self.concept_definitions[concept].kind_type.value,
                }
                sci_ctx, folk_ctx = self.generate_scientific_vs_folk_contexts(
                    concept, samples_per_test // 2
                )
                entry["scientific_contexts"] = sci_ctx
                entry["folk_contexts"] = folk_ctx
                dataset[category_key][concept] = entry

        _populate("natural_kinds", natural_concepts)
        _populate("nominal_kinds", nominal_concepts)
        _populate("artifact_kinds", ["clock", "hammer"])

        dataset["comparison_tests"] = self._generate_comparison_tests(
            natural_concepts, nominal_concepts, samples_per_test // 2
        )
        return dataset

    def _generate_comparison_tests(self, natural_concepts: List[str],
                                   nominal_concepts: List[str],
                                   num_samples: int) -> Dict:
        templates = [
            "Unlike {nom}, {nat} has an essential nature that is {prop}.",
            "While {nom} is defined by human purposes, {nat} is naturally {prop}.",
            "The difference between {nat} and {nom} is that the former has {prop}.",
            "Scientific discovery can revise our understanding of {nat} as {prop}, but not {nom}.",
            "Essential properties like {prop} determine {nat} membership, unlike {nom}.",
            "Natural kinds like {nat} have {prop}, while conventional kinds like {nom} are human-defined.",
            "The essence of {nat} involves {prop}, whereas {nom} exists by social agreement.",
            "Science discovers that {nat} is {prop}, but {nom} is stipulated by humans.",
        ]
        comparisons = []
        for _ in range(num_samples):
            nat = random.choice(natural_concepts)
            nom = random.choice(nominal_concepts)
            prop = random.choice(self.concept_definitions[nat].essential_properties)
            comparisons.append(
                random.choice(templates).format(nat=nat, nom=nom, prop=prop)
            )

        contrasts = []
        for nat in natural_concepts:
            for nom in nominal_concepts:
                nd = self.concept_definitions[nat]
                ld = self.concept_definitions[nom]
                contrasts.append({
                    "natural_concept": nat,
                    "nominal_concept": nom,
                    "natural_essential": random.choice(nd.essential_properties),
                    "nominal_essential": random.choice(ld.essential_properties),
                    "natural_superficial": random.choice(nd.superficial_properties),
                    "nominal_superficial": random.choice(ld.superficial_properties),
                    "prediction": f"{nat} should be more stable to superficial changes than {nom}",
                })

        return {
            "direct_comparisons": comparisons,
            "natural_vs_nominal_contrasts": contrasts,
        }

    def save_dataset(self, dataset: Dict,
                     filename: str = "natural_nominal_kinds_dataset.json") -> None:
        with open(filename, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filename}")

    def load_dataset(self, filename: str = "natural_nominal_kinds_dataset.json") -> Dict:
        with open(filename, "r") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Analyzer (full implementation)
# ---------------------------------------------------------------------------

class NaturalNominalAnalyzer:
    """
    Analyses whether AI systems track essential vs superficial properties in
    natural vs nominal kinds.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Vector extraction
    # ------------------------------------------------------------------

    def extract_concept_vector(self, concept: str, contexts: List[str],
                               layer: int = -1) -> np.ndarray:
        """
        Extract a concept representation by averaging hidden states at the
        position(s) of *concept* across all *contexts*.

        Uses the last hidden layer by default; pass a negative index to
        count from the end (e.g. layer=-6 for the 6th-from-last layer).
        Falls back to the last non-pad token when the concept token cannot
        be located in a context.
        """
        concept_vectors: List[np.ndarray] = []

        # Pre-compute token ids for the concept (take the first sub-token)
        concept_ids = self.tokenizer.encode(concept, add_special_tokens=False)
        target_id: Optional[int] = concept_ids[0] if concept_ids else None

        for context in contexts:
            try:
                tokens = self.tokenizer(
                    context, return_tensors="pt",
                    truncation=True, max_length=512,
                )
                input_ids = tokens["input_ids"][0]

                # Locate concept position(s)
                if target_id is not None:
                    positions = (input_ids == target_id).nonzero(as_tuple=True)[0]
                else:
                    positions = torch.tensor([])

                if len(positions) == 0:
                    # Fallback: last non-padding token
                    concept_pos = len(input_ids) - 1
                else:
                    concept_pos = int(positions[-1])  # last occurrence

                with torch.no_grad():
                    outputs = self.model(**tokens, output_hidden_states=True)
                    # hidden_states: tuple of (num_layers+1) tensors, each (1, seq, d)
                    hidden = outputs.hidden_states[layer]  # (1, seq, d_model)
                    vec = hidden[0, concept_pos, :].cpu().numpy()
                    concept_vectors.append(vec)

            except Exception:
                continue

        if not concept_vectors:
            raise ValueError(f"No valid vectors extracted for '{concept}'")

        return np.mean(concept_vectors, axis=0)

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def test_essential_vs_superficial_sensitivity(self, concept: str,
                                                  dataset: Dict) -> Dict:
        """
        Test whether the concept is more sensitive to essential than superficial
        property changes.

        Sensitivity is defined as (1 − cosine_similarity) between a property-
        specific context vector and a mixed baseline vector.
        """
        kind_type, concept_data = self._find_concept(concept, dataset)

        essential_vector = self.extract_concept_vector(
            concept, concept_data["essential_contexts"][:50]
        )
        superficial_vector = self.extract_concept_vector(
            concept, concept_data["superficial_contexts"][:50]
        )

        mixed = (concept_data["essential_contexts"][:25] +
                 concept_data["superficial_contexts"][:25])
        baseline_vector = self.extract_concept_vector(concept, mixed)

        def sim(u: np.ndarray, v: np.ndarray) -> float:
            return float(cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0])

        essential_sensitivity = 1.0 - sim(baseline_vector, essential_vector)
        superficial_sensitivity = 1.0 - sim(baseline_vector, superficial_vector)

        # Philosophical prediction
        if kind_type == "natural_kinds":
            # Natural kinds: concept should be STABLE under essential-context
            # variation (high similarity → low sensitivity) because the essential
            # properties form its core.  Superficial changes should alter the
            # context-vector more.
            tracks_essences = essential_sensitivity < superficial_sensitivity
        else:
            # Nominal/artifact kinds: functional (essential) properties matter
            # at least as much as superficial ones.
            tracks_essences = essential_sensitivity >= superficial_sensitivity

        return {
            "concept": concept,
            "kind_type": kind_type,
            "essential_sensitivity": float(essential_sensitivity),
            "superficial_sensitivity": float(superficial_sensitivity),
            "sensitivity_difference": float(superficial_sensitivity - essential_sensitivity),
            "tracks_essences_appropriately": bool(tracks_essences),
            "philosophical_interpretation": self._interpret_sensitivity_pattern(
                essential_sensitivity, superficial_sensitivity, kind_type
            ),
        }

    def _interpret_sensitivity_pattern(self, essential_sens: float,
                                       superficial_sens: float,
                                       kind_type: str) -> str:
        diff = superficial_sens - essential_sens
        if kind_type == "natural_kinds":
            if diff > 0.1:
                return "Strong evidence for essential property tracking (good for natural kinds)"
            elif diff > 0.05:
                return "Moderate evidence for essential property tracking"
            elif diff < -0.1:
                return "Concerning: more sensitive to superficial than essential properties"
            else:
                return "Unclear: similar sensitivity to essential and superficial properties"
        else:
            if essential_sens > superficial_sens + 0.05:
                return "Appropriately tracks functional/definitional properties"
            elif superficial_sens > essential_sens + 0.1:
                return "Concerning: overemphasises superficial properties for functional kind"
            else:
                return "Mixed: moderate sensitivity to both functional and superficial properties"

    def test_cross_domain_stability(self, concept: str, dataset: Dict) -> Dict:
        """Test whether the concept representation remains stable across domains."""
        _, concept_data = self._find_concept(concept, dataset)

        if "cross_domain_stability" not in concept_data:
            raise ValueError(f"Cross-domain data not found for '{concept}'")

        stability_data = concept_data["cross_domain_stability"]
        domain_vectors: Dict[str, np.ndarray] = {}

        for domain, contexts in stability_data.items():
            domain_vectors[domain] = self.extract_concept_vector(concept, contexts[:30])

        domain_names = list(domain_vectors.keys())
        similarities: Dict[str, float] = {}
        for i, d1 in enumerate(domain_names):
            for d2 in domain_names[i + 1:]:
                sim = float(cosine_similarity(
                    domain_vectors[d1].reshape(1, -1),
                    domain_vectors[d2].reshape(1, -1),
                )[0, 0])
                similarities[f"{d1}_vs_{d2}"] = sim

        mean_stability = float(np.mean(list(similarities.values()))) if similarities else 0.0

        return {
            "concept": concept,
            "domain_similarities": similarities,
            "mean_cross_domain_stability": mean_stability,
            "high_stability": mean_stability > 0.8,
            "philosophical_interpretation": (
                "High cross-domain stability (concept tracks stable properties)"
                if mean_stability > 0.8
                else "Low stability (concept varies markedly by context)"
                if mean_stability < 0.6
                else "Moderate cross-domain stability"
            ),
        }

    def test_typicality_effects(self, concept: str, dataset: Dict) -> Dict:
        """Test whether the concept shows prototype structure."""
        _, concept_data = self._find_concept(concept, dataset)

        if "typicality_gradient" not in concept_data:
            raise ValueError(f"Typicality data not found for '{concept}'")

        typicality_data = concept_data["typicality_gradient"]

        typical_vector = self.extract_concept_vector(
            concept, typicality_data["typical"][:30]
        )
        atypical_vector = self.extract_concept_vector(
            concept, typicality_data["atypical"][:30]
        )

        mixed = (concept_data["essential_contexts"][:20] +
                 concept_data["superficial_contexts"][:20])
        baseline_vector = self.extract_concept_vector(concept, mixed)

        def sim(u: np.ndarray, v: np.ndarray) -> float:
            return float(cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0])

        typical_sim = sim(baseline_vector, typical_vector)
        atypical_sim = sim(baseline_vector, atypical_vector)
        typicality_effect = typical_sim - atypical_sim

        return {
            "concept": concept,
            "typical_similarity_to_baseline": float(typical_sim),
            "atypical_similarity_to_baseline": float(atypical_sim),
            "typicality_effect": float(typicality_effect),
            "shows_prototype_structure": typicality_effect > 0.05,
            "philosophical_interpretation": (
                "Strong prototype structure (typical instances closer to concept centre)"
                if typicality_effect > 0.1
                else "Moderate prototype effects"
                if typicality_effect > 0.05
                else "Weak prototype structure"
                if typicality_effect >= 0
                else "Reverse typicality (atypical instances more central)"
            ),
        }

    # ------------------------------------------------------------------
    # Comprehensive pipeline
    # ------------------------------------------------------------------

    def comprehensive_natural_nominal_analysis(self, dataset: Dict,
                                               concepts: List[str] = None) -> Dict:
        """Run all analyses and compile results."""
        if concepts is None:
            concepts = []
            for cat in ["natural_kinds", "nominal_kinds", "artifact_kinds"]:
                concepts.extend(dataset.get(cat, {}).keys())

        results: Dict = {
            "essential_vs_superficial_tests": {},
            "cross_domain_stability_tests": {},
            "typicality_effect_tests": {},
            "natural_vs_nominal_comparison": {},
            "philosophical_conclusions": {},
        }

        for concept in concepts:
            print(f"  Analysing {concept}…")
            try:
                results["essential_vs_superficial_tests"][concept] = (
                    self.test_essential_vs_superficial_sensitivity(concept, dataset)
                )
            except Exception as e:
                print(f"    essential/superficial test failed for {concept}: {e}")

            try:
                results["cross_domain_stability_tests"][concept] = (
                    self.test_cross_domain_stability(concept, dataset)
                )
            except Exception as e:
                print(f"    cross-domain stability test failed for {concept}: {e}")

            try:
                results["typicality_effect_tests"][concept] = (
                    self.test_typicality_effects(concept, dataset)
                )
            except Exception as e:
                print(f"    typicality test failed for {concept}: {e}")

        results["natural_vs_nominal_comparison"] = (
            self._compare_natural_vs_nominal(results, dataset)
        )
        results["philosophical_conclusions"] = (
            self._generate_philosophical_conclusions(results)
        )
        return results

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _find_concept(self, concept: str, dataset: Dict) -> Tuple[str, Dict]:
        """Return (kind_category_key, concept_data) or raise ValueError."""
        for cat in ["natural_kinds", "nominal_kinds", "artifact_kinds"]:
            if concept in dataset.get(cat, {}):
                return cat, dataset[cat][concept]
        raise ValueError(f"Concept '{concept}' not found in dataset")

    def _compare_natural_vs_nominal(self, results: Dict, dataset: Dict) -> Dict:
        natural_concepts = list(dataset.get("natural_kinds", {}).keys())
        nominal_concepts = list(dataset.get("nominal_kinds", {}).keys())

        natural_stats = self._calculate_kind_statistics(results, natural_concepts)
        nominal_stats = self._calculate_kind_statistics(results, nominal_concepts)

        return {
            "natural_kind_patterns": natural_stats,
            "nominal_kind_patterns": nominal_stats,
            "key_differences": {
                "essential_tracking": (
                    natural_stats["mean_essential_tracking"]
                    - nominal_stats["mean_essential_tracking"]
                ),
                "stability": natural_stats["mean_stability"] - nominal_stats["mean_stability"],
                "typicality": natural_stats["mean_typicality"] - nominal_stats["mean_typicality"],
            },
            "philosophical_assessment": self._assess_kind_differences(
                natural_stats, nominal_stats
            ),
        }

    def _calculate_kind_statistics(self, results: Dict,
                                   concepts: List[str]) -> Dict:
        essential_tracking, stability_scores, typicality_scores = [], [], []

        for concept in concepts:
            if concept in results["essential_vs_superficial_tests"]:
                t = results["essential_vs_superficial_tests"][concept]
                essential_tracking.append(1.0 if t["tracks_essences_appropriately"] else 0.0)
            if concept in results["cross_domain_stability_tests"]:
                stability_scores.append(
                    results["cross_domain_stability_tests"][concept]["mean_cross_domain_stability"]
                )
            if concept in results["typicality_effect_tests"]:
                typicality_scores.append(
                    results["typicality_effect_tests"][concept]["typicality_effect"]
                )

        return {
            "concepts": concepts,
            "mean_essential_tracking": float(np.mean(essential_tracking)) if essential_tracking else 0.0,
            "mean_stability": float(np.mean(stability_scores)) if stability_scores else 0.0,
            "mean_typicality": float(np.mean(typicality_scores)) if typicality_scores else 0.0,
            "sample_size": len(concepts),
        }

    def _assess_kind_differences(self, natural_stats: Dict,
                                 nominal_stats: Dict) -> str:
        essential_diff = (natural_stats["mean_essential_tracking"]
                          - nominal_stats["mean_essential_tracking"])
        stability_diff = natural_stats["mean_stability"] - nominal_stats["mean_stability"]

        if essential_diff > 0.2 and stability_diff > 0.1:
            return "Strong evidence AI distinguishes natural from nominal kinds appropriately"
        elif essential_diff > 0.1 or stability_diff > 0.05:
            return "Moderate evidence for natural/nominal kind distinction"
        elif essential_diff < -0.1:
            return "Concerning: AI treats natural kinds like nominal kinds"
        else:
            return "Unclear: no strong differentiation between natural and nominal kinds"

    def _generate_philosophical_conclusions(self, results: Dict) -> Dict:
        es_tests = results.get("essential_vs_superficial_tests", {})
        appropriate = sum(
            1 for t in es_tests.values() if t.get("tracks_essences_appropriately", False)
        )
        total = len(es_tests)
        tracking_ratio = appropriate / total if total > 0 else 0.0

        stability_scores = [
            t["mean_cross_domain_stability"]
            for t in results.get("cross_domain_stability_tests", {}).values()
        ]
        mean_stability = float(np.mean(stability_scores)) if stability_scores else 0.0

        if tracking_ratio > 0.75 and mean_stability > 0.8:
            overall = "Strong evidence AI tracks essential properties and natural kinds appropriately"
            safety = "POSITIVE: AI concepts align with scientific understanding of natural categories"
        elif tracking_ratio > 0.5 and mean_stability > 0.6:
            overall = "Moderate evidence for appropriate essential property tracking"
            safety = "MIXED: Some alignment with scientific categories, but inconsistent"
        else:
            overall = "Weak evidence for essential property tracking"
            safety = "CONCERNING: AI may not distinguish essential from superficial properties"

        return {
            "overall_assessment": overall,
            "safety_implication": safety,
            "essential_tracking_ratio": float(tracking_ratio),
            "mean_concept_stability": float(mean_stability),
            "total_concepts_analysed": total,
        }


# ---------------------------------------------------------------------------
# Experiment orchestrator
# ---------------------------------------------------------------------------

class NaturalKindsExperiment:
    """
    Orchestrates the complete natural kinds vs nominal kinds experiment.

    Tests Kripke-Putnam predictions:
    1. Natural kinds have essential properties that determine membership.
    2. Superficial properties can change without affecting kind membership.
    3. Scientific discovery can revise essential properties.
    4. Natural kinds are more stable across contexts than nominal kinds.
    """

    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto" else device
        )

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

        self.dataset_generator = NaturalNominalDatasetGenerator()
        self.analyzer = NaturalNominalAnalyzer(self.model, self.tokenizer)

        print("✓ Natural Kinds Experiment initialised")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def generate_experimental_dataset(self,
                                      natural_concepts: List[str] = None,
                                      nominal_concepts: List[str] = None,
                                      samples_per_test: int = 200) -> Dict:
        if natural_concepts is None:
            natural_concepts = ["water", "gold", "tiger"]
        if nominal_concepts is None:
            nominal_concepts = ["chair", "game", "bachelor"]

        print("\n" + "=" * 50)
        print("GENERATING NATURAL VS NOMINAL KINDS DATASET")
        print("=" * 50)
        print(f"Natural kinds: {natural_concepts}")
        print(f"Nominal kinds: {nominal_concepts}")
        print(f"Samples per test: {samples_per_test}")

        dataset = self.dataset_generator.create_natural_vs_nominal_dataset(
            natural_concepts=natural_concepts,
            nominal_concepts=nominal_concepts,
            samples_per_test=samples_per_test,
        )

        with open("natural_nominal_experiment_dataset.json", "w") as f:
            json.dump(dataset, f, indent=2)
        print("✓ Dataset saved to 'natural_nominal_experiment_dataset.json'")
        return dataset

    # ------------------------------------------------------------------
    # Analysis steps (exposed individually for granular reporting)
    # ------------------------------------------------------------------

    def run_essential_vs_superficial_analysis(self, dataset: Dict,
                                              concepts: List[str] = None) -> Dict:
        concepts = concepts or self._all_concepts(dataset)
        print("\n" + "=" * 50)
        print("ESSENTIAL VS SUPERFICIAL PROPERTY ANALYSIS")
        print("=" * 50)

        results = {}
        for concept in concepts:
            print(f"\nAnalysing {concept}…")
            try:
                result = self.analyzer.test_essential_vs_superficial_sensitivity(
                    concept, dataset
                )
                results[concept] = result
                print(f"  Essential sensitivity:        {result['essential_sensitivity']:.3f}")
                print(f"  Superficial sensitivity:      {result['superficial_sensitivity']:.3f}")
                tick = "✓" if result["tracks_essences_appropriately"] else "✗"
                print(f"  Tracks essences appropriately: {tick}")
                print(f"  Interpretation: {result['philosophical_interpretation']}")
            except Exception as e:
                print(f"  Error: {e}")
        return results

    def run_cross_domain_stability_analysis(self, dataset: Dict,
                                            concepts: List[str] = None) -> Dict:
        concepts = concepts or self._all_concepts(dataset)
        print("\n" + "=" * 50)
        print("CROSS-DOMAIN STABILITY ANALYSIS")
        print("=" * 50)

        results = {}
        for concept in concepts:
            print(f"\nAnalysing cross-domain stability for {concept}…")
            try:
                result = self.analyzer.test_cross_domain_stability(concept, dataset)
                results[concept] = result
                tick = "✓" if result["high_stability"] else "✗"
                print(f"  Mean cross-domain stability: {result['mean_cross_domain_stability']:.3f}")
                print(f"  High stability: {tick}")
                print(f"  Interpretation: {result['philosophical_interpretation']}")
                print("  Domain similarities:")
                for pair, sim in result["domain_similarities"].items():
                    print(f"    {pair}: {sim:.3f}")
            except Exception as e:
                print(f"  Error: {e}")
        return results

    def run_typicality_analysis(self, dataset: Dict,
                                concepts: List[str] = None) -> Dict:
        concepts = concepts or self._all_concepts(dataset)
        print("\n" + "=" * 50)
        print("TYPICALITY EFFECT ANALYSIS")
        print("=" * 50)

        results = {}
        for concept in concepts:
            print(f"\nAnalysing typicality effects for {concept}…")
            try:
                result = self.analyzer.test_typicality_effects(concept, dataset)
                results[concept] = result
                tick = "✓" if result["shows_prototype_structure"] else "✗"
                print(f"  Typicality effect:                      {result['typicality_effect']:.3f}")
                print(f"  Shows prototype structure:              {tick}")
                print(f"  Typical similarity to baseline:         {result['typical_similarity_to_baseline']:.3f}")
                print(f"  Atypical similarity to baseline:        {result['atypical_similarity_to_baseline']:.3f}")
                print(f"  Interpretation: {result['philosophical_interpretation']}")
            except Exception as e:
                print(f"  Error: {e}")
        return results

    def run_intervention_analysis(self, dataset: Dict,
                                  concepts: List[str] = None) -> Dict:
        if concepts is None:
            concepts = ["water", "gold", "chair", "game"]
        print("\n" + "=" * 50)
        print("PROPERTY INTERVENTION ANALYSIS")
        print("=" * 50)

        results = {}
        for concept in concepts:
            print(f"\nGenerating intervention tests for {concept}…")
            try:
                intervention_data = self.dataset_generator.generate_intervention_dataset(
                    concept, 20
                )
                print(f"  Generated {len(intervention_data['essential_interventions'])} "
                      f"essential interventions")
                print(f"  Generated {len(intervention_data['superficial_interventions'])} "
                      f"superficial interventions")
                if intervention_data["essential_interventions"]:
                    print(f"  Sample essential: {intervention_data['essential_interventions'][0]}")
                if intervention_data["superficial_interventions"]:
                    print(f"  Sample superficial: {intervention_data['superficial_interventions'][0]}")
                results[concept] = intervention_data
            except Exception as e:
                print(f"  Error: {e}")
        return results

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_complete_analysis(self, dataset: Dict) -> Dict:
        print("\n" + "=" * 60)
        print("RUNNING COMPLETE NATURAL KINDS ANALYSIS")
        print("=" * 60)

        # Run sub-analyses for console output
        self.run_essential_vs_superficial_analysis(dataset)
        self.run_cross_domain_stability_analysis(dataset)
        self.run_typicality_analysis(dataset)
        intervention_results = self.run_intervention_analysis(dataset)

        # Compile everything through the analyzer's comprehensive method
        comprehensive_results = self.analyzer.comprehensive_natural_nominal_analysis(dataset)
        comprehensive_results["intervention_tests"] = intervention_results
        return comprehensive_results

    def run_full_experiment(self,
                            natural_concepts: List[str] = None,
                            nominal_concepts: List[str] = None,
                            samples_per_test: int = 200) -> Dict:
        print("\n" + "=" * 60)
        print("NATURAL KINDS EXPERIMENT: FULL ANALYSIS")
        print("=" * 60)
        print("Testing whether AI systems track essential vs superficial properties")
        print("in natural kinds (water, gold) vs nominal kinds (chair, game)")

        dataset = self.generate_experimental_dataset(
            natural_concepts=natural_concepts,
            nominal_concepts=nominal_concepts,
            samples_per_test=samples_per_test,
        )

        results = self.run_complete_analysis(dataset)
        self.visualize_results(results)
        self.generate_report(results, dataset)

        with open("natural_kinds_experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)

        if "philosophical_conclusions" in results:
            c = results["philosophical_conclusions"]
            print(f"Overall Assessment:       {c['overall_assessment']}")
            print(f"AI Safety Implication:    {c['safety_implication']}")
            print(f"Essential Tracking Ratio: {c.get('essential_tracking_ratio', 0):.1%}")

        print("\nGenerated Files:")
        print("  • natural_kinds_experiment_results.json")
        print("  • natural_nominal_experiment_dataset.json")
        print("  • natural_kinds_report.md")
        print("  • natural_kinds_analysis.png")

        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_results(self, results: Dict,
                          save_path: str = "natural_kinds_analysis.png") -> None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Natural vs Nominal Kinds: AI Concept Analysis", fontsize=16)

        es_tests = results.get("essential_vs_superficial_tests", {})
        cd_tests = results.get("cross_domain_stability_tests", {})
        typ_tests = results.get("typicality_effect_tests", {})

        # 1. Essential sensitivity boxplot
        ax = axes[0, 0]
        natural_ess = [v["essential_sensitivity"] for v in es_tests.values()
                       if v.get("kind_type") == "natural_kinds"]
        nominal_ess = [v["essential_sensitivity"] for v in es_tests.values()
                       if v.get("kind_type") == "nominal_kinds"]
        if natural_ess or nominal_ess:
            data = [x for x in [natural_ess, nominal_ess] if x]
            labels = [l for l, x in zip(["Natural Kinds", "Nominal Kinds"],
                                         [natural_ess, nominal_ess]) if x]
            ax.boxplot(data, tick_labels=labels)
        ax.set_title("Essential Property Sensitivity")
        ax.set_ylabel("Sensitivity Score")

        # 2. Cross-domain stability boxplot
        ax = axes[0, 1]
        natural_stab = [v["mean_cross_domain_stability"] for k, v in cd_tests.items()
                        if es_tests.get(k, {}).get("kind_type") == "natural_kinds"]
        nominal_stab = [v["mean_cross_domain_stability"] for k, v in cd_tests.items()
                        if es_tests.get(k, {}).get("kind_type") == "nominal_kinds"]
        if natural_stab or nominal_stab:
            data = [x for x in [natural_stab, nominal_stab] if x]
            labels = [l for l, x in zip(["Natural Kinds", "Nominal Kinds"],
                                         [natural_stab, nominal_stab]) if x]
            ax.boxplot(data, tick_labels=labels)
        ax.set_title("Cross-Domain Stability")
        ax.set_ylabel("Stability Score")

        # 3. Typicality effects bar chart
        ax = axes[1, 0]
        if typ_tests:
            concepts = list(typ_tests.keys())
            effects = [typ_tests[c]["typicality_effect"] for c in concepts]
            colour_map = {"natural_kinds": "blue", "nominal_kinds": "red",
                          "artifact_kinds": "green"}
            colours = [colour_map.get(es_tests.get(c, {}).get("kind_type", ""), "grey")
                       for c in concepts]
            ax.bar(range(len(concepts)), effects, color=colours, alpha=0.7)
            ax.set_xticks(range(len(concepts)))
            ax.set_xticklabels(concepts, rotation=45)
        ax.set_title("Typicality Effects by Concept")
        ax.set_ylabel("Typicality Effect")

        # 4. Essence tracking % by kind type
        ax = axes[1, 1]
        nat_track = [1 if v["tracks_essences_appropriately"] else 0
                     for v in es_tests.values() if v.get("kind_type") == "natural_kinds"]
        nom_track = [1 if v["tracks_essences_appropriately"] else 0
                     for v in es_tests.values() if v.get("kind_type") == "nominal_kinds"]
        if nat_track or nom_track:
            pcts = [np.mean(x) * 100 for x in [nat_track, nom_track] if x]
            labs = [l for l, x in zip(["Natural Kinds", "Nominal Kinds"],
                                       [nat_track, nom_track]) if x]
            ax.bar(labs, pcts, color=["blue", "red"][: len(pcts)], alpha=0.7)
            ax.set_ylim(0, 100)
        ax.set_title("% Tracking Essences Appropriately")
        ax.set_ylabel("Percentage (%)")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Visualisation saved to {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self, results: Dict, dataset: Dict,
                        save_path: str = "natural_kinds_report.md") -> str:
        c = results.get("philosophical_conclusions", {})
        es_tests = results.get("essential_vs_superficial_tests", {})
        cd_tests = results.get("cross_domain_stability_tests", {})
        typ_tests = results.get("typicality_effect_tests", {})
        interventions = results.get("intervention_tests", {})
        comparison = results.get("natural_vs_nominal_comparison", {})

        tracking_ratio = c.get("essential_tracking_ratio", 0)

        report = f"""# Natural vs Nominal Kinds Analysis Report
## Testing AI Systems for Essential Property Tracking

### Executive Summary

**Key Finding**: {c.get("overall_assessment", "N/A")}

**AI Safety Implication**: {c.get("safety_implication", "N/A")}

**Quantitative Summary**:
- Essential tracking ratio: {tracking_ratio:.1%}
- Mean concept stability: {c.get("mean_concept_stability", 0):.3f}
- Total concepts analysed: {c.get("total_concepts_analysed", 0)}

---

### Essential vs Superficial Property Sensitivity

**Natural Kinds**:
"""
        for concept, data in es_tests.items():
            if data.get("kind_type") == "natural_kinds":
                tick = "✓" if data["tracks_essences_appropriately"] else "✗"
                report += (
                    f"- **{concept.capitalize()}**: "
                    f"Essential: {data['essential_sensitivity']:.3f}, "
                    f"Superficial: {data['superficial_sensitivity']:.3f} {tick}\n"
                )
        report += "\n**Nominal Kinds**:\n"
        for concept, data in es_tests.items():
            if data.get("kind_type") == "nominal_kinds":
                tick = "✓" if data["tracks_essences_appropriately"] else "✗"
                report += (
                    f"- **{concept.capitalize()}**: "
                    f"Essential: {data['essential_sensitivity']:.3f}, "
                    f"Superficial: {data['superficial_sensitivity']:.3f} {tick}\n"
                )

        report += "\n---\n\n### Cross-Domain Stability\n\n"
        for concept, data in cd_tests.items():
            stab = data["mean_cross_domain_stability"]
            level = "High" if data["high_stability"] else "Low"
            report += (
                f"**{concept.capitalize()}**: {stab:.3f} ({level} stability)\n"
                f"- {data['philosophical_interpretation']}\n\n"
            )

        report += "---\n\n### Typicality Effects\n\n"
        for concept, data in typ_tests.items():
            proto = "Yes" if data["shows_prototype_structure"] else "No"
            report += (
                f"**{concept.capitalize()}**: effect = {data['typicality_effect']:.3f} "
                f"(Prototype: {proto})\n"
                f"- {data['philosophical_interpretation']}\n\n"
            )

        if comparison:
            kd = comparison.get("key_differences", {})
            nat_p = comparison.get("natural_kind_patterns", {})
            nom_p = comparison.get("nominal_kind_patterns", {})
            report += f"""---

### Natural vs Nominal Comparison

**Key Differences**:
- Essential tracking: {kd.get("essential_tracking", 0):.3f}
- Stability: {kd.get("stability", 0):.3f}
- Typicality: {kd.get("typicality", 0):.3f}

**Assessment**: {comparison.get("philosophical_assessment", "N/A")}

Natural kinds — mean essential tracking: {nat_p.get("mean_essential_tracking", 0):.3f}, mean stability: {nat_p.get("mean_stability", 0):.3f}
Nominal kinds — mean essential tracking: {nom_p.get("mean_essential_tracking", 0):.3f}, mean stability: {nom_p.get("mean_stability", 0):.3f}

"""

        if interventions:
            report += "---\n\n### Property Intervention Analysis\n\n"
            for concept, data in interventions.items():
                report += (
                    f"**{concept.capitalize()}**: "
                    f"{len(data['essential_interventions'])} essential / "
                    f"{len(data['superficial_interventions'])} superficial interventions\n"
                    f"- Prediction: {data['philosophical_prediction']}\n"
                )
                if data["essential_interventions"]:
                    report += f'- Sample essential: "{data["essential_interventions"][0]}"\n'
                if data["superficial_interventions"]:
                    report += f'- Sample superficial: "{data["superficial_interventions"][0]}"\n'
                report += "\n"

        report += "---\n\n### AI Safety Implications\n\n"
        if tracking_ratio > 0.7:
            report += (
                "- ✅ AI concepts show good essential property tracking\n"
                "- ✅ Interpretability methods may detect genuine understanding\n"
                "- ✅ AI concepts align with scientific categorisation\n"
            )
        elif tracking_ratio > 0.4:
            report += (
                "- ⚠️ Mixed: AI shows partial essential property tracking\n"
                "- ⚠️ Investigate which concept types track essences appropriately\n"
            )
        else:
            report += (
                "- ❌ AI concepts primarily track superficial properties\n"
                "- ❌ May indicate statistical pattern matching rather than understanding\n"
                "- ❌ Interpretability methods may not detect genuine comprehension\n"
            )

        report += (
            "\n---\n\n"
            "*Report generated by Natural Kinds Experiment framework*\n"
            "*References: Kripke (1980) Naming and Necessity; "
            "Putnam (1975) The meaning of 'meaning'*\n"
        )

        with open(save_path, "w") as f:
            f.write(report)
        print(f"✓ Report saved to {save_path}")
        return report

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _all_concepts(dataset: Dict) -> List[str]:
        concepts: List[str] = []
        for cat in ["natural_kinds", "nominal_kinds", "artifact_kinds"]:
            concepts.extend(dataset.get(cat, {}).keys())
        return concepts


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def quick_demo() -> Dict:
    experiment = NaturalKindsExperiment(model_name="distilgpt2")
    return experiment.run_full_experiment(
        natural_concepts=["water", "gold"],
        nominal_concepts=["chair", "game"],
        samples_per_test=50,
    )


def main() -> Dict:
    print("Running simplified natural kinds experiment…")
    experiment = NaturalKindsExperiment(model_name="distilgpt2")
    return experiment.run_full_experiment(
        natural_concepts=["water", "gold"],
        nominal_concepts=["chair", "game"],
        samples_per_test=20,
    )


if __name__ == "__main__":
    main()