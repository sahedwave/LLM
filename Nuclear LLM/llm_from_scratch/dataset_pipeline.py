"""Unified Phase 4.5 dataset engine for deterministic nuclear concept corpora."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src import config
from src.artifact_lock import build_artifact_manifest, load_artifact_manifest, save_artifact_manifest
from src.data_loader import (
    CONCEPT_DICTIONARY,
    _build_vocab_impl,
    infer_concept,
    infer_entry_type,
    infer_topic,
    normalize_text,
    split_into_sentences,
)
from src.execution_graph import (
    assert_execution_allowed,
    assert_side_execution_forbidden,
    execution_guard,
    import_guard,
)
from src.locked_artifacts import write_locked_artifacts

GRAPH_NODE = "BUILD"

import_guard(GRAPH_NODE, require_artifacts=False)


DATA_DIR = config.PROJECT_DIR / "data"
SYNTHETIC_DATASET_PATH = DATA_DIR / "synthetic_concept_dataset.jsonl"
JSON_BUILDER_DIR = Path(
    os.environ.get(
        "NUCLEAR_LLM_JSON_BUILDER_DIR",
        str(config.PROJECT_DIR.parent.parent / "json builder"),
    )
).expanduser()
JSON_BUILDER_DATA_DIR = JSON_BUILDER_DIR / "data"
JSON_BUILDER_GLOB = "generated_dataset*.jsonl"
CANONICAL_TYPES = ("definition", "explanation", "mechanism", "safety_analysis")
TYPE_TARGETS = {
    "definition": 20,
    "explanation": 20,
    "mechanism": 100,
    "safety_analysis": 20,
}
SCENARIO_BY_TYPE = {
    "definition": "definition",
    "explanation": "operational",
    "mechanism": "operational",
    "safety_analysis": "accident",
}
INSTRUCTION_TOKEN_BY_TYPE = {
    "definition": "[DEFINE]",
    "explanation": "[EXPLAIN]",
    "mechanism": "[EXPLAIN]",
    "safety_analysis": "[SCENARIO]",
}
CAUSAL_MARKERS = (
    "when",
    "then",
    "because",
    "this leads to",
    "as a result",
    "consequently",
    "therefore",
    "in turn",
    "so that",
)
CONCEPT_CORE_TERMS = {
    "neutron physics": ("neutron", "flux", "moderation", "scattering", "resonance", "spectrum", "leakage"),
    "reactor kinetics": ("reactivity", "k-effective", "delayed neutrons", "control rods", "doppler", "criticality", "prompt"),
    "thermal hydraulics": ("coolant", "boiling", "steam", "flow", "temperature", "overheating", "heat transfer", "critical heat flux"),
    "materials behavior": ("fuel", "cladding", "swelling", "oxidation", "embrittlement", "corrosion", "pellet", "zircaloy"),
    "safety systems": ("loca", "containment", "emergency", "shutdown", "accident", "blackout", "protection", "eccs"),
}
QUERY_ALIASES = {
    "neutron flux": ("neutron flux",),
    "moderation": ("neutron moderation", "moderation"),
    "resonance absorption": ("resonance absorption",),
    "diffusion length": ("diffusion length",),
    "neutron leakage": ("neutron leakage",),
    "k-effective": ("k-effective",),
    "reactivity insertion": ("reactivity", "reactivity insertion"),
    "delayed neutrons": ("delayed neutrons",),
    "control rods": ("control rods",),
    "Doppler feedback": ("doppler effect in reactors", "doppler feedback"),
    "coolant flow": ("coolant flow",),
    "boiling heat transfer": ("coolant boiling", "boiling heat transfer"),
    "critical heat flux": ("critical heat flux",),
    "natural circulation": ("natural circulation",),
    "steam generator heat transfer": ("steam generator heat transfer",),
    "hot channel behavior": ("reactor overheating", "hot channel behavior"),
    "uranium dioxide thermal conductivity": ("fuel thermal conductivity", "uranium dioxide thermal conductivity"),
    "fuel swelling": ("fuel swelling",),
    "zircaloy oxidation": ("zircaloy oxidation",),
    "radiation embrittlement": ("radiation embrittlement",),
    "coolant chemistry corrosion": ("coolant chemistry corrosion",),
    "loss of coolant accident": ("loca", "loss of coolant accident"),
    "decay heat removal": ("decay heat", "decay heat removal"),
    "emergency core cooling system": ("emergency core cooling system",),
    "containment": ("containment",),
    "station blackout": ("station blackout",),
    "reactor trip system": ("reactor trip system", "reactor trip"),
}
GENERIC_FILLER_PHRASES = (
    "this concept matters",
    "this concept is important",
    "is used to describe",
    "matters because it matters",
    "helps explain how",
)
TEMPORAL_MARKERS = ("when", "then", "as a result", "consequently", "therefore", "in turn", "once", "after")
MECHANISM_MARKERS = ("because", "causes", "leads to", "results in", "drives", "raises", "reduces", "increases")
PHASE_3_ARCHITECTURE_NOTE = """Phase 3 Dataset Architecture (GPT-style scaling system)
Raw Data -> Cleaning -> Normalization -> Validation -> Causal Scoring -> Deterministic Selection -> Training Set

- Synthetic concept generation is deterministic, mechanism-heavy, and constrained to one nuclear concept per sample.
- Validation removes malformed, repetitive, multi-concept, or low-signal samples before they enter the training corpus.
- Causal scoring ranks samples by how clearly they express physical cause-and-effect chains rather than only checking format.
- Deterministic selection preserves concept balance while keeping mechanism samples as the dominant training signal.
- Version locking still binds the resulting text, tokenizer vocabulary, and checkpoints through the existing artifact manifest.
"""
SOURCE_PREFERENCE = {
    "json_builder": 0,
    "synthetic": 1,
}


@execution_guard("build_vocab", GRAPH_NODE)
def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Single vocab authority for the entire project."""
    if os.environ.get(config.ALLOW_VOCAB_BUILD_ENV) != "1":
        raise RuntimeError("VOCAB DRIFT VIOLATION")
    return _build_vocab_impl(text)


def source_priority(source: str) -> int:
    """Rank human-curated builder data ahead of synthetic backfill."""
    return SOURCE_PREFERENCE.get(source.split(":", 1)[0], 99)


def synthetic_backfill_enabled() -> bool:
    """Return whether synthetic records should be used to fill missing coverage."""
    return os.environ.get(config.ENABLE_SYNTHETIC_BACKFILL_ENV) == "1"


def json_builder_dataset_paths() -> list[Path]:
    """Locate generated JSONL datasets from the external raw-text builder project."""
    if not JSON_BUILDER_DATA_DIR.exists():
        return []
    return sorted(
        path
        for path in JSON_BUILDER_DATA_DIR.glob(JSON_BUILDER_GLOB)
        if path.is_file()
    )


def majority_value(values: list[str], fallback: str = "") -> str:
    """Return the deterministic majority choice from a small string list."""
    cleaned = [value.strip() for value in values if value and value.strip()]
    if not cleaned:
        return fallback
    counts = Counter(cleaned)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def normalize_json_builder_row(row: Dict[str, object]) -> Dict[str, str] | None:
    """Normalize one raw sentence-level builder row into a canonical training hint."""
    text = normalize_record_text(str(row.get("text", "")))
    if not text:
        return None

    provided_concept = str(row.get("concept", "")).strip()
    if provided_concept.lower() == "general":
        provided_concept = ""

    topic = infer_concept(text, provided_concept=provided_concept)
    if topic not in CONCEPT_DICTIONARY:
        return None

    category = infer_entry_type(text, provided_type=str(row.get("type", "")).strip())
    subject = infer_topic(text)
    if not subject or subject == "general reactor engineering":
        subject = topic

    return {
        "topic": topic,
        "subject": subject,
        "category": category,
        "text": text,
    }


def normalize_json_builder_structured_row(row: Dict[str, object]) -> Dict[str, object] | None:
    """Normalize one structured builder row into the trainer-ready record format."""
    answer = normalize_record_text(str(row.get("answer", "")))
    reasoning = normalize_record_text(str(row.get("reasoning", "")))
    effect = normalize_record_text(str(row.get("effect", "")))
    if not answer or not reasoning or not effect:
        return None

    provided_topic = str(row.get("topic") or row.get("concept") or "").strip()
    if provided_topic.lower() == "general":
        provided_topic = ""

    combined_text = normalize_record_text(" ".join([answer, reasoning, effect]))
    topic = infer_concept(combined_text, provided_concept=provided_topic)
    if topic not in CONCEPT_DICTIONARY:
        return None

    category = infer_entry_type(
        combined_text,
        provided_type=str(row.get("category") or row.get("type") or "").strip(),
    )
    if category not in CANONICAL_TYPES:
        return None

    subject = str(row.get("subject", "")).strip()
    if not subject:
        subject = infer_topic(combined_text)
    if not subject or subject == "general reactor engineering":
        subject = topic

    scenario = str(row.get("scenario", "")).strip() or SCENARIO_BY_TYPE[category]
    instruction = str(row.get("instruction", "")).strip() or INSTRUCTION_TOKEN_BY_TYPE[category]
    question = normalize_record_text(str(row.get("question", "")))
    if not question:
        question = build_fallback_question(
            {
                "topic": topic,
                "subject": subject,
                "category": category,
            }
        )

    return {
        "source": "json_builder",
        "topic": topic,
        "subject": subject,
        "category": category,
        "scenario": scenario,
        "instruction": instruction,
        "question": question,
        "answer": answer,
        "reasoning": reasoning,
        "effect": effect,
        "text": normalize_record_text(str(row.get("text", ""))) or answer,
        "source_chunk": normalize_record_text(str(row.get("source_chunk", ""))) or answer,
    }


def build_json_builder_candidates() -> List[Dict[str, object]]:
    """Upgrade sentence-level builder JSONL rows into structured training candidates."""
    candidates: List[Dict[str, object]] = []
    for path in json_builder_dataset_paths():
        sentence_rows: List[Dict[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            structured = normalize_json_builder_structured_row(row)
            if structured is not None:
                candidates.append(structured)
                continue
            normalized = normalize_json_builder_row(row)
            if normalized is not None:
                sentence_rows.append(normalized)

        for window_size in (3, 4):
            if len(sentence_rows) < window_size:
                continue
            for index in range(len(sentence_rows) - window_size + 1):
                window = sentence_rows[index : index + window_size]
                combined_text = normalize_record_text(" ".join(item["text"] for item in window))
                if len(split_into_sentences(combined_text)) < 3:
                    continue

                topic = infer_concept(
                    combined_text,
                    provided_concept=majority_value(
                        [item["topic"] for item in window if item["topic"] in CONCEPT_DICTIONARY]
                    ),
                )
                if topic not in CONCEPT_DICTIONARY:
                    continue

                category = infer_entry_type(
                    combined_text,
                    provided_type=majority_value(
                        [item["category"] for item in window if item["category"] in CANONICAL_TYPES],
                        fallback="explanation",
                    ),
                )
                if category not in CANONICAL_TYPES:
                    continue

                subject = infer_topic(combined_text)
                if not subject or subject == "general reactor engineering":
                    subject = infer_topic(window[0]["text"])
                if not subject or subject == "general reactor engineering":
                    subject = topic

                answer = window[0]["text"]
                reasoning = window[1]["text"]
                effect = normalize_record_text(" ".join(item["text"] for item in window[2:]))
                if not effect:
                    continue

                candidates.append(
                    {
                        "source": "json_builder",
                        "topic": topic,
                        "subject": subject,
                        "category": category,
                        "scenario": SCENARIO_BY_TYPE[category],
                        "instruction": INSTRUCTION_TOKEN_BY_TYPE[category],
                        "question": build_fallback_question(
                            {
                                "topic": topic,
                                "subject": subject,
                                "category": category,
                            }
                        ),
                        "answer": answer,
                        "reasoning": reasoning,
                        "effect": effect,
                        "text": combined_text,
                    }
                )

    return candidates


def spec(
    concept: str,
    term: str,
    definition: str,
    importance: str,
    detail: str,
    cause: str,
    lead: str,
    result: str,
    consequence: str,
    safety: str,
    control: str,
) -> Dict[str, str]:
    """Create one deterministic subject specification."""
    return {
        "concept": concept,
        "term": term,
        "definition": definition,
        "importance": importance,
        "detail": detail,
        "cause": cause,
        "lead": lead,
        "result": result,
        "consequence": consequence,
        "safety": safety,
        "control": control,
    }


SPECS: List[Dict[str, str]] = [
    spec(
        "neutron physics",
        "neutron flux",
        "Neutron flux is the rate at which neutrons pass through a unit area in a reactor region.",
        "It links neutron population directly to reaction rate and local power generation.",
        "Flux measurements are used to track how intensely the chain reaction is proceeding in different parts of the core.",
        "reactivity rises and the neutron population grows",
        "more neutrons pass through each part of the fuel every second",
        "the fission rate climbs in the affected region",
        "local power and fuel temperature rise until feedback or control action offsets the change",
        "Strong local flux peaking can erode thermal margin in nearby fuel rods.",
        "Flux mapping, control-rod positioning, and core design limits are used to keep the neutron field within safe bounds.",
    ),
    spec(
        "neutron physics",
        "moderation",
        "Moderation is the slowing of fast neutrons through repeated scattering collisions in the moderator.",
        "Thermal reactors depend on moderation because slower neutrons are more likely to induce fission in uranium-235.",
        "Moderator density and composition strongly influence the spectrum of neutrons that remains in the core.",
        "fast neutrons collide repeatedly with light nuclei in the moderator",
        "their kinetic energy falls step by step",
        "more neutrons enter the energy range where thermal fission is likely",
        "the reactor can sustain the intended spectrum with better neutron economy",
        "Poor moderation can shift the neutron spectrum away from the design basis and weaken core performance.",
        "Moderator purity, temperature limits, and lattice design are used to keep the slowing-down process predictable.",
    ),
    spec(
        "neutron physics",
        "resonance absorption",
        "Resonance absorption is the strong absorption that occurs when neutrons pass through specific energy ranges in certain nuclides.",
        "It removes neutrons during slowing down and directly affects neutron economy.",
        "Uranium-238 is a major contributor to resonance absorption in thermal reactor analysis.",
        "neutrons slow down from high energy toward the thermal range",
        "they pass through energy bands where absorption probability becomes sharply higher",
        "some neutrons are captured before they can reach more useful energies",
        "fewer neutrons remain available to sustain the later stages of the chain reaction",
        "Excess resonance absorption can lower reactivity and compress operating margin.",
        "Fuel geometry, moderation conditions, and burnable absorber strategy are chosen to manage resonance losses.",
    ),
    spec(
        "neutron physics",
        "diffusion length",
        "Diffusion length is the characteristic distance a thermal neutron travels before it is absorbed in a material.",
        "It helps determine leakage behavior in finite reactor systems.",
        "Materials with strong scattering and weaker absorption allow neutrons to wander farther before they disappear.",
        "thermal neutrons scatter many times inside the material",
        "their path extends as each collision changes direction without immediate absorption",
        "the chance of reaching the core boundary before absorption becomes larger",
        "neutron leakage rises if migration distance becomes large compared with the active core dimensions",
        "High leakage can reduce reactivity and distort the intended power shape near the core edge.",
        "Reflectors and core dimensions are selected so neutron migration remains compatible with criticality goals.",
    ),
    spec(
        "neutron physics",
        "neutron leakage",
        "Neutron leakage is the loss of neutrons from the active core before they can cause another useful interaction.",
        "Escaped neutrons do not support the chain reaction and therefore weaken neutron economy.",
        "Core shape, size, and reflector design all influence how much leakage occurs.",
        "neutrons are born near the outer region of the core",
        "their random walk carries some of them toward the physical boundary",
        "a fraction of those neutrons leaves the active region before causing another useful interaction",
        "the effective multiplication of the reactor falls because those neutrons no longer support fission",
        "Excessive leakage can force higher enrichment or stronger control adjustments than the design intended.",
        "Reflectors, loading patterns, and geometry optimization are used to keep leakage within acceptable limits.",
    ),
    spec(
        "reactor kinetics",
        "k-effective",
        "k-effective is the ratio of neutrons in one generation to neutrons in the previous generation.",
        "It indicates whether the chain reaction is shrinking, steady, or growing.",
        "A reactor with k-effective equal to one is critical and maintains a stable neutron population on average.",
        "k-effective rises above one",
        "each neutron generation produces more neutrons than the previous one",
        "the neutron population grows from generation to generation",
        "power increases until negative feedback or control action drives the core back toward criticality",
        "Persistent departure of k-effective from one creates unwanted power change and reduces operating margin.",
        "Control rods, soluble absorbers, and temperature feedback are used to hold k-effective near its target value.",
    ),
    spec(
        "reactor kinetics",
        "reactivity insertion",
        "Reactivity insertion is a change that moves the reactor away from exact criticality by changing neutron production or neutron loss.",
        "Small reactivity changes can produce measurable changes in neutron population and power.",
        "Positive reactivity pushes power upward, while negative reactivity pushes power downward.",
        "a control action or physical change alters the neutron balance",
        "the core moves away from exact criticality",
        "neutron population begins to rise or fall according to the sign of the insertion",
        "reactor power follows that neutron change until feedback and control establish a new balance",
        "Uncontrolled positive reactivity can cause rapid power rise, while excessive negative reactivity can destabilize operation.",
        "Reactivity management uses absorbers, moderator conditions, and careful procedures to keep transients bounded.",
    ),
    spec(
        "reactor kinetics",
        "delayed neutrons",
        "Delayed neutrons are neutrons emitted by certain fission products after the original fission event rather than immediately.",
        "They slow the effective time scale of reactor power change and make controlled operation possible.",
        "Without delayed neutrons, practical mechanical control of power would be far more difficult.",
        "certain fission products decay after a short delay",
        "additional neutrons are released after the prompt-neutron burst has already occurred",
        "the effective neutron generation time becomes much longer",
        "reactor power changes slowly enough for control systems and operators to respond safely",
        "If operation moves too close to the prompt regime, normal control time becomes too short for comfortable intervention.",
        "Protection systems and reactivity limits are designed to keep routine operation within the delayed-neutron regime.",
    ),
    spec(
        "reactor kinetics",
        "control rods",
        "Control rods are neutron-absorbing components inserted into the core to reduce reactivity and power.",
        "They provide direct and rapid control over the neutron population.",
        "By absorbing neutrons, control rods support startup control, power maneuvering, shutdown, and emergency scram action.",
        "control rods move deeper into the active core",
        "more neutrons are intercepted by absorber material",
        "fewer neutrons remain available to induce further fission in the fuel",
        "power falls until the new rod position and reactor feedback reach equilibrium",
        "Incorrect rod positioning can distort the power shape or reduce shutdown margin.",
        "Rod worth limits, insertion procedures, and scram capability are maintained to keep rod control reliable.",
    ),
    spec(
        "reactor kinetics",
        "Doppler feedback",
        "Doppler feedback is the negative reactivity effect produced when hotter fuel broadens resonance absorption peaks in the fuel.",
        "It is an important inherent stabilizing response during power increases.",
        "Broader resonance absorption means more neutrons are captured in fertile material without adding more fission support.",
        "fuel temperature rises during a power increase",
        "resonance absorption bands in the fuel become effectively broader",
        "more neutrons are absorbed without sustaining the fissile chain reaction",
        "reactivity decreases and opposes the original power rise",
        "Weak temperature feedback would leave the reactor less able to damp rapid positive disturbances on its own.",
        "Fuel composition and core design are chosen so Doppler feedback contributes meaningful negative reactivity during transients.",
    ),
    spec(
        "thermal hydraulics",
        "coolant flow",
        "Coolant flow is the movement of reactor coolant that transports heat away from the core.",
        "Continuous flow is required to keep fuel and cladding temperatures within design limits.",
        "The amount of heat removed depends on flow rate, temperature rise, and coolant thermophysical properties.",
        "fission deposits heat inside the fuel pellets",
        "that heat conducts through the cladding into the moving coolant",
        "the coolant carries the thermal energy away from the core region",
        "fuel and cladding temperatures remain controlled only while heat removal matches heat generation",
        "Reduced flow can shrink thermal margin and accelerate cladding temperature rise.",
        "Pump reliability, flow limits, and emergency cooling capability are used to preserve adequate core cooling.",
    ),
    spec(
        "thermal hydraulics",
        "boiling heat transfer",
        "Boiling heat transfer is the transfer of heat from a hot surface to a liquid coolant as vapor bubbles form and depart.",
        "It can remove large amounts of heat because bubble formation enhances convection and carries latent heat away.",
        "The detailed heat-transfer behavior depends on pressure, mass flow, surface condition, and local heat flux.",
        "the cladding surface becomes hot enough to nucleate vapor bubbles in the coolant",
        "bubble growth and departure stir the nearby liquid and strengthen heat removal",
        "the surface can transfer more energy while staying in the nucleate boiling regime",
        "heat transfer remains effective until local conditions move toward a less favorable boiling regime",
        "If boiling shifts out of the stable nucleate regime, surface temperature can rise sharply.",
        "Thermal limits are set so boiling remains in acceptable regimes throughout the approved operating envelope.",
    ),
    spec(
        "thermal hydraulics",
        "critical heat flux",
        "Critical heat flux is the heat-flux limit at which normal nucleate boiling can transition to a much less effective heat-transfer regime.",
        "It is a central thermal limit because cladding temperature can rise rapidly once that limit is exceeded.",
        "Critical heat flux depends on pressure, mass flux, geometry, coolant state, and local power distribution.",
        "heat flux at the fuel surface increases as reactor power rises",
        "local boiling conditions approach the limit where liquid contact with the surface becomes unstable",
        "a vapor-rich layer begins to interfere with effective surface cooling",
        "cladding temperature climbs quickly because heat transfer has degraded",
        "Exceeding critical heat flux threatens cladding integrity and sharply reduces thermal margin.",
        "Core design, peaking-factor limits, and protection systems are used to keep heat flux below this limit with margin.",
    ),
    spec(
        "thermal hydraulics",
        "natural circulation",
        "Natural circulation is coolant flow driven by density differences rather than by mechanical pumping.",
        "It is important because it can provide passive heat removal when forced circulation is reduced or unavailable.",
        "Warmer fluid becomes less dense and rises, while cooler denser fluid descends to complete the circulation loop.",
        "core heating raises coolant temperature and lowers its density",
        "the lighter hot coolant rises while cooler denser fluid moves downward elsewhere in the loop",
        "a buoyancy-driven circulation path is established through the connected system",
        "heat continues moving away from the core even without active pump power",
        "Weak natural circulation may be insufficient if geometry or pressure conditions do not support adequate flow.",
        "Passive safety features and emergency procedures preserve a flow path that can sustain buoyancy-driven cooling when needed.",
    ),
    spec(
        "thermal hydraulics",
        "steam generator heat transfer",
        "Steam generator heat transfer is the process by which heat from primary coolant is transferred to the secondary side to produce steam.",
        "It links reactor heat production to the turbine cycle without mixing the two fluid systems.",
        "Reliable steam generator performance supports both plant efficiency and stable primary-side temperature control.",
        "hot primary coolant leaves the core carrying fission heat",
        "that coolant transfers energy through steam-generator tubes into secondary water",
        "secondary water boils into steam while the primary fluid is cooled",
        "the cooled primary coolant returns to the core and the steam drives the power conversion system",
        "Poor steam-generator performance can weaken heat removal from the primary system and disturb plant thermal balance.",
        "Tube integrity, water chemistry, and heat-balance monitoring are used to preserve stable heat transfer.",
    ),
    spec(
        "thermal hydraulics",
        "hot channel behavior",
        "Hot channel behavior describes the local thermal response of the most limiting fuel channel rather than the core average.",
        "It is important because local peaks, not averages, usually determine the limiting thermal margin.",
        "Analysts track hot channels to protect the region that experiences the highest combined power and temperature burden.",
        "power distribution becomes locally higher in one fuel channel than in neighboring channels",
        "that channel generates more heat than the core average and sees a larger temperature rise",
        "local fuel and cladding temperatures approach thermal limits earlier than the average channel would suggest",
        "the limiting channel determines whether overall operation remains safely inside the allowed envelope",
        "If local peaking is underestimated, a nominally acceptable average condition can still damage the hottest region.",
        "Peaking-factor limits and core design methods are used to protect the most limiting channel with margin.",
    ),
    spec(
        "materials behavior",
        "uranium dioxide thermal conductivity",
        "Uranium dioxide thermal conductivity is the ability of ceramic fuel to conduct heat from the pellet interior toward the surface.",
        "It matters because low conductivity increases the internal temperature gradient across the fuel pellet.",
        "Fuel temperature depends on both power density and how readily heat can move through the ceramic matrix.",
        "fission deposits heat throughout the fuel pellet volume",
        "that heat must conduct outward through a material whose conductivity is limited and can degrade with burnup",
        "a larger temperature difference develops between the pellet center and the surface",
        "centerline temperature rises and the fuel approaches thermal-performance limits sooner",
        "High fuel temperature can accelerate swelling, cracking, and fission-gas release.",
        "Fuel design and operating limits account for conductivity degradation so pellet temperature remains within margin.",
    ),
    spec(
        "materials behavior",
        "fuel swelling",
        "Fuel swelling is the gradual increase in fuel volume caused by fission-product buildup and irradiation-induced microstructural change.",
        "It matters because swelling changes pellet geometry and can influence pellet-cladding interaction.",
        "Swelling becomes more significant as burnup increases and fission products accumulate in the fuel matrix.",
        "fission products accumulate inside the ceramic fuel during irradiation",
        "the fuel microstructure changes and occupied volume gradually increases",
        "pellet geometry expands and the pellet-cladding gap narrows",
        "mechanical loading on the cladding becomes more likely as burnup continues",
        "Excessive swelling can tighten pellet-cladding contact and reduce mechanical margin in the fuel rod.",
        "Fuel-performance models and burnup limits are used to keep swelling behavior within analyzed bounds.",
    ),
    spec(
        "materials behavior",
        "zircaloy oxidation",
        "Zircaloy oxidation is the chemical growth of an oxide layer on zirconium-alloy cladding exposed to high-temperature water or steam.",
        "It matters because oxidation changes cladding thickness, strength, and heat-transfer behavior.",
        "Oxidation rate increases as temperature rises and as the chemical environment becomes more aggressive.",
        "hot zirconium alloy reacts with oxygen-bearing coolant or steam at the cladding surface",
        "an oxide layer forms and continues to grow with time and temperature",
        "the remaining metal wall becomes thinner and the thermal resistance of the surface changes",
        "cladding strength and heat-transfer margin both degrade if oxidation becomes excessive",
        "Rapid oxidation at high temperature is a key threat during severe cooling challenges.",
        "Chemistry control, temperature limits, and accident-mitigation systems are used to constrain oxidation damage.",
    ),
    spec(
        "materials behavior",
        "radiation embrittlement",
        "Radiation embrittlement is the loss of toughness that occurs when neutron irradiation alters the microstructure of reactor metals.",
        "It matters because embrittled metals tolerate thermal and mechanical shocks less effectively.",
        "Pressure-vessel steels are monitored closely because they experience neutron exposure over many years of operation.",
        "fast neutrons displace atoms within the metal lattice",
        "defects and precipitates accumulate in the material microstructure",
        "fracture toughness falls while brittle response becomes more likely",
        "the material has less margin against crack initiation or crack growth during demanding transients",
        "Embrittled structural materials can narrow safety margin during pressurized thermal shock and other severe events.",
        "Surveillance capsules and life-management programs are used to track and manage embrittlement over plant life.",
    ),
    spec(
        "materials behavior",
        "coolant chemistry corrosion",
        "Coolant chemistry corrosion is the material degradation that occurs when water chemistry promotes oxide growth, dissolution, or surface attack on reactor materials.",
        "It matters because corrosion can change wall thickness, surface condition, and heat-transfer behavior.",
        "Water purity, dissolved species, and temperature all influence how aggressively materials degrade in service.",
        "coolant chemistry drifts away from its controlled range",
        "surface reactions at metal boundaries become more aggressive or less protective",
        "oxide growth, dissolution, or corrosion product transport increases",
        "material condition and long-term component reliability degrade over time",
        "Poor chemistry control can accelerate cladding and structural degradation while increasing contamination transport.",
        "Chemistry surveillance and purification systems are used to keep material attack within acceptable limits.",
    ),
    spec(
        "safety systems",
        "loss of coolant accident",
        "A loss of coolant accident is an event in which reactor coolant is lost from the primary system faster than normal makeup can replace it.",
        "It matters because coolant is needed to remove both fission heat and decay heat from the core.",
        "Loss of coolant threatens heat removal even when the chain reaction has already been stopped.",
        "a break or failure allows coolant to escape from the primary boundary",
        "core inventory and pressure decrease while heat generation in the fuel continues",
        "the ability of the system to remove heat from the fuel weakens sharply",
        "fuel and cladding temperatures rise until emergency cooling restores adequate heat removal",
        "If coolant loss is not arrested and core cooling is not restored, fuel damage risk increases rapidly.",
        "Emergency core cooling, reactor trip, and containment functions are designed to manage this accident sequence.",
    ),
    spec(
        "safety systems",
        "decay heat removal",
        "Decay heat removal is the continued removal of heat produced by radioactive decay after the chain reaction stops.",
        "It matters because substantial heat remains in the core even after shutdown.",
        "Post-shutdown cooling must remain reliable until decay heat falls to a much lower level.",
        "fission products continue to decay after the reactor is shut down",
        "that radioactive decay keeps releasing thermal energy inside the fuel",
        "cooling systems must continue carrying that energy away from the core",
        "fuel temperature stays bounded only while residual heat removal remains available",
        "Loss of decay heat removal can lead to overheating even though the reactor is already subcritical.",
        "Residual heat removal systems, emergency procedures, and alternate cooling paths are used to protect this function.",
    ),
    spec(
        "safety systems",
        "emergency core cooling system",
        "The emergency core cooling system is a safety system that injects water or otherwise restores cooling after certain accidents.",
        "It matters because accident conditions can remove the normal cooling path while heat generation continues.",
        "Multiple injection stages are often provided so the system can function across a wide range of pressures and break sizes.",
        "an accident reduces or interrupts normal core cooling",
        "protective logic detects the event and initiates emergency injection or refill actions",
        "water is delivered to the vessel or primary system to restore core cooling",
        "fuel and cladding temperatures stop rising once adequate inventory and heat transfer are re-established",
        "Failure of emergency cooling would greatly increase the chance of severe fuel damage during a coolant-loss event.",
        "Redundancy, diversity, testing, and protected power supplies are used to keep emergency cooling available on demand.",
    ),
    spec(
        "safety systems",
        "containment",
        "Containment is the engineered barrier that surrounds key reactor systems and limits radioactive release during accidents.",
        "It matters because it provides the final physical barrier after fuel and coolant-pressure boundaries.",
        "Containment also provides a controlled volume for managing steam, pressure, and airborne radioactive material during severe transients.",
        "an accident releases steam, hydrogen, or radioactive material from damaged systems inside the plant",
        "the containment structure confines those materials within a robust sealed volume",
        "internal pressure-control features and isolation boundaries limit how much material can escape outward",
        "offsite release potential remains much lower than it would be without a functioning containment barrier",
        "A weak or bypassed containment function can greatly increase environmental release risk even if other systems respond.",
        "Containment integrity monitoring, isolation valves, and pressure-control systems are maintained to preserve this final barrier.",
    ),
    spec(
        "safety systems",
        "station blackout",
        "Station blackout is the loss of offsite power together with the loss or failure of the normal onsite AC power sources.",
        "It matters because many active cooling and control systems depend on reliable electrical power.",
        "Without power, pumps, controls, and much of the instrumentation needed for active heat removal can degrade over time.",
        "offsite power is lost and backup AC power is unavailable",
        "active cooling equipment and many support systems can no longer operate normally",
        "core heat removal becomes progressively harder to sustain as batteries deplete and conditions worsen",
        "passive features or rapid power recovery are then needed to prevent overheating",
        "Extended blackout can progress toward core damage if cooling and instrumentation cannot be maintained long enough.",
        "Backup generators, battery capacity, alternate AC strategies, and passive safety features are used to reduce blackout risk.",
    ),
    spec(
        "safety systems",
        "reactor trip system",
        "The reactor trip system is the protection function that rapidly shuts down the chain reaction when monitored conditions exceed safe limits.",
        "It matters because a fast shutdown response limits the escalation of abnormal transients.",
        "Protective trip logic monitors key reactor parameters and initiates shutdown before unsafe conditions can grow further.",
        "reactor parameters cross a protective trip setpoint during an abnormal event",
        "the protection system issues a shutdown signal without waiting for slower operator action",
        "control and shutdown devices drive the chain reaction toward a subcritical condition",
        "power stops rising and the plant transitions into a safer state for cooling and recovery actions",
        "If the trip system fails to act during a fast transient, the event can escalate before other protections regain control.",
        "Trip logic, surveillance testing, and independent actuation paths are used to keep the shutdown function dependable.",
    ),
]


def ordered_subject_specs() -> Dict[str, List[Dict[str, str]]]:
    """Group subject specifications by canonical concept."""
    grouped: Dict[str, List[Dict[str, str]]] = {concept: [] for concept in CONCEPT_DICTIONARY}
    for entry in SPECS:
        grouped[entry["concept"]].append(entry)
    return grouped


def sentence_starts(text: str) -> List[str]:
    """Extract lightweight sentence-start signatures."""
    starts: List[str] = []
    for sentence in split_into_sentences(text):
        words = re.findall(r"[A-Za-z0-9'-]+", sentence.lower())
        if not words:
            continue
        starts.append(" ".join(words[:2]))
    return starts


def lexical_tokens(text: str) -> List[str]:
    """Extract lightweight lexical tokens for similarity checks."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


def sentence_similarity(left: str, right: str) -> float:
    """Compute normalized lexical overlap between two sentences."""
    left_tokens = set(lexical_tokens(left))
    right_tokens = set(lexical_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def has_repeated_local_phrase(sentences: List[str]) -> bool:
    """Reject samples that repeat the same phrase across neighboring sentences."""
    for index in range(len(sentences) - 1):
        current_tokens = lexical_tokens(sentences[index])
        next_tokens = lexical_tokens(sentences[index + 1])
        current_ngrams = {
            " ".join(current_tokens[pos : pos + 3])
            for pos in range(max(0, len(current_tokens) - 2))
        }
        next_ngrams = {
            " ".join(next_tokens[pos : pos + 3])
            for pos in range(max(0, len(next_tokens) - 2))
        }
        repeated = {phrase for phrase in current_ngrams & next_ngrams if len(phrase) > 8}
        if repeated:
            return True
    return False


def has_self_referential_loop(text: str) -> bool:
    """Reject circular definition patterns and repeated restatements."""
    lowered = text.lower()
    if re.search(r"\b([a-z0-9-]+)\s+is\s+\1\b", lowered):
        return True
    return any(
        sentence_similarity(left, right) > 0.75
        for left, right in zip(split_into_sentences(text), split_into_sentences(text)[1:])
    )


def has_generic_filler(text: str) -> bool:
    """Reject generic phrases that promote repetitive tutoring language."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in GENERIC_FILLER_PHRASES)


def sample_aliases(entry: Dict[str, str]) -> Tuple[str, ...]:
    """Return the query-oriented aliases that should anchor one subject."""
    aliases = QUERY_ALIASES.get(entry["term"], (entry["term"],))
    return tuple(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))


def primary_alias(entry: Dict[str, str]) -> str:
    """Return the main alias used to anchor the first sentence of a sample."""
    return sample_aliases(entry)[0]


def definition_copula_and_fragment(entry: Dict[str, str]) -> Tuple[str, str]:
    """Remove the subject phrase from the raw definition so subject-first variants read cleanly."""
    definition = entry["definition"].strip()
    lowered = definition.lower()
    for candidate in (entry["term"],) + sample_aliases(entry):
        candidate_lower = candidate.lower()
        for copula in (" is ", " are "):
            prefix = candidate_lower + copula
            if lowered.startswith(prefix):
                fragment = definition[len(candidate) + len(copula) :].strip()
                return copula.strip(), fragment
    if " are " in lowered[: min(len(lowered), 40)]:
        return "are", definition
    return "is", definition


def score_causal_strength(text: str) -> float:
    """Score how clearly a sample expresses a cause-and-effect chain."""
    lowered = text.lower()
    markers = sum(lowered.count(marker) for marker in CAUSAL_MARKERS)
    steps = len(split_into_sentences(text))
    directional_terms = sum(lowered.count(word) for word in ("increase", "decrease", "rise", "fall", "reduce", "remove"))
    score = 0.2
    score += min(0.35, markers * 0.08)
    score += min(0.25, max(0, steps - 1) * 0.06)
    score += min(0.2, directional_terms * 0.03)
    return round(min(1.0, score), 4)


def score_causal_quality(text: str, sample_type: str, concept: str) -> float:
    """Score causal progression, mechanism specificity, and absence of filler."""
    lowered = text.lower()
    sentences = split_into_sentences(text)
    state_score = 1.0 if count_keyword_hits(lowered, CONCEPT_CORE_TERMS[concept]) >= 1 else 0.0
    cause_effect_clarity = min(1.0, sum(lowered.count(marker) for marker in MECHANISM_MARKERS) / 3.0)
    temporal_progression = min(1.0, sum(lowered.count(marker) for marker in TEMPORAL_MARKERS) / 3.0)
    mechanism_specificity = min(1.0, count_keyword_hits(lowered, CONCEPT_CORE_TERMS[concept]) / 3.0)
    sentence_progression = 1.0 if len(sentences) >= 3 else 0.6
    generic_penalty = 0.35 if has_generic_filler(text) else 0.0
    base = (
        0.2 * state_score
        + 0.25 * cause_effect_clarity
        + 0.2 * temporal_progression
        + 0.2 * mechanism_specificity
        + 0.15 * sentence_progression
    )
    if sample_type == "mechanism":
        base += 0.1
    if sample_type == "safety_analysis":
        base += 0.05
    return round(max(0.0, min(1.0, base - generic_penalty)), 4)


def count_keyword_hits(text: str, keywords: Tuple[str, ...]) -> int:
    """Count whole-keyword or phrase hits in one text block."""
    lowered = text.lower()
    return sum(1 for keyword in keywords if re.search(r"\b{0}\b".format(re.escape(keyword.lower())), lowered))


def concept_purity_score(text: str, concept: str) -> float:
    """Score how strongly one sample stays inside its declared concept boundary."""
    lowered = text.lower()
    on_topic = count_keyword_hits(lowered, CONCEPT_CORE_TERMS[concept])
    off_topic = 0
    for other_concept, keywords in CONCEPT_CORE_TERMS.items():
        if other_concept == concept:
            continue
        off_topic += count_keyword_hits(lowered, keywords)

    if on_topic == 0 and off_topic == 0:
        return 0.0

    raw_score = (on_topic + 1.0) / (on_topic + off_topic + 1.0)
    if off_topic == 0:
        raw_score += 0.15
    return round(min(1.0, raw_score), 4)


def infer_block_concept_score(text: str, concept: str) -> int:
    """Count concept-keyword hits for single-concept validation."""
    lowered = text.lower()
    return sum(1 for keyword in CONCEPT_DICTIONARY[concept] if keyword in lowered)


def validate_sample(text: str, concept: str, sample_type: str) -> bool:
    """Strict validation for one structured concept sample."""
    normalized = normalize_text(text)
    sentences = split_into_sentences(normalized)
    if sample_type not in CANONICAL_TYPES:
        return False
    if not 2 <= len(sentences) <= 5:
        return False
    lowered = normalized.lower()
    if "q:" in lowered or "a:" in lowered:
        return False
    if any(line.lstrip().startswith(("-", "*")) for line in normalized.splitlines()):
        return False
    starts = sentence_starts(normalized)
    start_counts = Counter(starts)
    if len(starts) > 1 and (len(start_counts) == 1 or any(count >= 3 for count in start_counts.values())):
        return False
    if has_repeated_local_phrase(sentences):
        return False
    if has_self_referential_loop(normalized):
        return False
    if has_generic_filler(normalized):
        return False
    if any(sentence_similarity(left, right) > 0.60 for left, right in zip(sentences, sentences[1:])):
        return False
    if concept_purity_score(normalized, concept) < 0.28:
        return False
    causal_quality = score_causal_quality(normalized, sample_type, concept)
    if sample_type == "mechanism" and causal_quality < 0.74:
        return False
    if sample_type == "safety_analysis" and causal_quality < 0.47:
        return False
    if sample_type == "explanation" and causal_quality < 0.44:
        return False
    if sample_type == "definition" and causal_quality < 0.30:
        return False
    return True


def is_valid_sample(sample: Dict[str, object]) -> bool:
    """Hard concept-boundary guard for one structured sample."""
    topic = str(sample.get("topic", ""))
    sample_type = str(sample.get("category", ""))
    text = normalize_record_text(str(sample.get("text", "")))
    if topic not in CONCEPT_DICTIONARY:
        return False
    return validate_sample(text, topic, sample_type)


def normalize_record_text(text: str) -> str:
    """Normalize a record body for deterministic comparison."""
    normalized = normalize_text(text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_definition_variants(entry: Dict[str, str]) -> List[str]:
    """Create definition samples for one concept subject."""
    alias = primary_alias(entry)
    copula, fragment = definition_copula_and_fragment(entry)
    importance_fragment = entry["importance"].strip()
    if importance_fragment.lower().startswith("it matters because "):
        importance_fragment = importance_fragment[len("It matters because ") :]
    return [
        f"{alias.capitalize()} {copula} {fragment} {entry['importance']} {entry['detail']}",
        f"In reactor analysis, {alias} {copula} {fragment} {entry['detail']}",
        f"{alias.capitalize()} {copula} {fragment} This property matters because {importance_fragment[0].lower() + importance_fragment[1:]}",
        f"Engineers define {alias} as {fragment} {entry['importance']}",
    ]


def build_explanation_variants(entry: Dict[str, str]) -> List[str]:
    """Create explanatory samples for one concept subject."""
    alias = primary_alias(entry)
    return [
        f"{alias.capitalize()} matters because {entry['importance'][0].lower() + entry['importance'][1:]} {entry['detail']}",
        f"In reactor operation, {alias} influences plant behavior because {entry['detail'][0].lower() + entry['detail'][1:]} {entry['importance']}",
        f"{alias.capitalize()} affects reactor behavior because {entry['importance'][0].lower() + entry['importance'][1:]} {entry['control']}",
        f"{alias.capitalize()} becomes important during operation because {entry['detail'][0].lower() + entry['detail'][1:]} {entry['importance']}",
    ]


MECH_SENTENCE_1 = (
    "When {cause}, {lead}.",
    "If {cause}, {lead}.",
    "As soon as {cause}, {lead}.",
    "During operation, when {cause}, {lead}.",
    "Whenever {cause}, {lead}.",
)
MECH_SENTENCE_2 = (
    "This leads to {result}.",
    "That shift causes {result}.",
    "The system then experiences {result}.",
    "The next effect is that {result}.",
    "As the change continues, {result}.",
)
MECH_SENTENCE_3 = (
    "As a result, {consequence}.",
    "Consequently, {consequence}.",
    "In turn, {consequence}.",
    "Therefore, {consequence}.",
)
MECH_SENTENCE_4 = (
    "{importance}",
    "{control}",
    "{safety}",
    "{detail}",
)


def build_mechanism_variants(entry: Dict[str, str]) -> List[str]:
    """Create mechanism-heavy samples with explicit cause-and-effect chains."""
    variants: List[str] = []
    for index, (first, second, third, fourth) in enumerate(product(MECH_SENTENCE_1, MECH_SENTENCE_2, MECH_SENTENCE_3, MECH_SENTENCE_4)):
        if index >= TYPE_TARGETS["mechanism"]:
            break
        variants.append(
            " ".join(
                [
                    first.format(cause=entry["cause"], lead=entry["lead"]),
                    second.format(result=entry["result"]),
                    third.format(consequence=entry["consequence"]),
                    fourth.format(
                        importance=entry["importance"],
                        control=entry["control"],
                        safety=entry["safety"],
                        detail=entry["detail"],
                    ),
                ]
            )
        )
    variants.extend(
        [
            " ".join(
                [
                    f"When {entry['cause']}, {entry['lead']}.",
                    f"This leads to {entry['result']}.",
                    f"As a result, {entry['consequence']}.",
                    entry["control"],
                ]
            ),
            " ".join(
                [
                    f"If {entry['cause']}, {entry['lead']}.",
                    f"The system then experiences {entry['result']}.",
                    f"Consequently, {entry['consequence']}.",
                    entry["importance"],
                ]
            ),
            " ".join(
                [
                    f"Once {entry['cause']}, {entry['lead']}.",
                    f"That change causes {entry['result']}.",
                    f"Therefore, {entry['consequence']}.",
                    entry["safety"],
                ]
            ),
        ]
    )
    return variants


def build_safety_variants(entry: Dict[str, str]) -> List[str]:
    """Create safety-analysis samples for one concept subject."""
    alias = primary_alias(entry)
    return [
        f"If {entry['safety'][0].lower() + entry['safety'][1:]} {entry['control']}",
        f"During abnormal operation, {alias} requires protection because {entry['safety'][0].lower() + entry['safety'][1:]} {entry['control']}",
        f"{alias.capitalize()} demands a clear safety response because {entry['safety'][0].lower() + entry['safety'][1:]} {entry['control']}",
        f"When protective systems work correctly, {entry['control'][0].lower() + entry['control'][1:]} {entry['importance']}",
    ]


def title_case_topic(topic: str) -> str:
    """Render a topic label for the structured training sample."""
    if topic.isupper():
        return topic
    return " ".join(part.capitalize() for part in topic.split())


def build_question(entry: Dict[str, str], sample_type: str) -> str:
    """Create one query-style prompt that matches the structured specialization target."""
    alias = primary_alias(entry)
    if sample_type == "definition":
        return f"What is {alias}?"
    if sample_type == "mechanism":
        return f"How does {alias} affect reactor behavior?"
    if sample_type == "safety_analysis":
        return f"What happens when {alias} becomes a safety concern?"
    return f"Explain {alias} in reactor operation."


def _unique_sentence_candidates(entry: Dict[str, str], body_text: str) -> List[str]:
    """Build a deduplicated sentence pool for Answer / Reasoning / Effect fields."""
    candidates = split_into_sentences(normalize_record_text(body_text))
    candidates.extend(
        [
            normalize_record_text(entry["definition"]),
            normalize_record_text(entry["detail"]),
            normalize_record_text(entry["importance"]),
            normalize_record_text(entry["consequence"]),
            normalize_record_text(entry["control"]),
            normalize_record_text(entry["safety"]),
        ]
    )

    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        for sentence in split_into_sentences(candidate):
            normalized = normalize_record_text(sentence)
            lowered = normalized.lower()
            if not normalized or lowered in seen:
                continue
            seen.add(lowered)
            unique.append(normalized)
    return unique


def build_structured_record(
    entry: Dict[str, str],
    sample_type: str,
    body_text: str,
    source: str = "synthetic",
) -> Dict[str, object]:
    """Convert one raw synthetic paragraph into Stage 3 / Stage 5 structured supervision."""
    sentences = _unique_sentence_candidates(entry, body_text)
    if len(sentences) < 3:
        raise RuntimeError(f"Unable to build structured record for {entry['term']} / {sample_type}.")

    answer = sentences[0]
    reasoning = sentences[1]
    effect_sentences = sentences[2:4]
    effect = normalize_record_text(" ".join(effect_sentences))
    text = normalize_record_text(" ".join([answer, reasoning, effect]))

    return {
        "source": source,
        "topic": entry["concept"],
        "subject": entry["term"],
        "category": sample_type,
        "scenario": SCENARIO_BY_TYPE[sample_type],
        "instruction": INSTRUCTION_TOKEN_BY_TYPE[sample_type],
        "question": build_question(entry, sample_type),
        "answer": answer,
        "reasoning": reasoning,
        "effect": effect,
        "explanation": answer,
        "mechanism": reasoning,
        "text": text,
        "causal_strength": score_causal_strength(text),
    }


def generate_structured_samples() -> List[Dict[str, object]]:
    """Generate deterministic structured synthetic samples across all concepts."""
    records: List[Dict[str, object]] = []
    builders = {
        "definition": build_definition_variants,
        "explanation": build_explanation_variants,
        "mechanism": build_mechanism_variants,
        "safety_analysis": build_safety_variants,
    }
    grouped = ordered_subject_specs()
    per_subject_targets = {
        "definition": 4,
        "explanation": 4,
        "mechanism": 20,
        "safety_analysis": 4,
    }

    for concept in CONCEPT_DICTIONARY:
        for entry in grouped[concept]:
            for sample_type in CANONICAL_TYPES:
                texts = builders[sample_type](entry)[: per_subject_targets[sample_type]]
                for text in texts:
                    records.append(build_structured_record(entry, sample_type, text, source="synthetic"))
    return records


def format_structured_block(record: Dict[str, object]) -> str:
    """Render one record to the on-disk structured block format."""
    return (
        "Concept: {concept}\n"
        "Topic: {topic}\n"
        "Type: {sample_type}\n"
        "Scenario: {scenario}\n"
        "Instruction: {instruction}\n\n"
        "Question:\n"
        "{question}\n\n"
        "Answer:\n"
        "{answer}\n\n"
        "Reasoning:\n"
        "{reasoning}\n\n"
        "Effect:\n"
        "{effect}"
    ).format(
        concept=record["topic"],
        topic=title_case_topic(str(record.get("subject", record["topic"]))),
        sample_type=record["category"],
        scenario=record.get("scenario", SCENARIO_BY_TYPE.get(str(record["category"]), "definition")),
        instruction=record.get("instruction", INSTRUCTION_TOKEN_BY_TYPE.get(str(record["category"]), "[EXPLAIN]")),
        question=record.get("question", build_fallback_question(record)),
        answer=record.get("answer", record.get("text", "")),
        reasoning=record.get("reasoning", record.get("text", "")),
        effect=record.get("effect", record.get("text", "")),
    )


def extract_concept(sample_text: str) -> str:
    """Parse the declared concept tag from a structured training sample."""
    for line in sample_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Concept: "):
            return stripped[len("Concept: ") :].strip()
    raise ValueError("Structured sample is missing a Concept: header.")


def build_fallback_question(record: Dict[str, object]) -> str:
    """Create a safe question for legacy or reconstructed records."""
    subject = str(record.get("subject", record.get("topic", "reactor behavior"))).strip().lower()
    sample_type = str(record.get("category", "explanation"))
    if sample_type == "definition":
        return f"What is {subject}?"
    if sample_type == "mechanism":
        return f"How does {subject} affect reactor behavior?"
    if sample_type == "safety_analysis":
        return f"What happens when {subject} becomes a safety concern?"
    return f"Explain {subject} in reactor operation."


def hydrate_structured_record(record: Dict[str, object]) -> Dict[str, object]:
    """Ensure records parsed from disk have the fields needed for structured training."""
    hydrated = dict(record)
    subject = str(hydrated.get("subject", hydrated.get("topic", ""))).strip() or str(hydrated["topic"])
    hydrated["subject"] = subject
    hydrated["scenario"] = str(
        hydrated.get("scenario", SCENARIO_BY_TYPE.get(str(hydrated["category"]), "definition"))
    )
    hydrated["instruction"] = str(
        hydrated.get("instruction", INSTRUCTION_TOKEN_BY_TYPE.get(str(hydrated["category"]), "[EXPLAIN]"))
    )

    base_sentences = _unique_sentence_candidates(
        {
            "definition": str(hydrated.get("answer") or hydrated.get("text", "")),
            "detail": str(hydrated.get("reasoning") or hydrated.get("text", "")),
            "importance": str(hydrated.get("effect") or hydrated.get("text", "")),
            "consequence": str(hydrated.get("effect") or hydrated.get("text", "")),
            "control": str(hydrated.get("effect") or hydrated.get("text", "")),
            "safety": str(hydrated.get("effect") or hydrated.get("text", "")),
        },
        str(hydrated.get("text", "")),
    )
    if len(base_sentences) < 3:
        raise RuntimeError(f"Unable to hydrate structured record for {hydrated['topic']} / {hydrated['category']}.")

    hydrated["question"] = str(hydrated.get("question") or build_fallback_question(hydrated))
    hydrated["answer"] = str(hydrated.get("answer") or base_sentences[0])
    hydrated["reasoning"] = str(hydrated.get("reasoning") or base_sentences[1])
    hydrated["effect"] = str(
        hydrated.get("effect") or normalize_record_text(" ".join(base_sentences[2:4]))
    )
    hydrated["explanation"] = str(hydrated.get("explanation") or hydrated["answer"])
    hydrated["mechanism"] = str(hydrated.get("mechanism") or hydrated["reasoning"])
    hydrated["text"] = normalize_record_text(
        " ".join([hydrated["answer"], hydrated["reasoning"], hydrated["effect"]])
    )
    hydrated["causal_strength"] = float(
        hydrated.get("causal_strength", score_causal_strength(str(hydrated["text"])))
    )
    hydrated["training_text"] = format_structured_block(hydrated)
    return hydrated


def write_synthetic_concept_dataset(path: Path = SYNTHETIC_DATASET_PATH) -> Path:
    """Write the structured synthetic dataset to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [format_structured_block(record) for record in generate_structured_samples()]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return path


def parse_structured_dataset(path: Path = SYNTHETIC_DATASET_PATH) -> List[Dict[str, object]]:
    """Parse the block-formatted synthetic concept dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Structured synthetic dataset not found at {path}.")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    blocks = re.split(r"\n\s*\n(?=Concept:\s)", text)
    records: List[Dict[str, object]] = []
    for block in blocks:
        raw_lines = [line.rstrip() for line in block.splitlines()]
        lines = [line for line in raw_lines if line.strip()]
        if len(lines) < 3:
            continue
        if not lines[0].startswith("Concept: "):
            continue
        fields: Dict[str, str] = {}
        current_key = ""
        buffer: List[str] = []

        def flush_buffer():
            nonlocal buffer
            if current_key:
                joined = " ".join(part for part in buffer if part.strip()).strip()
                if current_key in {"concept", "subject", "category", "scenario", "instruction"}:
                    fields[current_key] = joined
                else:
                    fields[current_key] = normalize_record_text(joined)
            buffer = []

        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Concept: "):
                flush_buffer()
                current_key = "concept"
                buffer = [stripped[len("Concept: ") :].strip()]
            elif stripped.startswith("Topic: "):
                flush_buffer()
                current_key = "subject"
                buffer = [stripped[len("Topic: ") :].strip()]
            elif stripped.startswith("Type: "):
                flush_buffer()
                current_key = "category"
                buffer = [stripped[len("Type: ") :].strip()]
            elif stripped.startswith("Scenario: "):
                flush_buffer()
                current_key = "scenario"
                buffer = [stripped[len("Scenario: ") :].strip()]
            elif stripped.startswith("Instruction: "):
                flush_buffer()
                current_key = "instruction"
                buffer = [stripped[len("Instruction: ") :].strip()]
            elif stripped == "Question:":
                flush_buffer()
                current_key = "question"
            elif stripped == "Answer:":
                flush_buffer()
                current_key = "answer"
            elif stripped == "Reasoning:":
                flush_buffer()
                current_key = "reasoning"
            elif stripped == "Effect:":
                flush_buffer()
                current_key = "effect"
            else:
                buffer.append(stripped)
        flush_buffer()

        concept = fields.get("concept", "").strip()
        sample_type = fields.get("category", "").strip()
        if not concept or not sample_type:
            continue

        legacy_body = normalize_record_text(" ".join(lines[2:])) if "answer" not in fields else ""
        record = {
            "source": "synthetic",
            "topic": concept,
            "subject": fields.get("subject", concept),
            "category": sample_type,
            "scenario": fields.get("scenario", SCENARIO_BY_TYPE.get(sample_type, "definition")),
            "instruction": fields.get("instruction", INSTRUCTION_TOKEN_BY_TYPE.get(sample_type, "[EXPLAIN]")),
            "question": fields.get("question", ""),
            "answer": fields.get("answer", ""),
            "reasoning": fields.get("reasoning", ""),
            "effect": fields.get("effect", ""),
            "text": normalize_record_text(
                " ".join(part for part in (fields.get("answer"), fields.get("reasoning"), fields.get("effect"), legacy_body) if part)
            ),
        }
        records.append(hydrate_structured_record(record))
    return records


def deduplicate_records(records: Iterable[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    """Remove exact and near-duplicate records deterministically."""
    deduped: List[Dict[str, object]] = []
    duplicate_count = 0
    by_bucket: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)

    def subject_signature(text: str) -> str:
        sentences = split_into_sentences(text)
        if not sentences:
            return ""
        tokens = lexical_tokens(sentences[0])
        return " ".join(tokens[:6])

    for record in records:
        category = str(record["category"])
        bucket_key = (
            str(record["topic"]),
            category,
            subject_signature(str(record["text"])),
        )
        text_key = normalize_record_text(str(record["text"])).lower()
        if any(normalize_record_text(str(existing["text"])).lower() == text_key for existing in by_bucket[bucket_key]):
            duplicate_count += 1
            continue
        similarity_threshold = 0.98 if category == "mechanism" else 0.88
        if any(sentence_similarity(str(record["text"]), str(existing["text"])) > similarity_threshold for existing in by_bucket[bucket_key]):
            duplicate_count += 1
            continue
        deduped.append(record)
        by_bucket[bucket_key].append(record)
    return deduped, duplicate_count


def validate_and_rank_records(records: Iterable[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    """Apply strict validation and keep records sorted by causal strength and stability."""
    validated: List[Dict[str, object]] = []
    rejected = 0
    for record in records:
        hydrated = hydrate_structured_record(record)
        topic = str(hydrated["topic"])
        category = str(hydrated["category"])
        text = normalize_record_text(str(hydrated["text"]))
        if topic not in CONCEPT_DICTIONARY or category not in CANONICAL_TYPES:
            rejected += 1
            continue
        if not validate_sample(text, topic, category):
            rejected += 1
            continue
        validated.append(
            {
                "source": str(hydrated.get("source", "synthetic")),
                "topic": topic,
                "subject": str(hydrated.get("subject", topic)),
                "category": category,
                "scenario": str(hydrated.get("scenario", SCENARIO_BY_TYPE.get(category, "definition"))),
                "instruction": str(hydrated.get("instruction", INSTRUCTION_TOKEN_BY_TYPE.get(category, "[EXPLAIN]"))),
                "question": str(hydrated["question"]),
                "answer": str(hydrated["answer"]),
                "reasoning": str(hydrated["reasoning"]),
                "effect": str(hydrated["effect"]),
                "explanation": str(hydrated.get("explanation", hydrated["answer"])),
                "mechanism": str(hydrated.get("mechanism", hydrated["reasoning"])),
                "text": text,
                "training_text": format_structured_block(hydrated),
                "concept_purity": concept_purity_score(text, topic),
                "causal_strength": score_causal_strength(text),
                "causal_quality": score_causal_quality(text, category, topic),
            }
        )

    deduped, duplicate_count = deduplicate_records(validated)
    ranked = sorted(
        deduped,
        key=lambda record: (
            str(record["topic"]),
            str(record["category"]),
            source_priority(str(record["source"])),
            -float(record["concept_purity"]),
            -float(record["causal_quality"]),
            -float(record["causal_strength"]),
            len(str(record["text"])),
            str(record["text"]).lower(),
        ),
    )
    return ranked, duplicate_count + rejected


def select_balanced_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Select a balanced, mechanism-heavy dataset deterministically."""
    buckets: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        concept: {sample_type: [] for sample_type in CANONICAL_TYPES}
        for concept in CONCEPT_DICTIONARY
    }
    for record in records:
        buckets[str(record["topic"])][str(record["category"])].append(record)

    selected: List[Dict[str, object]] = []
    for concept in CONCEPT_DICTIONARY:
        for sample_type in CANONICAL_TYPES:
            required = TYPE_TARGETS[sample_type]
            available = buckets[concept][sample_type]
            if len(available) < required:
                raise RuntimeError(
                    f"Dataset selection failed for {concept}/{sample_type}: {len(available)} available, {required} required."
                )
            selected.extend(available[:required])

    ordered: List[Dict[str, object]] = []
    max_bucket = max(TYPE_TARGETS.values())
    by_bucket: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        concept: {sample_type: [] for sample_type in CANONICAL_TYPES}
        for concept in CONCEPT_DICTIONARY
    }
    for record in selected:
        by_bucket[str(record["topic"])][str(record["category"])].append(record)

    for index in range(max_bucket):
        for concept in CONCEPT_DICTIONARY:
            for sample_type in CANONICAL_TYPES:
                bucket = by_bucket[concept][sample_type]
                if index < len(bucket):
                    ordered.append(bucket[index])
    return ordered


def select_locked_compatible_records(
    ranked_records: List[Dict[str, object]],
    raw_records: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Build a full balanced set by preferring ranked records and backfilling deterministically from raw data."""
    selected_by_bucket: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    seen_texts: Dict[Tuple[str, str], set[str]] = defaultdict(set)

    def maybe_add(record: Dict[str, object]) -> None:
        bucket = (str(record["topic"]), str(record["category"]))
        required = TYPE_TARGETS[str(record["category"])]
        text_key = normalize_record_text(str(record["text"])).lower()
        if len(selected_by_bucket[bucket]) >= required:
            return
        if text_key in seen_texts[bucket]:
            return
        selected_by_bucket[bucket].append(
            {
                "source": str(record.get("source", "synthetic")),
                "topic": str(record["topic"]),
                "subject": str(record.get("subject", record["topic"])),
                "category": str(record["category"]),
                "scenario": str(record.get("scenario", SCENARIO_BY_TYPE.get(str(record["category"]), "definition"))),
                "instruction": str(record.get("instruction", INSTRUCTION_TOKEN_BY_TYPE.get(str(record["category"]), "[EXPLAIN]"))),
                "question": str(record.get("question") or build_fallback_question(record)),
                "answer": str(record.get("answer") or record["text"]),
                "reasoning": str(record.get("reasoning") or record["text"]),
                "effect": str(record.get("effect") or record["text"]),
                "text": normalize_record_text(str(record["text"])),
                "causal_strength": float(record.get("causal_strength", score_causal_strength(str(record["text"])))),
            }
        )
        seen_texts[bucket].add(text_key)

    for record in ranked_records:
        maybe_add(record)

    for record in raw_records:
        maybe_add(record)

    ordered: List[Dict[str, object]] = []
    max_bucket = max(TYPE_TARGETS.values())
    for index in range(max_bucket):
        for concept in CONCEPT_DICTIONARY:
            for sample_type in CANONICAL_TYPES:
                bucket = selected_by_bucket[(concept, sample_type)]
                if index < len(bucket):
                    ordered.append(bucket[index])

    for concept in CONCEPT_DICTIONARY:
        for sample_type in CANONICAL_TYPES:
            bucket = selected_by_bucket[(concept, sample_type)]
            required = TYPE_TARGETS[sample_type]
            if len(bucket) < required:
                raise RuntimeError(
                    f"Locked-compatible selection failed for {concept}/{sample_type}: {len(bucket)} available, {required} required."
                )

    return ordered


def concept_distribution(records: List[Dict[str, object]]) -> Dict[str, int]:
    """Count records by concept."""
    return dict(Counter(str(record["topic"]) for record in records))


def type_distribution(records: List[Dict[str, object]]) -> Dict[str, int]:
    """Count records by type."""
    return dict(Counter(str(record["category"]) for record in records))


def source_breakdown(records: List[Dict[str, object]]) -> Dict[str, int]:
    """Count records by source."""
    return dict(Counter(str(record["source"]) for record in records))


@execution_guard("build_phase3_dataset", GRAPH_NODE)
def build_phase3_dataset() -> Dict[str, object]:
    """Build the deterministic structured concept dataset used by training."""
    json_builder_records = build_json_builder_candidates()
    use_synthetic_backfill = synthetic_backfill_enabled()
    if not json_builder_records and not use_synthetic_backfill:
        raise RuntimeError(
            "No JSON builder records were found. Generate data in '/Users/VIP/Documents/New project/json builder/data' "
            "or set {0}=1 to allow synthetic backfill.".format(config.ENABLE_SYNTHETIC_BACKFILL_ENV)
        )

    synthetic_records = generate_structured_samples() if use_synthetic_backfill else []
    candidate_records = json_builder_records + synthetic_records
    if not candidate_records:
        raise RuntimeError("No dataset candidates were available for build_phase3_dataset.")
    ranked_records, duplicate_count = validate_and_rank_records(candidate_records)
    if use_synthetic_backfill:
        try:
            records = select_balanced_records(ranked_records)
        except RuntimeError:
            records = select_locked_compatible_records(ranked_records, candidate_records)
    else:
        records = ranked_records
        if not records:
            raise RuntimeError("All JSON builder candidates were rejected during validation.")

    records = [hydrate_structured_record(record) for record in records]
    formatted_records = [record["training_text"] for record in records]
    text = "\n\n".join(formatted_records)
    stoi, itos = build_vocab(text)
    concept_groups: Dict[str, List[str]] = {concept: [] for concept in CONCEPT_DICTIONARY}
    for record in records:
        concept_groups[str(record["topic"])].append(str(record["training_text"]))

    package = {
        "records": records,
        "total_records": len(records),
        "training_texts": formatted_records,
        "text": text,
        "stoi": stoi,
        "itos": itos,
        "concept_texts": {concept: "\n\n".join(samples) for concept, samples in concept_groups.items()},
        "concept_labels": [extract_concept(sample_text) for sample_text in formatted_records],
        "concept_dictionary": list(CONCEPT_DICTIONARY.keys()),
        "source_breakdown": source_breakdown(records),
        "synthetic_stats": type_distribution([record for record in records if record["source"] == "synthetic"]),
        "concept_distribution": concept_distribution(records),
        "topic_distribution": concept_distribution(records),
        "type_distribution": type_distribution(records),
        "duplicate_count": duplicate_count,
    }

    print("total_records:", len(records))
    print("json_builder_candidates:", len(json_builder_records))
    print("synthetic_candidates:", len(synthetic_records))
    print("synthetic_backfill_enabled:", use_synthetic_backfill)
    print("concept_distribution:", package["concept_distribution"])
    print("type_distribution:", package["type_distribution"])
    print("duplicate_count:", package["duplicate_count"])
    print("vocab_size:", len(stoi))

    manifest = build_version_manifest(package, block_size=config.block_size)
    package["artifact_manifest"] = manifest

    regenerate_manifest = os.environ.get("NUCLEAR_LLM_REGENERATE_MANIFEST") == "1"
    if config.VERSION_PATH.exists():
        try:
            locked_manifest = load_version_manifest()
            if locked_manifest == manifest:
                print("artifact_manifest_status: locked")
            elif regenerate_manifest:
                write_version_manifest(manifest)
                print("artifact_manifest_status: regenerated")
                print("artifact_manifest_regenerated:", config.VERSION_PATH)
            else:
                print("artifact_manifest_status: mismatch")
        except Exception as exc:
            if regenerate_manifest:
                write_version_manifest(manifest)
                print("artifact_manifest_status: regenerated")
                print("artifact_manifest_regenerated:", config.VERSION_PATH)
            else:
                print("artifact_manifest_status: invalid ({0})".format(exc))
    elif regenerate_manifest:
        write_version_manifest(manifest)
        print("artifact_manifest_status: created")
        print("artifact_manifest_regenerated:", config.VERSION_PATH)
    else:
        print("artifact_manifest_status: missing")

    return package


def build_version_manifest(dataset_package: Dict[str, object], block_size: int | None = None) -> Dict[str, object]:
    """Create the version manifest that locks dataset, tokenizer, and checkpoint compatibility."""
    state = os.environ.get(config.EXECUTION_STATE_ENV)
    if state == "BUILD":
        assert_execution_allowed("build_version_manifest", GRAPH_NODE)
    elif state != "FREEZE":
        raise RuntimeError(f"EXECUTION GRAPH VIOLATION: FUNCTION build_version_manifest NOT ALLOWED IN STATE {state}")
    text = str(dataset_package["text"])
    stoi = dataset_package["stoi"]
    block = int(block_size if block_size is not None else config.block_size)
    manifest = build_artifact_manifest(
        text=text,
        stoi=stoi,
        block_size=block,
        model_version="phase5-semantic-transformer-v1",
        concept_prefix_enabled=True,
    )
    manifest["tokenizer_version"] = f"immutable-tokenizer-{manifest['tokenizer_hash'][:12]}"
    manifest["dataset_version"] = f"immutable-corpus-{manifest['dataset_hash'][:12]}"
    return manifest


def token_distribution(text: str, limit: int = 20) -> Dict[str, int]:
    """Return the top tokens in the current corpus."""
    previous_flag = os.environ.get(config.ALLOW_VOCAB_BUILD_ENV)
    os.environ[config.ALLOW_VOCAB_BUILD_ENV] = "1"
    try:
        vocab_tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]", text)
    finally:
        if previous_flag is None:
            os.environ.pop(config.ALLOW_VOCAB_BUILD_ENV, None)
        else:
            os.environ[config.ALLOW_VOCAB_BUILD_ENV] = previous_flag
    return dict(Counter(vocab_tokens).most_common(limit))


def freeze_dataset_artifacts(dataset_package: Dict[str, object]) -> Dict[str, object]:
    """Write the locked dataset/tokenizer artifacts for Phase 5."""
    assert_execution_allowed("freeze_dataset_artifacts")
    manifest = build_version_manifest(dataset_package, block_size=config.block_size)
    artifact_info = write_locked_artifacts(
        text=str(dataset_package["text"]),
        stoi=dataset_package["stoi"],
        itos=dataset_package["itos"],
        records=dataset_package["records"],
        manifest=manifest,
    )
    dataset_package["artifact_manifest"] = manifest
    dataset_package["manifest_id"] = artifact_info["manifest_id"]
    return {
        "manifest": manifest,
        "artifact_info": artifact_info,
        "token_distribution": token_distribution(str(dataset_package["text"])),
    }


def write_version_manifest(manifest: Dict[str, object]) -> Dict[str, object]:
    """Write the version manifest to version.json."""
    return save_artifact_manifest(manifest, str(config.VERSION_PATH))


def load_version_manifest() -> Dict[str, object]:
    """Load version.json from disk."""
    return load_artifact_manifest(str(config.VERSION_PATH))


def sentence_length_distribution(records: List[Dict[str, object]]) -> Dict[str, int]:
    """Summarize samples by sentence count."""
    buckets = {"2_sentences": 0, "3_sentences": 0, "4_sentences": 0, "5_sentences": 0, "other": 0}
    for record in records:
        count = len(split_into_sentences(str(record["text"])))
        key = f"{count}_sentences"
        if key in buckets:
            buckets[key] += 1
        else:
            buckets["other"] += 1
    return buckets


if __name__ == "__main__":
    assert_side_execution_forbidden()
