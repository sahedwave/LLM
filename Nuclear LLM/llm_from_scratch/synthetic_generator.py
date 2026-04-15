"""Deterministic synthetic nuclear corpus generator for balanced concept training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.execution_graph import assert_side_execution_forbidden


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "data" / "synthetic_concept_dataset.jsonl"
TYPE_COUNTS = {
    "definition": 5,
    "explanation": 5,
    "mechanism": 7,
    "safety_analysis": 3,
}
TYPE_ORDER = ("definition", "explanation", "mechanism", "safety_analysis")


def make_spec(
    concept: str,
    subject: str,
    definition: str,
    importance: str,
    explanation: str,
    mechanism: List[str],
    safety: str,
    control: str,
) -> Dict[str, object]:
    """Build one deterministic synthetic subject specification."""
    return {
        "concept": concept,
        "subject": subject,
        "definition": definition,
        "importance": importance,
        "explanation": explanation,
        "mechanism": mechanism,
        "safety": safety,
        "control": control,
    }


SPECS = [
    make_spec(
        "neutron physics",
        "neutron flux",
        "Neutron flux is the rate at which neutrons pass through a unit area per unit time.",
        "Reaction rates rise when neutron flux rises because more neutrons encounter fuel nuclei in the core.",
        "Operators monitor neutron flux because it links the neutron population directly to reactor power distribution.",
        [
            "A positive reactivity change increases the neutron population in the core.",
            "The larger neutron population raises neutron flux throughout the affected region.",
            "Higher flux causes more fission events per second in fissile material.",
            "Power and fuel temperature then rise until feedback or control action restores balance.",
        ],
        "Large local flux peaks can create local power peaks and reduce thermal margin in nearby fuel rods.",
        "Flux detectors, control rods, and core design limits are used to keep neutron flux within safe operating bounds.",
    ),
    make_spec(
        "neutron physics",
        "moderation",
        "Moderation is the process of slowing fast neutrons to lower energies through repeated scattering collisions.",
        "Thermal reactors depend on moderation because slower neutrons are more likely to induce fission in uranium-235.",
        "The effectiveness of moderation depends strongly on the scattering properties and density of the moderator material.",
        [
            "Fission releases neutrons at high kinetic energy.",
            "Repeated scattering collisions with light nuclei remove part of that energy from the neutrons.",
            "As the neutrons slow down, their probability of causing thermal fission in uranium-235 increases.",
            "The chain reaction then becomes easier to sustain at the intended neutron spectrum.",
        ],
        "Poor moderation can shift the neutron spectrum away from the design condition and reduce neutron economy.",
        "Moderator purity, density control, and temperature feedback are managed so the slowing-down process remains predictable.",
    ),
    make_spec(
        "neutron physics",
        "elastic scattering",
        "Elastic scattering is a neutron collision in which kinetic energy is redistributed without changing the internal state of the target nucleus.",
        "Elastic scattering is important because it is the main mechanism that slows neutrons in many reactor moderators.",
        "Light nuclei remove more neutron energy per collision, which is why water and graphite are effective moderators.",
        [
            "A moving neutron strikes a nucleus in the moderator.",
            "Part of the neutron kinetic energy is transferred to the nucleus during the collision.",
            "The neutron leaves the collision with lower energy but remains available for later interactions.",
            "Repeated elastic scattering events gradually shift the neutron population toward thermal energies.",
        ],
        "If scattering behavior is not well controlled, the reactor can develop a neutron spectrum that differs from the intended design basis.",
        "Moderator selection and neutron-transport calculations are used to keep scattering behavior consistent with the planned core response.",
    ),
    make_spec(
        "neutron physics",
        "resonance absorption",
        "Resonance absorption is the strong neutron absorption that occurs at specific neutron energies in certain nuclides.",
        "Resonance absorption matters because it removes neutrons during slowing down and directly affects neutron economy.",
        "Uranium-238 is a major contributor to resonance absorption in thermal reactor analysis.",
        [
            "Fast neutrons lose energy as they scatter in the moderator and fuel.",
            "While passing through resonance energies, some neutrons encounter sharply higher absorption probabilities in certain nuclides.",
            "Those neutrons are captured before reaching lower energies where they might otherwise support the chain reaction.",
            "The number of neutrons available for later fission or moderation therefore decreases.",
        ],
        "Excess resonance absorption can lower reactivity and reduce the margin available to sustain the intended power distribution.",
        "Fuel geometry, moderator conditions, and lattice design are chosen to manage resonance losses and preserve neutron economy.",
    ),
    make_spec(
        "neutron physics",
        "diffusion length",
        "Diffusion length is the characteristic distance a thermal neutron travels before it is absorbed in a material.",
        "Diffusion length is important because it helps determine leakage behavior in finite reactor systems.",
        "Materials with strong scattering and weak absorption tend to produce longer neutron migration distances.",
        [
            "A thermal neutron undergoes many scattering events while moving through a material.",
            "Each scattering event changes its direction and extends the path it travels before absorption.",
            "If the average travel distance becomes large compared with core dimensions, the chance of leakage increases.",
            "Leakage then reduces the neutron population available to sustain the chain reaction.",
        ],
        "Excessive neutron leakage can reduce reactivity and distort the intended power shape near core boundaries.",
        "Reflectors and core dimensions are selected so neutron migration remains compatible with criticality and power-shaping goals.",
    ),
    make_spec(
        "neutron physics",
        "macroscopic cross-section",
        "The macroscopic cross-section is the probability per unit path length that a neutron will interact in a material.",
        "It is important because it combines microscopic interaction probability with atom density into a quantity used directly in reactor calculations.",
        "Attenuation, mean free path, and reaction rate predictions all depend on the macroscopic cross-section.",
        [
            "Microscopic cross-section describes the interaction behavior of a single nucleus.",
            "Atom density determines how many nuclei are available in a unit volume of material.",
            "Multiplying the two produces the macroscopic interaction probability seen by a neutron traveling through the material.",
            "Reactor analysts then use that probability to predict absorption, scattering, and fission behavior.",
        ],
        "If macroscopic cross-sections are mischaracterized, predicted neutron behavior and core margins can be significantly distorted.",
        "Accurate material composition data and validated nuclear data libraries are used to keep these interaction estimates reliable.",
    ),
    make_spec(
        "neutron physics",
        "thermal utilization",
        "Thermal utilization is the fraction of thermal neutrons absorbed in the fuel rather than in nonfuel materials.",
        "Thermal utilization is important because only neutrons absorbed in the fuel can directly support thermal fission in a thermal reactor.",
        "Absorption in control materials, structural metals, or poisons lowers the number of neutrons available for useful fission.",
        [
            "Thermal neutrons move through fuel, moderator, coolant, and structural materials in the core.",
            "Some of those neutrons are absorbed in nonfuel materials before reaching fissile fuel nuclei.",
            "Every nonfuel absorption removes a neutron that could otherwise have contributed to the chain reaction.",
            "The thermal utilization factor therefore drops when parasitic absorption rises.",
        ],
        "Poor thermal utilization weakens neutron economy and can increase the control burden needed to maintain power.",
        "Core layout and material selection are designed to reduce unnecessary thermal-neutron absorption outside the fuel.",
    ),
    make_spec(
        "neutron physics",
        "neutron leakage",
        "Neutron leakage is the loss of neutrons from the active core before they can cause another useful interaction.",
        "Leakage is important because escaped neutrons do not contribute to sustaining the chain reaction.",
        "Core size, geometry, and reflector design all influence how much neutron leakage occurs.",
        [
            "Neutrons born near the edge of the core can travel outward while scattering through the material.",
            "If their path reaches the boundary before another useful interaction occurs, the neutrons leave the active region.",
            "Those leaked neutrons no longer contribute to absorption or fission inside the core.",
            "The effective multiplication of the reactor therefore decreases.",
        ],
        "High leakage can produce lower-than-expected reactivity and sharper power gradients near the core boundary.",
        "Reflectors, fuel loading patterns, and core dimensions are used to keep leakage within acceptable design limits.",
    ),
    make_spec(
        "reactor kinetics",
        "k-effective",
        "k-effective is the ratio of neutrons in one generation to neutrons in the previous generation.",
        "It is important because it indicates whether the chain reaction is subcritical, critical, or supercritical.",
        "A reactor with k-effective equal to one has a steady neutron population and stable power level.",
        [
            "If k-effective rises above one, each neutron generation produces more neutrons than the last.",
            "The neutron population then grows from generation to generation.",
            "Growing neutron population increases fission rate and reactor power.",
            "Negative feedback or control action must reduce reactivity to return the reactor toward critical conditions.",
        ],
        "Sustained deviation of k-effective above or below one drives unwanted power change and can erode operating margin.",
        "Control rods, soluble boron, and temperature feedback are used to keep k-effective near its intended value.",
    ),
    make_spec(
        "reactor kinetics",
        "reactivity",
        "Reactivity is a measure of how far a reactor is from exact criticality.",
        "Reactivity is important because small changes in it can produce measurable changes in neutron population and power.",
        "Positive reactivity pushes power upward, while negative reactivity pushes power downward.",
        [
            "A control action or physical change alters neutron production or neutron loss in the core.",
            "That change shifts the reactor away from exact criticality and produces a reactivity insertion.",
            "The neutron population responds by increasing or decreasing according to the sign of the insertion.",
            "Power then changes until feedback effects or operator action restore a balanced condition.",
        ],
        "Uncontrolled positive reactivity can cause rapid power rise, while excessive negative reactivity can challenge power stability and restart planning.",
        "Reactivity management is controlled through absorbers, moderator conditions, and carefully defined operating procedures.",
    ),
    make_spec(
        "reactor kinetics",
        "delayed neutrons",
        "Delayed neutrons are neutrons emitted by certain fission products after the original fission event rather than immediately.",
        "Delayed neutrons are important because they slow the effective time scale of reactor power change.",
        "That slower response makes controlled power operation possible with mechanical systems and human supervision.",
        [
            "Most fission neutrons appear essentially immediately after fission.",
            "A small fraction is emitted later when specific fission products undergo radioactive decay.",
            "Those delayed neutrons extend the time between successive effective neutron generations.",
            "The reactor then responds slowly enough for control rods and operators to manage power change safely.",
        ],
        "If the reactor enters prompt-dominated behavior, power can rise far faster than normal control systems are intended to handle.",
        "Design and operation are structured to keep the reactor within the delayed-neutron regime during normal maneuvering.",
    ),
    make_spec(
        "reactor kinetics",
        "control rods",
        "Control rods are neutron-absorbing components inserted into the core to reduce reactivity and power.",
        "They are important because they provide direct and rapid control over the neutron population.",
        "By absorbing neutrons, control rods help regulate startup, power maneuvers, shutdown, and emergency scram action.",
        [
            "When control rods move deeper into the core, they intercept more neutrons in the active region.",
            "Those absorbed neutrons are no longer available to cause fission in the fuel.",
            "The neutron population then decreases and the fission rate falls.",
            "Reactor power drops until the new rod position and reactor feedback reach equilibrium.",
        ],
        "Incorrect rod position can distort the power shape or leave insufficient shutdown margin during abnormal conditions.",
        "Rod worth limits, insertion procedures, and scram capability are used to keep rod control within safe bounds.",
    ),
    make_spec(
        "reactor kinetics",
        "doppler feedback",
        "Doppler feedback is the negative reactivity effect produced when rising fuel temperature broadens resonance absorption peaks in the fuel.",
        "It is important because it provides an inherent stabilizing response during power increases.",
        "Hotter fuel absorbs more neutrons in resonance ranges, which reduces the neutrons available for further fission support.",
        [
            "A power increase raises the temperature of the fuel pellets.",
            "Higher fuel temperature broadens the resonance absorption range of fertile material such as uranium-238.",
            "More neutrons are then absorbed without producing additional fission in the fissile fuel.",
            "The reactivity of the core decreases and opposes the original power rise.",
        ],
        "Weak temperature feedback would leave the reactor less able to counter rapid reactivity disturbances on its own.",
        "Fuel composition and reactor design are chosen so Doppler feedback contributes meaningful negative reactivity during transients.",
    ),
    make_spec(
        "reactor kinetics",
        "xenon-135 poisoning",
        "Xenon-135 poisoning is the reduction in reactivity caused by the buildup of xenon-135, a very strong neutron absorber.",
        "It is important because xenon concentration changes noticeably after power shifts and shutdown, affecting reactor maneuverability.",
        "Xenon transients can temporarily limit restart capability or change the control demand in the core.",
        [
            "Fission produces iodine-135 and xenon-135 among its fission products.",
            "After a power change or shutdown, iodine continues decaying into xenon while neutron burnup of xenon may decrease.",
            "The xenon concentration can therefore rise for a period even though the reactor power has changed.",
            "The added neutron absorption inserts negative reactivity and alters the power-control balance.",
        ],
        "Strong xenon transients can complicate restarts and, if mismanaged, may contribute to uneven power distributions.",
        "Operating procedures account for iodine and xenon behavior so control actions remain consistent with predicted poison transients.",
    ),
    make_spec(
        "reactor kinetics",
        "prompt criticality",
        "Prompt criticality is the condition in which the chain reaction can be sustained by prompt neutrons alone.",
        "It is important because it represents a very fast reactor response regime with sharply reduced control time.",
        "A reactor near prompt criticality can change power on the prompt-neutron time scale rather than the delayed-neutron time scale.",
        [
            "A sufficiently large positive reactivity insertion reduces the need for delayed neutrons to sustain the chain reaction.",
            "The neutron population then grows according to prompt-neutron dynamics instead of delayed-neutron dynamics.",
            "Power rises extremely rapidly because each prompt generation is very short.",
            "Normal operator response time becomes too slow unless the reactor protection system scrams the core immediately.",
        ],
        "Prompt-critical behavior can produce damaging power excursions before heat removal systems have time to respond.",
        "Reactor protection systems, negative feedback, and reactivity limits are designed to prevent this condition from being reached.",
    ),
    make_spec(
        "reactor kinetics",
        "reactor period",
        "The reactor period is the time required for reactor power to change by a factor of e.",
        "It is important because it expresses how quickly the neutron population and power are changing.",
        "Short reactor periods indicate fast power changes and demand prompt attention from control and protection systems.",
        [
            "A reactivity insertion changes the balance between neutron production and neutron loss.",
            "That change sets the exponential rate at which neutron population and power evolve.",
            "The reactor period becomes shorter as the positive reactivity insertion becomes larger.",
            "Control systems then use that information to judge whether the power change is acceptably slow or dangerously fast.",
        ],
        "Very short reactor periods can indicate an unstable transient that requires immediate corrective action.",
        "Period alarms, operator procedures, and protection setpoints are used to prevent unsafe acceleration of reactor power.",
    ),
    make_spec(
        "thermal hydraulics",
        "coolant flow",
        "Coolant flow is the movement of the reactor coolant that transports heat away from the core.",
        "Coolant flow is important because continuous heat removal is required to keep fuel and cladding temperatures within limits.",
        "The amount of heat removed by the coolant depends on flow rate, temperature rise, and fluid properties.",
        [
            "Fission deposits heat inside the fuel pellets.",
            "That heat conducts through the fuel and cladding to the coolant at the rod surface.",
            "Flowing coolant carries the absorbed thermal energy away from the core region.",
            "Fuel and cladding temperatures remain controlled only if the heat carried away matches the heat being produced.",
        ],
        "Reduced coolant flow can raise fuel and cladding temperature and shrink the thermal margin to boiling or damage limits.",
        "Pumps, natural circulation capability, and operating limits are used to maintain adequate core cooling under normal and abnormal conditions.",
    ),
    make_spec(
        "thermal hydraulics",
        "boiling heat transfer",
        "Boiling heat transfer is the transfer of heat from a hot surface to a liquid coolant as vapor bubbles form and depart.",
        "Boiling can be very effective because bubble formation enhances convection and carries latent heat away from the surface.",
        "The detailed heat-transfer behavior depends on pressure, mass flow, surface condition, and local heat flux.",
        [
            "As surface temperature rises above the local saturation condition, vapor bubbles begin to form at nucleation sites.",
            "Bubble growth and departure stir the nearby liquid and increase heat-transfer effectiveness.",
            "The coolant removes larger amounts of energy while the surface remains in the nucleate boiling regime.",
            "If heating continues too far, the boiling regime can shift toward less effective heat transfer.",
        ],
        "If boiling conditions move beyond stable nucleate boiling, the cladding surface can experience a rapid temperature increase.",
        "Thermal-hydraulic limits are set so boiling remains in acceptable regimes throughout the allowed operating envelope.",
    ),
    make_spec(
        "thermal hydraulics",
        "critical heat flux",
        "Critical heat flux is the heat-flux limit at which normal nucleate boiling can transition to a much less effective heat-transfer regime.",
        "It is important because cladding temperature can rise sharply once this limit is exceeded.",
        "Critical heat flux depends on local pressure, quality, mass flux, geometry, and flow conditions.",
        [
            "Heat flux at the rod surface increases as reactor power rises.",
            "At sufficiently high heat flux, the local boiling pattern changes and stable liquid contact with the surface is reduced.",
            "The surface can become partially or fully covered by vapor instead of well-cooled liquid.",
            "Heat transfer then degrades and cladding temperature increases quickly.",
        ],
        "Exceeding critical heat flux threatens cladding integrity and is one of the key thermal limits in reactor operation.",
        "Core design, operating margins, and protection systems are built to keep heat flux below the predicted limit with margin.",
    ),
    make_spec(
        "thermal hydraulics",
        "steam generator heat transfer",
        "Steam generator heat transfer is the process by which heat from the primary coolant is transferred to the secondary side to produce steam.",
        "It is important because it links reactor heat production to the turbine cycle without mixing the two fluid systems.",
        "Efficient steam generator performance supports both plant efficiency and stable primary-side temperatures.",
        [
            "Hot primary coolant leaves the reactor core carrying fission heat.",
            "That coolant flows through steam generator tubes while cooler secondary water surrounds the tubes.",
            "Heat passes through the tube walls into the secondary side and boils the secondary water into steam.",
            "The cooled primary fluid then returns to the reactor loop for another heat-removal cycle.",
        ],
        "Poor steam generator performance can reduce heat removal from the primary system and disturb plant thermal balance.",
        "Water chemistry control, tube integrity monitoring, and heat-balance surveillance are used to preserve reliable heat transfer.",
    ),
    make_spec(
        "thermal hydraulics",
        "natural circulation",
        "Natural circulation is coolant flow driven by density differences rather than by mechanical pumping.",
        "It is important because it can provide passive heat removal when forced circulation is reduced or unavailable.",
        "Warmer fluid becomes less dense and rises, while cooler denser fluid descends and completes the circulation loop.",
        [
            "Heating in the core raises coolant temperature and lowers its density.",
            "The lighter hot coolant tends to rise through the system.",
            "Cooler and denser fluid in other parts of the loop moves downward to replace it.",
            "The resulting buoyancy-driven flow carries heat away from the core without relying on pumps.",
        ],
        "Weak natural circulation may be insufficient if geometry, pressure conditions, or heat load do not support adequate flow.",
        "Passive safety designs and emergency procedures rely on preserving a flow path that allows buoyancy-driven cooling when needed.",
    ),
    make_spec(
        "thermal hydraulics",
        "pressurizer pressure control",
        "Pressurizer pressure control is the management of primary-system pressure so the coolant remains in the desired thermodynamic state.",
        "It is important because pressure affects saturation temperature, boiling margin, and overall primary-system behavior.",
        "Stable pressure helps maintain predictable coolant density and heat-transfer conditions in the reactor loop.",
        [
            "Heaters and spray systems adjust the state of water and steam inside the pressurizer volume.",
            "Those changes alter the pressure of the connected primary system.",
            "Primary pressure then influences the local boiling margin and the temperature at which bulk boiling would begin.",
            "Thermal-hydraulic behavior in the core remains more predictable when pressure is held near its design target.",
        ],
        "Loss of pressure control can narrow boiling margin and complicate the response to transients in the primary loop.",
        "Instrumentation, relief paths, and control systems are used to keep primary pressure within its allowed operating band.",
    ),
    make_spec(
        "thermal hydraulics",
        "hot channel factor",
        "The hot channel factor is a measure of how much local power or temperature in the hottest fuel channel exceeds the core average.",
        "It is important because local peaks, rather than average conditions, often determine the limiting thermal margin.",
        "Analysts use hot channel factors to account for nonuniform power distribution and manufacturing tolerances.",
        [
            "Power is not perfectly uniform across every fuel rod and every flow channel in the core.",
            "Some locations experience higher local heat generation than the core average.",
            "Those hotter locations produce higher local fuel and cladding temperatures than average conditions would suggest.",
            "Thermal design must therefore include margin for the most limiting channel, not just the mean channel.",
        ],
        "If local peaking is underestimated, a nominally acceptable average condition can still produce damage in the limiting channel.",
        "Peaking-factor limits, surveillance, and core-design methods are used to protect the hottest fuel region with margin.",
    ),
    make_spec(
        "thermal hydraulics",
        "fuel-to-coolant heat conduction",
        "Fuel-to-coolant heat conduction is the path by which heat moves from the fuel pellet through the cladding into the coolant.",
        "It is important because the temperature of the fuel and cladding depends on every resistance in this heat path.",
        "Gap conductance, cladding thickness, and coolant convection all influence the overall temperature drop.",
        [
            "Fission energy is deposited first inside the fuel pellet.",
            "Heat then conducts outward through the fuel and any pellet-cladding gap resistance.",
            "The cladding transfers that heat to the coolant by conduction and convection at the outer surface.",
            "If any step in this path becomes less effective, fuel and cladding temperatures rise for the same power level.",
        ],
        "Poor heat-transfer performance can reduce thermal margin and accelerate material degradation in the fuel rod.",
        "Fuel design, gap control, and coolant-flow limits are chosen to keep this heat path effective across the operating range.",
    ),
    make_spec(
        "materials behavior",
        "uranium dioxide thermal conductivity",
        "Uranium dioxide thermal conductivity is the ability of ceramic fuel to conduct heat from the pellet interior toward the surface.",
        "It is important because poor conductivity increases the temperature gradient inside the fuel pellet.",
        "Fuel temperature depends on both the power density and the ease with which heat can move through the ceramic.",
        [
            "Fission deposits heat throughout the fuel pellet volume.",
            "That heat must conduct through the ceramic to reach the cladding and coolant.",
            "If thermal conductivity is low, a larger temperature difference is required to move the same heat outward.",
            "The pellet centerline temperature therefore rises as conductivity degrades.",
        ],
        "High fuel temperature can accelerate swelling, fission-gas release, and other fuel-performance challenges.",
        "Fuel design and operating limits account for conductivity changes so pellet temperature stays within acceptable margins.",
    ),
    make_spec(
        "materials behavior",
        "fuel swelling",
        "Fuel swelling is the gradual increase in fuel volume caused by fission-product buildup and irradiation-induced microstructural change.",
        "It is important because swelling changes pellet geometry and can affect pellet-cladding interaction.",
        "Swelling becomes more significant as burnup increases and fission products accumulate in the fuel matrix.",
        [
            "Fission produces solid and gaseous products inside the fuel pellet during operation.",
            "Those products occupy space and alter the pellet microstructure over time.",
            "The fuel volume gradually increases as burnup and internal damage accumulate.",
            "Pellet-cladding gap conditions and internal stress then change along the fuel rod.",
        ],
        "Excessive swelling can tighten pellet-cladding contact and contribute to mechanical loading of the cladding.",
        "Fuel performance models and burnup limits are used to keep swelling effects within the designed mechanical margin.",
    ),
    make_spec(
        "materials behavior",
        "fuel cracking",
        "Fuel cracking is the formation of fractures in ceramic fuel caused by thermal stress and irradiation effects.",
        "It is important because crack formation changes the internal stress pattern and heat path in the pellet.",
        "Ceramic fuel is brittle, so steep temperature gradients can create stress large enough to cause cracking.",
        [
            "The center of a fuel pellet is usually hotter than the outer region during operation.",
            "Different radial temperatures cause different amounts of thermal expansion inside the ceramic.",
            "That uneven expansion creates internal stress in the brittle fuel material.",
            "The fuel responds by cracking and redistributing the stress field.",
        ],
        "Although controlled cracking can be expected, excessive cracking can influence fission-gas behavior and pellet mechanical response.",
        "Fuel design and power-ramp limits are used to manage thermal stress so cracking remains within analyzed behavior.",
    ),
    make_spec(
        "materials behavior",
        "pellet-cladding mechanical interaction",
        "Pellet-cladding mechanical interaction is the contact loading that develops when swelling or expanding fuel presses against the cladding.",
        "It is important because the cladding must remain intact while accommodating fuel deformation during irradiation.",
        "The interaction depends on gap size, burnup, fuel swelling, thermal expansion, and cladding strength.",
        [
            "Fuel temperature and irradiation cause the pellet to expand and swell over time.",
            "The gap between pellet and cladding gradually narrows as the fuel diameter increases.",
            "Once contact occurs, further pellet expansion transfers stress directly to the cladding wall.",
            "Cladding strain then rises and must remain below the analyzed failure threshold.",
        ],
        "Severe pellet-cladding interaction can contribute to cladding strain, cracking, or reduced fuel-rod reliability during power ramps.",
        "Power maneuver limits, fuel design choices, and material surveillance are used to manage this interaction safely.",
    ),
    make_spec(
        "materials behavior",
        "zircaloy oxidation",
        "Zircaloy oxidation is the chemical growth of an oxide layer on zirconium-alloy cladding exposed to high-temperature water or steam.",
        "It is important because oxidation changes cladding thickness, strength, and heat-transfer behavior.",
        "Oxidation rate increases as temperature rises and as the chemical environment becomes more aggressive.",
        [
            "Hot zirconium alloy reacts with oxygen-bearing coolant or steam at the cladding surface.",
            "An oxide layer forms and grows on the outside of the cladding wall.",
            "As oxidation progresses, the remaining metal wall becomes thinner and its properties change.",
            "Heat transfer and mechanical margin can then degrade if oxidation becomes excessive.",
        ],
        "Rapid high-temperature oxidation can seriously weaken cladding and is a key concern during severe cooling challenges.",
        "Chemistry control, temperature limits, and accident-mitigation systems are used to constrain oxidation damage.",
    ),
    make_spec(
        "materials behavior",
        "cladding creep",
        "Cladding creep is the slow time-dependent deformation of cladding under stress at elevated temperature.",
        "It is important because gradual shape change can alter fuel-rod geometry and mechanical margin during long service.",
        "Creep rate depends on temperature, stress level, irradiation condition, and material state.",
        [
            "The cladding experiences mechanical stress from internal pressure, coolant pressure, and pellet contact.",
            "Elevated temperature allows the metal lattice to deform gradually under that sustained load.",
            "Over long operating periods, the cladding shape changes even if the applied load remains similar.",
            "Gap size, strain distribution, and long-term integrity then evolve with time.",
        ],
        "Excessive creep can reduce dimensional margin and make the cladding more vulnerable during later transients.",
        "Material selection, stress limits, and fuel-rod design are used to keep cladding creep within acceptable bounds.",
    ),
    make_spec(
        "materials behavior",
        "radiation embrittlement",
        "Radiation embrittlement is the loss of toughness that occurs when neutron irradiation alters the microstructure of reactor metals.",
        "It is important because embrittled structural materials tolerate rapid thermal or mechanical loading less effectively.",
        "Pressure-vessel steels are monitored closely because they experience neutron exposure for many years of plant operation.",
        [
            "Fast neutrons displace atoms within the metal lattice during irradiation.",
            "Those defects and precipitates change the microstructure of the metal over time.",
            "The altered microstructure raises brittleness and lowers fracture toughness.",
            "The material then has less margin against crack initiation or crack growth during transients.",
        ],
        "Embrittled pressure-boundary materials can reduce structural safety margin during thermal shock or other demanding events.",
        "Surveillance capsules, operating limits, and life-management programs are used to track and manage embrittlement.",
    ),
    make_spec(
        "materials behavior",
        "hydrogen pickup in cladding",
        "Hydrogen pickup in cladding is the absorption of hydrogen into zirconium alloy during corrosion in reactor coolant.",
        "It is important because absorbed hydrogen can form brittle hydrides in the metal.",
        "Hydride formation changes ductility and can influence the way cladding responds to stress and temperature transients.",
        [
            "Corrosion reactions at the cladding surface release hydrogen as oxide forms.",
            "A portion of that hydrogen enters the zirconium alloy rather than remaining in the coolant.",
            "The absorbed hydrogen precipitates as hydrides when local conditions favor hydride formation.",
            "Those hydrides reduce ductility and alter the mechanical response of the cladding.",
        ],
        "High hydride content can increase the risk of brittle cladding behavior during storage, handling, or accident loading.",
        "Coolant chemistry control and cladding-alloy selection are used to limit hydrogen pickup and preserve ductility.",
    ),
    make_spec(
        "safety systems",
        "loss of coolant accident",
        "A loss of coolant accident is an event in which reactor coolant is lost from the primary system faster than it can be replaced normally.",
        "It is important because coolant is needed to remove both fission heat and decay heat from the core.",
        "Loss of coolant threatens heat removal even if the chain reaction has already been shut down.",
        [
            "A pipe break, valve failure, or other fault allows coolant to escape from the primary boundary.",
            "Core water inventory and pressure then decrease, reducing the ability of the system to remove heat from the fuel.",
            "Fuel and cladding temperatures begin to rise because heat generation continues while cooling weakens.",
            "Emergency cooling systems must inject water to restore cooling and limit damage progression.",
        ],
        "If coolant loss is not arrested and core cooling is not restored, fuel damage and fission-product release risk increase sharply.",
        "Emergency core cooling, reactor trip, and containment functions are designed to manage this accident sequence.",
    ),
    make_spec(
        "safety systems",
        "decay heat",
        "Decay heat is the heat released by radioactive decay after the fission chain reaction has been stopped.",
        "It is important because the core continues to produce substantial heat after shutdown.",
        "Decay heat decreases with time, but it remains large enough immediately after shutdown to require reliable cooling.",
        [
            "Fission creates many radioactive fission products during reactor operation.",
            "After shutdown, those nuclides continue to decay and release energy even though prompt fission has stopped.",
            "The remaining heat load keeps the fuel and structures hot unless coolant removes the energy.",
            "Cooling systems must therefore continue operating to prevent post-shutdown overheating.",
        ],
        "Ignoring decay heat can lead to overheating after shutdown, even when the reactor is no longer critical.",
        "Residual heat removal, emergency cooling, and well-defined shutdown procedures are used to manage decay heat safely.",
    ),
    make_spec(
        "safety systems",
        "emergency core cooling system",
        "The emergency core cooling system is a safety system that injects water or otherwise restores cooling after certain accidents.",
        "It is important because accident conditions can remove the normal cooling path while heat generation continues.",
        "Different injection stages are often provided so the system can operate across a wide range of pressures and break sizes.",
        [
            "An accident reduces or interrupts the normal ability of the primary system to cool the core.",
            "Sensors and protection logic detect the abnormal condition and initiate emergency cooling.",
            "Stored or pumped water is delivered to the reactor vessel or primary system to refill and cool the core.",
            "Restored coolant inventory improves heat transfer and limits further temperature rise in the fuel and cladding.",
        ],
        "Failure of emergency core cooling would greatly increase the chance of severe fuel damage during a coolant-loss event.",
        "Redundancy, diversity, testing, and independent power supplies are used to keep emergency cooling available when demanded.",
    ),
    make_spec(
        "safety systems",
        "containment",
        "Containment is the engineered barrier that surrounds key reactor systems and limits radioactive release during accidents.",
        "It is important because it provides the final physical barrier after fuel and coolant-pressure boundaries.",
        "Containment also provides a controlled volume for managing steam, pressure, and airborne radioactive material during severe transients.",
        [
            "An accident can release steam, hydrogen, or radioactive material from damaged systems inside the plant.",
            "Containment structures confine those materials within a robust sealed volume.",
            "Pressure-control features, sprays, and filtration systems then manage conditions inside the containment space.",
            "The amount of radioactive material that can escape to the environment is therefore reduced.",
        ],
        "A weak or bypassed containment function can greatly increase offsite release potential even if other safety systems operate.",
        "Containment integrity monitoring, isolation valves, and pressure-control systems are maintained to preserve this final barrier.",
    ),
    make_spec(
        "safety systems",
        "shutdown margin",
        "Shutdown margin is the amount of negative reactivity available to keep the reactor subcritical under specified conditions.",
        "It is important because the reactor must remain safely shut down despite temperature changes, xenon effects, or equipment configuration.",
        "Adequate shutdown margin ensures that control systems can overcome the most limiting reactivity conditions considered in design and operation.",
        [
            "When the reactor trips, control rods or other absorbers insert negative reactivity into the core.",
            "The total inserted negative reactivity must exceed any positive effects that still remain in the shutdown condition.",
            "If the margin is sufficient, the neutron population continues to fall and the core stays subcritical.",
            "If the margin were too small, the reactor could approach criticality again under unfavorable conditions.",
        ],
        "Insufficient shutdown margin weakens the guarantee that the reactor can be held safely subcritical after a trip or during refueling conditions.",
        "Control-rod requirements, boron concentration limits, and surveillance tests are used to confirm shutdown margin remains adequate.",
    ),
    make_spec(
        "safety systems",
        "defense in depth",
        "Defense in depth is the safety philosophy of using multiple independent and overlapping layers of protection against accidents and releases.",
        "It is important because no single barrier or system is assumed to be perfect under all conditions.",
        "The approach combines prevention, control, mitigation, and containment rather than relying on only one line of defense.",
        [
            "Plant design begins by reducing the likelihood of faults through conservative engineering and quality control.",
            "If an abnormal event still occurs, control and shutdown systems act to limit its progression.",
            "Additional cooling and containment functions then mitigate damage and limit radioactive release if earlier layers are insufficient.",
            "The overall risk falls because several barriers and safety functions must fail before severe consequences develop.",
        ],
        "Removing one layer of defense increases dependence on the remaining barriers and can narrow the margin against complex failures.",
        "Modern nuclear safety programs maintain redundant barriers and independent protective functions so defense in depth remains credible.",
    ),
    make_spec(
        "safety systems",
        "station blackout",
        "Station blackout is the loss of offsite power together with the loss or failure of the normal onsite AC power sources.",
        "It is important because many active cooling and control systems depend on reliable electrical power.",
        "Without power, instrumentation, pumps, and many safety-support systems may degrade over time.",
        [
            "A disturbance removes offsite electrical power to the plant.",
            "If emergency generators or alternate AC supplies are unavailable, powered cooling and support systems cannot operate normally.",
            "Core heat generation and decay heat removal then become increasingly difficult to manage as batteries deplete and thermal conditions worsen.",
            "The plant must rely on passive features, alternate power, or rapid recovery actions to prevent overheating.",
        ],
        "Extended station blackout can progress to core damage if cooling and instrumentation cannot be sustained long enough.",
        "Backup generators, battery capacity, alternate AC strategies, and passive cooling features are used to reduce blackout risk.",
    ),
    make_spec(
        "safety systems",
        "residual heat removal",
        "Residual heat removal is the continued removal of decay heat from the reactor after shutdown or during low-power states.",
        "It is important because stopping the chain reaction does not stop the heat load inside the core.",
        "Residual heat removal systems provide an organized path for carrying remaining thermal energy to a safe heat sink.",
        [
            "After shutdown, radioactive decay in fission products continues generating heat inside the fuel.",
            "Dedicated low-temperature or shutdown-cooling systems circulate coolant and transfer that heat to an external sink.",
            "As heat is removed, fuel and system temperatures continue falling toward stable shutdown conditions.",
            "The risk of overheating remains low only while this residual heat-removal function is maintained.",
        ],
        "Loss of residual heat removal after shutdown can still lead to dangerous temperature rise if cooling is not restored promptly.",
        "Shutdown cooling procedures, alternate heat-removal paths, and operator response guidance are used to protect this function.",
    ),
]


def build_definition_variants(spec: Dict[str, object]) -> List[str]:
    """Create deterministic definition-style samples for one subject."""
    definition = str(spec["definition"])
    importance = str(spec["importance"])
    explanation = str(spec["explanation"])
    mechanism = [str(step) for step in spec["mechanism"]]
    return [
        f"{definition} {importance} {explanation}",
        f"{definition} {explanation} {importance}",
        f"{definition} {importance} {mechanism[0]}",
        f"{definition} {explanation} {mechanism[1]}",
        f"{definition} {mechanism[0]} {importance}",
    ]


def build_explanation_variants(spec: Dict[str, object]) -> List[str]:
    """Create deterministic explanation-style samples for one subject."""
    importance = str(spec["importance"])
    explanation = str(spec["explanation"])
    mechanism = [str(step) for step in spec["mechanism"]]
    definition = str(spec["definition"])
    return [
        f"{importance} {explanation} {mechanism[0]}",
        f"{explanation} {importance} {mechanism[1]}",
        f"{importance} {mechanism[0]} {mechanism[1]}",
        f"{explanation} {mechanism[1]} {importance}",
        f"{importance} {definition} {mechanism[2]}",
    ]


def build_mechanism_variants(spec: Dict[str, object]) -> List[str]:
    """Create deterministic mechanism-style samples for one subject."""
    mechanism = [str(step) for step in spec["mechanism"]]
    importance = str(spec["importance"])
    explanation = str(spec["explanation"])
    return [
        " ".join(mechanism),
        f"{mechanism[0]} {mechanism[1]} {mechanism[2]} {importance}",
        f"{mechanism[0]} {mechanism[1]} {mechanism[3]} {explanation}",
        f"{importance} {mechanism[0]} {mechanism[2]} {mechanism[3]}",
        f"{mechanism[1]} {mechanism[2]} {mechanism[3]} {importance}",
        f"{mechanism[0]} {explanation} {mechanism[2]} {mechanism[3]}",
        f"{importance} {mechanism[1]} {mechanism[2]} {mechanism[3]}",
    ]


def build_safety_variants(spec: Dict[str, object]) -> List[str]:
    """Create deterministic safety-analysis samples for one subject."""
    safety = str(spec["safety"])
    control = str(spec["control"])
    importance = str(spec["importance"])
    mechanism = [str(step) for step in spec["mechanism"]]
    return [
        f"{safety} {control} {importance}",
        f"{safety} {mechanism[2]} {control}",
        f"{importance} {safety} {control}",
    ]


def generate_synthetic_nuclear_samples() -> List[Dict[str, str]]:
    """Generate a balanced synthetic nuclear corpus with fixed concept/type counts."""
    records: List[Dict[str, str]] = []
    builders = {
        "definition": build_definition_variants,
        "explanation": build_explanation_variants,
        "mechanism": build_mechanism_variants,
        "safety_analysis": build_safety_variants,
    }

    for spec in SPECS:
        concept = str(spec["concept"])
        for entry_type in TYPE_ORDER:
            variants = builders[entry_type](spec)
            required = TYPE_COUNTS[entry_type]
            if len(variants) != required:
                raise RuntimeError(
                    f"Synthetic generator expected {required} {entry_type} variants for {spec['subject']}, got {len(variants)}."
                )
            for text in variants:
                records.append(
                    {
                        "source": "synthetic",
                        "topic": concept,
                        "category": entry_type,
                        "text": " ".join(text.split()),
                    }
                )

    expected_total = len(SPECS) * sum(TYPE_COUNTS.values())
    if len(records) != expected_total:
        raise RuntimeError(f"Expected {expected_total} synthetic samples, produced {len(records)}.")
    return records


def write_synthetic_concept_dataset(path: Path = OUTPUT_PATH) -> Path:
    """Write the deterministic synthetic corpus to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = generate_synthetic_nuclear_samples()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "text": record["text"],
                "concept": record["topic"],
                "type": record["category"],
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return path


def main() -> None:
    """Generate the locked synthetic concept dataset."""
    output_path = write_synthetic_concept_dataset()
    records = generate_synthetic_nuclear_samples()
    concept_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    for record in records:
        concept_counts[record["topic"]] = concept_counts.get(record["topic"], 0) + 1
        type_counts[record["category"]] = type_counts.get(record["category"], 0) + 1
    print("synthetic_output:", output_path)
    print("synthetic_total_records:", len(records))
    print("synthetic_concept_distribution:", concept_counts)
    print("synthetic_type_distribution:", type_counts)


if __name__ == "__main__":
    assert_side_execution_forbidden()
