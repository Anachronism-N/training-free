#!/usr/bin/env python3
"""Build observational and controlled prompt suites for v134 head discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FACTORS = (
    "identity",
    "appearance",
    "action",
    "scene",
    "object",
    "camera",
    "atmosphere",
    "style",
)

COMPONENTS = {
    "identity": (
        ("A silver-haired woman with a narrow face", "A young man with curly black hair and a round face"),
        ("A middle-aged watchmaker with round brass glasses", "An elderly watchmaker with square black glasses"),
        ("A young Black cellist with a high braided bun", "A young East Asian cellist with a short bob haircut"),
        ("An elderly Japanese ceramic artist with cropped gray hair", "A young Japanese ceramic artist with long black hair"),
        ("A tall detective with a trimmed beard", "A short detective with a clean-shaven face"),
        ("A freckled bicycle courier with copper hair", "A dark-haired bicycle courier with an angular face"),
        ("A bearded ballroom dancer with olive skin", "A clean-shaven ballroom dancer with pale skin"),
        ("A woman geologist with a long auburn braid", "A man geologist with close-cropped blond hair"),
        ("A South Asian botanist with silver-rimmed glasses", "A Latina botanist with tortoiseshell glasses"),
        ("A broad-shouldered lighthouse keeper with a gray moustache", "A slender lighthouse keeper with a black moustache"),
        ("A young violin maker with a crescent-shaped birthmark", "An older violin maker with a straight scar on one cheek"),
        ("A Black female architect with close-cropped hair", "A white male architect with shoulder-length brown hair"),
        ("A teenage skateboarder with a shaved side haircut", "An adult skateboarder with long blond dreadlocks"),
        ("An Inuit wildlife photographer with a weathered face", "A Korean wildlife photographer with a smooth round face"),
        ("A female train conductor with a distinctive white forelock", "A male train conductor with a distinctive red beard"),
        ("An elderly astronomer with thick white eyebrows", "A young astronomer with thin dark eyebrows"),
    ),
    "appearance": (
        ("wearing a weathered red field jacket and black leather gloves", "wearing a pristine cobalt coat and cream wool gloves"),
        ("wearing a navy apron over a striped shirt", "wearing a burgundy waistcoat over a plain white shirt"),
        ("wearing a dark emerald concert dress and a silver pendant", "wearing an ivory concert suit and a gold brooch"),
        ("wearing an indigo work coat dusted with pale clay", "wearing a saffron work coat dusted with black clay"),
        ("wearing a charcoal overcoat and burgundy scarf", "wearing a tan trench coat and forest-green scarf"),
        ("wearing a yellow helmet and teal backpack", "wearing a white helmet and crimson messenger bag"),
        ("wearing a cream linen suit and polished brown shoes", "wearing a midnight-blue velvet suit and matte black shoes"),
        ("wearing an orange climbing shell and scratched silver helmet", "wearing a lime climbing shell and glossy black helmet"),
        ("wearing a moss-green utility vest and canvas trousers", "wearing a violet rain cape and dark denim trousers"),
        ("wearing a heavy navy sweater and brass key ring", "wearing a red oilskin coat and steel key ring"),
        ("wearing a brown leather apron and rolled linen sleeves", "wearing a black rubber apron and fitted cotton sleeves"),
        ("wearing a slate blazer and translucent drafting visor", "wearing a white coverall and opaque welding visor"),
        ("wearing a purple hoodie, knee pads, and red sneakers", "wearing a gray jacket, elbow pads, and blue boots"),
        ("wearing a white insulated parka with blue trim", "wearing a black insulated parka with orange trim"),
        ("wearing a bottle-green uniform and brass pocket watch", "wearing a maroon uniform and silver wristwatch"),
        ("wearing a charcoal cardigan and a copper hearing aid", "wearing a cream cardigan and a black headset"),
    ),
    "action": (
        ("walks steadily while checking a folded map", "runs quickly while waving the folded map overhead"),
        ("repairs a delicate clock mechanism with tweezers", "closes the clock case and winds it with a large key"),
        ("performs a slow expressive passage on a cello", "stops playing and carries the cello across the room"),
        ("shapes a tall clay vessel on a turning wheel", "paints geometric marks onto a finished clay vessel"),
        ("follows a set of wet footprints while taking notes", "erases the wet footprints and searches the ceiling"),
        ("repairs a bicycle chain with careful hand movements", "mounts the bicycle and pedals rapidly through the crowd"),
        ("stands almost still while guests circle nearby", "performs a sequence of fast turns through the guests"),
        ("kneels to inspect a rock sample and then climbs onward", "throws the sample away and descends toward the water"),
        ("catalogues rare leaves and adjusts specimen labels", "waters the plants and releases a cloud of butterflies"),
        ("polishes the rotating lamp housing and tests its gears", "rings an alarm bell and runs down the spiral stairs"),
        ("carves a violin bridge with a narrow hand tool", "strings the violin and performs a short melody"),
        ("studies a physical model and redraws one curved facade", "dismantles the model and assembles a tall square tower"),
        ("balances on a board and rolls through a gentle curve", "leaps from the board and sprints up a staircase"),
        ("tracks an animal through binoculars and takes photographs", "lowers the camera and digs a shallow shelter in the snow"),
        ("checks tickets while walking between occupied seats", "pulls an emergency lever and directs passengers toward a door"),
        ("aligns a telescope and records observations by hand", "closes the observatory dome and projects a star map on the wall"),
    ),
    "scene": (
        ("across a rain-slick European square beside a green tram", "through a dry desert market beside a blue bus"),
        ("at a crowded walnut workbench inside a narrow clock shop", "at a steel laboratory bench inside a bright clean room"),
        ("inside an abandoned glasshouse filled with white lilies", "inside a stone concert hall filled with red banners"),
        ("in a quiet wooden studio with paper windows and bowl shelves", "in an open concrete courtyard with metal shelves and murals"),
        ("along a foggy 1930s railway platform beneath iron arches", "inside a sunlit modern airport beneath glass beams"),
        ("in a crowded courtyard bordered by terracotta apartments", "on an empty rooftop bordered by mirrored office towers"),
        ("at the center of a candlelit ballroom with mirrored walls", "at the center of a fluorescent gymnasium with painted walls"),
        ("along a remote coast of black basalt columns", "along a dense forest path of pale birch trees"),
        ("inside a humid conservatory packed with tropical plants", "inside a dry seed vault lined with metal cabinets"),
        ("inside a storm-battered lighthouse above a rocky harbor", "inside a calm underground control room below a city"),
        ("in a warm violin workshop lined with maple instruments", "in a cold machine shop lined with aluminum parts"),
        ("inside a modern architecture studio overlooking a river", "inside a ruined stone library overlooking a canyon"),
        ("through a tiled metro plaza covered in bright murals", "through a wooden mountain village covered in fresh snow"),
        ("across a wind-scoured Arctic inlet beside blue ice ridges", "across a humid mangrove lagoon beside tangled roots"),
        ("through a restored dining carriage with brass lamps", "through an outdoor freight yard with sodium floodlights"),
        ("inside a hilltop observatory beneath a rotating dome", "inside a submarine navigation room beneath the ocean"),
    ),
    "object": (
        ("A green tram, black umbrella, and brass street clock remain visible", "A blue bus, white parasol, and digital billboard remain visible"),
        ("A green desk lamp, music box, and trays of tiny gears remain visible", "A violet ceiling lamp, microscope, and sealed glass samples remain visible"),
        ("The same cello, carved chair, and rows of lilies remain visible", "The same violin, metal stool, and rows of banners remain visible"),
        ("A cobalt clay vessel, bamboo tools, and pale bowls remain visible", "A crimson glass vessel, steel tools, and black plates remain visible"),
        ("A black notebook, silver fountain pen, and leather suitcase remain visible", "A tablet computer, red marker, and plastic carry-on remain visible"),
        ("An orange bicycle, hand pump, and laundry lines remain visible", "A white scooter, battery pack, and antenna cables remain visible"),
        ("A silver mask, crystal glass, and red floor medallion remain visible", "A basketball, plastic bottle, and blue center-circle remain visible"),
        ("A scratched rock hammer, sample pouch, and orange marker remain visible", "A polished walking stick, canvas satchel, and green marker remain visible"),
        ("A brass magnifier, blue labels, and flowering orchid remain visible", "A barcode scanner, orange labels, and sealed grain jar remain visible"),
        ("A brass lens assembly, oil can, and tide chart remain visible", "A glass monitor, ceramic mug, and transit map remain visible"),
        ("A half-built violin, curled wood shavings, and red clamp remain visible", "A turbine blade, metal filings, and blue vise remain visible"),
        ("A white scale model, graphite ruler, and yellow tracing paper remain visible", "A stone maquette, laser ruler, and transparent plastic sheet remain visible"),
        ("A teal skateboard, portable speaker, and painted bench remain visible", "A red sled, brass bell, and carved wooden bench remain visible"),
        ("A long camera lens, red tripod, and numbered field notebook remain visible", "A waterproof drone, yellow paddle, and laminated map remain visible"),
        ("A brass ticket punch, porcelain cup, and folded newspaper remain visible", "A radio handset, steel flask, and cargo manifest remain visible"),
        ("A brass telescope, red observation book, and mechanical clock remain visible", "A sonar console, blue navigation log, and pressure gauge remain visible"),
    ),
    "camera": (
        ("The camera uses a smooth waist-high tracking shot with brief face close-ups", "The camera uses a locked overhead wide shot with no close-ups"),
        ("The camera alternates macro hand details with a slow shoulder-level push", "The camera remains in a distant static profile view"),
        ("The camera performs a slow left-to-right track and a shallow orbit", "The camera performs a rapid crane rise followed by a steep zoom"),
        ("The camera holds a wide frontal composition and slowly approaches the hands", "The camera circles behind the subject in a tight handheld view"),
        ("The camera follows from behind before moving into a measured side profile", "The camera stays ahead in a fixed low-angle reverse tracking shot"),
        ("The camera moves with restrained documentary handheld motion", "The camera executes a perfectly smooth aerial orbit"),
        ("The camera is stabilized at chest height and slowly rotates clockwise", "The camera shakes at floor height and rotates counter-clockwise"),
        ("The camera shifts between a medium follow shot and close views of the hands", "The camera remains in an extreme wide shot from a cliff"),
        ("The camera glides between plant rows and returns to a centered portrait", "The camera remains above the ceiling in a vertical top-down view"),
        ("The camera climbs the spiral stair in one continuous following shot", "The camera stays outside and observes through one distant window"),
        ("The camera uses precise macro inserts within a gentle lateral dolly", "The camera uses a fisheye lens in a fast backward dolly"),
        ("The camera orbits the model slowly and pauses on the architect's face", "The camera locks onto the ceiling and never shows the model"),
        ("The camera tracks parallel at board height and then widens gradually", "The camera remains stationary at rooftop height"),
        ("The camera pans slowly with the photographer and holds long telephoto views", "The camera rushes forward with an ultra-wide action lens"),
        ("The camera travels down the aisle in a level continuous shot", "The camera looks straight down from above the carriage roof"),
        ("The camera makes a slow circular dolly between the desk and telescope", "The camera stays in a fixed close-up of the floor"),
    ),
    "atmosphere": (
        ("Blue-hour rain creates restrained reflections while distant pedestrians keep moving", "Harsh noon sunlight creates sharp shadows while wind lifts dry dust"),
        ("Soft morning window light and steady rain create a calm warm interior", "Pulsing red emergency light and drifting vapor create a tense cold interior"),
        ("Warm sunset light crosses dusty glass as a light breeze moves the flowers", "Cold moonlight crosses wet stone as heavy fog hides the background"),
        ("Diffuse morning light reveals clay dust floating in otherwise still air", "Strong green stage light reveals thick smoke moving through the room"),
        ("Amber lamps glow through platform fog while rain moves across the background", "White LEDs shine through clear air while crowds cast crisp moving shadows"),
        ("Bright afternoon sun shifts behind laundry while neighbors move naturally", "Nighttime lightning flashes behind antennas while the rooftop stays empty"),
        ("Warm candlelight flickers across mirrors while fabric moves in a gentle draft", "Flat fluorescent light stays constant while loose papers whip in strong wind"),
        ("Clear cold daylight gradually gives way to spray and fast coastal cloud", "Warm humid twilight stays clear as insects drift above still water"),
        ("Moist green light filters through glass while condensation runs down panes", "Dry white light remains uniform while dust settles on closed cabinets"),
        ("Gray storm light pulses with the rotating beam while waves strike below", "Soft amber light remains steady while ventilation fans turn quietly"),
        ("Late-afternoon light warms wood grain while fine dust hangs in the air", "Blue industrial light reflects from steel while sparks cross the background"),
        ("Neutral daylight shifts with passing clouds while river reflections move", "Orange firelight flickers constantly while ash drifts through broken windows"),
        ("Clear evening light turns gradually toward neon night as commuters pass", "Flat overcast daylight remains unchanged as snow falls heavily"),
        ("Low polar sunlight glints through blowing snow and moving ice fog", "Green monsoon light filters through heavy rain and rising water vapor"),
        ("Warm carriage lamps sway subtly while rain streaks the windows", "Cold yard floodlights remain fixed while snow blows between freight cars"),
        ("Deep night sky rotates slowly beyond the slit as instrument lights glow", "Dim red emergency lighting pulses while bubbles pass the outer windows"),
    ),
    "style": (
        ("naturalistic cinematic realism with restrained color and detailed skin", "flat cel animation with bold outlines and simplified faces"),
        ("tactile macro realism with accurate metal, wood, and hand motion", "glossy toy-commercial imagery with exaggerated reflections"),
        ("elegant 35mm concert cinematography with realistic anatomy and fabric", "pixel-art game imagery with deliberately blocky motion"),
        ("quiet documentary realism with accurate clay and wood textures", "high-contrast graphic-novel panels with halftone shading"),
        ("precise period-film realism with coherent architecture and clothing", "surreal watercolor imagery with melting architecture"),
        ("observational street-documentary realism with natural crowd motion", "miniature stop-motion imagery with visible handcrafted joints"),
        ("controlled dramatic realism with stable faces and physically plausible cloth", "bright children's cartoon imagery with elastic bodies"),
        ("large-format landscape realism with natural weather and body mechanics", "low-poly 3D imagery with faceted terrain and rigid motion"),
        ("botanical documentary realism with fine leaf and glass detail", "soft pastel illustration with deliberately flattened depth"),
        ("moody maritime realism with accurate machinery, water, and light", "retro monochrome newsreel imagery with heavy film damage"),
        ("warm workshop realism with precise hand-tool interaction", "neon cyberpunk animation with synthetic materials"),
        ("clean architectural-film realism with legible spatial relationships", "charcoal sketch animation with unstable hand-drawn lines"),
        ("energetic urban realism with coherent limbs and board physics", "paper-cut collage animation with layered flat shapes"),
        ("patient wildlife-documentary realism with detailed snow and optics", "infrared surveillance imagery with false-color subjects"),
        ("polished historical-drama realism with consistent carriage geometry", "silent-film slapstick imagery with accelerated motion"),
        ("scientific observational realism with accurate instruments and night light", "dreamlike oil painting with visible brushwork and soft geometry"),
    ),
}


def _render_prompt(fields: dict[str, str], *, paraphrase: bool) -> str:
    if paraphrase:
        return (
            f"Create one uninterrupted shot in {fields['style']}. "
            f"{fields['scene'].capitalize()}, {fields['identity'].lower()}, "
            f"{fields['appearance']}, {fields['action']}. "
            f"{fields['atmosphere']}. {fields['object']}. "
            f"{fields['camera']}. Preserve the described face, body proportions, "
            "wardrobe, principal objects, and spatial layout across the entire "
            "sequence while keeping motion continuous and physically coherent."
        )
    return (
        f"{fields['identity']}, {fields['appearance']}, {fields['action']} "
        f"{fields['scene']}. {fields['object']}. {fields['camera']}. "
        f"{fields['atmosphere']}. Render the sequence as {fields['style']}. "
        "Maintain the subject's face, body proportions, wardrobe, principal "
        "objects, and scene geometry throughout a long continuous shot while "
        "allowing natural non-repeating motion."
    )


def build_counterfactual_jobs() -> list[dict]:
    jobs = []
    component_count = len(COMPONENTS["identity"])
    for subject_index in range(component_count):
        for factor_index, factor in enumerate(FACTORS):
            dataset_index = len(jobs)
            fields = {}
            row_indices = {}
            for component in FACTORS:
                row_index = subject_index
                row_indices[component] = row_index
                fields[component] = COMPONENTS[component][row_index][0]
            semantic_fields = dict(fields)
            semantic_row_index = row_indices[factor]
            semantic_fields[factor] = COMPONENTS[factor][semantic_row_index][1]
            jobs.append(
                {
                    "dataset_index": dataset_index,
                    "job_id": f"cf_{subject_index:02d}_{factor}",
                    "kind": "counterfactual",
                    "family_id": f"subject_{subject_index:02d}",
                    "seed": subject_index,
                    "factor": factor,
                    "changed_fields": [factor],
                    "null_type": "template_paraphrase",
                    "base_prompt": _render_prompt(fields, paraphrase=False),
                    "semantic_prompt": _render_prompt(
                        semantic_fields, paraphrase=False
                    ),
                    "null_prompt": _render_prompt(fields, paraphrase=True),
                }
            )
    if len(jobs) != 128:
        raise AssertionError(f"expected 128 counterfactual jobs, got {len(jobs)}")
    return jobs


def build_observational_jobs(prompts: list[str]) -> list[dict]:
    if len(prompts) != 128:
        raise ValueError(
            f"MovieBench observational suite requires 128 prompts, got {len(prompts)}"
        )
    return [
        {
            "dataset_index": index,
            "job_id": f"moviebench_qwen_{index:03d}",
            "kind": "observational",
            "family_id": f"moviebench_{index:03d}",
            "seed": index,
            "factor": "natural",
            "base_prompt": prompt,
        }
        for index, prompt in enumerate(prompts)
    ]


def _write_suite(output_dir: Path, name: str, jobs: list[dict]) -> dict:
    prompt_path = output_dir / f"{name}.txt"
    manifest_path = output_dir / f"{name}.jsonl"
    prompt_text = "\n".join(job["base_prompt"] for job in jobs) + "\n"
    manifest_text = "".join(
        json.dumps(job, ensure_ascii=True, sort_keys=True) + "\n"
        for job in jobs
    )
    prompt_path.write_text(prompt_text, encoding="utf-8")
    manifest_path.write_text(manifest_text, encoding="utf-8")
    return {
        "name": name,
        "count": len(jobs),
        "prompt_path": str(prompt_path),
        "manifest_path": str(manifest_path),
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moviebench-qwen", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = [
        line.strip()
        for line in args.moviebench_qwen.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suites = [
        _write_suite(
            args.output_dir,
            "moviebench128_observational",
            build_observational_jobs(prompts),
        ),
        _write_suite(
            args.output_dir,
            "controlled128_counterfactual",
            build_counterfactual_jobs(),
        ),
    ]
    metadata = {
        "version": 1,
        "method": "v134_prompt_history_head_discovery",
        "moviebench_source": str(args.moviebench_qwen),
        "factors": list(FACTORS),
        "suites": suites,
    }
    metadata_path = args.output_dir / "suite_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[v134-suite] "
        + " ".join(f"{row['name']}={row['count']}" for row in suites)
        + f" metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
