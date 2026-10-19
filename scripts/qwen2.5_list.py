import torch
import cv2
import os
import json
import csv
import time
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# -------------------- CONFIG --------------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

#VIDEO_FOLDER = r"D:/Henry/VLMs/diving"          # Carpeta con los videos
VIDEO_FOLDER = r"D:/Henry/VLMs/diving/diving_samples_len_ori"          # Carpeta con los videos

OUTPUT_FOLDER = r"D:/Henry/VLMs/qwen_3B/results_p6" # Carpeta de salida para JSONs y CSV
CSV_OUTPUT = os.path.join(OUTPUT_FOLDER, "evaluaciones_todos_videos.csv")
INTERVALO_SEGUNDOS = 0.25
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Device: {device} | Model: {MODEL_ID}")


import scipy.io

# ---- CARGAR GRADO DE DIFICULTAD ----
MAT_FILE = r"D:/Henry/VLMs/diving/diving_difficulty_level.mat"  # ajusta la ruta
mat_data = scipy.io.loadmat(MAT_FILE)
difficulty_levels = mat_data["difficulty_level"].flatten()  # array de 370 valores
print(f"Grados de dificultad cargados: {len(difficulty_levels)} entradas")

# -------------------- PROMPTS --------------------
# Agrega, elimina o modifica prompts aquí. El "name" se guarda en el CSV.
# Solo los prompts con "active": True serán ejecutados.
PROMPTS = [
{
        "name": "prompt_6",
        "active": False,  
        "text": """
You are an official World Aquatics judge scoring a 10m platform dive under the current World Aquatics rules. You will receive a video clip of a single dive. Analyze it rigorously and produce a structured judgment.

============================================================
CRITICAL ANTI-BIAS DIRECTIVES — READ FIRST
============================================================

LLM judges tend to collapse toward a single "comfort-zone" value in the upper-middle of the scale, regardless of the dive's actual quality. This is FORBIDDEN. Follow these rules strictly:

1. NO ANCHORING. Do not start from any preferred or habitual value. Start each phase at 0 and build the score upward ONLY based on observed evidence.

2. EVIDENCE-FIRST. Before assigning any sub-score, list at least 2 concrete, observable facts from the video for that phase (e.g., "knees separated ~15 cm in tuck", "over-rotated past vertical by ~10°", "splash height ~40 cm", "feet crossed at entry"). Vague phrases like "good execution", "clean entry", "solid take-off" are PROHIBITED unless backed by a specific observation.

3. ASSUME IMPERFECTION. Real Olympic dives almost always contain visible flaws. If you report no flaws in a phase, you are likely wrong — re-examine the video.

4. USE THE FULL SCALE. Across many dives your scores must span the entire 0–10 range. Do not cluster outputs in a narrow band. Map your evidence to the correct band of the World Aquatics scale (see section D), not to a habitual value.

5. TIE-BREAK DOWNWARD. When hesitating between two adjacent values (e.g., 7.5 vs 8.0), choose the LOWER one. Judges punish, not reward, ambiguity.

6. NO COMFORT-ZONE OUTPUTS. Any final QoE that lands in the "Very Good" band (upper portion of the scale) requires that EVERY phase sub-score independently justifies that band, AND that the entry phase is among the strongest. If those conditions are not met, the final QoE MUST fall into a lower band.

7. VARIANCE REQUIREMENT. The five sub-scores must NOT all be identical or fall within a 0.5 spread of each other unless you provide explicit, phase-specific evidence that all five phases were executed at the same level. Phases naturally differ; identical sub-scores are a red flag for lazy scoring.

8. SELF-AUDIT. After producing the final QoE, ask yourself: "Would a different evaluator, given only my listed observations and defects, arrive at the same score?" If the observations describe clear flaws but the score sits in a high band, RECOMPUTE downward.

============================================================
REFERENCE FRAMEWORK (World Aquatics)
============================================================

A) DIVE GROUPS — Forward, Backward, Reverse, Inward, Twisting, Armstand.
B) POSITIONS — Straight (A), Pike (B), Tuck (C), Free (D).

C) QUALITY OF EXECUTION — 5 PHASES & MANDATORY DEFECT CHECKLIST

For EACH phase, scan for the following defects. Report which are present and which are absent.

1. Starting Position
   □ Lack of stillness / wobble
   □ Poor posture (rounded shoulders, bent knees)
   □ Arms not fixed
   □ Visible nerves / re-adjustment
   □ For armstand: unstable balance, shaking, walking hands

2. Approach
   □ Hesitation / balk
   □ Irregular rhythm or step count
   □ Weak hurdle / low knee drive
   □ Unstable press (back/inward)
   □ Lateral drift

3. Take-off
   □ Insufficient height
   □ Too close to / too far from platform
   □ Off-vertical lean
   □ Late or premature rotation initiation
   □ Weak leg drive
   □ Arms not coordinated

4. Flight
   □ Loose tuck (knees apart, hands not on shins)
   □ Bent knees in pike or straight position
   □ Toes not pointed
   □ Twists not on longitudinal axis (tilt)
   □ Under-rotation or over-rotation
   □ Late kick-out
   □ Asymmetric body shape

5. Entry
   □ Past vertical (over-rotated) — degrees estimate
   □ Short of vertical (under-rotated) — degrees estimate
   □ Splash size (small "rip" / medium / large)
   □ Bent arms or knees at entry
   □ Feet/hands not together
   □ Body break at hips on impact

D) JUDGING SCALE (World Aquatics, decimals allowed)
   - 10        : Excellent — flawless, exceptional height, ripped entry
   - 8.5 – 9.5 : Very Good — minor flaws, strong overall
   - 7.0 – 8.0 : Good — competent, multiple small flaws OR one moderate flaw
   - 5.0 – 6.5 : Satisfactory — visible execution problems
   - 2.5 – 4.5 : Deficient — major errors (significant over/under-rotation, large splash, broken position)
   - 0.5 – 2.0 : Unsatisfactory — failed dive but completed
   - 0         : Completely failed (balk, no attempt, wrong dive)

   Map evidence → band. Do NOT pick a value first and then justify it.

E) OVERALL QoE COMPUTATION (mandatory term-by-term, arithmetic-checked)

Weights: Entry=0.30, Flight=0.30, Take_off=0.20, Approach=0.10, Starting_Position=0.10.
(They sum to 1.00 — verify this.)

You MUST compute and report each weighted term INDIVIDUALLY, then sum them. Do NOT report only the final number. Do NOT collapse arithmetic into a single string.

Procedure (follow exactly):
  1. term_starting   = Starting_Position.sub_score * 0.10
  2. term_approach   = Approach.sub_score          * 0.10
  3. term_take_off   = Take_off.sub_score          * 0.20
  4. term_flight     = Flight.sub_score            * 0.30
  5. term_entry      = Entry.sub_score             * 0.30
  6. weighted_sum    = sum of the five terms above
  7. adjustment      = a value in [-0.5, +0.5] based on overall impression (default 0.0)
  8. final_qoe       = round(weighted_sum + adjustment, 1), clamped to [0.0, 10.0]

ARITHMETIC SELF-CHECK (mandatory):
  - Re-compute weighted_sum a second time independently and confirm both results match.
  - Confirm 0.10 + 0.10 + 0.20 + 0.30 + 0.30 = 1.00.
  - If all five sub_scores are equal to X, then weighted_sum MUST equal X. Verify this case explicitly when applicable.
  - If any term looks inconsistent with sub_score * weight, STOP and recompute.
  - The final_qoe field in the JSON MUST equal the QoE field. They cannot differ.

============================================================
TASK
============================================================

1. For each of the 5 phases:
   a. List ≥2 concrete observations.
   b. List which checklist defects are present.
   c. Assign an explicit numeric sub-score (0–10).
   d. State why the score is NOT 0.5 higher and NOT 0.5 lower.

2. Compute the weighted QoE, then state the final QoE.
3. If any phase is unjudgeable (occlusion, cut, angle), say so and score it at the LOW end of plausible range, not the middle.

============================================================
OUTPUT
============================================================

Return ONLY a JSON object exactly matching the schema below. No prose outside the JSON. No markdown. No code fences.

{
  "Phase_Analysis": {
    "Starting_Position": {
      "observations": ["<​obs1>", "<​obs2>"],
      "defects_present": ["<​defect>", "..."],
      "sub_score": <number 0-10>,
      "why_not_higher": "<​reason>",
      "why_not_lower": "<​reason>"
    },
    "Approach": {
      "observations": ["<​obs1>", "<​obs2>"],
      "defects_present": ["<​defect>", "..."],
      "sub_score": <number 0-10>,
      "why_not_higher": "<​reason>",
      "why_not_lower": "<​reason>"
    },
    "Take_off": {
      "observations": ["<​obs1>", "<​obs2>"],
      "defects_present": ["<​defect>", "..."],
      "sub_score": <number 0-10>,
      "why_not_higher": "<​reason>",
      "why_not_lower": "<​reason>"
    },
    "Flight": {
      "observations": ["<​obs1>", "<​obs2>"],
      "defects_present": ["<​defect>", "..."],
      "sub_score": <number 0-10>,
      "why_not_higher": "<​reason>",
      "why_not_lower": "<​reason>"
    },
    "Entry": {
      "observations": ["<​obs1>", "<​obs2>"],
      "defects_present": ["<​defect>", "..."],
      "sub_score": <number 0-10>,
      "why_not_higher": "<​reason>",
      "why_not_lower": "<​reason>"
    }
  },
  
  "Weighted_Computation": {
    "weights": {
      "Starting_Position": 0.10,
      "Approach": 0.10,
      "Take_off": 0.20,
      "Flight": 0.30,
      "Entry": 0.30
    },
    "weights_sum_check": 1.00,
    "terms": {
      "term_starting":  { "sub_score": <number>, "weight": 0.10, "value": <sub_score*0.10> },
      "term_approach":  { "sub_score": <number>, "weight": 0.10, "value": <sub_score*0.10> },
      "term_take_off":  { "sub_score": <number>, "weight": 0.20, "value": <sub_score*0.20> },
      "term_flight":    { "sub_score": <number>, "weight": 0.30, "value": <sub_score*0.30> },
      "term_entry":     { "sub_score": <number>, "weight": 0.30, "value": <sub_score*0.30> }
    },
    "weighted_sum": <sum of the five term.value>,
    "weighted_sum_recomputed": <independent recomputation, must equal weighted_sum>,
    "adjustment": <number in [-0.5, +0.5]>,
    "adjustment_reason": "<​short reason or 'none'>",
    "final_qoe": <weighted_sum + adjustment, rounded to 1 decimal, clamped 0-10>,
    "equal_subscores_sanity_check": "<​if all five sub_scores are equal, state the value X and confirm weighted_sum == X; otherwise 'N/A'>"
  },
  "QoE": <must equal final_qoe>,
  "QoE_Justification": {
    "Starting_Position": "<​concise summary tied to observations + sub-score>",
    "Approach": "<​concise summary + sub-score>",
    "Take_off": "<​concise summary + sub-score>",
    "Flight": "<​concise summary + sub-score>",
    "Entry": "<​concise summary + sub-score>"
  },
  "Limitations": "<​any phase that could not be clearly judged, or 'None'>",
  "Anti_Bias_Check": {
    "score_is_not_anchored": "<​explain why this specific score was chosen from evidence rather than from a habitual default; cite which observations drove the band selection>",
    "band_justification": "<​name the World Aquatics band the final QoE falls into and confirm the evidence matches that band's definition>",
    "full_scale_used": "<​confirmation that the score reflects observed quality, not a comfort-zone default>"
  }
}
"""
},
{
        "name": "prompt_8",
        "active": True,  
        "text": 
"""
You are a deterministic scoring engine adhering strictly to World Aquatics Rulebook Diving Section. Your output must be purely evidence-based, ignoring all human heuristics or comfort-zone biases.

Analyze the provided 10m platform dive video and output ONLY a valid JSON object. No markdown, no prose, no explanations outside the JSON.
You are FORBIDDEN from leaving "defects_present" empty.

SCORING RULES:
Evidence-Based: Assign sub-scores (0.0-10.0) for each of the 5 phases based strictly on observable defects. Do not anchor to average scores. Use the full 0.0-10.0 scale.
Phases & Weights:
Starting_Position (0.10)
Approach (0.10)
Take_off (0.20)
Flight (0.30)
Entry (0.30)
Calculation:
weighted_sum = Σ(sub_score * weight)
final_qoe = round(weighted_sum, 1). Clamped [0.0, 10.0].
QoE must equal final_qoe.

JSON SCHEMA:
{
  "Phase_Analysis": {
    "Starting_Position": {
      "observations": ["<​fact 1>", "<​fact 2>"],
      "defects_present": ["< defect>"],
      "sub_score": <float 0.0-10.0>
    },
    "Approach": {
      "observations": ["<​fact 1>", "<fact 2>"],
      "defects_present": ["<defect>"],
      "sub_score": <float 0.0-10.0>
    },
    "Take_off": {
      "observations": ["<​fact 1>", "<​fact 2>"],
      "defects_present": ["<defect>"],
      "sub_score": <float 0.0-10.0>
    },
    "Flight": {
      "observations": ["<​fact 1>", "<fact 2>"],
      "defects_present": ["<defect>"],
      "sub_score": <float 0.0-10.0>
    },
    "Entry": {
      "observations": ["<​fact 1>", "<fact 2>"],
      "defects_present": ["<defect>"],
      "sub_score": <float 0.0-10.0>
    }
  },
  "QoE": <must equal final_qoe>,
  "QoE_Justification": {
    "Starting_Position": "<​concise summary tied to observations + sub-score>",
    "Approach": "<​concise summary + sub-score>",
    "Take_off": "<​concise summary + sub-score>",
    "Flight": "<​concise summary + sub-score>",
    "Entry": "<​concise summary + sub-score>"
  }
}
"""
},
{
        "name": "prompt_10",
        "active": False,  
        "text": 
"""
You are a strict professional diving judge following FINA / World Aquatics judging criteria.

Evaluate the dive ONLY from observable evidence in the provided sampled frames.

IMPORTANT RULES:
- Be conservative and evidence-based.
- Do NOT assume perfect execution unless clearly visible.
- Penalize ambiguity, uncertainty, limited visibility, poor sampling coverage, or unclear execution.
- Scores above 8.5 must be rare and only used for clearly excellent dives.
- Scores between 9.0 and 10.0 should only be assigned for near-perfect execution.
- Most completed dives should naturally fall between 5.0 and 8.5 depending on observed defects.
- A visible completed dive must NEVER receive QoE = 0.0.
- Do NOT assign 0.0 unless the dive is completely failed, absent, or impossible to evaluate.
- For visible completed dives, phase sub-scores should normally be between 4.0 and 9.5.
- Every phase MUST contain a defects_present list.
- If no clear defect is visible, use:
  ["minor or unclear defect: no significant defect clearly visible from sampled frames"]
- Never use "None".
- Do NOT copy numeric placeholder values from the JSON schema.
- Replace every placeholder with actual computed values.
- Output ONLY valid JSON.
- No markdown.
- No explanations outside JSON.
- JSON must be syntactically valid.

Evaluate these phases:

1. Starting_Position
2. Approach
3. Take_off
4. Flight
5. Entry

Phase weights:
Starting_Position = 0.10
Approach = 0.10
Take_off = 0.20
Flight = 0.30
Entry = 0.30

Final score calculation:
weighted_sum = Σ(sub_score × weight)
QoE = round(weighted_sum, 1)

JSON FORMAT:
{
  "Phase_Analysis": {
    "Starting_Position": {
      "observations": [],
      "defects_present": [],
      "sub_score": "<actual score between 4.0 and 10.0 for visible completed dives>"
    },
    "Approach": {
      "observations": [],
      "defects_present": [],
      "sub_score": "<actual score between 4.0 and 10.0 for visible completed dives>"
    },
    "Take_off": {
      "observations": [],
      "defects_present": [],
      "sub_score": "<actual score between 4.0 and 10.0 for visible completed dives>"
    },
    "Flight": {
      "observations": [],
      "defects_present": [],
      "sub_score": "<actual score between 4.0 and 10.0 for visible completed dives>"
    },
    "Entry": {
      "observations": [],
      "defects_present": [],
      "sub_score": "<actual score between 4.0 and 10.0 for visible completed dives>"
    }
  },
  "QoE": "<actual weighted final score>",
  "QoE_Justification": {
    "Starting_Position": "",
    "Approach": "",
    "Take_off": "",
    "Flight": "",
    "Entry": ""
  }
}
"""
}
]

# Filtrar solo prompts activos
ACTIVE_PROMPTS = [p for p in PROMPTS if p.get("active", False)]
print(f"Prompts activos: {[p['name'] for p in ACTIVE_PROMPTS]}")

# -------------------- LOAD MODEL --------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    device_map="auto" if device == "cuda" else None,
).eval()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------- FUNCIONES --------------------
def extract_frames(video_path, intervalo_segundos):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    frame_interval = int(fps * intervalo_segundos)

    frames_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(Image.fromarray(frame_rgb))
        frame_count += 1

    cap.release()
    print(f"  Frames extraídos: {len(frames_list)} (fps={fps:.1f}, intervalo={frame_interval})")
    return frames_list


def run_inference(frames_list, prompt_text):
    content_list = [{"type": "image", "image": img} for img in frames_list]
    content_list.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content_list}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames_list, padding=True, return_tensors="pt").to(device)

    t_start = time.time()
    with torch.inference_mode():
        #generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=3072,      # suficiente para el esquema completo del prompt_6
            do_sample=False,          # determinista, evita "divagar" y desperdiciar tokens
            repetition_penalty=1.05,  # evita bucles repetitivos que truncan el JSON útil
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        result = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    t_end = time.time()

    elapsed = round(t_end - t_start, 2)
    clean_result = result.replace("```json", "").replace("```", "").strip()
    return clean_result, elapsed


def json_to_csv_row(video_name, prompt_name, inference_time, json_str):
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "video": video_name,
            "prompt_name": prompt_name,
            "inference_time_s": inference_time,
            "error": "JSON inválido",
            "raw_output": json_str[:300],
        }

    dive_id = data.get("Dive_Identification", {})
    qoe_just = data.get("QoE_Justification", {})
    
    # --- NUEVO: extraer Weighted_Computation.terms ---
    weighted = data.get("Weighted_Computation", {})
    terms = weighted.get("terms", {})

    def _t(key, field):
        return terms.get(key, {}).get(field, "")

    return {
        "video": video_name,
        "prompt_name": prompt_name,
        "inference_time_s": inference_time,
        "Dive_Number": dive_id.get("Dive_Number", ""),
        "Description": dive_id.get("Description", ""),
        "QoE": data.get("QoE", ""),
        "QoE_Starting_Position": qoe_just.get("Starting_Position", ""),
        "QoE_Approach": qoe_just.get("Approach", ""),
        "QoE_Take_off": qoe_just.get("Take_off", ""),
        "QoE_Flight": qoe_just.get("Flight", ""),
        "QoE_Entry": qoe_just.get("Entry", ""),
        
        # sub_scores
        "sub_score_starting": _t("term_starting", "sub_score"),
        "sub_score_approach": _t("term_approach", "sub_score"),
        "sub_score_take_off": _t("term_take_off", "sub_score"),
        "sub_score_flight":   _t("term_flight",   "sub_score"),
        "sub_score_entry":    _t("term_entry",    "sub_score"),
        # weights
        "weight_starting": _t("term_starting", "weight"),
        "weight_approach": _t("term_approach", "weight"),
        "weight_take_off": _t("term_take_off", "weight"),
        "weight_flight":   _t("term_flight",   "weight"),
        "weight_entry":    _t("term_entry",    "weight"),
        # values
        "value_starting": _t("term_starting", "value"),
        "value_approach": _t("term_approach", "value"),
        "value_take_off": _t("term_take_off", "value"),
        "value_flight":   _t("term_flight",   "value"),
        "value_entry":    _t("term_entry",    "value"),
        # agregados
        "weighted_sum": weighted.get("weighted_sum", ""),
        "adjustment":   weighted.get("adjustment", ""),
        "final_qoe":    weighted.get("final_qoe", ""),
        
        "Video_Quality_Note": data.get("Video_Quality_Note", ""),
        "error": "",
        "raw_output": "",
    }


# -------------------- PIPELINE PRINCIPAL --------------------
video_files = sorted([
    f for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith(VIDEO_EXTENSIONS)
])

if not video_files:
    print(f"No se encontraron videos en: {VIDEO_FOLDER}")
    exit()

print(f"Videos encontrados: {len(video_files)}")
print(f"Combinaciones a procesar: {len(video_files)} videos × {len(ACTIVE_PROMPTS)} prompts = {len(video_files) * len(ACTIVE_PROMPTS)} inferencias\n")

csv_fieldnames = [
    "video", "prompt_name", "inference_time_s",
    "Dive_Number", "Description", "QoE",
    "QoE_Starting_Position", "QoE_Approach", "QoE_Take_off", "QoE_Flight", "QoE_Entry",
    # --- NUEVO: sub_scores ---
    "sub_score_starting", "sub_score_approach", "sub_score_take_off",
    "sub_score_flight", "sub_score_entry",
    # --- NUEVO: weights ---
    "weight_starting", "weight_approach", "weight_take_off",
    "weight_flight", "weight_entry",
    # --- NUEVO: values (sub_score * weight) ---
    "value_starting", "value_approach", "value_take_off",
    "value_flight", "value_entry",
    # --- NUEVO: agregados útiles ---
    "weighted_sum", "adjustment", "final_qoe",
    "Video_Quality_Note", "error", "raw_output",
]

csv_rows = []
total = len(video_files) * len(ACTIVE_PROMPTS)
counter = 0

for video_idx, video_file in enumerate(video_files):
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    video_name = os.path.splitext(video_file)[0]

    print(f"\n📹 Video: {video_file}")
    frames = extract_frames(video_path, INTERVALO_SEGUNDOS)
    
    # Obtener DoD del video actual
    dod = float(difficulty_levels[video_idx]) if video_idx < len(difficulty_levels) else None
    dod_str = f"{dod:.1f}" if dod is not None else "unknown"
    print(f"\n📹 Video: {video_file} | DoD: {dod_str}")

    if not frames:
        for prompt in ACTIVE_PROMPTS:
            csv_rows.append({
                "video": video_name, "prompt_name": prompt["name"],
                "inference_time_s": 0, "error": "sin frames"
            })
        continue

    for prompt in ACTIVE_PROMPTS:
        counter += 1
        
        # Inyectar DoD en el prompt
        #dod_context = (
        #    f"\n====\nDegree of Difficulty CONTEXT\n====\n"
        #    f"The official Degree of Difficulty (DoD) based on Official World Aquatics table for this dive is: {dod_str}. "
        #    f"Use this value of ground truth as a reference for estimate Quality of Execution (QoE)."
        #)
        #enriched_prompt = dod_context + prompt["text"]
        
        #print(enriched_prompt)

        #print(f"  [{counter}/{total}] Prompt: {prompt['name']} — inferencia...")
        #evaluacion_json, elapsed = run_inference(frames, enriched_prompt)
        
        print(f"  [{counter}/{total}] Prompt: {prompt['name']} — inferencia...")
        evaluacion_json, elapsed = run_inference(frames, prompt["text"])

        # Guardar JSON individual: <video>_<prompt>.json
        json_filename = f"{video_name}_{prompt['name']}.json"
        json_out_path = os.path.join(OUTPUT_FOLDER, json_filename)
        with open(json_out_path, "w", encoding="utf-8") as f:
            f.write(evaluacion_json)

        print(f"  ✅ Tiempo: {elapsed}s | JSON: {json_filename}")

        row = json_to_csv_row(video_name, prompt["name"], elapsed, evaluacion_json)
        csv_rows.append(row)

# -------------------- GUARDAR CSV --------------------
with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames, extrasaction="ignore", delimiter=";")
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\n✅ Proceso terminado.")
print(f"   JSONs individuales en: {OUTPUT_FOLDER}")
print(f"   CSV consolidado en:    {CSV_OUTPUT}")