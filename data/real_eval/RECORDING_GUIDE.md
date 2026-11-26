# Real-Recording Benchmark Guide (`data/real_eval/`)

This folder holds **real-world acoustic recordings** (captured with a phone or laptop microphone in a real room) used to benchmark **EchoCount** separately from synthetic dataset mixtures (`FSD_Mix_*` and `Org_Mix_*`).

> **Why this matters:** Pre-rendered dataset mixtures are clean and in-distribution. Real room recordings include room reverberation, ambient HVAC noise, microphone colouration, and natural overlap. Always report metrics on `data/real_eval/` separately from synthetic benchmarks.

---

## 1. Target Recording Plan (20–30 Clips, 5–10 Seconds Each)

Record **16 kHz or 44.1/48 kHz mono/stereo WAV, M4A, or MP3** clips (EchoCount automatically resamples to 16 kHz mono).

### Part A — Isolated Single-Source Clips (`true_k = 1`, 12 clips)
Record 1 clip (`6–8 s`) of each of the 12 core everyday sounds alone in a quiet room:

| Suggested Filename | `true_k` | `true_labels` (Friendly Name) | `true_groups` | How to Record |
| :--- | :---: | :--- | :--- | :--- |
| `solo_hammer.wav` | `1` | `["Hammer"]` | `["Household/tools"]` | Tap a hammer or heavy tool on a wooden block 4–6 times |
| `solo_phone_vibration.wav` | `1` | `["Phone vibration"]` | `["Household/tools"]` | Set a phone to vibrate on a hard desk or table |
| `solo_dog_bark.wav` | `1` | `["Dog bark"]` | `["Animal"]` | Real dog barking (outdoor or indoor) |
| `solo_cat_meow.wav` | `1` | `["Cat meow"]` | `["Animal"]` | Real cat meowing |
| `solo_cello.wav` | `1` | `["Cello"]` | `["Musical instrument"]` | Acoustic cello or bowed string instrument |
| `solo_scissors.wav` | `1` | `["Scissors"]` | `["Household/tools"]` | Snip metal scissors in the air or cutting paper |
| `solo_telephone_ring.wav` | `1` | `["Telephone ring"]` | `["Household/tools"]` | Landline bell or classic phone ringtone |
| `solo_speech.wav` | `1` | `["Human voice"]` | `["Human voice"]` | Single person speaking naturally at 0.5–1.5 m |
| `solo_door_knock.wav` | `1` | `["Door knock"]` | `["Household/tools"]` | Knock on a wooden door 3–5 times |
| `solo_keyboard_typing.wav` | `1` | `["Keyboard typing"]` | `["Household/tools"]` | Type continuously on a mechanical or laptop keyboard |
| `solo_water.wav` | `1` | `["Water"]` | `["Environment/other"]` | Run a kitchen/bathroom faucet or pour water into a glass |
| `solo_car_horn.wav` | `1` | `["Car horn"]` | `["Environment/other"]` | Street recording with a car horn honking |

---

### Part B — 2-Source Real Simultaneous Mixtures (`true_k = 2`, 8–10 clips)
Record 2 sounds happening in the **same room at the same time** (not digitally summed):

| Suggested Filename | `true_k` | `true_labels` | `true_groups` | Scenario |
| :--- | :---: | :--- | :--- | :--- |
| `two_teachers_speaking.wav` | `2` | `["Human 1", "Human 2"]` | `["Human voice"]` | Two people speaking simultaneously |
| `speech_and_keyboard.wav` | `2` | `["Human voice", "Keyboard typing"]` | `["Human voice", "Household/tools"]` | Person talking while typing on a keyboard |
| `speech_and_door_knock.wav` | `2` | `["Human voice", "Door knock"]` | `["Human voice", "Household/tools"]` | Person talking while someone knocks on a door |
| `speech_and_phone_vibration.wav` | `2` | `["Human voice", "Phone vibration"]` | `["Human voice", "Household/tools"]` | Person talking while a phone vibrates on the table |
| `water_and_scissors.wav` | `2` | `["Water", "Scissors"]` | `["Environment/other", "Household/tools"]` | Running tap water while snipping scissors |
| `hammer_and_speech.wav` | `2` | `["Hammer", "Human voice"]` | `["Household/tools", "Human voice"]` | Hammering wood while a person speaks |
| `dog_and_speech.wav` | `2` | `["Dog bark", "Human voice"]` | `["Animal", "Human voice"]` | Dog barking while a person speaks |
| `telephone_and_keyboard.wav` | `2` | `["Telephone ring", "Keyboard typing"]` | `["Household/tools"]` | Phone ringing while typing on a keyboard |

---

### Part C — 3-Source Real Simultaneous Mixtures (`true_k = 3`, 5–8 clips)
Record 3 distinct sounds occurring in the same acoustic space:

| Suggested Filename | `true_k` | `true_labels` | `true_groups` | Scenario |
| :--- | :---: | :--- | :--- | :--- |
| `speech_keyboard_phone_vib.wav` | `3` | `["Human voice", "Keyboard typing", "Phone vibration"]` | `["Human voice", "Household/tools"]` | Office desk scene: speaking + typing + vibrating phone |
| `hammer_water_speech.wav` | `3` | `["Hammer", "Water", "Human voice"]` | `["Household/tools", "Environment/other", "Human voice"]` | Workshop/kitchen scene: tap running + hammering + voice |
| `dog_door_knock_speech.wav` | `3` | `["Dog bark", "Door knock", "Human voice"]` | `["Animal", "Household/tools", "Human voice"]` | Front door scene: knocking + dog barking + voice |
| `cello_scissors_telephone.wav` | `3` | `["Cello", "Scissors", "Telephone ring"]` | `["Musical instrument", "Household/tools"]` | Real-room counterpart to `mix_00035.wav` |

---

## 2. Manifest Format (`data/real_eval/manifest.json`)

Add each recorded WAV file to `data/real_eval/` and register an entry under `"clips"` in `data/real_eval/manifest.json`:

```json
{
  "file": "speech_and_keyboard.wav",
  "true_k": 2,
  "true_groups": ["Human voice", "Household/tools"],
  "true_labels": ["Human voice", "Keyboard typing"],
  "source_type": "real_room_microphone",
  "overlap_type": "simultaneous",
  "notes": "Recorded on smartphone mic at 1m distance in office"
}
```

## 3. Running the Real-Recording Benchmark

Once your clips are in `data/real_eval/`, run:

```powershell
cd C:\coding\SANADA\Sanada
.\.venv\Scripts\python evaluate_real_eval.py
```

This evaluates and writes `evaluation/real_eval_results.json` with:
1. **Counting accuracy** (Exact accuracy, Within-±1 accuracy, MAE)
2. **Open-vocabulary tagging F1 & Precision/Recall** against `true_labels` and `true_groups`
3. **Separation self-consistency pass rate & stem correlation gates**
