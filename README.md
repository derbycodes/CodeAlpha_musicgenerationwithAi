## AI-Driven Music Generation System

An artificial intelligence project built in Python that autonomously generates original musical sequences using Deep Learning (LSTM networks) and symbolic music processing (`music21`). 

The model parses polyphonic MIDI compositions, learns sequential melodic patterns and harmonic structures, and outputs new, coherent musical pieces exported directly back to MIDI format.

---

## Features

- **Symbolic Music Parsing:** Extracts notes, chords, pitch values, and rests from multi-track MIDI files using `music21`.
- **Sequential Pattern Learning:** Leverages a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers to learn long-range temporal dependencies in musical phrasing.
- **Polyphonic & Monophonic Support:** Handles single melodic lines as well as multi-note chord structures.
- **MIDI Synthesis & Export:** Converts raw model-generated pitch indices back into structured MIDI streams for playback in any standard DAW or media player.

---

## Tech Stack & Dependencies

- **Language:** Python 3.x
- **Core Libraries:**
  - `music21` – MIDI parsing, symbolic music data manipulation, and stream generation
  - `numpy` – Vectorized numerical operations and data reshaping
  - `tensorflow` / `keras` (or `torch`) – Neural network architecture, LSTM layers, and training pipeline
  - `glob` & `pickle` – Dataset handling and trained weight serialization

---

## Project Structure

```text
├── midi_songs/             # Raw MIDI training files (.mid)
├── Music generation_ai.py  # Data extraction, preprocessing, model architecture & training
├── data/                   # Serialized note sequences (pickle files)
├── weights/                # Checkpointed model weights (.h5 / .keras / .pt)
├── generated_output/       # AI-composed output MIDI files
└── README.md
