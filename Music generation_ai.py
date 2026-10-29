import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. DATA PREPROCESSING (using music21)
# ==========================================
def extract_notes_from_midi(midi_folder="midi_songs"):

    """
    Parses all MIDI files in the folder and extracts note pitches and chords.
    """
    notes = []
    midi_files = glob.glob(f"{midi_folder}/*.mid*")

    if not midi_files:
        print(f"No MIDI files found in '{midi_folder}' directory. Please add some .mid files!")
        return notes

    print(f"Found {len(midi_files)} files. Parsing MIDI data...")

    for file in midi_files:
        try:
            midi = converter.parse(file)
            print(f"Parsing: {file}")

            # Isolate instruments or flat stream
            try:
                parts = instrument.partitionByInstrument(midi)
                notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            except Exception:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    # Join note pitches with periods (e.g., 'C4.E4.G4')
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Could not parse {file}: {e}")

    # Save parsed notes for quick reloading
    with open('notes.pkl', 'wb') as f:
        pickle.dump(notes, f)

    return notes


# ==========================================
# 2. PREPARE SEQUENCES FOR LSTM
# ==========================================
def prepare_sequences(notes, n_vocab, sequence_length=32):
    """
    Converts note strings into numerical integer sequences for LSTM training.
    """
    # Sort and map unique pitches to integers
    pitchnames = sorted(list(set(notes)))
    note_to_int = {note: num for num, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]

        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    if n_patterns == 0:
        raise ValueError("Dataset is too small! Add more MIDI files or reduce sequence_length.")

    # Reshape input to [samples, time steps, features] and normalize
    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(n_vocab)

    # One-hot encode the outputs
    y = to_categorical(network_output, num_classes=n_vocab)

    return X, y, pitchnames


# ==========================================
# 3. BUILD THE LSTM MODEL ARCHITECTURE
# ==========================================
def create_network(network_input, n_vocab):
    """
    Defines a stacked LSTM model with Dropout to prevent overfitting.
    """
    model = Sequential([
        LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(n_vocab, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# ==========================================
# 4. GENERATE NEW MUSIC SEQUENCES
# ==========================================
def generate_notes(model, network_input, pitchnames, n_vocab, generate_length=100):
    """
    Uses the trained model to predict a new sequence of notes given a random seed.
    """
    int_to_note = {num: note for num, note in enumerate(pitchnames)}

    # Pick a random starting sequence from the training data as a seed
    start_index = np.random.randint(0, len(network_input) - 1)
    pattern = list(network_input[start_index])

    prediction_output = []

    print("Generating new music notes...")
    for _ in range(generate_length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        # Select note with highest probability
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        # Slide window forward by appending newly generated note
        pattern.append([index])
        pattern = pattern[1:]

    return prediction_output


# ==========================================
# 5. CONVERT SEQUENCES BACK TO MIDI FILE
# ==========================================
def create_midi(prediction_output, output_filename="generated_ai_output.mid"):
    """
    Converts note/chord strings back into a playable MIDI file.
    """
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # Pattern is a chord (contains dots or digits)
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a single note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Advance timeline for the next note (0.5 = eighth note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filename)
    print(f"\nSuccess! Generated MIDI saved as: {output_filename}")


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # 1. Parse Data
    notes = extract_notes_from_midi("midi_songs")

    if len(notes) > 35:
        n_vocab = len(set(notes))
        print(f"Total notes parsed: {len(notes)} | Unique pitches/vocab: {n_vocab}")

        # 2. Prepare Sequences
        X, y, pitchnames = prepare_sequences(notes, n_vocab, sequence_length=32)

        # 3. Build Model
        model = create_network(X, n_vocab)
        model.summary()

        # 4. Train the Model
        # Tip: Set epochs=50 to 100 for better musical coherence; 10 to 20 for a quick demo test
        print("\nStarting Training...")
        model.fit(X, y, epochs=15, batch_size=64)

        # 5. Generate and Export MIDI
        generated_notes = generate_notes(model, X, pitchnames, n_vocab, generate_length=100)
        create_midi(generated_notes, "generated_ai_output.mid")
    else:
        print("Please place at least a few MIDI files in 'midi_songs/' to run training.")