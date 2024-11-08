import os
import random
import pretty_midi
from midi2audio import FluidSynth
import csv
from datetime import datetime
import time
import uuid
import json
from pydub import AudioSegment

# Configuration
OUTPUT_DIR = "output"  # Directory to save MIDI, WAV, and CSV files
DEFAULT_SOUND_FONT_PATH = "./Fluid_related/FluidR3_GM.sf2"  # Update this path to your SoundFont
NUM_FILES_PER_BATCH = 5  # Number of MIDI-WAV-CSV sets to generate per batch
NUM_BATCHES = 3  # Number of batches to generate for diversity

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define musical scales
SCALES = {
    'C_major': [60, 62, 64, 65, 67, 69, 71, 72],
    'A_minor': [57, 59, 60, 62, 64, 65, 67, 69],
    'G_major': [67, 69, 71, 72, 74, 76, 78, 79],
    'E_minor': [64, 66, 67, 69, 71, 72, 74, 76],
    'D_major': [62, 64, 66, 67, 69, 71, 73, 74],
    'B_minor': [59, 61, 62, 64, 66, 67, 69, 71],
    'F_major': [65, 67, 69, 70, 72, 74, 76, 77],
    'D_minor': [62, 64, 65, 67, 69, 70, 72, 74],
    'A_major': [69, 71, 73, 74, 76, 78, 80, 81],
    'F_minor': [65, 67, 68, 70, 72, 73, 75, 77],
    # Add more scales as needed
}

CHORD_PROGRESSIONS = [
    ['C', 'F', 'G', 'C'],
    ['Am', 'Dm', 'G', 'C'],
    ['G', 'C', 'D', 'G'],
    ['Em', 'C', 'G', 'D'],
    ['D', 'G', 'A', 'D'],
    ['Am', 'G', 'F', 'C'],
    ['F', 'G', 'Am', 'F'],
    ['A', 'D', 'E', 'A'],
    ['Bm', 'G', 'D', 'A'],
    ['Em', 'Am', 'D', 'G'],
    # Add more progressions as needed
]

CHORDS = {
    'C': [60, 64, 67],     # C major
    'F': [65, 69, 72],     # F major
    'G': [67, 71, 74],     # G major
    'Am': [57, 60, 64],    # A minor
    'Dm': [62, 65, 69],    # D minor
    'Em': [64, 67, 71],    # E minor
    'D': [62, 66, 69],     # D major
    'A': [69, 73, 76],     # A major
    'Bm': [59, 62, 66],    # B minor
    'E': [64, 68, 71],     # E major
    'F#m': [66, 69, 73],   # F# minor
    'Bb': [70, 74, 77],    # B-flat major
    'Cm': [60, 63, 67],    # C minor
    'Fm': [65, 68, 72],    # F minor
    # Add more chords as needed
}

# Define available instruments
INSTRUMENTS = [
    'Acoustic Grand Piano',
    'Violin',
    'Flute',
    'Electric Guitar (jazz)'
    # 'Synth Lead',
    # Add more instruments as needed
]

def generate_unique_timestamp():
    """
    Generates a unique timestamp string with microsecond precision.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def get_wav_length(wav_path):
    """
    Returns the duration of the WAV file in seconds.
    """
    try:
        audio = AudioSegment.from_wav(wav_path)
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except Exception as e:
        print(f"Error reading WAV file {wav_path}: {e}")
        return None

def generate_random_midi(
    file_path,
    csv_path,
    enable_scale=False,
    scale_name='C_major',
    enable_chord_progression=False,
    progression=None,
    enable_multiple_instruments=False,
    instrument_list=None,
    enable_phrases=False,
    phrase_length_range=(4, 8),
    enable_augmentation=False,
    transpose_range=(-12, 12),
    enable_overlap=False,
    note_range=(60, 72),
    duration_range=(0.3, 1.0),
    velocity_range=(80, 120),
    num_notes_range=(20, 50),
    chord_probability=0.3,
    max_notes_per_chord=3,
    max_wav_length=10
):
    """
    Generates a random MIDI file with enhanced musical features and logs the note details in a CSV file.

    Parameters:
    - file_path (str): The output file path for the generated MIDI.
    - csv_path (str): The output file path for the CSV recording the notes.
    - enable_scale (bool): If True, generate notes based on a specific musical scale.
    - scale_name (str): The name of the scale to use (e.g., 'C_major').
    - enable_chord_progression (bool): If True, use predefined chord progressions.
    - progression (list): The chord progression to use. If None, a random progression is selected.
    - enable_multiple_instruments (bool): If True, use multiple instruments.
    - instrument_list (list): List of instrument names to choose from. If None, defaults to predefined INSTRUMENTS.
    - enable_phrases (bool): If True, organize notes into musical phrases.
    - phrase_length_range (tuple): Range of phrases lengths in number of chords or measures.
    - enable_augmentation (bool): If True, apply data augmentation techniques like transposition.
    - transpose_range (tuple): Range for transposing MIDI files in semitones.
    - enable_overlap (bool): If True, allows overlapping of notes and chords.
    - note_range (tuple): Tuple of (min_note, max_note) for random note pitches.
    - duration_range (tuple): Tuple of (min_duration, max_duration) for random note durations.
    - velocity_range (tuple): Tuple of (min_velocity, max_velocity) for note velocities.
    - num_notes_range (tuple): Tuple of (min_notes, max_notes) for number of notes to generate.
    - chord_probability (float): Probability of generating a chord instead of a single note.
    - max_notes_per_chord (int): Maximum number of notes in a chord.
    - max_wav_length (float): Maximum length of the generated audio in seconds. If None, no limit is applied.
    """

    # Initialize PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()

    # Select instrument(s)
    if enable_multiple_instruments and instrument_list:
        instruments = []
        for inst_name in instrument_list:
            program = pretty_midi.instrument_name_to_program(inst_name)
            instrument = pretty_midi.Instrument(program=program, name=inst_name)
            instruments.append(instrument)
    else:
        # Default to Acoustic Grand Piano
        program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        instruments = [pretty_midi.Instrument(program=program, name='Acoustic Grand Piano')]

    # Prepare CSV logging
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Note_Type', 'Instrument', 'Pitch', 'Start_Time', 'Duration', 'Velocity'])

        # Select scale
        if enable_scale and scale_name in SCALES:
            scale_pitches = SCALES[scale_name]
        else:
            # Default MIDI note range
            scale_pitches = list(range(note_range[0], note_range[1] + 1))

        # Select chord progression
        if enable_chord_progression:
            if not progression:
                progression = random.choice(CHORD_PROGRESSIONS)
        else:
            progression = None

        # Handle phrases
        if enable_phrases and progression:
            phrase_length = random.randint(*phrase_length_range)
            phrases = [progression[i:i + phrase_length] for i in range(0, len(progression), phrase_length)]
        else:
            phrases = [progression] if progression else [None]

        # Initialize start time
        start_time = 0

        # Initialize note counter
        total_notes = 0
        max_notes = random.randint(*num_notes_range)

        # Iterate through phrases
        for phrase in phrases:
            if phrase:
                for chord_name in phrase:
                    if chord_name in CHORDS:
                        chord_pitches = CHORDS[chord_name]
                    else:
                        # If chord not defined, skip
                        continue

                    # Assign instrument randomly if multiple instruments are enabled
                    if enable_multiple_instruments:
                        instrument = random.choice(instruments)
                    else:
                        instrument = instruments[0]

                    # Create chord
                    chord_notes = []
                    for pitch in chord_pitches:
                        if enable_scale:
                            # Ensure pitch is within scale
                            if pitch not in scale_pitches:
                                continue
                        else:
                            # Random pitch within range
                            pitch = random.randint(*note_range)

                        duration = random.uniform(*duration_range)

                        # Check if adding this note would exceed the max_wav_length
                        if max_wav_length and (start_time + duration) > max_wav_length:
                            break  # Stop generation if max length is reached

                        velocity = random.randint(*velocity_range)

                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_time,
                            end=start_time + duration
                        )
                        chord_notes.append(note)

                        # Log to CSV
                        csv_writer.writerow(['Chord', instrument.name, pitch, start_time, duration, velocity])

                        # Update total notes
                        total_notes += 1
                        if total_notes >= max_notes:
                            break

                    if chord_notes:
                        # Add chord to instrument
                        instrument.notes.extend(chord_notes)

                        # Update start time
                        if enable_overlap:
                            start_time += random.uniform(0, duration)  # Allow overlap
                        else:
                            start_time += duration

                    # Check if maximum number of notes reached
                    if total_notes >= max_notes:
                        break
            else:
                # No chord progression; generate single notes or chords based on probability
                while total_notes < max_notes:
                    if enable_multiple_instruments:
                        instrument = random.choice(instruments)
                    else:
                        instrument = instruments[0]

                    if random.random() < chord_probability and (total_notes + max_notes_per_chord) <= max_notes:
                        # Generate a chord
                        num_notes_in_chord = random.randint(2, max_notes_per_chord)
                        chord_notes = []
                        for _ in range(num_notes_in_chord):
                            pitch = random.choice(scale_pitches) if enable_scale else random.randint(*note_range)
                            duration = random.uniform(*duration_range)

                            # Check if adding this note would exceed the max_wav_length
                            if max_wav_length and (start_time + duration) > max_wav_length:
                                print("Max length reached.*****************>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                                break  # Stop generation if max length is reached

                            velocity = random.randint(*velocity_range)

                            note = pretty_midi.Note(
                                velocity=velocity,
                                pitch=pitch,
                                start=start_time,
                                end=start_time + duration
                            )
                            chord_notes.append(note)

                            # Log to CSV
                            csv_writer.writerow(['Chord', instrument.name, pitch, start_time, duration, velocity])

                            # Update total notes
                            if total_notes >= max_notes:
                                break

                        if chord_notes:
                            # Add chord to instrument
                            instrument.notes.extend(chord_notes)

                            # Update start time
                            if enable_overlap:
                                start_time += random.uniform(0, duration)  # Allow overlap
                            else:
                                start_time += duration
                        total_notes += 1
                    else:
                        # Generate a single note
                        pitch = random.choice(scale_pitches) if enable_scale else random.randint(*note_range)
                        duration = random.uniform(*duration_range)

                        # Check if adding this note would exceed the max_wav_length
                        if max_wav_length and (start_time + duration) > max_wav_length:
                                print("Max length reached.*****************>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                                break  # Stop generation if max length is reached

                        velocity = random.randint(*velocity_range)

                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_time,
                            end=start_time + duration
                        )
                        instrument.notes.append(note)

                        # Log to CSV
                        csv_writer.writerow(['Single', instrument.name, pitch, start_time, duration, velocity])

                        # Update start time
                        if enable_overlap:
                            start_time += random.uniform(0, duration)
                        else:
                            start_time += duration

                        # Update total notes
                        total_notes += 1

    # Add instruments to MIDI object
    midi.instruments.extend(instruments)

    # Write MIDI file
    midi.write(file_path)


def midi_to_wav(midi_path, wav_path, sound_font):
    """
    Converts a MIDI file to WAV using FluidSynth.

    Parameters:
    - midi_path (str): Path to the input MIDI file.
    - wav_path (str): Path to save the output WAV file.
    - sound_font (str): Path to the SoundFont (.sf2) file used for synthesis.
    """
    fs = FluidSynth(sound_font=sound_font)
    fs.midi_to_audio(midi_path, wav_path)

def initialize_metadata(metadata_path):
    """
    Initializes the metadata CSV file with headers.

    Parameters:
    - metadata_path (str): Path to the metadata CSV file.
    """
    headers = [
        'File_ID',
        'Number_of_Notes',
        'WAV_Length_Seconds',
        'Enable_Scale',
        'Scale_Name',
        'Enable_Chord_Progression',
        'Enable_Multiple_Instruments',
        'Enable_Phrases',
        'Phrase_Length_Range',
        'Enable_Augmentation',
        'Transpose_Range',
        'Enable_Overlap',
        'Note_Range',
        'Duration_Range',
        'Velocity_Range',
        'Num_Notes_Range',
        'Chord_Probability',
        'Max_Notes_Per_Chord'
    ]
    with open(metadata_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

def append_metadata(metadata_path, metadata_entry):
    """
    Appends a single metadata entry to the metadata CSV file.

    Parameters:
    - metadata_path (str): Path to the metadata CSV file.
    - metadata_entry (dict): Dictionary containing metadata fields.
    """
    with open(metadata_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            metadata_entry['File_ID'],
            metadata_entry['Number_of_Notes'],
            metadata_entry['WAV_Length_Seconds'],
            metadata_entry['Enable_Scale'],
            metadata_entry['Scale_Name'],
            metadata_entry['Enable_Chord_Progression'],
            metadata_entry['Enable_Multiple_Instruments'],
            metadata_entry['Enable_Phrases'],
            metadata_entry['Phrase_Length_Range'],
            metadata_entry['Enable_Augmentation'],
            metadata_entry['Transpose_Range'],
            metadata_entry['Enable_Overlap'],
            metadata_entry['Note_Range'],
            metadata_entry['Duration_Range'],
            metadata_entry['Velocity_Range'],
            metadata_entry['Num_Notes_Range'],
            metadata_entry['Chord_Probability'],
            metadata_entry['Max_Notes_Per_Chord']
        ])

def main(
    num_batches=NUM_BATCHES,
    num_files_per_batch=NUM_FILES_PER_BATCH,
    output_dir=OUTPUT_DIR,
    sound_font_path=DEFAULT_SOUND_FONT_PATH,
    custom_instrument_list=None
):
    """
    Main function to generate MIDI-WAV-CSV pairs with enhanced musical features across multiple batches.

    Parameters:
    - num_batches (int): Number of batches to generate for diversity.
    - num_files_per_batch (int): Number of file sets to generate per batch.
    - output_dir (str): Directory to save the generated files.
    - sound_font_path (str): Path to the SoundFont (.sf2) file.
    - custom_instrument_list (list): List of instrument names to use. If None, defaults are used.
    """
    metadata_path = os.path.join(output_dir, "metadata.csv")
    initialize_metadata(metadata_path)

    for batch_num in range(1, num_batches + 1):
        print(f"Starting batch {batch_num} of {num_batches}...")
        
        # Randomize generation parameters for diversity in each batch
        enable_scale = random.choice([True, False])
        scale_name = random.choice(list(SCALES.keys())) if enable_scale else ''
        enable_chord_progression = random.choice([True, False])
        enable_multiple_instruments = random.choice([True, False])
        enable_phrases = random.choice([True, False]) if enable_chord_progression else False
        phrase_length_range = (random.randint(3,5), random.randint(6,8)) if enable_phrases else (4,8)
        enable_augmentation = random.choice([True, False])
        transpose_range = (random.randint(-12, -6), random.randint(6, 12)) if enable_augmentation else (0,0)
        enable_overlap = random.choice([True, False])
        note_range = (random.randint(60, 64), random.randint(68, 72))
        min_duration_lower = 0.2
        min_duration_upper = 0.5
        max_duration_lower = 0.6
        max_duration_upper = 1.2

        # Generate the lower and upper bounds
        duration_lower_bound = round(random.uniform(min_duration_lower, min_duration_upper), 2)
        duration_upper_bound = round(random.uniform(max(duration_lower_bound, max_duration_lower), max_duration_upper), 2)

        # Ensure that the upper bound is greater than the lower bound
        if duration_upper_bound <= duration_lower_bound:
            duration_upper_bound = duration_lower_bound + 0.1  # Add a small value to ensure upper bound is greater

        duration_range = (duration_lower_bound, duration_upper_bound)
        velocity_range = (random.randint(70, 90), random.randint(100, 120))
        num_notes_range = (random.randint(15, 25), random.randint(30, 50))
        chord_probability = round(random.uniform(0.2, 0.5), 2)
        max_notes_per_chord = random.randint(2,4)

        # Define a unique identifier for the batch (optional)
        batch_id = str(uuid.uuid4())

        for i in range(1, num_files_per_batch + 1):
            # Generate a unique timestamp for filenames
            timestamp = generate_unique_timestamp()
            # Ensure uniqueness by waiting for microsecond changes
            time.sleep(0.001)  # Sleep for 1ms

            file_id = timestamp  # Using timestamp as File_ID

            # Define filenames based on timestamp
            midi_filename = f"{file_id}.mid"
            wav_filename = f"{file_id}.wav"
            csv_filename = f"{file_id}.csv"

            midi_path = os.path.join(output_dir, midi_filename)
            wav_path = os.path.join(output_dir, wav_filename)
            csv_path = os.path.join(output_dir, csv_filename)

            print(f"Generating file {i} of batch {batch_num}: {midi_filename}...")

            # Generate MIDI and CSV
            generate_random_midi(
                file_path=midi_path,
                csv_path=csv_path,
                enable_scale=enable_scale,
                scale_name=scale_name,
                enable_chord_progression=enable_chord_progression,
                progression=None,  # Let the function select a random progression
                enable_multiple_instruments=enable_multiple_instruments,
                instrument_list=custom_instrument_list if custom_instrument_list else INSTRUMENTS,
                enable_phrases=enable_phrases,
                phrase_length_range=phrase_length_range,
                enable_augmentation=enable_augmentation,
                transpose_range=transpose_range,
                enable_overlap=enable_overlap,
                # You can randomize or parameterize other settings as needed
            )

            print(f"Converting {midi_filename} to {wav_filename}...")
            # Convert MIDI to WAV
            midi_to_wav(midi_path, wav_path, sound_font_path)

            # Get number of notes from CSV (assuming the CSV has been written correctly)
            num_notes = 0
            with open(csv_path, mode='r') as csv_file_read:
                reader = csv.DictReader(csv_file_read)
                for _ in reader:
                    num_notes += 1

            # Get WAV length
            wav_length = get_wav_length(wav_path)

            # Prepare metadata entry
            metadata_entry = {
                'File_ID': file_id,
                'Number_of_Notes': num_notes,
                'WAV_Length_Seconds': wav_length if wav_length else '',
                'Enable_Scale': enable_scale,
                'Scale_Name': scale_name,
                'Enable_Chord_Progression': enable_chord_progression,
                'Enable_Multiple_Instruments': enable_multiple_instruments,
                'Enable_Phrases': enable_phrases,
                'Phrase_Length_Range': f"{phrase_length_range}",
                'Enable_Augmentation': enable_augmentation,
                'Transpose_Range': f"{transpose_range}",
                'Enable_Overlap': enable_overlap,
                'Note_Range': f"{note_range}",
                'Duration_Range': f"{duration_range}",
                'Velocity_Range': f"{velocity_range}",
                'Num_Notes_Range': f"{num_notes_range}",
                'Chord_Probability': chord_probability,
                'Max_Notes_Per_Chord': max_notes_per_chord
            }

            # Append metadata to metadata.csv
            append_metadata(metadata_path, metadata_entry)

            print(f"Pair {i} generated: {midi_filename}, {wav_filename}, {csv_filename}\n")

        print(f"Batch {batch_num} completed.\n")

    print("All MIDI-WAV-CSV pairs generated successfully.")

if __name__ == "__main__":
    # Example usage: Define global variables and run main
    NUM_BATCHES = 100  # Number of batches to generate
    NUM_FILES_PER_BATCH = 50  # Number of files per batch
    OUTPUT_DIR = "./output"
    SOUND_FONT_PATH = "FluidR3_GM.sf2"  # Update this path

    # Optionally, define a custom list of instruments
    CUSTOM_INSTRUMENTS = [
        'Acoustic Grand Piano',
        'Violin',
        'Flute',
        'Electric Guitar (jazz)'
        # 'Synth Lead'
    ]

    # Run the main function with desired flags
    main(
        num_batches=NUM_BATCHES,
        num_files_per_batch=NUM_FILES_PER_BATCH,
        output_dir=OUTPUT_DIR,
        sound_font_path=SOUND_FONT_PATH,
        custom_instrument_list=CUSTOM_INSTRUMENTS
    )
