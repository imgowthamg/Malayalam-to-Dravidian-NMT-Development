import subprocess
import sentencepiece as spm
import tempfile
import os
import csv
import time

# ---------------- Configuration ---------------- #

# Paths to SentencePiece models
SOURCE_SP_MODEL = r"./sourceM.model"    # Update path
TARGET_SP_MODEL = r"./targetM.model"    # Update path

# Path to your OpenNMT model
MODEL_PATH = "tamil_to_malay.pt"         # Update to your model file

# Translation command configuration
TRANSLATE_COMMAND = "onmt_translate"
GPU = "-1"
MIN_LENGTH = "1"

# Input/Output files
INPUT_FILE = "/home/icfoss/Tamil-Malayalm/sample_tamil.tm"         # Update to your actual input file
OUTPUT_CSV = "parallel_corpus.csv"  # Final output file

# ---------------- Helper Functions ---------------- #

def subword_text(text: str, sp_model_path: str) -> str:
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    tokens = sp.encode_as_pieces(text.strip())
    return " ".join(tokens)

def desubword_text(text: str, sp_model_path: str) -> str:
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    tokens = text.strip().split()
    return sp.decode_pieces(tokens)

# ---------------- Batch Processing ---------------- #

def process_batch(input_file: str, output_csv: str):
    with open(input_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    subworded_lines = [subword_text(line, SOURCE_SP_MODEL) for line in lines]

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as src_file:
        src_file.write("\n".join(subworded_lines) + "\n")
        src_file_path = src_file.name

    output_file = tempfile.NamedTemporaryFile(delete=False)
    output_file_path = output_file.name
    output_file.close()

    command = [
        TRANSLATE_COMMAND,
        "-model", MODEL_PATH,
        "-src", src_file_path,
        "-output", output_file_path,
        "-gpu", GPU,
        "-min_length", MIN_LENGTH,
        "-replace_unk",
        "-verbose"
    ]

    start_time = time.time()

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stderr.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print(f"Error during translation:\n{e.stderr.decode('utf-8')}")
        os.remove(src_file_path)
        os.remove(output_file_path)
        return

    elapsed = time.time() - start_time
    print(f"Translation completed in {elapsed:.2f} seconds.")

    with open(output_file_path, encoding="utf-8") as pred_file:
        subworded_translations = [line.strip() for line in pred_file.readlines()]

    desubworded_translations = [
        desubword_text(line, TARGET_SP_MODEL) for line in subworded_translations
    ]

    with open(output_csv, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Source", "Target"])
        for src, tgt in zip(lines, desubworded_translations):
            writer.writerow([src, tgt])

    os.remove(src_file_path)
    os.remove(output_file_path)

    print(f"Saved parallel corpus to '{output_csv}'")


if __name__ == "__main__":
    process_batch(INPUT_FILE, OUTPUT_CSV)
    print(f"✅ All done! Check '{OUTPUT_CSV}' for results.")
