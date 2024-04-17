import os
from tqdm import tqdm

# https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database
sentences_txt_path = os.path.join(r"C:\Users\priya\OneDrive\Desktop\PYTHONLESSONS-ML\mltu\Datasets", "metadata","sentences.txt")
sentences_folder_path = os.path.join(r"C:\Users\priya\OneDrive\Desktop\PYTHONLESSONS-ML\mltu\Datasets", "IAM_Sentences")

dataset, vocab, max_len = [], set(), 0
words = open(sentences_txt_path, "r").readlines()
for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.split(" ")
    if line_split[2] == "err":
        continue

    # folder1 = line_split[0][:3]
    # folder2 = "-".join(line_split[0].split("-")[:2])
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip("\n")

    label = label.replace("|", " ")

    rel_path = os.path.join(sentences_folder_path,file_name)
    if not os.path.exists(rel_path):
        print(f"File not found: {rel_path}")
        continue

    dataset.append([rel_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))
