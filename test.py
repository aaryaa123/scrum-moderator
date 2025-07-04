import pandas as pd
import ast
from datasets import load_dataset
import random

# Load the MOM-Summary-Dataset
mom_dataset = load_dataset("sasvata/MOM-Summary-Dataset")
mom_meetings = mom_dataset['train']

lines = []
N = len(mom_meetings)
for i in range(N):
    mt_raw = mom_meetings[i]['Meeting Transcript']
    try:
        mt_dict = ast.literal_eval(mt_raw)
        transcript = mt_dict.get('transcript', [])
        lines.extend(transcript)
    except Exception as e:
        print(f"Error parsing row {i}: {e}")

print("Random sample of 20 lines:")
for line in random.sample(lines, 20):
    print(line)
