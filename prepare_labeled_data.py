import pandas as pd
import ast
from datasets import load_dataset

def label_start_stop(text):
    text_l = text.lower().strip()

    # Flexible phrase lists (add more as needed based on your sample)
    start_phrases = [
        "let me begin", "i will start", "i am starting", "starting now", "i'll start", "my update",
        "i'd like to start", "i'd like to begin", "let's start", "let's begin", "i'd like to start the meeting",
        "i'd like to start our daily standup meeting", "good morning, team. i'd like to start",
        "good morning, team. let's start", "i will start my update"
    ]

    stop_phrases = [
        "i'm done", "that's it", "finished", "no more updates", "that's all", "i have nothing else",
        "i am finished", "done for now", "that concludes", "that is all", "thank you",
        "thank you for your participation", "okay, that's all", "that's it for today",
        "that's all for today", "keep up the good work", "meeting adjourned", "we'll meet again tomorrow"
    ]

    # Match if phrase appears anywhere (case-insensitive)
    for phrase in start_phrases:
        if phrase in text_l:
            return "start"
    for phrase in stop_phrases:
        if phrase in text_l:
            return "stop"
    return "other"

# Load the MOM-Summary-Dataset
mom_dataset = load_dataset("sasvata/MOM-Summary-Dataset")
mom_meetings = mom_dataset['train']

ss_data = []

N = len(mom_meetings)
for i in range(N):
    mt_raw = mom_meetings[i]['Meeting Transcript']
    try:
        mt_dict = ast.literal_eval(mt_raw)
        transcript = mt_dict.get('transcript', [])
        for line in transcript:
            label_ss = label_start_stop(line)
            ss_data.append({"text": line, "label": label_ss})
    except Exception as e:
        print(f"Error parsing row {i}: {e}")

df_ss = pd.DataFrame(ss_data)
df_ss.to_csv("start_stop_labeled.csv", index=False)
print(df_ss['label'].value_counts())
