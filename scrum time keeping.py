import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import time
from enum import Enum
import threading
import speech_recognition as sr
import logging
import pyttsx3
import random
import queue
from sentence_transformers import SentenceTransformer
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Standup data placeholder
standup_data = {}
STOP_PHRASES = ["done", "that's it", "finished", "i'm done", "i am done", "that's all"]

def generate_standup_for(name):
    yesterday_tasks = [
        "worked on refactoring old code",
        "completed assigned Jira tickets",
        "fixed critical production bugs",
        "finished writing unit tests",
        "reviewed pull requests"
    ]
    today_tasks = [
        "will focus on writing integration tests",
        "plan to implement new API endpoints",
        "am preparing deployment for staging",
        "will sync with the design team",
        "am debugging a memory leak"
    ]
    blockers = [
        "No blockers",
        "Waiting for a code review",
        "Blocked by merge conflicts",
        "Waiting for access permissions",
        "Stuck on flaky test cases"
    ]

    return [
        f"Yesterday, I {random.choice(yesterday_tasks)}.",
        f"Today, I {random.choice(today_tasks)}.",
        f"{random.choice(blockers)}."
    ]

class ParticipantState(Enum):
    WAITING = 1
    SPEAKING = 2
    EXCEEDED = 3

class ScrumTimekeeper:
    def __init__(self, root):
        self.root = root
        self.root.title("Scrum Timekeeper with CC")
        self.root.geometry("800x600")
        self.participants = {}
        self.current_speaker = None
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.meeting_active = False
        self.transcription_text = tk.StringVar()

        self.command_queue = queue.Queue()
        self.listening_thread = None
        self.stop_listening_flag = threading.Event()

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Define agenda topics dictionary
        self.AGENDA_TOPICS = {
            "Sprint Planning": "Discuss the goals and tasks for the upcoming sprint.",
            "Bug Fixes": "Report and resolve bugs found during testing.",
            "Code Review": "Review code submitted by team members.",
            "Deployment": "Prepare and plan deployment for the current sprint.",
            "Blockers": "Discuss any obstacles preventing progress."
        }

        # Precompute agenda embeddings
        self.agenda_texts = list(self.AGENDA_TOPICS.values())
        self.agenda_embeddings = self.embed_text(self.agenda_texts)

        self.setup_gui()

    def embed_text(self, texts):
        return self.embedding_model.encode(texts, convert_to_tensor=False)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def check_similarity_to_agenda(self, spoken_lines):
        if not spoken_lines:
            return 0.0
        discussion_text = " ".join(spoken_lines)
        discussion_embedding = self.embed_text([discussion_text])[0]
        similarities = [self.cosine_similarity(discussion_embedding, topic_emb) for topic_emb in self.agenda_embeddings]
        max_similarity = max(similarities)
        return max_similarity

    def setup_gui(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        self.setup_setup_tab()
        self.setup_meeting_tab()

    def monitor_speaker_time(self, participant):
        def monitor():
            while self.meeting_active:
                if self.current_speaker != participant:
                    break  # Stop monitoring if it's no longer their turn

                pdata = self.participants.get(participant)
                if pdata and pdata["state"] == ParticipantState.SPEAKING and pdata["start_time"]:
                    elapsed = time.time() - pdata["start_time"]
                    total_used = elapsed + pdata["T_used"]
                    if total_used >= pdata["T_alloc"]:
                        # Only handle once
                        pdata["state"] = ParticipantState.EXCEEDED
                        self.root.after(0, lambda: self.handle_time_exceeded(participant))
                        break
                time.sleep(1)
        threading.Thread(target=monitor, daemon=True).start()



    def handle_time_exceeded(self, participant):
        self.status_var.set(f"{participant.capitalize()} exceeded allocated time.")
        self.interrupt_speaker(participant)
        self.stop_speaker(participant)
        self.current_speaker = None  # allow new speaker to be started via command

    def interrupt_speaker(self, participant):
        try:
            engine = pyttsx3.init()
            engine.say(f"{participant.capitalize()}, your time is up. Please wrap it up.")
            engine.runAndWait()
        except Exception as e:
            logging.error(f"TTS error: {e}")

    def setup_setup_tab(self):
        self.setup_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_tab, text="Setup")

        frame = ttk.Frame(self.setup_tab, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="Participant Name:").grid(column=0, row=0, sticky=tk.W)
        self.name_entry = ttk.Entry(frame, width=30)
        self.name_entry.grid(column=1, row=0, sticky=(tk.W, tk.E))

        ttk.Label(frame, text="Allocated Time (minutes):").grid(column=0, row=1, sticky=tk.W)
        self.time_entry = ttk.Entry(frame, width=30)
        self.time_entry.grid(column=1, row=1, sticky=(tk.W, tk.E))

        ttk.Button(frame, text="Add Participant", command=self.add_participant_gui).grid(column=2, row=0, rowspan=2, padx=5)

        self.tree = ttk.Treeview(frame, columns=('Name', 'Allocated'), show='headings')
        self.tree.heading('Name', text='Name')
        self.tree.heading('Allocated', text='Allocated Time (min)')
        self.tree.grid(column=0, row=2, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        ttk.Button(frame, text="Remove Selected", command=self.remove_participant).grid(column=0, row=3, pady=5)
        ttk.Button(frame, text="Start Meeting", command=self.start_meeting).grid(column=2, row=3, pady=5)

    def setup_meeting_tab(self):
        self.meeting_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.meeting_tab, text="Meeting")

        frame = ttk.Frame(self.meeting_tab, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.meeting_tree = ttk.Treeview(frame, columns=('Name', 'State', 'Used', 'Allocated'), show='headings')
        self.meeting_tree.heading('Name', text='Name')
        self.meeting_tree.heading('State', text='State')
        self.meeting_tree.heading('Used', text='Used Time')
        self.meeting_tree.heading('Allocated', text='Allocated Time')
        self.meeting_tree.grid(column=0, row=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.status_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.status_var, wraplength=700).grid(column=0, row=1, columnspan=3, pady=10)
        ttk.Button(frame, text="End Meeting", command=self.end_meeting).grid(column=1, row=2, pady=5)
        ttk.Button(frame, text="Play Standup", command=self.play_standup_gui).grid(column=2, row=2, pady=5)

        ttk.Label(frame, text="Real-Time Transcription (CC):").grid(column=0, row=3, sticky=tk.W, pady=(10,0))
        transcription_label = ttk.Label(frame, textvariable=self.transcription_text, wraplength=700,
                                        background="#f9f9f9", relief="solid", anchor="w", padding=5)
        transcription_label.grid(column=0, row=4, columnspan=3, sticky=(tk.W, tk.E), pady=(0,10))

    def play_standup_gui(self):
        participant = simpledialog.askstring("Standup", "Enter participant name:", parent=self.root)
        if not participant:
            return
        participant = participant.lower()
        engine = pyttsx3.init()

        participant_data = self.participants.get(participant)
        if participant_data and participant_data["spoken_lines"]:
            engine.say(f"{participant.capitalize()}'s standup update based on their spoken words.")
            for sentence in participant_data["spoken_lines"]:
                engine.say(sentence)
        elif participant in standup_data:
            engine.say(f"{participant.capitalize()}'s standup update based on dummy data.")
            for sentence in standup_data[participant]:
                engine.say(sentence)
        else:
            engine.say(f"Sorry, I don't have any standup data for {participant}.")
            messagebox.showinfo("Not Found", f"No standup data for {participant}.")

        engine.runAndWait()

    def add_participant_gui(self):
        name = self.name_entry.get()
        time_value = self.time_entry.get()
        if not name or not time_value:
            messagebox.showerror("Error", "Please enter both name and time")
            return
        try:
            allocated_time = float(time_value) * 60
            self.add_participant(name, allocated_time)
            self.name_entry.delete(0, tk.END)
            self.time_entry.delete(0, tk.END)
        except ValueError:
            messagebox.showerror("Error", "Invalid time format")

    def add_participant(self, name, allocated_time_seconds):
        name = name.lower()
        if name in self.participants:
            messagebox.showerror("Error", f"Participant {name} already exists")
            return
        self.participants[name] = {
            "T_alloc": allocated_time_seconds,
            "T_used": 0,
            "state": ParticipantState.WAITING,
            "start_time": None,
            "spoken_lines": [],
        }
        self.tree.insert('', 'end', iid=name, values=(name, f"{allocated_time_seconds / 60:.2f}"))
        self.update_meeting_tree()

    def remove_participant(self):
        selected = self.tree.selection()
        for item in selected:
            del self.participants[item]
            self.tree.delete(item)
        self.update_meeting_tree()

    def update_meeting_tree(self):
        for i in self.meeting_tree.get_children():
            self.meeting_tree.delete(i)
        for name, pdata in self.participants.items():
            used_time_min = pdata["T_used"] / 60
            allocated_time_min = pdata["T_alloc"] / 60
            state_name = pdata["state"].name
            self.meeting_tree.insert('', 'end', iid=name, values=(name, state_name, f"{used_time_min:.1f}", f"{allocated_time_min:.1f}"))

    def start_meeting(self):
        if not self.participants:
            messagebox.showerror("Error", "No participants added")
            return
        self.meeting_active = True
        self.status_var.set("Meeting started.")
        self.notebook.select(self.meeting_tab)
        self.current_speaker = None
        self.update_meeting_tree()

        self.stop_listening_flag.clear()
        self.listening_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.listening_thread.start()

        self.start_next_speaker()

    def listen_loop(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        while self.meeting_active and not self.stop_listening_flag.is_set():
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
                try:
                    recognized_text = self.recognizer.recognize_google(audio).lower()
                    logging.debug(f"Recognized: {recognized_text}")
                    self.transcription_text.set(recognized_text)
                    self.process_recognition(recognized_text)
                except sr.UnknownValueError:
                    logging.debug("Could not understand audio")
                except sr.RequestError as e:
                    logging.error(f"API error: {e}")
            except sr.WaitTimeoutError:
                logging.debug("Listening timed out, no speech detected")

    def process_recognition(self, text):
        text = text.strip().lower()
        words = text.split()

        # Check for voice commands even if no one is speaking
        if len(words) >= 2:
            if words[0] == "start" and words[1] in self.participants:
                self.command_queue.put(("start", words[1]))
                return
            if words[-1] == "start" and words[0] in self.participants:
                self.command_queue.put(("start", words[0]))
                return
            if words[0] == "stop" and words[1] in self.participants:
                self.command_queue.put(("stop", words[1]))
                return
            if words[-1] == "stop" and words[0] in self.participants:
                self.command_queue.put(("stop", words[0]))
                return

        # Only collect spoken lines if someone is speaking
        if not self.current_speaker:
            return

        pdata = self.participants[self.current_speaker]
        pdata["spoken_lines"].append(text)

        if any(phrase in text for phrase in STOP_PHRASES):
            self.command_queue.put(("stop", self.current_speaker))


    def start_next_speaker(self):
        if not self.meeting_active:
            return
        waiting = [p for p, d in self.participants.items() if d["state"] == ParticipantState.WAITING]
        if not waiting:
            self.status_var.set("All participants have spoken. Meeting is ending.")
            self.end_meeting()
            return
        next_speaker = waiting[0]
        self.set_speaker(next_speaker)

    def set_speaker(self, name):
        if self.current_speaker:
            # Stop previous speaker timer and update used time
            prev = self.participants[self.current_speaker]
            if prev["start_time"] is not None:
                elapsed = time.time() - prev["start_time"]
                prev["T_used"] += elapsed
            prev["state"] = ParticipantState.WAITING
            prev["start_time"] = None

        self.current_speaker = name
        pdata = self.participants[name]
        pdata["state"] = ParticipantState.SPEAKING
        pdata["start_time"] = time.time()

        self.status_var.set(f"{name.capitalize()} is now speaking.")
        self.update_meeting_tree()
        self.monitor_speaker_time(name)


    def stop_speaker(self, name):
        pdata = self.participants.get(name)
        if not pdata or pdata["state"] != ParticipantState.SPEAKING:
            return

        if pdata["start_time"]:
            elapsed = time.time() - pdata["start_time"]
            pdata["T_used"] += elapsed

        pdata["state"] = ParticipantState.WAITING
        pdata["start_time"] = None
        self.update_meeting_tree()

        if self.current_speaker == name:
            self.current_speaker = None





    def end_meeting(self):
        self.meeting_active = False
        self.stop_listening_flag.set()
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
        self.status_var.set("Meeting ended.")
        self.show_meeting_summary()

    def show_meeting_summary(self):
        summary = "Meeting Summary:\n\n"
        for name, pdata in self.participants.items():
            summary += f"{name.capitalize()} (used {pdata['T_used'] / 60:.2f} min):\n"
            similarity = self.check_similarity_to_agenda(pdata["spoken_lines"])
            summary += f"Similarity to agenda: {similarity:.2f}\n"
            summary += "Spoken lines:\n"
            for line in pdata["spoken_lines"]:
                summary += f"  - {line}\n"
            summary += "\n"
        messagebox.showinfo("Meeting Summary", summary)

    def main_loop(self):
        def command_handler():
            while True:  # <--- allow handling even outside meeting time
                try:
                    command, participant = self.command_queue.get(timeout=0.5)
                    logging.debug(f"Handling command: {command} for {participant}")

                    if command == "stop" and participant == self.current_speaker:
                        self.stop_speaker(participant)
                    elif command == "start":
                        self.set_speaker(participant)
                except queue.Empty:
                    continue

        threading.Thread(target=command_handler, daemon=True).start()
        self.root.mainloop()



if __name__ == "__main__":
    root = tk.Tk()
    app = ScrumTimekeeper(root)
    app.main_loop()
