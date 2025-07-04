import tkinter as tk
from tkinter import ttk, messagebox
import time
from enum import Enum
import threading
import speech_recognition as sr
import logging
import pyttsx3
import queue
import joblib
import os

# NEW: For semantic similarity
from sentence_transformers import SentenceTransformer, util

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load classifiers
cat_vectorizer = joblib.load("category_vectorizer.joblib")
cat_clf = joblib.load("category_classifier.joblib")
ss_vectorizer = joblib.load("startstop_vectorizer.joblib")
ss_clf = joblib.load("startstop_classifier.joblib")

def categorize_statement(statement):
    X = cat_vectorizer.transform([statement])
    cat = cat_clf.predict(X)[0]
    logging.debug(f"Categorized '{statement}' as {cat}")
    return cat

def detect_start_stop(statement):
    X = ss_vectorizer.transform([statement])
    val = ss_clf.predict(X)[0]
    logging.debug(f"Start/stop classifier: '{statement}' -> {val}")
    return val

class ParticipantState(Enum):
    WAITING = 1
    SPEAKING = 2
    EXCEEDED = 3
    DONE = 4

class ScrumTimekeeper:
    def __init__(self, root):
        self.root = root
        self.root.title("Scrum Timekeeper")
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
        self.setup_gui()

        # --- SEMANTIC SIMILARITY SETUP ---
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.agenda = [
    "What did you do yesterday?",
    "What will you do today?",
    "Are there any blockers or impediments?"
]
        self.agenda_emb = self.sim_model.encode(self.agenda, convert_to_tensor=True)

    def setup_gui(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        self.setup_setup_tab()
        self.setup_meeting_tab()

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
        ttk.Button(frame, text="Show Similarity Report", command=self.show_similarity_report).grid(column=2, row=2, pady=5)
        ttk.Label(frame, text="Real-Time Transcription (CC):").grid(column=0, row=3, sticky=tk.W, pady=(10,0))
        transcription_label = ttk.Label(frame, textvariable=self.transcription_text, wraplength=700,
                                        background="#f9f9f9", relief="solid", anchor="w", padding=5)
        transcription_label.grid(column=0, row=4, columnspan=3, sticky=(tk.W, tk.E), pady=(0,10))
        # Manual entry for testing
        self.manual_entry = ttk.Entry(frame, width=60)
        self.manual_entry.grid(column=0, row=5, columnspan=2, pady=(10,0))
        ttk.Button(frame, text="Submit Statement", command=self.manual_statement).grid(column=2, row=5, pady=(10,0))

    def manual_statement(self):
        text = self.manual_entry.get()
        if text:
            self.process_recognition(text)
            self.manual_entry.delete(0, tk.END)

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
        logging.debug(f"Added participant {name} with {allocated_time_seconds/60:.2f} min")

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
        self.status_var.set("Meeting started. Say a start phrase (e.g. 'Alice, you can start').")
        self.notebook.select(self.meeting_tab)
        self.current_speaker = None
        self.update_meeting_tree()
        self.stop_listening_flag.clear()
        self.listening_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.listening_thread.start()
        logging.debug("Meeting started. Awaiting start phrase.")

    def listen_loop(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        while self.meeting_active and not self.stop_listening_flag.is_set():
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
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
        try:
            action = detect_start_stop(text)
        except Exception:
            action = None

        # Fallback keyword lists
        start_keywords = ["start", "begin", "you can start", "your turn"]
        stop_keywords = [
            "i'm done", "that's it", "finished", "no more updates", "that's all", "i have nothing else",
            "i am finished", "done for now", "that concludes", "that is all"
        ]

        # Start logic
        is_start = (action == "start") or any(phrase in text for phrase in start_keywords)
        is_stop = (action == "stop") or any(phrase in text for phrase in stop_keywords)

        # Only treat as start if either classifier OR keyword matches AND current_speaker is None
        if is_start and self.current_speaker is None:
            for name in self.participants:
                if name in text:
                    self.command_queue.put(("start", name))
                    logging.debug(f"Start command detected for {name}")
                    return
            self.command_queue.put(("start", self.get_next_waiting()))
            logging.debug("Start command detected for next waiting participant")
            return

        # Only treat as stop if either classifier OR keyword matches AND current_speaker is not None
        if is_stop and self.current_speaker is not None:
            self.command_queue.put(("stop", self.current_speaker))
            logging.debug(f"Stop command detected for {self.current_speaker}")
            return

        # Only add as a content line if NOT classified as start/stop by either method
        if self.current_speaker and not is_start and not is_stop:
            pdata = self.participants[self.current_speaker]
            if pdata["state"] == ParticipantState.SPEAKING:
                pdata["spoken_lines"].append(text)
                logging.debug(f"Added statement for {self.current_speaker}: {text}")
            else:
                logging.debug(f"Did NOT add statement: {text} (state is {pdata['state']})")
        else:
            logging.debug(f"No current speaker or action was start/stop. Ignored statement: {text}")

    def get_next_waiting(self):
        waiting = [p for p, d in self.participants.items() if d["state"] == ParticipantState.WAITING]
        logging.debug(f"Next waiting participant: {waiting[0] if waiting else None}")
        return waiting[0] if waiting else None

    def monitor_speaker_time(self, participant):
        def monitor():
            while self.meeting_active:
                if self.current_speaker != participant:
                    break
                pdata = self.participants.get(participant)
                if pdata and pdata["state"] == ParticipantState.SPEAKING and pdata["start_time"]:
                    elapsed = time.time() - pdata["start_time"]
                    total_used = elapsed + pdata["T_used"]
                    if total_used >= pdata["T_alloc"]:
                        pdata["state"] = ParticipantState.EXCEEDED
                        self.root.after(0, lambda: self.handle_time_exceeded(participant))
                        break
                time.sleep(1)
        threading.Thread(target=monitor, daemon=True).start()

    def handle_time_exceeded(self, participant):
        self.status_var.set(f"{participant.capitalize()} exceeded allocated time.")
        self.interrupt_speaker(participant)
        self.stop_speaker(participant)
        self.current_speaker = None
        logging.debug(f"{participant} exceeded time and was stopped.")

    def interrupt_speaker(self, participant):
        try:
            engine = pyttsx3.init()
            engine.say(f"{participant.capitalize()}, your time is up. Please wrap it up.")
            engine.runAndWait()
        except Exception as e:
            logging.error(f"TTS error: {e}")

    def start_next_speaker(self):
        if not self.meeting_active:
            return
        next_speaker = self.get_next_waiting()
        if not next_speaker:
            self.status_var.set("All participants have spoken. Meeting is ending.")
            self.end_meeting()
            return
        self.set_speaker(next_speaker)

    def set_speaker(self, name):
        logging.debug(f"set_speaker called for {name}")
        if self.current_speaker:
            prev = self.participants[self.current_speaker]
            if prev["start_time"] is not None:
                elapsed = time.time() - prev["start_time"]
                prev["T_used"] += elapsed
            prev["state"] = ParticipantState.WAITING
            prev["start_time"] = None
            logging.debug(f"Previous speaker was {self.current_speaker}, set to WAITING")
        self.current_speaker = name
        pdata = self.participants[name]
        pdata["state"] = ParticipantState.SPEAKING
        pdata["start_time"] = time.time()
        self.status_var.set(f"{name.capitalize()} is now speaking.")
        self.update_meeting_tree()
        self.monitor_speaker_time(name)
        logging.debug(f"{name} state set to SPEAKING")

    def stop_speaker(self, name):
        logging.debug(f"stop_speaker called for {name}")
        pdata = self.participants.get(name)
        if not pdata or pdata["state"] != ParticipantState.SPEAKING:
            return
        if pdata["start_time"]:
            elapsed = time.time() - pdata["start_time"]
            pdata["T_used"] += elapsed
        pdata["state"] = ParticipantState.DONE  # Set to DONE!
        pdata["start_time"] = None
        self.update_meeting_tree()
        if self.current_speaker == name:
            self.current_speaker = None
        logging.debug(f"{name} state set to DONE")

    def end_meeting(self):
        self.meeting_active = False
        self.stop_listening_flag.set()
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
        self.status_var.set("Meeting ended.")
        self.show_meeting_summary()
        similarity_report = self.get_similarity_report()
        messagebox.showinfo("Similarity Report", similarity_report)
        logging.debug("Meeting ended.")

    def show_meeting_summary(self):
        logging.debug("Generating meeting summary...")
        summary = "Meeting Summary:\n\n"
        for name, pdata in self.participants.items():
            logging.debug(f"{name}: {len(pdata['spoken_lines'])} statements recorded.")
            summary += f"{name.capitalize()} (used {pdata['T_used'] / 60:.2f} min):\n"
            if not pdata["spoken_lines"]:
                summary += "  No statements recorded.\n"
                continue
            categorized = {"yesterday": [], "today": [], "blocker": []}
            for line in pdata["spoken_lines"]:
                cat = categorize_statement(line)
                categorized[cat].append(line)
            for cat in ["yesterday", "today", "blocker"]:
                if categorized[cat]:
                    summary += f"{cat.capitalize()}:\n"
                    for line in categorized[cat]:
                        summary += f"  - {line}\n"
            summary += "\n"
        messagebox.showinfo("Meeting Summary", summary)

    # --- SEMANTIC SIMILARITY REPORT ---
    def get_similarity_report(self):
        report = ""
        for name, pdata in self.participants.items():
            if not pdata["spoken_lines"]:
                continue
            report += f"\n{name.capitalize()}:\n"
            for line in pdata["spoken_lines"]:
                line_emb = self.sim_model.encode(line, convert_to_tensor=True)
                sims = util.pytorch_cos_sim(line_emb, self.agenda_emb)[0]
                best_idx = sims.argmax().item()
                best_score = sims[best_idx].item()
                agenda_item = self.agenda[best_idx]
                report += f'  - "{line}" (agenda: "{agenda_item}", similarity: {best_score:.2f})\n'
        return report if report else "No statements to analyze."

    def show_similarity_report(self):
        similarity_report = self.get_similarity_report()
        messagebox.showinfo("Similarity Report", similarity_report)

    def main_loop(self):
        def command_handler():
            while True:
                try:
                    command, participant = self.command_queue.get(timeout=0.5)
                    logging.debug(f"Handling command: {command} for {participant}. Current speaker: {self.current_speaker}")
                    if command == "stop" and participant == self.current_speaker:
                        self.stop_speaker(participant)
                        # Do NOT call self.start_next_speaker() here!
                        # Wait for explicit start phrase for next participant
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
