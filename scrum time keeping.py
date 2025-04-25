import tkinter as tk
from tkinter import ttk, messagebox
import time
from enum import Enum
import threading
import speech_recognition as sr
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.setup_gui()

    def setup_gui(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.setup_tab = ttk.Frame(self.notebook)
        self.meeting_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="Setup")
        self.notebook.add(self.meeting_tab, text="Meeting")
        
        self.setup_setup_tab()
        self.setup_meeting_tab()

    def setup_setup_tab(self):
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

        ttk.Label(frame, text="Real-Time Transcription (CC):").grid(column=0, row=3, sticky=tk.W, pady=(10,0))
        transcription_label = ttk.Label(frame,
                                        textvariable=self.transcription_text,
                                        wraplength=700,
                                        background="#f9f9f9",
                                        relief="solid",
                                        anchor="w",
                                        padding=5)
        transcription_label.grid(column=0, row=4, columnspan=3, sticky=(tk.W, tk.E), pady=(0,10))

    def add_participant_gui(self):
        name = self.name_entry.get()
        time_value = self.time_entry.get()
        
        if not name or not time_value:
            messagebox.showerror("Error", "Please enter both name and time")
            return
        
        try:
            allocated_time = float(time_value) * 60  # Convert minutes to seconds
            self.add_participant(name, allocated_time)
            self.name_entry.delete(0, tk.END)
            self.time_entry.delete(0, tk.END)
            self.update_participant_list()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for time")

    def add_participant(self, name, allocated_time):
        self.participants[name.lower()] = {
            "state": ParticipantState.WAITING,
            "T_alloc": allocated_time,
            "T_used": 0,
            "start_time": None
        }
        logging.info(f"Added participant: {name}")

    def update_participant_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for name, data in self.participants.items():
            self.tree.insert('', tk.END, values=(name, f"{data['T_alloc']/60:.2f}"))

    def remove_participant(self):
        selected_item = self.tree.selection()
        if selected_item:
            name = self.tree.item(selected_item)['values'][0]
            del self.participants[name.lower()]
            self.update_participant_list()

    def start_meeting(self):
        if not self.participants:
            messagebox.showerror("Error", "Please add participants before starting the meeting")
            return
        self.meeting_active = True
        self.notebook.select(1)  # Switch to meeting tab
        self.update_meeting_list()
        self.status_var.set("Meeting started. Say 'start [name]' or 'stop [name]' to control timing.")
        threading.Thread(target=self.run_meeting, daemon=True).start()

    def end_meeting(self):
        self.meeting_active = False
        self.status_var.set("Meeting ended.")
        self.show_meeting_summary()

    def show_meeting_summary(self):
        summary = "Meeting Summary:\n\n"
        for name, data in self.participants.items():
            summary += f"{name}: Used {data['T_used']/60:.2f} min / Allocated {data['T_alloc']/60:.2f} min\n"
        messagebox.showinfo("Meeting Summary", summary)

    def update_meeting_list(self):
        for item in self.meeting_tree.get_children():
            self.meeting_tree.delete(item)
        for name, data in self.participants.items():
            self.meeting_tree.insert('', tk.END, values=(
                name, 
                data['state'].name, 
                f"{data['T_used']/60:.2f} min", 
                f"{data['T_alloc']/60:.2f} min"
            ))

    def process_command(self, command, participant):
        logging.debug(f"Processing command: {command} for {participant}")
        participant = participant.lower()
        if participant not in self.participants:
            logging.warning(f"Participant {participant} not found")
            return

        if command == "start" and self.participants[participant]["state"] == ParticipantState.WAITING:
            self.transition_state(participant, ParticipantState.SPEAKING)
            self.participants[participant]["start_time"] = time.time()
            self.current_speaker = participant
            self.status_var.set(f"Started timing for {participant}")
            logging.info(f"Started timing for {participant}")
        elif command == "stop" and self.participants[participant]["state"] in [ParticipantState.SPEAKING, ParticipantState.EXCEEDED]:
            self.update_time(participant)
            self.transition_state(participant, ParticipantState.WAITING)
            self.current_speaker = None
            self.status_var.set(f"Stopped timing for {participant}")
            logging.info(f"Stopped timing for {participant}")
        else:
            logging.warning(f"Invalid command state: {command} for {participant} in state {self.participants[participant]['state']}")

        self.check_exceeded(participant)
        self.update_meeting_list()

    def transition_state(self, participant, new_state):
        old_state = self.participants[participant]["state"]
        self.participants[participant]["state"] = new_state
        logging.info(f"Transitioned {participant} from {old_state} to {new_state}")

    def update_time(self, participant):
        if self.participants[participant]["state"] == ParticipantState.SPEAKING:
            current_time = time.time()
            elapsed = current_time - self.participants[participant]["start_time"]
            self.participants[participant]["T_used"] += elapsed
            self.participants[participant]["start_time"] = current_time
            logging.debug(f"Updated time for {participant}: {elapsed:.2f} seconds")

    def check_exceeded(self, participant):
        if self.participants[participant]["T_used"] > self.participants[participant]["T_alloc"]:
            self.transition_state(participant, ParticipantState.EXCEEDED)
            self.status_var.set(f"Warning: {participant} has exceeded their allocated time!")
            logging.warning(f"{participant} has exceeded their allocated time")

    def listen_for_command(self):
        with self.microphone as source:
            self.status_var.set("Listening for command...")
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio).lower()
            self.transcription_text.set(text)  # Update real-time transcription
            logging.debug(f"Recognized speech: {text}")
            words = text.split()
            if len(words) == 2 and words[0] in ["start", "stop"]:
                return words[0], words[1]
        except sr.UnknownValueError:
            self.status_var.set("Could not understand audio")
            logging.warning("Could not understand audio")
        except sr.RequestError as e:
            self.status_var.set(f"Could not request results; {e}")
            logging.error(f"Could not request results; {e}")
        return None, None

    def run_meeting(self):
        while self.meeting_active:
            command, participant = self.listen_for_command()
            if command and participant:
                self.process_command(command, participant)
            if self.current_speaker:
                self.update_time(self.current_speaker)
            self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = ScrumTimekeeper(root)
    root.mainloop()
