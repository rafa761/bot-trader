import os
import re
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from queue import Queue
from typing import Dict

import customtkinter as ctk


class BotLauncherGUI:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Bot Trader Launcher")
        self.root.geometry("800x600")

        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.output_queue = Queue()

        self.create_widgets()
        self.load_bots()
        self.start_queue_monitor()

    def create_widgets(self):
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Bot Trader Launcher",
            font=("Roboto", 24, "bold")
        )
        self.title_label.pack(pady=10)

        self.bots_frame = ctk.CTkFrame(self.main_frame)
        self.bots_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.console = ctk.CTkTextbox(self.main_frame, height=200)
        self.console.pack(fill="x", padx=10, pady=10)
        self.console.configure(state="disabled")

    def load_bots(self):
        """Carrega dinamicamente os bots disponíveis no diretório atual"""
        current_path = Path(".")
        if not current_path.exists():
            self.log_message("Erro: Diretório atual não encontrado!")
            return

        for bot_dir in current_path.iterdir():
            if bot_dir.is_dir() and not bot_dir.name.startswith('__'):
                self.create_bot_row(bot_dir)

    def create_bot_row(self, bot_dir: Path):
        frame = ctk.CTkFrame(self.bots_frame)
        frame.pack(fill="x", padx=5, pady=5)

        name_label = ctk.CTkLabel(
            frame,
            text=bot_dir.name,
            font=("Roboto", 14)
        )
        name_label.pack(side="left", padx=10)

        play_button = ctk.CTkButton(
            frame,
            text="▶ Play",
            width=100,
            command=lambda: self.run_bot(bot_dir)
        )
        play_button.pack(side="right", padx=10)

    def run_bot(self, bot_dir: Path):
        # Constrói o caminho correto para o main.py
        main_file = bot_dir / "main.py"

        # Log do caminho para debug
        self.log_message(f"Tentando executar: {main_file.absolute()}")

        if not main_file.exists():
            self.log_message(f"Erro: Arquivo main.py não encontrado em {bot_dir.name}")
            return

        if bot_dir.name in self.running_processes:
            self.log_message(f"Bot {bot_dir.name} já está em execução!")
            return

        try:
            # Configura o ambiente para o processo
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # Inicia o processo com redirecionamento de saída
            process = subprocess.Popen(
                [sys.executable, str(main_file.absolute())],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=str(bot_dir.absolute())  # Define o diretório de trabalho como a pasta do bot
            )

            self.running_processes[bot_dir.name] = process

            # Inicia threads para monitorar stdout e stderr
            threading.Thread(
                target=self.monitor_output,
                args=(process.stdout, bot_dir.name, "OUT"),
                daemon=True
            ).start()

            threading.Thread(
                target=self.monitor_output,
                args=(process.stderr, bot_dir.name, "ERR"),
                daemon=True
            ).start()

            self.log_message(f"Bot {bot_dir.name} iniciado com sucesso!")

        except Exception as e:
            self.log_message(f"Erro ao executar {bot_dir.name}: {str(e)}")

    def monitor_output(self, pipe, bot_name: str, pipe_name: str):
        """Monitora a saída (stdout ou stderr) do processo"""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    # Procura por URLs no output
                    urls = re.findall(
                        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)

                    if urls:
                        for url in urls:
                            self.output_queue.put((bot_name, f"Dashboard disponível em: {url}"))
                    else:
                        self.output_queue.put((bot_name, f"[{pipe_name}] {line.strip()}"))
        except Exception as e:
            self.output_queue.put((bot_name, f"Erro ao monitorar saída: {str(e)}"))
        finally:
            if bot_name in self.running_processes:
                process = self.running_processes[bot_name]
                if process.poll() is not None:
                    del self.running_processes[bot_name]
                    self.output_queue.put((bot_name, "Processo finalizado"))

    def start_queue_monitor(self):
        def monitor():
            try:
                while not self.output_queue.empty():
                    bot_name, message = self.output_queue.get_nowait()
                    self.log_message(f"[{bot_name}] {message}")
            except Exception as e:
                print(f"Erro no monitor: {str(e)}")
            finally:
                self.root.after(100, monitor)

        self.root.after(100, monitor)

    def log_message(self, message: str):
        self.console.configure(state="normal")
        self.console.insert("end", message + "\n")
        self.console.configure(state="disabled")
        self.console.see("end")

        if "http" in message:
            url = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                            message)
            if url:
                url = url.group()
                self.make_url_clickable(url)

    def make_url_clickable(self, url: str):
        self.console.tag_config("url", foreground="blue", underline=True)
        self.console.tag_bind("url", "<Button-1>", lambda e: webbrowser.open(url))

        start = "1.0"
        while True:
            pos = self.console.search(url, start, "end")
            if not pos:
                break
            self.console.tag_add("url", pos, f"{pos}+{len(url)}c")
            start = f"{pos}+{len(url)}c"

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = BotLauncherGUI()
    app.run()
