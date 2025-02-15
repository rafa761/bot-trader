import os
import re
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from queue import Queue
from typing import Literal

import customtkinter as ctk


class Tooltip:
    """
    Classe responsável por criar e gerenciar tooltips em widgets da interface gráfica.
    """

    def __init__(self, widget: tk.Widget, text: str):
        """
        Inicializa o tooltip.

        :param widget: Widget ao qual o tooltip será associado.
        :param text: Texto que será exibido no tooltip.
        """
        self.widget = widget
        self.text = text
        self.tooltip: tk.Toplevel | None = None
        self.widget.bind('<Enter>', self.show_tooltip)
        self.widget.bind('<Leave>', self.hide_tooltip)

    def show_tooltip(self, event=None):
        """
        Exibe o tooltip próximo ao widget.
        """
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tooltip,
            text=self.text,
            background="#2B2B2B",
            foreground="white",
            relief="solid",
            borderwidth=1,
            font=("Roboto", 10)
        )
        label.pack()

    def hide_tooltip(self, event=None):
        """
        Oculta o tooltip.
        """
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class ProcessManager:
    """
    Classe responsável por gerenciar a execução de processos (bots e treinamentos de IA).
    """

    def __init__(self, output_queue: Queue):
        """
        Inicializa o gerenciador de processos.

        :param output_queue: Fila para armazenar as saídas dos processos.
        """
        self.running_processes: dict[str, subprocess.Popen] = {}
        self.output_queue = output_queue

    def run_file(self, bot_dir: Path, file_path: Path, process_type: Literal["Bot", "AI Training"]) -> None:
        """
        Executa um arquivo Python (bot ou treinamento de IA).

        :param bot_dir: Diretório do bot.
        :param file_path: Caminho do arquivo Python a ser executado.
        :param process_type: Tipo do processo ("Bot" ou "AI Training").
        """
        if not file_path.exists():
            self.output_queue.put((bot_dir.name, f"Erro: Arquivo {file_path.name} não encontrado em {bot_dir.name}"))
            return

        process_key = f"{bot_dir.name}_{process_type}"
        if process_key in self.running_processes:
            self.output_queue.put((process_key, f"{process_type} do {bot_dir.name} já está em execução!"))
            return

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(
                [sys.executable, str(file_path.absolute())],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=str(bot_dir.absolute())
            )

            self.running_processes[process_key] = process

            threading.Thread(
                target=self.monitor_output,
                args=(process.stdout, process_key, "OUT"),
                daemon=True
            ).start()

            threading.Thread(
                target=self.monitor_output,
                args=(process.stderr, process_key, "ERR"),
                daemon=True
            ).start()

            self.output_queue.put((process_key, f"{process_type} do {bot_dir.name} iniciado com sucesso!"))

        except Exception as e:
            self.output_queue.put((process_key, f"Erro ao executar {process_type} do {bot_dir.name}: {str(e)}"))

    def monitor_output(self, pipe, process_key: str, pipe_name: Literal["OUT", "ERR"]) -> None:
        """
        Monitora a saída de um processo e a envia para a fila de saída.

        :param pipe: Pipe de saída do processo.
        :param process_key: Chave do processo.
        :param pipe_name: Nome do pipe ("OUT" ou "ERR").
        """
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    urls = re.findall(
                        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)

                    if urls:
                        for url in urls:
                            self.output_queue.put((process_key, f"Dashboard disponível em: {url}"))
                    else:
                        self.output_queue.put((process_key, f"[{pipe_name}] {line.strip()}"))
        except Exception as e:
            self.output_queue.put((process_key, f"Erro ao monitorar saída: {str(e)}"))
        finally:
            if process_key in self.running_processes:
                process = self.running_processes[process_key]
                if process.poll() is not None:
                    del self.running_processes[process_key]
                    self.output_queue.put((process_key, "Processo finalizado"))


class BotLauncherGUI:
    """
    Classe principal da interface gráfica para lançamento de bots de trade.
    """

    def __init__(self):
        """
        Inicializa a interface gráfica.
        """
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Bot Trader Launcher")
        self.root.geometry("800x600")

        self.output_queue = Queue()
        self.process_manager = ProcessManager(self.output_queue)

        # Lista para manter referência aos tooltips
        self.tooltips = []

        self.create_widgets()
        self.load_bots()
        self.start_queue_monitor()

    def create_widgets(self):
        """
        Cria os widgets da interface gráfica.
        """
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

    def create_tooltip(self, widget: tk.Widget, text: str) -> Tooltip:
        """
        Cria um tooltip para um widget.

        :param widget: Widget ao qual o tooltip será associado.
        :param text: Texto do tooltip.
        :return: Instância do tooltip criado.
        """
        tooltip = Tooltip(widget, text)
        self.tooltips.append(tooltip)
        return tooltip

    def load_bots(self):
        """
        Carrega os bots disponíveis no diretório atual.
        """
        current_path = Path(".")
        if not current_path.exists():
            self.log_message("Erro: Diretório atual não encontrado!")
            return

        for bot_dir in current_path.iterdir():
            if bot_dir.is_dir() and not bot_dir.name.startswith('__'):
                self.create_bot_row(bot_dir)

    def create_bot_row(self, bot_dir: Path):
        """
        Cria uma linha na interface para um bot específico.

        :param bot_dir: Diretório do bot.
        """
        frame = ctk.CTkFrame(self.bots_frame)
        frame.pack(fill="x", padx=5, pady=5)

        # Nome do bot
        name_label = ctk.CTkLabel(
            frame,
            text=bot_dir.name,
            font=("Roboto", 14)
        )
        name_label.pack(side="left", padx=10)

        # Container para os botões (alinhamento à direita)
        buttons_frame = ctk.CTkFrame(frame)
        buttons_frame.pack(side="right", padx=10)

        # Verifica a existência dos arquivos
        main_exists = (bot_dir / "main.py").exists()
        ai_exists = (bot_dir / "ai_training.py").exists()

        # Botão AI Training
        ai_button = ctk.CTkButton(
            buttons_frame,
            text="🤖 A.I Training",
            width=120,
            command=lambda: self.process_manager.run_file(bot_dir, bot_dir / "ai_training.py", "AI Training"),
            state="normal" if ai_exists else "disabled",
            fg_color="#4B0082" if ai_exists else "#666666"
        )
        ai_button.pack(side="right", padx=5)

        if not ai_exists:
            self.create_tooltip(ai_button, "Arquivo ai_training.py não encontrado")

        # Botão Play
        play_button = ctk.CTkButton(
            buttons_frame,
            text="▶ Play",
            width=100,
            command=lambda: self.process_manager.run_file(bot_dir, bot_dir / "main.py", "Bot"),
            state="normal" if main_exists else "disabled",
            fg_color="#006400" if main_exists else "#666666"
        )
        play_button.pack(side="right", padx=5)

        if not main_exists:
            self.create_tooltip(play_button, "Arquivo main.py não encontrado")

    def start_queue_monitor(self):
        """
        Inicia o monitoramento da fila de saída dos processos.
        """

        def monitor():
            try:
                while not self.output_queue.empty():
                    process_key, message = self.output_queue.get_nowait()
                    self.log_message(f"[{process_key}] {message}")
            except Exception as e:
                print(f"Erro no monitor: {str(e)}")
            finally:
                self.root.after(100, monitor)

        self.root.after(100, monitor)

    def log_message(self, message: str):
        """
        Exibe uma mensagem no console da interface gráfica.

        :param message: Mensagem a ser exibida.
        """
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
        """
        Torna uma URL clicável no console.

        :param url: URL a ser tornada clicável.
        """
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
        """
        Inicia a execução da interface gráfica.
        """
        self.root.mainloop()


if __name__ == "__main__":
    app = BotLauncherGUI()
    app.run()
