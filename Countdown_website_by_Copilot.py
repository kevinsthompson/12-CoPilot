# %%
{
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

# %%





# %%
"""
Cell 1: Instrumentation utilities
Each cell prints start/end debug messages with timing and optional memory stats.
Works in Jupyter and as a plain Python script.
"""
import time, sys, gc, traceback, datetime, os

try:
    import psutil
    _PSUTIL = True
    _PROC = psutil.Process(os.getpid())
except ImportError:
    _PSUTIL = False

_CELL_COUNTER = 0

def _fmt_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def cell_start(name: str):
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    print(f"[CELL {_CELL_COUNTER} START] {_fmt_ts()} :: {name}")
    if _PSUTIL:
        rss = _PROC.memory_info().rss / (1024**2)
        print(f"[CELL {_CELL_COUNTER} MEM] RSS={rss:.2f}MB")
    return time.perf_counter()

def cell_end(start_time: float, success: bool = True):
    elapsed = time.perf_counter() - start_time
    status = "OK" if success else "ERROR"
    if _PSUTIL:
        rss = _PROC.memory_info().rss / (1024**2)
        print(f"[CELL {_CELL_COUNTER} MEM] RSS={rss:.2f}MB")
    print(f"[CELL {_CELL_COUNTER} END] {_fmt_ts()} :: {status} :: {elapsed:.3f}s")
    sys.stdout.flush()

_st = cell_start("Init instrumentation utilities")
# (No work needed here beyond defining utilities)
cell_end(_st)




# %%
"""
Cell 2: Dependencies (Gradio) setup
- Ensures gradio is available (installs if missing)
"""
_st = cell_start("Dependencies setup")

import importlib, subprocess

def ensure_package(pkg: str, import_name: str | None = None, version: str | None = None):
    modname = import_name or pkg
    try:
        return importlib.import_module(modname)
    except ImportError:
        print(f"[DEPS] Installing '{pkg}{'=='+version if version else ''}'...")
        pkgspec = f"{pkg}=={version}" if version else pkg
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkgspec])
        return importlib.import_module(modname)

gr = ensure_package("gradio")

cell_end(_st)



# %%
"""
Cell 3: Game stubs and helpers
Two simple game stubs to be implemented later:
- Guess the Number (stub)
- Rock-Paper-Scissors (stub)
"""
_st = cell_start("Define game stubs and helpers")

# Guess the Number (stub)
def guess_game_submit(guess):
    try:
        if guess is None or (isinstance(guess, float) and not guess == guess):
            return "Please enter a number."
        return f"Stub: You guessed {int(guess)}. Game logic will be added later."
    except Exception:
        traceback.print_exc()
        return "Error: Invalid input."

def guess_game_reset():
    return ""

# Rock-Paper-Scissors (stub)
_CHOICES = ["Rock", "Paper", "Scissors"]

def rps_game_submit(choice):
    if not choice:
        return "Please choose Rock, Paper, or Scissors."
    return f"Stub: You chose {choice}. Game logic will be added later."

def rps_game_reset():
    return ""

cell_end(_st)




# %%
"""
Cell 4: Build Gradio UI (website)
- Two tabs hosting the two game stubs
"""
_st = cell_start("Build Gradio UI")

import gradio as gr  # type: ignore

def build_app():
    with gr.Blocks(title="Mini Games Hub") as demo:
        gr.Markdown("# Mini Games Hub\nTwo simple games (stubs). Functionality coming soon.")
        with gr.Tab("Guess the Number (stub)"):
            gr.Markdown("Enter an integer and press Submit. We'll implement the logic later.")
            guess_input = gr.Number(label="Your guess (integer)", precision=0, value=0)
            with gr.Row():
                guess_submit = gr.Button("Submit", variant="primary")
                guess_reset_btn = gr.Button("Reset")
            guess_output = gr.Textbox(label="Result", interactive=False)

            guess_submit.click(fn=guess_game_submit, inputs=guess_input, outputs=guess_output)
            guess_reset_btn.click(fn=guess_game_reset, inputs=None, outputs=guess_output)

        with gr.Tab("Rock-Paper-Scissors (stub)"):
            gr.Markdown("Pick your move and press Play. We'll add an opponent later.")
            rps_choice = gr.Dropdown(choices=_CHOICES, value=None, label="Your move")
            with gr.Row():
                rps_submit = gr.Button("Play", variant="primary")
                rps_reset_btn = gr.Button("Reset")
            rps_output = gr.Textbox(label="Result", interactive=False)

            rps_submit.click(fn=rps_game_submit, inputs=rps_choice, outputs=rps_output)
            rps_reset_btn.click(fn=rps_game_reset, inputs=None, outputs=rps_output)

        gr.Markdown("— © Mini Games Hub")
    return demo

demo = build_app()

cell_end(_st)
```python
# %%
"""
Cell 5: Launch app
- Launches inline in notebooks
- Opens browser when run as a script
"""
_st = cell_start("Launch app")

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and hasattr(ip, "config") and "IPKernelApp" in ip.config
    except Exception:
        return False

def launch_app(app):
    running_in_notebook = _in_notebook()
    common = dict(debug=True)
    if running_in_notebook:
        print("[LAUNCH] Detected notebook. Launching inline...")
        app.queue().launch(inline=True, prevent_thread_lock=True, **common)
    else:
        print("[LAUNCH] Detected script. Launching in browser...")
        # On Windows, open default browser automatically
        app.queue().launch(inbrowser=True, prevent_thread_lock=True, **common)

if __name__ == "__main__":
    launch_app(demo)

cell_end(_st)

# %%






# %%
# "jupyter": "This is a Jupyter notebook.  Each cell is delimited by # %%.  And please provide some debugging output as each cell starts to run and complete. Finally, please ensure that any code you provide is valid Python code that can run in a Jupyter notebook environment or as a single python script:"   
