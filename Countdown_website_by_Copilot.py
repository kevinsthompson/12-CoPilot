# %%
{
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

# %%


# "jupyter": "This is a Jupyter notebook.  Each cell is delimited by # %%.  And please provide some debugging output as each cell starts to run and complete. Finally, please ensure that any code you provide is valid Python code that can run in a Jupyter notebook environment or as a single python script:"   


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
Cell 3: Countdown Numbers Game + RPS helpers
Replaces the old 'Guess the Number' with the Countdown Numbers game.
"""
_st = cell_start("Define game logic")

import random

# Countdown Numbers Game setup
_LARGE = [25, 50, 75, 100]
_SMALL = [n for n in range(1, 11)] * 2  # two of each 1-10
_current_numbers: list[int] = []
_current_target: int | None = None

def countdown_deal(large_count: int):
    global _current_numbers, _current_target
    try:
        if not (0 <= large_count <= 4):
            return "Choose between 0 and 4 large numbers."
        large = random.sample(_LARGE, large_count)
        small = random.sample(_SMALL, 6 - large_count)
        _current_numbers = large + small
        random.shuffle(_current_numbers)
        _current_target = random.randint(100, 999)
        nums = " ".join(str(n) for n in _current_numbers)
        return f"Target: {_current_target}\nNumbers: {nums}"
    except Exception:
        traceback.print_exc()
        return "Error dealing numbers."

def countdown_check(final_result):
    if _current_target is None or not _current_numbers:
        return "Deal numbers first."
    if final_result is None:
        return "Enter a result."
    try:
        val = int(final_result)
    except Exception:
        return "Result must be an integer."
    diff = abs(val - _current_target)
    if diff == 0:
        return f"Perfect! You hit {val} exactly."
    elif diff <= 5:
        return f"Close! {val} (off by {diff})."
    else:
        return f"{val} (off by {diff}). Target was {_current_target}."

def countdown_reset():
    global _current_numbers, _current_target
    _current_numbers = []
    _current_target = None
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
- Countdown Numbers Game
- Rock-Paper-Scissors stub
"""
_st = cell_start("Build Gradio UI")

import gradio as gr  # type: ignore

def build_app():
    with gr.Blocks(title="Mini Games Hub") as demo:
        gr.Markdown("# Mini Games Hub\nCountdown Numbers Game + Rock-Paper-Scissors (stub).")
        with gr.Tab("Countdown Numbers Game"):
            gr.Markdown("Select large numbers count (0–4) then Deal. Try to reach the target using arithmetic. Enter the final result you achieved.")
            large_input = gr.Slider(minimum=0, maximum=4, step=1, value=2, label="Large numbers count")
            with gr.Row():
                deal_btn = gr.Button("Deal", variant="primary")
                reset_btn = gr.Button("Reset")
            numbers_output = gr.Textbox(label="Board", interactive=False)
            final_result = gr.Number(label="Your final result", precision=0)
            check_btn = gr.Button("Check")
            check_output = gr.Textbox(label="Feedback", interactive=False)

            deal_btn.click(fn=countdown_deal, inputs=large_input, outputs=numbers_output)
            reset_btn.click(fn=countdown_reset, inputs=None, outputs=numbers_output)
            check_btn.click(fn=countdown_check, inputs=final_result, outputs=check_output)

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

stop_app()

%
