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
import time
from functools import lru_cache
from typing import List, Optional, Tuple

_LARGE = [25, 50, 75, 100]
_SMALL = [n for n in range(1, 11)] * 2  # two of each 1-10

_current_numbers: List[int] = []
_current_target: Optional[int] = None


def countdown_deal_random(large_count: int):
    """Randomly deal numbers (does NOT change target)."""
    global _current_numbers
    if not (0 <= large_count <= 4):
        return [None]*7
    large = random.sample(_LARGE, large_count)
    small = random.sample(_SMALL, 6 - large_count)
    nums = large + small
    random.shuffle(nums)
    _current_numbers = nums[:]
    return [*nums, _current_target]

def countdown_generate_target():
    """Generate a random target (keeping chosen numbers)."""
    global _current_target
    _current_target = random.randint(100, 999)
    return _current_target

def countdown_reset():
    """Clear board."""
    global _current_numbers, _current_target
    _current_numbers = []
    _current_target = None
    return [None, None, None, None, None, None, None, ""]

def _validate_selection(nums: List[Optional[int]]) -> Optional[str]:
    picked = [n for n in nums if n is not None]
    if len(picked) != 6:
        return "Pick exactly 6 numbers."
    large_used = sum(1 for n in picked if n in _LARGE)
    if large_used > 4:
        return "At most 4 large numbers."
    for L in _LARGE:
        if picked.count(L) > 1:
            return f"Large number {L} at most once."
    for s in range(1, 11):
        if picked.count(s) > 2:
            return f"Small number {s} at most twice."
    return None

def _solve(numbers: List[int], target: int, time_limit: float = 3.0) -> Tuple[int, List[str]]:
    start = time.perf_counter()
    best_val: Optional[int] = None
    best_steps: List[str] = []
    best_diff = float("inf")

    def update(val: int, steps: List[str]):
        nonlocal best_val, best_steps, best_diff
        d = abs(val - target)
        if d < best_diff or (d == best_diff and len(steps) < len(best_steps)):
            best_val, best_steps, best_diff = val, steps[:], d

    @lru_cache(maxsize=None)
    def search(state: Tuple[int, ...]) -> None:
        if time.perf_counter() - start > time_limit:
            return
        if len(state) == 1:
            update(state[0], [])
            return
        n = len(state)
        for i in range(n):
            a = state[i]
            for j in range(i+1, n):
                b = state[j]
                rest = list(state[:i] + state[i+1:j] + state[j+1:])
                ops: List[Tuple[int, str]] = []
                ops.append((a + b, f"{a} + {b} = {a + b}"))
                if a != 1 and b != 1:
                    ops.append((a * b, f"{a} × {b} = {a * b}"))
                if a > b:
                    ops.append((a - b, f"{a} - {b} = {a - b}"))
                if b > a:
                    ops.append((b - a, f"{b} - {a} = {b - a}"))
                if b != 0 and a % b == 0 and b != 1:
                    ops.append((a // b, f"{a} ÷ {b} = {a // b}"))
                if a != 0 and b % a == 0 and a != 1:
                    ops.append((b // a, f"{b} ÷ {a} = {b // a}"))
                for val, desc in ops:
                    if val <= 0:
                        continue
                    new_state = tuple(sorted(rest + [val]))
                    prev_best = (best_val, best_diff)
                    search(new_state)
                    if best_val != prev_best[0] or abs(best_val - target) < prev_best[1]:
                        # If improved, prepend this step (build from bottom)
                        if desc not in best_steps:
                            best_steps.insert(0, desc)
                    if best_diff == 0:
                        return

    search(tuple(sorted(numbers)))
    assert best_val is not None
    return best_val, best_steps

def countdown_solve(n1, n2, n3, n4, n5, n6, target):
    nums = [n1, n2, n3, n4, n5, n6]
    msg = _validate_selection(nums)
    if msg:
        return msg
    if target is None:
        return "Enter a target (100–999) or use Random Target."
    try:
        t = int(target)
    except:
        return "Target must be integer."
    if not (100 <= t <= 999):
        return "Target must be 100–999."
    val, steps = _solve([int(x) for x in nums], t)
    diff = abs(val - t)
    header = f"Target: {t}\nNumbers: {' '.join(str(x) for x in nums)}\nBest: {val} (off by {diff})"
    if not steps:
        return header + "\nNo operations needed."
    body = "\n".join(f"{i+1}) {s}" for i, s in enumerate(steps))
    return header + "\n\nSteps:\n" + body

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

_ALL_CHOICES = [*range(1, 11), 25, 50, 75, 100]

def build_app():
    with gr.Blocks(title="Mini Games Hub") as demo:
        gr.Markdown("# Mini Games Hub\nCountdown Numbers Game + Rock-Paper-Scissors (stub).")

        with gr.Tab("Countdown Numbers Game"):
            gr.Markdown("Select 6 numbers manually or use Random Deal. Then set/roll a target and click Solve.")
            with gr.Row():
                large_input = gr.Slider(minimum=0, maximum=4, step=1, value=2, label="Large numbers count (Random Deal)")
                deal_btn = gr.Button("Random Deal", variant="primary")
                rand_target_btn = gr.Button("Random Target")
                reset_btn = gr.Button("Reset")

            with gr.Row():
                n1 = gr.Dropdown(choices=_ALL_CHOICES, label="Number 1")
                n2 = gr.Dropdown(choices=_ALL_CHOICES, label="Number 2")
                n3 = gr.Dropdown(choices=_ALL_CHOICES, label="Number 3")
                n4 = gr.Dropdown(choices=_ALL_CHOICES, label="Number 4")
                n5 = gr.Dropdown(choices=_ALL_CHOICES, label="Number 5")
                n6 = gr.Dropdown(choices=_ALL_CHOICES, label="Number 6")

            target_in = gr.Number(label="Target (100–999)", precision=0)
            solve_btn = gr.Button("Solve", variant="primary")
            solution_out = gr.Textbox(label="Solution", interactive=False, lines=10)
            deal_btn.click(countdown_deal_random, inputs=large_input, outputs=[n1, n2, n3, n4, n5, n6, target_in])
            rand_target_btn.click(countdown_generate_target, outputs=target_in)
            reset_btn.click(countdown_reset, outputs=[n1, n2, n3, n4, n5, n6, target_in, solution_out])
            solve_btn.click(countdown_solve, inputs=[n1, n2, n3, n4, n5, n6, target_in], outputs=solution_out)

        with gr.Tab("Rock-Paper-Scissors (stub)"):
            gr.Markdown("Pick your move and press Play.")
            rps_choice = gr.Dropdown(choices=_CHOICES, label="Your move")
            with gr.Row():
                rps_submit = gr.Button("Play", variant="primary")
                rps_reset_btn = gr.Button("Reset")
            rps_output = gr.Textbox(label="Result", interactive=False)
            rps_submit.click(rps_game_submit, inputs=rps_choice, outputs=rps_output)
            rps_reset_btn.click(rps_game_reset, outputs=rps_output)

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


# %%
