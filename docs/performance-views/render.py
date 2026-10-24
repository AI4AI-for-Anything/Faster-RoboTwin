#!/usr/bin/env python3
"""Render the Faster-RoboTwin README figures from archived numbers; no GPU or simulator work.

Inputs: estimate.json (full-benchmark estimate), ladder.json (measured throughput ladder) and
time-budget.json (measured cost breakdowns).
Outputs: 01-estimate.svg, 02-ladder.svg, 03-time-budget.svg, 04-ceiling.svg and ../change-views/module-map.svg.
Usage: python docs/performance-views/render.py   (needs matplotlib)
"""
from pathlib import Path
import json
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
RED = '#8C0000'
MID = '#B5797C'
LIGHT = '#D7BDBF'
PALE = '#EBDCDD'
GREY = '#A99A9C'
MUTED = '#79696B'
INK = '#292122'
GRID = '#EADFE0'
CARD = '#FAF5F5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'svg.hashsalt': 'faster-robotwin-crimson', 'font.size': 12, 'hatch.color': LIGHT})


def load(name):
    return json.loads((ROOT / name).read_text())


def save(fig, path):
    target = path.with_suffix('.svg')
    fig.savefig(target, facecolor='white', metadata={'Date': None})
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)


def frame(ax, axis='x'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, labelsize=10.5, colors=MUTED, pad=8)
    ax.grid(axis=axis, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def check_estimate(est):
    wall, gpu, sp = est['wall_h'], est['gpu_h'], est['speedup']
    assert math.isclose(wall['official_default'] / wall['faster'], sp['wall_vs_official_default'], abs_tol=6e-4)
    assert math.isclose(wall['official_longest_first'] / wall['faster'], sp['wall_vs_official_longest_first'], abs_tol=6e-4)
    assert math.isclose(gpu['official'] / gpu['faster'], sp['gpu_time'], abs_tol=6e-4)


def hero():
    est = load('estimate.json')
    check_estimate(est)
    wall, gpu = est['wall_h'], est['gpu_h']
    values = [wall['official_default'], wall['faster']]
    fig = plt.figure(figsize=(9, 4.9), facecolor='white')
    fig.text(.06, .885, 'Faster-RoboTwin, optimized with an AI agent', color=RED, weight='bold', fontsize=19)
    fig.text(.06, .79, 'Official π0.5 evaluation · 50 tasks × 2 settings × 100 episodes · 8× H20 · videos written as usual',
             color=MUTED, fontsize=10.3)
    ax = fig.add_axes([.25, .31, .4, .38])
    ax.barh([1, 0], values, color=[LIGHT, RED], height=.26, zorder=3)
    ax.set_yticks([1, 0], ['Official pipeline', 'Faster-RoboTwin'])
    for y, value in zip([1, 0], values):
        ax.text(value + 1.5, y, f'{value:.1f} h', va='center', color=INK, fontsize=11.5, weight='bold')
    ax.set_xlim(0, 92)
    ax.set_ylim(-.6, 1.6)
    ax.set_xticks([0, 20, 40, 60, 80])
    frame(ax)
    ax.set_xlabel('Wall clock on 8 GPUs, estimated (hours) ↓', color=MUTED, labelpad=10, fontsize=10.5)
    fig.text(.855, .55, f"{est['speedup']['wall_vs_official_default']:.2f}×", color=RED, fontsize=34, weight='bold',
             ha='center')
    fig.text(.855, .45, 'estimated speedup', color=MUTED, fontsize=11.5, ha='center')
    fig.text(.855, .385, 'every episode bitwise identical', color=MUTED, fontsize=9.5, ha='center')
    fig.text(.06, .055, f"Estimate, not a full run: per-step and per-episode costs measured on 6 of 100 jobs "
             f"({est['measured_step_coverage'] * 100:.0f}% of steps), step counts from the public\nleaderboard success "
             f"rates. GPU time {gpu['official']:.1f} → {gpu['faster']:.1f} GPU-h "
             f"({est['speedup']['gpu_time']:.2f}×). Official = eval_policy_multitask.py, 1 job per GPU.",
             color=MUTED, fontsize=9.2, linespacing=1.45)
    save(fig, ROOT / '01-estimate')


def ladder():
    data = load('ladder.json')
    rows = data['rows']
    ceil = data['ceiling']
    base = rows[0]['steps_per_s']
    fig = plt.figure(figsize=(9, 8.4), facecolor='white')
    H = 8.4
    top = lambda inches: 1 - inches / H
    fig.text(.06, top(.42), 'Faster-RoboTwin', fontsize=11, weight='bold', color=RED)
    fig.text(.06, top(.85), 'Measured throughput ladder', fontsize=21, weight='bold', color=INK)
    fig.text(.06, top(1.22), 'place_container_plate / clean · one H20 · steady-state steps per second per GPU · '
             'cumulative configurations', fontsize=10.2, color=MUTED)
    ax = fig.add_axes([.47, top(4.55), .44, 3.0 / H])
    labels = [r['label'] for r in rows] + [ceil['label']]
    values = [r['steps_per_s'] for r in rows] + [ceil['steps_per_s']]
    n = len(values)
    ys = list(range(n - 1, -1, -1))
    for i, (y, v) in enumerate(zip(ys, values)):
        if i == n - 1:
            ax.barh(y, v, height=.56, color='white', edgecolor=RED, linewidth=1.4, hatch='///', zorder=3)
        else:
            ax.barh(y, v, height=.56, color=RED if i == n - 2 else (LIGHT if i == 0 else MID), zorder=3)
        tag = 'derived' if i == n - 1 else 'measured'
        ax.text(v + .2, y, f'{v:.2f}  ({v / base:.2f}×, {tag})', va='center', fontsize=10, color=INK,
                weight='bold' if i == n - 2 else 'normal')
    ax.set_yticks(ys, [f"{lab}  [{r}]" for lab, r in zip(labels, [r['round'] for r in rows] + [ceil['round']])],
                  fontsize=9.6)
    ax.set_xlim(0, 17)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_ylim(-.6, n - .4)
    frame(ax)
    ax.tick_params(axis='y', colors=INK)
    ax.set_xlabel('steps per second per GPU ↑', color=MUTED, labelpad=8, fontsize=10)

    per = data['per_task']['rows']
    fig.text(.06, top(5.25), 'Per task: GPU time of one 100-episode job, official ÷ Faster-RoboTwin (estimate)',
             fontsize=12.5, weight='bold', color=INK)
    bx = fig.add_axes([.47, top(7.55), .44, 1.95 / H])
    order = sorted(per, key=lambda r: r['ratio'])
    for y, r in enumerate(order):
        bx.plot([3.5, r['ratio']], [y, y], color=GRID, linewidth=1.2, zorder=2)
        bx.plot([r['ratio']], [y], marker='o', markersize=8, color=RED, zorder=3)
        bx.text(r['ratio'] + .04, y, f"{r['ratio']:.2f}×", va='center', fontsize=10, color=INK)
    bx.set_yticks(range(len(order)), [f"{r['job']}  ({r['jobs_per_gpu']} jobs/GPU)" for r in order], fontsize=9.6)
    bx.set_xlim(3.5, 5.0)
    bx.set_xticks([3.5, 4.0, 4.5, 5.0], ['3.5×', '4.0×', '4.5×', '5.0×'])
    bx.set_ylim(-.6, len(order) - .4)
    frame(bx)
    bx.tick_params(axis='y', colors=INK)
    note = ('Top: one task; every bar is a measured configuration whose episodes were bitwise identical to the official '
            'baseline; the hatched bar is the GPU-bound ceiling derived from a per-step frame log. Bottom: measured '
            'per-step and per-episode costs combined with leaderboard-based step counts.')
    fig.text(.06, top(8.2), textwrap.fill(note, 122), fontsize=9.1, color=MUTED, linespacing=1.4)
    save(fig, ROOT / '02-ladder')


def stacked(ax, y, parts, colors, height=.5):
    left = 0
    for value, color in zip(parts.values(), colors):
        ax.barh(y, value, left=left, height=height, color=color, edgecolor='white', linewidth=2, zorder=3)
        left += value
    return left


def time_budget():
    data = load('time-budget.json')
    step = data['per_step_one_job']
    job = data['per_job_fixed']
    H = 8.3
    fig = plt.figure(figsize=(9, H), facecolor='white')
    top = lambda inches: 1 - inches / H
    fig.text(.06, top(.55), 'Where the time goes', color=INK, weight='bold', fontsize=21)
    fig.text(.06, top(.92), 'Measured with timing wrappers on 8× H20 (smoke scale); official pipeline vs. '
             'Faster-RoboTwin', color=MUTED, fontsize=10.2)

    fig.text(.06, top(1.55), 'Per step, one job alone on a GPU · stack_blocks_three / clean', fontsize=12.5,
             weight='bold', color=INK)
    ax = fig.add_axes([.17, top(3.25), .77, 1.4 / H])
    off_total = stacked(ax, 1, step['official_ms'], [MID, LIGHT])
    fast_total = stacked(ax, 0, step['faster_ms'], [RED, MID, LIGHT, PALE, PALE])
    render = step['official_ms']['Rendering: 4 cameras every step']
    ax.text(render / 2, 1, f'4-camera rendering every step  {render}', color='white', ha='center', va='center',
            fontsize=10, weight='bold', zorder=4)
    head = step['faster_ms']['Head camera (eval video)']
    ax.text(head / 2, 0, f'head camera {head}', color='white', ha='center', va='center', fontsize=10,
            weight='bold', zorder=4)
    ax.text(off_total + 5, 1, f'{off_total:.0f} ms', va='center', fontsize=10.5, color=INK, weight='bold')
    ax.text(fast_total + 5, 0, f'{fast_total:.0f} ms', va='center', fontsize=10.5, color=INK, weight='bold')
    ax.set_yticks([1, 0], ['Official', 'Faster'])
    ax.set_ylim(-.55, 1.55)
    ax.set_xlim(0, 450)
    frame(ax)
    ax.tick_params(axis='y', colors=INK, labelsize=11)
    ax.set_xlabel('ms per step', color=MUTED, fontsize=10, labelpad=6)
    fast = step['faster_ms']
    note = ('Faster, left to right: ' + ' · '.join(f'{k} {v}' for k, v in fast.items()) + '. Official: everything '
            f"except rendering (actions, inference, communication) {step['official_ms']['Actions, inference and the rest']}.")
    fig.text(.06, top(4.05), textwrap.fill(note, 122), fontsize=9.2, color=MUTED, linespacing=1.4)

    fig.text(.06, top(4.75), 'Per job, fixed start-up', fontsize=12.5, weight='bold', color=INK)
    bx = fig.add_axes([.17, top(6.4), .77, 1.35 / H])
    off_job = stacked(bx, 1, job['official_s'], [RED, MID, LIGHT, PALE, GREY])
    left = 0
    for value in job['official_s'].values():
        if value >= 10:
            bx.text(left + value / 2, 1, f'{value:.0f}', color='white', ha='center', va='center', fontsize=10,
                    weight='bold', zorder=4)
        left += value
    res = job['faster_s']['Resident worker and server, 2nd job onward']
    bx.barh(0, res, height=.5, color=RED, zorder=3)
    bx.text(off_job + 1, 1, f'≈{off_job:.0f} s', va='center', fontsize=10.5, color=INK, weight='bold')
    bx.text(res + 1, 0, '2–3 s from the 2nd job of a resident worker', va='center', fontsize=10.5, color=INK,
            weight='bold')
    bx.set_yticks([1, 0], ['Official', 'Faster'])
    bx.set_ylim(-.55, 1.55)
    bx.set_xlim(0, 75)
    frame(bx)
    bx.tick_params(axis='y', colors=INK, labelsize=11)
    bx.set_xlabel('seconds per job', color=MUTED, fontsize=10, labelpad=6)
    note = 'Official, left to right: ' + ' · '.join(f'{k} {v:g}' for k, v in job['official_s'].items()) + '.'
    fig.text(.06, top(7.2), textwrap.fill(note, 122), fontsize=9.2, color=MUTED, linespacing=1.4)
    lo, hi = data['per_episode_expert_s']['official']
    fig.text(.06, top(8.0), f'Per episode, the expert check (cuRobo planning + physics) costs {lo}–{hi} s for every '
             'tried seed; Faster-RoboTwin replays a recorded\noutcome instead. In the full benchmark the per-step '
             'rendering dominates; start-up is about 1% of the GPU time.', fontsize=9.2, color=MUTED,
             linespacing=1.4)
    save(fig, ROOT / '03-time-budget')


def ceiling():
    est = load('estimate.json')
    check_estimate(est)
    sat = load('time-budget.json')['per_step_gpu_saturated']
    parts = sat['gpu_ms']
    need = sum(parts.values())
    interval = sat['measured_step_interval_ms']
    five = sat['official_step_ms'] / 5
    H = 8.4
    fig = plt.figure(figsize=(9, H), facecolor='white')
    top = lambda inches: 1 - inches / H
    fig.text(.06, top(.55), 'Why it stops near 4.4×', color=INK, weight='bold', fontsize=21)
    fig.text(.06, top(.92), 'With the eval video rendered during evaluation, every step needs one ray-traced head '
             'frame', color=MUTED, fontsize=10.2)

    fig.text(.06, top(1.55), 'GPU work per step · place_container_plate / clean · 14 jobs per GPU', fontsize=12.5,
             weight='bold', color=INK)
    ax = fig.add_axes([.06, top(3.45), .88, 1.25 / H])
    left = 0
    for value, color in zip(parts.values(), [RED, MID, LIGHT]):
        ax.barh(0, value, left=left, height=.42, color=color, edgecolor='white', linewidth=2, zorder=3)
        left += value
    head = parts['Head frame (eval video)']
    ax.text(head / 2, 0, f'head frame (eval video) {head:.1f} ms', color='white', ha='center', va='center',
            fontsize=10.5, weight='bold', zorder=4)
    marks = [(five, f'5× pace {five:.1f}', RED, 'right'), (interval, f'measured {interval:.1f}', INK, 'left')]
    for value, label, color, side in marks:
        ax.axvline(value, color=color, linestyle=(0, (4, 3)), linewidth=1.2, zorder=5)
        ax.text(value + (-.8 if side == 'right' else .8), .5, label, ha=side, va='bottom', fontsize=9.5, color=color)
    ax.text(need - .4, -.4, f'required {need:.1f}', ha='right', va='top', fontsize=9.5, color=INK)
    ax.set_yticks([])
    ax.set_ylim(-.8, .85)
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    frame(ax)
    ax.set_xlabel('ms of GPU time per step', color=MUTED, fontsize=10, labelpad=6)
    note = (f"Required = head frame {head} + wrist frames at chunk start {parts['Wrist frames at chunk start']} + "
            f"inference {parts['Inference']} ms (amortized per step); the run reached {need / interval * 100:.0f}% of it. "
            f"A 5× pace on this task ({sat['official_step_ms']:.0f} ms per official step ÷ 5) is shorter than the head "
            "frame alone. Frames from several processes or threads do not overlap on the GPU; H20 has no RT cores, "
            "and the frame time scales with the 32 samples per pixel.")
    fig.text(.06, top(4.75), textwrap.fill(note, 122), fontsize=9.2, color=MUTED, linespacing=1.4)

    fig.text(.06, top(5.3), 'Full benchmark on 8 GPUs (estimate)', fontsize=12.5, weight='bold', color=INK)
    bx = fig.add_axes([.27, top(7.25), .67, 1.45 / H])
    wall = est['wall_h']
    rows = [('Official, list order', wall['official_default'], LIGHT),
            ('Official, longest first', wall['official_longest_first'], PALE),
            ('Faster-RoboTwin', wall['faster'], RED)]
    for y, (label, value, color) in zip([2, 1, 0], rows):
        bx.barh(y, value, height=.55, color=color, zorder=3)
        bx.text(value + 1, y, f"{value:.1f} h ({wall['official_default'] / value:.2f}×)", va='center', fontsize=10,
                color=INK, weight='bold' if label == 'Faster-RoboTwin' else 'normal')
    cap = wall['official_default'] / est['ceiling']['speedup']
    target = wall['official_default'] / 5
    bx.axvline(target, color=RED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)
    bx.text(target - .6, 2.55, f'5× = {target:.1f} h', ha='right', va='bottom', fontsize=9.2, color=RED)
    bx.axvline(cap, color=INK, linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)
    bx.text(cap + .6, 2.55, f'≈{est["ceiling"]["speedup"]}× ceiling = {cap:.1f} h', ha='left', va='bottom',
            fontsize=9.2, color=INK)
    bx.set_yticks([2, 1, 0], [r[0] for r in rows])
    bx.set_ylim(-.55, 3.1)
    bx.set_xlim(0, 92)
    frame(bx)
    bx.tick_params(axis='y', colors=INK, labelsize=10)
    bx.set_xlabel('hours ↓', color=MUTED, fontsize=10, labelpad=6)
    d = est['deferred_video_optional']
    note = ("Only fewer frames per step goes further. The optional deferred-video mode (not the default) skips the "
            "head frame during evaluation and rebuilds the videos afterwards: about "
            f"{d['speedup_wall'][0]}–{d['speedup_wall'][1]}× to scores (estimate); scores matched on 84 + 25 jobs, "
            "rebuilt videos were md5-identical on 6 episodes.")
    fig.text(.06, top(8.2), textwrap.fill(note, 122), fontsize=9.2, color=MUTED, linespacing=1.4)
    save(fig, ROOT / '04-ceiling')


def modulemap():
    fig = plt.figure(figsize=(9, 8.8), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.055, .94, 'What changed in Faster-RoboTwin', fontsize=22, color=INK, weight='bold')
    ax.text(.055, .905, 'Every switch is off by default; with none set, the wrappers run the official code.',
            fontsize=10.5, color=MUTED)
    cards = [
        ('Render only what is read', 'No front camera (read by neither policy\nnor video); mid-chunk steps render only\n'
         'the head camera, no update_obs', 'EVALOPT_SKIP_UNUSED_CAMS · EVALOPT_LAZY_OBS\nprof_eval.py · opt_loop.py'),
        ('Expert checks recorded once', 'Policy-independent expert-check outcome\nper seed recorded, then replayed\n'
         '(no cuRobo planning or physics)', 'EVALOPT_EXPERT_RECORD / _REPLAY\nexpert_cache.py'),
        ('Resident processes', 'One simulator process and one server run\nmany jobs; server reset to key(0) per job;\n'
         'cuRobo built only when used', 'prof_eval.py --jobs · EVALOPT_SERVER_NEW_JOB\nEVALOPT_LAZY_CUROBO · '
         'prof_server.py'),
        ('Many jobs per GPU', 'One multi-session policy server per GPU:\nshared weights, per-job observation and\n'
         'random-number chain; 10–14 jobs per GPU', 'EVALOPT_MULTISESSION · EVALOPT_MEM_FRACTION\n'
         'multisession_server.py · run_ms_workers.sh'),
        ('Host-side waits and caches', 'CUDA waits block instead of spin\n(76.9 → 0.8 ms CPU per frame);\n'
         'SAPIEN render cache kept', 'EVALOPT_CUDA_BLOCKING_SYNC · EVALOPT_KEEP_CACHE\ncuda_sched.py · prof_eval.py'),
        ('Reproducible baseline', 'XLA autotune results pinned to a file\nrecorded on an idle GPU, so the official\n'
         'baseline repeats bit for bit', 'EVALOPT_AUTOTUNE\nserve_policy.sh · autotune/*.textproto'),
    ]
    for i, (title, description, paths) in enumerate(cards):
        x, y = .055 + (i % 2) * .455, .632 - (i // 2) * .255
        ax.add_patch(FancyBboxPatch((x, y), .425, .225, boxstyle='round,pad=0.009,rounding_size=0.012',
                                    facecolor=CARD, edgecolor=GRID))
        ax.plot([x, x + .425], [y + .225, y + .225], color=RED, linewidth=2, solid_capstyle='round')
        ax.text(x + .014, y + .19, f'{i + 1:02d}', fontsize=11, color=RED, weight='bold')
        ax.text(x + .06, y + .19, title, fontsize=12.5, color=INK, weight='bold')
        ax.text(x + .014, y + .16, description, fontsize=10, color=INK, va='top', linespacing=1.5)
        ax.text(x + .014, y + .018, paths, fontsize=8.9, color=MUTED, linespacing=1.5)
    ax.text(.055, .022, 'Optional, not default: EVALOPT_NO_VIDEO + EVALOPT_RECORD_ACTIONS defer the eval video and\n'
            'replay_video.py rebuilds it. All code is under evalopt/ and scripts/evalopt/; upstream RoboTwin and\n'
            'XPolicyLab files are unchanged.', fontsize=9.6, color=MUTED, linespacing=1.5)
    save(fig, ROOT.parent / 'change-views/module-map')


if __name__ == '__main__':
    hero()
    ladder()
    time_budget()
    ceiling()
    modulemap()
