"""Shared acquisition-clock histogram for the overview and detailed page."""
from __future__ import annotations

import numpy as np
from matplotlib.ticker import MultipleLocator

OVERVIEW_NORMAL = '#2176ae'
OVERVIEW_RECEIVE_GAP = '#15803d'
OVERVIEW_FPGA_LOSS = '#c0392b'
OVERVIEW_TEXT_SIZE = 7

STATS_BOX = dict(boxstyle='round,pad=0.45', fc='#fbf6d9', ec='#c9c3a0', lw=.8)


def _draw_bars(ax, positions, steps, counts, fontsize, *, color='#2a78d6'):
    bars = ax.bar(positions, counts, width=.85, color=color, lw=0)
    ax.set_yscale('log')
    for bar, step, count in zip(bars, steps, counts):
        ax.annotate(f'{step} ms: {count:,}',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                    rotation=90, fontsize=fontsize)
    ax.set_ylim(.6, max(int(counts.max()) if counts.size else 1, 1) * 100)
    ax.grid(axis='y', color='#e6e5e0', lw=.5)
    ax.set_axisbelow(True)


def draw_clock_histogram(ax, fpga, *, compact=False):
    """Show steps above 100 ms separately, at their exact recorded values."""
    delta = fpga['delta'][fpga['valid']]
    steps, counts = np.unique(delta, return_counts=True)
    tail = steps > 100
    colors = '#2a78d6'
    if compact:
        loss_steps = np.unique(fpga['delta'][fpga['lost'] > 0])
        colors = np.where(np.isin(steps, loss_steps), OVERVIEW_FPGA_LOSS, OVERVIEW_NORMAL)
    _draw_bars(ax, steps[~tail], steps[~tail], counts[~tail], 6 if compact else 8,
               color=colors[~tail] if compact else colors)
    for multiple in range(2, 5):
        ax.axvline(multiple * 19.2, color='#52514e', lw=.6, ls=':', alpha=.45, zorder=0)
    ax.set_xlim(-2, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel('FPGA clock step / ms')
    ax.set_ylabel('Clock step count (log)' if compact else 'Intervals (log)')
    ax.grid(True, color='#e6e5e0', lw=.5)
    ax.set_axisbelow(True)
    normal = int(np.count_nonzero(delta <= 20))
    total = len(delta)
    percent = lambda count: count / total if total else 0.
    box = (f'valid intervals: {total:,}\n'
           f'Δ ≤ 20 ms: {normal:,} ({percent(normal):.2%})\n'
           f'Δ > 20 ms: {total - normal:,} ({percent(total - normal):.3%})\n'
           f'frames lost: {fpga["frames_lost"]:,} ({fpga["drop_rate"]:.3%})\n'
           f'time lost: {fpga["lost_seconds"]:.2f} s')
    if compact:
        events = int(np.count_nonzero(fpga['lost'] > 0))
        box = (f'Valid clock steps: {total:,}\n'
               f'Inferred loss events: {events:,}\n'
               f'Inferred missing frames: {fpga["frames_lost"]:,}\n'
               f'Inferred frame loss rate: {fpga["drop_rate"]:.3%}')
    if fpga['n_nonpositive_steps']:
        box += f'\nReset/wrap steps excluded: {fpga["n_nonpositive_steps"]:,}; loss incomplete'
    ax.text(.985, .97, box, transform=ax.transAxes, ha='right', va='top',
            fontsize=OVERVIEW_TEXT_SIZE if compact else 8,
            bbox=STATS_BOX)
    if tail.any():
        inset = ax.inset_axes([.65, .37, .33, .31])
        # Even spacing keeps rare steps visible when their magnitudes differ greatly.
        _draw_bars(inset, np.arange(tail.sum()), steps[tail], counts[tail], 6,
                   color=colors[tail] if compact else colors)
        inset.set_xticks([])
        inset.tick_params(axis='y', labelsize=6)
        inset.set_title('Steps >100 ms (log count)', fontsize=7)
    return dict(position_ms=steps, count=counts, is_overflow=tail)
