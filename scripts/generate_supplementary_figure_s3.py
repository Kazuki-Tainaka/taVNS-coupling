#!/usr/bin/env python3
"""Generate the data-driven Supplementary Figure S3 specification landscape."""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cairosvg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BLUE = '#2C7FB8'
TEAL = '#1B9E77'
PURPLE = '#756BB1'
ORANGE = '#D99000'
MID = '#777C83'
LIGHT = '#D8DADD'
VERY_LIGHT = '#F4F4F4'
DARK = '#222222'
GRID = '#E5E5E5'

REFERENCE_SETTING = 'lag1_r0p80_sbp1p0_len3_mean'
SETTING_RE = re.compile(
    r'^lag(?P<lag>[012])_r(?P<r>0p(?:70|80|85))_sbp(?P<sbp>[01]p[05])_len(?P<length>[34])_(?P<aggregation>mean|median)$'
)


@dataclass(frozen=True)
class Specification:
    setting_id: str
    estimate: float
    n: int
    p_value: float | None
    lag: int
    r_threshold: float
    sbp_threshold: float
    min_length: int
    aggregation: str


def _decode_decimal(token: str) -> float:
    return float(token.replace('p', '.'))


def _parse_setting(setting_id: str, estimate: float, n: int, p_value: float | None) -> Specification:
    match = SETTING_RE.match(setting_id)
    if not match:
        raise ValueError(f'Unrecognized full-factorial setting_id: {setting_id}')
    groups = match.groupdict()
    return Specification(
        setting_id=setting_id,
        estimate=estimate,
        n=n,
        p_value=p_value,
        lag=int(groups['lag']),
        r_threshold=_decode_decimal(groups['r']),
        sbp_threshold=_decode_decimal(groups['sbp']),
        min_length=int(groups['length']),
        aggregation=groups['aggregation'],
    )


def _read_rows(path: Path) -> tuple[list[Specification], dict[str, list[float]]]:
    all_specs: list[Specification] = []
    by_direction: dict[str, list[float]] = {'all': [], 'up': [], 'down': []}

    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'record_type', 'metric', 'direction', 'setting_id', 'value', 'n', 'p_value'}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing required columns in {path.name}: {sorted(missing)}')

        for row in reader:
            if row['record_type'] != 'full_factorial':
                continue
            if row['metric'] != 'mean_difference_stim_minus_pre':
                continue
            direction = row['direction'].strip().lower()
            if direction not in by_direction:
                continue
            value = float(row['value'])
            by_direction[direction].append(value)
            if direction == 'all':
                p_text = row.get('p_value', '').strip()
                p_value = None if p_text in {'', 'NA', 'NaN', 'nan'} else float(p_text)
                all_specs.append(
                    _parse_setting(
                        setting_id=row['setting_id'].strip(),
                        estimate=value,
                        n=int(float(row['n'])),
                        p_value=p_value,
                    )
                )

    if len(all_specs) != 72:
        raise ValueError(f'Expected 72 all-sequence specifications, found {len(all_specs)}')
    for direction, values in by_direction.items():
        if len(values) != 72:
            raise ValueError(f'Expected 72 {direction} estimates, found {len(values)}')
    if sum(spec.setting_id == REFERENCE_SETTING for spec in all_specs) != 1:
        raise ValueError(f'Reference setting {REFERENCE_SETTING!r} was not found exactly once')
    return all_specs, by_direction


def configure_style() -> None:
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Arimo', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 7.5,
        'axes.titlesize': 8.8,
        'axes.titleweight': 'bold',
        'axes.labelsize': 7.8,
        'xtick.labelsize': 6.8,
        'ytick.labelsize': 6.8,
        'axes.linewidth': 0.8,
        'svg.fonttype': 'none',
        'svg.hashsalt': 'tavns-srep-s3-specification-landscape',
        'savefig.transparent': False,
        'axes.unicode_minus': True,
    })


def _panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.07) -> None:
    ax.text(
        x, y, label, transform=ax.transAxes, ha='left', va='top',
        fontsize=12.5, fontweight='bold', color=DARK, clip_on=False,
    )


def _clean_axis(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(DARK)
    ax.spines['bottom'].set_color(DARK)
    if ygrid:
        ax.grid(axis='y', color=GRID, linewidth=0.65, zorder=0)
    if xgrid:
        ax.grid(axis='x', color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def make_s3_specification_landscape(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    configure_style()
    all_specs, by_direction = _read_rows(data_path)
    specs = sorted(all_specs, key=lambda spec: (spec.estimate, spec.setting_id))
    x = np.arange(1, len(specs) + 1)
    estimates = np.array([spec.estimate for spec in specs], dtype=float)
    n_values = np.array([spec.n for spec in specs], dtype=int)
    ref_index = next(i for i, spec in enumerate(specs) if spec.setting_id == REFERENCE_SETTING)
    ref_x = x[ref_index]
    ref_value = estimates[ref_index]
    median_value = float(np.median(estimates))

    fig = plt.figure(figsize=(7.25, 5.15))
    gs = fig.add_gridspec(
        3, 2,
        left=0.135, right=0.985, bottom=0.105, top=0.965,
        wspace=0.27, hspace=0.18,
        width_ratios=[4.75, 1.55],
        height_ratios=[2.05, 0.72, 2.35],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)
    ax_d = fig.add_subplot(gs[:, 1])

    # Panel a
    point_colors = np.where(estimates < 0, BLUE, ORANGE)
    ax_a.plot(x, estimates, color=LIGHT, linewidth=0.7, zorder=1)
    ax_a.scatter(x, estimates, c=point_colors, s=18, edgecolors='white', linewidths=0.35, zorder=3)
    ax_a.scatter([ref_x], [ref_value], marker='D', s=58, facecolors='white', edgecolors=DARK, linewidths=1.25, zorder=5)
    ax_a.axhline(0, color=DARK, linewidth=0.9, linestyle=(0, (3, 3)), zorder=2)
    ax_a.axhline(median_value, color=MID, linewidth=0.85, linestyle=(0, (1.5, 2.2)), zorder=2)
    ax_a.axvline(ref_x, color=DARK, linewidth=0.55, alpha=0.55, zorder=0)
    ax_a.set_xlim(0.25, 72.75)
    ax_a.set_ylim(-2.95, 0.62)
    ax_a.set_yticks([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5])
    ax_a.set_ylabel('Mean paired Stim–Pre difference\n(ms/mmHg)')
    ax_a.set_title('All-sequence BRS specification curve', loc='left', pad=4.5)
    ax_a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    _clean_axis(ax_a)
    _panel_label(ax_a, 'a', x=-0.105, y=1.08)

    positive_count = int(np.sum(estimates > 0))
    p_positive = [spec.p_value for spec in specs if spec.estimate > 0 and spec.p_value is not None]
    p_note = f'all nominal $p$ ≥ {min(p_positive):.3f}' if p_positive else 'no inferential support'
    ax_a.text(
        0.985, 0.06,
        f'Range: {estimates.min():.2f} to +{estimates.max():.2f}\n'
        f'{positive_count}/72 positive (orange; {p_note})',
        transform=ax_a.transAxes, ha='right', va='bottom', fontsize=6.6, color=DARK,
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=LIGHT, linewidth=0.6),
    )
    ax_a.annotate(
        f'Reference = {ref_value:.2f}',
        xy=(ref_x, ref_value), xycoords='data',
        xytext=(22, -8), textcoords='offset points',
        ha='left', va='top', fontsize=6.35, color=DARK,
        arrowprops=dict(arrowstyle='-', color=DARK, linewidth=0.6),
    )
    ax_a.text(
        13.2, median_value + 0.08, f'Median = {median_value:.2f}',
        ha='left', va='bottom', fontsize=6.2, color=MID,
    )

    # Panel b
    ax_b.scatter(x, n_values, s=20, color='#B9C0C7', edgecolors='none', zorder=2)
    ax_b.scatter([ref_x], [n_values[ref_index]], marker='D', s=35, facecolors='white', edgecolors=DARK, linewidths=1.05, zorder=4)
    ax_b.axvline(ref_x, color=DARK, linewidth=0.55, alpha=0.55, zorder=0)
    ax_b.set_ylim(10.5, 18.6)
    ax_b.set_yticks([11, 14, 18])
    ax_b.set_ylabel('Evaluable\nparticipants')
    ax_b.set_title('All-sequence evaluable sample size', loc='left', pad=2.2)
    ax_b.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    _clean_axis(ax_b, ygrid=True)
    _panel_label(ax_b, 'b', x=-0.105, y=1.17)
    ax_b.text(0.995, 0.89, f'$n$ = {n_values.min()}–{n_values.max()}', transform=ax_b.transAxes,
              ha='right', va='top', fontsize=6.3, color=MID)

    # Panel c
    level_rows: list[tuple[str, object]] = [
        ('Lag 0', ('lag', 0)),
        ('Lag 1', ('lag', 1)),
        ('Lag 2', ('lag', 2)),
        ('$r$ ≥ 0.70', ('r_threshold', 0.70)),
        ('$r$ ≥ 0.80', ('r_threshold', 0.80)),
        ('$r$ ≥ 0.85', ('r_threshold', 0.85)),
        ('SBP ≥ 0.5', ('sbp_threshold', 0.5)),
        ('SBP ≥ 1.0', ('sbp_threshold', 1.0)),
        ('Min. length 3', ('min_length', 3)),
        ('Min. length 4', ('min_length', 4)),
        ('Mean', ('aggregation', 'mean')),
        ('Median', ('aggregation', 'median')),
    ]
    y_positions = np.arange(len(level_rows))[::-1]
    group_bounds = [(0, 3), (3, 6), (6, 8), (8, 10), (10, 12)]
    for gi, (start, stop) in enumerate(group_bounds):
        if gi % 2 == 0:
            low = len(level_rows) - stop - 0.5
            high = len(level_rows) - start - 0.5
            ax_c.axhspan(low, high, color=VERY_LIGHT, zorder=0)
    for row_idx, (_, (attr, level)) in enumerate(level_rows):
        y = y_positions[row_idx]
        active_x = [idx for idx, spec in zip(x, specs) if getattr(spec, attr) == level]
        ax_c.scatter(active_x, np.full(len(active_x), y), s=9.5, color=DARK, linewidths=0, zorder=3)
    for sep in [2.5, 5.5, 7.5, 9.5]:
        ax_c.axhline(len(level_rows) - 1 - sep, color=LIGHT, linewidth=0.75, zorder=1)
    ax_c.axvline(ref_x, color=DARK, linewidth=0.65, alpha=0.65, zorder=2)
    ax_c.set_yticks(y_positions, [label for label, _ in level_rows])
    ax_c.set_ylim(-0.65, len(level_rows) - 0.35)
    ax_c.set_xticks([1, 12, 24, 36, 48, 60, 72])
    ax_c.set_xlabel('Specifications ordered by all-sequence Stim–Pre estimate')
    ax_c.set_title('Parameter settings for each specification', loc='left', pad=3.5)
    ax_c.tick_params(axis='y', length=0, pad=2)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.spines['left'].set_visible(False)
    ax_c.spines['bottom'].set_color(DARK)
    _panel_label(ax_c, 'c', x=-0.105, y=1.08)

    # Panel d
    direction_order = ['all', 'up', 'down']
    labels = ['All', 'Up', 'Down']
    colors = [BLUE, TEAL, PURPLE]
    counts = [sum(v < 0 for v in by_direction[d]) for d in direction_order]
    percentages = [100.0 * count / 72.0 for count in counts]
    y_d = np.array([2, 1, 0])
    ax_d.barh(y_d, percentages, color=colors, height=0.50, edgecolor='none', zorder=2)
    for y, percentage, count in zip(y_d, percentages, counts):
        ax_d.text(min(percentage - 2.0, 96.0), y, f'{count}/72', ha='right', va='center',
                  fontsize=7.2, color='white', fontweight='bold', zorder=3)
    ax_d.set_xlim(0, 100)
    ax_d.set_ylim(-0.85, 2.55)
    ax_d.set_yticks(y_d, labels)
    ax_d.set_xticks([0, 50, 100])
    ax_d.set_xlabel('Specifications with a negative\nStim–Pre estimate (%)')
    ax_d.set_title('Direction consistency', loc='left', pad=4.5)
    _clean_axis(ax_d, xgrid=True, ygrid=False)
    _panel_label(ax_d, 'd', x=-0.20, y=1.035)
    ax_d.text(
        0.015, -0.44,
        'Descriptive total:\n188/216 negative (87.0%)',
        transform=ax_d.get_yaxis_transform(), ha='left', va='top', fontsize=6.7, color=DARK,
        bbox=dict(boxstyle='round,pad=0.28', facecolor=VERY_LIGHT, edgecolor=LIGHT, linewidth=0.6),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / 'supplementary_figure_s3_brs_specification_landscape.svg'
    png_path = output_dir / 'supplementary_figure_s3_brs_specification_landscape.png'
    fig.savefig(
        svg_path,
        format='svg',
        facecolor='white',
        metadata={
            'Title': 'Supplementary Figure S3. BRS implementation specification landscape',
            'Creator': 'generate_supplementary_figure_s3.py',
            'Description': 'Vector-only live-text SVG; all 72 all-sequence specifications, evaluable sample size, parameter matrix, and direction summary.',
            'Date': None,
        },
    )
    plt.close(fig)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=300, background_color='white')
    return svg_path, png_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Generate the Supplementary Figure S3 specification landscape.'
    )
    parser.add_argument('--supplementary-data-3', type=Path,
                        default=root / 'expected_outputs' / 'publication_source_data'
                        / 'supplementary_data_3_brs_sensitivity_and_coupling_significance.csv',
                        help='Path to Supplementary Data 3 CSV.')
    parser.add_argument('--output-dir', type=Path,
                        default=root / 'generated_outputs',
                        help='Output directory for SVG and PNG.')
    args = parser.parse_args()
    svg_path, png_path = make_s3_specification_landscape(args.supplementary_data_3.resolve(), args.output_dir.resolve())
    print(svg_path)
    print(png_path)


if __name__ == '__main__':
    main()
