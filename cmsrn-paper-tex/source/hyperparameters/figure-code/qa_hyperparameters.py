"""Validate the final compact hyperparameter figures without modifying source data."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import pymupdf


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill-scripts', type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[3]
    source = package / 'source/hyperparameters'
    audit = source / 'figure-audit'
    figures = package / 'figures'

    def audit_script(script, target, *options):
        result = subprocess.run(
            [sys.executable, '-B', str(args.skill_scripts / script),
             str(target.relative_to(package)), *options, '--json'],
            cwd=package, capture_output=True, text=True, encoding='utf8', check=True)
        return json.loads(result.stdout)

    preflight = audit_script('validate_figure.py', source / 'figure-code/plot_hyperparameters.py')
    write_json(audit / 'source-validation.json', preflight)
    assert preflight['summary']['counts']['FAIL'] == 0
    originals = json.loads((source / 'source-sha256.json').read_text(encoding='utf8'))
    assert all(sha256(source / 'original' / name) == digest for name, digest in originals.items())
    csv_path = source / 'original/chengdu_hyperparameter_values.csv'
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    key = lambda row: (row['parameter'], row['value'], row['channel'])
    lookup = {key(row): row for row in rows}
    assert len(rows) == len(lookup) == 60
    seen = set()
    manifest = []
    for mode, old_height, height, count in [('joint', 122, 106, 4), ('flow', 202, 184, 8)]:
        stem = f'chengdu-{mode}-hyperparameter-paper'
        pdf = figures / f'hyperparameter-{mode}.pdf'
        alignment = json.loads((audit / f'{stem}.alignment.json').read_text(encoding='utf8'))
        assert alignment['verdict'] == 'PASS'
        panels = alignment['layout']['panels']
        assert len(panels) == count
        assert all(panel['panel_label_anchor_pt'][1] < panel['bbox_pt'][1] for panel in panels)
        collisions = json.loads((audit / f'{stem}.collision.json').read_text(encoding='utf8'))
        assert collisions['verdict'] == 'PASS' and collisions['summary']['warn'] == 0
        write_json(audit / f'{stem}.audit_figure_collisions.json', collisions)
        text_audit = audit_script('audit_pdf_text.py', pdf, '--min-pt', '5')
        assert text_audit['auditable'] and text_audit['below_minimum_count'] == 0
        write_json(audit / f'{stem}.audit_pdf_text.json', text_audit)
        data = json.loads((audit / f'{stem}.plot-data.json').read_text(encoding='utf8'))['data']
        expected = []
        for row in data:
            row_key = key(row)
            assert row_key not in seen
            seen.add(row_key)
            for metric in ['mae', 'rmse']:
                assert float(row[metric]) == float(lookup[row_key][metric])
                expected.append(f'{float(row[metric]):.4f}')
        with pymupdf.open(pdf) as document:
            page = document[0]
            dimensions = [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72]
            assert abs(dimensions[0] - 160) < .001 and abs(dimensions[1] - height) < .001
            numeric_labels = re.findall(r'\b[01]\.\d{4}\b', page.get_text())
            assert Counter(numeric_labels) == Counter(expected)
            page.get_pixmap(dpi=150).save(audit / f'{stem}.final-preview.png')
        manifest.append({
            'figure': stem, 'size_mm': dimensions, 'previous_height_mm': old_height,
            'height_reduction_percent': round(100 * (old_height - height) / old_height, 2),
            'minimum_font_pt': text_audit['minimum_found_pt'],
            'alignment': alignment['verdict'], 'collision': collisions['verdict'],
            'panel_captions': 'All below plot areas; flow names included in bottom captions',
            'numeric_labels_checked': len(expected), 'sha256': sha256(pdf),
        })
    assert len(seen) == 60
    write_json(audit / 'final-figure-manifest.json', manifest)
    write_json(audit / 'compact-numeric-audit.json', {
        'source_rows': len(rows), 'plotted_rows': len(seen), 'plotted_values': 120,
        'all_source_files_unchanged': True, 'pdf_numeric_labels_match_source': True,
        'all_panel_captions_below_plot_areas': True,
        'source_csv_sha256': sha256(csv_path),
    })
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
