"""Reposition existing vector content; no diagram redrawing or font replacement.

The archived Visio exports are immutable inputs. Body objects retain their
relative coordinates; only the three panel frames and caption positions change.
Run after finish_figure_export.py; --refresh-input replaces the archived input
when the original Visio export has deliberately been regenerated.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from pypdf.generic import FloatObject, RectangleObject

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
FIG = ROOT / "figures"
ARCHIVE = SRC / "before-bottom-panel-titles"
NEW_H = 253.0
TOP_SHIFT, LOWER_SHIFT = 25.0, 14.0
TOP_TRIM = 17.69


def path_indices(group):
    """Distinguish painted geometry from the original full-page clip paths."""
    painted, clips, pending = set(), [], []
    for i, (args, op) in enumerate(group):
        if op in (b"m", b"l", b"c", b"re", b"h"):
            pending.append(i)
        elif op in (b"W", b"W*"):
            clips.append(pending[:])
            pending = []
        elif op in (b"S", b"s", b"f", b"f*", b"B", b"B*"):
            painted.update(pending)
            pending = []
        elif op == b"n":
            pending = []
    return painted, clips


def y_values(args, op):
    if op == b"re":
        return [float(args[1]), float(args[1]) + float(args[3])]
    if op in (b"m", b"l", b"c"):
        return [float(args[i]) for i in range(1, len(args), 2)]
    if op == b"Tm":
        return [float(args[5])]
    return []


def adjust_pdf():
    reader = PdfReader(ARCHIVE / "model-architecture.pdf")
    page = reader.pages[0]
    before_text = page.extract_text()
    ops = page.get_contents().operations
    groups, current = [], []
    for args, op in ops:
        current.append((copy.deepcopy(args), op))
        if op == b"EMC":
            groups.append(current)
            current = []
    assert not current and len(groups) == 256
    counts = {"top": 0, "lower": 0, "caption_a_text_runs": 0,
              "caption_b_text_runs": 0, "caption_c_text_runs": 0}
    result = []
    for group in groups:
        mcid = int(group[0][0][1]["/MCID"])
        painted, clips = path_indices(group)
        text_indices = [i for i, (_, op) in enumerate(group) if op == b"Tm"]
        ys = [y for i in painted | set(text_indices)
              for y in y_values(*group[i])]
        # Original page-outline path was already disabled and stays disabled.
        if not ys:
            shift = 0
        else:
            assert max(ys) < 134.0 or min(ys) > 135.0, (mcid, ys)
            shift = TOP_SHIFT if min(ys) > 135 else LOWER_SHIFT
            counts["top" if shift == TOP_SHIFT else "lower"] += 1

        for i in painted | set(text_indices):
            args, op = group[i]
            if op == b"Tm":
                old_y = float(args[5])
                if abs(old_y - 231.86) < .01:
                    args[5] = FloatObject(150.0)
                    counts["caption_a_text_runs"] += 1
                elif abs(old_y - 125.02) < .01 and abs(float(args[4])-77.76) < .01:
                    args[4] = FloatObject(float(args[4]) - 1.926)
                    args[5] = FloatObject(4.5)
                    counts["caption_b_text_runs"] += 1
                elif abs(old_y - 122.33) < .01 and abs(float(args[4])-304.56) < .01:
                    args[4] = FloatObject(float(args[4]) + 15.045)
                    args[5] = FloatObject(4.5)
                    counts["caption_c_text_runs"] += 1
                else:
                    args[5] = FloatObject(old_y + shift)
            elif op in (b"m", b"l", b"c"):
                for j in range(1, len(args), 2):
                    y = float(args[j])
                    # Shorten only the formerly caption-bearing frame edges.
                    if mcid == 1 and y > 230:
                        y -= TOP_TRIM
                    elif mcid == 4 and y > 125:
                        y -= 3.12
                    elif mcid == 7 and y > 125:
                        y -= 2.22
                    args[j] = FloatObject(y + shift)
            elif op == b"re":
                args[1] = FloatObject(float(args[1]) + shift)
        # The original full-page clips were smaller than the relocated page.
        clip_replacement, clip_skip = {}, set()
        for indices in clips:
            clip_ys = [y for i in indices for y in y_values(*group[i])]
            assert min(clip_ys) < .01 and max(clip_ys) > 244.55
            clip_replacement[indices[0]] = ([FloatObject(0), FloatObject(0),
                FloatObject(float(page.mediabox.width)), FloatObject(NEW_H)], b"re")
            clip_skip.update(indices[1:])
        result.extend(clip_replacement.get(i, row) for i, row in enumerate(group)
                      if i not in clip_skip)
    assert [counts[k] for k in ("caption_a_text_runs", "caption_b_text_runs",
                                "caption_c_text_runs")] == [10, 1, 1]
    content = page.get_contents()
    content.operations = result
    writer = PdfWriter()
    writer.add_page(page)
    output = writer.pages[0]
    output.mediabox = RectangleObject([0, 0, float(page.mediabox.width), NEW_H])
    output.cropbox = RectangleObject(output.mediabox)
    output.replace_contents(content)
    with (FIG / "model-architecture.pdf").open("wb") as stream:
        writer.write(stream)
    after = PdfReader(FIG / "model-architecture.pdf")
    assert after.pages[0].extract_text() == before_text, "Figure text changed"
    return counts


def adjust_svg():
    # SVG retains the native Visio shapes and their text, including math glyphs.
    namespace = "http://www.w3.org/2000/svg"
    visio = "http://schemas.microsoft.com/visio/2003/SVGExtensions/"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    ET.register_namespace("v", visio)
    root = ET.parse(ARCHIVE / "model-architecture.svg").getroot()
    _, _, old_w, old_h = map(float, root.get("viewBox").split())
    root.set("height", f"{NEW_H/72:.9f}in")
    root.set("viewBox", f"0 0 {old_w:g} {NEW_H:g}")
    ns = {"v": "http://schemas.microsoft.com/office/visio/2012/main"}
    with zipfile.ZipFile(SRC / "formatted-model-architecture.vsdx") as z:
        source = ET.fromstring(z.read("visio/pages/page1.xml"))
    positions = {s.get("ID"): float(s.find("v:Cell[@N='PinY']", ns).get("V"))*72
                 for s in source.findall(".//v:Shape", ns)}
    page_group = root.find(f"{{{namespace}}}g")
    before_texts = ["".join(t.itertext()) for t in root.iter(f"{{{namespace}}}text")]
    for index, group in enumerate(list(page_group)):
        sid = group.get(f"{{{visio}}}mID")
        if sid not in positions:
            continue
        dx = 0.0
        dy = NEW_H-old_h-(TOP_SHIFT if positions[sid] > 135 else LOWER_SHIFT)
        if sid in ("1255", "1258", "1259"):
            delta = {"1255": TOP_TRIM, "1258": 3.12, "1259": 2.22}[sid]
            rect = group.find(f"{{{namespace}}}rect")
            rect.set("y", str(float(rect.get("y"))+delta))
            rect.set("height", str(float(rect.get("height"))-delta))
        if sid in ("1256", "1263", "2474"):
            title = group.find(f"{{{namespace}}}text")
            tx, ty = map(float, re.search(r"translate\(([^,]+),([^\)]+)\)",
                                          group.get("transform")).groups())
            baseline = 150.0 if sid == "1256" else 4.5
            dy = NEW_H-baseline-(ty+float(title.get("y")))
            dx = {"1256": 0, "1263": -1.926, "2474": 15.045}[sid]
        wrapper = ET.Element(f"{{{namespace}}}g", {"transform": f"translate({dx:g},{dy:g})"})
        page_group.remove(group)
        wrapper.append(group)
        page_group.insert(index, wrapper)
    assert before_texts == ["".join(t.itertext()) for t in root.iter(f"{{{namespace}}}text")]
    ET.ElementTree(root).write(FIG / "model-architecture.svg", encoding="utf-8",
                               xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-input", action="store_true")
    args = parser.parse_args()
    ARCHIVE.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        destination = ARCHIVE / f"model-architecture.{suffix}"
        if not destination.exists() or args.refresh_input:
            shutil.copy2(FIG / destination.name, destination)
    counts = adjust_pdf()
    adjust_svg()
    doc = pdfium.PdfDocument(str(FIG / "model-architecture.pdf"))
    img = doc[0].render(scale=300/72).to_pil()
    img.save(FIG / "model-architecture.png", dpi=(300, 300))
    report = {"pdf_points": [491.04, NEW_H], "png_pixels": list(img.size),
              "vector_text_identical": True, "title_baselines_pdf_y": [150.0, 4.5, 4.5],
              "top_panel_bottom_pdf_y": 161.47,
              "lower_panel_bottom_pdf_y": [13.88876, 14.26376], **counts}
    (SRC / "panel-title-layout-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
