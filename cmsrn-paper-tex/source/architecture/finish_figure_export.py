"""Remove Visio's page-outline stroke and render the final vector PDF at 300 dpi.

Requires pypdf and pypdfium2. Does not crop or rasterize the PDF.
"""
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream
import pypdfium2 as pdfium

root = Path(__file__).resolve().parents[2]
pdf_path = root / "figures/model-architecture.pdf"
reader = PdfReader(pdf_path)
page = reader.pages[0]
width, height = float(page.mediabox.width), float(page.mediabox.height)
content = page.get_contents()
ops = content.operations
removed = 0
for i, (args, op) in enumerate(ops[:-1]):
    # Native Visio export adds one page-sized rectangle immediately stroked.
    # Clipping rectangles, colored panel frames, and all other paths are retained.
    if (op == b"re" and ops[i + 1][1] == b"S"
            and abs(float(args[0])) < 1 and abs(float(args[1])) < 1
            and abs(float(args[2]) - width) < 1
            and abs(float(args[3]) - height) < 1):
        ops[i + 1] = ([], b"n")
        removed += 1
assert removed <= 1, "Unexpected multiple page outlines; inspect export"
content.operations = ops
writer = PdfWriter()
writer.add_page(page)
writer.pages[0].replace_contents(content)
temporary = pdf_path.with_name("model-architecture-clean.pdf")
with temporary.open("wb") as stream:
    writer.write(stream)
reader.close()
temporary.replace(pdf_path)
doc = pdfium.PdfDocument(str(pdf_path))
image = doc[0].render(scale=300 / 72).to_pil()
image.save(pdf_path.with_suffix(".png"), dpi=(300, 300))
print(f"Removed {removed} page-outline stroke(s); exported vector PDF and 300 dpi PNG.")
