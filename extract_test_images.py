"""
extract_test_images.py
----------------------
Extracts pages from PDFs in the pdfs/ folder and saves them as high-res PNGs
into the test_images/ folder for pipeline testing.

Uses PyMuPDF (fitz) — no poppler required.

Usage:
    python extract_test_images.py
    python extract_test_images.py --pdf pdfs/blueprint_2.pdf --pages 0 1
    python extract_test_images.py --all   # one page from each PDF
"""

import os
import argparse
import fitz  # PyMuPDF


DPI = 300
ZOOM = DPI / 72  # PDF native is 72 DPI


def extract_pages(pdf_path: str, page_indices: list, out_dir: str):
    """Renders specified pages from a PDF as 300-DPI PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    mat = fitz.Matrix(ZOOM, ZOOM)
    saved = []

    for page_idx in page_indices:
        if page_idx >= len(doc):
            print(f"  [SKIP] {pdf_path} has only {len(doc)} pages, skipping page index {page_idx}")
            continue
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        out_path = os.path.join(out_dir, f"{base}_page{page_idx}.png")
        pix.save(out_path)
        print(f"  Saved: {out_path}  ({pix.width}x{pix.height} px)")
        saved.append(out_path)

    doc.close()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract PDF pages to PNG for pipeline testing")
    parser.add_argument("--pdf", default=None, help="Path to a single PDF (default: pdfs/blueprint_0.pdf)")
    parser.add_argument("--pages", nargs="+", type=int, default=[0, 1],
                        help="Page indices to extract (0-based, default: 0 1)")
    parser.add_argument("--all", action="store_true",
                        help="Extract page 0 from every PDF in pdfs/")
    parser.add_argument("--out", default="test_images", help="Output folder (default: test_images)")
    args = parser.parse_args()

    pdf_dir = "pdfs"

    if args.all:
        pdfs = sorted([
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.lower().endswith(".pdf")
        ])
        print(f"Found {len(pdfs)} PDFs — extracting page 0 from each...")
        for pdf_path in pdfs:
            extract_pages(pdf_path, page_indices=[0], out_dir=args.out)
    else:
        pdf_path = args.pdf if args.pdf else os.path.join(pdf_dir, "blueprint_0.pdf")
        print(f"Extracting pages {args.pages} from: {pdf_path}")
        extract_pages(pdf_path, page_indices=args.pages, out_dir=args.out)

    print(f"\nDone. Images saved to: {args.out}/")


if __name__ == "__main__":
    main()
