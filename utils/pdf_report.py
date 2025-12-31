# utils/pdf_report.py
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

STATIC_OUT = os.path.join("static", "output_videos")
os.makedirs(STATIC_OUT, exist_ok=True)

def generate_pdf_report(analysis_data, base_name):
    """
    Create a simple PDF summarizing key charts and top players.
    analysis_data: dict produced by process_pipeline (analysis JSON)
    base_name: base filename to use
    Returns path to pdf
    """
    pdf_path = os.path.join(STATIC_OUT, f"report_{base_name}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    w, h = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, h - 60, "FootyVision — Match Report")

    # Basic stats
    fps = analysis_data.get("fps", "")
    frames = analysis_data.get("total_frames", "")
    c.setFont("Helvetica", 11)
    c.drawString(40, h - 100, f"FPS: {fps}")
    c.drawString(160, h - 100, f"Frames: {frames}")

    # Insert images (speed/distance/possession/radar) if exist
    y = h - 140
    imgs = analysis_data.get("images", {})
    positions = [("speed", 40, y), ("distance", 300, y)]
    for key, x, y0 in positions:
        p = imgs.get(key)
        if p and os.path.exists(p):
            try:
                img = ImageReader(p)
                c.drawImage(img, x, y0 - 140, width=240, height=120, preserveAspectRatio=True)
            except Exception:
                pass

    # next row
    y = y - 160
    radar = imgs.get("radar")
    poss = imgs.get("possession")
    if radar and os.path.exists(radar):
        try:
            c.drawImage(ImageReader(radar), 40, y - 140, width=240, height=140, preserveAspectRatio=True)
        except Exception:
            pass
    if poss and os.path.exists(poss):
        try:
            c.drawImage(ImageReader(poss), 300, y - 140, width=240, height=140, preserveAspectRatio=True)
        except Exception:
            pass

    # Player top performance summary
    y2 = y - 180
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y2, "Top Players Performance")
    c.setFont("Helvetica", 10)
    player_stats = analysis_data.get("player_stats", {})
    top_lines = []
    # make a short list of top distance or speed if available
    for pid, pdata in sorted(player_stats.items(), key=lambda x: -x[1].get("distance", 0))[:5]:
        top_lines.append(f"P{pid}: distance {pdata.get('distance',0):.2f}, max speed {pdata.get('max_speed',0):.2f}")
    for i, line in enumerate(top_lines):
        c.drawString(40, y2 - 18*(i+1), line)

    c.showPage()
    c.save()
    return pdf_path

# optional helper used in pipeline for generating a small prepped object (not required)
def make_report_data(analysis):
    return analysis
