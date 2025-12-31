# app.py
import os
import json
from flask import Flask, render_template, request, url_for, send_file, send_from_directory, redirect
from process_pipeline import process_video
from utils.pdf_report import generate_pdf_report

app = Flask(__name__, static_folder="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "input_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Handle favicon.ico requests (browsers often request this at root)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Landing page
@app.route("/")
def home():
    return render_template("home.html")

# Upload page (GET shows form, POST handles file)
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload_page.html")
    # POST: handle file upload
    if "video" not in request.files:
        return "No file uploaded", 400
    file = request.files["video"]
    if file.filename == "":
        return "Empty filename", 400

    # Save file
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    # Process video (this is blocking — may take time)
    analysis = process_video(save_path)
    if not analysis:
        return "Processing failed", 500

    # Build static URLs for template
    video_url = url_for("static", filename=f"output_videos/{analysis['processed_filename']}")
    json_rel = analysis.get("analysis_json", "")
    json_url = None
    if json_rel:
        json_url = url_for("static", filename=f"output_videos/{json_rel}")

    return render_template("result.html", video_url=video_url, analysis=analysis, analysis_json_url=json_url)

# Download generated PDF report by base name
@app.route("/download_report/<base>")
def download_report(base):
    json_path = os.path.join(app.static_folder, "output_videos", f"analysis_{base}.json")
    if not os.path.exists(json_path):
        return "Report not found", 404

    with open(json_path, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

    pdf_path = generate_pdf_report(analysis_data, base)
    if not os.path.exists(pdf_path):
        return "Failed to create PDF", 500

    return send_file(pdf_path, as_attachment=True)

# Convenience route to serve any output video
@app.route("/output_videos/<path:filename>")
def output_videos(filename):
    return send_from_directory(os.path.join(app.static_folder, "output_videos"), filename)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

