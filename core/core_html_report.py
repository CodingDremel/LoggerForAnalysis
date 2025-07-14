import os
import json
from pathlib import Path

def generate_html_report(
    log_folder: str,
    report_path: str,
    log_capture: list[str],
    output_html: str = "operation_summary.html"
):
    """
    Generate a single HTML file with:
    - Hierarchical links to all files in the log_folder
    - Results from operation_report.json
    - All captured log/print output
    """

    def file_tree_html(root_path: Path, rel_path=""):
        html = "<ul>"
        for entry in sorted(root_path.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
            full_rel = os.path.join(rel_path, entry.name)
            if entry.is_dir():
                html += f'<li><b>{entry.name}/</b>{file_tree_html(entry, full_rel)}</li>'
            else:
                html += f'<li><a href="{full_rel}" target="_blank">{entry.name}</a></li>'
        html += "</ul>"
        return html

    # 1. File tree
    log_folder_path = Path(log_folder)
    file_tree = file_tree_html(log_folder_path)

    # 2. Operation report
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        report_html = f"<pre>{json.dumps(report, indent=2)}</pre>"
    except Exception as e:
        report_html = f"<pre>Error loading report: {e}</pre>"

    # 3. Captured logs
    logs_html = "<pre>" + "\n".join(log_capture) + "</pre>"

    # Compose HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Operation Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        ul {{ list-style-type: none; }}
        pre {{ background: #f8f8f8; padding: 1em; border: 1px solid #ddd; }}
        a {{ text-decoration: none; color: #0366d6; }}
    </style>
</head>
<body>
    <h1>Hierarchical Log Folder Structure</h1>
    {file_tree}
    <h1>Operation Report</h1>
    {report_html}
    <h1>Log Output</h1>
    {logs_html}
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    return output_html