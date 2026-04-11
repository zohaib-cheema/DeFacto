"""An inspectable HTML report with no remote scripts or external assets."""

from __future__ import annotations

import html
import json
from pathlib import Path


def render_report(run: dict, destination: str | Path) -> Path:
    def escape(value):
        return html.escape(str(value), quote=True)

    metrics = "".join(
        f"<tr><th scope='row'>{escape(name.replace('_', ' '))}</th><td>{value:.4f}</td></tr>"
        for name, value in run["metrics"].items()
    )
    classes = run["classes"]
    headings = "".join(f"<th scope='col'>{escape(label)}</th>" for label in classes)
    confusion = "".join(
        f"<tr><th scope='row'>{escape(label)}</th>"
        + "".join(f"<td>{value}</td>" for value in row)
        + "</tr>"
        for label, row in zip(classes, run["confusion"], strict=True)
    )
    quality = "".join(f"<li>{escape(message)}</li>" for message in run["train"]["limitations"])
    low, high = run["accuracy_interval"]
    content = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DeFacto experiment {escape(run["run_id"])}</title>
<style>
:root{{font-family:system-ui,sans-serif;color:#182b25;background:#fff}}
body{{max-width:960px;margin:48px auto;padding:0 24px;line-height:1.6}}
h1{{font-size:40px;letter-spacing:-1px}}h2{{margin-top:40px}}p{{max-width:75ch}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;border-bottom:1px solid #dce5de;padding:10px 12px}}
thead,caption{{background:#eff5f0}}caption{{text-align:left;padding:12px;font-weight:600}}
code{{overflow-wrap:anywhere}}.notice{{border-left:3px solid #21624c;padding:4px 20px}}
details{{margin:24px 0}}pre{{overflow:auto;background:#eff5f0;padding:18px}}
@media(max-width:600px){{body{{margin:24px auto}}h1{{font-size:30px}}th,td{{padding:8px}}}}
</style>
<header><p>DeFacto</p><h1>Classification experiment</h1>
<p>Model: <strong>{escape(run["model"])}</strong> · Mode: {escape(run["mode"])}<br>
Run <code>{escape(run["run_id"])}</code> · {escape(run["created_at"])}</p></header>
<main><div class="notice"><p>This is a research evaluation against dataset labels.
Predictions do not establish whether a real world claim is true.
The prepared numeric tables cannot classify an arbitrary article.</p></div>
<h2>Evaluation</h2><p>{run["train"]["rows"]} training rows and
{run["test"]["rows"]} external evaluation rows. Preprocessing is fitted within
each training fold. The external split is used only for final evaluation.</p>
<table><caption>Measured scores</caption><tbody>{metrics}</tbody></table>
<p>Accuracy bootstrap interval: {low:.4f} to {high:.4f}. This percentile interval
uses 500 resamples and describes sampling uncertainty for this fixed model.
It does not account for training variability or dataset shift.</p>
<h2>Confusion matrix</h2><p>Rows are actual labels. Columns are predicted labels.</p>
<table><thead><tr><th scope="col">Actual / Predicted</th>{headings}</tr></thead>
<tbody>{confusion}</tbody></table>
<h2>Data checks</h2><ul>{quality}<li>Matching feature rows across splits:
{run["overlap"]["matching_feature_rows"]}. Numeric matches may represent different statements.</li>
<li>Excluded columns: {escape(", ".join(run["train"]["excluded_columns"]) or "None")}.</li>
<li>Dataset origin and upstream preprocessing must be independently verified.</li></ul>
<details><summary>Reproduction metadata</summary><pre>{escape(json.dumps(run, indent=2))}</pre></details>
</main><footer><p>Generated from a recorded experiment. Scores are measured, not promises.</p></footer>
</html>"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination
