# Internal 
from __future__ import annotations
import base64, io, os, shutil, tempfile, traceback, types
import cv2
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd
from PIL import Image
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
import zipfile, io as _io


try:
    from traitly.fruit_phenotyping import FruitInternalAnalyzer, FruitExternalAnalyzer
    from traitly import __version__
except ImportError:
    __version__ = "dev"

    class _Stub:
        def __init__(self, p):
            self.image_path = p
    class FruitInternalAnalyzer(_Stub): pass
    class FruitExternalAnalyzer(_Stub): pass


def arr_to_b64(arr):
    if arr is None: return ""
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def img_tag(arr, style="width:100%;border-radius:8px;margin-top:.5rem"):
    b = arr_to_b64(arr)
    return f'<img src="data:image/png;base64,{b}" style="{style}">' if b else ""

def df_csv(df): return df.to_csv(index=False).encode()


_CSS = """
/* light theme (default) colors */
:root {
  --header-bg: rgba(0,0,0,1);
  --header-text: rgba(185,185,186,.75);
  --header-text-hover: #fff;
  --header-active-bg: rgba(255,255,255,.3);
  --header-hover-bg: rgba(96,158,160,.5);
  --brand-color: white;
  --ver-color: #94a3b8;
  --ver-bg: rgba(0,255,255,.20);
  --theme-btn-color: rgba(255,255,255,.7);
  --theme-btn-hover: rgba(255,255,255,.08);
  --gh-btn-bg: rgba(255,255,255,.15);
  --gh-btn-color: rgba(255,255,255,.85);
  --gh-btn-border: rgba(255,255,255,.22);
  --gh-btn-hover-bg: rgba(255,255,255,.13);
  --gh-btn-hover-color: #fff;
  --gh-stat-color: rgba(255,255,255,.5);
  --sidebar-bg: #fff;
  --sidebar-border: #e2e8f0;
  --body-bg: #f0f2f5;
  --panel-title-color: #1e293b;
  --step-hover-bg: #f1f5f9;
  --step-hover-color: #1e293b;
  --step-active-bg: #eff6ff;
  --step-active-color: #1d4ed8;
  --step-color: #475569;
  --step-num-bg: #e2e8f0;
  --step-num-color: #94a3b8;
  --reset-bg: transparent;
  --reset-border: #e2e8f0;
  --reset-color: #64748b;
  --reset-hover-bg: #fee2e2;
  --reset-hover-border: #fca5a5;
  --reset-hover-color: #dc2626;
  --home-text: #1a1a2e;
  --home-h2-color: #1a1a2e;
  --home-h3-color: #426fb1;
  --home-h4-color: #1a1a2e;
  --home-info-bg: rgba(212,141,158,0.3);
  --home-info-border: #91597a;
  --home-link-color: #2d63bc;
  --home-muted: #64748b;
  --body-text: #1a1a2e;
}

/* dark theme colors */
body.dark-theme {
  --header-bg: rgba(255,255,255,1);
  --header-text: rgba(50,50,50,.75);
  --header-text-hover: #000;
  --header-active-bg: rgba(0,0,0,.15);
  --header-hover-bg: rgba(96,158,160,.4);
  --brand-color: rgba(30,30,30,0.8);
  --ver-color: #475569;
  --ver-bg: rgba(0,100,100,.15);
  --theme-btn-color: rgba(0,0,0,.7);
  --theme-btn-hover: rgba(0,0,0,.08);
  --gh-btn-bg: rgba(0,0,0,.07);
  --gh-btn-color: rgba(0,0,0,.85);
  --gh-btn-border: rgba(0,0,0,.22);
  --gh-btn-hover-bg: rgba(0,0,0,.13);
  --gh-btn-hover-color: #000;
  --gh-stat-color: rgba(0,0,0,.5);
  --sidebar-bg: #1e1e2e;
  --sidebar-border: #3d3d5c;
  --body-bg: #2d2d3f;
  --panel-title-color: #f1f5f9;
  --step-hover-bg: #2d3748;
  --step-hover-color: #171f2d;
  --step-active-bg: #1e3a5f;
  --step-active-color: #90cdf4;
  --step-color: #94a3b8;
  --step-num-bg: #3d3d5c;
  --step-num-color: #cbd5e1;
  --reset-bg: transparent;
  --reset-border: #3d3d5c;
  --reset-color: #94a3b8;
  --reset-hover-bg: #3d1515;
  --reset-hover-border: #ef4444;
  --reset-hover-color: #fca5a5;
  --home-text: #bfc4ca;
  --home-h2-color: #bfc4ca;
  --home-h3-color: #5491c5;
  --home-h4-color: white;
  --home-info-bg: rgba(134,98,120,0.75);
  --home-info-border: #c084a0;
  --home-link-color: #90cdf4;
  --home-muted: #94a3b8;
  --body-text: #e2e8f0;
}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
body { font-family: 'Inter','Segoe UI',sans-serif; background: var(--body-bg); color:#1a1a2e; margin:0; transition: background .2s; }

/* hide shiny default nav */
.navbar, nav.navbar, .bslib-page-title { display:none !important; }

/* header bar */
.traitly-header {
  position:fixed; top:0; left:0; right:0; height:150px;
  background: var(--header-bg);
  display:flex; align-items:center; padding:0 1.5rem;
  z-index:1050; box-shadow:0 1px 4px rgba(0,0,0,.3);
  gap:1rem; transition: background .2s;
}

/* traitly main title */
.t-brand {
  display:flex; align-items:center; gap:1.5rem; color: var(--brand-color);
  font-size:4.0rem; font-weight:700; letter-spacing:2px;
  white-space:nowrap; text-decoration:none;
  margin-left: 2.5rem;
  margin-right: 3rem;
}

/* version badge next to traitly */
.t-brand .ver {
  font-size:1.32rem; font-weight:400;
  background: var(--ver-bg); padding:.4rem .35rem; border-radius:20px; color: var(--ver-color);
}

/* nav buttons in top */
.t-nav { display:flex; gap:.9rem; align-items:center; }
.t-nav a,
.t-nav a:link,
.t-nav a:visited {
  color: var(--header-text) !important; text-decoration:none; font-size:1.84rem;
  font-weight:500; padding:.8rem .7rem; border-radius:4px;
  transition:all .15s; cursor:pointer; border:none; background:none; white-space:nowrap;
}
.t-nav a:hover { color: var(--header-text-hover) !important; background: var(--header-hover-bg); border-radius:20px; }
.t-nav a.active { color: var(--header-text-hover) !important; background: var(--header-active-bg); border-radius:20px; }

/* right buttons */
.t-right { margin-left:auto; display:flex; align-items:center; gap:2.5rem; }

/* dark/light toggle */
.theme-btn {
  background:none; border:none; color: var(--theme-btn-color);
  cursor:pointer; padding:.3rem; border-radius:4px;
  display:flex; align-items:center; transition:all .15s; font-size:1rem;
}
.theme-btn:hover { color: var(--header-text-hover); background: var(--theme-btn-hover); }

/* github button */
.gh-btn {
  display:flex; align-items:center; gap:.4rem;
  background: var(--gh-btn-bg); color: var(--gh-btn-color);
  border:1px solid var(--gh-btn-border); border-radius:30px;
  padding:.28rem .7rem; font-size:1.58rem; font-weight:300;
  text-decoration:none; transition:all .15s;
}
.gh-btn:hover { background: var(--gh-btn-hover-bg); color: var(--gh-btn-hover-color); }
.gh-stat {
  display:flex; align-items:center; gap:.25rem;
  font-size:1.43rem; color: var(--gh-stat-color);
}
.gh-stat svg { width:12px; height:12px; }

/* push content below header */
.bslib-page-main, main {
  margin-top: 155px !important;
  margin-left: 90px !important;
  padding-top: .5rem;
}

/* sidebar */
.bslib-sidebar-layout > .sidebar {
  background: var(--sidebar-bg) !important;
  border-right:1px solid var(--sidebar-border) !important;
  padding-top:.7rem; top:155px !important;
}

.sb-label {
  font-size:1.9rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.09em; color:#94a3b8; margin: 5.9rem 6.9 5.3rem 0.8rem;
  gap: 5rem;
}
.sb-mode-badge {
  display:inline-flex; align-items:center; gap:.35rem;
  padding:.3rem .7rem; border-radius:20px; font-size:1.8rem; font-weight:600;
  margin-bottom:5rem;
}
.sb-mode-badge.internal { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }
.sb-mode-badge.external { background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; }

/* step nav links */
.step-link {
  display:flex; align-items:center; gap:1.5rem;
  padding:.4rem .6rem; margin:.08rem 0;
  border-radius:8px; cursor:pointer;
  font-size:1.82rem; font-weight:500; color:#475569;
  transition:all .14s; border:none; background:none;
  width:100%; text-align:left;
}
.step-link { color: var(--step-color); }
.step-link:first-of-type { margin-top: 1.5rem; }
.step-link:hover { background: var(--step-hover-bg); color: var(--step-hover-color); }
.step-link.active { background: var(--step-active-bg); color: var(--step-active-color); font-weight:600; }
.step-link.done .step-num { background:#059669; color:#fff; }
.step-link.active .step-num { background:#3b82f6; color:#fff; }

.step-num {
  width: 38px; height: 38px; border-radius: 50%;
  background: var(--step-num-bg); color: var(--step-num-color);
  font-size: .8rem; font-weight: 800;
  display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.step-check { color:#059669; font-size:.72rem; margin-left:auto; }

/* reset button in sidebar */
.btn-reset {
  width:100%; border-radius:8px; padding:.4rem; font-size:1.8rem;
  transition:all .14s; cursor:pointer; margin-top:.6rem;
  background: var(--step-hover-bg);
  border: 1px solid var(--sidebar-border);
  color: var(--step-color);
}
.btn-reset:hover {
  background: var(--step-active-bg);
  border-color: var(--step-active-color);
  color: var(--step-active-color);
}

/* home page style */
.home-content {
  padding:1.5rem; font-size:.95rem; line-height:1.7;
  color: var(--home-text);
}
.home-title {
  font-size:4rem; font-weight:700; margin-bottom:4rem;
  color: var(--home-text);
}
.home-title-sub {
  font-weight:300; color: var(--home-muted);
}
.home-h2 {
  font-size:3rem; font-weight:600; margin-bottom:1rem;
  color: var(--home-h2-color);
}
.home-h3 {
  font-size:2.5rem; font-weight:600; margin:.8rem 0 .3rem;
  color: var(--home-h3-color);
}

.home-content h4 {
  color: var(--home-h4-color) !important;
  font-weight: normal;
}

.home-body {
  font-size:2rem; line-height:1.5;
}
.home-info-box {
  background: var(--home-info-bg);
  border-left:4px solid var(--home-info-border);
  padding:.75rem 1rem; border-radius:10px;
}
.home-link {
  color: var(--home-link-color);
}

/* hide sidebar in home */
body.on-home .bslib-sidebar-layout > .sidebar {
  display: none !important;
}

body.on-home .bslib-sidebar-layout {
  grid-template-columns: 0 1fr !important;
  padding-left: 0 !important;
}

body.on-home .bslib-sidebar-layout > .main,
body.on-home .bslib-page-main,
body.on-home main {
  margin-left: 0 !important;
  padding-left: 1.5rem !important;
  grid-column: 1 / -1 !important;
  width: 100% !important;
  max-width: 100% !important;
}

/* cards */
.home-section {
  background:#fff; border-radius:12px; border:1px solid #e2e8f0;
  padding:1.25rem 1.4rem; margin-bottom:.9rem;
  box-shadow:0 1px 3px rgba(0,0,0,.06);
}
.feature-box {
  padding:.95rem 1.1rem; background:#f8fafc;
  border-left:4px solid #3b82f6; border-radius:0 8px 8px 0; margin:.65rem 0;
}
.pipeline-item {
  display:flex; align-items:center; gap:.42rem;
  padding:.32rem .6rem; margin:.15rem 0;
  background:#f1f5f9; border-left:3px solid #3b82f6;
  border-radius:0 5px 5px 0; font-size:.83rem;
}

/* hide the default shiny nav-tabs */
.nav-tabs { display:none !important; }

/* action buttons full-width */
button.action-button, a.action-button {
  width:100% !important; display:block !important;
  text-align:center !important; border-radius:8px !important;
  font-weight:500 !important; padding:.46rem 1rem !important; transition:all .14s !important;
}
.btn-primary {
  background:linear-gradient(135deg,#3b82f6,#1d4ed8) !important;
  border:none !important; box-shadow:0 2px 6px rgba(59,130,246,.3) !important;
}
.btn-primary:hover {
  background:linear-gradient(135deg,#2563eb,#1e3a8a) !important;
  transform:translateY(-1px);
}

/* inputs */
.form-control, .form-select {
  border-radius:7px !important; border:1px solid #e2e8f0 !important; font-size:.86rem !important;
}
.form-control:focus, .form-select:focus {
  border-color:#3b82f6 !important; box-shadow:0 0 0 3px rgba(59,130,246,.15) !important;
}
.folder-picked {
  font-size:.81rem; color:#475569; padding:.36rem .62rem;
  background:#f1f5f9; border-radius:6px; border:1px solid #e2e8f0;
  margin:.22rem 0 .42rem; word-break:break-all;
}
.panel-title {
  font-size: 4rem;
  font-weight: 600;
  color: var(--panel-title-color);
  margin-bottom: 3rem;
}
.shiny-input-container {
  width: 100% !important;
}
.shiny-input-container .input-group {
  width: 100% !important;
}
/* tabs content label sizes */
.tab-content label,
.tab-content .form-label,
.tab-content .form-check-label {
  font-size: 1.8rem !important;
}
.tab-content .form-control,
.tab-content .form-select {
  font-size: 1.6rem !important;
}

/* controlling img output step 1 */
#step1_preview {
  display: flex !important;
  justify-content: center;
  align-items: flex-start;
  padding-left: 2rem;
}

#step1_preview img {
  max-height: 1500px;
  max-width: 1500px;
  object-fit: contain;
}

/* step 1 result boxes */
.bslib-value-box {
  border-radius: 10px !important;
  border: 1px solid #e2e8f0 !important;
  min-height: 70px !important; 
  background: rgba(98,123,140,0.8) ! important;  
}

.bslib-value-box .value-box-value {
  font-size: 3.0rem !important;  
}

.bslib-value-box .value-box-title {
  font-size: 2.1rem !important;
  color: #fff !important;
}

#step2_results img {
  max-height: 1500px;
  max-width: 100%;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}

#step3_results img {
  max-height: 1500px;
  max-width: 100%;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}

#step4_results img {
  max-height: 1500px;
  max-width: 100%;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}

/* step 5 */
#detect_results img {
  max-height: 1500px;
  max-width: 100%;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}

/* step 6 - morphology */
#morph_results img {
  max-height: 1500px;
  max-width: 100%;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}

/* step 7 - color */
#color_results img {
  max-height: 1500px;
  max-width: 100%;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}

/* image zoom */
.img-zoomable { cursor: zoom-in; transition: opacity .15s; }
.img-zoomable:hover { opacity: .9; }

#lightbox-overlay {
  display: none; position: fixed; inset: 0; z-index: 9999;
  background: rgba(0,0,0,.85); align-items: center; justify-content: center;
  cursor: zoom-out;
}
#lightbox-overlay.active { display: flex; }
#lightbox-overlay img {
  max-width: 92vw; max-height: 92vh;
  border-radius: 8px; box-shadow: 0 8px 40px rgba(0,0,0,.6);
  object-fit: contain;
}
#lightbox-close {
  position: absolute; top: 1rem; right: 1.5rem;
  color: #fff; font-size: 2.5rem; cursor: pointer;
  background: none; border: none; line-height: 1;
}
#lightbox-overlay img {
  max-width: 92vw; max-height: 92vh;
  border-radius: 8px; box-shadow: 0 8px 40px rgba(0,0,0,.6);
  object-fit: contain;
  transition: transform .1s ease;
  cursor: grab;
  user-select: none;
}
#lightbox-overlay img.dragging { cursor: grabbing; }
#lightbox-zoom-btns {
  position: absolute; bottom: 1.5rem; left: 50%; transform: translateX(-50%);
  display: flex; gap: .5rem; align-items: center;
}
#lightbox-zoom-btns button {
  background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.3);
  color: #fff; border-radius: 8px; padding: .4rem .9rem;
  font-size: 1.4rem; cursor: pointer; transition: background .15s;
}
#lightbox-zoom-btns button:hover { background: rgba(255,255,255,.28); }
#lightbox-zoom-level {
  color: rgba(255,255,255,.7); font-size: 1.2rem; min-width: 3.5rem; text-align: center;
}

/* info messages */
.tooltip-wrap {
  position: relative; display: inline-flex; align-items: center;
}
.tooltip-wrap .tooltip-box {
  visibility: hidden; opacity: 0;
  background: #1e293b; color: #fff;
  font-size: 1.3rem; font-weight: 400;
  border-radius: 7px; padding: .5rem .8rem;
  position: absolute; left: 2rem; top: 50%; transform: translateY(-50%);
  white-space: normal; width: 220px; z-index: 999;
  box-shadow: 0 4px 12px rgba(0,0,0,.25);
  transition: opacity .15s;
  pointer-events: none;
}
.tooltip-wrap:hover .tooltip-box { visibility: visible; opacity: 1; }

/* steps text */
body.dark-theme .tab-content,
body.dark-theme .tab-content * {
  color: var(--body-text) !important;
}

body.dark-theme .tab-content .text-success { color: #4ade80 !important; }
body.dark-theme .tab-content .text-danger  { color: #f87171 !important; }
body.dark-theme .tab-content .text-info    { color: #38bdf8 !important; }
body.dark-theme .tab-content .text-muted   { color: #94a3b8 !important; }
body.dark-theme .tab-content .panel-title  { color: var(--panel-title-color) !important; }
body.dark-theme .tab-content .tooltip-box  { color: #fff !important; }
body.dark-theme .tab-content .bslib-value-box .value-box-value,
body.dark-theme .tab-content .bslib-value-box .value-box-title { color: #fff !important; }

/* inputs background */
body.dark-theme .tab-content .form-control,
body.dark-theme .tab-content .form-select {
  background-color: #2d3748 !important;
  border-color: #4a5568 !important;
}
body.dark-theme .tab-content .form-control[type="file"],
body.dark-theme .tab-content input[type="file"] {
  background-color: #2d3748 !important;
  border-color: #4a5568 !important;
}
body.dark-theme .tab-content .input-group-text {
  background-color: #3d3d5c !important;
  border-color: #4a5568 !important;
}
body.dark-theme .tab-content input[type="file"]::file-selector-button {
  background-color: #4a5568 !important;
  color: var(--body-text) !important;
  border-color: #64748b !important;
}
body.dark-theme .tab-content hr { border-color: #3d3d5c !important; }
body.dark-theme .tab-content details summary { color: #94a3b8 !important; }

/* hr lines in dark */
body.dark-theme hr {
  border-color: rgba(105,141,151, 0.8) !important;
  opacity: 1 !important;
}
body.dark-theme .tab-content hr {
  border-color: rgba(105,141,151, 0.8) !important;
  opacity: 1 !important;
}

/* ? bubble in dark */
body.dark-theme .tooltip-wrap > span {
  background: #3d3d5c !important;
  color: #cbd5e1 !important;
}

.bslib-value-box { border-radius:10px !important; border:1px solid #e2e8f0 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#f1f5f9; }
::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:3px; }
"""


_HEADER = f"""
<div id="lightbox-overlay">
  <button id="lightbox-close" onclick="closeLightbox()">✕</button>
  <img id="lightbox-img" src="" alt="zoom">
  <div id="lightbox-zoom-btns">
    <button onclick="lbZoom(0.25)">＋</button>
    <span id="lightbox-zoom-level">100%</span>
    <button onclick="lbZoom(-0.25)">－</button>
    <button onclick="closeLightbox(); _lbScale=1; _lbDragX=0; _lbDragY=0; _lbApply()">↺</button>
  </div>
</div>

<div class="traitly-header">
  <a class="t-brand">
    Traitly <span class="ver">v{__version__}</span>
  </a>

  <nav class="t-nav" id="t-nav">
    <a id="hn-0" onclick="goMainTab('tab_home',0)"   class="active">Home</a>
    <a id="hn-1" onclick="goMainTab('tab_analysis',1,'internal')">Internal</a>
    <a id="hn-2" onclick="goMainTab('tab_analysis',2,'external')">External</a>
    <a id="hn-3" onclick="goMainTab('tab_bg',3)">Background Helper</a>
    <a id="hn-4" onclick="goMainTab('tab_batch',4)">Batch Analysis</a>
  </nav>

  <div class="t-right">
    <button class="theme-btn" onclick="toggleTheme()" title="Toggle theme">
        <i id="theme-icon" class="fa-solid fa-moon" style="font-size:3rem"></i>
    </button>

    <a class="gh-btn" href="https://github.com/mariameraz/traitly" target="_blank">
      <i class="fa-brands fa-github-alt" style="font-size:2.4rem"></i>
      mariameraz/traitly
      <span class="gh-stat">
        <svg viewBox="0 0 16 16" fill="rgba(255,255,255,.5)">
          <path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0
                   1 .416 1.279l-3.046 2.97.719 4.192a.751.751 0 0 1-1.088.791L8
                   12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75
                   0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Z"/>
        </svg>
        0
      </span>
    </a>
  </div>
</div>

<script>
function goStep(stepValue) {{
  Shiny.setInputValue('js_step_click', stepValue, {{priority: 'event'}});
}}

var _dark = false;
function toggleTheme() {{
  _dark = !_dark;
  document.body.classList.toggle('dark-theme', _dark);
  var icon = document.getElementById('theme-icon');
  icon.className = _dark ? 'fa-regular fa-sun' : 'fa-solid fa-moon';
}}

function goMainTab(tabValue, idx, mode) {{
  for (var i = 0; i < 5; i++) {{
    var el = document.getElementById('hn-' + i);
    if (el) el.classList.toggle('active', i === idx);
  }}
  document.body.classList.toggle('on-home', tabValue === 'tab_home');
  Shiny.setInputValue('js_main_tab', tabValue, {{priority: 'event'}});
  if (mode) Shiny.setInputValue('js_mode', mode, {{priority: 'event'}});
}}

// lightbox
var _lbScale = 1, _lbDragX = 0, _lbDragY = 0, _lbStartX = 0, _lbStartY = 0, _lbDragging = false;

function _lbApply() {{
  var img = document.getElementById('lightbox-img');
  img.style.transform = 'translate(' + _lbDragX + 'px,' + _lbDragY + 'px) scale(' + _lbScale + ')';
  document.getElementById('lightbox-zoom-level').textContent = Math.round(_lbScale * 100) + '%';
}}

function lbZoom(delta) {{
  _lbScale = Math.min(8, Math.max(0.2, _lbScale + delta));
  _lbApply();
}}

function closeLightbox() {{
  document.getElementById('lightbox-overlay').classList.remove('active');
  _lbScale = 1; _lbDragX = 0; _lbDragY = 0;
  _lbApply();
}}

document.addEventListener('click', function(e) {{
  if (e.target.closest('#lightbox-zoom-btns')) return;
  var img = e.target.closest('.img-zoomable');
  if (img) {{
    document.getElementById('lightbox-img').src = img.src;
    _lbScale = 1; _lbDragX = 0; _lbDragY = 0;
    document.getElementById('lightbox-overlay').classList.add('active');
    _lbApply();
  }} else if (e.target.id === 'lightbox-overlay' || e.target.id === 'lightbox-close') {{
    closeLightbox();
  }}
}});

document.getElementById('lightbox-overlay') && document.addEventListener('wheel', function(e) {{
  if (!document.getElementById('lightbox-overlay').classList.contains('active')) return;
  e.preventDefault();
  lbZoom(e.deltaY < 0 ? 0.15 : -0.15);
}}, {{ passive: false }});

var _lbImg = null;
document.addEventListener('mousedown', function(e) {{
  if (e.target.id !== 'lightbox-img') return;
  _lbDragging = true; _lbStartX = e.clientX - _lbDragX; _lbStartY = e.clientY - _lbDragY;
  e.target.classList.add('dragging');
}});
document.addEventListener('mousemove', function(e) {{
  if (!_lbDragging) return;
  _lbDragX = e.clientX - _lbStartX; _lbDragY = e.clientY - _lbStartY;
  _lbApply();
}});
document.addEventListener('mouseup', function(e) {{
  _lbDragging = false;
  var img = document.getElementById('lightbox-img');
  if (img) img.classList.remove('dragging');
}});

document.addEventListener('keydown', function(e) {{
  if (!document.getElementById('lightbox-overlay').classList.contains('active')) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === '+' || e.key === '=') lbZoom(0.2);
  if (e.key === '-') lbZoom(-0.2);
}});

document.body.classList.add('on-home');
</script>
"""


# panels for each step
def _panel(val, title, *children):
    return ui.nav_panel(title,
        ui.div(
            ui.HTML(f'<p class="panel-title">{title}</p>'),
            *children,
        ),
        value=val,
    )

step_setup = _panel("step_setup", "Setup Image Measurements",
    ui.layout_columns(
        ui.div(
            ui.output_ui("upload_input_ui"),
            ui.hr(),
            ui.input_checkbox("detect_label", "Detect label text", False),
            ui.input_checkbox("skip_qr", "Skip QR detection", False),
            ui.input_checkbox("detect_color_checker", "Detect color checker", False),
            ui.input_slider("confidence", "Detection confidence", 0.0, 1.0, 0.6, step=0.01),
            ui.hr(),
            ui.input_checkbox("use_dimensions", "Use physical dimensions", False),
            ui.output_ui("dimensions_ui"),
            ui.input_numeric("diameter_cm", "Reference diameter (cm)", 2.5, min=0.0, step=0.01),
            ui.hr(),
            ui.input_checkbox("use_crop", "Crop image", False),
            ui.output_ui("crop_ui"),
            ui.hr(),
            ui.input_action_button("run_step1", "▶  Run Setup", class_="btn btn-primary",
                                   style="font-size: 2rem; padding: .8rem 1.5rem;")
        ),
        ui.div(
            ui.output_image("step1_preview"),
            ui.output_ui("step1_results"),),
            col_widths=[3,9]
    ),
)

step_mask = _panel("step_mask", "Generate Fruit Mask",
    ui.layout_columns(
        ui.div(
            ui.output_ui("mask_bg_ui"),
            ui.input_checkbox("remove_roi", "Remove label/reference regions", True),
            ui.input_checkbox("use_manual_hsv", "Apply manual color threshold", False),
            ui.p("Find HSV range in Background Helper",style="font-size:1.4rem; color:#94a3b8; margin-top:-.5rem;"),
            ui.output_ui("hsv_ui"),
            ui.hr(),
            ui.HTML('''
                    <details style="margin-bottom:.8rem">
                        <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                                        padding:.4rem .2rem;color:#475569;user-select:none">
                            Advanced parameters
                        </summary>
                    <div id="advanced-mask-params" style="padding:.6rem 0 0 .4rem">
            '''),
            ui.hr(),
            ui.input_checkbox("stamp", "Invert image colors (stamp)", False),
            ui.input_checkbox("apply_convex_hull", "Apply convex hull", False),
            ui.input_checkbox("fill_holes", "Fill holes", False),
            ui.hr(),
            ui.input_slider("n_iteration", "Morphology iterations", 1, 5, 1),
            ui.input_slider("roi_expansion", "ROI expansion (px)", -80, 80, 10),
            ui.input_slider("kernel_blur", "Blur kernel", 1, 17, 1, step=2),
            ui.input_slider("kernel_open", "Opening kernel", 1, 17, 1, step=2),
            ui.input_slider("kernel_close", "Closing kernel", 1, 17, 1, step=2),
            ui.input_numeric("erosion_px", "Erosion (px)", 0, min=0, step=1),
            ui.HTML('</div></details>'),
            ui.hr(),
            ui.input_action_button("run_step2", "▶  Generate Mask", class_="btn btn-primary",
                       style="font-size: 2rem; padding: .8rem 1.5rem;"),
        ),
        ui.div(ui.output_ui("step2_results")),
        col_widths=[3, 9],
    ),
)

step_contrast = _panel("step_contrast", "Enhance Locule Contrast <span class=home-title-sub> – Optional</span>",
    ui.layout_columns(
        ui.div(
            ui.input_select("contrast_method", "Contrast method",
                            choices=["none", "gamma", "sigmoid", "exp"]),
            ui.output_ui("contrast_params_ui"),
            ui.hr(),
            ui.input_checkbox("compare_method", "Compare all methods", False),
            ui.hr(),
            ui.HTML('''
            <details style="margin-bottom:.8rem">
              <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                              padding:.4rem .2rem;color:#475569;user-select:none">
                Advanced parameters
              </summary>
              <div style="padding:.6rem 0 0 .4rem">
            '''),
            ui.input_slider("kernel_blur3", "Blur kernel", 1, 15, 1, step=2),
            ui.input_numeric("clip_limit", "CLAHE clip limit (0 = off)", 0, min=0, step=1),
            ui.input_numeric("tile_grid_size", "CLAHE tile grid size", 12, min=1, step=1),
            ui.HTML('</div></details>'),
            ui.hr(),
            ui.input_action_button("run_step3", "▶  Enhance Contrast", class_="btn btn-primary",
                                   style="font-size: 2rem; padding: .8rem 1.5rem;"),
        ),
        ui.div(ui.output_ui("step3_results")),
        col_widths=[3, 9],
    ),
)

step_locule = _panel("step_locule", "Generate Locule Mask <span class=home-title-sub> – Optional</span>",
    ui.layout_columns(
        ui.div(
            ui.input_checkbox("gen_histogram", "Generate L-channel histogram", False),
            ui.output_ui("histogram_params_ui"),
            ui.hr(),
            ui.input_checkbox("use_thresh", "Set intensity threshold", False),
            ui.output_ui("thresh_ui"),
            ui.hr(),
            ui.input_checkbox("use_otsu", "Set Otsu threshold", False),
            ui.output_ui("otsu_ui"),
            ui.hr(),
            ui.input_numeric("min_fruit_area_lm", "Min fruit area (px)", 5000, min=100, step=100),
            ui.input_numeric("min_locule_area_lm", "Min locule area (px)", 0, min=0, step=10),
            ui.hr(),
            ui.HTML('''
            <details style="margin-bottom:.8rem">
              <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                              padding:.4rem .2rem;color:#475569;user-select:none">
                Advanced parameters
              </summary>
              <div style="padding:.6rem 0 0 .4rem">
            '''),
            ui.input_checkbox("invert_locule", "Invert locule mask", False),
            ui.hr(),
            ui.input_slider("kernel_blur4",  "Blur kernel",    1, 17, 1, step=2),
            ui.input_slider("kernel_open4",  "Opening kernel", 1, 17, 1, step=2),
            ui.input_slider("kernel_close4", "Closing kernel", 1, 17, 1, step=2),
            ui.input_numeric("erosion_px4",  "Erosion (px)",   10, min=0, step=1),
            ui.HTML('</div></details>'),
            ui.hr(),
            ui.input_action_button("run_step4", "▶  Generate Locule Mask", class_="btn btn-primary",
                                   style="font-size: 2rem; padding: .8rem 1.5rem;"),
        ),
        ui.div(ui.output_ui("step4_results")),
        col_widths=[3, 9],
    ),
)

step_detect = _panel("step_detect", "Detect Fruits",
    ui.layout_columns(
        ui.div(
            ui.input_slider("min_fruit_circularity", "Min circularity", 0.0, 1.0, 0.5, step=0.05),
            ui.input_numeric("min_fruit_area_det", "Min fruit area (px)", 500, min=1, step=100),
            ui.input_numeric("max_fruit_area_det", "Max fruit area (px)", 0, min=0, step=100),
            ui.p("Set to 0 for no upper limit.", style="font-size:1.4rem; color:#94a3b8; margin-top:-.5rem;"),
            ui.output_ui("detect_locule_params_ui"),
            ui.hr(),
            ui.HTML('''
            <details style="margin-bottom:.8rem">
              <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                              padding:.4rem .2rem;color:#475569;user-select:none">
                Advanced parameters
              </summary>
              <div style="padding:.6rem 0 0 .4rem">
            '''),
            ui.input_numeric("rescale_factor_det", "Rescale factor", 0, min=0, step=0.1),
            ui.p("Set to 0 for no upper limit.", style="font-size:1.4rem;color:#94a3b8;margin-top:-.5rem;"),
            ui.hr(),
            ui.input_slider("contour_thickness_det", "Fruit contour thickness", 1, 10, 2),
            ui.input_slider("locule_thickness_det",  "Locule contour thickness", 1, 10, 2),
            ui.input_text("contour_color_det", "Fruit contour color (R,G,B)", "0,255,0"),
            ui.input_text("locule_color_det",  "Locule contour color (R,G,B)", "255,0,255"),
            ui.HTML('</div></details>'),
            ui.hr(),
            ui.input_action_button("run_detect", "▶  Detect Fruits", class_="btn btn-primary",
                                   style="font-size: 2rem; padding: .8rem 1.5rem;"),
        ),
        ui.div(ui.output_ui("detect_results")),
        col_widths=[3, 9],
    ),
)

step_morph = _panel("step_morph", "Morphological Analysis",
    ui.layout_columns(
        ui.div(
            ui.input_select("contour_mode", "Contour mode",
                            choices=["raw", "hull", "approx", "ellipse", "circle"]),
            ui.output_ui("epsilon_ui"),
            ui.hr(),
            ui.output_ui("morph_locule_params_ui"),
            ui.hr(),
            ui.input_checkbox("save_params_morph", "Save analysis parameters", False),
            ui.hr(),
            ui.HTML('''
            <details style="margin-bottom:.8rem">
            <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                            padding:.4rem .2rem;color:#475569;user-select:none">
                Advanced parameters
            </summary>
            <div style="padding:.6rem 0 0 .4rem">
            '''),
            ui.input_numeric("alpha_morph", "Alpha (Int. pericarp concave hull)", 0, min=0, step=0.05),
            ui.input_numeric("angle_shifts_morph", "Angle shifts (symmetry)", 500, min=0, step=50),
            ui.input_numeric("num_rays_morph", "Num rays (pericarp thickness)", 90, min=0, step=10),
            ui.HTML('</div></details>'),
            ui.hr(),
            ui.HTML('''
            <details style="margin-bottom:.8rem">
              <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                              padding:.4rem .2rem;color:#475569;user-select:none">
                Plot styling
              </summary>
              <div style="padding:.6rem 0 0 .4rem">
            '''),
            ui.input_slider("font_size_morph", "Font size", 0.5, 4.0, 1.5, step=0.1),
            ui.input_numeric("font_thickness_morph", "Font thickness", 2, min=1, step=1),
            ui.input_select("label_position_morph", "Label position",
                            choices=["top", "bottom", "left", "right"]),
            ui.input_text("font_color_morph",           "Font color (R,G,B)",            "0,0,0"),
            ui.input_text("label_color_morph",          "Label background (R,G,B)",      "255,255,255"),
            ui.input_text("pericarp_ext_color_morph",   "Ext. pericarp color (R,G,B)",   "0,240,0"),
            ui.input_numeric("pericarp_ext_thick_morph","Ext. pericarp thickness", 2, min=1, step=1),
            ui.input_text("pericarp_int_color_morph",   "Int. pericarp color (R,G,B)",   "0,240,240"),
            ui.input_numeric("pericarp_int_thick_morph","Int. pericarp thickness", 2, min=1, step=1),
            ui.input_text("locule_color_morph",         "Locule color (R,G,B)",          "255,0,255"),
            ui.input_numeric("locule_thick_morph",      "Locule thickness", 2, min=1, step=1),
            ui.input_text("centroid_fruit_color_morph", "Fruit centroid color (R,G,B)",  "255,255,51"),
            ui.input_numeric("centroid_fruit_thick_morph", "Fruit centroid size", 2, min=1, step=1),
            ui.input_text("centroid_locule_color_morph","Locule centroid color (R,G,B)", "0,255,255"),
            ui.input_numeric("centroid_locule_thick_morph","Locule centroid size", 2, min=1, step=1),
            ui.HTML('</div></details>'),
            ui.hr(),
            ui.input_action_button("run_morph", "▶  Analyze Morphology", class_="btn btn-primary",
                                   style="font-size: 2rem; padding: .8rem 1.5rem;"),
        ),
        ui.div(ui.output_ui("morph_results")),
        col_widths=[3, 9],
    ),
)

# ── Color step ────────────────────────────────────────────────────────────────
step_color = _panel("step_color", "Color Analysis",
    ui.layout_columns(
        ui.div(
            ui.input_select("stat", "Statistical measure", choices=["mean", "median"]),
            ui.output_ui("color_tissue_ui"),
            ui.input_select("color_space", "Color space",
                            choices=["all", "rgb", "lab", "hsv", "gray"]),
            ui.hr(),
            ui.input_checkbox("get_color_histogram", "Get color histogram", False),
            ui.input_numeric("dark_thresh_color", "Dark pixel threshold", 20, min=0, step=1),
            ui.hr(),
            ui.input_checkbox("save_params_color", "Save analysis parameters", False),
            ui.hr(),
            # ── Plot styling (collapsible) ─────────────────────────────────
            ui.HTML('''
            <details style="margin-bottom:.8rem">
              <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                              padding:.4rem .2rem;color:#475569;user-select:none">
                Plot styling
              </summary>
              <div style="padding:.6rem 0 0 .4rem">
            '''),
            ui.input_slider("font_size_color", "Font size", 0.5, 4.0, 2.0, step=0.1),
            ui.input_numeric("font_thickness_color", "Font thickness", 2, min=1, step=1),
            ui.input_select("label_position_color", "Label position",
                            choices=["top", "bottom", "left", "right"]),
            ui.input_slider("label_opacity_color", "Label opacity", 0.0, 1.0, 0.7, step=0.05),
            ui.input_text("font_color_color",         "Font color (R,G,B)",          "0,0,0"),
            ui.input_text("label_color_color",        "Label background (R,G,B)",    "255,255,255"),
            ui.input_text("pericarp_ext_color_color", "Ext. pericarp color (R,G,B)", "0,255,0"),
            ui.input_numeric("pericarp_ext_thick_color", "Ext. pericarp thickness", 2, min=1, step=1),
            # internal-only annotation params (rendered conditionally in server)
            ui.output_ui("color_int_styling_ui"),
            ui.HTML('</div></details>'),
            ui.hr(),
            ui.input_action_button("run_color", "▶  Analyze Color", class_="btn btn-primary",
                                   style="font-size: 2rem; padding: .8rem 1.5rem;"),
        ),
        ui.div(ui.output_ui("color_results")),
        col_widths=[3, 9],
    ),
)


# home 
tab_home = ui.nav_panel("Home",
    ui.div(
        ui.HTML('''
        <div class="home-content">
            <h1 class="home-title">
                Welcome to Traitly <span class="home-title-sub">Interactive Analyzer</span>
            </h1>
            <h2>
                <p style="margin-bottom:1.2rem">
                <strong>Traitly</strong> is a Python library designed to <strong>automate fruit image analysis</strong>, from a single sample to hundreds of fruits in one run. Using standard RGB images, it extracts 
                <strong>color, shape, and size traits</strong> from both internal (cross-sections) and external 
                (surface) fruit images, with no manual measurements required.
                </p>
                <p style="margin-bottom:1.2rem">
                Traitly is committed to <strong>open and reproducible science</strong>: every analysis automatically 
                generates a session report with all parameters and versions used, ensuring complete traceability of results.
                </p>
            </h2>

            <h4 class="home-h4">
                <div class="home-body">
                    <br>
                    <p class="home-info-box">
                    This is the web application to run <strong>Internal</strong> and <strong>External</strong> analyses 
                    interactively. For a deeper understanding of parameters, pipeline functions, input image requirements, 
                    and expected outputs, visit the full documentation at 
                    <a class="home-link" href="https://traitly.readthedocs.io/" target="_blank">traitly.readthedocs.io</a>
                    </p>
                </div>
            </h4>

            <br><br>

            <h2 class="home-h2">What does Traitly analyze?</h2>
            <br>

            <p style="margin-bottom:1.2rem;font-size:2rem">Traitly works with two main types of fruit images:</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:2rem;margin:.8rem 0">
                <div>
                    <h3 class="home-h3">⭑ Internal images (cross-section)</h3>
                    <ul style="margin:.2rem 0 .8rem;padding-left:5rem;font-size:2rem">
                        <li>Internal fruit morphology</li>
                        <li>Number and distribution of locules</li>
                        <li>Pericarp thickness</li>
                        <li>Symmetry</li>
                        <li>Color of internal tissues</li>
                    </ul>
                </div>
                <div>
                    <h3 class="home-h3">⭑ External images (surface)</h3>
                    <ul style="margin:.2rem 0 .8rem;padding-left:5rem;font-size:2rem">
                        <li>General fruit shape</li>
                        <li>Size</li>
                        <li>Surface color</li>
                    </ul>
                </div>
            </div>
            <p style="font-size:2rem">
                <i>Optionally</i>, Traitly can 
                <strong>convert pixels to real metric units</strong> through automatic detection of a size 
                reference marker present in the image.
            </p>

            <br><br>

            <h2 class="home-h2">Methodological approach</h2>
            <div class="home-body">
                <p style="margin-bottom:1.2rem">
                    The core of Traitly\'s analysis is based primarily on <strong>classical segmentation and 
                    traditional image processing</strong>, complemented by pre-trained models for auxiliary tasks 
                    such as label or size reference detection.
                </p>
                <p>
                This design prioritizes <strong>robustness, interpretability, and reproducibility</strong>, 
                and allows the method to be <strong>easily adaptable</strong> beyond fruits. With minimal parameter 
                adjustments, the same approach can be applied to other tissues such as <strong>seeds or leaves</strong>, 
                without redefining the pipeline architecture.
                </p>
            </div>

            <br><br>

            <h2 class="home-h2">Key features</h2>
            <div class="home-body">
                <ul style="margin:.2rem 0 .8rem;padding-left:5rem;font-size:2rem;margin-bottom:1.2rem">
                    <li style="margin-bottom:.4rem"><strong>Single image or batch processing</strong>: analyze a single image or entire folders in one run.</li>
                    <li style="margin-bottom:.4rem"><strong>Per-fruit measurements</strong>: each detected fruit receives a unique ID and is measured independently.</li>
                    <li style="margin-bottom:.4rem"><strong>Fully automated</strong>: detection, segmentation, calibration, and trait extraction without manual measurements.</li>
                    <li style="margin-bottom:.4rem"><strong>Pre-trained models included</strong>: automatic detection of size markers and sample labels.</li>
                    <li style="margin-bottom:.4rem"><strong>Color correction</strong>: Macbeth Color Checker detection to standardize color across experiments.</li>
                    <li style="margin-bottom:.4rem"><strong>Automatic sample identification</strong>: detection of QR codes and text labels.</li>
                    <li style="margin-bottom:.4rem"><strong>PDF support</strong>: direct conversion of scanned PDF files to images.</li>
                    <li style="margin-bottom:.4rem"><strong>Session reports</strong>: automatically saves parameters, dependency versions, and metadata for every run.</li>
                </ul>
            </div>

            <br><br>

            <h2 class="home-h2">Built on solid foundations</h2>
            <div class="home-body">
                <p style="margin-bottom:1.2rem">
                    Traitly relies on well-established libraries from the Python scientific ecosystem. Core processing 
                    uses <strong>OpenCV (contrib)</strong>, <strong>NumPy</strong>, <strong>SciPy</strong>, 
                    <strong>pandas</strong>, and <strong>matplotlib</strong>, all with C/C++ backends that guarantee 
                    high performance even in large-scale batch analyses.
                </p>
                <p>
                    This makes Traitly particularly well-suited for <strong>high-throughput phenotyping experiments</strong> 
                    in plant breeding and genetics, where analyzing large populations is common.
                </p>
            </div>
        </div>
        '''),
    ),
    value="tab_home",
)

tab_analysis = ui.nav_panel("Analysis",
    ui.navset_hidden(
        step_setup, step_mask, step_contrast, step_locule,
        step_detect, step_morph, step_color,
        id="pipeline_step", selected="step_setup",
    ),
    value="tab_analysis",
)

tab_bg = ui.nav_panel("BG Helper",
    ui.h5("🎨 Background Color Helper", style="font-weight:600;margin-bottom:.4rem"),
    ui.p("Upload an image, inspect HSV pixel colors, then tune your background thresholds.",
         class_="text-muted small"),
    ui.hr(),
    ui.input_file("bg_upload", "Upload image",
                  accept=[".jpg",".jpeg",".png",".bmp",".tiff",".tif"]),
    ui.output_ui("bg_main_ui"),
    value="tab_bg",
)

tab_batch = ui.nav_panel("Batch",
    ui.h5("📊 Batch Analysis", style="font-weight:600;margin-bottom:.4rem"),
    ui.p("Select all images you want to process — hold Ctrl/Cmd to pick multiple files.",
         class_="text-muted small"),
    ui.hr(),
    ui.layout_columns(
        ui.div(
            ui.HTML('<div class="sb-label">Images</div>'),
            ui.input_file(
                "batch_files", "Select images",
                accept=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
                multiple=True,
            ),
            ui.output_ui("batch_file_info"),
            ui.hr(),
            ui.HTML('<div class="sb-label">Analysis Mode</div>'),
            ui.input_radio_buttons(
                "batch_mode", None,
                choices={"external": "⋆˙⟡ External Fruit Analysis", "internal": "⋆˙⟡ Internal"},
                selected="external",
                inline=True,
            ),
            ui.hr(),
            ui.HTML('<div class="sb-label">Pipeline</div>'),
            ui.input_checkbox("run_morphology",  "Analyze morphology", True),
            ui.input_checkbox("run_color_batch", "Analyze color", True),
            ui.hr(),
            ui.HTML('<div class="sb-label">Quick Config</div>'),
            ui.input_select("bg_color_batch", "Background", choices=["blue","black","white"]),
            ui.input_numeric("min_fruit_area_batch", "Min fruit area (px)", 500, min=1, step=100),
            ui.input_slider("min_circ_batch", "Min circularity", 0.0, 1.0, 0.5, step=0.05),
            ui.hr(),
            ui.input_action_button("run_batch", "▶  Run Batch Analysis", class_="btn btn-primary"),
        ),
        ui.div(ui.output_ui("batch_results")),
        col_widths=[3,9],
    ),
    value="tab_batch",
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
sidebar_ui = ui.sidebar(
    ui.output_ui("sidebar_content"),
    width="400px",
    open="always",
)

# ══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
app_ui = ui.page_sidebar(
    sidebar_ui,
    ui.tags.head(
        ui.tags.link(rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"),
    ),
    ui.tags.style(_CSS),
    ui.HTML(_HEADER),
    ui.navset_hidden(
        tab_home, tab_analysis, tab_bg, tab_batch,
        id="main_tab",
        selected="tab_home",
    ),
    title="Traitly",
    fillable=False,
)


# ══════════════════════════════════════════════════════════════════════════════
# SERVER
# ══════════════════════════════════════════════════════════════════════════════
def server(input: Inputs, output: Outputs, session: Session):

    # ── reactive state ────────────────────────────────────────────────────────
    r_analyzer   = reactive.value(None)
    r_completed  = reactive.value([])
    r_mode       = reactive.value("home")
    r_cur_step   = reactive.value("step_setup")
    r_bg_analyzer = reactive.value(None)
    r_output_folder = reactive.value("")
    r_batch_zip  = reactive.value(None)
    r_morph_zip  = reactive.value(None)
    r_morph_base = reactive.value("morphology")
    r_color_zip  = reactive.value(None)          # ← new
    r_color_base = reactive.value("color")       # ← new
    r_img_shape  = reactive.value((100, 100))
    r_step1_result = reactive.value(ui.div())
    r_setup_done = reactive.value(0)
    r_img_ready  = reactive.value(False)
    r_upload_key = reactive.value(0)

    # ── step definitions per mode ─────────────────────────────────────────────
    def _steps(mode):
        if mode == "internal":
            return [
                ("step_setup",    "", "Setup"),
                ("step_mask",     "", "Fruit Mask"),
                ("step_contrast", "", "Enhance Contrast"),
                ("step_locule",   "", "Locule Mask"),
                ("step_detect",   "", "Detect Fruits"),
                ("step_morph",    "", "Morphology"),
                ("step_color",    "", "Color"),
            ]
        return [
            ("step_setup",  "", "Setup"),
            ("step_mask",   "", "Fruit Mask"),
            ("step_detect", "", "Detect Fruits"),
            ("step_morph",  "", "Morphology"),
            ("step_color",  "", "Color"),
        ]

    def mark_done(idx):
        d = list(r_completed.get())
        if idx not in d: d.append(idx)
        r_completed.set(d)

    # ── JS → switch main tab ──────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.js_main_tab)
    def _on_main_tab():
        tab = input.js_main_tab()
        ui.update_navs("main_tab", selected=tab, session=session)
        if tab not in ("tab_analysis",):
            r_mode.set(tab.replace("tab_", ""))

    @reactive.effect
    @reactive.event(input.js_mode)
    def _on_mode():
        new_mode = input.js_mode()
        if new_mode != r_mode.get():
            r_mode.set(new_mode)
            r_completed.set([])
            r_cur_step.set("step_setup")
            r_analyzer.set(None)
            r_img_ready.set(False)
            r_step1_result.set(ui.div())
            r_upload_key.set(r_upload_key.get() + 1)
            ui.update_navs("pipeline_step", selected="step_setup", session=session)

    # ── JS → switch step ─────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.js_step_click)
    def _on_step():
        sid = input.js_step_click()
        ui.update_navs("pipeline_step", selected=sid, session=session)
        r_cur_step.set(sid)

    # ── sidebar render ────────────────────────────────────────────────────────
    @render.ui
    def sidebar_content():
        mode = r_mode.get()
        done = r_completed.get()
        cur  = r_cur_step.get()

        if mode not in ("internal", "external"):
            return ui.div(
                ui.HTML('<div class="sb-label">Navigation</div>'),
                ui.HTML('<p style="font-size:1.0rem;color:#64748b;padding:.3rem .5rem">'
                        'Select Internal or External from the top bar to start the pipeline.</p>'),
            )

        steps = _steps(mode)
        items = []
        for i, (sid, icon, label) in enumerate(steps):
            is_done   = i in done
            is_active = sid == cur
            cls = "step-link"
            if is_active: cls += " active"
            if is_done:   cls += " done"
            chk = '<span class="step-check">✓</span>' if is_done else ""
            items.append(ui.HTML(
                f'<button class="{cls}" '
                f'onclick="goStep(\'{sid}\')">'
                f'<span class="step-num">{i+1}</span>'
                f'<span class="step-label">{icon} {label}</span>'
                f'{chk}</button>'
            ))

        return ui.div(
            ui.HTML('<div class="sb-label">Pipeline Steps</div>'),
            *items,
            ui.hr(),
            ui.input_action_button("reset_btn", "↻ Reset", class_="btn btn-reset"),
        )

    # reset
    @reactive.effect
    @reactive.event(input.reset_btn)
    def _reset():
        r_analyzer.set(None); r_completed.set([]); r_cur_step.set("step_setup")
        ui.update_navs("pipeline_step", selected="step_setup", session=session)

    @render.ui
    def upload_input_ui():
        r_upload_key.get()
        return ui.input_file("upload_img", "Upload a fruit image",
                            accept=[".jpg",".jpeg",".png",".bmp",".tiff",".tif"])

    # step 1
    @render.ui
    def dimensions_ui():
        if input.use_dimensions():
            return ui.div(
                ui.input_numeric("width_cm",  "Width (cm)",  21.59, min=0.0, step=0.01),
                ui.input_numeric("length_cm", "Length (cm)", 27.94, min=0.0, step=0.01),
            )
        return ui.div()

    @render.ui
    def crop_ui():
        img_h, img_w = r_img_shape.get()
        if input.use_crop():
            return ui.div(
                ui.layout_columns(
                    ui.input_numeric("crop_x", "x (left px)", 0,     min=0, step=1),
                    ui.input_numeric("crop_y", "y (top px)",  0,     min=0, step=1),
                    col_widths=[6,6],
                ),
                ui.layout_columns(
                    ui.input_numeric("crop_w", "w (width px)", img_w, min=1, step=1),
                    ui.input_numeric("crop_h", "h (height px)", img_h, min=1, step=1),
                    col_widths=[6,6],
                ),
                ui.layout_columns(
                    ui.input_action_button("apply_crop", "Apply Crop",
                                           class_="btn btn-secondary",
                                           style="font-size:1.6rem; margin-top:.5rem;"),
                    ui.input_action_button("reset_crop", "↻ Reset Size",
                                           class_="btn btn-outline-secondary",
                                           style="font-size:1.6rem; margin-top:.5rem;"),
                    col_widths=[6,6],
                ),
            )
        return ui.div()

    def _do_load_image():
        f = input.upload_img()
        if not f: return
        path = f[0]["datapath"]
        mode = r_mode.get()
        az = (FruitInternalAnalyzer(path) if mode == "internal"
              else FruitExternalAnalyzer(path))
        az.load_image(plot=False)
        r_img_shape.set(az.img_shape)
        if input.use_crop():
            az.load_image(
                plot=False,
                x=input.crop_x(), y=input.crop_y(),
                w=input.crop_w(), h=input.crop_h(),
            )
        r_analyzer.set(az)
        r_completed.set([])
        r_step1_result.set(ui.div())
        r_img_ready.set(True)

    @reactive.effect
    @reactive.event(input.upload_img)
    def _load_image(): _do_load_image()

    @reactive.effect
    @reactive.event(input.apply_crop)
    def _on_apply_crop(): _do_load_image()

    @reactive.effect
    @reactive.event(input.reset_crop)
    def _on_reset_crop():
        img_h, img_w = r_img_shape.get()
        ui.update_numeric("crop_x", value=0,     session=session)
        ui.update_numeric("crop_y", value=0,     session=session)
        ui.update_numeric("crop_w", value=img_w, session=session)
        ui.update_numeric("crop_h", value=img_h, session=session)
        f = input.upload_img()
        if not f: return
        path = f[0]["datapath"]
        mode = r_mode.get()
        az = (FruitInternalAnalyzer(path) if mode == "internal"
              else FruitExternalAnalyzer(path))
        az.load_image(plot=False)
        r_analyzer.set(az)
        r_completed.set([])

    @render.image
    def step1_preview():
        r_setup_done.get()
        if not r_img_ready.get():
            return None
        az = r_analyzer.get()
        if az is None or az.img_rgb is None:
            f = input.upload_img()
            if not f: return None
            return {"src": f[0]["datapath"], "alt": "Uploaded", "width": "100%"}
        display_img = cv2.cvtColor(az.img_copy, cv2.COLOR_BGR2RGB) if \
            (hasattr(az, "img_copy") and az.img_copy is not None) else az.img_rgb
        if input.use_crop():
            fig, ax = plt.subplots(figsize=(9,9))
            ax.imshow(display_img if display_img.shape[2] == 3 and display_img.dtype == np.uint8
                      else display_img)
            ax.axis("on")
            fig.tight_layout()
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig.savefig(tmp.name, dpi=100)
            plt.close(fig)
            return {"src": tmp.name, "alt": "Preview", "class": "img-zoomable"}
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(display_img).save(tmp.name)
        return {"src": tmp.name, "alt": "Preview", "class": "img-zoomable"}

    @reactive.effect
    @reactive.event(input.run_step1)
    def _run_step1():
        az = r_analyzer.get()
        if az is None:
            r_step1_result.set(ui.p("Upload an image first.", class_="text-info"))
            return
        try:
            az.setup_measurements(
                detect_label=input.detect_label(),
                confidence=input.confidence(),
                skip_qr=input.skip_qr(),
                detect_color_checker=input.detect_color_checker(),
                width_cm=input.width_cm()   if input.use_dimensions() else None,
                length_cm=input.length_cm() if input.use_dimensions() else None,
                diameter_cm=input.diameter_cm(), verbose=False,
            )
            mark_done(0)
            h, w = az.img.shape[:2] if az.img is not None else (0, 0)
            n_refs = len(az.ref_roi) if az.ref_roi else 0
            r_step1_result.set(ui.div(
                ui.p("Setup complete!",
                     style="font-size:3.4rem; text-align:center; max-width:700px; "
                            "margin:0 auto; color: #97c8ec; font-weight:700; background-color:rgba(49,63,65,0.8);"),
                ui.div(
                    ui.layout_columns(
                        ui.value_box("Label:", az.label_text or "N/A", theme="primary"),
                        ui.value_box("px/cm density:", f"{az.px_per_cm:.1f}" if az.px_per_cm else "–", theme="primary"),
                        col_widths=[6, 6],
                    ),
                    ui.layout_columns(
                        ui.value_box("Image size:", f"{w}×{h}", theme="secondary"),
                        ui.value_box("References:", str(n_refs), theme="success" if n_refs > 0 else "danger"),
                        col_widths=[6, 6],
                    ),
                    style="max-width: 700px; margin: 0 auto;",
                ),
            ))
            r_setup_done.set(r_setup_done.get() + 1)
        except Exception as e:
            r_step1_result.set(
                ui.div(ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc()))
            )

    @render.ui
    def step1_results():
        return r_step1_result.get()

    # ── Background helper preview (live) ──────────────────────────────────────
    @render.ui
    def bg_preview():
        az = r_bg_analyzer.get()
        if az is None or az.img_hsv is None: return ui.div()
        lo = np.array([input.l_h(), input.l_s(), input.l_v()], dtype=np.uint8)
        hi = np.array([input.u_h(), input.u_s(), input.u_v()], dtype=np.uint8)
        mask = cv2.inRange(az.img_hsv, lo, hi)
        prev = cv2.cvtColor(az.img_bgr.copy(), cv2.COLOR_BGR2RGB)
        grn  = np.zeros_like(prev); grn[mask > 0] = [0, 220, 80]
        prev = cv2.addWeighted(prev, .65, grn, .35, 0)
        pct  = 100 * mask.sum() / 255 / mask.size
        return ui.div(
            ui.layout_columns(
                ui.HTML(img_tag(prev) + '<p class="text-muted small">🟢 Background selected</p>'),
                ui.HTML(img_tag(cv2.bitwise_not(mask)) + '<p class="text-muted small">Fruit mask (white = fruit)</p>'),
                col_widths=[6,6],
            ),
            ui.p(f"Coverage: {pct:.1f}% of pixels", class_="text-muted small"),
        )

    @render.ui
    @reactive.event(input.bg_detect_btn)
    def bg_detect_out():
        az = r_bg_analyzer.get()
        if az is None: return ui.p("Upload an image first.", class_="text-info")
        lo = [input.l_h(), input.l_s(), input.l_v()]
        hi = [input.u_h(), input.u_s(), input.u_v()]
        try:
            az.generate_fruit_mask(lower_hsv=lo, upper_hsv=hi, plot=False)
            az.detect_fruits(min_fruit_circularity=input.bg_circ(),
                            min_fruit_area=input.bg_area(), verbose=False, plot=False)
            n   = len(az.fruit_locule_map) if az.fruit_locule_map else 0
            msg = "✅ Mask looks good!" if n > 0 else "⚠️ No fruits — adjust thresholds."
            return ui.div(
                ui.value_box("Fruits detected", n, theme="success" if n > 0 else "warning"),
                ui.p(msg, class_="text-success" if n > 0 else "text-warning"),
            )
        except Exception as e:
            return ui.div(ui.p(f"❌ {e}", class_="text-danger"), ui.pre(traceback.format_exc()))

    @render.ui
    def bg_final_code():
        lo   = [input.l_h(), input.l_s(), input.l_v()]
        hi   = [input.u_h(), input.u_s(), input.u_v()]
        code = (f"lower_hsv = {lo}\nupper_hsv  = {hi}\n\n"
                f"min_fruit_circularity = {input.bg_circ()}\n"
                f"min_fruit_area        = {input.bg_area()}")
        return ui.div(
            ui.p("Use in Individual Analysis → Generate Mask or Batch:", class_="text-success"),
            ui.pre(code, style="background:#f1f5f9;padding:1rem;border-radius:6px;font-size:.81rem"),
        )

    # step 2
    @render.ui
    def mask_bg_ui():
        if r_mode.get() == "external":
            return ui.input_select("bg_color", "Background color",
                                   choices=["blue","black","white"])
        return ui.div()

    @render.ui
    @reactive.event(input.run_step2)
    def step2_results():
        az = r_analyzer.get()
        if az is None:
            return ui.p("Complete Step 1 first.", class_="text-info")
        is_int = r_mode.get() == "internal"
        try:
            lower_hsv = [input.h_range()[0], input.s_range()[0], input.v_range()[0]] if input.use_manual_hsv() else None
            upper_hsv = [input.h_range()[1], input.s_range()[1], input.v_range()[1]] if input.use_manual_hsv() else None
            kw = dict(
                stamp=input.stamp(), remove_roi=input.remove_roi(),
                lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                n_iteration=input.n_iteration(), roi_expansion=input.roi_expansion(),
                kernel_blur=input.kernel_blur() or None,
                kernel_open=input.kernel_open() or None,
                kernel_close=input.kernel_close() or None,
                apply_convex_hull=input.apply_convex_hull(),
                fill_holes=input.fill_holes(), erosion_px=input.erosion_px(),
                plot=True, plot_size=(20,20)
            )
            if not is_int:
                kw["background_color"] = input.bg_color()
            az.generate_fruit_mask(**kw)
            mark_done(1)
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")
            return ui.HTML(
                f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                f'style="width:100%;border-radius:8px;margin-top:.5rem">'
            )
        except Exception as e:
            return ui.div(ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc()))

    @render.ui
    def hsv_ui():
        if not input.use_manual_hsv():
            return ui.div()
        return ui.div(
            ui.input_slider("h_range", "H", 0, 180, [0, 180]),
            ui.input_slider("s_range", "S", 0, 255, [0, 255]),
            ui.input_slider("v_range", "V", 0, 255, [0, 255]),
            style="padding:.4rem 0 .2rem 0",
        )

    # step 3
    @render.ui
    @reactive.event(input.run_step3)
    def step3_results():
        az = r_analyzer.get()
        if az is None:
            return ui.p("Complete earlier steps first.", class_="text-info")
        try:
            method = input.contrast_method()
            gamma  = input.gamma()  if method == "gamma"   else 1.5
            gain   = input.gain()   if method == "sigmoid" else 5.0
            cutoff = input.cutoff() if method == "sigmoid" else 0.5
            c      = input.c_val()  if method == "exp"     else 0.5
            az.enhance_locule_contrast(
                contrast_method=method, gamma=gamma, gain=gain, cutoff=cutoff, c=c,
                plot=True, compare_method=input.compare_method(),
                kernel_blur=input.kernel_blur3(),
                clip_limit=input.clip_limit() or None,
                tile_grid_size=input.tile_grid_size(),
            )
            mark_done(2)
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")
            return ui.HTML(
                f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                f'style="width:100%;border-radius:8px;margin-top:.5rem">'
            )
        except Exception as e:
            return ui.div(ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc()))

    @render.ui
    def contrast_params_ui():
        method = input.contrast_method()
        if method == "gamma":
            return ui.input_slider("gamma", "Gamma", 0.1, 3.0, 1.5, step=0.1)
        elif method == "sigmoid":
            return ui.div(
                ui.input_slider("gain",   "Gain",   1.0, 10.0, 5.0, step=0.5),
                ui.input_slider("cutoff", "Cutoff", 0.0,  1.0, 0.5, step=0.05),
            )
        elif method == "exp":
            return ui.input_slider("c_val", "C value", 0.1, 2.0, 0.5, step=0.1)
        return ui.div()

    # step 4
    @render.ui
    def histogram_params_ui():
        if input.gen_histogram():
            return ui.input_slider("otsu_offset", "Otsu offset", -50, 50, 0, step=1)
        return ui.div()

    @render.ui
    @reactive.event(input.run_step4)
    def step4_results():
        az = r_analyzer.get()
        if az is None:
            return ui.p("Complete earlier steps first.", class_="text-info")
        try:
            thresh   = input.thresh_min()     if input.use_thresh() else 120
            otsu_off = input.otsu_offset_lm() if input.use_otsu()   else None
            if input.gen_histogram():
                az.generate_l_channel_histogram(otsu_offset=input.otsu_offset())
                buf = io.BytesIO()
                plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                plt.close("all")
                return ui.div(ui.HTML(
                    f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                    f'style="width:100%;border-radius:8px;margin-top:.5rem">'
                ))
            az.generate_locule_mask(
                thresh_min=thresh, otsu_offset=otsu_off,
                min_fruit_area=input.min_fruit_area_lm(),
                min_locule_area=input.min_locule_area_lm(),
                invert_locule=input.invert_locule(),
                kernel_blur=input.kernel_blur4() or None,
                kernel_open=input.kernel_open4() or None,
                kernel_close=input.kernel_close4() or None,
                erosion_px=input.erosion_px4(),
                plot=True, plot_size=(20,20)
            )
            mark_done(3)
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")
            return ui.div(ui.HTML(
                f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                f'style="width:100%;border-radius:8px;margin-top:.5rem">'
            ))
        except Exception as e:
            return ui.div(ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc()))

    @render.ui
    def thresh_ui():
        if input.use_thresh():
            return ui.input_slider("thresh_min", "Min threshold", 0, 255, 120)
        return ui.div()

    @render.ui
    def otsu_ui():
        if input.use_otsu():
            return ui.input_slider("otsu_offset_lm", "Otsu offset", -50, 50, 0, step=1)
        return ui.div()

    # step 5
    @render.ui
    def detect_locule_params_ui():
        if r_mode.get() == "internal":
            return ui.div(
                ui.input_numeric("min_locule_area",      "Min locule area (px)", 50,  min=1, step=10),
                ui.input_numeric("min_locule_per_fruit", "Min locules/fruit",     1,  min=0, step=1),
            )
        return ui.div()

    @render.ui
    @reactive.event(input.run_detect)
    def detect_results():
        az = r_analyzer.get()
        if az is None:
            return ui.p("Complete earlier steps first.", class_="text-info")
        is_int = r_mode.get() == "internal"
        try:
            def _parse_color(s):
                return tuple(int(x.strip()) for x in s.split(","))
            kw = dict(
                min_fruit_circularity=input.min_fruit_circularity(),
                min_fruit_area=input.min_fruit_area_det(),
                max_fruit_area=input.max_fruit_area_det() or None,
                rescale_factor=input.rescale_factor_det() or None,
                contour_thickness=input.contour_thickness_det(),
                contour_color=_parse_color(input.contour_color_det()),
                verbose=False, plot=True, plot_size=(20,20)
            )
            if is_int:
                kw["min_locule_area"]      = input.min_locule_area()
                kw["min_locule_per_fruit"] = input.min_locule_per_fruit()
                kw["locule_thickness"]     = input.locule_thickness_det()
                kw["locule_color"]         = _parse_color(input.locule_color_det())
            az.detect_fruits(**kw)
            idx = 4 if is_int else 2
            mark_done(idx)
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")
            return ui.HTML(
                f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                f'style="width:100%;border-radius:8px;margin-top:.5rem">'
            )
        except Exception as e:
            return ui.div(ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc()))

    # step 6 – morphology
    @render.ui
    def epsilon_ui():
        if input.contour_mode() == "approx":
            return ui.input_numeric("epsilon_morph", "Epsilon (approx simplification)", 0.001, min=0.0001, step=0.001)
        return ui.div()

    @render.ui
    def morph_locule_params_ui():
        if r_mode.get() == "internal":
            return ui.div(
                ui.input_numeric("min_locule_area_morph", "Min locule area (px)", 10, min=0, step=10),
                ui.input_numeric("max_locule_area_morph", "Max locule area (px)", 0, min=0, step=100),
                ui.p("Set to 0 for no upper limit.", style="font-size:1.4rem;color:#94a3b8;margin-top:-.5rem;"),
            )
        return ui.div()

    @render.ui
    @reactive.event(input.run_morph)
    def morph_results():
        az = r_analyzer.get()
        if az is None:
            return ui.p("Complete earlier steps first.", class_="text-info")
        is_int = r_mode.get() == "internal"
        try:
            def _pc(s):
                return tuple(int(x.strip()) for x in s.split(","))
            epsilon   = input.epsilon_morph() if input.contour_mode() == "approx" else 0.001
            alpha_val = input.alpha_morph()   if input.alpha_morph() > 0 else None
            max_loc   = input.max_locule_area_morph() or None if is_int else None
            kw = dict(
                contour_mode=input.contour_mode(), epsilon=epsilon,
                font_size=input.font_size_morph(), font_thickness=input.font_thickness_morph(),
                font_color=_pc(input.font_color_morph()),
                label_position=input.label_position_morph(),
                label_color=_pc(input.label_color_morph()),
                pericarp_ext_color=_pc(input.pericarp_ext_color_morph()),
                pericarp_ext_thickness=input.pericarp_ext_thick_morph(),
                display_table=True, plot=True,
            )
            if is_int:
                kw.update(dict(
                    alpha=alpha_val,
                    angle_shifts=input.angle_shifts_morph(),
                    num_rays=input.num_rays_morph(),
                    min_locule_area=input.min_locule_area_morph(),
                    max_locule_area=max_loc,
                    pericarp_int_color=_pc(input.pericarp_int_color_morph()),
                    pericarp_int_thickness=input.pericarp_int_thick_morph(),
                    locule_color=_pc(input.locule_color_morph()),
                    locule_thickness=input.locule_thick_morph(),
                    centroid_fruit_color=_pc(input.centroid_fruit_color_morph()),
                    centroid_fruit_thickness=input.centroid_fruit_thick_morph(),
                    centroid_locule_color=_pc(input.centroid_locule_color_morph()),
                    centroid_locule_thickness=input.centroid_locule_thick_morph(),
                ))
            df = az.analyze_morphology(**kw)
            idx = 5 if is_int else 3
            mark_done(idx)
            plt.close("all")
            b64 = arr_to_b64(cv2.cvtColor(az.results.annotated_image, cv2.COLOR_BGR2RGB))
            parts = [ui.HTML(
                f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                f'style="width:100%;border-radius:8px;margin-top:.5rem">'
            )]
            if df is not None and not df.empty:
                csv_b = df_csv(df)
                parts += [
                    ui.output_data_frame("morph_table"),
                    ui.download_button("dl_morph", "⬇ Download CSV",
                                    class_="btn btn-primary mt-2",
                                    style="margin-right: 1.5rem !important"),
                ]
                @render.data_frame
                def morph_table(): return render.DataGrid(df, height="320px")
                @render.download(filename="morphology_results.csv")
                async def dl_morph(): yield csv_b
            # Zip results
            tmp_dir = tempfile.mkdtemp()
            base = os.path.splitext(os.path.basename(az.img_path))[0]
            ann_path = os.path.join(tmp_dir, f"{base}_annotated.png")
            if az.results is not None and az.results.annotated_image is not None:
                cv2.imwrite(ann_path, az.results.annotated_image)
            if df is not None and not df.empty:
                df.to_csv(os.path.join(tmp_dir, f"{base}_morphology_results.csv"), index=False)
            params_saved = False
            if input.save_params_morph():
                params_saved = True
                az.save_parameters(output_path=tmp_dir)
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname in os.listdir(tmp_dir):
                    zf.write(os.path.join(tmp_dir, fname), arcname=fname)
            r_morph_zip.set(zip_buf.getvalue())
            r_morph_base.set(base)
            parts.append(ui.download_button(
                "dl_morph_zip", "⬇ Download image + csv (.zip)",
                class_="btn btn-primary mt-2",
            ))
            if params_saved:
                parts.append(ui.p(
                    "Parameters files created and included in the .zip!",
                    style="font-size:1.6rem; text-align:center; max-width:700px; "
                          "color:#97c8ec; font-weight:700; "
                          "background-color:rgba(49,63,65,0.9); border-radius:6px; padding:.4rem;"
                ))
            return ui.div(*parts)
        except Exception as e:
            return ui.div(ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc()))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 – Color
    # ══════════════════════════════════════════════════════════════════════════

    @render.ui
    def color_tissue_ui():
        if r_mode.get() == "internal":
            return ui.input_select("tissue", "Tissue",
                                   choices=["all", "total_pericarp", "outer_pericarp",
                                            "inner_pericarp", "locules"])
        return ui.div()

    # Internal-only plot styling fields (shown inside the Plot styling collapsible)
    @render.ui
    def color_int_styling_ui():
        if r_mode.get() == "internal":
            return ui.div(
                ui.input_text("pericarp_int_color_color",   "Int. pericarp color (R,G,B)",  "255,255,0"),
                ui.input_numeric("pericarp_int_thick_color","Int. pericarp thickness", 2, min=1, step=1),
                ui.input_text("locule_color_color",         "Locule color (R,G,B)",         "255,0,255"),
                ui.input_numeric("locule_thick_color",      "Locule thickness", 2, min=1, step=1),
            )
        return ui.div()

    @render.ui
    @reactive.event(input.run_color)
    def color_results():
        az = r_analyzer.get()
        if az is None:
            return ui.p("Complete earlier steps first.", class_="text-info")
        is_int = r_mode.get() == "internal"
        try:
            def _pc(s):
                return tuple(int(x.strip()) for x in s.split(","))

            # ── Build kwargs ───────────────────────────────────────────────
            kw = dict(
                stat=input.stat(),
                color_space=input.color_space(),
                get_color_histogram=input.get_color_histogram(),
                display_table=True,
                plot=False,                       # we render color_image manually
                font_size=input.font_size_color(),
                font_thickness=input.font_thickness_color(),
                font_color=_pc(input.font_color_color()),
                label_position=input.label_position_color(),
                label_color=_pc(input.label_color_color()),
                label_opacity=input.label_opacity_color(),
                pericarp_ext_color=_pc(input.pericarp_ext_color_color()),
                pericarp_ext_thickness=input.pericarp_ext_thick_color(),
                dark_thresh=input.dark_thresh_color(),
            )
            if is_int:
                kw["tissue"]                 = input.tissue()
                kw["pericarp_int_color"]     = _pc(input.pericarp_int_color_color())
                kw["pericarp_int_thickness"] = input.pericarp_int_thick_color()
                kw["locule_color"]           = _pc(input.locule_color_color())
                kw["locule_thickness"]       = input.locule_thick_color()

            df = az.analyze_color(**kw)
            idx = 6 if is_int else 4
            mark_done(idx)

            # ── Display color_image (not annotated_image) ─────────────────
            color_img = getattr(az.results, "color_image", None)
            if color_img is None:
                color_img = getattr(az.results, "annotated_image", None)

            parts = []
            if color_img is not None:
                b64 = arr_to_b64(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
                parts.append(ui.HTML(
                    f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                    f'style="width:100%;border-radius:8px;margin-top:.5rem">'
                ))

            # ── Table + CSV download ───────────────────────────────────────
            if df is not None and not df.empty:
                csv_b = df_csv(df)
                parts += [
                    ui.output_data_frame("color_table"),
                    ui.download_button("dl_color_csv", "⬇ Download CSV",
                                       class_="btn btn-primary mt-2",
                                       style="margin-right: 1.5rem !important"),
                ]
                @render.data_frame
                def color_table(): return render.DataGrid(df, height="320px")

                @render.download(filename="color_results.csv")
                async def dl_color_csv(): yield csv_b

            # ── Zip: prefer annotated_image, fall back to color_image ─────
            tmp_dir = tempfile.mkdtemp()
            base    = os.path.splitext(os.path.basename(az.img_path))[0]

            ann_img  = getattr(az.results, "annotated_image", None)
            col_img  = getattr(az.results, "color_image", None)

            if ann_img is not None:
                img_to_save  = ann_img
                img_filename = f"{base}_annotated.png"
            elif col_img is not None:
                img_to_save  = col_img
                img_filename = f"{base}_color.png"
            else:
                img_to_save  = None
                img_filename = None

            if img_to_save is not None:
                cv2.imwrite(os.path.join(tmp_dir, img_filename), img_to_save)

            if df is not None and not df.empty:
                df.to_csv(os.path.join(tmp_dir, f"{base}_color_results.csv"), index=False)

            params_saved = False
            if input.save_params_color():
                params_saved = True
                az.save_parameters(output_path=tmp_dir)

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname in os.listdir(tmp_dir):
                    zf.write(os.path.join(tmp_dir, fname), arcname=fname)
            r_color_zip.set(zip_buf.getvalue())
            r_color_base.set(base)

            parts.append(ui.download_button(
                "dl_color_zip", "⬇ Download image + csv (.zip)",
                class_="btn btn-primary mt-2",
            ))
            if params_saved:
                parts.append(ui.p(
                    "Parameters files created and included in the .zip!",
                    style="font-size:1.6rem; text-align:center; max-width:700px; "
                          "color:#97c8ec; font-weight:700; "
                          "background-color:rgba(49,63,65,0.9); border-radius:6px; padding:.4rem;"
                ))
            return ui.div(*parts)

        except Exception as e:
            return ui.div(ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc()))

    # ── Color zip download handler ─────────────────────────────────────────
    @render.download(filename=lambda: f"{r_color_base.get()}_color.zip")
    async def dl_color_zip():
        data = r_color_zip.get()
        if data:
            yield data

    # ══════════════════════════════════════════════════════════════════════════
    # BACKGROUND HELPER
    # ══════════════════════════════════════════════════════════════════════════
    @reactive.effect
    @reactive.event(input.bg_upload)
    def _load_bg():
        f = input.bg_upload()
        if not f: return
        az = FruitExternalAnalyzer(f[0]["datapath"])
        az.load_image(plot=False)
        r_bg_analyzer.set(az)

    @render.ui
    def bg_main_ui():
        az = r_bg_analyzer.get()
        if az is None:
            return ui.p("👆 Upload an image above.", class_="text-info")
        return ui.div(
            ui.HTML(img_tag(cv2.cvtColor(az.img, cv2.COLOR_BGR2RGB),
                            "width:100%;max-height:280px;object-fit:contain;"
                            "border-radius:8px;margin-bottom:.8rem")),
            ui.hr(),
            ui.h6("2️⃣ Inspect HSV distribution"),
            ui.layout_columns(
                ui.div(
                    ui.input_numeric("bg_sample","Sample size",10000,min=1000,max=50000,step=1000),
                    ui.input_action_button("bg_scatter_btn","▶  Generate scatterplot",
                                           class_="btn btn-primary"),
                ),
                ui.output_ui("bg_scatter_out"),
                col_widths=[3,9],
            ),
            ui.hr(),
            ui.h6("3️⃣ Define HSV thresholds — live preview"),
            ui.input_select("bg_preset","Preset",
                            choices=["blue","white","black","gray (example)","custom"]),
            ui.layout_columns(
                ui.div(
                    ui.HTML('<div style="font-size:1.6rem;font-weight:600;margin-bottom:.4rem">Lower HSV</div>'),
                    ui.input_slider("lower_h", "H min", 0, 180, 0),
                    ui.input_slider("lower_s", "S min", 0, 255, 0),
                    ui.input_slider("lower_v", "V min", 0, 255, 0),
                ),
                ui.div(
                    ui.HTML('<div style="font-size:1.6rem;font-weight:600;margin-bottom:.4rem">Upper HSV</div>'),
                    ui.input_slider("upper_h", "H max", 0, 180, 180),
                    ui.input_slider("upper_s", "S max", 0, 255, 255),
                    ui.input_slider("upper_v", "V max", 0, 255, 255),
                ),
                col_widths=[6, 6],
            ),
            ui.output_ui("bg_preview"),
            ui.hr(),
            ui.h6("4️⃣ Verify detections"),
            ui.layout_columns(
                ui.div(
                    ui.input_slider("bg_circ","Min circularity",0.0,1.0,0.5,step=0.05),
                    ui.input_numeric("bg_area","Min fruit area (px)",500,min=1,step=100),
                    ui.input_action_button("bg_detect_btn","▶  Run detect_fruits",
                                           class_="btn btn-primary"),
                ),
                ui.output_ui("bg_detect_out"),
                col_widths=[4,8],
            ),
            ui.hr(),
            ui.h6("5️⃣ Copy these values"),
            ui.output_ui("bg_final_code"),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # BATCH ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    @render.ui
    def batch_file_info():
        files = input.batch_files()
        if not files:
            return ui.p("No files selected yet.", class_="text-muted small")
        n = len(files)
        names = [f["name"] for f in files[:5]]
        preview = ", ".join(names) + ("…" if n > 5 else "")
        return ui.div(
            ui.p(f"📷 {n} image(s) selected", class_="text-success small fw-bold"),
            ui.p(preview, class_="text-muted small",
                 style="word-break:break-all;font-size:.75rem"),
        )

    @render.ui
    @reactive.event(input.run_batch)
    def batch_results():
        files = input.batch_files()
        if not files:
            return ui.p("Select images first.", class_="text-info")

        n_total     = len(files)
        is_int      = input.batch_mode() == "internal"
        tmp_dir     = tempfile.mkdtemp()
        output_path = os.path.join(tmp_dir, "Results")
        os.makedirs(output_path, exist_ok=True)

        all_morphology, all_color, errors = [], [], []
        total_fruits = 0
        saved_images = []

        cfg = dict(
            background_color      = input.bg_color_batch(),
            min_fruit_area        = input.min_fruit_area_batch(),
            min_fruit_circularity = input.min_circ_batch(),
        )

        with ui.Progress(min=0, max=n_total, session=session) as p:
            for i, f in enumerate(files):
                fname = f["name"]
                p.set(value=i, message=f"Processing {i+1}/{n_total}", detail=fname)
                try:
                    src  = f["datapath"]
                    dest = os.path.join(tmp_dir, fname)
                    shutil.copy2(src, dest)
                    az = (FruitInternalAnalyzer(dest) if is_int
                          else FruitExternalAnalyzer(dest))
                    az.load_image(plot=False)
                    df_m, df_c, err, n_fruits, ann_img = az.process_single_file(
                        config=cfg,
                        analyze_morphology=input.run_morphology(),
                        analyze_color=input.run_color_batch(),
                        save_image=True,
                        output_path=output_path,
                    )
                    if err:
                        errors.append({"filename": fname, **err})
                    else:
                        if df_m is not None: all_morphology.append(df_m)
                        if df_c is not None: all_color.append(df_c)
                        total_fruits += (n_fruits or 0)
                        stem     = os.path.splitext(fname)[0]
                        ann_path = os.path.join(output_path, f"{stem}_annotated.jpg")
                        if ann_img is not None and not os.path.exists(ann_path):
                            import cv2 as _cv2
                            _cv2.imwrite(ann_path, ann_img)
                        if os.path.exists(ann_path):
                            saved_images.append(ann_path)
                except Exception as e:
                    errors.append({"filename": fname, "status": str(e)})
            p.set(value=n_total, message="Done!", detail="")

        df_morph_all = pd.concat(all_morphology, ignore_index=True) if all_morphology else None
        df_color_all = pd.concat(all_color,      ignore_index=True) if all_color      else None
        if df_morph_all is not None:
            df_morph_all.to_csv(os.path.join(output_path, "morphology_results.csv"), index=False)
        if df_color_all is not None:
            df_color_all.to_csv(os.path.join(output_path, "color_results.csv"), index=False)

        zip_buf = _io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in os.listdir(output_path):
                full = os.path.join(output_path, fpath)
                if os.path.isfile(full):
                    zf.write(full, arcname=fpath)
        r_batch_zip.set(zip_buf.getvalue())

        n_ok  = n_total - len(errors)
        parts = [
            ui.p(f"✅ Batch complete! {n_ok}/{n_total} images processed.",
                 class_="text-success fw-bold"),
            ui.layout_columns(
                ui.value_box("Processed",        n_ok,              theme="success"),
                ui.value_box("Fruits detected",  total_fruits,      theme="primary"),
                ui.value_box("Annotated images", len(saved_images), theme="info"),
                ui.value_box("Errors",           len(errors),       theme="warning" if errors else "secondary"),
            ),
            ui.p(f"🖼️ {len(saved_images)} annotated image(s) + CSVs included in the zip.",
                 class_="text-info small", style="margin-top:.5rem"),
            ui.download_button("dl_batch_zip", "📦 Download all results (.zip)",
                               class_="btn btn-primary mt-1 mb-3"),
        ]
        if errors:
            err_lines = "\n".join(
                f"  • {e.get('filename', '?')}: {e.get('status', 'unknown')}" for e in errors
            )
            parts.append(ui.div(
                ui.p(f"⚠️ {len(errors)} error(s):", class_="text-warning small fw-bold"),
                ui.pre(err_lines,
                       style="background:#fff8ed;padding:.7rem;border-radius:6px;"
                             "font-size:.78rem;color:#92400e"),
            ))
        for i, (label, df) in enumerate([("Morphology", df_morph_all), ("Color", df_color_all)]):
            if df is not None and not df.empty:
                uid   = f"bdf_{i}"
                dl_id = f"dl_batch_{i}"
                fn    = f"{'morphology' if i == 0 else 'color'}_results.csv"
                csv_b = df_csv(df)
                parts += [
                    ui.h6(f"📊 {label} ({len(df)} rows)"),
                    ui.output_data_frame(uid),
                    ui.download_button(dl_id, f"📥 Download {fn}",
                                       class_="btn btn-outline-primary btn-sm mb-2"),
                ]
                def _reg(df=df, csv_b=csv_b, uid=uid, dl_id=dl_id, fn=fn):
                    @session.output(id=uid)
                    @render.data_frame
                    def _tbl(): return render.DataGrid(df, height="280px")
                    @session.output(id=dl_id)
                    @render.download(filename=fn)
                    async def _dl(): yield csv_b
                _reg()
        return ui.div(*parts)

    # ── Download handlers ─────────────────────────────────────────────────────
    @render.download(filename=lambda: f"{r_morph_base.get()}_morphology.zip")
    async def dl_morph_zip():
        data = r_morph_zip.get()
        if data: yield data

    @render.download(filename="traitly_results.zip")
    async def dl_batch_zip():
        data = r_batch_zip.get()
        if data: yield data


app = App(app_ui, server)