# traitly/shiny_app/app.py

#########################################################################################
# STANDARD LIBRARY
#########################################################################################
from __future__ import annotations

import base64
import io
import json as _json
import os
import shutil
import tempfile
import traceback
import zipfile

#########################################################################################
# THIRD-PARTY
#########################################################################################
import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np
import pandas as pd

try:
    from shiny import App, Inputs, Outputs, Session, reactive, render, ui
except ImportError:
    raise RuntimeError(
        f"Traitly App requires shiny installed. To install, run:\n"
        'pip install "traitly[app]"'
    )
#########################################################################################
# INTERNAL IMPORTS
#########################################################################################
try:
    from traitly import __version__
    from traitly.fruit_phenotyping import FruitExternalAnalyzer, FruitInternalAnalyzer

    traitly_available = True
except ImportError:
    __version__ = "dev"
    traitly_available = False


_CSS = """
/* for HF */
/* html { font-size: clamp(7px, 0.55vw, 12px); } */
/* body { font-size: clamp(7px, 0.55vw, 12px); } */

html { font-size: clamp(9px, 0.55vw, 12px); }
body { font-size: clamp(9px, 0.55vw, 12px); }

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

/* header bar HF*/
.traitly-header {
    position:fixed; top:0; left:0; right:0; height:100px;
    /* position:fixed; top:0; left:0; right:0; height:150px; */
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

/* push content below header HF */
.bslib-page-main, main {
    /* margin-top: 155px !important; */
    margin-top: 105px !important;
    margin-left: 20px !important;
    padding-top: .5rem;
}

.bslib-sidebar-layout > .main {
    padding-left: 0.5rem !important;
}

/* sidebar HF */
.bslib-sidebar-layout > .sidebar {
    background: var(--sidebar-bg) !important;
    border-right:1px solid var(--sidebar-border) !important;
    /* padding-top:.7rem; top:155px !important; */
    padding-top:.7rem; top:105px !important;
}

.sb-label {
    font-size:1.9rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.09rem; color:#94a3b8; margin: 1rem 6.9rem 1rem 0.8rem;
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

/* HF */
.step-num {
    width: 38px; height: 38px; border-radius: 50%;
    background: var(--step-num-bg); color: var(--step-num-color);
    font-size: 1.5rem; font-weight: 800;
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.step-check { color:#059669; font-size:1.6rem; margin-left:auto; }

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

body.on-nosidebar .bslib-sidebar-layout > .sidebar {
    display: none !important;
}

body.on-nosidebar .bslib-sidebar-layout {
    grid-template-columns: 0 1fr !important;
    padding-left: 0 !important;
}

body.on-nosidebar .bslib-sidebar-layout > .main,
body.on-nosidebar .bslib-page-main,
body.on-nosidebar main {
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
    max-width: 100%;
    overflow: hidden;
}

#step1_preview img {
    max-height: 1500px;
    max-width: 100%;
    object-fit: contain;
    display: block;
    margin: 0 auto;
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

/* mask editor */
#img_mask {
    image-rendering: auto;
}
#img_orig {
    image-rendering: auto;
}

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


/* hr lines in dark */
body.dark-theme hr {
    border-color: rgba(105,141,151, 0.8) !important;
    opacity: 1 !important;
}
body.dark-theme .tab-content hr {
    border-color: rgba(105,141,151, 0.8) !important;
    opacity: 1 !important;
}

body.dark-theme .tab-content details summary { color: #94a3b8 !important; }

/* ? bubble in dark */
body.dark-theme .tooltip-wrap > span {
    background: #3d3d5c !important;
    color: #cbd5e1 !important;
}


/* slider track + filled portion */
.irs--shiny .irs-bar {
    background: #3b82f6 !important;
    border-top-color: #3b82f6 !important;
    border-bottom-color: #3b82f6 !important;
}

.irs--shiny .irs-line {
    background: #64748b !important;
    border-color: #64748b !important;
    height: 3px !important;
    border-radius: 3px !important;
}

body.dark-theme .irs--shiny .irs-line {
    background: #94a3b8 !important;
    border-color: #94a3b8 !important;
}

/* slider handle */
.irs--shiny .irs-handle {
    background: #fff !important;
    border-color: #3b82f6 !important;
}
body.dark-theme .irs--shiny .irs-handle {
    background: #cbd5e1 !important;
    border-color: #3b82f6 !important;
}

/* min/max and current value labels */
.irs--shiny .irs-min,
.irs--shiny .irs-max,
.irs--shiny .irs-single,
.irs--shiny .irs-from,
.irs--shiny .irs-to {
    background: var(--step-num-bg) !important;
    color: var(--body-text) !important;
}

body.dark-theme .irs--shiny .irs-line {
    background: #94a3b8 !important;
    border-color: #94a3b8 !important;
}
body.dark-theme .irs--shiny .irs-bar {
    background: #3b82f6 !important;
}


.bslib-value-box { border-radius:10px !important; border:1px solid #e2e8f0 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#f1f5f9; }
::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:3px; }
"""


_HEADER = f"""
<div id="toast-msg"></div>

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
        <a id="hn-5" onclick="goMainTab('tab_pdf',5)">PDF Extractor</a>
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
                <span id="gh-stars">–</span>
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
    for (var i = 0; i < 6; i++) {{
    var el = document.getElementById('hn-' + i);
    if (el) el.classList.toggle('active', i === idx);
}}
    document.body.classList.toggle('on-home', tabValue === 'tab_home');
    document.body.classList.toggle('on-nosidebar', tabValue === 'tab_bg' || tabValue === 'tab_batch' || tabValue === 'tab_pdf');
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

fetch('https://api.github.com/repos/mariameraz/traitly')
    .then(function(r) {{ return r.json(); }})
    .then(function(d) {{
        var el = document.getElementById('gh-stars');
        if (el && d.stargazers_count !== undefined) {{
            el.textContent = d.stargazers_count;
        }}
    }})
    .catch(function() {{}});

function showToast(msg) {{
    var t = document.getElementById('toast-msg');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(function() {{ t.classList.remove('show'); }}, 2000);
}}

Shiny.addCustomMessageHandler('get_current_mode', function() {{
    var internalBtn = document.getElementById('hn-1');
    var externalBtn = document.getElementById('hn-2');
    var mode = 'internal';
    if (externalBtn && externalBtn.classList.contains('active')) {{
        mode = 'external';
    }} else if (internalBtn && internalBtn.classList.contains('active')) {{
        mode = 'internal';
    }}
    Shiny.setInputValue('js_mode', mode, {{priority: 'event'}});
}});

</script>
"""

####################
# Helper functions #
####################


def arr_to_b64(arr):
    """
    Convert a numpy array to a base64-encoded PNG string.
    """
    if arr is None:
        return ""
    success, encoded = cv2.imencode(".png", arr.astype(np.uint8))
    if not success:
        return ""
    return base64.b64encode(encoded.tobytes()).decode()


def img_tag(arr, style="width:100%;border-radius:8px;margin-top:.5rem"):
    """
    Return an HTML <img> tag with the array embedded as base64 PNG
    """
    b = arr_to_b64(arr)
    return f'<img src="data:image/png;base64,{b}" style="{style}">' if b else ""


def _df_to_datatable(
    df: "pd.DataFrame", table_id: str, page_length: int = 5, cols_per_page: int = 7
) -> str:
    """
    Convert pd.DataFrame to an interactive HTML table
    """

    cols = df.columns.tolist()
    n_cols = len(cols)
    n_col_pages = -(-n_cols // cols_per_page)

    fruit_id_idx = next(
        (i for i, c in enumerate(cols) if "fruit_id" in c.lower()), None
    )
    search_placeholder = (
        "Search by fruit_id..." if fruit_id_idx is not None else "Search..."
    )

    rows_data = []
    for _, row in df.iterrows():
        rows_data.append(["" if pd.isna(v) else str(v) for v in row])

    thead = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead>"
    tbody = "<tbody></tbody>"
    btn_style = (
        "display:inline-block;padding:.35rem .9rem;border-radius:6px;border:none;"
        "background:linear-gradient(135deg,#3b82f6,#1d4ed8);color:#fff;"
        "font-size:1.5rem;font-weight:500;cursor:pointer;"
        "box-shadow:0 2px 6px rgba(59,130,246,.3);margin:0 .25rem;"
        "text-decoration:none;transition:opacity .15s;"
    )

    # results table
    footer = (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-top:.5rem;font-size:1.4rem;flex-wrap:wrap;gap:.4rem;">'
        # left, entries selector + info
        f'<div style="display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;">'
        f'<label style="color:#475569;">Show '
        f'<select id="{table_id}_len" style="border:1px solid #e2e8f0;border-radius:5px;'
        f'padding:.15rem .3rem;font-size:1.3rem;">'
        f'<option value="5">5</option>'
        f'<option value="10" selected>10</option>'
        f'<option value="25">25</option>'
        f'<option value="50">50</option>'
        f'<option value="100">100</option>'
        f'<option value="-1">All</option>'
        f"</select> rows</label>"
        f'<span id="{table_id}_info" style="color:#64748b;"></span>'
        f"</div>"
        # right, previous / Next col buttons
        f"<div>"
        f'<button id="{table_id}_prev" style="{btn_style}opacity:.45">Previous</button>'
        f'<button id="{table_id}_next" style="{btn_style}">Next</button>'
        f"</div>"
        f"</div>"
    )

    init = f"""
    <script>
    (function() {{
        var _allRows = {_json.dumps(rows_data)};
        var _filtered = _allRows.slice();
        var _colPage = 0;
        var _rowPage = 0;
        var _pageLen = {page_length};
        var _colsPerPage = {cols_per_page};
        var _totalCols  = {n_cols};
        var _totalColPages = {n_col_pages};
        var _fruitIdIdx = {fruit_id_idx if fruit_id_idx is not None else "null"};
        var _sortCol = -1;
        var _sortAsc = true;

        var _tbody = document.querySelector('#{table_id} tbody');
        var _lenSel = document.getElementById('{table_id}_len');
        var _info = document.getElementById('{table_id}_info');
        var _prev = document.getElementById('{table_id}_prev');
        var _next = document.getElementById('{table_id}_next');

        function _renderRows() {{
            var start = _rowPage * (_pageLen === -1 ? _filtered.length : _pageLen);
            var end = _pageLen === -1 ? _filtered.length : Math.min(start + _pageLen, _filtered.length);
            var html = '';
            for (var i = start; i < end; i++) {{
                html += '<tr>';
                for (var j = 0; j < _totalCols; j++) {{
                    html += '<td>' + (_allRows.indexOf(_filtered[i]), _filtered[i][j]) + '</td>';
                }}
                html += '</tr>';
            }}
            _tbody.innerHTML = html;

            // col visibility
            var cs = _colPage * _colsPerPage;
            var ce = Math.min(cs + _colsPerPage, _totalCols);
            var tbl = document.getElementById('{table_id}');
            var headers = tbl.querySelectorAll('th');
            var cells = tbl.querySelectorAll('td');
            for (var h = 0; h < headers.length; h++) {{
                headers[h].style.display = (h >= cs && h < ce) ? '' : 'none';
            }}
            var rows = _tbody.querySelectorAll('tr');
            for (var r = 0; r < rows.length; r++) {{
                var tds = rows[r].querySelectorAll('td');
                for (var c = 0; c < tds.length; c++) {{
                    tds[c].style.display = (c >= cs && c < ce) ? '' : 'none';
                }}
            }}

            // info
            var rowTotal = _filtered.length;
            var rowStart = rowTotal === 0 ? 0 : start + 1;
            var rowEnd2  = Math.min(end, rowTotal);
            _info.innerHTML =
                'Cols <b>' + (cs+1) + '–' + ce + '</b> of ' + _totalCols +
                ' &nbsp;|&nbsp; Rows ' + rowStart + '–' + rowEnd2 + ' of ' + rowTotal;

            // button opacity
            _prev.style.opacity = _colPage === 0 ? '0.45' : '1';
            _next.style.opacity = _colPage >= _totalColPages - 1 ? '0.45' : '1';
        }}

        function _applySearch(val) {{
            val = val.toLowerCase();
            if (val === '' || _fruitIdIdx === null) {{
                _filtered = _allRows.slice();
            }} else {{
                _filtered = _allRows.filter(function(r) {{
                    return r[_fruitIdIdx].toLowerCase().indexOf(val) !== -1;
                }});
            }}
            _rowPage = 0;
            _renderRows();
        }}


        _lenSel.onchange = function() {{
            _pageLen = parseInt(this.value);
            _rowPage = 0;
            _renderRows();
        }};

        _prev.onclick = function() {{
            if (_colPage > 0) {{ _colPage--; _renderRows(); }}
        }};
        _next.onclick = function() {{
            if (_colPage < _totalColPages - 1) {{ _colPage++; _renderRows(); }}
        }};

        var _ths = document.querySelectorAll('#{table_id} thead th');
        for (var hi = 0; hi < _ths.length; hi++) {{
            (function(idx) {{
                _ths[idx].style.cursor = 'pointer';
                _ths[idx].onclick = function() {{
                    if (_sortCol === idx) {{ _sortAsc = !_sortAsc; }}
                    else {{ _sortCol = idx; _sortAsc = true; }}
                    _filtered.sort(function(a, b) {{
                        var av = a[idx], bv = b[idx];
                        var an = parseFloat(av), bn = parseFloat(bv);
                        if (!isNaN(an) && !isNaN(bn)) {{ return _sortAsc ? an-bn : bn-an; }}
                        return _sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
                    }});
                    _rowPage = 0;
                    _renderRows();
                }};
            }})(hi);
        }}

        function _initDT() {{
            if (typeof $ === 'undefined' || !$.fn || !$.fn.DataTable) {{
                setTimeout(_initDT, 80); return;
            }}
            if ($.fn.DataTable.isDataTable('#{table_id}')) {{
                $('#{table_id}').DataTable().destroy();
            }}
            $('#{table_id}').DataTable({{
                paging: false,
                info: false,
                ordering: false,
                lengthChange: false,
                searching: true,
                language: {{
                    search: "Search:",
                    searchPlaceholder: "{search_placeholder}"
                }}
            }});
            $('#{table_id}_wrapper .dataTables_filter input')
                .off()
                .on('input', function() {{ _applySearch(this.value); }});

            _renderRows();
        }}
        _initDT();
    }})();
    </script>
    """

    return (
        f'<div style="overflow-x:auto;font-size:1.5rem;margin-top:.6rem">'
        f'<table id="{table_id}" class="display" style="width:100%">'
        f"{thead}{tbody}</table></div>" + footer + init
    )


################################
# panels for each step         #
################################


def _panel(val, title, *children):
    return ui.nav_panel(
        title,
        ui.div(
            ui.HTML(f'<p class="panel-title">{title}</p>'),
            *children,
        ),
        value=val,
    )


# step 1 - setup measurement
step_setup = _panel(
    "step_setup",
    "Setup Image Measurements",
    ui.layout_columns(
        ui.div(
            ui.output_ui("upload_input_ui"),
            ui.hr(),
            ui.input_checkbox("detect_label", "Detect label text", False),
            ui.input_checkbox("skip_qr", "Skip QR detection", False),
            ui.input_checkbox("detect_color_checker", "Detect color checker", False),
            ui.input_slider(
                "confidence", "Detection confidence", 0.0, 1.0, 0.6, step=0.01
            ),
            ui.hr(),
            ui.input_checkbox("use_dimensions", "Use physical dimensions", False),
            ui.output_ui("dimensions_ui"),
            ui.input_numeric(
                "diameter_cm", "Reference diameter (cm)", 2.5, min=0.0, step=0.01
            ),
            ui.hr(),
            ui.input_checkbox("use_crop", "Crop image", False),
            ui.output_ui("crop_ui"),
            ui.hr(),
            ui.input_action_button(
                "run_step1",
                "▶  Run Setup",
                class_="btn btn-primary",
                style="font-size: 2rem; padding: .8rem 1.5rem;",
            ),
        ),
        ui.div(
            ui.div(
                ui.output_ui("step1_preview"),
                ui.div(
                    ui.output_ui("step1_results"),
                    style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%);  "
                    "width:90%; z-index:10;",
                ),
                style="position:relative;",
            ),
        ),
        col_widths=[3, 9],
    ),
)

# step 2 - create a mask
step_mask = _panel(
    "step_mask",
    "Generate Fruit Mask",
    ui.layout_columns(
        ui.div(
            ui.output_ui("mask_bg_ui"),
            ui.input_checkbox("remove_roi", "Remove label/reference regions", True),
            ui.input_checkbox("use_manual_hsv", "Apply manual color threshold", False),
            ui.p(
                "Find HSV range in Background Helper",
                style="font-size:1.4rem; color:#94a3b8; margin-top:-.5rem;",
            ),
            ui.output_ui("hsv_ui"),
            ui.hr(),
            ui.HTML("""
                    <details style="margin-bottom:.8rem">
                        <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                                        padding:.4rem .2rem;color:#475569;user-select:none">
                            Advanced parameters
                        </summary>
                    <div id="advanced-mask-params" style="padding:.6rem 0 0 .4rem">
            """),
            ui.hr(),
            ui.output_ui("mask_stamp_ui"),
            ui.input_checkbox("apply_convex_hull", "Apply convex hull", False),
            ui.input_checkbox("fill_holes", "Fill holes", False),
            ui.hr(),
            ui.input_slider("n_iteration", "Morphology iterations", 1, 5, 1),
            ui.input_slider("roi_expansion", "ROI expansion (px)", -80, 80, 10),
            ui.input_slider("kernel_blur", "Blur kernel", 1, 17, 1, step=2),
            ui.input_slider("kernel_open", "Opening kernel", 1, 17, 1, step=2),
            ui.input_slider("kernel_close", "Closing kernel", 1, 17, 1, step=2),
            ui.input_numeric("erosion_px", "Erosion (px)", 0, min=0, step=1),
            ui.HTML("</div></details>"),
            ui.hr(),
            ui.input_action_button(
                "run_step2",
                "▶  Generate Mask",
                class_="btn btn-primary",
                style="font-size: 2rem; padding: .8rem 1.5rem;",
            ),
        ),
        ui.div(ui.output_ui("step2_results")),
        col_widths=[3, 9],
    ),
)

# step 5 - open an interactive mask editor (step 3 in external analysis)
step_edit_mask = _panel(
    "step_edit_mask",
    "Interactive Mask Editor <span class=home-title-sub> – Optional</span>",
    ui.layout_columns(
        ui.div(
            ui.p(
                "Draw polygons on the mask to add or remove regions.",
                style="font-size:1.6rem;color:#64748b;margin-bottom:.8rem;",
            ),
            ui.HTML(
                """<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.8rem">"""
            ),
            ui.input_action_button(
                "mask_edit_mode_add",
                "＋ ADD (white)",
                class_="btn btn-primary",
                style="font-size:1.6rem;padding:.4rem .8rem;flex:1",
            ),
            ui.input_action_button(
                "mask_edit_mode_remove",
                "－ REMOVE (black)",
                class_="btn btn-outline-secondary",
                style="font-size:1.6rem;padding:.4rem .8rem;flex:1",
            ),
            ui.HTML("</div>"),
            ui.output_ui("mask_edit_mode_indicator"),
            # ui.hr(),
            # ui.input_slider("mask_overlay_alpha", "Overlay transparency", 0.0, 1.0, 0.4, step=0.05),
            ui.hr(),
            ui.input_action_button(
                "mask_edit_apply",
                ui.HTML('<i class="fa-solid fa-check"></i> Apply polygon'),
                class_="btn btn-primary",
                style="font-size:1.6rem;padding:.4rem .8rem;width:100%;margin-bottom:.4rem",
            ),
            ui.input_action_button(
                "mask_edit_undo",
                ui.HTML('<i class="fa-solid fa-clock-rotate-left"></i> Undo last'),
                class_="btn btn-outline-secondary",
                style="font-size:1.6rem;padding:.4rem .8rem;width:100%;margin-bottom:.4rem",
            ),
            ui.input_action_button(
                "mask_edit_clear",
                ui.HTML('<i class="fa-solid fa-xmark"></i> Clear points'),
                class_="btn btn-outline-secondary",
                style="font-size:1.6rem;padding:.4rem .8rem;width:100%;margin-bottom:.8rem",
            ),
            ui.hr(),
            ui.input_action_button(
                "mask_edit_save",
                ui.HTML('<i class="fa-solid fa-floppy-disk"></i> Save & continue'),
                class_="btn btn-primary",
                style="font-size:1.8rem;padding:.6rem 1rem;width:100%;margin-bottom:.4rem",
            ),
            ui.input_action_button(
                "mask_edit_discard",
                ui.HTML('<i class="fa-solid fa-trash-arrow-up"></i> Discard changes'),
                class_="btn btn-outline-secondary",
                style="font-size:1.6rem;padding:.4rem .8rem;width:100%",
            ),
        ),
        ui.div(
            ui.output_ui("mask_editor_canvas"),
        ),
        col_widths=[3, 9],
    ),
)

# step 3 enhance locule contrast (internal analysis only)
step_contrast = _panel(
    "step_contrast",
    "Enhance Locule Contrast <span class=home-title-sub> – Optional</span>",
    ui.layout_columns(
        ui.div(
            ui.input_select(
                "contrast_method",
                "Contrast method",
                choices=["none", "gamma", "sigmoid", "exp"],
            ),
            ui.output_ui("contrast_params_ui"),
            ui.hr(),
            ui.input_checkbox("compare_method", "Compare all methods", False),
            ui.hr(),
            ui.HTML("""
            <details style="margin-bottom:.8rem">
                <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                                padding:.4rem .2rem;color:#475569;user-select:none">
                    Advanced parameters
                </summary>
            <div style="padding:.6rem 0 0 .4rem">
            """),
            ui.input_slider("kernel_blur3", "Blur kernel", 1, 15, 1, step=2),
            ui.input_numeric(
                "clip_limit", "CLAHE clip limit (0 = off)", 0, min=0, step=1
            ),
            ui.input_numeric(
                "tile_grid_size", "CLAHE tile grid size", 12, min=1, step=1
            ),
            ui.HTML("</div></details>"),
            ui.hr(),
            ui.input_action_button(
                "run_step3",
                "▶  Enhance Contrast",
                class_="btn btn-primary",
                style="font-size: 2rem; padding: .8rem 1.5rem;",
            ),
        ),
        ui.div(ui.output_ui("step3_results")),
        col_widths=[3, 9],
    ),
)

# step 4 - create locule mask (internal analysis only)
step_locule = _panel(
    "step_locule",
    "Generate Locule Mask <span class=home-title-sub> – Optional</span>",
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
            ui.input_numeric(
                "min_fruit_area_lm", "Min fruit area (px)", 5000, min=100, step=100
            ),
            ui.input_numeric(
                "min_locule_area_lm", "Min locule area (px)", 0, min=0, step=10
            ),
            ui.hr(),
            ui.HTML("""<details style="margin-bottom:.8rem">
                <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                        padding:.4rem .2rem;color:#475569;user-select:none">
                    Advanced parameters
                </summary>
                <div style="padding:.6rem 0 0 .4rem">
                """),
            ui.input_checkbox("invert_locule", "Invert locule mask", False),
            ui.hr(),
            ui.input_slider("kernel_blur4", "Blur kernel", 1, 17, 1, step=2),
            ui.input_slider("kernel_open4", "Opening kernel", 1, 17, 1, step=2),
            ui.input_slider("kernel_close4", "Closing kernel", 1, 17, 1, step=2),
            ui.input_numeric("erosion_px4", "Erosion (px)", 10, min=0, step=1),
            ui.HTML("</div></details>"),
            ui.hr(),
            ui.input_action_button(
                "run_step4",
                "▶  Generate Locule Mask",
                class_="btn btn-primary",
                style="font-size: 2rem; padding: .8rem 1.5rem;",
            ),
        ),
        ui.div(ui.output_ui("step4_results")),
        col_widths=[3, 9],
    ),
)

# step 4/6 detect fruits from binary mask
step_detect = _panel(
    "step_detect",
    "Detect Fruits",
    ui.layout_columns(
        ui.div(
            ui.input_slider(
                "min_fruit_circularity", "Min circularity", 0.0, 1.0, 0.5, step=0.05
            ),
            ui.input_numeric(
                "min_fruit_area_det", "Min fruit area (px)", 500, min=1, step=100
            ),
            ui.input_numeric(
                "max_fruit_area_det", "Max fruit area (px)", 0, min=0, step=100
            ),
            ui.p(
                "Set to 0 for no upper limit.",
                style="font-size:1.4rem; color:#94a3b8; margin-top:-.5rem;",
            ),
            ui.output_ui("detect_locule_params_ui"),
            ui.hr(),
            ui.HTML("""
                <details style="margin-bottom:.8rem">
                    <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                                    padding:.4rem .2rem;color:#475569;user-select:none">
                        Advanced parameters
                    </summary>
                <div style="padding:.6rem 0 0 .4rem">
            """),
            ui.output_ui("detect_dilation_ui"),
            ui.input_numeric(
                "rescale_factor_det", "Rescale factor", 0, min=0, step=0.1
            ),
            ui.p(
                "Set to 0 for no upper limit.",
                style="font-size:1.4rem;color:#94a3b8;margin-top:-.5rem;",
            ),
            ui.hr(),
            ui.input_slider(
                "contour_thickness_det", "Fruit contour thickness", 1, 10, 2
            ),
            ui.input_text(
                "contour_color_det", "Fruit contour color (R,G,B)", "0,255,0"
            ),
            ui.output_ui("detect_int_styling_ui"),
            ui.HTML("</div></details>"),
            ui.hr(),
            ui.input_action_button(
                "run_detect",
                "▶  Detect Fruits",
                class_="btn btn-primary",
                style="font-size: 2rem; padding: .8rem 1.5rem;",
            ),
        ),
        ui.div(ui.output_ui("detect_results")),
        col_widths=[3, 9],
    ),
)

# step 7/5 calculate fruit morphology
step_morph = _panel(
    "step_morph",
    "Morphological Analysis",
    ui.layout_columns(
        ui.div(
            ui.input_select(
                "contour_mode",
                "Contour mode",
                choices=["raw", "hull", "approx", "ellipse", "circle"],
            ),
            ui.output_ui("epsilon_ui"),
            ui.output_ui("morph_locule_params_ui"),
            ui.input_checkbox("save_params_morph", "Save analysis parameters", False),
            ui.output_ui("morph_advanced_section_ui"),
            ui.hr(),
            ui.HTML("""
                <details style="margin-bottom:.8rem">
                    <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                                    padding:.4rem .2rem;color:#475569;user-select:none">
                        Plot styling
                    </summary>
                <div style="padding:.6rem 0 0 .4rem">
            """),
            ui.input_slider("font_size_morph", "Font size", 0.5, 4.0, 1.5, step=0.1),
            ui.input_numeric(
                "font_thickness_morph", "Font thickness", 2, min=1, step=1
            ),
            ui.input_select(
                "label_position_morph",
                "Label position",
                choices=["top", "bottom", "left", "right"],
            ),
            ui.input_text("font_color_morph", "Font color (R,G,B)", "0,0,0"),
            ui.input_text(
                "label_color_morph", "Label background (R,G,B)", "255,255,255"
            ),
            ui.input_text(
                "pericarp_ext_color_morph", "Ext. pericarp color (R,G,B)", "0,240,0"
            ),
            ui.input_numeric(
                "pericarp_ext_thick_morph", "Ext. pericarp thickness", 2, min=1, step=1
            ),
            ui.output_ui("morph_int_styling_ui"),
            ui.HTML("</div></details>"),
            ui.hr(),
            ui.input_action_button(
                "run_morph",
                "▶  Analyze Morphology",
                class_="btn btn-primary",
                style="font-size: 2rem; padding: .8rem 1.5rem;",
            ),
        ),
        ui.div(ui.output_ui("morph_results")),
        col_widths=[3, 9],
    ),
)

# step 8/6 color extraction
step_color = _panel(
    "step_color",
    "Color Analysis",
    ui.layout_columns(
        ui.div(
            ui.input_select("stat", "Statistical measure", choices=["mean", "median"]),
            ui.output_ui("color_tissue_ui"),
            ui.input_select(
                "color_space",
                "Color space",
                choices=["all", "rgb", "lab", "hsv", "gray"],
            ),
            ui.hr(),
            ui.input_checkbox("get_color_histogram", "Get color histogram", False),
            ui.input_numeric(
                "dark_thresh_color", "Dark pixel threshold", 20, min=0, step=1
            ),
            ui.hr(),
            ui.input_checkbox("save_params_color", "Save analysis parameters", False),
            ui.hr(),
            ui.HTML("""
                <details style="margin-bottom:.8rem">
                    <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                                    padding:.4rem .2rem;color:#475569;user-select:none">
                        Plot styling
                    </summary>
                <div style="padding:.6rem 0 0 .4rem">
            """),
            ui.input_slider("font_size_color", "Font size", 0.5, 4.0, 2.0, step=0.1),
            ui.input_numeric(
                "font_thickness_color", "Font thickness", 2, min=1, step=1
            ),
            ui.input_select(
                "label_position_color",
                "Label position",
                choices=["top", "bottom", "left", "right"],
            ),
            ui.input_slider(
                "label_opacity_color", "Label opacity", 0.0, 1.0, 0.7, step=0.05
            ),
            ui.input_text("font_color_color", "Font color (R,G,B)", "0,0,0"),
            ui.input_text(
                "label_color_color", "Label background (R,G,B)", "255,255,255"
            ),
            ui.input_text(
                "pericarp_ext_color_color", "Ext. pericarp color (R,G,B)", "0,255,0"
            ),
            ui.input_numeric(
                "pericarp_ext_thick_color", "Ext. pericarp thickness", 2, min=1, step=1
            ),
            ui.output_ui("color_int_styling_ui"),
            ui.HTML("</div></details>"),
            ui.hr(),
            ui.input_action_button(
                "run_color",
                "▶  Analyze Color",
                class_="btn btn-primary",
                style="font-size: 2rem; padding: .8rem 1.5rem;",
            ),
        ),
        ui.div(ui.output_ui("color_results")),
        col_widths=[3, 9],
    ),
)

# tab 1 - home
tab_home = ui.nav_panel(
    "Home",
    ui.div(
        ui.HTML("""
        <div class="home-content">
            <h1 class="home-title">
                Welcome to Traitly <span class="home-title-sub">Interactive Analyzer</span>
            </h1>

            <div class="home-body">
                <p style="margin-bottom:1.2rem">
                    <strong>Traitly</strong> is a Python library designed to <strong>automate fruit image analysis</strong>,
                        from a single sample to hundreds of fruits in one run. Using standard RGB images, it extracts
                    <strong>color, shape, and size traits</strong> from both internal (cross-sections) and external
                    (surface) fruit images, with no manual measurements required.
                </p>
                <p style="margin-bottom:1.2rem">
                    Traitly is committed to <strong>open and reproducible science</strong>: every analysis automatically
                    generates a session report with all parameters and versions used, ensuring complete traceability of results.
                </p>
            </div>

            <h4 class="home-h4">
                <div class="home-body">
                    <br>
                        <p class="home-info-box">
                        This is the web application to run <strong>Internal</strong> and <strong>External</strong> analyses
                        interactively. For a deeper understanding of parameters, pipeline functions, input image requirements,
                        and expected outputs, visit the full documentation at
                        <a class="home-link" href="https://traitly.readthedocs.io/" target="_blank">traitly.readthedocs.io</a>
                        <br><br>
                        <span style="color: #a70085;"><strong>┈➤ Download example images to get started</strong></span>
                        <a class="home-link" href="https://github.com/mariameraz/traitly/tree/main/tutorials/images" target="_blank"><strong>here</strong></a> ˎˊ˗
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

            <h2 class="home-h2">Citation</h2>
            <div class="home-body">
                <p style="margin-bottom:1.2rem">
                    If you use Traitly in your research, please cite it as:
                </p>

                <p style="margin-bottom:1.2rem; margin-left:2.8rem; font-style: italic;">
                    Torres-Meraz, M. A., Lopez-Moreno, H., & Zalapa, J. (2026). Traitly: A Python Toolkit for High-Throughput Fruit Phenotyping. Zenodo.
                    <a href="https://doi.org/10.5281/zenodo.18738366" target="_blank" rel="noopener noreferrer"
                    style="color: var(--home-link-color, #2d63bc); text-decoration: none; border-bottom: 1px dotted var(--home-link-color, #2d63bc);">
                        https://doi.org/10.5281/zenodo.18738366
                    </a>
                </p>
            </div>

            <br><br>

            <h2 class="home-h2">Acknowledgements</h2>
            <div class="home-body">
                <p style="margin-bottom:1.2rem">
                    We thank the developers of
                    <a href="https://opencv.org/" target="_blank" rel="noopener noreferrer">OpenCV</a>,
                    <a href="https://github.com/ultralytics/ultralytics" target="_blank" rel="noopener noreferrer">Ultralytics</a>,
                    <a href="https://github.com/JaidedAI/EasyOCR" target="_blank" rel="noopener noreferrer">EasyOCR</a>,
                    <a href="https://numpy.org/" target="_blank" rel="noopener noreferrer">NumPy</a>,
                    <a href="https://pandas.pydata.org/" target="_blank" rel="noopener noreferrer">Pandas</a>,
                    <a href="https://matplotlib.org/" target="_blank" rel="noopener noreferrer">Matplotlib</a>, and
                    <a href="https://shiny.posit.co/py/" target="_blank" rel="noopener noreferrer">Shiny</a>, as well as all open-source libraries that
                    made this project possible.
                </p>
            </div>

        <br><br>

        <h2 class="home-h2">Contributors</h2>
            <p style="margin-bottom: 1.5rem;">
                <a href="https://github.com/mariameraz" target="_blank" style="text-decoration: none; margin-right: 1rem; display: inline-flex; align-items: center; gap: 0.5rem;">
                    <img src="https://github.com/mariameraz.png" width="74" height="74" style="border-radius: 80%;">
                </a>
                <a href="https://github.com/hector-LM" target="_blank" style="text-decoration: none; margin-right: 1rem; display: inline-flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem;">
                    <img src="https://github.com/hector-LM.png" width="74" height="74" style="border-radius: 80%;">
                </a>
            </p>

        </div>
        """),
    ),
    value="tab_home",
)

tab_analysis = ui.nav_panel(
    "Analysis",
    ui.navset_hidden(
        step_setup,
        step_mask,
        step_edit_mask,
        step_contrast,
        step_locule,
        step_detect,
        step_morph,
        step_color,
        id="pipeline_step",
        selected="step_setup",
    ),
    value="tab_analysis",
)

# tab 4- Background helper
tab_bg = ui.nav_panel(
    "BG Helper",
    ui.div(
        ui.div(
            ui.HTML('<p class="panel-title">Background Color Helper</p>'),
            ui.p(
                "⋆˙⟡ Upload an image to inspect its HSV pixel distribution, then tune thresholds to isolate the background.",
                style="font-size: 2rem; margin-bottom: 1rem",
            ),
            ui.HTML("<br>"),
            ui.hr(),
            ui.layout_columns(
                # left col, upload + sample size
                ui.div(
                    ui.input_file(
                        "bg_upload",
                        "Upload image",
                        accept=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
                    ),
                    ui.input_numeric(
                        "bg_sample",
                        "Pixel sample size",
                        10000,
                        min=1000,
                        max=100000,
                        step=1000,
                    ),
                    ui.p(
                        "Increase for denser scatter, decrease for speed.",
                        style="font-size:1.4rem;color:#94a3b8;margin-top:-.4rem;",
                    ),
                ),
                # right col, scatterplot
                ui.div(ui.output_ui("bg_scatter_out")),
                col_widths=[4, 8],
            ),
            style="border-radius:12px;padding:1.4rem;margin-bottom:1rem;",
        ),
        # preview panel
        ui.div(
            ui.HTML(
                '<p style="font-size:2rem;font-weight:600;color:var(--panel-title-color);'
                'margin-bottom:.8rem;">HSV Thresholds — live preview</p>'
            ),
            ui.hr(),
            ui.layout_columns(
                ui.input_select(
                    "bg_preset",
                    "Preset values",
                    choices=["custom", "blue", "white", "black"],
                ),
                ui.div(),
                col_widths=[4, 8],
            ),
            ui.HTML("<br>"),
            ui.layout_columns(
                ui.div(
                    ui.HTML(
                        '<div style="font-size:1.7rem;font-weight:600;margin-bottom:.5rem;'
                        'color:#3b82f6;">Lower bound</div>'
                    ),
                    ui.input_slider("bg_h_lo", "H min", 0, 180, 0),
                    ui.input_slider("bg_s_lo", "S min", 0, 255, 0),
                    ui.input_slider("bg_v_lo", "V min", 0, 255, 0),
                ),
                ui.div(
                    ui.HTML(
                        '<div style="font-size:1.7rem;font-weight:600;margin-bottom:.5rem;'
                        'color:#1d4ed8;">Upper bound</div>'
                    ),
                    ui.input_slider("bg_h_hi", "H max", 0, 180, 180),
                    ui.input_slider("bg_s_hi", "S max", 0, 255, 255),
                    ui.input_slider("bg_v_hi", "V max", 0, 255, 255),
                ),
                col_widths=[6, 6],
            ),
            # plot imgs
            ui.output_ui("bg_preview"),
            style="border-radius:12px;padding:1.4rem;margin-bottom:1rem;",
        ),
        # hsv threshold message
        ui.div(
            ui.HTML(
                '<p style="font-size:2rem;font-weight:600;color:var(--panel-title-color);'
                'margin-bottom:.5rem;">Copy these values</p>'
            ),
            ui.output_ui("bg_final_code"),
            style="background:var(--sidebar-bg);border-radius:12px;border:1px solid var(--sidebar-border);"
            "padding:1.4rem;box-shadow:0 1px 3px rgba(0,0,0,.06);",
        ),
    ),
    value="tab_bg",
)

# tab 5 - Batch analysis
tab_batch = ui.nav_panel(
    "Batch",
    ui.layout_columns(
        ui.div(
            ui.HTML('<p class="panel-title">Batch Analysis</p>'),
            ui.input_file(
                "batch_files",
                "Select images",
                accept=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
                multiple=True,
            ),
            # ui.output_ui("batch_file_info"),
            ui.hr(),
            ui.HTML('<div class="sb-label">Analysis Mode</div>'),
            ui.input_radio_buttons(
                "batch_mode",
                None,
                choices={"external": "External", "internal": "Internal"},
                selected="external",
                inline=True,
            ),
            ui.hr(),
            ui.HTML('<div class="sb-label">Pipeline</div>'),
            ui.input_checkbox("run_morphology", "Analyze morphology", True),
            ui.input_checkbox("run_color_batch", "Analyze color", True),
            ui.hr(),
            ui.input_file(
                "batch_json",
                "Load parameters (.json)",
                accept=[".json"],
            ),
            ui.p(
                "If provided, all analysis parameters are loaded from this file. ",
                ui.HTML("<br>"),
                ui.span("➤ Don't have one yet? ", style="color:#94a3b8;"),
                ui.download_button(
                    "dl_base_json",
                    "Download the base json here",
                    style="color: #2d63bc;text-decoration:underline;cursor:pointer;"
                    "background:none;border:none;padding:0;font-size:1.4rem;",
                ),
                style="font-size:1.4rem;color:#94a3b8;margin-top:-.4rem;",
            ),
            ui.hr(),
            ui.input_numeric("batch_num_cores", "Number of cores", 1, min=1, step=1),
            ui.p(
                "Use more cores to speed up processing on large datasets.",
                style="font-size:1.4rem;color:#94a3b8;margin-top:-.4rem;",
            ),
            ui.hr(),
            ui.input_action_button(
                "run_batch",
                "▶  Run Batch Analysis",
                class_="btn btn-primary",
                style="font-size:2rem;padding:.8rem 1.5rem;",
            ),
        ),
        ui.div(ui.output_ui("batch_results")),
        col_widths=[3, 9],
    ),
    value="tab_batch",
)

# tab 6 - extract images from pdf
tab_pdf = ui.nav_panel(
    "PDF Extractor",
    ui.layout_columns(
        ui.div(
            ui.HTML('<p class="panel-title">PDF Extractor</p>'),
            ui.input_file("pdf_file", "Upload PDF", accept=[".pdf"], multiple=True),
            # ui.output_ui("pdf_file_info"),
            ui.hr(),
            ui.HTML('<div class="sb-label">Options</div>'),
            ui.input_numeric(
                "pdf_dpi", "Resolution (DPI)", 150, min=72, max=600, step=50
            ),
            ui.p(
                "Higher DPI = better quality, larger files.",
                style="font-size:1.4rem;color:#94a3b8;margin-top:-.4rem;",
            ),
            ui.hr(),
            ui.input_select(
                "pdf_format",
                "Output format",
                choices=["jpg", "png", "tiff"],
                selected="jpg",
            ),
            ui.hr(),
            ui.input_checkbox("pdf_qr_label", "Rename pages using QR code", False),
            ui.p(
                "If checked, pages with a detected QR code will be renamed using its text.",
                style="font-size:1.4rem;color:#94a3b8;margin-top:-.4rem;",
            ),
            ui.hr(),
            ui.input_action_button(
                "run_pdf",
                "▶  Extract Images",
                class_="btn btn-primary",
                style="font-size:2rem;padding:.8rem 1.5rem;",
            ),
        ),
        ui.div(ui.output_ui("pdf_results")),
        col_widths=[3, 9],
    ),
    value="tab_pdf",
)

# side bar config
sidebar_ui = ui.sidebar(
    ui.output_ui("sidebar_content"),
    # width="220px", ## HF
    width="250px",
    open="always",
)

app_ui = ui.page_sidebar(
    sidebar_ui,
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css",
        ),
        ui.tags.link(
            rel="stylesheet",
            href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css",
        ),
        ui.tags.script(
            src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"
        ),
    ),
    ui.tags.style(_CSS),
    ui.HTML(_HEADER),
    ui.navset_hidden(
        tab_home,
        tab_analysis,
        tab_bg,
        tab_batch,
        tab_pdf,
        id="main_tab",
        selected="tab_home",
    ),
    title="Traitly",
    fillable=False,
)


# servers
def server(input: Inputs, output: Outputs, session: Session):

    r_analyzer = reactive.value(None)
    r_completed = reactive.value([])
    r_mode = reactive.value(None)
    r_cur_step = reactive.value("step_setup")
    r_bg_analyzer = reactive.value(None)
    # r_output_folder = reactive.value("")
    r_batch_zip = reactive.value(None)
    r_pdf_zip = reactive.value(None)
    r_morph_zip = reactive.value(None)
    r_morph_base = reactive.value("morphology")
    r_color_zip = reactive.value(None)
    r_color_base = reactive.value("color")
    r_axis_b64 = reactive.value("")
    r_img_shape = reactive.value((100, 100))
    r_step1_result = reactive.value(ui.div())
    r_setup_done = reactive.value(0)
    r_img_ready = reactive.value(False)
    r_upload_key = reactive.value(0)
    r_original_img_name = reactive.value("")
    r_mode_version = reactive.value(0)
    r_step2_result = reactive.value(ui.div())
    r_step3_result = reactive.value(ui.div())
    r_step4_result = reactive.value(ui.div())
    r_detect_result = reactive.value(ui.div())
    r_morph_result = reactive.value(ui.div())
    r_color_result = reactive.value(ui.div())
    # For the interactive editor
    # r_mask_editor_active = reactive.value(False)
    r_mask_edited = reactive.value(None)
    r_mask_points = reactive.value([])
    r_mask_mode = reactive.value("white")
    r_mask_history = reactive.value([])

    def _steps(mode):
        if mode == "internal":
            return [
                ("step_setup", "", "Setup"),
                ("step_mask", "", "Fruit Mask"),
                ("step_contrast", "", "Enhance Contrast"),
                ("step_locule", "", "Locule Mask"),
                ("step_edit_mask", "", "Edit Mask"),
                ("step_detect", "", "Detect Fruits"),
                ("step_morph", "", "Morphology"),
                ("step_color", "", "Color"),
            ]
        return [
            ("step_setup", "", "Setup"),
            ("step_mask", "", "Fruit Mask"),
            ("step_edit_mask", "", "Edit Mask"),
            ("step_detect", "", "Detect Fruits"),
            ("step_morph", "", "Morphology"),
            ("step_color", "", "Color"),
        ]

    def mark_done(idx):
        d = list(r_completed.get())
        if idx not in d:
            d.append(idx)
        r_completed.set(d)

    @render.download(filename="parameters.json")
    async def dl_base_json():
        json_path = os.path.join(os.path.dirname(__file__), "www", "parameters.json")
        with open(json_path, "rb") as f:
            yield f.read()

    @reactive.effect
    @reactive.event(input.js_main_tab)
    def _on_main_tab():
        tab = input.js_main_tab()
        ui.update_navs("main_tab", selected=tab, session=session)

        if tab == "tab_analysis" and r_mode.get() is None:
            session.send_custom_message("get_current_mode", {})

    @reactive.effect
    @reactive.event(input.js_mode)
    def _on_mode():
        new_mode = input.js_mode()
        if new_mode in ("internal", "external"):
            r_mode.set(new_mode)
            r_completed.set([])
            r_cur_step.set("step_setup")
            r_analyzer.set(None)
            r_img_ready.set(False)
            r_step1_result.set(ui.div())
            r_upload_key.set(r_upload_key.get() + 1)
            r_original_img_name.set("")
            r_mode_version.set(r_mode_version.get() + 1)
            r_step2_result.set(ui.div())
            r_step3_result.set(ui.div())
            r_step4_result.set(ui.div())
            r_detect_result.set(ui.div())
            r_morph_result.set(ui.div())
            r_color_result.set(ui.div())
            r_mask_edited.set(None)
            r_mask_points.set([])
            r_mask_history.set([])
            r_mask_mode.set("white")
            ui.update_navs("pipeline_step", selected="step_setup", session=session)

    @reactive.effect
    @reactive.event(input.js_step_click)
    def _on_step():
        sid = input.js_step_click()
        ui.update_navs("pipeline_step", selected=sid, session=session)
        r_cur_step.set(sid)

    @render.ui
    def sidebar_content():
        mode = r_mode.get()

        if mode is None:
            return ui.div(
                ui.HTML('<div class="sb-label">Pipeline Steps</div>'),
                ui.p("Click Internal or External to start.",
                    style="font-size:1.4rem; color:#64748b; padding: 1rem;")
            )
        done = r_completed.get()
        cur = r_cur_step.get()

        steps = _steps(mode)
        items = []

        if mode == "internal":
            mode_badge = ('<div class="sb-mode-badge internal" style="margin: 0.5rem 0 1rem 0; padding: 0.3rem 0.7rem; background: #ffeffd; '
                        'color: #c91f91; border: 1px solid #febfe7; border-radius: 20px; font-size: 1.4rem; '
                        'font-weight: 600; text-align: center;">Internal Analysis</div>')
        else:
            mode_badge = ('<div class="sb-mode-badge external" style="margin: 0.5rem 0 1rem 0; padding: 0.3rem 0.7rem; '
                            'background: #f0f7fd; color: #156b80; border: 1px solid #bbc9f7; border-radius: 20px; font-size: 1.4rem; '
                            'font-weight: 600; text-align: center;">External Analysis</div>')


        for i, (sid, icon, label) in enumerate(steps):
            is_done = i in done
            is_active = sid == cur
            cls = "step-link"
            if is_active:
                cls += " active"
            if is_done:
                cls += " done"
            chk = '<span class="step-check">✓</span>' if is_done else ""
            items.append(
                ui.HTML(
                    f'<button class="{cls}" '
                    f"onclick=\"goStep('{sid}')\">"
                    f'<span class="step-num">{i + 1}</span>'
                    f'<span class="step-label">{icon} {label}</span>'
                    f"{chk}</button>"
                )
            )

        return ui.div(
            ui.HTML('<div class="sb-label">Pipeline Steps</div>'),
            ui.HTML(mode_badge),
            *items,
            ui.hr(),
            ui.input_action_button("reset_btn", "↻ Reset", class_="btn btn-reset"),
        )

    # reset
    @reactive.effect
    @reactive.event(input.reset_btn)
    def _reset():
        r_analyzer.set(None)
        r_completed.set([])
        r_cur_step.set("step_setup")
        r_original_img_name.set("")
        r_mode_version.set(r_mode_version.get() + 1)
        r_step2_result.set(ui.div())
        r_step3_result.set(ui.div())
        r_step4_result.set(ui.div())
        r_detect_result.set(ui.div())
        r_morph_result.set(ui.div())
        r_color_result.set(ui.div())
        r_img_ready.set(False)
        r_upload_key.set(r_upload_key.get() + 1)
        r_step1_result.set(ui.div())
        ui.update_navs("pipeline_step", selected="step_setup", session=session)

    @render.ui
    def upload_input_ui():
        r_upload_key.get()
        return ui.input_file(
            "upload_img",
            "Upload a fruit image",
            accept=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
        )

    # step 1
    @render.ui
    def dimensions_ui():
        if input.use_dimensions():
            return ui.div(
                ui.input_numeric(
                    "width_cm", "Width (cm)", 21.59, min=0.0, step=0.01
                ),  # default letter size
                ui.input_numeric("length_cm", "Length (cm)", 27.94, min=0.0, step=0.01),
            )
        return ui.div()

    @render.ui
    def crop_ui():
        img_h, img_w = r_img_shape.get()
        if input.use_crop():
            return ui.div(
                ui.layout_columns(
                    ui.input_numeric("crop_x", "x (left px)", 0, min=0, step=1),
                    ui.input_numeric("crop_y", "y (top px)", 0, min=0, step=1),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.input_numeric("crop_w", "w (width px)", img_w, min=1, step=1),
                    ui.input_numeric("crop_h", "h (height px)", img_h, min=1, step=1),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.input_action_button(
                        "apply_crop",
                        "Apply Crop",
                        class_="btn btn-secondary",
                        style="font-size:1.6rem; margin-top:.5rem;",
                    ),
                    ui.input_action_button(
                        "reset_crop",
                        "↻ Reset Size",
                        class_="btn btn-outline-secondary",
                        style="font-size:1.6rem; margin-top:.5rem;",
                    ),
                    col_widths=[6, 6],
                ),
            )
        return ui.div()

    def _copy_with_original_name(file_info: dict) -> str:
        original_name = file_info["name"]
        src_path = file_info["datapath"]
        tmp_dir = tempfile.mkdtemp()
        dest_path = os.path.join(tmp_dir, original_name)
        shutil.copy2(src_path, dest_path)
        return dest_path

    def _do_load_image():
        if not traitly_available:
            r_step1_result.set(
                ui.p(
                    '<i class="fa-solid fa-triangle-exclamation"></i> traitly is not installed. Run: `pip install traitly` first.',
                    class_="text-danger",
                    style="font-size:2rem;",
                )
            )
            return

        f = input.upload_img()
        if not f:
            return
        # use original image name
        original_name = f[0]["name"]
        r_original_img_name.set(os.path.splitext(original_name)[0])
        path = _copy_with_original_name(f[0])
        mode = r_mode.get()
        az = (
            FruitInternalAnalyzer(path)
            if mode == "internal"
            else FruitExternalAnalyzer(path)
        )
        az.load_image(plot=False)
        r_img_shape.set(az.img_shape)
        if input.use_crop():
            az.load_image(
                plot=False,
                x=input.crop_x(),
                y=input.crop_y(),
                w=input.crop_w(),
                h=input.crop_h(),
            )
        r_analyzer.set(az)
        r_completed.set([])
        r_step1_result.set(ui.div())
        r_img_ready.set(True)

    @reactive.effect
    @reactive.event(input.upload_img)
    def _load_image():
        _do_load_image()

    @reactive.effect
    @reactive.event(input.apply_crop)
    def _on_apply_crop():
        r_axis_b64.set("")
        _do_load_image()

    @reactive.effect
    @reactive.event(input.reset_crop)
    def _on_reset_crop():
        img_h, img_w = r_img_shape.get()
        ui.update_numeric("crop_x", value=0, session=session)
        ui.update_numeric("crop_y", value=0, session=session)
        ui.update_numeric("crop_w", value=img_w, session=session)
        ui.update_numeric("crop_h", value=img_h, session=session)

        f = input.upload_img()
        if not f:
            return

        path = _copy_with_original_name(f[0])

        mode = r_mode.get()
        az = (
            FruitInternalAnalyzer(path)
            if mode == "internal"
            else FruitExternalAnalyzer(path)
        )
        az.load_image(plot=False)
        r_analyzer.set(az)
        r_completed.set([])

    @reactive.effect
    @reactive.event(input.use_crop)
    def _on_use_crop_toggle():
        az = r_analyzer.get()
        if az is None:
            return
        if input.use_crop():
            plt.close("all")
            fig, ax_plot = plt.subplots(figsize=(12, 9))
            rgb = cv2.cvtColor(az.img, cv2.COLOR_BGR2RGB)
            ax_plot.imshow(rgb)
            ax_plot.set_xlabel("x  (px)", fontsize=11)
            ax_plot.set_ylabel("y  (px)", fontsize=11)
            ax_plot.set_title(
                "Reference axes — use these coordinates to set the crop region",
                fontsize=12,
            )
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
            buf.seek(0)
            r_axis_b64.set(base64.b64encode(buf.read()).decode())
            plt.close("all")
        else:
            r_axis_b64.set("")

    @render.ui
    def step1_preview():
        r_setup_done.get()
        if not r_img_ready.get():
            return ui.div()
        az = r_analyzer.get()
        if az is None:
            return ui.div()
        display_img = (
            cv2.cvtColor(az.img_copy, cv2.COLOR_BGR2RGB)
            if (hasattr(az, "img_copy") and az.img_copy is not None)
            else cv2.cvtColor(az.img, cv2.COLOR_BGR2RGB)
        )

        if input.use_crop():  # ← ejes ON mientras crop esté activo
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.imshow(display_img)
            ax.axis("on")
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            return ui.HTML(
                f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                f'style="width:100%;border-radius:8px;margin-top:.5rem">'
            )

        b64 = arr_to_b64(cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
        return ui.HTML(
            f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
            f'style="width:100%;border-radius:8px;margin-top:.5rem">'
        )

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
                width_cm=input.width_cm() if input.use_dimensions() else None,
                length_cm=input.length_cm() if input.use_dimensions() else None,
                diameter_cm=input.diameter_cm(),
                verbose=False,
            )
            mark_done(0)
            h, w = az.img.shape[:2] if az.img is not None else (0, 0)
            n_refs = len(az.ref_roi) if az.ref_roi else 0
            r_step1_result.set(
                ui.div(
                    ui.p(
                        "Setup complete!",
                        style="font-size:3.4rem; text-align:center; max-width:700px; "
                        "margin:0 auto; color: #97c8ec; font-weight:700; background-color:rgba(49,63,65,0.8);",
                    ),
                    ui.div(
                        ui.layout_columns(
                            ui.value_box(
                                "Label:", az.label_text or "N/A", theme="primary"
                            ),
                            ui.value_box(
                                "px/cm density:",
                                f"{az.px_per_cm:.1f}" if az.px_per_cm else "–",
                                theme="primary",
                            ),
                            col_widths=[6, 6],
                        ),
                        ui.layout_columns(
                            ui.value_box("Image size:", f"{w}×{h}", theme="secondary"),
                            ui.value_box(
                                "References:",
                                str(n_refs),
                                theme="success" if n_refs > 0 else "danger",
                            ),
                            col_widths=[6, 6],
                        ),
                        style="max-width: 700px; margin: 0 auto;",
                    ),
                )
            )
            r_setup_done.set(r_setup_done.get() + 1)
        except Exception as e:
            r_step1_result.set(
                ui.div(
                    ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc())
                )
            )

    @render.ui
    def step1_results():
        return r_step1_result.get()

    # step 2
    @render.ui
    def mask_bg_ui():
        if r_mode.get() == "external":
            return ui.input_select(
                "bg_color", "Background color", choices=["blue", "black", "white"]
            )
        return ui.div()

    @render.ui
    def mask_stamp_ui():
        if r_mode.get() == "internal":
            return ui.input_checkbox("stamp", "Invert image colors (stamp)", False)
        return ui.div()

    @reactive.effect
    @reactive.event(input.run_step2)
    def _run_step2():
        az = r_analyzer.get()
        if az is None:
            r_step2_result.set(ui.p("Complete Step 1 first.", class_="text-info"))
            return
        is_int = r_mode.get() == "internal"
        try:
            lower_hsv = (
                [input.h_range()[0], input.s_range()[0], input.v_range()[0]]
                if input.use_manual_hsv()
                else None
            )
            upper_hsv = (
                [input.h_range()[1], input.s_range()[1], input.v_range()[1]]
                if input.use_manual_hsv()
                else None
            )
            kw = dict(
                stamp=input.stamp() if r_mode.get() == "internal" else False,
                remove_roi=input.remove_roi(),
                lower_hsv=lower_hsv,
                upper_hsv=upper_hsv,
                n_iteration=input.n_iteration(),
                roi_expansion=input.roi_expansion(),
                kernel_blur=input.kernel_blur() or None,
                kernel_open=input.kernel_open() or None,
                kernel_close=input.kernel_close() or None,
                apply_convex_hull=input.apply_convex_hull(),
                fill_holes=input.fill_holes(),
                erosion_px=input.erosion_px(),
                plot=True,
                plot_size=(20, 20),
            )

            if not is_int:
                kw["background_color"] = input.bg_color()

            az.generate_fruit_mask(**kw)
            mark_done(1)
            if az.mask_fruit is not None:
                r_mask_edited.set(az.mask_fruit.copy())
                r_mask_history.set([])
                r_mask_points.set([])
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")
            r_step2_result.set(
                ui.HTML(
                    f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                    f'style="width:100%;border-radius:8px;margin-top:.5rem">'
                )
            )
        except Exception as e:
            r_step2_result.set(
                ui.div(
                    ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc())
                )
            )

    @render.ui
    def step2_results():
        return r_step2_result.get()

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

    def _render_editor_panels():
        az = r_analyzer.get()
        edited = r_mask_edited.get()

        try:
            overlay_alpha = float(input.mask_overlay_alpha())
        except Exception:
            overlay_alpha = 0.5

        if az is None or edited is None:
            return None, None

        img_h, img_w = edited.shape
        MAX_PX = 8000
        scale = min(1.0, MAX_PX / max(img_h, img_w))
        PANEL_W = int(img_w * scale)
        PANEL_H = int(img_h * scale)

        # left panel (mask)
        left = cv2.resize(
            cv2.cvtColor(edited, cv2.COLOR_GRAY2BGR),
            (PANEL_W, PANEL_H),
            interpolation=cv2.INTER_NEAREST,
        )

        # right panel (overlay)
        orig_rgb = az.img_rgb if az.img_rgb is not None else None

        if orig_rgb is not None:
            orig_rgb_correct = cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2RGB)
            orig_resized = cv2.resize(
                orig_rgb_correct, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA
            )
        else:
            orig_resized = cv2.resize(
                cv2.cvtColor(edited, cv2.COLOR_GRAY2BGR),
                (PANEL_W, PANEL_H),
                interpolation=cv2.INTER_AREA,
            )

        mask_resized = cv2.resize(
            edited, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST
        )
        mask_vis = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        right_full = cv2.addWeighted(
            orig_resized, 1.0 - overlay_alpha, mask_vis, overlay_alpha, 0
        )

        b64_mask = arr_to_b64(left)
        b64_orig = arr_to_b64(right_full)
        return b64_mask, b64_orig

    def _push_editor_panels():
        b64_mask, b64_orig = _render_editor_panels()
        if b64_mask and b64_orig:
            pts_list = [[p[0], p[1]] for p in r_mask_points.get()]
            session.send_custom_message(
                "mask_editor_update",
                {
                    "b64_mask": b64_mask,
                    "b64_orig": b64_orig,
                    "pts": pts_list,
                    "mode": r_mask_mode.get(),
                },
            )

    @render.ui
    def mask_editor_canvas():
        az = r_analyzer.get()
        edited = r_mask_edited.get()
        if az is None or edited is None:
            return ui.p(
                ui.HTML(
                    ' <i class="fa-solid fa-triangle-exclamation"></i> Create a mask first.'
                ),
                class_="text-info",
                style="font-size: 2rem",
            )

        with reactive.isolate():
            img_h, img_w = edited.shape
            b64_mask, b64_orig = _render_editor_panels()
            if not b64_mask:
                return ui.div()

            pts_json = _json.dumps([[p[0], p[1]] for p in r_mask_points.get()])
            mode_init = r_mask_mode.get()

            try:
                overlay_alpha_display = float(input.mask_overlay_alpha())
            except Exception:
                overlay_alpha_display = 0.5

        return ui.HTML(f"""
        <div style="display:flex;gap:6px;width:100%;">

        <!-- left panel for mask -->
        <div id="wrap_mask" style="position:relative;overflow:hidden;flex:1;
            border-radius:8px;background:#111;cursor:crosshair;user-select:none;">
            <img id="img_mask" src="data:image/png;base64,{b64_mask}"
                style="width:100%;display:block;transform-origin:0 0;" draggable="false">

            <canvas id="cvs_mask" style="position:absolute;top:0;left:0;
                    width:100%;height:100%;pointer-events:none;"></canvas>

            <div style="position:absolute;bottom:6px;left:50%;transform:translateX(-50%);
                background:rgba(0,0,0,.5);color:#fff;font-size:2.8rem;padding:2px 8px;
                border-radius:4px;pointer-events:none;">MASK</div>
        </div>

        <!-- right panel for image overlay -->
        <div id="wrap_orig" style="position:relative;overflow:hidden;flex:1;
                border-radius:8px;background:#111;cursor:crosshair;user-select:none;">
            <img id="img_orig" src="data:image/png;base64,{b64_orig}"
                style="width:100%;display:block;transform-origin:0 0;" draggable="false">
            <canvas id="cvs_orig" style="position:absolute;top:0;left:0;
                    width:100%;height:100%;pointer-events:none;"></canvas>
            <div style="position:absolute;bottom:6px;left:50%;transform:translateX(-50%);
                background:rgba(0,0,0,.5);color:#fff;font-size:2.8rem;padding:2px 8px;
                border-radius:4px;pointer-events:none;">OVERLAY</div>
        </div>

        </div>
        <script>
        (function() {{
            var IMG_W = {img_w};
            var IMG_H = {img_h};

                var panels = [
                {{wrap:document.getElementById('wrap_mask'),
                    img:document.getElementById('img_mask'), cvs:document.getElementById('cvs_mask')}},
                {{wrap:document.getElementById('wrap_orig'),
                    img:document.getElementById('img_orig'), cvs:document.getElementById('cvs_orig')}}
                ];

            if (!panels[0].wrap || !panels[1].wrap) return;

            var _zoom = 1.0, _panX = 0, _panY = 0;
            var _dragging = false, _dragSX, _dragSY, _panOX, _panOY;

            function _applyTransform() {{
                panels.forEach(function(p) {{
                p.img.style.transform =
                    'translate(' + _panX + 'px,' + _panY + 'px) scale(' + _zoom + ')';
            }});
            }}

            function _clampPan(rect) {{
                _panX = Math.min(0, Math.max(_panX, -(rect.width  * (_zoom - 1))));
                _panY = Math.min(0, Math.max(_panY, -(rect.height * (_zoom - 1))));
            }}

            panels.forEach(function(panel) {{
                // sync zoom
                panel.wrap.addEventListener('wheel', function(e) {{
                e.preventDefault();
                var rect = panel.wrap.getBoundingClientRect();
                var mx = e.clientX - rect.left, my = e.clientY - rect.top;
                var prev = _zoom;
                _zoom = Math.min(10.0, Math.max(1.0, _zoom * (e.deltaY < 0 ? 1.2 : 1/1.2)));
                _panX = mx - (_zoom/prev)*(mx - _panX);
                _panY = my - (_zoom/prev)*(my - _panY);
                if (_zoom === 1.0) {{ _panX = 0; _panY = 0; }}
                    _clampPan(rect);
                    _applyTransform();
                    _drawOverlay();
                }},
                {{passive:false}});

            panel.wrap.addEventListener('mousedown', function(e) {{
                if (e.button !== 2) return;
                e.preventDefault();
                _dragging = true;
                _dragSX = e.clientX; _dragSY = e.clientY;
                _panOX = _panX; _panOY = _panY;
                panels.forEach(function(p) {{ p.wrap.style.cursor = 'grabbing'; }});
            }});
            panel.wrap.addEventListener('contextmenu', function(e) {{ e.preventDefault(); }});
            }});

            document.addEventListener('mousemove', function(e) {{
                if (!_dragging) return;
                    _panX = _panOX + (e.clientX - _dragSX);
                    _panY = _panOY + (e.clientY - _dragSY);
                    _clampPan(panels[0].wrap.getBoundingClientRect());
                    _applyTransform();
                    _drawOverlay();
            }});

            document.addEventListener('mouseup', function(e) {{
            if (e.button !== 2) return;
                _dragging = false;
                panels.forEach(function(p) {{ p.wrap.style.cursor = 'crosshair'; }});
            }});

            // polygon points
            var _pts  = {pts_json};
            var _mode = '{mode_init}';

            function _screenToImg(clientX, clientY, panel) {{
                var rect = panel.wrap.getBoundingClientRect();
                var scaleX = panel.wrap.offsetWidth  > 0 ? rect.width  / panel.wrap.offsetWidth  : 1;
                var scaleY = panel.wrap.offsetHeight > 0 ? rect.height / panel.wrap.offsetHeight : 1;
                var ix = Math.round((clientX - rect.left - _panX) / _zoom / rect.width  * IMG_W * scaleX);
                var iy = Math.round((clientY - rect.top  - _panY) / _zoom / rect.height * IMG_H * scaleY);
                return {{x: Math.max(0,Math.min(ix,IMG_W-1)), y: Math.max(0,Math.min(iy,IMG_H-1))}};
            }}

            function _imgToScreen(ix, iy, panel) {{
                var rect = panel.wrap.getBoundingClientRect();
                return {{
                x: ix / IMG_W * rect.width  * _zoom + _panX,
                y: iy / IMG_H * rect.height * _zoom + _panY
                }};
            }}

            function _drawOnPanel(panel) {{
                var r = panel.wrap.getBoundingClientRect();
                panel.cvs.width = r.width; panel.cvs.height = r.height;
                var ctx = panel.cvs.getContext('2d');
                ctx.clearRect(0, 0, r.width, r.height);
                if (_pts.length === 0) return;
                var stroke = _mode === 'white' ? 'rgba(80,220,80,1)'  : 'rgba(206,18,1,1)';
                var fill   = _mode === 'white' ? 'rgba(0,180,0,0.25)' : 'rgba(200,0,0,0.25)';
                var spts = _pts.map(function(p) {{ return _imgToScreen(p[0], p[1], panel); }});
                ctx.beginPath();
                ctx.moveTo(spts[0].x, spts[0].y);
                for (var i=1;i<spts.length;i++) ctx.lineTo(spts[i].x, spts[i].y);
                if (_pts.length >= 3) {{ ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); }}
                ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke();
                spts.forEach(function(sp) {{
                // points size
                ctx.beginPath(); ctx.arc(sp.x, sp.y, 8, 0, Math.PI*2);
                ctx.fillStyle = stroke; ctx.fill();
                }});
                ctx.font = 'bold 32px sans-serif';
                var txt = 'Points: ' + _pts.length;
                var txtW = ctx.measureText(txt).width;
                var pad = 6;
                var txtX = 10;
                var txtY = 34;
                ctx.fillStyle = 'rgba(0,0,0,0.55)';
                ctx.beginPath();
                ctx.roundRect(txtX - pad, txtY - 24, txtW + pad*2, 30, 5);
                ctx.fill();
                ctx.fillStyle = 'rgba(255,255,255,1)';
                ctx.fillText(txt, txtX, txtY);
            }}

        function _drawOverlay() {{ panels.forEach(_drawOnPanel); }}
        _drawOverlay();


        panels.forEach(function(panel) {{
            panel.img.addEventListener('click', function(e) {{
                if (_dragging) return;
                var pt = _screenToImg(e.clientX, e.clientY, panel);
                _pts.push([pt.x, pt.y]);
                _drawOverlay();
                Shiny.setInputValue('mask_click', {{x: pt.x, y: pt.y}}, {{priority:'event'}});
            }});
        }});

        document.addEventListener('click', function(e) {{
            var btn = e.target && e.target.closest ? e.target.closest('[id]') : e.target;
            if (!btn) return;
            if (btn.id === 'mask_edit_mode_add')    {{ _mode = 'white'; _drawOverlay(); }}
            if (btn.id === 'mask_edit_mode_remove') {{ _mode = 'black'; _drawOverlay(); }}
            if (btn.id === 'mask_edit_apply')       {{ showToast('✓ Polygon applied'); }}
            if (btn.id === 'mask_edit_save')        {{ showToast('✓ Mask saved!'); }}
        }});


        Shiny.addCustomMessageHandler('mask_editor_update', function(data) {{
            panels[0].img.src = 'data:image/png;base64,' + data.b64_mask;
            panels[1].img.src = 'data:image/png;base64,' + data.b64_orig;
            if (data.pts  !== undefined) _pts  = data.pts;
            if (data.mode !== undefined) _mode = data.mode;
            _drawOverlay();
        }});

        }})();
        </script>
        """)

    ########################
    ## Interactive editor ##
    ########################

    @render.ui
    def mask_edit_mode_indicator():
        mode = r_mask_mode.get()
        color = "#059669" if mode == "white" else "#dc2626"
        label = "ADD region" if mode == "white" else "REMOVE region"
        return ui.HTML(
            f'<div style="font-size:1.6rem;font-weight:600;color:{color};'
            f"padding:.3rem .6rem;border-radius:6px;border:2px solid {color};"
            f'display:inline-block;margin-bottom:.5rem;">Mode: {label}</div>'
        )

    @reactive.effect
    @reactive.event(input.mask_overlay_alpha)
    def _on_alpha_change():
        _push_editor_panels()

    @reactive.effect
    @reactive.event(input.mask_click)
    def _on_mask_click():
        click = input.mask_click()
        if not click:
            return
        pts = list(r_mask_points.get())
        pts.append((int(click["x"]), int(click["y"])))
        r_mask_points.set(pts)

    @reactive.effect
    @reactive.event(input.mask_edit_mode_add)
    def _mask_mode_add():
        r_mask_mode.set("white")

    @reactive.effect
    @reactive.event(input.mask_edit_mode_remove)
    def _mask_mode_remove():
        r_mask_mode.set("black")

    # Apply polygon
    @reactive.effect
    @reactive.event(input.mask_edit_apply)
    def _mask_apply():
        pts = r_mask_points.get()
        if len(pts) < 3:
            return
        edited = r_mask_edited.get().copy()
        hist = list(r_mask_history.get())
        hist.append(edited.copy())
        poly = np.array(pts, dtype=np.int32)
        fill = 255 if r_mask_mode.get() == "white" else 0
        cv2.fillPoly(edited, [poly], fill)
        r_mask_history.set(hist)
        r_mask_edited.set(edited)
        r_mask_points.set([])
        _push_editor_panels()

    # Undo
    @reactive.effect
    @reactive.event(input.mask_edit_undo)
    def _mask_undo():
        hist = list(r_mask_history.get())
        if len(hist) > 0:
            prev = hist.pop()
            r_mask_history.set(hist)
            r_mask_edited.set(prev)
        r_mask_points.set([])
        _push_editor_panels()

    # Clear points
    @reactive.effect
    @reactive.event(input.mask_edit_clear)
    def _mask_clear():
        r_mask_points.set([])
        _push_editor_panels()

    @reactive.effect
    @reactive.event(input.mask_edit_save)
    def _mask_save():
        az = r_analyzer.get()
        edited = r_mask_edited.get()
        if az is None or edited is None:
            return

        if (
            r_mode.get() == "internal"
            and hasattr(az, "mask_locule")
            and az.mask_locule is not None
        ):
            az.mask_locule = edited.copy()
        else:
            az.mask_fruit = edited.copy()
        r_analyzer.set(az)
        mark_done(1)
        r_mask_points.set([])

    @reactive.effect
    @reactive.event(input.mask_edit_discard)
    def _mask_discard():
        az = r_analyzer.get()
        if az is None:
            return
        hist = r_mask_history.get()
        if hist:
            r_mask_edited.set(hist[0].copy())
        r_mask_points.set([])
        r_mask_history.set([])
        _push_editor_panels()

    # step 3
    @reactive.effect
    @reactive.event(input.run_step3)
    def _run_step3():
        az = r_analyzer.get()
        if az is None:
            r_step3_result.set(
                ui.p("Complete earlier steps first.", class_="text-info")
            )
            return
        try:
            method = input.contrast_method()
            gamma = input.gamma() if method == "gamma" else 1.5
            gain = input.gain() if method == "sigmoid" else 5.0
            cutoff = input.cutoff() if method == "sigmoid" else 0.5
            c = input.c_val() if method == "exp" else 0.5
            az.enhance_locule_contrast(
                contrast_method=method,
                gamma=gamma,
                gain=gain,
                cutoff=cutoff,
                c=c,
                plot=True,
                compare_method=input.compare_method(),
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
            r_step3_result.set(
                ui.HTML(
                    f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                    f'style="width:100%;border-radius:8px;margin-top:.5rem">'
                )
            )
        except Exception as e:
            r_step3_result.set(
                ui.div(
                    ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc())
                )
            )

    @render.ui
    def step3_results():
        return r_step3_result.get()

    @render.ui
    def contrast_params_ui():
        method = input.contrast_method()
        if method == "gamma":
            return ui.input_slider("gamma", "Gamma", 0.1, 3.0, 1.5, step=0.1)
        elif method == "sigmoid":
            return ui.div(
                ui.input_slider("gain", "Gain", 1.0, 10.0, 5.0, step=0.5),
                ui.input_slider("cutoff", "Cutoff", 0.0, 1.0, 0.5, step=0.05),
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

    @reactive.effect
    @reactive.event(input.run_step4)
    def _run_step4():
        az = r_analyzer.get()
        if az is None:
            r_step4_result.set(
                ui.p("Complete earlier steps first.", class_="text-info")
            )
            return
        try:
            thresh = input.thresh_min() if input.use_thresh() else 120
            otsu_off = input.otsu_offset_lm() if input.use_otsu() else None
            if input.gen_histogram():
                az.generate_l_channel_histogram(otsu_offset=input.otsu_offset())
                buf = io.BytesIO()
                plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                plt.close("all")
                r_step4_result.set(
                    ui.div(
                        ui.HTML(
                            f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                            f'style="width:100%;border-radius:8px;margin-top:.5rem">'
                        )
                    )
                )
                return
            az.generate_locule_mask(
                thresh_min=thresh,
                otsu_offset=otsu_off,
                min_fruit_area=input.min_fruit_area_lm(),
                min_locule_area=input.min_locule_area_lm(),
                invert_locule=input.invert_locule(),
                kernel_blur=input.kernel_blur4() or None,
                kernel_open=input.kernel_open4() or None,
                kernel_close=input.kernel_close4() or None,
                erosion_px=input.erosion_px4(),
                plot=True,
                plot_size=(20, 20),
            )
            mark_done(3)

            _edit_src = (
                az.mask_locule
                if (hasattr(az, "mask_locule") and az.mask_locule is not None)
                else az.mask_fruit
                if az.mask_fruit is not None
                else None
            )
            if _edit_src is not None:
                r_mask_edited.set(_edit_src.copy())
                r_mask_history.set([_edit_src.copy()])
                r_mask_points.set([])
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")
            r_step4_result.set(
                ui.div(
                    ui.HTML(
                        f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                        f'style="width:100%;border-radius:8px;margin-top:.5rem">'
                    )
                )
            )
        except Exception as e:
            r_step4_result.set(
                ui.div(
                    ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc())
                )
            )

    @render.ui
    def step4_results():
        return r_step4_result.get()

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
                ui.input_numeric(
                    "min_locule_area", "Min locule area (px)", 50, min=1, step=10
                ),
                ui.input_numeric(
                    "min_locule_per_fruit", "Min locules/fruit", 1, min=0, step=1
                ),
            )
        return ui.div()

    @render.ui
    def detect_int_styling_ui():
        if r_mode.get() == "internal":
            return ui.div(
                ui.input_slider(
                    "locule_thickness_det", "Locule contour thickness", 1, 10, 2
                ),
                ui.input_text(
                    "locule_color_det", "Locule contour color (R,G,B)", "255,0,255"
                ),
                ui.input_slider(
                    "pericarp_int_thickness_det",
                    "Int. pericarp contour thickness",
                    1,
                    10,
                    2,
                ),
                ui.input_text(
                    "pericarp_int_color_det",
                    "Int. pericarp contour color (R,G,B)",
                    "0,255,255",
                ),
            )
        return ui.div()

    @render.ui
    def detect_dilation_ui():
        if r_mode.get() == "internal":
            return ui.input_numeric(
                "dilation_factor", "Dilation factor", 0, min=0, step=0.1
            )
        return ui.div()

    @render.ui
    def morph_advanced_section_ui():
        if r_mode.get() != "internal":
            return ui.div()
        return ui.div(
            ui.HTML("""
            <details style="margin-bottom:.8rem">
            <summary style="font-size:1.7rem;font-weight:600;cursor:pointer;
                            padding:.4rem .2rem;color:#475569;user-select:none">
                Advanced parameters
            </summary>
            <div style="padding:.6rem 0 0 .4rem">
            """),
            ui.input_numeric(
                "dilation_morph", "Dilation factor (int. pericarp)", 0, min=0, step=0.05
            ),
            ui.input_numeric(
                "angle_shifts_morph", "Angle shifts (symmetry)", 500, min=0, step=50
            ),
            ui.input_numeric(
                "num_rays_morph", "Num rays (pericarp thickness)", 90, min=0, step=10
            ),
            ui.HTML("</div></details>"),
        )

    @render.ui
    def morph_int_styling_ui():
        if r_mode.get() == "internal":
            return ui.div(
                ui.input_text(
                    "pericarp_int_color_morph",
                    "Int. pericarp color (R,G,B)",
                    "0,240,240",
                ),
                ui.input_numeric(
                    "pericarp_int_thick_morph",
                    "Int. pericarp thickness",
                    2,
                    min=1,
                    step=1,
                ),
                ui.input_text(
                    "locule_color_morph", "Locule color (R,G,B)", "255,0,255"
                ),
                ui.input_numeric(
                    "locule_thick_morph", "Locule thickness", 2, min=1, step=1
                ),
                ui.input_text(
                    "centroid_fruit_color_morph",
                    "Fruit centroid color (R,G,B)",
                    "255,255,51",
                ),
                ui.input_numeric(
                    "centroid_fruit_thick_morph",
                    "Fruit centroid size",
                    2,
                    min=1,
                    step=1,
                ),
                ui.input_text(
                    "centroid_locule_color_morph",
                    "Locule centroid color (R,G,B)",
                    "0,255,255",
                ),
                ui.input_numeric(
                    "centroid_locule_thick_morph",
                    "Locule centroid size",
                    2,
                    min=1,
                    step=1,
                ),
            )
        return ui.div()

    @reactive.effect
    @reactive.event(input.run_detect)
    def _run_detect():
        az = r_analyzer.get()
        if az is None:
            r_detect_result.set(
                ui.p("Complete earlier steps first.", class_="text-info")
            )
            return
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
                verbose=False,
                plot=True,
                plot_size=(20, 20),
            )
            if is_int:
                kw["min_locule_area"] = input.min_locule_area()
                kw["min_locule_per_fruit"] = input.min_locule_per_fruit()
                kw["locule_thickness"] = input.locule_thickness_det()
                kw["locule_color"] = _parse_color(input.locule_color_det())
                kw["dilation_factor"] = input.dilation_factor() or None
                kw["pericarp_int_color"] = _parse_color(input.pericarp_int_color_det())
                kw["pericarp_int_thickness"] = input.pericarp_int_thickness_det()
            az.detect_fruits(**kw)
            idx = 5 if is_int else 3
            mark_done(idx)
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")
            r_detect_result.set(
                ui.HTML(
                    f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                    f'style="width:100%;border-radius:8px;margin-top:.5rem">'
                )
            )
        except Exception as e:
            r_detect_result.set(
                ui.div(
                    ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc())
                )
            )

    @render.ui
    def detect_results():
        return r_detect_result.get()

    # step 6 – morphology
    @render.ui
    def epsilon_ui():
        if input.contour_mode() == "approx":
            return ui.input_numeric(
                "epsilon_morph",
                "Epsilon (approx simplification)",
                0.001,
                min=0.0001,
                step=0.001,
            )
        return ui.div()

    @render.ui
    def morph_locule_params_ui():
        if r_mode.get() == "internal":
            return ui.div(
                ui.input_numeric(
                    "min_locule_area_morph", "Min locule area (px)", 10, min=0, step=10
                ),
                ui.input_numeric(
                    "max_locule_area_morph", "Max locule area (px)", 0, min=0, step=100
                ),
                ui.p(
                    "Set to 0 for no upper limit.",
                    style="font-size:1.4rem;color:#94a3b8;margin-top:-.5rem;",
                ),
            )
        return ui.div()

    @reactive.effect
    @reactive.event(input.run_morph)
    def _run_morph():
        az = r_analyzer.get()
        if az is None:
            r_morph_result.set(
                ui.p("Complete earlier steps first.", class_="text-info")
            )
            return
        is_int = r_mode.get() == "internal"
        try:

            def _pc(s):
                return tuple(int(x.strip()) for x in s.split(","))

            epsilon = (
                input.epsilon_morph() if input.contour_mode() == "approx" else 0.001
            )
            dilation_val = (
                (input.dilation_morph() if input.dilation_morph() > 0 else None)
                if is_int
                else None
            )
            max_loc = (input.max_locule_area_morph() or None) if is_int else None
            kw = dict(
                contour_mode=input.contour_mode(),
                epsilon=epsilon,
                font_size=input.font_size_morph(),
                font_thickness=input.font_thickness_morph(),
                font_color=_pc(input.font_color_morph()),
                label_position=input.label_position_morph(),
                label_color=_pc(input.label_color_morph()),
                pericarp_ext_color=_pc(input.pericarp_ext_color_morph()),
                pericarp_ext_thickness=input.pericarp_ext_thick_morph(),
                display_table=True,
                plot=True,
                plot_size=(20, 20),
            )
            if is_int:
                kw.update(
                    dict(
                        dilation_factor=dilation_val,
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
                    )
                )
            df = az.analyze_morphology(**kw)
            idx = 6 if is_int else 4
            mark_done(idx)
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")
            parts = [
                ui.HTML(
                    f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                    f'style="width:100%;border-radius:8px;margin-top:.5rem">'
                )
            ]
            if df is not None and not df.empty:
                csv_b = df.to_csv(index=False).encode()
                parts += [
                    ui.output_ui("morph_table_dt"),
                    ui.download_button(
                        "dl_morph",
                        ui.HTML(
                            '<i class="fa-solid fa-file-arrow-down"></i> Download CSV'
                        ),
                        class_="btn btn-primary",
                        style="font-size:1.8rem;padding:.7rem 1.2rem;"
                    ),
                ]

                @render.ui
                def morph_table_dt():
                    return ui.HTML(
                        _df_to_datatable(
                            df, "morph_dt_tbl", page_length=5, cols_per_page=7
                        )
                    )

                @render.download(filename="morphology_results.csv")
                async def dl_morph():
                    yield csv_b

            tmp_dir = tempfile.mkdtemp()
            base = (
                r_original_img_name.get()
                or os.path.splitext(os.path.basename(az.input_path))[0]
            )
            ann_path = os.path.join(tmp_dir, f"{base}_processed.png")
            if az.results is not None and az.results.morphology_image is not None:
                cv2.imwrite(ann_path, az.results.morphology_image)
            if df is not None and not df.empty:
                df.to_csv(
                    os.path.join(tmp_dir, f"{base}_morphology_results.csv"), index=False
                )
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
            parts.append(
                ui.download_button(
                    "dl_morph_zip",
                    ui.HTML(
                        '<i class="fa-solid fa-file-arrow-down"></i> Download CSV + Image (.zip)'
                    ),
                    class_="btn btn-primary",
                    style="margin-left: 1.5rem; font-size:1.8rem;padding:.7rem 1.2rem;"
                )
            )
            if params_saved:
                parts.append(
                    ui.p(
                        "Parameters files created and included in the .zip!",
                        style="font-size:1.6rem; text-align:center; max-width:700px; "
                        "color:#97c8ec; font-weight:700; background-color:rgba(49,63,65,0.9); border-radius:6px; padding:.4rem;",
                    )
                )
            r_morph_result.set(ui.div(*parts))
        except Exception as e:
            r_morph_result.set(
                ui.div(
                    ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc())
                )
            )

    @render.ui
    def morph_results():
        return r_morph_result.get()

    # step 7 - color
    @render.ui
    def color_tissue_ui():
        if r_mode.get() == "internal":
            return ui.input_select(
                "tissue",
                "Tissue",
                choices=[
                    "all",
                    "total_pericarp",
                    "outer_pericarp",
                    "inner_pericarp",
                    "locules",
                ],
            )
        return ui.div()

    @render.ui
    def color_int_styling_ui():
        if r_mode.get() == "internal":
            return ui.div(
                ui.input_text(
                    "pericarp_int_color_color",
                    "Int. pericarp color (R,G,B)",
                    "0,255,255",
                ),
                ui.input_numeric(
                    "pericarp_int_thick_color",
                    "Int. pericarp thickness",
                    2,
                    min=1,
                    step=1,
                ),
                ui.input_text(
                    "locule_color_color", "Locule color (R,G,B)", "255,0,255"
                ),
                ui.input_numeric(
                    "locule_thick_color", "Locule thickness", 2, min=1, step=1
                ),
            )
        return ui.div()

    @reactive.effect
    @reactive.event(input.run_color)
    def _run_color():
        az = r_analyzer.get()
        if az is None:
            r_color_result.set(
                ui.p("Complete earlier steps first.", class_="text-info")
            )
            return
        is_int = r_mode.get() == "internal"
        try:

            def _pc(s):
                return tuple(int(x.strip()) for x in s.split(","))

            want_histogram = input.get_color_histogram()

            if want_histogram:
                try:
                    from traitly.fruit_phenotyping import plot_dark_threshold
                except ImportError:
                    from traitly.fruit_phenotyping.color_plot import plot_dark_threshold
                plt.close("all")
                plot_dark_threshold(
                    az.img,
                    az.mask_fruit,
                    dark_threshold=input.dark_thresh_color(),
                )
                buf = io.BytesIO()
                plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                plt.close("all")
                r_color_result.set(
                    ui.HTML(
                        f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                        f'style="width:100%;border-radius:8px;margin-top:.5rem">'
                    )
                )
                return

            kw = dict(
                stat=input.stat(),
                color_space=input.color_space(),
                get_color_histogram=False,
                display_table=True,
                plot=True,
                plot_size=(20, 20),
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
                kw["tissue"] = input.tissue()
                kw["pericarp_int_color"] = _pc(input.pericarp_int_color_color())
                kw["pericarp_int_thickness"] = input.pericarp_int_thick_color()
                kw["locule_color"] = _pc(input.locule_color_color())
                kw["locule_thickness"] = input.locule_thick_color()

            df = az.analyze_color(**kw)
            idx = 7 if is_int else 5
            mark_done(idx)

            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")

            parts = [
                ui.HTML(
                    f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                    f'style="width:100%;border-radius:8px;margin-top:.5rem">'
                )
            ]

            if df is not None and not df.empty:
                csv_b = df.to_csv(index=False).encode()
                parts += [
                    ui.output_ui("color_table_dt"),
                    ui.download_button(
                        "dl_color_csv",
                        ui.HTML(
                            '<i class="fa-solid fa-file-arrow-down"></i> Download CSV'
                        ),
                        class_="btn btn-primary",
                        style="font-size:1.8rem;padding:.7rem 1.2rem;"
                    ),
                ]

                @render.ui
                def color_table_dt():
                    return ui.HTML(
                        _df_to_datatable(
                            df, "color_dt_tbl", page_length=5, cols_per_page=7
                        )
                    )

                @render.download(filename="color_results.csv")
                async def dl_color_csv():
                    yield csv_b

            tmp_dir = tempfile.mkdtemp()
            base = (
                r_original_img_name.get()
                or os.path.splitext(os.path.basename(az.input_path))[0]
            )

            ann_img = getattr(az.results, "morphology_image", None)
            col_img = getattr(az.results, "color_image", None)
            if ann_img is not None:
                img_to_save, img_filename = ann_img, f"{base}_processed.png"
            elif col_img is not None:
                img_to_save, img_filename = col_img, f"{base}_color.png"
            else:
                img_to_save = img_filename = None

            if img_to_save is not None:
                cv2.imwrite(os.path.join(tmp_dir, img_filename), img_to_save)
            if df is not None and not df.empty:
                df.to_csv(
                    os.path.join(tmp_dir, f"{base}_color_results.csv"), index=False
                )

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

            parts.append(
                ui.download_button(
                    "dl_color_zip",
                    ui.HTML(
                        '<i class="fa-solid fa-file-arrow-down"></i> Download CSV + Image (.zip)'
                    ),
                    class_="btn btn-primary",
                    style="margin-left: 1.5rem; font-size:1.8rem;padding:.7rem 1.2rem;"
                )
            )
            if params_saved:
                parts.append(
                    ui.p(
                        "Parameters files created and included in the .zip!",
                        style="font-size:1.6rem; text-align:center; max-width:700px; "
                        "color:#97c8ec; font-weight:700; "
                        "background-color:rgba(49,63,65,0.9); border-radius:6px; padding:.4rem;",
                    )
                )
            r_color_result.set(ui.div(*parts))

        except Exception as e:
            r_color_result.set(
                ui.div(
                    ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc())
                )
            )

    @render.ui
    def color_results():
        return r_color_result.get()

    @render.download(filename=lambda: f"{r_color_base.get()}_color.zip")
    async def dl_color_zip():
        data = r_color_zip.get()
        if data:
            yield data

    # bg helper
    bg_thresh = {
        "blue": ([90, 50, 50], [130, 255, 255]),
        "white": ([0, 0, 180], [180, 40, 255]),
        "black": ([0, 0, 0], [180, 255, 50]),
    }

    @reactive.effect
    @reactive.event(input.bg_upload)
    def _load_bg():
        f = input.bg_upload()
        if not f:
            return
        path = _copy_with_original_name(f[0])
        az = FruitExternalAnalyzer(path)
        az.load_image(plot=False)
        r_bg_analyzer.set(az)

    @reactive.effect
    @reactive.event(input.bg_preset)
    def _apply_bg_preset():
        preset = input.bg_preset()
        if preset not in bg_thresh:
            return
        lo, hi = bg_thresh[preset]
        ui.update_slider("bg_h_lo", value=lo[0], session=session)
        ui.update_slider("bg_s_lo", value=lo[1], session=session)
        ui.update_slider("bg_v_lo", value=lo[2], session=session)
        ui.update_slider("bg_h_hi", value=hi[0], session=session)
        ui.update_slider("bg_s_hi", value=hi[1], session=session)
        ui.update_slider("bg_v_hi", value=hi[2], session=session)

    @render.ui
    def bg_scatter_out():
        az = r_bg_analyzer.get()
        if az is None:
            return ui.div(
                ui.HTML(
                    '<div style="display:flex;align-items:center;justify-content:center;'
                    "height:200px;border:2px dashed #e2e8f0;border-radius:10px;"
                    'color:#94a3b8;font-size:1.8rem;">Upload an image to see the HSV scatter plot</div>'
                )
            )
        try:
            plt.close("all")
            az.generate_color_scatterplot(
                sample_size=input.bg_sample(), plot_size=(14, 4)
            )
            buf = io.BytesIO()
            plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")
            return ui.HTML(
                f'<img src="data:image/png;base64,{b64}" class="img-zoomable" '
                f'style="width:100%;border-radius:8px;">'
            )
        except Exception as e:
            return ui.div(
                ui.p(f"{e}", class_="text-danger"), ui.pre(traceback.format_exc())
            )

    @render.ui
    def bg_preview():
        az = r_bg_analyzer.get()
        if az is None:
            return ui.div()
        lo = np.array(
            [input.bg_h_lo(), input.bg_s_lo(), input.bg_v_lo()], dtype=np.uint8
        )
        hi = np.array(
            [input.bg_h_hi(), input.bg_s_hi(), input.bg_v_hi()], dtype=np.uint8
        )
        mask = cv2.inRange(az.img_hsv, lo, hi)
        orig = az.img.copy()
        fruit_mask = cv2.bitwise_not(mask)
        pct = 100 * mask.sum() / 255 / mask.size
        return ui.div(
            ui.layout_columns(
                ui.div(
                    ui.HTML(
                        img_tag(orig, "width:100%;border-radius:8px;margin-top:.3rem")
                    ),
                    ui.p(
                        "Original image",
                        style="font-size:1.5rem;color:#64748b;text-align:center;margin-top:.3rem;",
                    ),
                ),
                ui.div(
                    ui.HTML(
                        img_tag(
                            fruit_mask, "width:100%;border-radius:8px;margin-top:.3rem"
                        )
                    ),
                    ui.p(
                        f"Fruit mask — background coverage: {pct:.1f}%",
                        style="font-size:1.5rem;color:#64748b;text-align:center;margin-top:.3rem;",
                    ),
                ),
                col_widths=[6, 6],
            ),
        )

    @render.ui
    def bg_final_code():
        lo = [input.bg_h_lo(), input.bg_s_lo(), input.bg_v_lo()]
        hi = [input.bg_h_hi(), input.bg_s_hi(), input.bg_v_hi()]
        code = f"lower_hsv = {lo}\nupper_hsv  = {hi}"
        return ui.div(
            ui.p(
                'Use these color threshold values in "Generate Fruit Mask" Section:',
                style="font-size:1.7rem;color:#059669;margin-bottom:.4rem;",
            ),
            ui.pre(
                code,
                style="background:var(--step-num-bg);padding:1rem;border-radius:6px;"
                "font-size:1.5rem;color:var(--body-text);",
            ),
        )

    ## batch analysis

    @render.ui
    @reactive.event(input.run_batch)
    def batch_results():
        files = input.batch_files()
        if not files:
            return ui.p("Select images first.", class_="text-info")

        is_int = input.batch_mode() == "internal"
        num_cores = max(1, int(input.batch_num_cores()))

        tmp_root = tempfile.mkdtemp()
        img_dir = os.path.join(tmp_root, "images")
        output_path = os.path.join(tmp_root, "Results")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        n_total = len(files)
        with ui.Progress(min=0, max=n_total, session=session) as p:
            for i, f in enumerate(files):
                p.set(value=i, message=f"Copying {i + 1}/{n_total}", detail=f["name"])
                dest = os.path.join(img_dir, f["name"])
                shutil.copy2(f["datapath"], dest)
            p.set(value=n_total, message="Running analysis…", detail="")

        json_f = input.batch_json()
        json_path = json_f[0]["datapath"] if json_f else None

        try:
            az = (
                FruitInternalAnalyzer(img_dir)
                if is_int
                else FruitExternalAnalyzer(img_dir)
            )
            with ui.Progress(min=0, max=1, session=session) as p:
                p.set(
                    value=0,
                    message=f"Running analysis on {n_total} images…",
                    detail="This may take a while depending on dataset size and cores.",
                )
                az.analyze_folder(
                    analyze_morphology=input.run_morphology(),
                    analyze_color=input.run_color_batch(),
                    json_path=json_path,
                    output_path=output_path,
                    num_cores=num_cores,
                    verbose=False,
                )
                p.set(value=1, message="Done!", detail="Building results…")
        except Exception as e:
            return ui.div(
                ui.p(f"{e}", class_="text-danger"),
                ui.pre(traceback.format_exc()),
            )

        out_files = os.listdir(output_path)
        ann_images = [
            f
            for f in out_files
            if f.endswith("_processed.jpg") or f.endswith("_processed.png")
        ]
        morph_csv = next((f for f in out_files if f == "morphology_results.csv"), None)
        color_csv = next((f for f in out_files if f == "color_results.csv"), None)
        session_txt = next((f for f in out_files if f == "session_report.txt"), None)
        error_txt = next((f for f in out_files if f == "error_report.txt"), None)

        n_ok = n_total
        n_errors = 0
        total_fruits = 0
        if session_txt:
            try:
                with open(os.path.join(output_path, session_txt)) as fh:
                    for line in fh:
                        if "images ok" in line:
                            n_ok = int(line.split(":")[-1].strip())
                        elif "images failed" in line:
                            n_errors = int(line.split(":")[-1].strip())
                        elif "total fruits" in line:
                            total_fruits = int(line.split(":")[-1].strip())
            except Exception:
                pass

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in out_files:
                full = os.path.join(output_path, fname)
                if os.path.isfile(full):
                    zf.write(full, arcname=fname)
        r_batch_zip.set(zip_buf.getvalue())

        status_color = (
            "#dc2626"
            if n_errors == n_total
            else "#d97706"
            if n_errors > 0
            else "#059669"
        )
        status_icon = (
            '<i class="fa-solid fa-circle-exclamation"></i>'
            if n_errors == n_total
            else '<i class="fa-solid fa-triangle-exclamation"></i>'
            if n_errors > 0
            else '<i class="fa-solid fa-circle-check"></i>'
        )
        status_msg = (
            f"{status_icon} Batch complete. Check error details below."
            if n_errors > 0
            else f"{status_icon} Batch complete"
        )

        parts = [
            ui.p(
                ui.HTML(
                    f'<p style="font-size:2rem;font-weight:700;color:{status_color};margin-bottom:1rem;">'
                    f"{status_msg}</p>"
                )
            ),
            ui.layout_columns(
                ui.value_box("Images found", n_total, theme="secondary"),
                ui.value_box("Successfully analyzed", n_ok, theme="success"),
                ui.value_box("Total fruits", total_fruits, theme="primary"),
                ui.value_box(
                    "Errors", n_errors, theme="warning" if n_errors else "secondary"
                ),
            ),
            ui.hr(),
            ui.HTML(
                f'<div style="font-size:1.7rem;color:#475569;margin:.5rem 0;">'
                f"<b>Files generated:</b><br>"
                f"&nbsp;&nbsp;★ {len(ann_images)} annotated image(s)<br>"
                + ("&nbsp;&nbsp;★ morphology_results.csv<br>" if morph_csv else "")
                + ("&nbsp;&nbsp;★ color_results.csv<br>" if color_csv else "")
                + ("&nbsp;&nbsp;★ session_report.txt<br>" if session_txt else "")
                + ("&nbsp;&nbsp;★ error_report.txt<br>" if error_txt else "")
                + "</div>"
            ),
            ui.hr(),
            ui.download_button(
                "dl_batch_zip",
                ui.HTML(
                    '<i class="fa-solid fa-file-arrow-down"></i> Download all results (.zip)'
                ),
                class_="btn btn-primary",
                style="font-size:1.8rem;padding:.7rem 1.2rem;",
            ),
        ]

        # Show error details but only if error report exists
        if error_txt:
            try:
                with open(os.path.join(output_path, error_txt)) as fh:
                    err_content = fh.read()
                parts.append(
                    ui.div(
                        ui.HTML("<br>"),
                        ui.HTML(
                            '<p style="font-size:1.7rem;font-weight:600;color:#d97706;margin-top:1rem;"> Details:</p>'
                        ),
                        ui.pre(
                            err_content,
                            style="background:#fff8ed;padding:.8rem;border-radius:6px; font-size:1.3rem;color:#92400e;max-height:800px;overflow-y:auto;",
                        ),
                    )
                )
            except Exception:
                pass

        return ui.div(*parts)

    @render.download(filename=lambda: f"{r_morph_base.get()}_morphology.zip")
    async def dl_morph_zip():
        data = r_morph_zip.get()
        if data:
            yield data

    @render.download(filename="traitly_results.zip")
    async def dl_batch_zip():
        data = r_batch_zip.get()
        if data:
            yield data

    ## PDF extractor
    @render.ui
    @reactive.event(input.run_pdf)
    def pdf_results():
        files = input.pdf_file()
        if not files:
            return ui.p("Upload a PDF file first.", class_="text-info")
        try:
            from traitly.pdf import pdf_to_img
        except ImportError:
            return ui.div(
                ui.p(
                    '<i class="fa-solid fa-triangle-exclamation"></i> PyMuPDF is not installed.',
                    class_="text-danger",
                ),
                ui.pre('Install it with:  pip install "traitly[pdf]"'),
            )

        all_saved = []
        all_errors = []

        with ui.Progress(min=0, max=len(files), session=session) as p:
            for idx, f in enumerate(files):
                pdf_name = f["name"]
                p.set(
                    value=idx,
                    message=f"Extracting {idx + 1}/{len(files)}…",
                    detail=pdf_name,
                )
                tmp_in = tempfile.mkdtemp()
                tmp_out = tempfile.mkdtemp()
                pdf_path = os.path.join(tmp_in, pdf_name)
                shutil.copy2(f["datapath"], pdf_path)
                try:
                    saved = pdf_to_img(
                        pdf_path,
                        dpi=input.pdf_dpi(),
                        output_path=tmp_out,
                        verbose=False,
                        detect_qr=input.pdf_qr_label(),
                        output_format=input.pdf_format(),
                    )
                    all_saved.extend(saved)
                except Exception as e:
                    all_errors.append(f"{pdf_name}: {e}")
            p.set(value=len(files), message="Done!", detail="")

        n_pages = len(all_saved)
        if n_pages == 0:
            return ui.div(
                ui.p("No images were generated.", class_="text-warning"),
                *[ui.p(e, class_="text-danger") for e in all_errors],
            )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in all_saved:
                zf.write(fpath, arcname=os.path.basename(fpath))
        r_pdf_zip.set(zip_buf.getvalue())

        thumb_html = ""
        for fpath in all_saved[:6]:
            try:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                h, w = img.shape[:2]
                scale = min(300 / w, 300 / h)
                if scale < 1.0:
                    img = cv2.resize(
                        img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                success, encoded = cv2.imencode(".png", img)
                if not success:
                    continue
                b64 = base64.b64encode(encoded.tobytes()).decode()
                fname = os.path.basename(fpath)
                thumb_html += f"""
                <div style="border:1px solid #e2e8f0;border-radius:8px;padding:.5rem;
                            background:#f8fafc;overflow:hidden;">
                    <img src="data:image/png;base64,{b64}" class="img-zoomable"
                        style="width:100%;height:200px;object-fit:contain;
                                border-radius:6px;display:block;">
                    <p style="font-size:1.3rem;color:#64748b;text-align:center;
                            margin-top:.4rem;word-break:break-all;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                    title="{fname}">{fname}</p>
                </div>
                """
            except Exception:
                pass

        n_pdfs = len(files)
        msg = (
            ui.HTML(
                f'<i class="fa-solid fa-circle-check"></i> {n_pages} page(s) extracted from {n_pdfs} PDF(s)'
            )
            if n_pdfs > 1
            else ui.HTML(
                f' <i class="fa-solid fa-circle-check"></i> {n_pages} page(s) extracted from {files[0]["name"]}'
            )
        )

        parts = [
            ui.p(
                msg,
                style="font-size:2rem;font-weight:700;color:#059669;margin-bottom:1rem;",
            ),
            ui.download_button(
                "dl_pdf_zip",
                ui.HTML(
                    '<i class="fa-solid fa-file-arrow-down"></i> Download all images (.zip)'
                ),
                class_="btn btn-primary",
                style="font-size:1.8rem;padding:.7rem 1.2rem;margin-bottom:1.5rem;",
            ),
            ui.hr(),
        ]
        if all_errors:
            parts.append(
                ui.div(
                    ui.HTML(
                        '<p style="font-size:1.7rem;font-weight:600;color:#d97706;">Errors</p>'
                    ),
                    *[
                        ui.p(e, class_="text-danger", style="font-size:1.4rem;")
                        for e in all_errors
                    ],
                    ui.hr(),
                )
            )
        if thumb_html:
            parts += [
                ui.p(
                    "Preview (first 6 pages):",
                    style="font-size:1.6rem;color:#475569;margin-bottom:.8rem;",
                ),
                ui.HTML(f"""
                <div style="display:grid;grid-template-columns:repeat(3,1fr);
                            gap:1rem;width:100%;">
                    {thumb_html}
                </div>
                """),
            ]
        return ui.div(*parts)

    @render.download(
        filename=lambda: (
            f"{os.path.splitext(input.pdf_file()[0]['name'])[0]}_images.zip"
            if input.pdf_file() and len(input.pdf_file()) == 1
            else "pdf_images.zip"
        )
    )
    async def dl_pdf_zip():
        data = r_pdf_zip.get()
        if data:
            yield data


app = App(app_ui, server)


# Run the app with CLI
def run():
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Run Traitly Shiny app")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", default=8000, type=int, help="Port number")
    args = parser.parse_args()
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run(
        [
            "shiny",
            "run",
            app_path,
            "--reload",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    )
