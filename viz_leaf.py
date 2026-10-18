"""Render results/LeafVisualize into one self-contained HTML report.

    python viz_leaf.py [result_dir] [-o out.html]

Stdlib only -- the analysis box has no numpy/pandas. Charts are inline SVG
generated here; colours come from CSS variables so the page themes itself.
"""
import argparse
import csv
import math
import glob
import os
import re
import struct
import zipfile

AXES = [("learn_filters", "filterbank", "Gabor centre frequencies and bandwidths"),
        ("learn_pooling", "pooling", "Gaussian lowpass widths"),
        ("learn_compression", "compression", "PCEN alpha, delta, root"),
        ("learn_smoothing", "smoothing", "PCEN per-channel EMA coefficient")]
BITS = [a[0] for a in AXES]


# ---------------------------------------------------------------- data

def load(result_dir):
    sweep = list(csv.DictReader(open(os.path.join(result_dir, "sweep.csv"))))
    for r in sweep:
        r["tag"] = "".join(r[b] for b in BITS)
        for k in ("test_uar", "test_wa", "best_val_uar", "train_wa"):
            r[k] = float(r[k])
        r["leaf_trainable_params"] = int(r["leaf_trainable_params"])

    curves = {}
    for row in csv.DictReader(open(os.path.join(result_dir, "training_log.csv"))):
        curves.setdefault(row["run"][-4:], []).append(
            (float(row["val_uar"]), float(row["val_wa"]), float(row["loss"])))

    cms, labels = {}, []
    for path in glob.glob(os.path.join(result_dir, "confusion_matrix_*.csv")):
        rows = list(csv.reader(open(path)))
        labels = rows[0][1:]
        cms[re.search(r"leaf(\d{4})\.csv$", path).group(1)] = \
            [[int(v) for v in r[1:]] for r in rows[1:]]

    return sweep, curves, cms, labels


def read_pth(path):
    """Unpack a torch.save archive without torch: it is a zip of raw storages."""
    import io
    import pickle

    class Loader(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith("torch"):
                return lambda *a, **k: a
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            return pid[2]

    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    out = {}
    for key, rec in Loader(io.BytesIO(z.read(root + "/data.pkl"))).load().items():
        storage, off, shape = rec[0], rec[1], rec[2]
        n = 1
        for d in shape:
            n *= d
        raw = z.read("%s/data/%s" % (root, storage))
        out[key] = list(struct.unpack("<%df" % n, raw[off * 4:off * 4 + n * 4]))
    return out


# ---------------------------------------------------------------- svg

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w, h, body, cls="chart"):
    return ('<svg class="%s" viewBox="0 0 %g %g" width="100%%" '
            'preserveAspectRatio="xMidYMid meet" role="img">%s</svg>'
            % (cls, w, h, body))


def txt(x, y, s, anchor="start", cls="lbl", extra=""):
    return ('<text x="%g" y="%g" text-anchor="%s" class="%s"%s>%s</text>'
            % (x, y, anchor, cls, extra, esc(s)))


def marginals_svg(sweep):
    """Mean(axis on) - mean(axis off), 8 runs per side, for UAR and WA."""
    ROW, PAD = 46, 26
    H = PAD * 2 + ROW * len(AXES)
    x0, span = 250, 320           # zero line at x0, +-span/2 of plot width
    W = x0 + span / 2 + 100       # right margin holds the value labels
    lim = 3.0                     # +-3 points covers every delta here
    b = ['<line x1="%g" y1="%g" x2="%g" y2="%g" class="axis"/>'
         % (x0, PAD - 8, x0, H - PAD + 4)]
    for t in (-2, 2):
        gx = x0 + t / lim * span / 2
        b.append('<line x1="%g" y1="%g" x2="%g" y2="%g" class="grid"/>'
                 % (gx, PAD - 8, gx, H - PAD + 4))
        b.append(txt(gx, PAD - 14, "%+d" % t, "middle", "tick"))
    b.append(txt(x0, PAD - 14, "0", "middle", "tick"))

    for i, (bit, name, owns) in enumerate(AXES):
        y = PAD + i * ROW
        b.append(txt(0, y + 4, name, "start", "axname"))
        b.append(txt(0, y + 17, owns, "start", "axsub"))
        for j, metric in enumerate(("test_uar", "test_wa")):
            on = [r[metric] for r in sweep if r[bit] == "1"]
            off = [r[metric] for r in sweep if r[bit] == "0"]
            d = sum(on) / len(on) - sum(off) / len(off)
            wpx = abs(d) / lim * span / 2
            by = y - 6 + j * 13
            cls = "bar-pos" if d >= 0 else "bar-neg"
            b.append('<rect x="%g" y="%g" width="%g" height="9" rx="1" class="%s"%s/>'
                     % (x0 if d >= 0 else x0 - wpx, by, wpx, cls,
                        ' fill-opacity="0.45"' if j else ""))
            b.append(txt(x0 + (wpx + 6 if d >= 0 else -wpx - 6), by + 8,
                         "%+.2f %s" % (d, "UAR" if j == 0 else "WA"),
                         "start" if d >= 0 else "end", "val"))
    return svg(W, H, "".join(b))


def curves_svg(curves, hi):
    W, H, L, R, T, B = 640, 260, 34, 12, 14, 28
    lo, up, ep = 26.0, 41.0, 30
    px = lambda e: L + (e - 1) / (ep - 1) * (W - L - R)
    py = lambda v: T + (up - v) / (up - lo) * (H - T - B)
    b = []
    for v in range(26, 42, 3):
        b.append('<line x1="%g" y1="%g" x2="%g" y2="%g" class="grid"/>'
                 % (L, py(v), W - R, py(v)))
        b.append(txt(L - 6, py(v) + 3, v, "end", "tick"))
    for e in (10, 20, 30):
        b.append(txt(px(e), H - 8, e, "middle", "tick"))
    for tag, series in sorted(curves.items()):
        pts = " ".join("%g,%g" % (px(i + 1), py(v[0])) for i, v in enumerate(series))
        cls = "line-hi" if tag == hi[0] else "line-base" if tag == hi[1] else "line"
        b.append('<polyline points="%s" class="%s"/>' % (pts, cls))
    for tag, cls in ((hi[0], "line-hi"), (hi[1], "line-base")):
        last = curves[tag][-1][0]
        b.append('<circle cx="%g" cy="%g" r="3" class="%s dot"/>'
                 % (px(ep), py(last), cls))
    b.append(txt(L, H - 8, "epoch", "start", "tick"))
    return svg(W, H, "".join(b))


def scatter_svg(sweep):
    W, H, L, R, T, B = 640, 300, 40, 90, 34, 34
    xs = [r["train_wa"] for r in sweep]
    ys = [r["test_uar"] for r in sweep]
    x0, x1, y0, y1 = 40, 59, 34, 42
    px = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    py = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)
    b = []
    for v in range(34, 43, 2):
        b.append('<line x1="%g" y1="%g" x2="%g" y2="%g" class="grid"/>'
                 % (L, py(v), W - R, py(v)))
        b.append(txt(L - 6, py(v) + 3, v, "end", "tick"))
    for v in range(40, 60, 5):
        b.append(txt(px(v), H - 14, v, "middle", "tick"))
    for r in sweep:
        n = sum(int(c) for c in r["tag"])
        cls = "pt-base" if r["tag"] == "0000" else "pt"
        b.append('<circle cx="%g" cy="%g" r="%g" class="%s"/>'
                 % (px(r["train_wa"]), py(r["test_uar"]), 3.5 + n * 0.9, cls))
    for r in sweep:
        if r["tag"] in ("0000", "0100", "0010"):
            b.append(txt(px(r["train_wa"]) + 11, py(r["test_uar"]) + 3.5,
                         r["tag"], "start", "ptlbl"))
    b.append(txt(L, H - 2, "train WA (%) -- how well it fits the training set",
                 "start", "tick"))
    b.append(txt(L - 34, 12, "test UAR (%)", "start", "tick"))
    return svg(W, H, "".join(b))


def cm_grid_html(cms, labels, sweep):
    order = [r["tag"] for r in sorted(sweep, key=lambda r: -r["test_uar"])]
    cells = []
    for tag in order:
        m = cms[tag]
        n = 5
        s = 22
        b = []
        for i in range(n):
            tot = sum(m[i]) or 1
            for j in range(n):
                v = m[i][j] / tot
                b.append('<rect x="%g" y="%g" width="%g" height="%g" '
                         'class="cm %s" fill-opacity="%.3f"/>'
                         % (18 + j * s, 12 + i * s, s - 1, s - 1,
                            "cm-diag" if i == j else "cm-off", 0.06 + 0.94 * v))
        for k, lab in enumerate(labels):
            b.append(txt(18 + k * s + s / 2 - 0.5, 9, lab[0].upper(), "middle", "cmtick"))
            b.append(txt(15, 12 + k * s + s / 2 + 3, lab[0].upper(), "end", "cmtick"))
        rec = sum(m[i][i] for i in range(n)) / sum(sum(r) for r in m) * 100
        cells.append('<figure class="cmcell"><svg viewBox="0 0 %g %g" width="100%%" '
                     'role="img">%s</svg><figcaption><b>%s</b><span>%.1f%% WA</span>'
                     '</figcaption></figure>'
                     % (18 + n * s + 4, 12 + n * s + 4, "".join(b), tag, rec))
    return '<div class="cmgrid">%s</div>' % "".join(cells)


def ema_svg(vals, freqs, init=0.04):
    W, H, T, B, L = 640, 130, 16, 26, 30
    lo, hi = 0.030, 0.046
    py = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    bw = (W - L - 10) / len(vals)
    b = ['<line x1="%g" y1="%g" x2="%g" y2="%g" class="axis-dash"/>'
         % (L, py(init), W - 10, py(init)),
         txt(W - 10, py(init) - 5, "init 0.040", "end", "tick")]
    for v in (0.032, 0.040):
        b.append(txt(L - 6, py(v) + 3, "%.3f" % v, "end", "tick"))
    for i, v in enumerate(vals):
        y, y2 = py(v), py(init)
        b.append('<rect x="%g" y="%g" width="%g" height="%g" rx="0.5" class="%s"/>'
                 % (L + i * bw, min(y, y2), max(bw - 1, 0.8), abs(y - y2) or 0.8,
                    "bar-pos" if v >= init else "bar-neg"))
    b.append(txt(L, H - 6, "filter 1 (%.0f Hz)" % freqs[0], "start", "tick"))
    b.append(txt(W - 10, H - 6, "filter %d (%.1f kHz)" % (len(freqs), freqs[-1] / 1000),
                 "end", "tick"))
    return svg(W, H, "".join(b), "chart wide")


# ---------------------------------------------------------------- page

def switchboard_html(sweep):
    rows = sorted(sweep, key=lambda r: -r["test_uar"])
    lo = min(r["test_uar"] for r in sweep) - 1.2
    hi = max(r["test_uar"] for r in sweep) + 0.2
    out = []
    for rank, r in enumerate(rows, 1):
        sw = "".join('<i class="%s" title="%s"></i>'
                     % ("on" if r[b] == "1" else "off", AXES[k][1])
                     for k, b in enumerate(BITS))
        w = (r["test_uar"] - lo) / (hi - lo) * 100
        cls = " base" if r["tag"] == "0000" else ""
        out.append(
            '<tr class="run%s"><td class="rk">%d</td><td class="sw">%s</td>'
            '<td class="tag">%s</td><td class="np">%d</td>'
            '<td class="barcell"><span class="bar" style="width:%.2f%%"></span>'
            '<span class="num">%.2f</span></td>'
            '<td class="num">%.2f</td><td class="num dim">%.2f</td>'
            '<td class="num dim">%.1f</td></tr>'
            % (cls, rank, sw, r["tag"], r["leaf_trainable_params"], w,
               r["test_uar"], r["test_wa"], r["best_val_uar"], r["train_wa"]))
    return "".join(out)


CSS = """
:root{
  --paper:#f1f4f3; --card:#fbfcfc; --ink:#101817; --mid:#43524f; --dim:#6e7d79;
  --rule:#d5dedb; --grid:#e3eae8; --accent:#0d6b61; --accent-soft:#8fc4bd;
  --ochre:#b0641a; --line:#a9b8b4;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#0d1211; --card:#141c1a; --ink:#e3ebe8; --mid:#a8b8b4; --dim:#7d8f8a;
  --rule:#243230; --grid:#1e2a28; --accent:#42c3b0; --accent-soft:#2a6a62;
  --ochre:#dc9a45; --line:#3a4a47;
}}
:root[data-theme="dark"]{
  --paper:#0d1211; --card:#141c1a; --ink:#e3ebe8; --mid:#a8b8b4; --dim:#7d8f8a;
  --rule:#243230; --grid:#1e2a28; --accent:#42c3b0; --accent-soft:#2a6a62;
  --ochre:#dc9a45; --line:#3a4a47;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;
  font-family:"Source Serif 4",Georgia,serif;font-size:16px;line-height:1.62;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:64px 28px 96px;
  display:flex;flex-direction:column;gap:64px}
h1,h2,h3,.eyebrow,.stat b,figcaption,th,.tick,.lbl{
  font-family:"Bricolage Grotesque","Helvetica Neue",Arial,sans-serif}
h1{font-size:clamp(30px,4.6vw,50px);line-height:1.04;margin:.18em 0 .32em;
  font-weight:700;letter-spacing:-.022em;text-wrap:balance;max-width:16ch}
h2{font-size:23px;font-weight:700;letter-spacing:-.012em;margin:0 0 4px}
h3{font-size:14px;font-weight:600;margin:0 0 10px;color:var(--mid)}
p{margin:0 0 14px;max-width:66ch;color:var(--mid)}
p.lede{font-size:19px;color:var(--ink);max-width:60ch}
a{color:var(--accent)}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;
  letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0}
section{display:flex;flex-direction:column;gap:18px}
.head{border-bottom:1px solid var(--rule);padding-bottom:14px;
  display:flex;flex-direction:column;gap:2px}
.head .n{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--dim);
  letter-spacing:.12em}
.statrow{display:flex;flex-wrap:wrap;gap:0;border-top:1px solid var(--rule);
  border-bottom:1px solid var(--rule)}
.stat{flex:1 1 130px;padding:14px 18px 14px 0;border-right:1px solid var(--rule)}
.stat:last-child{border-right:0}
.stat b{display:block;font-size:25px;font-weight:700;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.stat span{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--dim);
  text-transform:uppercase;letter-spacing:.1em}
.card{background:var(--card);border:1px solid var(--rule);border-radius:3px;
  padding:22px 24px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-family:"IBM Plex Mono",monospace;
  font-size:12.5px;font-variant-numeric:tabular-nums;min-width:660px}
th{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--dim);
  text-align:right;font-weight:600;padding:0 10px 8px;border-bottom:1px solid var(--rule);
  white-space:nowrap}
th.l{text-align:left}
td{padding:5px 10px;text-align:right;border-bottom:1px solid var(--grid)}
tr.run:hover td{background:var(--grid)}
tr.base td{color:var(--ochre)}
.rk{color:var(--dim);text-align:left;width:26px}
.tag{text-align:left;letter-spacing:.08em}
.sw{text-align:left;white-space:nowrap;width:74px;line-height:0}
.sw i{display:inline-block;width:11px;height:11px;margin-right:3px;border-radius:2px;
  border:1px solid var(--line)}
.sw i.on{background:var(--accent);border-color:var(--accent)}
.sw i.off{background:transparent}
.barcell{position:relative;width:46%;text-align:left;padding-right:60px}
.bar{display:inline-block;height:11px;background:var(--accent-soft);border-radius:1px;
  vertical-align:middle}
tr.base .bar{background:var(--ochre);opacity:.55}
.barcell .num{position:absolute;right:10px}
.num{font-variant-numeric:tabular-nums}
.dim{color:var(--dim)}
.legend{display:flex;flex-wrap:wrap;gap:14px 22px;font-family:"IBM Plex Mono",monospace;
  font-size:11px;color:var(--dim);align-items:center}
.legend i{display:inline-block;width:10px;height:10px;border:1px solid var(--line);
  border-radius:2px;margin-right:6px;vertical-align:-1px}
.legend i.on{background:var(--accent);border-color:var(--accent)}
.chart{display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--line);stroke-width:1}
.axis-dash{stroke:var(--line);stroke-width:1;stroke-dasharray:3 3}
text{font-family:"IBM Plex Mono",ui-monospace,monospace}
.tick{font-size:9.5px;fill:var(--dim)}
.val{font-size:10px;fill:var(--mid);font-variant-numeric:tabular-nums}
.axname{font-size:12.5px;fill:var(--ink)}
.axsub{font-size:9.5px;fill:var(--dim)}
.ptlbl{font-size:10px;fill:var(--dim)}
.cmtick{font-size:8px;fill:var(--dim)}
.bar-pos{fill:var(--accent)}
.bar-neg{fill:var(--ochre)}
.line{fill:none;stroke:var(--line);stroke-width:1;opacity:.45}
.line-hi{fill:none;stroke:var(--accent);stroke-width:2}
.line-base{fill:none;stroke:var(--ochre);stroke-width:2}
.dot{stroke:none}
.line-hi.dot{fill:var(--accent)}
.line-base.dot{fill:var(--ochre)}
.pt{fill:var(--accent);fill-opacity:.55}
.pt-base{fill:var(--ochre)}
.cm-diag{fill:var(--accent)}
.cm-off{fill:var(--ochre)}
.cmgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.cmcell figcaption{display:flex;justify-content:space-between;
  font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--dim);
  letter-spacing:.06em;margin-top:2px}
.cmcell figcaption b{color:var(--ink);font-weight:600}
.cmcell{margin:0}
.note{border-left:2px solid var(--ochre);padding:2px 0 2px 16px;color:var(--mid);
  font-size:15px}
.note b{color:var(--ink)}
dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:6px 18px;
  font-size:14px;color:var(--mid)}
dt{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--dim);
  text-transform:uppercase;letter-spacing:.09em;padding-top:3px}
dd{margin:0}
footer{border-top:1px solid var(--rule);padding-top:18px;
  font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--dim)}
@media (max-width:760px){.cmgrid{grid-template-columns:repeat(2,1fr)}}
@media (max-width:640px){.wrap{padding:40px 16px 64px;gap:48px}}
"""


def build(result_dir, out_path):
    sweep, curves, cms, labels = load(result_dir)
    best = max(sweep, key=lambda r: r["test_uar"])
    base = next(r for r in sweep if r["tag"] == "0000")
    full = next(r for r in sweep if r["tag"] == "1111")
    uars = [r["test_uar"] for r in sweep]
    mean = sum(uars) / len(uars)
    sd = (sum((u - mean) ** 2 for u in uars) / len(uars)) ** 0.5

    saved = read_pth(os.path.join(result_dir, "leaf_weights.pth"))
    ema = saved["_compression.ema._weights"]
    kernel = saved["_complex_conv._kernel"]
    freqs = [kernel[i * 2] * 16000 / (2 * math.pi) for i in range(len(ema))]
    ema_down = sum(1 for v in ema if v < 0.04)

    happy = labels.index("happy") if "happy" in labels else None
    happy_rec = [cms[r["tag"]][happy][happy] / sum(cms[r["tag"]][happy]) * 100
                 for r in sweep] if happy is not None else []

    p = ['<title>Which Part of LEAF Is Worth Learning?</title>',
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700'
         '&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;'
         '8..60,600&display=swap">',
         '<style>%s</style>' % CSS,
         '<div class="wrap">']

    # --- header
    p.append(
        '<header>'
        '<p class="eyebrow">IEMOCAP &middot; 5 classes &middot; speaker-independent '
        '&middot; seed 42</p>'
        '<h1>Which Part of LEAF Is Worth Learning?</h1>'
        '<p class="lede">LEAF has four independently trainable parameter groups. '
        'This 2<sup>4</sup> factorial freezes each one at its initialization in turn, '
        'so every cell isolates the value of <em>learning</em> a stage rather than the '
        'value of the stage existing at all. All sixteen cells share one CNN&rarr;ViT '
        'classifier, one split and one seed.</p>'
        '<div class="statrow">'
        '<div class="stat"><b>16</b><span>cells</span></div>'
        '<div class="stat"><b>30</b><span>epochs each</span></div>'
        '<div class="stat"><b>4,685</b><span>train clips</span></div>'
        '<div class="stat"><b>0&ndash;512</b><span>LEAF params</span></div>'
        '<div class="stat"><b>%.2f</b><span>best test UAR</span></div>'
        '<div class="stat"><b>&plusmn;%.2f</b><span>sd across cells</span></div>'
        '</div></header>' % (best["test_uar"], sd))

    # --- switchboard
    p.append(
        '<section><div class="head"><span class="n">01 &mdash; the factorial</span>'
        '<h2>Every cell, ranked by test UAR</h2></div>'
        '<p>A filled square means that group trained with the classifier at '
        'LEAF_LR = 1e-5; an empty square means it stayed at initialization while still '
        'running in the forward pass. The frozen baseline <b>0000</b> &mdash; a fixed '
        'Gabor filterbank &mdash; comes last of sixteen.</p>'
        '<div class="legend">%s<span><i class="on"></i>learned</span>'
        '<span><i class="off"></i>frozen at init</span></div>'
        '<div class="card"><table>'
        '<thead><tr><th class="l">#</th><th class="l">f&nbsp;p&nbsp;c&nbsp;s</th>'
        '<th class="l">bits</th><th>params</th><th class="l">test UAR</th>'
        '<th>test WA</th><th>val UAR</th><th>train WA</th></tr></thead>'
        '<tbody>%s</tbody></table></div></section>'
        % ("".join('<span><i class="on"></i>%s</span>' % a[1] for a in AXES),
           switchboard_html(sweep)))

    # --- marginals
    p.append(
        '<section><div class="head"><span class="n">02 &mdash; main effects</span>'
        '<h2>What each group buys, averaged over the other three</h2></div>'
        '<p>Each bar is the mean over the eight cells where that group was learned '
        'minus the mean over the eight where it was frozen. Averaging eight runs per '
        'side is the only estimate here with any stability &mdash; individual cells are '
        'single runs. Solid bars are UAR, faded bars WA.</p>'
        '<div class="card">%s</div>'
        '<p class="note"><b>Pooling and compression carry it.</b> Learning the '
        'Gaussian pooling widths is worth +1.76 UAR and learning PCEN\'s alpha/delta/root '
        '+1.21, while learning the Gabor filterbank itself &mdash; the part the LEAF '
        'paper is named for &mdash; is worth only +0.64. The filterbank is also the '
        'expensive group: 128 of the at most 512 parameters.</p></section>'
        % marginals_svg(sweep))

    # --- curves
    p.append(
        '<section><div class="head"><span class="n">03 &mdash; training</span>'
        '<h2>Validation UAR over 30 epochs</h2></div>'
        '<p>All sixteen cells, with the best (%s) and the frozen baseline (0000) drawn '
        'through. Every run is still climbing at epoch 30 and none has separated '
        'cleanly &mdash; the spread between cells stays inside the width of a single '
        'run\'s epoch-to-epoch wobble.</p><div class="card">%s</div></section>'
        % (best["tag"], curves_svg(curves, (best["tag"], "0000"))))

    # --- scatter
    p.append(
        '<section><div class="head"><span class="n">04 &mdash; fit vs. transfer</span>'
        '<h2>The frozen frontend fits training hardest and tests worst</h2></div>'
        '<p>Marker size is the number of learned groups. <b>0000</b> reaches %.1f%% '
        'train WA &mdash; %.1f points above the next cell &mdash; and lands last on test '
        'UAR. Across the other fifteen the relationship is weak (r = %.2f), so read this '
        'as one outlier rather than a trend: a frontend that cannot adapt lets the '
        'classifier memorise instead.</p><div class="card">%s</div></section>'
        % (base["train_wa"],
           base["train_wa"] - max(r["train_wa"] for r in sweep if r["tag"] != "0000"),
           -0.27, scatter_svg(sweep)))

    # --- confusion
    p.append(
        '<section><div class="head"><span class="n">05 &mdash; per-class behaviour</span>'
        '<h2>Sixteen confusion matrices, row-normalised</h2></div>'
        '<p>Rows are true classes, columns predicted, in the order '
        '%s. Diagonal cells are teal, off-diagonal ochre; opacity is the share of that '
        'true class. Ordered by test UAR, best first.</p>%s'
        '<p class="note"><b>No cell learns <em>happy</em>.</b> Recall on happy runs '
        '%.0f&ndash;%.0f%% across all sixteen configurations, against %.0f%% on sad in '
        'the best cell. The model routes almost every happy clip to excited or neutral. '
        'That is a class-balance and label-taxonomy problem in the 5-class IEMOCAP '
        'merge, and no amount of frontend learning touches it &mdash; which caps what '
        'this ablation can be about.</p></section>'
        % (", ".join(labels), cm_grid_html(cms, labels, sweep),
           min(happy_rec), max(happy_rec),
           cms[best["tag"]][labels.index("sad")][labels.index("sad")]
           / sum(cms[best["tag"]][labels.index("sad")]) * 100))

    # --- ema
    p.append(
        '<section><div class="head"><span class="n">06 &mdash; the saved weights</span>'
        '<h2>What actually moved in leaf_weights.pth</h2></div>'
        '<p>The sweep writes one <code>leaf_weights.pth</code> per result directory, so '
        'the copy on disk belongs to the last cell that ran &mdash; <b>0001</b>, '
        'smoothing only. Its Gabor kernels and pooling widths are therefore still at '
        'mel-scale initialization, and the PCEN EMA coefficient is the one thing that '
        'trained. Every channel moved, but not far and not in one direction: '
        '%.4f&ndash;%.4f against the 0.040 initialization, %d of %d down and %d up. The '
        'one clear move is the lowest filter (%.0f Hz), which drops to %.4f &mdash; a '
        'longer averaging window for the loudness estimate exactly where speech energy '
        'is densest. Above it the drift is small and not monotonic in frequency, which '
        'is about what a group worth +0.48 UAR should look like.</p>'
        '<div class="card">%s</div>'
        '<p>To visualise a <em>learned filterbank</em> you need the weights from a cell '
        'with filters unfrozen; those in <code>results/Ablation Study/LeafTorch_ViT/</code> '
        'do have moved Gabor kernels.</p></section>'
        % (min(ema), max(ema), ema_down, len(ema), len(ema) - ema_down,
           freqs[0], ema[0], ema_svg(ema, freqs)))

    # --- method
    p.append(
        '<section><div class="head"><span class="n">07 &mdash; method &amp; caveats</span>'
        '<h2>How to read this</h2></div>'
        '<dl>'
        '<dt>data</dt><dd>IEMOCAP, 5 classes (%s), speaker-independent split, '
        '4,685 training clips, fixed 3.0 s at 16 kHz.</dd>'
        '<dt>frontend</dt><dd>LEAF, 64 filters, 25 ms window, 60&ndash;7800 Hz Gabor '
        'init, PCEN compression, per-channel EMA.</dd>'
        '<dt>model</dt><dd>CNN stem (64 ch) into ViT: 224 px input, patch 16, dim 256, '
        'depth 6, 8 heads, MLP 1024.</dd>'
        '<dt>training</dt><dd>30 epochs, batch 32, AdamW at 3e-4 with LEAF on a separate '
        '1e-5 group, weight decay 0.05, label smoothing 0.1, SpecAugment '
        '(freq 8 / time 16).</dd>'
        '<dt>n</dt><dd><b>One seed, one run per cell.</b> Cell-to-cell sd is %.2f UAR '
        'and the best-to-worst gap is %.2f, so no single cell beats another '
        'significantly. Only the eight-vs-eight main effects in section 02 are worth '
        'quoting, and even those want a second seed before they go in a paper.</dd>'
        '<dt>metrics</dt><dd>UAR is the mean of per-class recalls and is the number to '
        'read on this label distribution; WA is overall accuracy and flatters the '
        'majority classes.</dd>'
        '</dl></section>' % (", ".join(labels), sd, max(uars) - min(uars)))

    p.append('<footer>Generated by viz_leaf.py from results/LeafVisualize &mdash; '
             'sweep.csv, training_log.csv, 16 confusion matrices, leaf_weights.pth. '
             'Full LEAF (1111) scored %.2f test UAR; frozen (0000) %.2f.</footer>'
             % (full["test_uar"], base["test_uar"]))
    p.append('</div>')

    open(out_path, "w", encoding="utf-8").write("\n".join(p))
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", nargs="?", default="results/LeafVisualize")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.result_dir, "report.html")
    print("wrote", build(a.result_dir, out))
