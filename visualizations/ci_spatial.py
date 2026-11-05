#!/usr/bin/env python
"""Spatial structure of grounding-line posterior uncertainty.

Produces two visualisations plus per-parameter maps:

  ci_spatial_maps.png      - GL coloured by CI width and each model parameter
  ci_uncertainty_band.png  - the credible interval drawn as a band along the GL

The CI width is dominated by the differential tide (a temporal driver), so a plain
average over acquisition pairs washes out any spatial pattern. Instead the maps condition
on the inferred tide: informative pairs (|diff_tides| >= STRONG_TIDE) and weak pairs
(|diff_tides| < WEAK_TIDE), each reduced to a per-flow-bend median. The input is the
retained pair-interval product, so every pair already satisfies the configured filters.

Usage
-----
    python -m gl_from_range_offsets.visualizations.ci_spatial --config config.yaml \
        --flowlines ... --pair-intervals ... --maps-out ... --band-out ...
"""
from __future__ import annotations

import argparse
import os
import pathlib
import warnings

import pyproj

os.environ.setdefault("PROJ_DATA", pyproj.datadir.get_data_dir())  # silence GDAL PROJ warning
warnings.filterwarnings("ignore")

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr
from shapely.geometry import Polygon

from gl_from_range_offsets.postprocessing import filtering
from gl_from_range_offsets.visualizations import style

# plot in EPSG:3031 (the data CRS) and draw only the graticule in geographic coordinates
PROJ = ccrs.SouthPolarStereo(central_longitude=0, true_scale_latitude=-71)
GEO = ccrs.PlateCarree()

# --------------------------------------------------------------------------- config
STRONG_TIDE = 1.5   # |dA| (m) above which a pair is "informative"
WEAK_TIDE = 0.5     # |dA| (m) below which a pair carries almost no flexure signal
MIN_PAIRS = 5       # minimum pairs per flow-bend within a stratum
GAP_M = 2_000       # break the GL into runs where consecutive points jump further than this
SMOOTH_LINES = 9    # rolling-median window (in flow-bends, 500 m apart) for the band edges

sns.set_theme(context="paper", style="ticks", font_scale=1.0)
style.use_style()  # after seaborn, so its rcParams win
# credible-interval colour pair: dark = 68% (inner), light = 95% (outer)
C68, C95 = "#8E0152", "#F768A1"


# ----------------------------------------------------------------------------- load
def load_lines(path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path, engine="pyogrio")[["Id", "geometry"]]
    g = g[g.geometry.notna()]
    g = g[g.geometry.length > 1_000]          # drop degenerate stubs
    return g.set_index("Id")


def load_ci(path, level: int, ness_min: float) -> pd.DataFrame:
    """Per (flow-bend, pair) credible intervals at `level` from the pair-interval product,
    with the absolute inferred differential tide as `absdA`."""
    df = pd.read_parquet(path)
    df = df.loc[df["level"].astype(int).eq(level)]
    df = df.rename(columns={"flowline_id": "Id", "primary_time": "p", "secondary_time": "s"})
    df["Id"] = df["Id"].astype(int)
    df["absdA"] = df["diff_tides"].abs()
    df = df.dropna(subset=["absdA", "width"])
    n0 = len(df)
    df = df.loc[filtering.converged_mask(df, ness_min)]
    print(f"  ci_{level}: {n0} rows -> {len(df)} after nESS>={ness_min} "
          f"({(df.width < filtering.COLLAPSED_CI_M).sum()} collapsed rows remain)")
    return df


# ------------------------------------------------------------------------ aggregate
def per_line(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the (flow-bend x pair) panel to one row per flow-bend."""
    strong = df[df.absdA >= STRONG_TIDE]
    weak = df[df.absdA < WEAK_TIDE]

    cols = {
        "w_strong": strong.groupby("Id")["width"].median(),
        "n_strong": strong.groupby("Id").size(),
        "w_weak": weak.groupby("Id")["width"].median(),
        "n_weak": weak.groupby("Id").size(),
        "left": strong.groupby("Id")["left"].median(),
        "right": strong.groupby("Id")["right"].median(),
        "left_weak": weak.groupby("Id")["left"].median(),
        "right_weak": weak.groupby("Id")["right"].median(),
        "mean_absdA": df.groupby("Id")["absdA"].mean(),
    }
    # model parameters / covariates (static or time-averaged). A covariate whose raster
    # was not readable when the intervals were written is simply absent from the export,
    # so aggregate only what is there and let the figure drop the matching panel.
    for name, how in (("ice_thickness", "median"),
                      ("ice_thickness_uncertainty", "median"),
                      ("error", "mean"),
                      ("correlation_coeff", "mean")):
        if name in df.columns:
            cols[name] = getattr(df.groupby("Id")[name], how)()
    missing = {"ice_thickness", "ice_thickness_uncertainty", "error",
               "correlation_coeff"} - set(cols)
    if missing:
        print(f"  covariates absent from the interval export, panels skipped: "
              f"{', '.join(sorted(missing))}")

    out = pd.DataFrame(cols)
    return out[(out.n_strong >= MIN_PAIRS) & (out.n_weak >= MIN_PAIRS)]


def attach_geometry(tab: pd.DataFrame, lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Locate each flow-bend at its inferred GL point (midpoint of the credible interval)."""
    tab = gpd.GeoDataFrame(tab.join(lines, how="inner"), geometry="geometry", crs=lines.crs)
    mid = (tab["left"] + tab["right"]) / 2.0
    keep = mid <= tab.geometry.length          # drop clipped lines whose CI lies past the end
    tab, mid = tab[keep], mid[keep]
    pts = [ln.interpolate(d) for ln, d in zip(tab.geometry, mid)]
    return gpd.GeoDataFrame(tab.drop(columns="geometry"), geometry=pts, crs=lines.crs)


def latlon_grid(ax, dlon: float = 1.0, dlat: float = 0.5, labels: bool = True) -> None:
    """Draw a geographic (EPSG:4326) graticule over an EPSG:3031 map, labelled on all four
    sides at `dlon`/`dlat` spacing."""
    gl = ax.gridlines(crs=GEO, draw_labels=labels, linewidth=0.4,
                      color="0.75", alpha=0.8, linestyle="--",
                      x_inline=False, y_inline=False, rotate_labels=False)
    gl.xlocator = mticker.MultipleLocator(dlon)
    gl.ylocator = mticker.MultipleLocator(dlat)
    gl.xlabel_style = {"size": style.GRATICULE_FONTSIZE}
    gl.ylabel_style = {"size": style.GRATICULE_FONTSIZE}


def map_extent(gdf: gpd.GeoDataFrame, pad: float = 15_000) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) in EPSG:3031 -- the order both cartopy's
    `set_extent` and the basemap reader expect."""
    x, y = gdf.geometry.x, gdf.geometry.y
    return (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad)


def set_extent(ax, gdf: gpd.GeoDataFrame, pad: float = 15_000) -> None:
    ax.set_extent(map_extent(gdf, pad), crs=PROJ)


def gl_runs(gdf: gpd.GeoDataFrame) -> list[np.ndarray]:
    """Split the (Id-ordered) GL into contiguous runs, breaking across inlets/segments."""
    gdf = gdf.sort_index()
    xy = np.c_[gdf.geometry.x, gdf.geometry.y]
    step = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    return np.split(np.arange(len(gdf)), np.where(step > GAP_M)[0])


def savefig(fig, savepath) -> None:
    """Write `fig` to `savepath` via `style.savefig`."""
    style.savefig(fig, savepath)


# -------------------------------------------------------------------------- fig one
def truncate(name: str, lo: float = 0.0, hi: float = 1.0):
    """Colormap `name` with its range restricted to [`lo`, `hi`], dropping the near-black
    and near-white ends that blend into the greyscale mosaic."""
    cmap = plt.get_cmap(name)
    return mcolors.LinearSegmentedColormap.from_list(
        f"{name}_{lo}_{hi}", cmap(np.linspace(lo, hi, 256)))


def letter(ax, tag: str) -> None:
    """Panel tag in the upper-left corner, in place of a title."""
    ax.text(0.025, 0.975, tag, transform=ax.transAxes, ha="left", va="top",
            fontsize=style.FONTSIZE, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="0.6", lw=0.6, alpha=0.9))


def figure_maps(gdf: gpd.GeoDataFrame, panel: pd.DataFrame, savepath) -> None:
    """GL coloured by CI width (tide-conditioned) and by each model parameter, saved to
    `savepath`. Panels (a)-(c) share one log-scale colorbar drawn beside (c)."""
    norm = LogNorm(vmin=120, vmax=2000)
    ci_cmap = "viridis"
    par_cmap = truncate("plasma", 0.05, 0.92)
    cov_cmap = truncate("magma", 0.12, 0.90)
    specs = [
        ("w_strong", ci_cmap, norm, None),                 # colorbar shared, drawn at (c)
        ("w95_strong", ci_cmap, norm, None),
        ("w_weak", ci_cmap, norm, "CI width (m)"),
        ("ice_thickness", par_cmap, None, "$h$ (m)"),
        ("ice_thickness_uncertainty", par_cmap, None, r"$\sigma_h$ (m)"),
        ("mean_absdA", cov_cmap, None, r"$|\Delta A|$ (m)"),
        ("error", cov_cmap, None, r"$\sigma$ (m)"),
    ]
    # a covariate missing from the export gets no panel, and the rest are relettered so
    # the sequence stays gapless
    specs = [s for s in specs if s[0] in gdf.columns and gdf[s[0]].notna().any()]
    specs = [(f"({chr(ord('a') + i)})",) + s for i, s in enumerate(specs)]

    # axes placed by hand: the layout engines shrink each geo-axis to the projection aspect,
    # so a colorbar panel ends up smaller; explicit rectangles keep all seven maps identical
    extent = map_extent(gdf)
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])   # dy / dx
    FIGW, FIGH = 16.0, 14.5
    MAPW = 4.0                       # inches
    MAPH = MAPW * aspect
    CBW, CBPAD = 0.16, 0.12          # colorbar width and its gap to the map
    LEFT, BOTTOM = 0.55, 0.55        # outer margins (inches)
    HGAP, VGAP = 1.35, 0.75          # room for the colorbar + its label / for the graticule

    fig = plt.figure(figsize=(FIGW, FIGH))
    fig.set_layout_engine("none")

    def rect(r, c, w, h, dx=0.0):
        x = LEFT + c * (MAPW + HGAP) + dx
        y = FIGH - BOTTOM - (r + 1) * MAPH - r * VGAP
        return [x / FIGW, y / FIGH, w / FIGW, h / FIGH]

    cells = [(r, c) for r in range(3) for c in range(3)]
    for (tag, col, cmap, nrm, lab), (r, c) in zip(specs, cells):
        ax = fig.add_axes(rect(r, c, MAPW, MAPH), projection=PROJ)
        sub = gdf[gdf[col].notna()]
        kw = dict(norm=nrm) if nrm is not None else dict(
            vmin=np.nanpercentile(sub[col], 2), vmax=np.nanpercentile(sub[col], 98))
        ax.set_facecolor("white")
        # mosaic washed towards white so the markers carry the ink, embedded at panel size
        style.add_basemap(ax, extent, PROJ, alpha=0.55,
                          max_px=int(MAPW * 300), interpolation="none")
        sc = ax.scatter(sub.geometry.x, sub.geometry.y, c=sub[col], s=6,
                        cmap=cmap, transform=PROJ, linewidths=0, zorder=2, **kw)
        ax.set_extent(extent, crs=PROJ)
        letter(ax, tag)
        if lab is not None:
            cax = fig.add_axes(rect(r, c, CBW, MAPH, dx=MAPW + CBPAD))
            fig.colorbar(sc, cax=cax, label=lab)

    # final panel, interaction: the thickness control only appears once the flexure
    # signal is strong. Needs ice thickness, so it goes only when that covariate is there.
    if panel.empty:
        savefig(fig, savepath)
        return
    prow, pcol = cells[len(specs)]
    ax = fig.add_axes(rect(prow, pcol, 2 * MAPW + HGAP - 1.2, MAPH, dx=0.9))
    ax.plot(panel["centre"], panel["rho"], "o-", color=C68, lw=2, ms=7)
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    for c, r, n in zip(panel["centre"], panel["rho"], panel["n"]):
        ax.annotate(f"n={n // 1000}k" if n >= 1000 else f"n={n}", (c, r), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=style.FONTSIZE, color="0.45")
    ax.set_xlabel(r"$|\Delta A|$ (m)")
    ax.set_ylabel(r"Spearman $\rho$ (CI width, $h$)")
    ax.grid(alpha=0.3); sns.despine(ax=ax)
    letter(ax, f"({chr(ord('a') + len(specs))})")

    savefig(fig, savepath)


# -------------------------------------------------------------------------- fig two
def _band(gdf: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, lcol: str, rcol: str) -> list[Polygon]:
    polys = []
    for run in gl_runs(gdf):
        if len(run) < 10:
            continue
        sub = gdf.sort_index().iloc[run]
        lo = sub[lcol].rolling(SMOOTH_LINES, center=True, min_periods=1).median()
        hi = sub[rcol].rolling(SMOOTH_LINES, center=True, min_periods=1).median()
        geom = lines.loc[sub.index, "geometry"]
        left = [ln.interpolate(min(d, ln.length)) for ln, d in zip(geom, lo)]
        right = [ln.interpolate(min(d, ln.length)) for ln, d in zip(geom, hi)]
        ring = [(p.x, p.y) for p in left] + [(p.x, p.y) for p in reversed(right)]
        if len(ring) > 3:
            polys.append(Polygon(ring).buffer(0))
    return polys


def figure_band(gdf: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, ci95: pd.DataFrame,
                savepath) -> None:
    """Draw the credible interval as a band hugging the grounding line, saved to
    `savepath`. The overview locates the zoom windows; the bands are shown in the zoom
    panels, where they are wide enough to see."""
    g = gdf.join(ci95[["left", "right", "left_weak", "right_weak"]].rename(
        columns={"left": "l95", "right": "r95",
                 "left_weak": "l95w", "right_weak": "r95w"}), how="inner")

    bands = {  # (95% band, 68% band) for each tide stratum
        "strong": (gpd.GeoSeries(_band(g, lines, "l95", "r95"), crs=lines.crs),
                   gpd.GeoSeries(_band(g, lines, "left", "right"), crs=lines.crs)),
        "weak": (gpd.GeoSeries(_band(g, lines, "l95w", "r95w"), crs=lines.crs),
                 gpd.GeoSeries(_band(g, lines, "left_weak", "right_weak"), crs=lines.crs)),
    }

    # One dense contiguous run; both zooms share it so the tide contrast is like-for-like.
    srt = gdf.sort_index()
    runs = [r for r in gl_runs(gdf) if len(r) >= 40]
    run = max(runs, key=len)
    centre = srt.iloc[run[len(run) // 2]].geometry
    zooms = [("strong", f"informative pairs |$\\Delta A$| $\\geq$ {STRONG_TIDE} m "
                        f"(median 68% CI = {gdf.w_strong.median():.0f} m)"),
             ("weak", f"weak pairs |$\\Delta A$| < {WEAK_TIDE} m "
                      f"(median 68% CI = {gdf.w_weak.median():.0f} m)")]
    HALF = 6_000  # m

    fig = plt.figure(figsize=(17, 9), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])

    overview = map_extent(gdf)
    ax0 = fig.add_subplot(gs[:, 0], projection=PROJ)
    style.add_basemap(ax0, overview, PROJ)
    sc = ax0.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf["w_strong"], s=3,
                     cmap="viridis", norm=LogNorm(vmin=120, vmax=2000),
                     transform=PROJ, linewidths=0, zorder=2)
    fig.colorbar(sc, ax=ax0, shrink=0.6, label="68% CI width (m)")
    ax0.add_patch(plt.Rectangle((centre.x - HALF, centre.y - HALF), 2 * HALF, 2 * HALF,
                                fill=False, ec="crimson", lw=1.6, transform=PROJ, zorder=5))
    ax0.set_title(f"Overview: GL coloured by 68% CI width\n(informative pairs, "
                  f"|$\\Delta A$| $\\geq$ {STRONG_TIDE} m)")
    ax0.set_extent(overview, crs=PROJ)
    latlon_grid(ax0)

    zoom_extent = (centre.x - HALF, centre.x + HALF, centre.y - HALF, centre.y + HALF)
    for k, (stratum, ttl) in enumerate(zooms):
        ax = fig.add_subplot(gs[k, 1], projection=PROJ)
        b95, b68 = bands[stratum]
        style.add_basemap(ax, zoom_extent, PROJ)
        ax.add_geometries(list(b95), crs=PROJ, facecolor=C95, edgecolor="none", alpha=0.60)
        ax.add_geometries(list(b68), crs=PROJ, facecolor=C68, edgecolor="none", alpha=0.90)
        ax.scatter(gdf.geometry.x, gdf.geometry.y, c="0.15", s=3, transform=PROJ,
                   linewidths=0, zorder=3)
        ax.set_extent(zoom_extent, crs=PROJ)
        ax.spines["geo"].set_edgecolor("crimson")
        ax.spines["geo"].set_linewidth(1.6)
        ax.set_title(ttl)
        latlon_grid(ax, dlon=0.1, dlat=0.05)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=C95, alpha=0.60),
               plt.Rectangle((0, 0), 1, 1, fc=C68, alpha=0.90),
               plt.Line2D([], [], color="0.15", marker="o", ls="", ms=6)]
    fig.legend(handles, ["95% credible band", "68% credible band", "GL (CI midpoint)"],
               loc="outside lower center", ncol=3, frameon=False)

    fig.suptitle("Grounding line with its 68% and 95% credible bands "
                 "(bands are only resolvable at zoom scale)")
    style.savefig(fig, savepath)


# ------------------------------------------------------------------------ main
def interaction_curve(df: pd.DataFrame) -> pd.DataFrame:
    edges = [0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3.1]
    rows = []
    if "ice_thickness" not in df.columns:
        return pd.DataFrame(columns=["centre", "rho", "n"])   # no thickness, no curve
    d = df.dropna(subset=["ice_thickness"])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = d[(d.absdA >= lo) & (d.absdA < hi)]
        if len(m) > 500:
            rows.append({"centre": (lo + hi) / 2,
                         "rho": spearmanr(m["width"], m["ice_thickness"]).statistic,
                         "n": len(m)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    ap.add_argument("--flowlines", type=pathlib.Path, required=True)
    ap.add_argument("--pair-intervals", type=pathlib.Path, required=True)
    ap.add_argument("--maps-out", type=pathlib.Path, required=True)
    ap.add_argument("--band-out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    import yaml

    with args.config.open() as stream:
        ness_min, _ = filtering.thresholds(yaml.safe_load(stream))
    lines = load_lines(args.flowlines)
    d68 = load_ci(args.pair_intervals, 68, ness_min)
    d95 = load_ci(args.pair_intervals, 95, ness_min)

    t68, t95 = per_line(d68), per_line(d95)
    gdf = attach_geometry(t68, lines)
    gdf["w95_strong"] = t95["w_strong"].reindex(gdf.index)

    n_bad = int((gdf.w_strong < filtering.COLLAPSED_CI_M).sum())
    print(f"flow-bends mapped: {len(gdf)}  (collapsed posteriors remaining: {n_bad})")
    print(f"median 68% CI width: informative {gdf.w_strong.median():.0f} m | "
          f"weak {gdf.w_weak.median():.0f} m ({gdf.w_weak.median()/gdf.w_strong.median():.1f}x wider)")

    panel = interaction_curve(d68)
    print(panel.to_string(index=False))

    if "ice_thickness" in gdf.columns:
        pair = d68.dropna(subset=["absdA", "ice_thickness"])
        line = gdf[["mean_absdA", "ice_thickness"]].dropna()
        print(f"rho(tide, h): per (flow-bend, pair) "
              f"{spearmanr(pair.absdA, pair.ice_thickness).statistic:+.3f} "
              f"(n={len(pair)}) | per flow-bend "
              f"{spearmanr(line.mean_absdA, line.ice_thickness).statistic:+.3f} "
              f"(n={len(line)})")

    for p in (args.maps_out, args.band_out):
        p.parent.mkdir(parents=True, exist_ok=True)
    figure_maps(gdf, panel, args.maps_out)
    figure_band(gdf, lines, t95, args.band_out)

if __name__ == "__main__":
    main()
