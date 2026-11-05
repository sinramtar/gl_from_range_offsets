"""Inferred grounding-line points map: points coloured by acquisition year, the
cumulative credible intervals as nested polygons, and temporal coverage.

Usage
-----
    python -m gl_from_range_offsets.visualizations.gl_points_map --config config.yaml \
        --postprocessing-dir ... --reference-gl ... --flow-bends ... --dinsars-dir ... \
        --radar-mosaic ... --figures-dir ...
"""

import argparse
import pathlib
import re

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import rasterio as rio
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib_map_utils.core.scale_bar import ScaleBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

CRS_3031 = ccrs.SouthPolarStereo(true_scale_latitude=-71)
CRS_4326 = ccrs.PlateCarree()
# navy / vermillion / dark green, colour-vision-deficiency safe and distinct from the mosaic
YEAR_COLORS = {2019: "#4426D6", 2020: "#F77008", 2021: "#10943C"}
# two shades of one hue so the nesting of the intervals reads at a glance
CI_68_COLOR = "#8E0152"   # dark magenta, inner bound
CI_95_COLOR = "#F768A1"   # bright pink, outer bound
# grey band the SAR mosaic is compressed into, so the 1-pt points stay readable over it
MOSAIC_GREY_RANGE = (0.2, 0.80)
DINSAR_DATE = re.compile(r"(\d{8})")

# type sizes: axis labels / legends / panel letters, then lon-lat labels, scale bar,
# ticks and DInSAR-marker annotations
LABEL_FS, LEGEND_FS, PANEL_FS = 16, 14, 16
TICK_FS, GRID_FS, SCALE_FS, ANNOT_FS = 14, 11, 9, 13


def load_points(path):
    """Read the paper point product with parsed time columns."""
    df = gpd.read_parquet(path)
    for col in ("primary_time", "secondary_time"):
        df[col] = pd.to_datetime(df[col])
    return df


def ci_envelope(path, flowlines, level, max_gap = 1500, smooth=9):
    """Landward and seaward bounds of the cumulative credible interval, as polygon runs.

    Each row gives one flow-bend line's interval as two arc-length positions. Runs are
    cut where consecutive lines are more than `max_gap` apart geographically, and the
    arc-length bounds are smoothed with a rolling median over `smooth` lines (a display
    choice that does not touch the reported widths) before being mapped back to the geometry.
    Returns a list of (landward, seaward) coordinate arrays, one per contiguous run, and the
    unsmoothed per-flow-bend interval widths."""
    rows = pd.read_parquet(path)
    rows = rows.loc[rows["level"].astype(int).eq(level)]
    lo, hi, width = "left", "right", "width"
    rows = rows.rename(columns={"flowline_id": "Id"})
    rows = rows.dropna(subset=[lo, hi, width])
    rows = rows.merge(flowlines[["Id", "geometry"]], on="Id").sort_values("Id")

    # cut where consecutive lines are geographically apart, by their midpoint positions
    mid = np.array([[g.interpolate(g.length / 2).x, g.interpolate(g.length / 2).y]
                    for g in rows["geometry_y"]])
    apart = np.linalg.norm(np.diff(mid, axis=0), axis=1) > max_gap
    run_id = np.concatenate([[0], np.cumsum(apart)])

    runs = []
    for _, block in rows.groupby(run_id):
        if len(block) < 2:
            continue
        arc_lo = block[lo].rolling(smooth, center=True, min_periods=1).median()
        arc_hi = block[hi].rolling(smooth, center=True, min_periods=1).median()
        landward, seaward = [], []
        for geom, a, b in zip(block["geometry_y"], arc_lo, arc_hi):
            p, q = geom.interpolate(a), geom.interpolate(b)
            landward.append([p.x, p.y])
            seaward.append([q.x, q.y])
        runs.append((np.array(landward), np.array(seaward)))

    return runs, rows[width].to_numpy()


def read_mosaic(path, bounds, max_px=2200):
    """Windowed read of the SAR mosaic over `bounds`, downsampled to at most `max_px` on the
    long side and rescaled into the muted display grey band. Returns (array, extent)."""
    with rio.open(path) as src:
        window = rio.windows.from_bounds(*bounds, transform=src.transform)
        scale = max(window.height, window.width) / max_px
        out_shape = (max(1, int(window.height / scale)), max(1, int(window.width / scale)))
        arr = src.read(1, window=window, out_shape=out_shape,
                       resampling=rio.enums.Resampling.average).astype(float)
    arr[arr == 0] = np.nan
    # robust normalisation, then compression into the muted display band; the imshow
    # calls pass vmin=0, vmax=1 so autoscaling does not stretch the greys back out
    lo, hi = np.nanpercentile(arr, [2, 98])
    g0, g1 = MOSAIC_GREY_RANGE
    arr = g0 + (g1 - g0) * np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr, [bounds[0], bounds[2], bounds[1], bounds[3]]


def dinsar_scene_dates(dinsar_dir):
    """Primary date and mission of every DInSAR scene, for the coverage panel."""
    rows = []
    for f in sorted(pathlib.Path(dinsar_dir).glob("*.geojson")):
        dates = DINSAR_DATE.findall(f.stem)
        if not dates:
            continue
        mission = "CSK" if f.stem.startswith("CSK") else "S1"
        rows.append({"mission": mission, "date": pd.to_datetime(dates[0], format="%Y%m%d")})
    return pd.DataFrame(rows)


def coverage_table(points):
    """Retained GL estimates per acquisition pair."""
    pairs = (points.groupby(["primary_time", "secondary_time"])
                   .size()
                   .rename("n_points")
                   .reset_index())
    pairs["year"] = pairs["primary_time"].dt.year
    return pairs.sort_values("primary_time")


def draw_ref_gl(ax, gdf, linewidth, zorder, halo=True):
    """Draw the reference GL on `ax`. `halo` adds a white stroke, used in the insets where
    the line runs through the point cloud."""
    halo = [pe.withStroke(linewidth=linewidth + 1.0, foreground="white")] if halo else None
    for geom in gdf.geometry:
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            x, y = part.coords.xy
            ax.plot(x, y, color="black", linewidth=linewidth, zorder=zorder,
                    path_effects=halo)


def draw_envelope(ax, runs, facecolor, alpha, zorder, edge=None, linewidth=0.0, dashed=True):
    """Draw the credible interval on `ax` as a filled ribbon between its landward and
    seaward bounds, optionally outlined with `edge`."""
    style = (0, (4, 2)) if dashed else "solid"
    for left, right in runs:
        ring = np.vstack([left, right[::-1]])
        ax.fill(ring[:, 0], ring[:, 1], facecolor=facecolor, alpha=alpha,
                edgecolor="none", zorder=zorder, rasterized=True)
        if edge is not None:
            # above the points: band and point cloud are the same width, so lower bounds hide
            for bound in (left, right):
                ax.plot(bound[:, 0], bound[:, 1], color=edge, linewidth=linewidth,
                        linestyle=style, zorder=5,
                        path_effects=[pe.withStroke(linewidth=linewidth + 0.6,
                                                    foreground="white")])


def widest_ci_bbox(path, flowlines, window=20, half_width=13_000.0):
    """Bounding box centred on the stretch where the credible interval is widest."""
    rows = pd.read_parquet(path)
    rows = rows.loc[rows["level"].astype(int).eq(95)]
    width = "width"
    rows = rows.rename(columns={"flowline_id": "Id"})
    rows = rows.dropna(subset=[width])
    rows = rows.merge(flowlines[["Id", "geometry"]], on="Id").sort_values("Id")
    smoothed = rows[width].rolling(window, center=True, min_periods=window // 4).mean()
    geom = rows.loc[smoothed.idxmax(), "geometry_y"]
    c = geom.interpolate(geom.length / 2)
    return (c.x - half_width, c.y - half_width, c.x + half_width, c.y + half_width)


def add_inset(ax, bounds, mosaic_path, anchor, edgecolor, draw):
    """A zoom panel on `ax`, with its extent outlined on the parent map."""
    minx, miny, maxx, maxy = bounds
    ax.add_patch(mpl.patches.Rectangle((minx, miny), maxx - minx, maxy - miny,
                                       linewidth=1.2, edgecolor=edgecolor,
                                       facecolor="none", zorder=10))
    axin = inset_axes(ax, width="42%", height="42%", loc="upper right",
                      axes_class=GeoAxes, axes_kwargs=dict(projection=CRS_3031),
                      bbox_to_anchor=anchor, bbox_transform=ax.transAxes)
    mosaic, extent = read_mosaic(mosaic_path, bounds, max_px=1000)
    axin.imshow(mosaic, cmap="gray", vmin=0.0, vmax=1.0, extent=extent, zorder=0)
    axin.set_extent([minx, maxx, miny, maxy], crs=CRS_3031)
    draw(axin, bounds)
    for spine in axin.spines.values():
        spine.set_edgecolor(edgecolor)
        spine.set_linewidth(1.2)
    return axin


def map_panel(fig, spec, extent, mosaic, mosaic_extent, label):
    ax = fig.add_subplot(spec, projection=CRS_3031)
    ax.set_facecolor('black')
    ax.set_alpha(0.6)
    ax.set_extent(extent, crs=CRS_3031)
    ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, linestyle="--", linewidth=0.5,
                 color="dimgray", crs=CRS_4326, dms=True, x_inline=False, y_inline=False,
                 xlabel_style={"size": GRID_FS, "color": "gray"},
                 ylabel_style={"size": GRID_FS, "color": "gray"},
                 xpadding=3, ypadding=3, rotate_labels=False)
    ax.imshow(mosaic, cmap="gray", vmin=0.0, vmax=1.0, extent=mosaic_extent, zorder=0)
    ax.add_artist(ScaleBar(
        location="lower right", style="boxes",
        bar={"projection": 3031, "minor_type": "none", "height": 0.05,
             "major_div": 2, "max": 50},
        labels={"style": "minor_first", "loc": "above", "pad": 0},
        units={"loc": "text", "pad": 0},
        text={"stroke_color": "black", "textcolor": "black", "fontsize": SCALE_FS,
              "stroke_width": 0.3, "fontweight": "light", "fontfamily": "sans-serif"}))
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=PANEL_FS, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="square", fc="white", alpha=0.7, linewidth=0))
    return ax


def make_figure(points, ref_gl, mosaic, mosaic_extent, extent, mosaic_path, inset_boxes,
                dinsar, ci68, ci68_w, ci95, ci95_w, out_stem):
    pairs = coverage_table(points)
    years = sorted(YEAR_COLORS)

    fig = plt.figure(figsize=(15, 10.5), dpi=300)
    gs = mpl.gridspec.GridSpec(2, 2, figure=fig, height_ratios=[3.3, 1.0],
                               hspace=0.16, wspace=0.08)

    # (a) points and (b) intervals in separate panels: they are the same width, so on one
    # map whichever is drawn on top hides the other
    ax_p = map_panel(fig, gs[0, 0], extent, mosaic, mosaic_extent, "(a)")
    shuffled = points.sample(frac=1.0, random_state=0)  # so no year is buried under the last
    ax_p.scatter(shuffled.geometry.x, shuffled.geometry.y, s=2.0, alpha=0.85, linewidths=0,
                 color=[YEAR_COLORS[y] for y in shuffled["primary_time"].dt.year], zorder=3,
                 rasterized=True)
    # no halo at shelf scale: the white stroke would bleach whole stretches of the GL
    draw_ref_gl(ax_p, ref_gl, linewidth=0.5, zorder=4, halo=False)
    ax_p.legend(handles=[mpl.lines.Line2D([], [], marker="o", linestyle="none", markersize=5,
                                          color=YEAR_COLORS[y], label=str(y)) for y in years]
                + [mpl.lines.Line2D([], [], color="black", linewidth=0.8,
                                    label="Reference GL")],
                loc="lower left", fontsize=LEGEND_FS, framealpha=0.85,
                title="Acquisition year", title_fontsize=LEGEND_FS)

    ax_c = map_panel(fig, gs[0, 1], extent, mosaic, mosaic_extent, "(b)")
    draw_envelope(ax_c, ci95, facecolor=CI_95_COLOR, alpha=0.9, zorder=2)
    draw_envelope(ax_c, ci68, facecolor=CI_68_COLOR, alpha=0.9, zorder=2.1)
    draw_ref_gl(ax_c, ref_gl, linewidth=0.4, zorder=4, halo=False)
    ax_c.legend(handles=[mpl.patches.Patch(facecolor=CI_68_COLOR, edgecolor="none",
                                           label="Cumulative 68% CI"),
                         mpl.patches.Patch(facecolor=CI_95_COLOR, edgecolor="none",
                                           label="Cumulative 95% CI"),
                         mpl.lines.Line2D([], [], color="black", linewidth=0.8,
                                          label="Reference GL")],
                loc="lower left", fontsize=LEGEND_FS, framealpha=0.85)

    def draw_points(axin, bounds):
        sel = points.clip(bounds).sample(frac=1.0, random_state=0)
        axin.scatter(sel.geometry.x, sel.geometry.y, s=5.0, alpha=0.8, linewidths=0,
                     color=[YEAR_COLORS[y] for y in sel["primary_time"].dt.year], zorder=3,
                     rasterized=True)
        draw_ref_gl(axin, ref_gl.clip(bounds), linewidth=0.9, zorder=4)

    def draw_intervals(axin, bounds):
        draw_envelope(axin, ci95, facecolor=CI_95_COLOR, alpha=0.9, zorder=2)
        draw_envelope(axin, ci68, facecolor=CI_68_COLOR, alpha=0.9, zorder=2.1)
        draw_ref_gl(axin, ref_gl.clip(bounds), linewidth=0.6, zorder=4, halo=False)

    # two zooms per map at the same places: the reference region and the widest-95%-CI stretch
    anchors = [(0.0, 0.0, 0.99, 0.99), (0.0, 0.0, 0.99, 0.56)]
    for ax, draw in ((ax_p, draw_points), (ax_c, draw_intervals)):
        for (bounds, edge), anchor in zip(inset_boxes, anchors):
            add_inset(ax, bounds, mosaic_path, anchor, edge, draw)

    # (c) temporal coverage
    axc = fig.add_subplot(gs[1, :])
    for year in years:
        sel = pairs[pairs["year"] == year]
        axc.bar(sel["primary_time"], sel["n_points"], width=5.0,
                color=YEAR_COLORS[year], linewidth=0)
    axc.set_ylabel("Retained GL\nestimates per pair", fontsize=LABEL_FS)
    axc.set_xlabel("Acquisition time", fontsize=LABEL_FS)
    axc.tick_params(axis="both", labelsize=TICK_FS)
    axc.margins(x=0.01)
    axc.grid(axis="y", linestyle=":", linewidth=0.5, color="lightgray")
    axc.set_axisbelow(True)
    for spine in ("top", "right"):
        axc.spines[spine].set_visible(False)

    # clip to the 2019-2021 range-offset record; all_dinsar also holds later scenes
    t0, t1 = pd.Timestamp("2018-12-01"), pd.Timestamp("2022-02-01")
    axc.set_xlim(t0, t1)
    dinsar = dinsar[(dinsar["date"] >= t0) & (dinsar["date"] <= t1)]

    ymax = pairs["n_points"].max()
    axc.set_ylim(0, ymax * 1.34)
    for mission, color, offset in (("S1", "dimgray", 1.26), ("CSK", "maroon", 1.13)):
        sel = dinsar[dinsar["mission"] == mission]
        axc.plot(sel["date"], np.full(len(sel), ymax * offset), "v", markersize=3.5,
                 color=color, linestyle="none", clip_on=False)
        axc.text(t1 - pd.Timedelta(days=8), ymax * offset, f" DInSAR {mission}",
                 fontsize=ANNOT_FS, color=color, va="center", ha="left")
    axc.text(0.012, 0.70, "(c)", transform=axc.transAxes, fontsize=PANEL_FS, fontweight="bold",
             va="top", ha="left")

    # the map axes shrink to their data aspect; match the coverage panel to their edges
    fig.canvas.draw()
    left = ax_p.get_position().x0
    right = ax_c.get_position().x1
    box = axc.get_position()
    axc.set_position([left, box.y0, right - left, box.height])

    # 300 dpi affects only the rasterized layers (basemap, points, bands); text stays vector
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    n_pairs = len(pairs)
    print(f"acquisition pairs retained : {n_pairs}")
    print(f"GL estimates               : {len(points)}")
    print(f"flow-bend lines            : {points['flowline_id'].nunique()}")
    print(f"GL estimates per pair      : median {pairs['n_points'].median():.0f}")
    print("per year (pairs / points)  : " + ", ".join(
        f"{y}: {len(pairs[pairs.year == y])} / {len(points[points.primary_time.dt.year == y])}"
        for y in years))
    print(f"DInSAR scenes              : S1 {(dinsar.mission == 'S1').sum()}, "
          f"CSK {(dinsar.mission == 'CSK').sum()}")
    print(f"cumulative CI width        : 68% median {np.median(ci68_w):.0f} m, "
          f"95% median {np.median(ci95_w):.0f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    ap.add_argument("--postprocessing-dir", type=pathlib.Path, required=True)
    ap.add_argument("--reference-gl", type=pathlib.Path, required=True)
    ap.add_argument("--flow-bends", type=pathlib.Path, required=True)
    ap.add_argument("--dinsars-dir", type=pathlib.Path, required=True)
    ap.add_argument("--radar-mosaic", type=pathlib.Path, required=True)
    ap.add_argument("--figures-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    import yaml

    config_path = args.config.resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    points = load_points(args.postprocessing_dir / "paper_points.parquet")
    ref_gl = gpd.read_file(args.reference_gl).to_crs("EPSG:3031")
    flowlines = gpd.read_file(args.flow_bends).to_crs("EPSG:3031")
    dinsar = dinsar_scene_dates(args.dinsars_dir)

    # extent from the reference GL, not the ROI, which is mostly empty open shelf
    pad = 12_000.0
    b = ref_gl.total_bounds
    bounds = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    cumulative = args.postprocessing_dir / "cumulative_intervals.parquet"
    ci68, ci68_w = ci_envelope(cumulative, flowlines, 68)
    ci95, ci95_w = ci_envelope(cumulative, flowlines, 95)

    mosaic_path = args.radar_mosaic
    mosaic, mosaic_extent = read_mosaic(mosaic_path, bounds)

    crane = gpd.read_file(pathlib.Path(__file__).resolve().parents[1] / "rois" / "Crane.geojson").to_crs("EPSG:3031").total_bounds
    widest = widest_ci_bbox(cumulative, flowlines)
    inset_boxes = [(tuple(crane), "maroon"), (widest, "#0072B2")]

    out = args.figures_dir / "filtered_gl_points"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(points, ref_gl, mosaic, mosaic_extent, extent, mosaic_path, inset_boxes,
                dinsar, ci68, ci68_w, ci95, ci95_w, out)


if __name__ == "__main__":
    main()
