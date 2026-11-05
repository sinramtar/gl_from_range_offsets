"""Flow-bend line / profile figure. Panel (b) draws the gradient field, flow and
flow-bend lines and reference GL in EPSG:3031 map coordinates with a scale bar; panels
(c, d) draw the range-offset profiles along the flow-bend and flow lines, coloured by
acquisition year. The shaded bending zone uses the smoothing and thresholds of
`Flowline_params` in the configuration.

`--data-dir` must hold `hybrid_flowlines.gpkg`, `tmoc_gl_2020.geojson`,
`hybrid_grad_field.tif`, the flowline `flowlines/<id>.geojson`, and the extracted profiles
`clean_offsets_hybrid/<id>/*.npz` and `clean_offsets_flow/<id>/*.npz`.

Usage
-----
    python -m visualizations.flowbend_profiles --config config.yaml --data-dir ../data
"""

import argparse
import pathlib

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import shapely.ops
from matplotlib_map_utils.core.scale_bar import ScaleBar
from scipy.ndimage import gaussian_filter1d

LANDWARD = SEAWARD = 2.0e4
# navy / vermillion / dark green, the same year colours as the GL points map
YEAR_COLORS = {2019: "#332288", 2020: "#D55E00", 2021: "#117733"}

# type sizes: axis titles and colorbar label / tick labels / panel labels / legend
LABEL_FS, TICK_FS, PANEL_FS, LEGEND_FS = 16, 13, 16, 12


def load_profiles(offset_dir):
    """Range offset profile per acquisition pair, along the extraction line."""
    profiles = {}
    for f in sorted(pathlib.Path(offset_dir).glob("*.npz")):
        with np.load(f) as data:
            profiles[f.stem] = data["arr_0"][0, :]
    return profiles


def bending_zone(profiles, pixel_size, sigma_px, lower, upper):
    """Along-line extent of the flexure zone. Thresholds the Gaussian-smoothed pixel-wise
    standard deviation of the profiles (`sigma_px` pixels) into [`lower`, `upper`], the
    bending band. Returns the arc-length interval, in metres from the landward end of the
    line, of the longest contiguous run inside the band."""
    stack = np.array(list(profiles.values()))
    n = stack.shape[1]
    step = (LANDWARD + SEAWARD) / (n - 1)
    sd = gaussian_filter1d(np.nanstd(stack, axis=0), sigma=sigma_px * pixel_size / step,
                           mode="nearest")

    inside = np.flatnonzero((sd >= lower) & (sd <= upper))
    if inside.size == 0:
        return None
    runs = np.split(inside, np.flatnonzero(np.diff(inside) > 1) + 1)
    run = max(runs, key=len)
    return run[0] * step, run[-1] * step


def draw_axis_and_zone(ax, line, zone, offset=700.0):
    """Draw the flow-bend line's coordinate axis (the line itself, offset to one side so it
    annotates rather than covers) and the flexure `zone` it passes through."""
    # black with a white halo, legible over both ends of the plasma colormap
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]

    axis = line.offset_curve(offset)
    coords = np.array(axis.coords)
    ax.plot(coords[:, 0], coords[:, 1], color="black", lw=1.4, linestyle=(0, (5, 3)),
            zorder=4, path_effects=halo)

    if zone is not None:
        start, end = zone
        seg = shapely.ops.substring(line, start, end)
        ax.add_patch(mpl.patches.Polygon(
            np.array(seg.buffer(700.0).exterior.coords), closed=True,
            facecolor="lightgreen", edgecolor="darkgreen", alpha=0.5, lw=0.8, zorder=3))

    # arrow at the seaward end marking increasing x, labelled on the outer side of the axis
    tail = axis.interpolate(axis.length - 3500.0)
    head = axis.interpolate(axis.length - 200.0)
    ax.annotate("", xy=(head.x, head.y), xytext=(tail.x, tail.y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.6",
                                color="black", lw=1.6, shrinkA=0, shrinkB=0,
                                path_effects=halo), zorder=5)

    step = np.array([head.x - tail.x, head.y - tail.y])
    normal = np.array([-step[1], step[0]]) / np.linalg.norm(step)
    if np.sign(offset) < 0:
        normal = -normal
    label = np.array([head.x, head.y]) + normal * 1100.0
    ax.text(label[0], label[1], "$x$", color="black", fontsize=LABEL_FS, ha="center",
            va="center", zorder=5, path_effects=halo)


def plot_profiles(ax, profiles):
    """Profiles coloured by acquisition year, in random order so no year is buried."""
    order = list(profiles)
    np.random.default_rng(0).shuffle(order)
    for pair in order:
        offset = profiles[pair]
        x = np.linspace(-LANDWARD, SEAWARD, offset.shape[0]) / 1e3
        year = int(pair[:4])
        ax.plot(x, offset, color=YEAR_COLORS[year], linewidth=0.8, alpha=0.75)
    ax.axvline(0.0, color="red", linewidth=1.2)
    ax.set_ylim(-3, 3)
    ax.tick_params(axis="both", labelsize=TICK_FS)


def make_figure(grad, extent, uv, flowline, hybrid_line, ref_line,
                prof_hybrid, prof_flow, zone, out_stem):
    fig = plt.figure(figsize=(15, 7.5), constrained_layout=True)
    gs = mpl.gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 4], height_ratios=[1, 1])

    ax_c = fig.add_subplot(gs[1, 1])
    ax_b = fig.add_subplot(gs[0, 1], sharex=ax_c, sharey=ax_c)
    ax_a = fig.add_subplot(gs[0:2, 0])

    xmin, xmax, ymin, ymax = extent
    im = ax_a.imshow(grad, cmap="plasma", alpha=0.8, extent=extent)

    # quiver in map coordinates; v is flipped because imshow's origin='upper' descends
    u, v = uv
    ny, nx = grad.shape
    X, Y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymax, ymin, ny))
    angle = np.arctan2(-v, u)
    ax_a.quiver(X[::20, ::20], Y[::20, ::20],
                np.cos(angle)[::20, ::20], np.sin(angle)[::20, ::20],
                scale=10, color="violet")

    fx, fy = flowline.geometry.values[0].coords.xy
    hfx, hfy = hybrid_line.geometry.values[0].coords.xy
    ax_a.plot(fx, fy, color="cyan", lw=2.0, label="Flowline")
    ax_a.plot(hfx, hfy, color="white", lw=2.0, label="Flow-bend line")
    draw_axis_and_zone(ax_a, hybrid_line.geometry.values[0], zone)
    ref_line.plot(ax=ax_a, color="red", linewidth=2.0)
    ax_a.plot([], [], color="red", lw=2.0, label="Reference GL")

    ax_a.set_xlim(xmin, xmax)
    ax_a.set_ylim(ymin, ymax)
    ax_a.set_aspect("equal")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    leg = ax_a.legend(fontsize=LEGEND_FS)
    leg.get_frame().set_facecolor("darkgray")
    leg.set_alpha(0.5)
    ax_a.text(0.02, 0.98, "(b)", transform=ax_a.transAxes, fontsize=PANEL_FS,
              fontweight="bold", va="top", ha="left",
              bbox=dict(boxstyle="square", fc="white", alpha=0.5))
    # colorbar as an inset of ax_a: aspect='equal' shrinks the axes box to the image
    cax = ax_a.inset_axes([0.0, -0.055, 1.0, 0.025])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal",
                        label="Normalised flow-bend gradient field (m/d)")
    cbar.set_label("Normalised flow-bend gradient field (m/d)", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    # 5 km bar, about a quarter of the ~20 km panel width
    ax_a.add_artist(ScaleBar(
        location="lower left", style="boxes",
        bar={"projection": 3031, "minor_type": "none", "height": 0.05, "major_div": 1,
             "max": 5, "facecolors": "white"},
        labels={"style": "minor_first", "loc": "above", "pad": 0},
        units={"loc": "text", "pad": 0},
        text={"stroke_color": "white", "textcolor": "white", "fontsize": 8,
              "stroke_width": 0.3, "fontweight": "light", "fontfamily": "sans-serif"}))

    plot_profiles(ax_b, prof_hybrid)
    plot_profiles(ax_c, prof_flow)

    # one y label for both panels, anchored to the top edge of the lower one
    ax_c.set_ylabel("Vertical tide deflection (m)", y=1.0, fontsize=LABEL_FS)
    ax_c.set_xlabel("Distance along extraction line (km)", fontsize=LABEL_FS)
    ax_b.text(0.02, 0.97, "(c)", transform=ax_b.transAxes, fontsize=PANEL_FS,
              fontweight="bold", va="top", ha="left")
    ax_c.text(0.02, 0.97, "(d)", transform=ax_c.transAxes, fontsize=PANEL_FS,
              fontweight="bold", va="top", ha="left")

    handles = [mpl.lines.Line2D([], [], color=c, lw=2.0, label=str(y))
               for y, c in YEAR_COLORS.items()]
    ax_b.legend(handles=handles, title="Acquisition year", fontsize=LEGEND_FS,
                title_fontsize=LEGEND_FS, loc="upper left", bbox_to_anchor=(0.08, 1.0),
                framealpha=0.85)

    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"profiles (flow-bend / flow): {len(prof_hybrid)} / {len(prof_flow)}")
    print(f"panel (b) width            : {(xmax - xmin) / 1e3:.1f} km")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    ap.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    ap.add_argument("--flowline-id", type=int, default=1563)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("figures/flowbend_line_profiles"))
    args = ap.parse_args()

    import yaml

    with args.config.open() as stream:
        flowline_params = yaml.safe_load(stream)["Flowline_params"]
    # the flexure zone is where the smoothed standard deviation of the offsets time series
    # lies between the configured thresholds
    lower, upper = float(flowline_params["lower_stddev"]), float(flowline_params["upper_stddev"])
    sigma_px = float(flowline_params["sigma"])

    fid = args.flowline_id
    flowline = gpd.read_file(args.data_dir / "flowlines" / f"{fid}.geojson").to_crs("EPSG:3031")
    hybrid = gpd.read_file(args.data_dir / "hybrid_flowlines.gpkg").to_crs("EPSG:3031")
    hybrid_line = hybrid[hybrid.Id == fid]
    ref_line = gpd.read_file(args.data_dir / "tmoc_gl_2020.geojson").to_crs("EPSG:3031")

    frame = gpd.GeoSeries([flowline.geometry.values[0], hybrid_line.geometry.values[0]],
                          crs="EPSG:3031").scale(xfact=1.2, yfact=1.2, origin="center")
    bounds = frame.total_bounds

    with rio.open(args.data_dir / "hybrid_grad_field.tif") as tif:
        window = rio.windows.from_bounds(*bounds, transform=tif.transform)
        data = tif.read(window=window)
        wb = rio.windows.bounds(window, tif.transform)
        res = tif.res
    extent = [wb[0], wb[2], wb[1], wb[3]]

    prof_hybrid = load_profiles(args.data_dir / "clean_offsets_hybrid" / str(fid))
    prof_flow = load_profiles(args.data_dir / "clean_offsets_flow" / str(fid))

    zone = bending_zone(prof_hybrid, pixel_size=res[0], sigma_px=sigma_px,
                        lower=lower, upper=upper)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(data[2], extent, (data[0], data[1]),
                flowline, hybrid_line, ref_line.clip(bounds),
                prof_hybrid, prof_flow, zone, args.out)

    if zone is not None:
        print(f"bending zone               : {zone[0] - LANDWARD:.0f} to "
              f"{zone[1] - LANDWARD:.0f} m from the reference GL "
              f"(stddev in [{lower}, {upper}], sigma {sigma_px} px)")


if __name__ == "__main__":
    main()
