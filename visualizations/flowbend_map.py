"""Shelf-wide flow-bend gradient field map: the normalised gradient field the profiles are
extracted along, the flow-bend lines, the reference GL, and boxes marking the two lines
the analysis works through (1563 and 2574). Produces
`flowbend_lines_without_inset.pdf`; the box geometries come from `hybrid_flowlines.gpkg`.

Usage
-----
    python -m visualizations.flowbend_map --data-dir ../data
"""

import argparse
import pathlib

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import rasterio as rio

from visualizations import style

# the two flow-bend lines the analysis works through
EXAMPLE_ID = 1563
BAD_ID = 2574

# band 3 ("combined") of hybrid_grad_field.tif: tidal-flexure gradient near the GL blended
# with the ice-flow direction elsewhere
COMBINED_BAND = 3

# white flow-bend lines and red reference GL against the plasma field; limegreen boxes
FLOWBEND_COLOR = "white"
REF_GL_COLOR = "red"
HIGHLIGHT_COLOR = "limegreen"

# a sample of the flow-bend lines reads as a field where drawing all of them fills solid
N_SAMPLE = 2000
SEED = 0

# authored at the same width as flowbend_line_profiles.pdf so text renders at the same size
# when both are included at \linewidth; aspect is preserved so this costs no page height
FIG_WIDTH_IN = 15.12   # flowbend_line_profiles.pdf width (1088.4 pt)
MAP_ASPECT = 1.1234    # grad field extent: 418800 m / 372800 m


def load(data_dir, roi_path):
    roi = gpd.read_file(roi_path).to_crs("EPSG:3031")
    flowbends = gpd.read_file(data_dir / "hybrid_flowlines.gpkg").to_crs("EPSG:3031")
    ref_gl = gpd.read_file(data_dir / "tmoc_gl_2020.geojson").to_crs("EPSG:3031")

    with rio.open(data_dir / "hybrid_grad_field.tif") as tif:
        window = rio.windows.from_bounds(
            left=roi.total_bounds[0], right=roi.total_bounds[2],
            bottom=roi.total_bounds[1], top=roi.total_bounds[3], transform=tif.transform)
        grad = tif.read(COMBINED_BAND, window=window)
        bounds = rio.windows.bounds(window, tif.transform)

    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    return roi, flowbends, ref_gl, grad, extent


def highlight(ax, geom_bounds, linestyle="-"):
    minx, miny, maxx, maxy = geom_bounds
    ax.add_patch(mpl.patches.Rectangle(
        (minx, miny), maxx - minx, maxy - miny, linewidth=1.5,
        edgecolor=HIGHLIGHT_COLOR, facecolor="none", linestyle=linestyle, zorder=10))


def build(roi, flowbends, ref_gl, grad, extent):
    # +2 in of height for the horizontal colorbar and the graticule labels; savefig's
    # bbox_inches="tight" crops whatever is left over.
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN / MAP_ASPECT + 2),
                     dpi=300, layout="constrained")
    ax = fig.add_subplot(1, 1, 1, projection=style.CRS_3031)
    ax.set_extent(extent, crs=style.CRS_3031)

    field = ax.imshow(grad, cmap="plasma", alpha=0.9, extent=extent,
                      transform=style.CRS_3031, origin="upper")
    style.add_graticule(ax)
    fig.colorbar(field, ax=ax, label="Normalised flow-bend gradient field (m/d)",
                 pad=0.046, fraction=0.046, orientation="horizontal")

    flowbends.sample(min(N_SAMPLE, len(flowbends)), random_state=SEED).plot(
        ax=ax, color=FLOWBEND_COLOR, linewidth=0.5, alpha=0.7)
    ref_gl.clip(roi.total_bounds).plot(ax=ax, color=REF_GL_COLOR, linewidth=1.0)

    line = mpl.lines.Line2D
    ax.legend(
        facecolor="gray",
        handles=[line([0, 1], [0, 0], color=FLOWBEND_COLOR, linewidth=2.5),
                 line([0, 1], [0, 0], color=REF_GL_COLOR, linewidth=2.5),
                 line([0, 1], [0, 0], color=HIGHLIGHT_COLOR, linewidth=2.0),
                 line([0, 1], [0, 0], color=HIGHLIGHT_COLOR, linewidth=2.0, linestyle="--")],
        labels=["Flow-bend lines", "Reference GL",
                f"Flow-bend line {EXAMPLE_ID}", f"Flow-bend line {BAD_ID}"])

    highlight(ax, flowbends.loc[flowbends.Id == EXAMPLE_ID].total_bounds)
    highlight(ax, flowbends.loc[flowbends.Id == BAD_ID].total_bounds, linestyle="--")

    ax.margins(0)
    style.add_scalebar(ax, max_km=50, major_div=1, fontsize=style.GRATICULE_FONTSIZE)
    style.panel_label(ax, "(a)")
    return fig


def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=repo / "data")
    parser.add_argument("--roi", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1] / "rois" / "LarsenC.geojson")
    parser.add_argument("--out", type=pathlib.Path,
                        default=repo / "figures" / "flowbend_lines_without_inset.pdf")
    args = parser.parse_args()

    style.use_style()
    fig = build(*load(args.data_dir, args.roi))
    style.savefig(fig, args.out)


if __name__ == "__main__":
    main()
