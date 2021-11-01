# ------------------------------------------------------------------------------
#   CONFIGURATION
# ------------------------------------------------------------------------------

# Dependencies
import sys, os, csv
import gitlab
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from matplotlib import rcParams
import contextily as ctx
import mapclassify as mc

# CRS
WGS84 = "EPSG:4326"

# Plot theme
MAPS_THEME = {"font.size": 6, "figure.dpi": 600, "text.usetex": False}
rcParams.update(MAPS_THEME)
FONT_NAME = "Noto Sans"
rcParams["font.family"] = FONT_NAME

# ZORDER
Z_BASE, Z_ADM, Z_MARINE, Z_DATA, Z_LAKE, Z_LABEL = 10, 20, 30, 40, 50, 60

# Color scheme
HIGHLIGHT_COLOR = "Black"
DEFAULT_COLOR = "DarkSlateGray"
WATER_COLOR = "LightSteelBlue"
BORDER_COLOR = "Silver"
WATER_BORDER_COLOR = "SteelBlue"
ALPH = 0.8

# Marker size
MARKER_SIZE = 6

# Cities path
PLACE_PATH = os.path.join("..", "data", "places", "worldcities.csv")

# World admin path
WORLD_PATH = os.path.join("..", "data", "admin", "geoBoundariesCGAZ_ADM0.shp")

# Marine paths
LAKE_PATH = os.path.join("..", "data", "marine", "ne_10m_lakes.shp")
MARINE_PATH = os.path.join("..", "data", "marine",
                           "ne_10m_geography_marine_polys.shp")

# Basemap
BASEMAP_PATH = os.path.join("..", "data", "basemap",
                            "eo_base_2020_clean_geo.tif")

# Country settings path
SETTINGS_PATH = os.path.join("..", "data", "country_settings.xlsx")
SETTINGS_SHEET = 'params'

# Invalid countries -- countries that fail WGS84 to WEBMERC conversion
INVALID_ISO = ["RUS", "FJI"]

# Output folder
OUT_PATH = os.path.join("..", "contributions")

# ------------------------------------------------------------------------------
#   UTILITY
# ------------------------------------------------------------------------------


# Add representative point
def add_representative_point(gdf):
    gdf["rep"] = gdf.geometry.representative_point()
    return gdf


# Get bounding box with specified buffer
def get_buffered_bounds(total_bounds, buffer=0.1):

    # Set axis limits
    minx, miny, maxx, maxy = total_bounds
    lenx = maxx - minx
    leny = maxy - miny
    maxl = max(lenx, leny)

    # Bounding box
    width = height = maxl * (1.0 + buffer)
    xmargin = (width - lenx) / 2.0
    ymargin = (height - leny) / 2.0
    left = minx - xmargin
    right = maxx + xmargin
    top = maxy + ymargin
    bottom = miny - ymargin

    # Return result
    return left, right, top, bottom


# Style mapclassify legend
def style_mapclassify_legend(legend_label, legend_units, data_max, ax):

    # Handle legend
    leg = ax.get_legend()
    leg.set_title(legend_label, prop={
                                        "weight": "medium",
                                        "size": "small"
                                      })
    leg._legend_box.align = "left"
    leg.set_zorder(100)
    # leg.set_bbox_to_anchor((1.5,0.5))

    # Fix legend text appearance
    for lbl in leg.get_texts():
        label_text = lbl.get_text()
        lower = label_text.split(",")[0].replace("[", "").replace("(", "")
        upper = label_text.split(",")[-1].replace("]", "").replace(")", "")
        if data_max > 999:
            new_text = f'{float(lower):,.0f}{legend_units} to {float(upper):,.0f}{legend_units}'
        elif data_max > 99:
            new_text = f'{float(lower):,.1f}{legend_units} to {float(upper):,.1f}{legend_units}'
        elif data_max > 9:
            new_text = f'{float(lower):,.2f}{legend_units} to {float(upper):,.2f}{legend_units}'
        else:
            new_text = f'{float(lower):,.3g}{legend_units} to {float(upper):,.3g}{legend_units}'
        lbl.set_text(new_text)
        lbl.set(fontsize="small", fontweight="medium")

    # Fix legend handle sizes so they don't overlap
    for hdl in leg.legendHandles:
        hdl._legmarker.set_markersize(6)

# ------------------------------------------------------------------------------
#   PARSERS
# ------------------------------------------------------------------------------


# Read in desired settings
def read_settings(iso):

    # Read settings file
    settings = pd.read_excel(SETTINGS_PATH,
                             SETTINGS_SHEET).set_index("iso_name")

    # Get relevant index
    settings_iso = settings.loc[iso]

    # Return result
    return settings_iso


# Read world administrative boundaries
def read_world_admin(settings):

    # Read world admin
    world_admin = gpd.read_file(WORLD_PATH)

    # Remove invalid countries - countries that fail conversion
    idx = world_admin.ISO_CODE.isin(INVALID_ISO)

    # Add representative point
    world_admin = add_representative_point(world_admin)

    # Construct points gdf
    world_points = world_admin.copy()
    world_points.set_geometry("rep", inplace=True)

    # Convert coordinate system
    world_admin = world_admin[~idx].to_crs(settings['WEBMERC'])
    world_points = world_points[~idx].to_crs(settings['WEBMERC'])

    # Return poly and points
    return world_admin, world_points


# Read populated places
def read_populated_places(iso, settings):

    #  Read populated places
    cities_df = pd.read_csv(PLACE_PATH).set_index("id")
    cities = gpd.GeoDataFrame(
        cities_df,
        geometry=gpd.points_from_xy(cities_df.longitude, cities_df.latitude),
        crs=WGS84,
    )

    # Projection
    cities = cities.to_crs(settings['WEBMERC'])

    # Isolated capital cities
    idx = (cities["capital"] == "primary")

    # Isolate those places above a population threshold
    kdx = (cities["population"] >= settings['POPTHRESH_NGB']) & (cities["iso3"] != iso)
    mdx = (cities["population"] >= settings['POPTHRESH_ISO']) & (cities["iso3"] == iso)
    jdx = kdx | mdx

    # Reduce places per thresholds
    capital_cities = cities[idx & jdx].copy()
    populated_places = cities[~idx & jdx].copy()

    # Return points
    return capital_cities, populated_places


# Read lakes
def read_lakes(settings):

    # Lake poly
    lakes = gpd.read_file(LAKE_PATH)

    # Add representative point
    lakes = add_representative_point(lakes)

    # Get points
    lakes_points = lakes.drop(columns="geometry").copy()
    lakes_points.set_geometry("rep", inplace=True)
    lakes_points = lakes_points[lakes_points["scalerank"] < settings['SCALERANK']]

    # Project CRS
    lakes = lakes.to_crs(settings['WEBMERC'])
    lakes_points = lakes_points.to_crs(settings['WEBMERC'])

    # Return result
    return lakes, lakes_points


# Read marine
def read_marine(settings):

    # Lake poly
    marine = gpd.read_file(MARINE_PATH)

    # Add representative point
    marine = add_representative_point(marine)

    # Get points
    marine_points = marine.copy()
    marine_points.set_geometry("rep", inplace=True)
    marine_points = marine_points[marine_points["scalerank"] < settings['SCALERANK']]

    # Project CRS
    marine = marine.to_crs(settings['WEBMERC'])
    marine_points = marine_points.to_crs(settings['WEBMERC'])

    # Return result
    return marine, marine_points

# ------------------------------------------------------------------------------
#   LAYOUT
# ------------------------------------------------------------------------------


# Plot country profile
def set_layout(ax):

    # Construct figure
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None
    )
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Starting font
    rcParams["font.family"] = FONT_NAME

# ------------------------------------------------------------------------------
#   BASEMAP
# ------------------------------------------------------------------------------


# Set basemap
def set_basemap(settings, ax):

    ctx.add_basemap(
        ax,
        source=BASEMAP_PATH,
        crs=settings['WEBMERC'],
        alpha=0.5,
        cmap="gray",
        vmin=0,
        vmax=255,
        zorder=Z_BASE,
    )


# ------------------------------------------------------------------------------
#   WORLD ADMIN
# ------------------------------------------------------------------------------


# Add world boundaries
def add_world_boundaries(world_admin, iso, ax):
    world_admin.geometry.boundary.plot(
        edgecolor=BORDER_COLOR,
        linewidth=0.5,
        ax=ax,
        zorder=Z_ADM,
    )


# Add world labels
def add_world_labels(world_points, iso, ax):
    for i, row in world_points.iterrows():
        color = DEFAULT_COLOR
        if row.ISO_CODE == iso:
            color = HIGHLIGHT_COLOR
        # String handling for name
        country_name = row.ISO_CODE
        # Set annotations
        if country_name and country_name != "None":
            ax.annotate(country_name,
                        xy=row.rep.coords[0],
                        ha="center",
                        fontweight="bold",
                        fontsize="small",
                        color=color,
                        zorder=Z_LABEL,
                        path_effects=[
                                    PathEffects.withStroke(
                                        linewidth=1,
                                        foreground="w",
                                        alpha=ALPH)
                                    ],
                        )

# ------------------------------------------------------------------------------
#   POPULATED PLACES
# ------------------------------------------------------------------------------


def add_populated_places(populated_places, ax):

    # Markers
    populated_places.plot(
            ax=ax,
            marker="o",
            markersize=MARKER_SIZE,
            color="White",
            edgecolor=DEFAULT_COLOR,
            zorder=Z_LABEL,
            linewidth=0.25,
        )

    # Labels
    for i, row in populated_places.iterrows():
        ax.annotate(row['name'], xy=row.geometry.coords[0],
                    xytext=(1, 1),
                    textcoords="offset points",
                    fontsize="x-small",
                    fontweight="medium",
                    color=DEFAULT_COLOR,
                    zorder=Z_LABEL,
                    path_effects=[
                        PathEffects.withStroke(
                                                linewidth=1,
                                                foreground="w",
                                                alpha=ALPH
                        )],
                    )
        if pd.isna(row["localname"]):
            continue
        rcParams["font.family"] = row["fontname"]
        ax.annotate(
            row["localname"],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(1, -4),
            textcoords="offset points",
            fontsize="x-small",
            fontweight="medium",
            color=DEFAULT_COLOR,
            zorder=Z_LABEL,
            path_effects=[
                PathEffects.withStroke(linewidth=1, foreground="w", alpha=ALPH)
            ],
            bbox=dict(
                boxstyle="square,pad=0.1",
                facecolor="white",
                alpha=0.05,
                edgecolor="none",
            ),
        )
        rcParams["font.family"] = FONT_NAME


def add_capital_cities(capital_cities, ax):

    # Markers
    capital_cities.plot(
                ax=ax,
                marker="o",
                markersize=MARKER_SIZE,
                color="White",
                edgecolor=DEFAULT_COLOR,
                linewidth=0.25,
                zorder=Z_LABEL,
            )
    capital_cities.plot(
        ax=ax,
        marker="*",
        markersize=MARKER_SIZE*2./3.,
        color=DEFAULT_COLOR,
        linewidth=0.25,
        zorder=Z_LABEL,
    )

    # Labels
    for i, row in capital_cities.iterrows():
        ax.annotate(row['name'], xy=row.geometry.coords[0],
                    xytext=(1, 1),
                    textcoords="offset points",
                    fontsize="x-small",
                    fontweight="bold",
                    color=DEFAULT_COLOR,
                    zorder=Z_LABEL,
                    path_effects=[
                        PathEffects.withStroke(
                                                linewidth=1,
                                                foreground="w",
                                                alpha=ALPH
                        )],
                    )
        if pd.isna(row["localname"]):
            continue
        rcParams["font.family"] = row["fontname"]
        ax.annotate(
            row["localname"],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(1, -4),
            textcoords="offset points",
            fontsize="x-small",
            fontweight="bold",
            color=DEFAULT_COLOR,
            zorder=Z_LABEL,
            path_effects=[
                PathEffects.withStroke(linewidth=1, foreground="w", alpha=ALPH)
            ],
            bbox=dict(
                boxstyle="square,pad=0.1",
                facecolor="white",
                alpha=0.05,
                edgecolor="none",
            ),
        )
        rcParams["font.family"] = FONT_NAME


# ------------------------------------------------------------------------------
#   WATER FEATURES
# ------------------------------------------------------------------------------


# Add water boundaries
def add_water_boundaries(water, z, ax):
    water.geometry.plot(
            color=WATER_COLOR,
            alpha=0.95,
            edgecolor=WATER_BORDER_COLOR,
            linewidth=0.05,
            ax=ax,
            zorder=z,
    )


# Add water labels
def add_water_labels(water_points, ax):
    for i, row in water_points.iterrows():
        if row['name']:
            ax.annotate(
                row['name'].replace(" ", "\n"),
                xy=(row.rep.x, row.rep.y),
                xytext=(1, 1),
                textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="center_baseline",
                fontsize="x-small",
                fontweight="medium",
                color=WATER_BORDER_COLOR,
                zorder=Z_LABEL,
                path_effects=[
                    PathEffects.withStroke(
                        linewidth=1,
                        foreground="w",
                        alpha=ALPH)
                ],
            )
