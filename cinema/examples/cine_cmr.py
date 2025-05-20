"""Script to visualize the SAX and LAX slices in real coordinate space."""

from pathlib import Path

import numpy as np
import plotly.graph_objs as go
import SimpleITK as sitk  # noqa: N813
from plotly.offline import iplot


def image_to_real_space(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rot: np.ndarray,
    origin: np.ndarray,
    pixel_spacing: tuple[float, float],
    slice_spacing: float,
) -> np.ndarray:
    """Transform image coordinates to real space coordinates.

    Args:
        x: x axis index (n_points, )
        y: y axis index (n_points, )
        z: z axis index (n_points, )
        rot: 3x3 matrix
        origin: x0, y0, z0
        pixel_spacing: dx, dy
        slice_spacing: dz

    Returns:
        real space coordinates (3, n_points)
    """
    coords = np.array([x, y, z])
    spacing = np.array([pixel_spacing[0], pixel_spacing[1], slice_spacing])
    real_coords = rot @ (coords * spacing[:, None]) + origin[:, None]
    return real_coords


def get_meshgrid(
    height: int,
    width: int,
    z: int,
    rot: np.ndarray,
    origin: np.ndarray,
    pixel_spacing: tuple[float, float],
    slice_spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get meshgrid for LAX view.

    Args:
        height: image height
        width: image width
        z: image location
        rot: 3x3 matrix
        origin: x0, y0, z0
        pixel_spacing: dx, dy
        slice_spacing: dz

    Returns:
        x: x coordinates (width, height)
        y: y coordinates (width, height)
        z: z coordinates (width, height)
    """
    x, y = np.meshgrid(np.linspace(0, width, width), np.linspace(0, height, height), indexing="ij")
    z = z + np.zeros((width, height))
    coords = image_to_real_space(x.reshape(-1), y.reshape(-1), z.reshape(-1), rot, origin, pixel_spacing, slice_spacing)
    x = coords[0, :].reshape(width, height)
    y = coords[1, :].reshape(width, height)
    z = coords[2, :].reshape(width, height)
    return x, y, z


def plot_cmr_views(
    lax_2c_image: sitk.Image,
    lax_3c_image: sitk.Image,
    lax_4c_image: sitk.Image,
    sax_image: sitk.Image,
    t_to_show: int,
    depth_to_show: int,
) -> go.Figure:
    """Plot SAX and LAX images in 3D space."""
    width_2c, height_2c, _, _ = lax_2c_image.GetSize()
    *pixel_spacing_2c, _, _ = lax_2c_image.GetSpacing()
    origin_2c = np.array(lax_2c_image.GetOrigin())[:3]
    rot_2c = np.array(lax_2c_image.GetDirection()).reshape(4, 4)[:3, :3]

    width_3c, height_3c, _, _ = lax_3c_image.GetSize()
    *pixel_spacing_3c, _, _ = lax_3c_image.GetSpacing()
    origin_3c = np.array(lax_3c_image.GetOrigin())[:3]
    rot_3c = np.array(lax_3c_image.GetDirection()).reshape(4, 4)[:3, :3]

    width_4c, height_4c, _, _ = lax_4c_image.GetSize()
    *pixel_spacing_4c, _, _ = lax_4c_image.GetSpacing()
    origin_4c = np.array(lax_4c_image.GetOrigin())[:3]
    rot_4c = np.array(lax_4c_image.GetDirection()).reshape(4, 4)[:3, :3]

    width_sax, height_sax, depth_sax, _ = sax_image.GetSize()
    *pixel_spacing_sax, slice_spacing_sax, _ = sax_image.GetSpacing()
    origin_sax = np.array(sax_image.GetOrigin()[:3])
    rot_sax = np.array(sax_image.GetDirection()).reshape(4, 4)[:3, :3]

    x_2c = np.array([0, 0, width_2c, width_2c, 0])
    y_2c = np.array([0, height_2c, height_2c, 0, 0])
    coords_2c = image_to_real_space(x_2c, y_2c, x_2c * 0, rot_2c, origin_2c, pixel_spacing_2c, 0)

    x_3c = np.array([0, 0, width_3c, width_3c, 0])
    y_3c = np.array([0, height_3c, height_3c, 0, 0])
    coords_3c = image_to_real_space(x_3c, y_3c, x_3c * 0, rot_3c, origin_3c, pixel_spacing_3c, 0)

    x_4c = np.array([0, 0, width_4c, width_4c, 0])
    y_4c = np.array([0, height_4c, height_4c, 0, 0])
    coords_4c = image_to_real_space(x_4c, y_4c, x_4c * 0, rot_4c, origin_4c, pixel_spacing_4c, 0)

    trace_2c = go.Scatter3d(
        x=coords_2c[0, :],
        y=coords_2c[1, :],
        z=coords_2c[2, :],
        mode="lines",
        name="LAX 2C",
        marker={"color": "#B85450"},
        line={"width": 8},
    )
    trace_3c = go.Scatter3d(
        x=coords_3c[0, :],
        y=coords_3c[1, :],
        z=coords_3c[2, :],
        mode="lines",
        name="LAX 3C",
        marker={"color": "#D6B656"},
        line={"width": 8},
    )
    trace_4c = go.Scatter3d(
        x=coords_4c[0, :],
        y=coords_4c[1, :],
        z=coords_4c[2, :],
        mode="lines",
        name="LAX 4C",
        marker={"color": "#82B366"},
        line={"width": 8},
    )

    np_image_2c = np.transpose(sitk.GetArrayFromImage(lax_2c_image))[..., 0, t_to_show]  # (width, height)
    x, y, z = get_meshgrid(height_2c, width_2c, 0, rot_2c, origin_2c, pixel_spacing_2c, 0)
    image_trace_2c = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=np_image_2c,
        cmin=0,
        cmax=255,
        colorscale="gray",
        showscale=False,
    )
    np_image_3c = np.transpose(sitk.GetArrayFromImage(lax_3c_image))[..., 0, t_to_show]  # (width, height)
    x, y, z = get_meshgrid(height_3c, width_3c, 0, rot_3c, origin_3c, pixel_spacing_3c, 0)
    image_trace_3c = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=np_image_3c,
        cmin=0,
        cmax=255,
        colorscale="gray",
        showscale=False,
    )
    np_image_4c = np.transpose(sitk.GetArrayFromImage(lax_4c_image))[..., 0, t_to_show]  # (width, height)
    x, y, z = get_meshgrid(height_4c, width_4c, 0, rot_4c, origin_4c, pixel_spacing_4c, 0)
    image_trace_4c = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=np_image_4c,
        cmin=0,
        cmax=255,
        colorscale="gray",
        showscale=False,
    )

    traces = [
        image_trace_2c,
        image_trace_3c,
        image_trace_4c,
    ]
    for d in range(depth_sax):
        x_sax = np.array([0, 0, width_sax, width_sax, 0])
        y_sax = np.array([0, height_sax, height_sax, 0, 0])
        z_sax = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) + d

        coords_sax = image_to_real_space(x_sax, y_sax, z_sax, rot_sax, origin_sax, pixel_spacing_sax, slice_spacing_sax)

        trace_sax = go.Scatter3d(
            x=coords_sax[0, :],
            y=coords_sax[1, :],
            z=coords_sax[2, :],
            mode="lines",
            name="SAX",
            marker={"color": "#6C8EBF"},
            line={"width": 6},
        )
        np_image_sax = np.transpose(sitk.GetArrayFromImage(sax_image))[..., d, t_to_show]
        x, y, z = get_meshgrid(height_sax, width_sax, d, rot_sax, origin_sax, pixel_spacing_sax, slice_spacing_sax)
        image_trace_sax = go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=np_image_sax,
            cmin=0,
            cmax=255,
            colorscale="gray",
            showscale=False,
        )

        if d == depth_to_show:
            traces.append(image_trace_sax)
        else:
            trace_sax["showlegend"] = False
        traces.append(trace_sax)
    traces += [trace_2c, trace_3c, trace_4c]
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene_camera={
            "eye": {"x": -1.5, "y": 0.75, "z": 1.5},
        },
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "center",
            "x": 0.5,
            "title": "",
            "font": {"color": "black"},
        },
    )
    fig.update_layout(
        scene={
            "xaxis": {
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
                "title": "",
            },
            "yaxis": {"showticklabels": False, "showgrid": False, "zeroline": False, "title": ""},
            "zaxis": {"showticklabels": False, "showgrid": False, "zeroline": False, "title": ""},
        }
    )
    return fig


def run() -> None:
    """Visualize SAX and LAX images in 3D space."""
    # load data
    t_to_show = 4
    depth_to_show = 4
    uid = "1"
    data_dir = Path(__file__).parent.resolve()

    lax_2c_image = sitk.ReadImage(data_dir / f"data/ukb/{uid}/{uid}_lax_2c.nii.gz")
    lax_3c_image = sitk.ReadImage(data_dir / f"data/ukb/{uid}/{uid}_lax_3c.nii.gz")
    lax_4c_image = sitk.ReadImage(data_dir / f"data/ukb/{uid}/{uid}_lax_4c.nii.gz")
    sax_image = sitk.ReadImage(data_dir / f"data/ukb/{uid}/{uid}_sax.nii.gz")

    fig = plot_cmr_views(lax_2c_image, lax_3c_image, lax_4c_image, sax_image, t_to_show, depth_to_show)
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
    )
    fig.write_image("cine_cmr_sax_lax.png", scale=5)
    iplot(fig)


if __name__ == "__main__":
    run()
