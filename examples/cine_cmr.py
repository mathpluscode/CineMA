"""Script to visualize the SAX and LAX slices in real coordinate space."""

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


if __name__ == "__main__":
    # load data
    lax_2c_image = sitk.ReadImage("data/ukb/1/1_lax_2c.nii.gz")
    height_2c, width_2c, _ = lax_2c_image.GetSize()
    *pixel_spacing_2c, _ = lax_2c_image.GetSpacing()
    origin_2c = np.array(lax_2c_image.GetOrigin())
    rot_2c = np.array(lax_2c_image.GetDirection()).reshape(3, 3)

    lax_3c_image = sitk.ReadImage("data/ukb/1/1_lax_3c.nii.gz")
    height_3c, width_3c, _ = lax_3c_image.GetSize()
    *pixel_spacing_3c, _ = lax_3c_image.GetSpacing()
    origin_3c = np.array(lax_3c_image.GetOrigin())
    rot_3c = np.array(lax_3c_image.GetDirection()).reshape(3, 3)

    lax_4c_image = sitk.ReadImage("data/ukb/1/1_lax_4c.nii.gz")
    height_4c, width_4c, _ = lax_4c_image.GetSize()
    *pixel_spacing_4c, _ = lax_4c_image.GetSpacing()
    origin_4c = np.array(lax_4c_image.GetOrigin())
    rot_4c = np.array(lax_4c_image.GetDirection()).reshape(3, 3)

    sax_image = sitk.ReadImage("data/ukb/1/1_sax.nii.gz")
    height_sax, width_sax, depth_sax, _ = sax_image.GetSize()
    *pixel_spacing_sax, slice_spacing_sax, _ = sax_image.GetSpacing()
    origin_sax = np.array(sax_image.GetOrigin()[:3])
    rot_sax = np.array(sax_image.GetDirection()).reshape(4, 4)[:3, :3]

    # visualize
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
        line={"width": 4},
    )
    trace_3c = go.Scatter3d(
        x=coords_3c[0, :],
        y=coords_3c[1, :],
        z=coords_3c[2, :],
        mode="lines",
        name="LAX 3C",
        marker={"color": "#D6B656"},
        line={"width": 4},
    )
    trace_4c = go.Scatter3d(
        x=coords_4c[0, :],
        y=coords_4c[1, :],
        z=coords_4c[2, :],
        mode="lines",
        name="LAX 4C",
        marker={"color": "#82B366"},
        line={"width": 4},
    )
    traces = [trace_2c, trace_3c, trace_4c]
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
            name=f"SAX Slice {d}",
            marker={"color": "#6C8EBF"},
            line={"width": 4},
        )
        traces.append(trace_sax)

    fig = go.Figure(data=traces)
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.write_image("cine_cmr_sax_lax.png", scale=5)
    iplot(fig)
