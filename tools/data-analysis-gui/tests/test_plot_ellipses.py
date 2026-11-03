"""Geometry and export of the shared direction-spread ellipses."""
import numpy as np
from matplotlib.figure import Figure

from data_analysis_gui.core.style import direction_ellipses
from data_analysis_gui.storage import extract_plot_data


def test_rotated_one_sd_ellipse_and_export():
    angle = np.radians(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.array([[-2, 0], [2, 0], [0, -1], [0, 1]]) @ rotation.T + [3, -2]
    # Exclude a nonfinite pair without changing the group's centre or covariance.
    points = np.vstack([points, [np.nan, 1]])
    fig = Figure()
    ax = fig.subplots()
    patches = direction_ellipses(ax, points, np.zeros(5, int))
    assert len(patches) == 1
    ellipse = patches[0]
    np.testing.assert_allclose(ellipse.center, [3, -2])
    np.testing.assert_allclose([ellipse.width, ellipse.height], 2 * np.sqrt([8 / 3, 2 / 3]))
    np.testing.assert_allclose(ellipse.angle % 180, 30)
    rows = extract_plot_data(fig)
    assert len(rows) == 1
    assert rows[0]['artist'] == 'ellipse'
    assert rows[0]['series'] == '0° 1 SD'
    np.testing.assert_allclose([rows[0]['x'], rows[0]['y']], [3, -2])
    assert 'width=' in rows[0]['extra'] and 'angle_deg=' in rows[0]['extra']


def test_sparse_and_degenerate_directions_have_no_ellipse():
    points = np.array([[0, 0], [1, 1], [2, 2], [4, 1], [5, 3], [7, 7], [7, 7], [7, 7]])
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    ax = Figure().subplots()
    assert direction_ellipses(ax, points, labels) == []
