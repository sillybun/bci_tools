"""Two-row, three-column overview with one method per column."""

import numpy as np

from ...core import style
from ..base import new_figure
from .diagnostics import plot_angular_error


def overview(fit,residual_fit,spectrum,dirs,blocks,embedding,held,obs,
             scatter,banner, *, figure=None):
    fig = new_figure(16, 9) if figure is None else figure
    if embedding is None:
        axes = fig.subplots(2, 3)
    else:
        grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, .045, 1])
        axes = np.array([[fig.add_subplot(grid[row, col]) for col in (0, 1, 3)]
                         for row in range(2)])
        color_axis = fig.add_subplot(grid[1, 2])
    fig._overview_panels = list(axes.ravel())
    scatter(axes[0,0],fit['scores'][:,:2],dirs,
            f'PC1 ({100*spectrum[0]:.1f}% variance)',f'PC2 ({100*spectrum[1]:.1f}% variance)',
            'A  PCA: direction structure (in-sample)',legend=True)
    ax=axes[1,0]
    if residual_fit is not None:
        variance=residual_fit['explained']
        scatter(ax,residual_fit['scores'][:,:2],dirs,
                f'Residual PC1 ({100*variance[0]:.1f}% residual var.)',
                f'Residual PC2 ({100*variance[1]:.1f}% residual var.)',
                'D  Residual PCA: total-rate adjustment (in-sample)')
    else:style.text_panel(ax,['Residual PCA unavailable','Insufficient residual feature dimensions.'])
    if embedding is None:
        for ax in (axes[0,1],axes[1,1]):
            style.text_panel(ax,['UMAP disabled','Enable UMAP in S04 parameters.'])
    else:
        scores=embedding['scores']
        scatter(axes[0,1],scores,dirs,'UMAP 1','UMAP 2','B  UMAP: target-direction organization')
        points=axes[1,1].scatter(scores[:,0],scores[:,1],c=blocks,cmap='viridis',s=12,alpha=.7,lw=0)
        axes[1,1].set(xlabel='UMAP 1',ylabel='UMAP 2',title='E  UMAP: recording-block organization')
        axes[1,1].set_aspect('equal',adjustable='box')
        fig.colorbar(points, cax=color_axis, label='Recording block')
        # Match the square plot boxes while preserving equal UMAP units and all points.
        low, high = scores.min(axis=0), scores.max(axis=0)
        centre = (low + high) / 2
        span = float((high - low).max())
        radius = span / 2 + max(span * .08, .05)
        for ax in axes[:, 1]:
            ax.set(xlim=(centre[0] - radius, centre[0] + radius),
                   ylim=(centre[1] - radius, centre[1] + radius))
        for text in list(axes[0,1].texts):
            if 'outside view' in text.get_text():text.remove()
    scatter(axes[0,2],held,dirs,'TDR cos','TDR sin','C  TDR: direction structure (held-out)')
    plot_angular_error(axes[1,2], obs, 'F  TDR')
    for ax in axes.ravel():
        ax.set_box_aspect(1)
    fig.suptitle(banner+'\nPCA: explain variance  |  UMAP: explore neighborhoods  |  TDR: test task structure',fontsize=10)
    fig.supxlabel('Original/residual PCA and UMAP use the full cohort; TDR projections are held out. '
                  'Residual variance percentages refer to restandardized residual features.\n'
                  'Centroids and unfilled 1 SD ellipses describe trial spread. '
                  'Train-only residual PCA projections are provided in Supplementary.',fontsize=8)
    return fig
