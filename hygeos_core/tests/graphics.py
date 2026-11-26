import xarray as xr
import numpy as np

def xrimshow(
    da: xr.DataArray,
    title: str | None=None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    display_size: float=3.0,
    margin_top: float=0.1,
    margin_left: float=0.6,
    margin_bottom: float=0.5,
    margin_right: float=0.1,
    cbar_thickness: float=0.15,
    cbar_length: float=0.5,
    cbar_gap_horizontal: float=0.5,
    cbar_gap_vertical: float=0.2,
    cbar_label_space: float=0.5,
    title_height: float=0.4,
    yincrease: bool = False,
):
    """
    Plot a 2D DataArray using imshow with consistent absolute margins.

    The figure size adapts to the data aspect ratio while maintaining
    constant image area (width * height = display_size²).  A square
    image is therefore display_size × display_size.

    The colorbar is placed below the image for wide data (aspect >= 1)
    and to the right for tall data (aspect < 1).  All margins, gaps and
    the colorbar thickness are specified in inches and stay constant
    regardless of data shape, ensuring a visually consistent layout.

    Parameters
    ----------
    da : xarray.DataArray
        2D DataArray to plot.
    title : str, optional
        Title for the plot.
    vmin, vmax : float, optional
        Colormap range.
    cmap : str, optional
        Colormap name (e.g., 'viridis', 'plasma'). If None, uses xarray's default colormap.
    display_size : float, default=3.
        Controls display size.  The image area in inches² equals
        display_size², so a square image would be
        display_size × display_size.
    margin_top : float, default=0.1
        Top margin in inches (used when there is no title).
    margin_left : float, default=0.5
        Left margin in inches.
    margin_bottom : float, default=0.3
        Bottom margin in inches.
    margin_right : float, default=0.1
        Right margin in inches.
    cbar_thickness : float, default=0.15
        Colorbar thickness in inches.
    cbar_length : float, default=0.5
        Colorbar length as a fraction of the image extent along the
        same axis (0 to 1).  The colorbar is centered.
    cbar_gap_horizontal : float, default=0.5
        Gap between the image and a horizontal colorbar (below) in inches.
    cbar_gap_vertical : float, default=0.2
        Gap between the image and a vertical colorbar (right) in inches.
    cbar_label_space : float, default=0.4
        Extra space in inches reserved for colorbar tick labels.
        Added to the right for vertical colorbars, to the bottom for
        horizontal colorbars.
    title_height : float, default=0.4
        Space reserved for the title in inches (replaces margin_top
        when *title* is set).
    yincrease : bool, default=False
        If True, the y-axis increases upward.
        If False, the y-axis increases downward.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    im : matplotlib.image.AxesImage
    """
    from matplotlib import pyplot as plt

    # Data aspect ratio (width / height in pixels)
    aspect = da.shape[1] / da.shape[0]

    # Image dimensions in inches, preserving area = display_size²
    img_w = (display_size ** 2 * aspect) ** 0.5
    img_h = (display_size ** 2 / aspect) ** 0.5

    if title is None and isinstance(da.name, str):
        title = da.name

    top_space = title_height if title else margin_top

    # Set vmin/vmax
    da = da.compute()
    vmin = vmin or float(np.nanpercentile(da, 5))
    vmax = vmax or float(np.nanpercentile(da, 95))

    # Compute figure size and axes rectangles [left, bottom, w, h]
    if aspect >= 1:
        # Wide image → horizontal colorbar below
        fig_w = margin_left + img_w + margin_right
        fig_h = top_space + img_h + cbar_gap_horizontal + cbar_thickness + cbar_label_space

        img_rect = (
            margin_left / fig_w,
            (cbar_label_space + cbar_thickness + cbar_gap_horizontal) / fig_h,
            img_w / fig_w,
            img_h / fig_h,
        )
        cbar_actual_w = img_w * cbar_length
        cbar_x = margin_left + (img_w - cbar_actual_w) / 2
        cbar_rect = (
            cbar_x / fig_w,
            cbar_label_space / fig_h,
            cbar_actual_w / fig_w,
            cbar_thickness / fig_h,
        )
        cbar_orientation = 'horizontal'
    else:
        # Tall image → vertical colorbar on the right
        fig_w = margin_left + img_w + cbar_gap_vertical + cbar_thickness + cbar_label_space + margin_right
        fig_h = top_space + img_h + margin_bottom

        img_rect = (
            margin_left / fig_w,
            margin_bottom / fig_h,
            img_w / fig_w,
            img_h / fig_h,
        )
        cbar_actual_h = img_h * cbar_length
        cbar_y = margin_bottom + (img_h - cbar_actual_h) / 2
        cbar_rect = (
            (margin_left + img_w + cbar_gap_vertical) / fig_w,
            cbar_y / fig_h,
            cbar_thickness / fig_w,
            cbar_actual_h / fig_h,
        )
        cbar_orientation = 'vertical'

    # Create figure with dedicated image and colorbar axes
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(img_rect)
    cax = fig.add_axes(cbar_rect)

    # Plot using xarray's imshow
    im = da.plot.imshow(
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        add_colorbar=True,
        cbar_ax=cax,
        cbar_kwargs={'orientation': cbar_orientation},
        yincrease=yincrease,
    )

    if title is not None:
        ax.set_title(title, pad=3)

    return fig, ax, im


def downsample(da: xr.DataArray, size: int = 500) -> xr.DataArray:
    """
    Downsample a DataArray by striding so that its smallest
    dimension has between `size` and `2 x size` elements.

    Assigns arange coordinates to any dimension that lacks them,
    so that the strided result preserves meaningful axis values.

    Parameters
    ----------
    da : xr.DataArray
        Input array of any number of dimensions.
    size : int, default=500
        Target approximate number of elements along the smallest dimension.

    Returns
    -------
    xr.DataArray
        Strided view of the input with reduced resolution.
    """
    # Add arange coordinates for dimensions that have none
    for dim, dimsize in zip(da.dims, da.shape):
        if dim not in da.coords:
            da = da.assign_coords({dim: np.arange(dimsize)})
    m = min(da.shape)
    stride = max(1, m//size)
    s = slice(None, None, stride)
    return da[(s,)*da.ndim]