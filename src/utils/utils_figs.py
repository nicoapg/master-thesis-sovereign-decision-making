def save_figure(fig, path_file: str, 
    width: float=1_600/1.5, height: float=900/1.5, scale: float=3) -> None:
    """Save a figure to file as .html & .png.

    Args:
        fig: figure to save.
        path_file (str): path to file.
        width (float): width of the figure.
        height (float): height of the figure.
        scale (float): scale of the figure.
    """

    fig.write_html(f"{path_file}.html")
    fig.write_image(f"{path_file}.png", width=width, height=height, scale=scale)
