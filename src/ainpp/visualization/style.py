import matplotlib.pyplot as plt
import seaborn as sns

def set_style(config=None, context="paper", style="whitegrid", palette="deep", font_family="sans-serif", dpi=300):
    """
    Sets the plotting style using Seaborn and Matplotlib.
    
    Args:
        config (dict or DictConfig, optional): Configuration dictionary. 
            If provided, overrides key arguments.
        context (str): Seaborn context.
        style (str): Seaborn style.
        palette (str): Color palette.
        font_family (str): Font family.
        dpi (int): Figure DPI.
    """
    if config:
        context = config.get("context", context)
        style = config.get("style", style)
        palette = config.get("palette", palette)
        font_family = config.get("font_family", font_family)
        dpi = config.get("dpi", dpi)

    try:
        sns.set_theme(context=context, style=style, palette=palette)
    except ImportError:
        # Fallback
        plt.style.use('ggplot')
        
    plt.rcParams.update({
        'font.family': font_family,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': dpi,
        'savefig.bbox': 'tight',
        'image.cmap': 'viridis'
    })
