"""
Centralized plot configuration module for consistent styling across all plots.
This module provides a single source of truth for all plotting configurations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional

# Font size configuration
FONT_SIZES = {
    'title': 18,        # Main plot titles
    'label': 14,        # Axis labels
    'legend': 12,       # Legend text
    'tick': 11,         # Tick labels
    'annotation': 10    # Additional text/annotations
}

# Figure size presets
FIGURE_SIZES = {
    'default': (12, 8),
    'wide': (14, 7),
    'square': (10, 10),
    'small': (8, 6),
    'heatmap': (12, 8)
}

# Color palettes
COLOR_PALETTES = {
    'coolwarm': 'coolwarm',
    'cubehelix': sns.cubehelix_palette(start=0.8, rot=-0.5, as_cmap=True),
    'cubehelix_reverse': sns.cubehelix_palette(start=0.8, rot=-0.5, as_cmap=True, reverse=True)
}

def setup_seaborn_style():
    """Initialize seaborn with default styling."""
    plt.style.use('default')
    sns.set_palette("coolwarm")
    plt.rcParams['figure.figsize'] = FIGURE_SIZES['default']
    plt.rcParams['font.size'] = FONT_SIZES['tick']

def get_coolwarm_colors(n_colors: int) -> list:
    """Get n colors from the coolwarm palette."""
    return sns.color_palette("coolwarm", n_colors)

def get_color_mapping(items: list) -> Dict[str, str]:
    """Create a color mapping for a list of items using coolwarm palette."""
    colors = get_coolwarm_colors(len(items))
    return dict(zip(items, colors))

class PlotConfig:
    """Base configuration class for all plots."""
    
    def __init__(self, figure_size: Optional[Tuple[int, int]] = None):
        self.figure_size = figure_size or FIGURE_SIZES['default']
        self.title_fontsize = FONT_SIZES['title']
        self.label_fontsize = FONT_SIZES['label']
        self.legend_fontsize = FONT_SIZES['legend']
        self.tick_fontsize = FONT_SIZES['tick']
        self.annotation_fontsize = FONT_SIZES['annotation']

class HeatmapConfig(PlotConfig):
    """Configuration specific to heatmap plots."""
    
    def __init__(self):
        super().__init__(FIGURE_SIZES['heatmap'])
        self.grid_kws = {"height_ratios": (0.9, 0.05), "hspace": 0.3}
        self.linewidths = 0.5
        self.cbar_orientation = "horizontal"
        self.max_xticks = 25

class ScatterPlotConfig(PlotConfig):
    """Configuration specific to scatter plots."""
    
    def __init__(self):
        super().__init__(FIGURE_SIZES['default'])
        self.marker_size = 50
        self.alpha = 1.0

class LineplotConfig(PlotConfig):
    """Configuration specific to line plots."""
    
    def __init__(self):
        super().__init__(FIGURE_SIZES['default'])
        self.marker = 'o'
        self.linewidth = 2

class FeaturePlotConfig(PlotConfig):
    """Configuration specific to feature plots."""
    
    def __init__(self):
        super().__init__(FIGURE_SIZES['wide'])
        self.patch_alpha = 0.3
        self.patch_color = 'yellow'
        self.drift_line_color = 'red'
        self.drift_line_style = '--'
        self.drift_line_width = 1.5

def apply_common_styling(ax, config: PlotConfig):
    """Apply common styling to a matplotlib axis."""
    ax.grid(True)
    ax.tick_params(labelsize=config.tick_fontsize)
    
    # Set label font sizes if labels exist
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=config.label_fontsize)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=config.label_fontsize)
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=config.title_fontsize)

def setup_figure_with_config(config: PlotConfig):
    """Create a figure with the specified configuration."""
    fig, ax = plt.subplots(figsize=config.figure_size)
    return fig, ax
