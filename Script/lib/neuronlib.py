"""
NeuronLib
=========

A library containing useful tools for analyzing data from neural network simulations.

Provides:
  1. `class Grapher` for drawing density distribution, matrix relation plots
  2. `class NeuralNetwork` for extracting information about neural networks, such as connection probability, degree
  3. `class NeualDynamics` for graph plotting of neural dynamics, such as firing rate and ISI distribution
  4. Other useful tools, such as reading/writing matrix from/into files, matrix and array operations

How to use the library
----------------------
Place the folder "lib" in your working directory. Do not modify its name or content.\n
Import this library as follow:
>>> import lib.neuronlib as nlib

To import specific class:
>>> from lib.neuronlib import NeuralDynamics

See class docstrings for help or further instructions, or you can use the following\n
to view all instructions such as additional arguments for graph plotting.
>>> nlib.ask_for_help()
>>> nlib.help_graph_formatting()

----------

Version: alpha03
Last update: 08 June 2022

In progress:
-

"""


import os
import csv
import math
import numpy as np
from decimal import Decimal
import matplotlib as mpl
from scipy import linalg, stats
from matplotlib import pyplot as plt, collections as mcol, ticker as tck, patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from lib.cppvector import Vector_fl


def font_settings_for_matplotlib(font: dict):
    # Use ">> print(plt.rcParams.keys())" to view all options
    plt.rc('font', **font)
    mpl.rcParams['mathtext.default'] = 'regular'

font_settings = {
    'family' : 'Charter',
    'size'   : 20
}
font_settings_for_matplotlib(font_settings)


class Grapher:

    def __init__(self) -> None:
        self._grapher_mode = 'grapher'

        self.xdata         = None
        self.ydata         = None
        self.fig           = None
        self.axes          = []
        self.ax            = None
        self._x            = []
        self._y            = []
 
        self._c            = 'b'
        self._m            = '^'
        self._ms           = 5
        self._ls           = 'none'
        self._lw           = 2
        self._mfc          = None
        self._mf           = True

        self._title        = ''
        self._plotlabel    = ''
        self._axislabel    = ['','']
        self._xlabelnplot  = []
        self._ylabelnplot  = []
        self._grid         = True
        self._gridnplot    = []
        self._minortick    = True
        self._legend       = False
        self._legendnplot  = None
        self._legendcomb   = False
        self._textbox      = ''
        self._textboxypad  = .05

        self._binsize      = None

        self._xlim         = (None, None)
        self._ylim         = (None, None)
        self._xlogscale    = False
        self._ylogscale    = False
        self._xsymlogscale = False
        self._xlinthresh   = 1
        self._ysymlogscale = False
        self._ylinthresh   = 1

    def create_plot(self, nrows=1, ncols=1, figsize=(7, 7), **options):
        """Create plot/subplots, invoke before adding data.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default `(7, 7)`\n
        nrows, ncols : int
            number of rows/columns of the subplot grid, by default `1`\n
        sharex, sharey : bool or {`'none'`, `'all'`, `'row'`, `'col'`}
            control sharing of properties among x or y axes, by default `False`:
            - `True` or `'all'`: x- or y-axis will be shared among all subplots
            - `False` or `'none'`: each subplot x- or y-axis will be independent
            - `'row'`: each subplot row will share an x- or y-axis
            - `'col'`: each subplot column will share an x- or y-axis\n
        hratio, wratio : array of int
            define the relative height of rows/width of columns
            e.g., hratio=`[1, 3, 2]` for nrows: 3\n
        dpi : int, optional
            DPI, by default `150`\n
        """
        sharex = False
        sharey = False
        dpi = 150
        hratio = None
        wratio = None
        for key, value in options.items():
            if key == 'sharex': sharex = value
            elif key == 'sharey': sharey = value
            elif key == 'hratio': hratio = value
            elif key == 'wratio': wratio = value
        self.__init__()
        if nrows == 1 and ncols == 1: self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi, gridspec_kw={'height_ratios':hratio, 'width_ratios':wratio})
        else:
            self.fig, self.axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey, figsize=figsize, dpi=dpi, gridspec_kw={'height_ratios':hratio, 'width_ratios':wratio})
            self.axes = self.axes.flatten()

    def inset_to_parent_plot(self, parent_plot: object, **inset_format):
        """Add to a parent plot as an inset plot.

        Parameters
        ----------
        parent_plot : object of Grapher
            the parent plot to be inset into
        **inset_format : dict, required
            inset axes formatting (see Notes)

        Notes
        -----
        inset axes formatting
        - width : float or str, e.g., `1` : 1 inch ; `"70%"` : 70% of parent_bbox
        - height : float or str, e.g., `1` : 1 inch ; `"70%"` : 70% of parent_bbox
        - loc : int, `1` or `2` or `3` or `4`
        - borderpad : float
        - bbox_to_anchor : tuple, e.g., `(.34, .34, .75, .576)`

        Examples
        --------
        >>> g = GraphHistogramDistribution()
        ... g.create_plot()
        ... g.add_data(data, 0.2)
        ... g.make_plot()
        ... 
        >>> inset_format = {
        ...     "width":"70%",
        ...     "height":"70%",
        ...     "loc":1,
        ...     "borderpad":3,
        ...     "bbox_to_anchor":(.34, .34, .75, .576),
        ... }
        ... g_inset = GraphHistogramDistribution()
        ... g_inset.inset_to_parent_plot(g, **inset_format)
        ... g_inset.add_data(data, 0.2)
        ... g_inset.stylize_plot(dict(c='r'))
        ... g_inset.make_plot()
        ... 
        >>> g.show_plot()
        """
        self.ax = inset_axes(parent_plot.ax, **inset_format, bbox_transform=parent_plot.ax.transAxes)
        self.ax.tick_params(labelleft=True, labelbottom=True)
        self.fig = parent_plot.fig

    def add_data(self, xdata, ydata, **kwargs):
        """Add data to be plotted.

        Parameters
        ----------
        xdata : numpy.ndarray
            data in x-axis
        ydata : numpy.ndarray
            data in y-axis
        """
        for key, value in kwargs.items():
            print('Warning: the optional argument [{}] is not supported'.format(key))
        self._x = np.array(xdata).flatten()
        self._y = np.array(ydata).flatten()
        if len(self._x) != len(self._y):
            err = 'DataValueError: xdata and ydata must have same size but xdata with size: {} and ydata with size: {} are given'.format(len(self._x), len(self._y))
            raise ValueError(err)

    def stylize_plot(self, style):
        """Stylize matplotlib axes.

        Parameters
        ----------
        color | c : str
            color of markers and lines, e.g., `'b', 'C1', 'darkorange'`\n
        marker | m : str
            marker style, e.g., `'o', '^', 'D', ','`\n
        markersize | ms : float
            marker size\n
        markerfacecolor | mfc : str
            marker face color, default: same as `color`
        markerfilled | mf : bool
            solid marker or hollow marker, default `True`
        linestyle | ls : str
            line style, e.g., `'-', '--', '-.', ':'`\n
        lineweight | lw : float
            line weight\n
        """
        for key, value in style.items():
            if key == 'color' or key == 'c': self._c = value
            elif key == 'marker' or key == 'm': self._m = value
            elif key == 'markersize' or key == 'ms': self._ms = value
            elif key == 'markerfacecolor' or key == 'mfc': self._mfc = value
            elif key == 'markerfilled' or key == 'mf': self._mf = value
            elif key == 'linestyle' or key == 'ls': self._ls = value
            elif key == 'lineweight' or key == 'lw': self._lw = value
            else: print('Warning: the optional argument [{}] is not supported'.format(key))
        if not self._mf: self._mfc = 'none'

    def label_plot(self, label):
        """Label matplotlib axes.

        Parameters
        ----------
        title : str
            figure title\n
        plotlabel : str
            plot label to be displayed in legend\n
        axislabel : list of str
            x-axis and y-axis labels, e.g., `['time', 'voltage']`\n
        xlabel : str
            x-axis label\n
        ylabel : str
            y-axis label\n
        xlabelnplot : list of int | int
            show y-axis label in selected subplots, otherwise disable\n
        ylabelnplot : list of int | int
            show y-axis label in selected subplots, otherwise disable\n
        legend : bool | list
            - bool: legend on/off, default: `False`
            - list: list of plotlabels (for single plot)
            - list: `[(subplotindex, plotlabels ...), ...]` (for subplots)\n
        legendcombine : bool
            combine legends if there are multiple subplots, default: `False`\n
        textbox : str | list
            information to be displayed at top-left corner
            - list: `[(subplotindex, text ...), ...]` (for subplots)\n
        """
        for key, value in label.items():
            if key == 'title' or key == 'tl':
                self._title = value
                if type(self._title) != str: raise TypeError('the argument [title]/[tl] must be of type "str"')
            elif key == 'plotlabel' or key == 'pl':
                self._plotlabel = value
                if type(self._plotlabel) != str: raise TypeError('the argument [plotlabel]/[pl] must be of type "str"')
            elif key == 'axislabel' or key == 'al':
                if type(self._axislabel) != list: raise TypeError('the argument [axislabel]/[al] must be of type "list of str"')
                self._axislabel[0] = '{}'.format(value[0])
                self._axislabel[1] = '{}'.format(value[1])
                if type(self._axislabel[0]) != str and type(self._axislabel[1]) != str: raise TypeError('the argument [axislabel]/[al] must be of type "list of str"')
            elif key == 'xlabel' or key == 'xl':
                self._axislabel[0] = '{}'.format(value)
                if type(self._axislabel[0]) != str: raise TypeError('the argument [xlabel]/[xl] must be of type "str"')
            elif key == 'ylabel' or key == 'yl':
                self._axislabel[1] = '{}'.format(value)
                if type(self._axislabel[1]) != str: raise TypeError('the argument [ylabel]/[yl] must be of type "str" but {} is given instead'.format(type(self._axislabel[1])))
            elif key == 'xlabelnplot' or key == 'xln':
                self._xlabelnplot = value
                if type(self._xlabelnplot) != list and type(self._xlabelnplot) != int: raise TypeError('the argument [xlabelnplot]/[xln] must be of type "list of int" or "int" but {} is given instead'.format(type(self._xlabelnplot)))
            elif key == 'ylabelnplot' or key == 'yln':
                self._ylabelnplot = value
                if type(self._ylabelnplot) != list and type(self._ylabelnplot) != int: raise TypeError('the argument [ylabelnplot]/[yln] must be of type "list of int" or "int" but {} is given instead'.format(type(self._ylabelnplot)))
            elif key == 'legend' or key == 'lg':
                self._legend = value
                if type(self._legend) != bool and type(self._legend) != list: raise TypeError('the argument [legend]/[lg] must be of type "bool" or "list" but {} is given instead'.format(type(self._legend)))
            elif key == 'legendnplot' or key == 'lgn':
                self._legendnplot = value
                print('Warning: the optional argument [legendnplot] is not yet supported')
            elif key == 'legendcombine' or key == 'lgc':
                self._legendcomb = value
                if type(self._legendcomb) != bool: raise TypeError('the argument [legendcomb]/[lgc] must be of type "bool"')
            elif key == 'textbox' or key == 'tb':
                self._textbox = value
                if type(self._textbox) != str and type(self._textbox) != list: raise TypeError('the argument [textbox]/[tb] must be of type "str" or "list of str"')
            else: print('Warning: the optional argument [{}] is not supported'.format(key))
        if type(self._xlabelnplot) == int: self._xlabelnplot = [self._xlabelnplot]
        elif type(self._xlabelnplot) == list: pass
        else: raise TypeError('the argument [xlabelnplot]/[xln] must be of type "list of int" or "int" but {} is given instead'.format(type(self._xlabelnplot)))
        if type(self._ylabelnplot) == int: self._ylabelnplot = [self._ylabelnplot]
        elif type(self._ylabelnplot) == list: pass
        else: raise TypeError('the argument [ylabelnplot]/[yln] must be of type "list of int" or "int" but {} is given instead'.format(type(self._ylabelnplot)))

    def set_scale(self, scale, nplot=None):
        """Set scale of matplotlib axes.

        Parameters
        ----------
        nplot : int
            set the scale of the n-th sub-plot if there are multiple subplots\n
        grid : bool
            grid on/off, default: `True`\n
        gridnplot : list of int
            the indexes of subplot whose grid is on, default: all grids on\n
        minortick : bool
            minortick on/off, default: `True`\n
        xlim : tuple
            horizontal plot range, default: fitted to data\n
        ylim : tuple
            vertical plot range, default: fitted to data\n
        xlogscale : bool
            use log scale in x-axis, default: `False`\n
        ylogscale : bool
            use log scale in y-axis, default: `False`\n
        xsymlogscale : bool
            use symmetric log scale in x-axis (should be used with `xlinthresh`), default: `False`\n
        xlinthresh : float
            the threshold of linear range when using `xsymlogscale`, default: `1`\n
        ysymlogscale : bool
            use symmetric log scale in y-axis (should be used with `ylinthresh`), default: `False`\n
        ylinthresh : float
            the threshold of linear range when using `ysymlogscale`, default: `1`\n
        """
        for key, value in scale.items():
            if key == 'grid' or key == 'gd':
                self._grid = value
                if type(self._grid) != bool: raise TypeError('the argument [grid]/[gd] must be of type "bool"')
            elif key == 'gridnplot' or key == 'gdn':
                self._gridnplot = value
                if type(self._gridnplot) != list: raise TypeError('the argument [gridnplot] must be of type "list"')
            elif key == 'minortick':
                self._minortick = value
                if type(self._minortick) != bool: raise TypeError('the argument [minortick] must be of type "bool"')
            elif key == 'xlim': self._xlim = value
            elif key == 'ylim': self._ylim = value
            elif key == 'xlogscale' or key == 'xlg':
                self._xlogscale = value
                if type(self._xlogscale) != bool: raise TypeError('the argument [xlogscale]/[xlg] must be of type "bool"')
                # if self._grapher_mode == 'densitydistribution':
                #     print('Warning: to show data points in log scale, use argument [logdata] in "add_data" instead')
            elif key == 'ylogscale' or key == 'ylg':
                self._ylogscale = value
                if type(self._ylogscale) != bool: raise TypeError('the argument [ylogscale]/[ylg] must be of type "bool"')
            elif key == 'xsymlogscale':
                self._xsymlogscale = value
                if type(self._xsymlogscale) != bool: raise TypeError('the argument [xsymlogscale] must be of type "bool"')
                # if self._grapher_mode == 'densitydistribution':
                #     print('Warning: to show data points in log scale, use argument [symlogdata] in "add_data" instead')
            elif key == 'xlinthresh':
                self._xlinthresh = value
                if self._grapher_mode == 'densitydistribution':
                    print('Warning: use argument [linthresh] in "add_data" instead')
            elif key == 'ysymlogscale':
                self._ysymlogscale = value
                if type(self._ysymlogscale) != bool: raise TypeError('the argument [ysymlogscale] must be of type "bool"')
            elif key == 'ylinthresh': self._ylinthresh = value
            else: print('Warning: the optional argument [{}] is not supported'.format(key))
        # self.ax.tick_params(axis='both', direction='in', which='both')
        # locmin = tck.LogLocator(base=10.0, subs=(.2,.4,.6,.8), numticks=12)
        # self.ax.yaxis.set_minor_locator(locmin)
        # self.ax.yaxis.set_minor_formatter(tck.NullFormatter())
        # self.ax.yaxis.set_minor_locator(tck.AutoMinorLocator(4))
        if nplot == None:
            if self.ax == None:
                for ax in self.axes:
                    if self._minortick: ax.minorticks_on()
                    else: ax.minorticks_off()
                    if not all(x is None for x in self._xlim): ax.set_xlim(self._xlim)
                    if not all(x is None for x in self._ylim): ax.set_ylim(self._ylim)
                    if self._ylogscale: ax.set_yscale('log')
                    elif self._ysymlogscale: ax.set_yscale('symlog', linthresh=self._ylinthresh)
                    if self._xlogscale: ax.set_xscale('log')
                    elif self._xsymlogscale: ax.set_xscale('symlog', linthresh=self._xlinthresh)
            else:
                if self._minortick: self.ax.minorticks_on()
                else: self.ax.minorticks_off()
                if not all(x is None for x in self._xlim): self.ax.set_xlim(self._xlim)
                if not all(x is None for x in self._ylim): self.ax.set_ylim(self._ylim)
                if self._ylogscale: self.ax.set_yscale('log')
                elif self._ysymlogscale: self.ax.set_yscale('symlog', linthresh=self._ylinthresh)
                if self._xlogscale: self.ax.set_xscale('log')
                elif self._xsymlogscale: self.ax.set_xscale('symlog', linthresh=self._xlinthresh)
        elif type(nplot) == int:
            if self._minortick: self.axes[nplot].minorticks_on()
            else: self.axes[nplot].minorticks_off()
            if not all(x is None for x in self._xlim): self.axes[nplot].set_xlim(self._xlim)
            if not all(x is None for x in self._ylim): self.axes[nplot].set_ylim(self._ylim)
            if self._ylogscale: self.axes[nplot].set_yscale('log')
            elif self._ysymlogscale: self.axes[nplot].set_yscale('symlog', linthresh=self._ylinthresh)
            if self._xlogscale: self.axes[nplot].set_xscale('log')
            elif self._xsymlogscale: self.axes[nplot].set_xscale('symlog', linthresh=self._xlinthresh)
        elif type(nplot) == list and type(nplot[0]) == int:
            for n in nplot:
                if self._minortick: self.axes[nplot].minorticks_on()
                else: self.axes[nplot].minorticks_off()
                if not all(x is None for x in self._xlim): self.axes[n].set_xlim(self.xlim)
                if not all(x is None for x in self._ylim): self.axes[n].set_ylim(self.ylim)
                if self._ylogscale: self.axes[n].set_yscale('log')
                elif self._ysymlogscale: self.axes[n].set_yscale('symlog', linthresh=self._ylinthresh)
                if self._xlogscale: self.axes[n].set_xscale('log')
                elif self._xsymlogscale: self.axes[n].set_xscale('symlog', linthresh=self._xlinthresh)
        else:
            print('ArgumentTypeError: \"nplot\" should be type: int or list of int, but {} was given'.format(str(type(nplot))))

    def _set_fmt(self):
        textbox_fontsize_multiplier = 0.75
        if self._minortick: plt.minorticks_on()
        else: plt.minorticks_off()
        if self.ax != None:
            if self._textbox == '': _titlepad = 15
            else: _titlepad = 30
            if self._title != '': self.ax.set_title(self._title, pad=_titlepad, loc='left')
            if self._grid:
                self.ax.grid(True, which='major', axis='both', color='0.6')
                self.ax.grid(True, which='minor', axis='both', color='0.85', linestyle='--')
            else: self.ax.grid(False)
            if type(self._legend) == list: self.ax.legend(self._legend)
            elif self._legend: self.ax.legend()
            self.ax.set(xlabel=self._axislabel[0], ylabel=self._axislabel[1])
            if self._textbox != '':
                props = dict(boxstyle='round', pad=0.1, facecolor='white', edgecolor='none', alpha=0.75)
                self.ax.text(0.00001, 1.0+self._textboxypad, self._textbox, fontsize=font_settings['size']*textbox_fontsize_multiplier,
                             verticalalignment='top', transform=self.ax.transAxes, bbox=props)
        elif len(self.axes) != 0:
            if self._textbox == '': _titlepad = 15
            else: _titlepad = 30
            if self._title != '': self.axes[0].set_title(self._title, pad=_titlepad, loc='left')
            if self._textbox != '':
                if type(self._textbox) == str:
                    for n in range((self.axes)):
                        props = dict(boxstyle='round', pad=0.08, facecolor='white', edgecolor='none', alpha=0.75)
                        self.axes[n].text(0.00001, 1.0+self._textboxypad, self._textbox, fontsize=font_settings['size']*textbox_fontsize_multiplier,
                                            verticalalignment='top', transform=self.axes[0].transAxes, bbox=props)
                elif type(self._textbox) == list:
                    for tb in self._textbox:
                        props = dict(boxstyle='round', pad=0.08, facecolor='white', edgecolor='none', alpha=0.75)
                        self.axes[tb[0]].text(0.00001, 1.0+self._textboxypad, tb[1], fontsize=font_settings['size']*textbox_fontsize_multiplier,
                                              verticalalignment='top', transform=self.axes[tb[0]].transAxes, bbox=props)
            for ax in self.axes:
                if self._grid:
                    ax.grid(True, which='major', axis='both', color='0.6')
                    ax.grid(True, which='minor', axis='both', color='0.85', linestyle='--')
                else: ax.grid(False)
                if type(self._legend) == bool and self._legend and not self._legendcomb: ax.legend()
                if len(self._xlabelnplot) == 0: ax.set(xlabel=self._axislabel[0])
                if len(self._ylabelnplot) == 0: ax.set(ylabel=self._axislabel[1])
            for n in self._xlabelnplot: self.axes[n].set(xlabel=self._axislabel[0])
            for n in self._ylabelnplot: self.axes[n].set(ylabel=self._axislabel[1])
            if len(self._gridnplot) != 0:
                for n in range((self.axes)):
                    if n in self._gridnplot:
                        self.axes[n].grid(True, which='major', axis='both', color='0.6')
                        self.axes[n].grid(True, which='minor', axis='both', color='0.85', linestyle='--')
                    else: self.axes[n].grid(False)
            if type(self._legend) == list:
                for lg in self._legend: self.axes[lg[0]].legend(lg[1:])
            if self._legend and self._legendcomb:
                bbox = self.axes[0].get_position()
                self.axes[0].set_position([bbox.x0, bbox.y0, bbox.width * 0.9, bbox.height])
                self.axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2))
            # ax.legend(title='Node index', loc='center left', bbox_to_anchor=(1, 0.5))
            pass
        else:
            err = 'the graph has not been initialized, please invoke method \"create_plot()\" to create plot/subplots'
            raise Exception(err)

    def apply_format(self):
        self._set_fmt()

    def make_plot(self, nplot=0, **options):
        """Make plot, invoke before exporting plot.

        Parameters
        ----------
        nplot : int, optional
            the index of subplot to be taken effect on, by default 0
        apply_format : bool, optional
            apply all formatings, e.g., style, label, etc., default `False`
        """
        apply_all_format = False
        apply_style = True
        for key, value in options.items():
            if key == 'apply_all_format': apply_all_format = value
            elif key == 'apply_style': apply_style = value
            else: print('Warning: the optional argument [{}] is not supported'.format(key))
        if apply_all_format: self._set_fmt()
        if self.ax != None:
            if self._x.any() == None or len(self._x) == 0 or self._y.any() == None or len(self._y) == 0:
                self.ax.add_patch(patches.Rectangle((.01, .01), .98, .98, color='1', alpha=.75, fill=True, clip_on=False, transform=self.ax.transAxes, zorder=5))
                self.ax.text(.5, .5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes, zorder=6)
            else:
                self.ax.plot(self._x, self._y, c=self._c, marker=self._m, ms=self._ms, mfc=self._mfc,
                    ls=self._ls, lw=self._lw, label=self._plotlabel, zorder=2)
        elif len(self.axes) != 0:
            if self._x.any() == None or len(self._x) == 0 or self._y.any() == None or len(self._y) == 0:
                self.axes[nplot].add_patch(patches.Rectangle((.01, .01), .98, .98, color='1', alpha=.75, fill=True, clip_on=False, transform=self.axes[nplot].transAxes, zorder=5))
                self.axes[nplot].text(.5, .5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=self.axes[nplot].transAxes, zorder=6)
            else:
                self.axes[nplot].plot(self._x, self._y, c=self._c, marker=self._m, ms=self._ms, mfc=self._mfc,
                    ls=self._ls, lw=self._lw, label=self._plotlabel, zorder=2)
        else:
            err = 'the graph has not been initialized, please invoke method \"create_plot()\" to create plot/subplots'
            raise Exception(err)

    def save_plot(self, filename: str, label='', ext='png', path='', tight_layout=True):
        """Save plot into a file.

        Parameters
        ----------
        filename : str
            name of the output file
        label : str, optional
            label attached at the end of the file name, by default `''`
        ext : str, optional
            file extension, by default `'png'`
        path : str, optional
            path to the output file, by default `''`
        tight_layout : bool, optional
            enable tight layout, by default `True`
        """
        self._set_fmt()
        if label != '': filename += ' ({})'.format(label)
        if ext != '': filename += '.' + ext
        if tight_layout: self.fig.tight_layout()
        self.fig.savefig(os.path.join(path, filename))

    def show_plot(self):
        """Preview plot in matplotlib build-in console.
        """
        self._set_fmt()
        self.fig.tight_layout()
        plt.show()

    def save_plot_info(self, filename: str, label='', path=''):
        """Save plot information, e.g., binsize, data points

        Parameters
        ----------
        filename : str
            name of the output file
        label : str, optional
            label attached at the end of the file name, by default ''
        path : str, optional
            path to the output file, by default ''
        """
        try:
            if path != '': os.mkdir(path)
        except OSError: pass
        if label != '': filename += ' ({})'.format(label)
        filepath = os.path.join(path, filename)
        self._save_plot_info(filepath)
        if self._grapher_mode == 'densitydistribution':
            self._save_data_points(filepath)

    def _save_plot_info(self, filepath: str):
        with open(filepath+'_plotinfo.txt', 'w') as f:
            if self._grapher_mode == 'densitydistribution':
                f.write('bin size:\t{}\n'.format(self._binsize))

    def _save_data_points(self, filepath: str):
        with open(filepath+'_datapoints.txt', 'w') as f:
            for x, y in zip(self._x, self._y):
                f.write('{}\t{}\n'.format(x, y))

class GraphDataRelation(Grapher):

    def __init__(self) -> None:
        """Graph the relation between two sets of data.

        Notes
        -----

        Steps (functions to be called in order):
        1. `create_plot()`
        2. `add_data()`
        3. (optional)
        4. `make_plot()`
        5. `save_plot()` or `show_plot()`
        
        Optional (step 4):
        - `stylize_plot()`
        - `label_plot()`
        - `set_scale()`
        - `draw_xyline()`
        - `fit_linear()`

        Examples
        --------
        >>> A = [[1, 3], [2, 2]]
        ... B = [[2, 3], [1, 4]]
        >>> g = GraphDataRelation()
        ... g.create_plot()
        ... g.add_data(A, B)
        ... coef = g.fit_linear()
        ... g.make_plot()
        ... g.show_plot()
        """
        super().__init__()
        self._m            = 'o'
        self._ms           = 1.5
        self._ls           = 'none'
        self._grapher_mode = 'datarelation'

    def add_data(self, xdata, ydata, **kwargs):
        """Add data to be plotted.

        Parameters
        ----------
        xdata : numpy.ndarray
            data in x-axis
        ydata : numpy.ndarray
            data in y-axis
        """
        for key, value in kwargs.items():
            print('Warning: the optional argument [{}] is not supported'.format(key))
        self._x = np.array(xdata).flatten()
        self._y = np.array(ydata).flatten()
        if len(self._x) != len(self._y):
            err = 'DataValueError: xdata and ydata must have same size but xdata with size: {} and ydata with size: {} are given'.format(len(self._x), len(self._y))
            raise ValueError(err)

    def draw_xyline(self, color='k'):
        min_elem = min(np.amin(self._x[self._x != None]), np.amin(self._y[self._y != None]))
        max_elem = max(np.amax(self._x[self._x != None]), np.amax(self._y[self._y != None]))
        elem_range = np.linspace(min_elem, max_elem)
        self.ax.plot(elem_range, elem_range, c=color, ls='--', lw=1, zorder=1)

    def fit_linear(self, color='darkorange') -> tuple:
        min_elem = min(np.amin(self._x), np.amin(self._y))
        max_elem = max(np.amax(self._x), np.amax(self._y))
        elem_range = np.linspace(min_elem, max_elem)
        coef = np.polyfit(self._x, self._y, 1)  # slope = coef[0], y-intercept = coef[1]
        poly1d_fn = np.poly1d(coef)
        self.ax.plot(elem_range, poly1d_fn(elem_range), c=color, ls='-.', lw=1.25, zorder=3)
        return coef

class GraphMatrixRelation(GraphDataRelation):

    def __init__(self) -> None:
        """Graph the relation between two matrices.

        Notes
        -----

        Steps (functions to be called in order):
        1. `create_plot()`
        2. `add_data()`
        3. (optional)
        4. `make_plot()`
        5. `save_fig()` or `show_fig()`
        
        Optional (step 4):
        - `stylize_plot()`
        - `label_plot()`
        - `set_scale()`
        - `draw_xyline()`
        - `fit_linear()`

        Examples
        --------

        >>> A = [[1, 3], [2, 2]]
        ... B = [[2, 3], [1, 4]]
        >>> g = GraphMatrixRelation()
        ... g.create_plot()
        ... g.add_data(A, B)
        ... coef = g.fit_linear()
        ... g.make_plot()
        ... g.show_plot()
        """
        super().__init__()
        self._onlyOffDiag  = False
        self._onlyDiag     = False
        self._ms           = 1
        self._grapher_mode = 'matrixrelation'

    def add_data(self, xdata, ydata, **kwargs):
        """Add data to be plotted.

        Parameters
        ----------
        xdata : numpy.ndarray
            data in x-axis
        ydata : numpy.ndarray
            data in y-axis
        onlyOffDiag : bool
            show only off-diagonal elements, default: `False`
        onlyDiag : bool
            show only diagonal elements, default: `False`
        """
        for key, value in kwargs.items():
            if key == 'onlyDiag': self._onlyDiag = value
            elif key == 'onlyOffDiag': self._onlyOffDiag = value
            else: print('Warning: the optional argument [{}] is not supported'.format(key))
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        if self.onlyOffDiag: # keep only off-diagonals
            self._x = self._rm_diag(self.xdata).flatten()
            self._y = self._rm_diag(self.ydata).flatten()
            self._axislabel[0] += ' (off-diagonals)'
            self._axislabel[1] += ' (off-diagonals)'
        elif self._onlyDiag: # keep only diagonals
            self._x = np.diag(self.xdata)
            self._y = np.diag(self.ydata)
            self._axislabel[0] += ' (diagonals)'
            self._axislabel[1] += ' (diagonals)'
        else:
            self._x = self.xdata.flatten()
            self._y = self.ydata.flatten()
        if self.xdata.shape != self.ydata.shape:
            err = 'DataValueError: xdata and ydata must be matrices of same size but xdata with size: {} and ydata with size: {} are given'.format(self.xdata.shape, self.ydata.shape)
            raise ValueError(err)

    def _rm_diag(self, A: np.ndarray) -> np.ndarray:
        if A.shape[0] == A.shape[1]:
            return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)
        else:
            err = 'DataValueError: the input matrix must be square in order to remove its diagonal elements'
            raise ValueError(err)

class GraphHistogramDistribution(Grapher):

    def __init__(self) -> None:
        """Graph the density distribution of a set of data.

        Steps (functions to be called in order):
        1. `create_plot()`
        2. `add_data()`
        3. (optional)
        4. `make_plot()`
        5. `save_plot()` or `show_plot()`
        
        Optional (step 4):
        - `stylize_plot()`
        - `label_plot()`
        - `set_scale()`
        - `fit_gaussian()`

        Examples
        --------

        >>> A = np.random.normal(0, 10, 5000)
        >>> g = GraphHistogramDistribution()
        ... g.create_plot()
        ... g.add_data(A, 0.1)
        ... g.make_plot()
        ... g.show_plot()
        """
        super().__init__()
        self._ms = 7.5
        self._lw = 2
        self._ls = '-'
        self._density_normalize = False
        self._normalize_to_factor = 1
        self._axislabel[1] = 'Number of occurrence'
        self._grapher_mode = 'histogramdistribution'

    def add_data(self, data, binsize: float, **kwargs):
        """Add data to be plotted.

        Parameters
        ----------
        data : np.ndarray
            data
        binsize : float
            bin size
        normalize_to_factor : float
            normalize density distribution to a factor, default: 1.0 (normalized)
        logdata : bool
            use log scale in x-data, force logscale in x-axis, default: `False`
        symlogdata : bool
            use symmetric log scale in x-data (should be used with `linthresh`), force symlogscale in x-axis, default: `False`
        linthresh : float
            the threshold of linear range when using `symlogdata`, default: 1
        """
        for key, value in kwargs.items():
            if key == 'normalize_to_factor': self._normalize_to_factor = value
            elif key == 'logdata': self._xlogscale = value
            elif key == 'symlogdata': self._xsymlogscale = value
            elif key == 'linthresh': self._xlinthresh = value
            else: print('Warning: the optional argument [{}] is not supported'.format(key))

        if data.size == 0 or all(data == None):
            self._x, self._y = np.array([]), np.array([])
        else:
            self._binsize = binsize
            self.xdata = np.array(data).flatten()
            min_elem = np.amin(self.xdata); max_elem = np.amax(self.xdata)
            if self._xlogscale:
                pos_nearzero_datum = np.amin(np.array(data)[np.array(data) > 0])
                number_of_bins = math.ceil((math.log10(max_elem) - math.log10(pos_nearzero_datum)) / self._binsize)
                density, binedge = np.histogram(
                    data, bins=np.logspace(math.log10(pos_nearzero_datum), math.log10(pos_nearzero_datum)
                                        + number_of_bins*self._binsize, number_of_bins)
                )
            elif self._xsymlogscale:
                # pos_nearzero_datum = np.amin(np.array(data)[np.array(data) > 0])
                # neg_nearzero_datum = np.amax(np.array(data)[np.array(data) < 0])
                pos_nearzero_datum = self._xlinthresh
                neg_nearzero_datum = -self._xlinthresh
                lin_bins_amt = math.ceil((max_elem - min_elem) / self.binsize)
                pos_bins_amt = math.ceil((math.log10(max_elem) - math.log10(pos_nearzero_datum)) / self._binsize)
                neg_bins_amt = math.ceil((math.log10(-min_elem) - math.log10(-neg_nearzero_datum)) / self._binsize)
                linbins = np.linspace(neg_nearzero_datum, pos_nearzero_datum, lin_bins_amt)
                pos_logbins = np.logspace(math.log10(pos_nearzero_datum), math.log10(pos_nearzero_datum)+pos_bins_amt*self._binsize, pos_bins_amt)
                neg_logbins = -np.flip(np.logspace(math.log10(-neg_nearzero_datum), math.log10(-neg_nearzero_datum)+neg_bins_amt*self._binsize, neg_bins_amt))
                density, binedge = np.histogram(data, bins=np.concatenate((neg_logbins, linbins, pos_logbins)), density=self._density_normalize)
            else:
                bins_amt = math.ceil((max_elem - min_elem) / self._binsize)
                density, binedge = np.histogram(data, bins=np.linspace(min_elem, min_elem+bins_amt*self._binsize, bins_amt), density=self._density_normalize)
            if self._density_normalize:
                density = np.array(density, dtype=float)
                density /= np.dot(density, np.diff(binedge)) # normalization
                if self._normalize_to_factor != 1: density *= self._normalize_to_factor
            self._x = (binedge[1:] + binedge[:-1]) / 2
            self._y = density

    def fit_gaussian(self, nplot=0, **kwargs):
        """Fit the data with a Gaussian curve.

        Parameters
        ----------
        nplot : int, optional
            the index of subplot to be taken effect on, by default 0
        color | c : str
            color of markers and lines, default: `'r'`
        lineweight | lw : float
            line weight, default: `2.5`
        """
        c = 'r'; lw = 2.5
        for key, value in kwargs.items():
            if key == 'color' or key == 'c': c = value
            elif key == 'lineweight' or key == 'lw': lw = value
            else: print('Warning: the optional argument [{}] is not supported'.format(key))
        mu = np.mean(self.xdata); sigma = np.std(self.xdata)
        norm_xval = np.linspace(mu - 4*sigma, mu + 4*sigma, 150)
        if self.ax != None:
            self.ax.plot(norm_xval, stats.norm.pdf(norm_xval, mu, sigma), '--', c=c, lw=lw, label='Normal distribution', zorder=1)
        else:
            self.axes[nplot].plot(norm_xval, stats.norm.pdf(norm_xval, mu, sigma), '--', c=c, lw=lw, label='Normal distribution', zorder=1)

class GraphDensityDistribution(GraphHistogramDistribution):

    def __init__(self) -> None:
        """Graph the density distribution of a set of data.

        Notes
        -----

        Steps (functions to be called in order):
        1. `create_plot()`
        2. `add_data()`
        3. (optional)
        4. `make_plot()`
        5. `save_plot()` or `show_plot()`
        
        Optional (step 4):
        - `stylize_plot()`
        - `label_plot()`
        - `set_scale()`
        - `fit_gaussian()`

        Examples
        --------

        >>> A = np.random.normal(0, 10, 5000)
        >>> g = GraphDensityDistribution()
        ... g.create_plot()
        ... g.add_data(A, 0.1)
        ... g.make_plot()
        ... g.show_plot()
        """
        super().__init__()
        self._ms = 7.5
        self._lw = 2
        self._ls = '-'
        self._density_normalize = True
        self._normalize_to_factor = 1
        self._axislabel[1] = 'Probability density'
        self._grapher_mode = 'densitydistribution'


class Tool:

    def __init__(self) -> None:
        self._mode = ''

    def _init_input_output_path(self, input_folder: str, output_folder: str) -> tuple:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        index = this_dir.rfind('\\')
        if input_folder != None:
            if ':' in input_folder: input_path = input_folder
            else: input_path = os.path.join(this_dir[:index], input_folder)
        else: input_path = ''
        if output_folder != None:
            if ':' in output_folder: output_path = output_folder
            else: output_path = os.path.join(this_dir[:index], output_folder)
            try: os.mkdir(output_path)
            except FileExistsError: pass
        else: output_path = ''
        return input_path, output_path

    def _graph_distribution_subroutine(self, data: np.ndarray, binsize: float, style=None, label=None, scale=None, graph=None, **options):
        _xlabel = 'variable x'
        _filename = 'Distribution plot'
        title = ''
        textbox = 'default'
        fitgaussian = 'none'
        filelabel = ''
        fileextension = 'png'
        subplotindex = 0
        saveplotinfo = False

        xlogscale = False
        density = True
        histogram = False

        for key, value in options.items():
            if key == '_xlabel': _xlabel = value
            elif key == '_filename': _filename = value
            elif key == 'title': title = value
            elif key == 'textbox': textbox = value
            elif key == 'fitgaussian': fitgaussian = value
            elif key == 'filelabel': filelabel = value
            elif key == 'fileextension': fileextension = value
            elif key == 'subplotindex': subplotindex = value
            elif key == 'saveplotinfo': saveplotinfo = value
            elif key == 'xlogscale': xlogscale = value
            elif key == 'density': density = value
            elif key == 'histogram': histogram = value
            else: print('Warning: the optional argument [{}] is not supported'.format(key))
        if histogram: density = False

        if data.size != 0:
            # Remove 'None', if exist, from data
            data = data[data != None]

            min_datum = np.amin(data)
            max_datum = np.amax(data)
            if (max_datum - min_datum)/10 < binsize:
                print('Warning: the chosen bin size = {} may be too large, it is recommended to use a value smaller than {}'.format(binsize, round_to_n((max_datum-min_datum)/10, 2)))

        if graph == None:
            graph = GraphDensityDistribution()
            graph.create_plot(figsize=(9,7))
            graph.add_data(data, binsize, logdata=xlogscale)
            if fitgaussian != 'none':
                if type(fitgaussian) == tuple: graph.fit_gaussian(c=fitgaussian[0], lw=fitgaussian[1])
                else: graph.fit_gaussian()
            if textbox == 'default':
                if self._mode == 'NeuralNetwork':
                    textbox = 'min: {:.3} Hz | max: {:.3} | bin size: {:.5}'.format(float(min_datum), float(max_datum), float(binsize))
                elif self._mode == 'NeuralDynamics':
                    if not xlogscale:
                        textbox = 'T: {:.7} ms | dt: {:.5} | min: {:.3} Hz | max: {:.3} | bin size: {:.5}'.format(float(self._config[2]),
                                float(self._config[1]), float(min_datum), float(max_datum), float(binsize))
                    else:
                        textbox = 'T: {:.7} ms | dt: {:.5} | min: {:.3} Hz | max: {:.3} | bin size (log): {:.5}'.format(float(self._config[2]),
                                float(self._config[1]), float(min_datum), float(max_datum), float(binsize))
                graph.label_plot(dict(xlabel=_xlabel, title=title, textbox=textbox))
            else: graph.label_plot(dict(xlabel=_xlabel, title=title, textbox=textbox))
            if style != None: graph.stylize_plot(style)
            if label != None: graph.label_plot(label)
            if xlogscale: graph.set_scale(dict(xlogscale=True))
            if scale != None: graph.set_scale(scale)
            graph.make_plot()
            graph.save_plot(_filename, filelabel, ext=fileextension, path=self._output_path)
            if saveplotinfo: graph.save_plot_info(_filename, filelabel, path=self._output_path)
        else:
            if style != None: graph.stylize_plot(style)
            if label != None: graph.label_plot(label)
            graph.add_data(data, binsize, logdata=xlogscale)
            graph.make_plot(subplotindex)

    def _graph_timeseries_subroutine(self, timeseries: np.ndarray, neuron_index: int, trim=(None, None), style=None, label=None, scale=None, graph=None, **options):
        time = np.arange(0, len(timeseries)) * self._config[1] / 1000 # total time steps * dt in seconds

        _ylabel = 'timeseries'
        _filename = 'Timeseries plot'
        title = ''
        textbox = 'default'
        filelabel = ''
        fileextension = 'png'
        subplotindex = None
        saveplotinfo = False
        for key, value in options.items():
            if key == '_ylabel': _ylabel = value
            elif key == '_filename': _filename = value
            elif key == 'title': title = value
            elif key == 'textbox': textbox = value
            elif key == 'filelabel': filelabel = value
            elif key == 'fileextension': fileextension = value
            elif key == 'subplotindex': subplotindex = value
            elif key == 'saveplotinfo': saveplotinfo = value
            else: print('Warning: the optional argument [{}] is not supported'.format(key))

        if graph == None:
            graph = GraphDataRelation()
            graph.create_plot(figsize=(14,7))
            graph.add_data(time, timeseries)

            graph.label_plot(dict(axislabel=['Time (s)', _ylabel], title=title))
            if textbox == 'default':
                textbox = 'Neuron index: {:d} | T: {:.7} ms | dt: {:.5}'.format(
                    int(neuron_index), float(self._config[2]), float(self._config[1]))
                graph.label_plot(dict(textbox=textbox))
            elif textbox == 'none': pass
            else: graph.label_plot(dict(textbox=textbox))
            if style != None: graph.stylize_plot(style)
            else: graph.stylize_plot(dict(ls='-', lw=1, ms=0))
            if scale != None: graph.set_scale(scale)
            if trim != (None, None): graph.set_scale(dict(xlim=trim))
            graph.make_plot()
            graph.save_plot(_filename, label=filelabel, ext=fileextension, path=self._output_path)
            if saveplotinfo: graph.save_plot_info(_filename, path=self._output_path)
        else:
            graph.stylize_plot(dict(ls='-', lw=1, ms=0))
            graph.set_scale(dict(grid=True))
            if trim != (None, None): graph.set_scale(dict(xlim=trim))
            else: graph.set_scale(dict(xlim=(np.amin(time), np.amax(time))))
            if style != None: graph.stylize_plot(style)
            if label != None: graph.label_plot(label)
            graph.add_data(time, timeseries)
            graph.make_plot(subplotindex)

    def _graph_phasediagram_subroutine(self, xdata: np.ndarray, ydata: np.ndarray, neuron_index: int, trim=(None, None), label=None, graph=None, **options):
        _axislabel = ['variable x', 'variable y']
        _filename = 'Phase diagram'
        cmap = 'copper'
        title = ''
        textbox = 'default'
        filelabel = ''
        fileextension = 'png'
        subplotindex = None
        saveplotinfo = False
        for key, value in options.items():
            if key == '_axislabel': _axislabel = value
            elif key == '_filename': _filename = value
            elif key == 'cmap': cmap = value
            elif key == 'title': title = value
            elif key == 'textbox': textbox = value
            elif key == 'filelabel': filelabel = value
            elif key == 'fileextension': fileextension = value
            elif key == 'subplotindex': subplotindex = value
            elif key == 'saveplotinfo': saveplotinfo = value
            else: print('Warning: the optional argument [{}] is not supported'.format(key))

        if trim[0] == None: start_time = 0
        else: start_time = trim[0]
        if trim[1] == None: end_time = self._config[2]
        else: end_time = trim[1]
        begin = int(start_time*1000/self._config[1])
        end = int(end_time*1000/self._config[1])

        if not graph:
            graph = GraphDataRelation()
            graph.create_plot(figsize=(9,7))
            ax = graph.ax
            outputplot = True
        elif subplotindex:
            ax = graph.axes[subplotindex]
        else: ax = graph.ax

        lc = _colorline(xdata[begin:end], ydata[begin:end], cmap=cmap, linewidth=1, ax=ax)
        cbar = plt.colorbar(lc)
        cbar.ax.yaxis.set_major_locator(tck.FixedLocator(cbar.ax.get_yticks().tolist()))
        cbar.ax.set_yticklabels(['{:f}'.format(Decimal('{}'.format(x)).normalize()) for x in np.arange(start_time, end_time+(end_time-start_time)/5, (end_time-start_time)/5)])
        cbar.set_label('Time (s)', rotation=90, labelpad=15)
        graph.ax.autoscale_view()

        if outputplot:
            graph.label_plot(dict(axislabel=_axislabel, title=title))
            if textbox == 'default':
                textbox = 'Neuron index: {:d} | dt: {:.5} ms'.format(int(neuron_index), float(self._config[1]))
                graph.label_plot(dict(textbox=textbox))
            elif textbox == 'none': pass
            else: graph.label_plot(dict(textbox=textbox))
            graph.save_plot(_filename, label=filelabel, ext=fileextension, path=self._output_path)
            if saveplotinfo: graph.save_plot_info(_filename, path=self._output_path)
        else:
            if not label: graph.label_plot(label)

    def __del__(self): pass


class NeuralNetwork(Tool):

    def __init__(self, synaptic_weight_matrix, input_folder=None, output_folder=None, delimiter=' ', matrix_format='nonzero'):
        """A tool for analyzing neural networks.

        Parameters
        ----------
        synaptic_weight_matrix : str or numpy.ndarray
            a matrix storing the synaptic weights/coupling strengths of neurons in a network:
            - str: system path to a local matrix file
            - numpy.ndarray: the matrix in form of a numpy.ndarray\n
        input_folder : str, optional
            full or relative path of the input folder, by default local folder\n
        output_folder : str, optional
            full or relative path of the output folder, by default local folder\n
        delimiter : str or chr, optional
            delimiter of the matrix file, by default `' '`(space)\n
        matrix_format : {`'nonzero'`, `'full'`}, optional
            - `'full'`: the matrix file stores the full matrix with elements in each row
            - `'nonzero'`: the matrix file stores only the nonzero elements with format: {j i w_ji}\n

        Raises
        ------
        FileNotFoundError
            invalid os path for input or output files\n
        TypeError
            invalid argument type for `synaptic_weight_matrix`\n
        ValueError
            invalid argument value for `matrix_format`\n
        """
        if type(synaptic_weight_matrix).__module__ == np.__name__:
            _, self._output_path = self._init_input_output_path(None, output_folder)
            self.synaptic_weight_matrix = synaptic_weight_matrix
            print('Synaptic weight matrix is imported as a numpy.ndarray')
        elif type(synaptic_weight_matrix) == str:
            input_path, self._output_path = self._init_input_output_path(input_folder, output_folder)
            self.synaptic_weight_matrix = self._init_synaptic_weight_matrix_from_file(synaptic_weight_matrix, input_path, delimiter, matrix_format)
            print('Synaptic weight matrix is imported from a local file: \"{}\"'.format(synaptic_weight_matrix))
        else:
            err = 'the input argument \"synaptic_weight_matrix\" must either be a numpy.ndarray or a string storing the path to a local matrix file'
            raise TypeError(err)
        np.fill_diagonal(self.synaptic_weight_matrix, 0) # assume no self-linkage
        self._isAdjMatModified = False
        self._mode = 'NeuralNetwork'

    def _init_synaptic_weight_matrix_from_file(self, synaptic_weight_matrix_file: str, input_path: str, delimiter: chr, matrix_format: str):
        try:
            with open(os.path.join(input_path, synaptic_weight_matrix_file), 'r', newline='') as fp:
                content = list(csv.reader(fp, delimiter=delimiter))
                if matrix_format == 'nonzero':
                    network_size = int(content[0][0]) # N
                    for i in range(1, len(content)):
                        content[i] = remove_all_occurrences('', content[i])
                        content[i][0] = int(content[i][0])-1 # j
                        content[i][1] = int(content[i][1])-1 # i
                        content[i][2] = float(content[i][2]) # w_ij
                    synaptic_weight_matrix = np.zeros((network_size, network_size))
                    for item in content[1:]:
                        synaptic_weight_matrix[item[1]][item[0]] = item[2]
                    del content
                    synaptic_weight_matrix = np.array(synaptic_weight_matrix).astype(float)
                elif matrix_format == 'full':
                    synaptic_weight_matrix = np.array(content).astype(float)
        except FileNotFoundError:
            err = 'Synaptic weight matrix file \"{}\" cannot be found.'.format(os.path.join(input_path, synaptic_weight_matrix_file))
            raise FileNotFoundError(err)
        return synaptic_weight_matrix

    def _init_synaptic_weight(self, **kwargs) -> tuple:
        synaptic_weight_matrix = self.synaptic_weight_matrix
        for key, value in kwargs:
            if key == 'synaptic_weight_matrix': synaptic_weight_matrix = value
        negative_weights = np.array(synaptic_weight_matrix.flatten()[synaptic_weight_matrix.flatten() < 0])
        positive_weights = np.array(synaptic_weight_matrix.flatten()[synaptic_weight_matrix.flatten() > 0])
        nonzero_weights = np.array(synaptic_weight_matrix.flatten()[synaptic_weight_matrix.flatten() != 0])
        return negative_weights, positive_weights, nonzero_weights

    def write_synaptic_weight_matrix_info_file(self, filename='synaptic_weight_matrix', filelabel='', ext='txt', matrix_format='nonzero', delimiter=' ', **kwargs):
        """Write synaptic weight matrix of the network into a local file."""
        custom = False
        for key, value in kwargs:
            if key == 'synaptic_weight_matrix':
                synaptic_weight_matrix = value
                custom = True
        if not custom: synaptic_weight_matrix = self.synaptic_weight_matrix
        if filelabel == '': filepath = '{}.{}'.format(filename, ext)
        else: filepath = '{}_{}.{}'.format(filename, filelabel, ext)
        with open(os.path.join(self._output_path, filepath), 'w') as fp:
            if matrix_format == 'nonzero':
                fp.write(str(len(synaptic_weight_matrix)))
                for j in range(len(synaptic_weight_matrix)):
                    for i in range(len(synaptic_weight_matrix)):
                        if synaptic_weight_matrix[i][j] != 0:
                            fp.write('\n{:d}{}{:d}{}{:.10f}'.format(j+1, delimiter, i+1, delimiter, synaptic_weight_matrix[i][j]))
            elif matrix_format == 'full':
                for row in synaptic_weight_matrix:
                    if row[0] == 0: fp.write('{:.0f}'.format(row[0])) # To reduce file size
                    else:           fp.write( '{:.8}'.format(row[0]))
                    for element in row[1:]:
                        if element == 0:    fp.write('\t{:.0f}'.format(element)) # To reduce file size
                        else:               fp.write( '\t{:.8}'.format(element))
                    fp.write('\n')

            else:
                err = 'the optional argument \"matrix_format\" can only be \'nonezero\' or \'full\''
                raise ValueError(err)
        print('Writing synaptic weight matrix: export to \"{}\" in directory \"{}\"'.format(filepath, self._output_path))

    def statistics_of_synaptic_weights(self, link_type: str, q=0.5, **kwargs) -> tuple:
        """Find the statistics of synaptic weights of the network.

        Parameters
        ----------
        link_type : str
            the type of links to be returned, options: 'inh' or 'exc' or 'all'
        q : float, optional
            q-th percentile of the return data, by default 0.5

        Returns
        -------
        tuple (scalar, scalar, scalar)
            mean, standanrd deviation, q-th percentile of the requested type of synaptic weights
        """
        custom = False
        for key, value in kwargs:
            if key == 'synaptic_weight_matrix':
                synaptic_weight_matrix = value
                custom = True
        if not custom: synaptic_weight_matrix = self.synaptic_weight_matrix
        syn_w_neg, syn_w_pos, syn_w = self._init_synaptic_weight(synaptic_weight_matrix=synaptic_weight_matrix)
        mean_neg, mean_pos, mean_all = np.mean(syn_w_neg), np.mean(syn_w_pos), np.mean(syn_w)
        std_neg, std_pos, std_all = np.std(syn_w_neg), np.std(syn_w_pos), np.std(syn_w)
        pct_neg, pct_pos, pct_all = np.percentile(syn_w_neg, q), np.percentile(syn_w_pos, q), np.percentile(syn_w, q)
        if link_type == 'inh': return mean_neg, std_neg, pct_neg
        if link_type == 'exc': return mean_pos, std_pos, pct_pos
        else: return mean_all, std_all, pct_all

    def connection_probability(self, **kwargs) -> float:
        """Return the connection probability of the network."""
        custom = False
        for key, value in kwargs:
            if key == 'synaptic_weight_matrix':
                synaptic_weight_matrix = value
                custom = True
        if custom:
            matrix_size = np.shape(synaptic_weight_matrix)[0]
            _, _, nonzero_weights = self._init_synaptic_weight(custom_synaptic_weight_matrix=synaptic_weight_matrix)
        else:
            matrix_size = np.shape(self.synaptic_weight_matrix)[0]
            _, _, nonzero_weights = self._init_synaptic_weight()
        num_of_links = len(nonzero_weights)
        connection_probability = num_of_links / (matrix_size * (matrix_size - 1))
        return connection_probability

    def number_of_links(self, linktype: str, **kwargs) -> np.ndarray:
        """Return the number of links in the network.

        Parameters
        ----------
        link_type : str
            the type of links to be returned, options: 'inh' or 'exc' or 'all'

        Returns
        -------
        numpy.ndarray
            an array of number of links of requested type, else an array with 0th, 1st, 2nd element
            storing the array of number of links of inhibitory, excitatory, all neurons correspondingly
        """
        custom = False
        for key, value in kwargs:
            if key == 'synaptic_weight_matrix':
                synaptic_weight_matrix = value
                custom = True
        if custom: negative_weights, positive_weights, nonzero_weights = self._init_synaptic_weight(synaptic_weight_matrix)
        else: negative_weights, positive_weights, nonzero_weights = self._init_synaptic_weight()
        number_of_links = []
        number_of_links.append(len(negative_weights))
        number_of_links.append(len(positive_weights))
        number_of_links.append(len(nonzero_weights))
        if linktype == 'inh': return np.array(number_of_links)[0]
        if linktype == 'exc': return np.array(number_of_links)[1]
        if linktype == 'all': return np.array(number_of_links)[2]
        else: return np.array(number_of_links)

    def neuron_type(self, **kwargs) -> np.ndarray:
        """Return the electrophysiological class of neurons in the network."""
        synaptic_weight_matrix = self.synaptic_weight_matrix
        for key, value in kwargs:
            if key == 'custom_synaptic_weight_matrix': synaptic_weight_matrix = value
        matrix_size = np.shape(synaptic_weight_matrix)[0]
        neuron_type = np.zeros(matrix_size)
        for row in synaptic_weight_matrix:
            for idx in range(matrix_size):
                if row[idx] < 0:
                    if neuron_type[idx] == 1: print('Warning: inconsistent classification in neuron type')
                    neuron_type[idx] = -1
                elif row[idx] > 0:
                    if neuron_type[idx] == -1: print('Warning: inconsistent classification in neuron type')
                    neuron_type[idx] = 1
        return neuron_type

    def excitatory_to_inhibitory_ratio(self) -> float:
        """Return the ratio of excitatory neurons to inhibitory neurons."""
        exc = np.count_nonzero(self.neuron_type() == +1)
        inh = np.count_nonzero(self.neuron_type() == -1)
        if inh == 0: return np.Inf
        else: return exc/inh

    def incoming_degree(self, link_type: str, **kwargs) -> np.ndarray:
        """Return the incoming degree of neurons in the network.

        Parameters
        ----------
        link_type : str
            the type of links to be returned, options: 'inh' or 'exc' or 'all'

        Returns
        -------
        numpy.ndarray
            an array of incoming degrees of requested type, else an array with 0th, 1st, 2nd element
            storing the array of incoming degrees of inhibitory, excitatory, all neurons correspondingly
        """
        custom = False
        for key, value in kwargs:
            if key == 'synaptic_weight_matrix':
                synaptic_weight_matrix = value
                custom = True
        if not custom: synaptic_weight_matrix = self.synaptic_weight_matrix
        incoming_degree, temp = [], []
        for row in synaptic_weight_matrix:
            temp.append(len(row[row < 0]))  #inh
            temp.append(len(row[row > 0]))  #exc
            temp.append(len(row[row != 0])) #all
            incoming_degree.append(np.array(temp))
            temp.clear()
        if link_type == 'inh': return np.array(incoming_degree).T[0]
        if link_type == 'exc': return np.array(incoming_degree).T[1]
        if link_type == 'all': return np.array(incoming_degree).T[2]
        else: return np.array(incoming_degree).T

    def outgoing_degree(self, **kwargs) -> np.ndarray:
        """Return the outgoing degree of neurons in the network."""
        custom = False
        for key, value in kwargs:
            if key == 'synaptic_weight_matrix':
                synaptic_weight_matrix = value
                custom = True
        if not custom: synaptic_weight_matrix = self.synaptic_weight_matrix
        outgoing_degree = []
        for row in synaptic_weight_matrix.T: outgoing_degree.append(len(row[row != 0]))
        return np.array(outgoing_degree)

    def average_synaptic_weights_of_incoming_links(self, linktype: str, **kwargs) -> np.ndarray:
        """Return the avergae synaptic weights of incoming links of neurons in the network.

        Parameters
        ----------
        link_type : str
            the type of links to be returned, options: 'inh' or 'exc' or 'all'

        Returns
        -------
        numpy.ndarray
            an array of average incoming synaptic weights of requested type, else an array with 0th, 1st, 2nd element
            storing the array of average incoming synaptic weights of inhibitory, excitatory, all neurons correspondingly
        """
        custom = False
        for key, value in kwargs:
            if key == 'synaptic_weight_matrix':
                synaptic_weight_matrix = value
                custom = True
        if not custom: synaptic_weight_matrix = self.synaptic_weight_matrix
        average_weights_in, temp = [], []
        for row in synaptic_weight_matrix:
            k_inh = len(row[row < 0])
            k_exc = len(row[row > 0])
            k_all = len(row[row != 0])
            if k_inh != 0: temp.append(np.sum(row[row < 0]) / k_inh)
            else: temp.append(None)
            if k_exc != 0: temp.append(np.sum(row[row > 0]) / k_exc)
            else: temp.append(None)
            if k_all != 0: temp.append(np.sum(row[row != 0]) / k_all)
            else: temp.append(None)
            average_weights_in.append(np.array(temp))
            temp.clear()
        if linktype == 'inh': return np.array(average_weights_in).T[0]
        if linktype == 'exc': return np.array(average_weights_in).T[1]
        if linktype == 'all': return np.array(average_weights_in).T[2]
        else: return np.array(average_weights_in).T

    def average_synaptic_weights_of_outgoing_links(self, linktype: str, **kwargs) -> np.ndarray:
        """Return the avergae synaptic weights of outgoing links of neurons in the network.

        Parameters
        ----------
        link_type : str
            the type of links to be returned, options: 'inh' or 'exc' or 'all'

        Returns
        -------
        numpy.ndarray
            an array of average outgoing synaptic weights of requested type, else an array with 0th, 1st, 2nd element
            storing the array of average outgoing synaptic weights of inhibitory, excitatory, all neurons correspondingly
        """
        custom = False
        for key, value in kwargs:
            if key == 'synaptic_weight_matrix':
                synaptic_weight_matrix = value
                custom = True
        if not custom: synaptic_weight_matrix = self.synaptic_weight_matrix
        average_weights_out, temp = [], []
        for col in synaptic_weight_matrix.T:
            k_inh = len(col[col < 0])
            k_exc = len(col[col > 0])
            k_all = len(col[col != 0])
            if k_inh != 0: temp.append(np.sum(col[col < 0]) / k_inh)
            else: temp.append(None)
            if k_exc != 0: temp.append(np.sum(col[col > 0]) / k_exc)
            else: temp.append(None)
            if k_all != 0: temp.append(np.sum(col[col != 0]) / k_all)
            else: temp.append(None)
            average_weights_out.append(np.array(temp))
            temp.clear()
        if linktype == 'inh': return np.array(average_weights_out).T[0]
        if linktype == 'exc': return np.array(average_weights_out).T[1]
        if linktype == 'all': return np.array(average_weights_out).T[2]
        else: return np.array(average_weights_out).T

    def plot_synaptic_weight_distribution_NEG(self, binsize: float, style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of negative synaptic weights."""
        print('Drawing graph: distribution of negative synaptic weights')
        negative_weight = self._init_synaptic_weight()[0]

        _xlabel = r'Negative synaptic weight w$_{ij}^-$'
        _filename = 'Distribution of negative synaptic weights'

        self._graph_distribution_subroutine(negative_weight, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_synaptic_weight_distribution_POS(self, binsize: float, style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of positive synaptic weights."""
        print('Drawing graph: distribution of positive synaptic weights')
        positive_weight = self._init_synaptic_weight()[1]

        _xlabel = r'Positive synaptic weight w$_{ij}^+$'
        _filename = 'Distribution of positive synaptic weights'

        self._graph_distribution_subroutine(positive_weight, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_synaptic_weight_distribution(self, binsize: float, style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of synaptic weights."""
        print('Drawing graph: distribution of synaptic weights')
        nonzero_weight = self._init_synaptic_weight()[2]

        _xlabel = r'Synaptic weight w$_{ij}$'
        _filename = 'Distribution of synaptic weights'

        self._graph_distribution_subroutine(nonzero_weight, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_incoming_degree_distribution_INH(self, binsize: float, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of inhibitory incoming degrees."""
        print('Drawing graph: distribution of inhibitory incoming degree')
        inc_deg_inh = self.incoming_degree('inh')
        if len(specific_neurons) != 0: inc_deg_inh = inc_deg_inh[specific_neurons]

        _xlabel = r'Inhibitory incoming degree k$_{in}^-$'
        _filename = 'Distribution of inhibitory incoming degrees'

        self._graph_distribution_subroutine(inc_deg_inh, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_incoming_degree_distribution_EXC(self, binsize: float, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of excitatory incoming degrees."""
        print('Drawing graph: distribution of excitatory incoming degree')
        inc_deg_exc = self.incoming_degree('exc')
        if len(specific_neurons) != 0: inc_deg_exc = inc_deg_exc[specific_neurons]

        _xlabel = r'Excitatory incoming degree k$_{in}^+$'
        _filename = 'Distribution of excitatory incoming degrees'

        self._graph_distribution_subroutine(inc_deg_exc, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_incoming_degree_distribution(self, binsize: float, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of incoming degrees."""
        print('Drawing graph: distribution of incoming degree')
        inc_deg_all = self.incoming_degree('all')
        if len(specific_neurons) != 0: inc_deg_all = inc_deg_all[specific_neurons]

        _xlabel = r'Incoming degree k$_{in}$'
        _filename = 'Distribution of incoming degrees'

        self._graph_distribution_subroutine(inc_deg_all, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_outgoing_degree_distribution(self, binsize: float, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of outgoing degrees."""
        print('Drawing graph: distribution of outgoing degrees')
        out_deg = self.outgoing_degree()
        if len(specific_neurons) != 0: out_deg = out_deg[specific_neurons]

        _xlabel = r'Outgoing degree k$_{out}$'
        _filename = 'Distribution of outgoing degrees'

        self._graph_distribution_subroutine(out_deg, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_incoming_average_weight_distribution_INH(self, binsize: float, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of average synaptic weights of inhibitory incoming links."""
        print('Drawing graph: distribution of average synaptic weights of inhibitory incoming links')
        avg_w_inc_inh = self.average_synaptic_weights_of_incoming_links('inh')
        if len(specific_neurons) != 0: avg_w_inc_inh = avg_w_inc_inh[specific_neurons]

        _xlabel = r'Average synaptic weight of inhibitory incoming links s$_{in}^-$'
        _filename = 'Distribution of average synaptic weights of inhibitory incoming links'

        self._graph_distribution_subroutine(avg_w_inc_inh, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_incoming_average_weight_distribution_EXC(self, binsize: float, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of average synaptic weights of excitatory incoming links."""
        print('Drawing graph: distribution of average synaptic weights of excitatory incoming links')
        avg_w_inc_exc = self.average_synaptic_weights_of_incoming_links('exc')
        if len(specific_neurons) != 0: avg_w_inc_exc = avg_w_inc_exc[specific_neurons]

        _xlabel = r'Average synaptic weight of excitatory incoming links s$_{in}^-$'
        _filename = 'Distribution of average synaptic weights of excitatory incoming links'

        self._graph_distribution_subroutine(avg_w_inc_exc, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_incoming_average_weight_distribution(self, binsize: float, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of average synaptic weights of incoming links."""
        print('Drawing graph: distribution of average synaptic weights of incoming links')
        avg_w_inc = self.average_synaptic_weights_of_incoming_links('all')
        if len(specific_neurons) != 0: avg_w_inc = avg_w_inc[specific_neurons]

        _xlabel = r'Average synaptic weight of incoming links s$_{in}$'
        _filename = 'Distribution of average synaptic weights of incoming link'

        self._graph_distribution_subroutine(avg_w_inc, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_outgoing_average_weight_distribution(self, binsize: float, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of average synaptic weights of outgoing links."""
        print('Drawing graph: distribution of average synaptic weights of outgoing links')
        avg_w_out = self.average_synaptic_weights_of_outgoing_links('all')
        if len(specific_neurons) != 0: avg_w_out = avg_w_out[specific_neurons]

        _xlabel = r'Average synaptic weight of outgoing links s$_{out}$'
        _filename = 'Distribution of average synaptic weights of outgoing link'

        self._graph_distribution_subroutine(avg_w_out, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_links_of_network(self, link_type: str, threshold=0.5, figsize=(9, 9), dpi=150,
                              file_name='strongest_network_links', file_type=['svg','png']):
        """Plot the strongest links in the network.

        Parameters
        ----------
        link_type : str
            _description_
        threshold : float, optional
            links with strength exceed the threshold will be shown, by default 0.5
        figsize : tuple, optional
            figure size, by default (9, 9)
        dpi : int, optional
            dots per inch, by default 150
        file_name : str, optional
            name or path of the output plot, by default 'strongest_network_links'
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']
        """

        def set_background_with_node_type(node_amt, row_size):
            node_type = self.classify_node_type(self.synaptic_weight_matrix)
            node_count = 0
            for i in range(row_size):
                for j in range(row_size):
                    if node_type[node_count] == -1: ax.scatter([j+1], [i+1], s=5, c='c')
                    elif node_type[node_count] == 1: ax.scatter([j+1], [i+1], s=5, c='m')
                    else: ax.scatter([j+1], [i+1], s=5, c='darkgrey')
                    node_count += 1
                    if node_count == node_amt: break

        def join_two_points(i: int, j: int, row_size: int, color=''):
            ax.plot([i%row_size+1, j%row_size+1], [int(i/row_size)+1, int(j/row_size)+1], color+'-', lw=0.7)

        print('Drawing graph: strongest links in network')
        node_amt = np.shape(self.synaptic_weight_matrix)[0]
        row_size = round(math.sqrt(node_amt))

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        set_background_with_node_type(node_amt, row_size)
        mean, sd, pct = self.statistics_of_synaptic_weights(100-threshold)
        pos_threshold = pct#mean + 1.5 * sd
        mean, sd, pct = self.statistics_of_synaptic_weights(threshold)
        neg_threshold = pct#mean + 1.5 * sd
        for i in range(node_amt):
            for j in range(node_amt):
                if self.synaptic_weight_matrix[i][j] < neg_threshold:
                    if link_type in ['inh', 'all']: join_two_points(i, j, row_size, 'b')
                elif self.synaptic_weight_matrix[i][j] > pos_threshold:
                    if link_type in ['exc', 'all']: join_two_points(i, j, row_size, 'r')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_xlim(0, row_size+1)
        ax.set_ylim(0, row_size+1)
        for ext in file_type: fig.savefig(os.path.join(self.output_path, file_name+'.'+ext))

    def revert_modifications(self):
        self.synaptic_weight_matrix = self._unmodified_adjmat.copy()
        self.isAdjMatModified = False

    def suppression_to_inhibition(self, k: float) -> np.ndarray:
        """Suppress inhibition of the network.
        
        Replace the negative synaptic weights wij- by ( wij- + k * sigma_inh ).
        If wij- exceeds zero, replace it by zero to prevent convertion to excitatory links.
        Modified the adjacency matrix in place.
        Revert to the unmodified network by calling `Network.revert_modifications()`.
        The effect of suppression is not accumulated.

        Parameters
        ----------
        k : float
            suppression level

        Returns
        -------
        np.ndarray
            adjacency matrix of the suppressed network
        """
        if not self.isAdjMatModified: self._unmodified_adjmat = self.synaptic_weight_matrix.copy()
        self.isAdjMatModified = True
        self.synaptic_weight_matrix = self._unmodified_adjmat.copy()
        sigma_inh = self.statistics_of_synaptic_weights('inh')[1]
        for i in range(self.number_of_neurons):
            for j in range(self.number_of_neurons):
                if self.synaptic_weight_matrix[i][j] < 0:
                    self.synaptic_weight_matrix[i][j] += k * sigma_inh
                    if self.synaptic_weight_matrix[i][j] > 0:
                        self.synaptic_weight_matrix[i][j] = 0
        return self.synaptic_weight_matrix


class NeuralDynamics(Tool):

    def __init__(self, input_folder=None, output_folder=None, **options):
        """A tool for analyzing neural dynamics.

        Parameters
        ----------
        input_folder : str, optional
            full or relative path of the input folder, by default local folder\n
        output_folder : str, optional
            full or relative path of the output folder, by default local folder\n

        Raises
        ------
        FileNotFoundError
            invalid os path for input or output files\n
        IndexError
            neuron index out of bound\n
        TypeError
            invalid argument type for `configuration`\n
            invalid argument type for `neural_dynamics_original`\n
        ValueError
            invalid value for object of `neural_dynamics_original`\n
        """

        configuration = 'cont.dat'
        spiking_data_file = 'spks.txt'
        membrane_potential_timeseries_file = 'memp.bin'
        recovery_variable_timeseries_file = 'recv.bin'
        synaptic_current_timeseries_file = 'curr.bin'
        delimiter = '\t'
        for key, value in options.items():
            if key == 'configuration': configuration = value
            elif key == 'spiking_data_file': spiking_data_file = value
            elif key == 'membrane_potential_timeseries_file': membrane_potential_timeseries_file = value
            elif key == 'recovery_variable_timeseries_file': recovery_variable_timeseries_file = value
            elif key == 'synaptic_current_timeseries_file': synaptic_current_timeseries_file = value
            elif key == 'delimiter': delimiter = value

        input_path, self._output_path = self._init_input_output_path(input_folder, output_folder)
        if type(configuration) == list:
            self._spike_count, self._spike_times, self._config = self._init_spiking_data_from_file(spiking_data_file, configuration, delimiter, input_path)
        elif type(configuration) == str:
            self._spike_count, self._spike_times, self._config = self._init_spiking_data_from_file(spiking_data_file, configuration, delimiter, input_path)
        else:
            err = 'the input argument \"config\" must either be a list of [N, dt, T] or a string storing the path to a local file'
            raise TypeError(err)
        self._membrane_potential_timeseries_file = os.path.join(input_path, membrane_potential_timeseries_file)
        self._recovery_variable_timeseries_file = os.path.join(input_path, recovery_variable_timeseries_file)
        self._synaptic_current_timeseries_file = os.path.join(input_path, synaptic_current_timeseries_file)
        if input_folder == '': input_folder = 'current directory'
        print('Spiking data imported from a local file: \"{}\" in \"{}\"'.format(spiking_data_file, input_folder))
        self._mode = 'NeuralDynamics'

    def _init_spiking_data_from_file(self, spiking_data_file: str, configuration, delimiter: str, input_path: str) -> tuple:
        if type(configuration) == list:
            config = configuration
            matrix_size = config[0]
        elif type(configuration) == str:
            try:
                with open(os.path.join(input_path, configuration), 'r') as fp:
                    reader = csv.reader(fp, delimiter='|')
                    config = np.array(list(reader), dtype=object)
                config = config[1]
                config[0] = int(config[0])   # config[0] = N
                config[1] = float(config[1]) # config[1] = dt
                config[2] = float(config[2]) # config[2] = T
                matrix_size = int(config[0])
            except FileNotFoundError:
                err = 'configuration file \"{}\" cannot be found'.format(os.path.join(input_path, configuration))
                raise FileNotFoundError(err)
        try:
            with open(os.path.join(input_path, spiking_data_file), 'r') as fp:
                spike_times = np.empty(matrix_size, dtype=object)
                spike_count = np.zeros(matrix_size)
                reader = csv.reader(fp, delimiter=delimiter)
                counter = 0
                for row in reader:
                    try: spike_times[counter] = float(config[1])*np.delete(np.array(list(row)).astype('float'), [0], 0)
                    except ValueError: pass
                    spike_count[counter] = int(row[0])
                    counter += 1
        except FileNotFoundError:
            err = 'spiking data file \"{}\" cannot be found'.format(os.path.join(input_path, spiking_data_file))
            raise FileNotFoundError(err)
        return spike_count, spike_times, config

    def _init_firing_rate(self, _spike_count=None, _config=None, _custom=False) -> np.ndarray:
        if _custom == False: _spike_count, _config = self._spike_count, self._config
        return _spike_count / _config[2] * 1000 # config[2] = T

    def _init_firing_rate_change(self, neural_dynamics_original: object, specific_neurons=[]) -> np.ndarray:
        if type(neural_dynamics_original) != NeuralDynamics:
            err = 'the input argument \"neural_dynamics_original\" must be an object of class::NeuralDynamics.'
            raise TypeError(err)
        spike_count_orig = neural_dynamics_original._spike_count
        config_orig = neural_dynamics_original._config
        if len(self._spike_count) != len(spike_count_orig):
            err = 'the numbers of neurons from two numerical simulations do not match.'
            raise ValueError(err)
        firing_rate_orig = self._init_firing_rate(spike_count_orig, config_orig, _custom=True)
        firing_rate = self._init_firing_rate()
        if specific_neurons == []: return firing_rate - firing_rate_orig
        else:
            specific_neurons = [x-1 for x in specific_neurons]
            return firing_rate[specific_neurons] - firing_rate_orig[specific_neurons]

    def _init_interspike_interval(self, specific_neurons=[]) -> np.ndarray:
        if specific_neurons == []:
            interspike_interval = np.empty(self._config[0], dtype=object)
            for neuron in range(self._config[0]):
                try: interspike_interval[neuron] = np.array(np.diff(self._spike_times[neuron]), dtype=float)
                except ValueError: interspike_interval[neuron] = np.diff(np.array([0]))
        else:
            count = 0
            specific_neurons = [x-1 for x in specific_neurons]
            interspike_interval = np.empty(len(specific_neurons), dtype=object)
            for neuron in range(self._config[0]):
                if neuron in specific_neurons:
                    try: interspike_interval[count] = np.array(np.diff(self._spike_times[neuron]), dtype=float)
                    except ValueError: interspike_interval[count] = np.diff(np.array([0]))
                    count += 1
        interspike_interval = np.concatenate([item for item in interspike_interval.flatten()], 0) / 1000
        return interspike_interval

    def _init_membrane_potential(self, neuron_index: int, _membrane_potential_timeseries=None, _config=None, _custom=False) -> np.ndarray:
        if _custom == False: _membrane_potential_timeseries, _config = self._membrane_potential_timeseries_file, self._config
        neuron_index -= 1
        if not 0 <= neuron_index <= self._config[0]-1:
            err = 'neuron index {:d} is out of bound, index starts from 1, ends in {:d}'.format(neuron_index+1, self._config[0])
            raise IndexError(err)
        time_series = Vector_fl()
        if time_series.read_from_binary(_membrane_potential_timeseries, neuron_index, _config[0]) == 0: pass
        else:
            err = 'time series of membrane potential \"{}\" cannot be found.'.format(
                  self._membrane_potential_timeseries_file)
            raise FileNotFoundError(err)
        return np.array(time_series)

    def _init_recovery_variable(self, neuron_index: int, _recovery_variable_timeseries=None, _config=None, _custom=False) -> np.ndarray:
        if _custom == False: _recovery_variable_timeseries, _config = self._recovery_variable_timeseries_file, self._config
        neuron_index -= 1
        if not 0 <= neuron_index <= self._config[0]-1:
            err = 'neuron index {:d} is out of bound, index starts from 1, ends in {:d}'.format(neuron_index+1, self._config[0])
            raise IndexError(err)
        time_series = Vector_fl()
        if time_series.read_from_binary(_recovery_variable_timeseries, neuron_index, _config[0]) == 0: pass
        else:
            err = 'time series of recovery variable \"{}\" cannot be found.'.format(
                  self._recovery_variable_timeseries_file)
            raise FileNotFoundError(err)
        return np.array(time_series)

    def _init_synaptic_current(self, neuron_index: int, _synaptic_current_timeseries=None, _config=None, _custom=False) -> np.ndarray:
        if _custom == False: _synaptic_current_timeseries, _config = self._synaptic_current_timeseries_file, self._config
        neuron_index -= 1
        if not 0 <= neuron_index <= self._config[0]-1:
            err = 'neuron index {:d} is out of bound, index starts from 1, ends in {:d}'.format(neuron_index+1, self._config[0])
            raise IndexError(err)
        time_series = Vector_fl()
        if time_series.read_from_binary(_synaptic_current_timeseries, neuron_index, _config[0]) == 0: pass
        else:
            err = 'time series of current \"{}\" cannot be found.'.format(
                  self._synaptic_current_timeseries_file)
            raise FileNotFoundError(err)
        return np.array(time_series)

    def spike_counts(self) -> np.ndarray:
        """Return the spike counts of each neuron in the network.

        Returns
        -------
        numpy.ndarray
            the i-th array element coressponds to the spike counts of neuron i
        """
        return self._spike_count

    def firing_rates(self) -> np.ndarray:
        """Return the firing rate of each neuron in the network.

        Returns
        -------
        numpy.ndarray
            the i-th array element coressponds to the firing rate of neuron i
        """
        return self._init_firing_rate()

    def firing_rate_changes(self, neural_dynamics_original: object) -> np.ndarray:
        """Return the changes in firing rate of each neuron in two networks.

        Parameters
        ----------
        neural_dynamics_original : object
            object of class NeuralDynamics

        Returns
        -------
        numpy.ndarray
            the i-th array element coressponds to the change in firing rate of neuron i in two networks
        """
        return self._init_firing_rate_change(neural_dynamics_original)

    def interspike_intervals(self) -> np.ndarray:
        """Return the inter-spike intervals of action potential spikes in the network.

        Returns
        -------
        numpy.ndarray
            the inter-spike intervals of action potential spikes in the network
        """
        return self._init_interspike_interval()

    def plot_firing_rate_distribution(self, binsize: float, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of firing rates.

        Parameters
        ----------
        binsize : float
            bin size
        specific_neurons : list, optional
            neurons to be considered, index starts from 1, by default `[]`
        style : dict, optional
            style of the plot, e.g., color, line weight, etc., by default `None`
        scale : dict, optional
            scale of the plot, e.g., x range, log scale, etc., by default `None`
        graph : object of Grapher, optional
            supply a Grapher object here for overlaying plots, by default `None`
        
        **options
        
        textbox : str, optional
            plot information to be displayed at top-left corner, show basic info by default
        fitgaussian : bool, optional
            fit the data with a Gaussian curve, by default `False`
        filelabel : str, optional
            label attached at the end of the file name, by default `''`
        fileextension : str, optional
            file extension, by default `'png'`
        subplotindex : int, optional
            index of sub-plot to be drawn into, by default `0`
        """
        print('Drawing graph: distribution of firing rates')
        firing_rate = self._init_firing_rate()
        if len(specific_neurons) != 0: firing_rate = firing_rate[[x-1 for x in specific_neurons]]

        _xlabel = 'Firing rate (Hz)'
        _filename = 'Distribution of firing rates'

        self._graph_distribution_subroutine(firing_rate, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_firing_rate_change_distribution(self, neural_dynamics_original: object, binsize: float, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of changes in firing rate.

        Parameters
        ----------
        neural_dynamics_original : object of NeuralDynamics
            object of original neural dynamics
        binsize : float
            bin size
        specific_neurons : list, optional
            neurons to be considered, index starts from 1, by default `[]`
        style : dict, optional
            style of the plot, e.g., color, line weight, etc., by default `None`
        scale : dict, optional
            scale of the plot, e.g., x range, log scale, etc., by default `None`
        graph : object of Grapher, optional
            supply a Grapher object here for overlaying plots, by default `None`
        
        **options
        
        textbox : str, optional
            plot information to be displayed at top-left corner, show basic info by default
        fitgaussian : bool, optional
            fit the data with a Gaussian curve, by default `False`
        filelabel : str, optional
            label attached at the end of the file name, by default `''`
        fileextension : str, optional
            file extension, by default `'png'`
        subplotindex : int, optional
            index of sub-plot to be drawn into, by default `0`
        """
        print('Drawing graph: distribution of changes in firing rate')
        firing_rate_chg = self._init_firing_rate_change(neural_dynamics_original, specific_neurons)

        _xlabel = 'Change in firing rate (Hz)'
        _filename = 'Distribution of changes in firing rate'

        self._graph_distribution_subroutine(firing_rate_chg, binsize, style, label, scale, graph, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_interspike_interval_distribution(self, binsize: float, xlogscale=True, specific_neurons=[], style=None, label=None, scale=None, graph=None, **options):
        """Plot the distribution of inter-spike intervals (ISI).

        Parameters
        ----------
        binsize : float
            bin size
        xlogscale : bool, optional
            log scale in x-axis, by default `True`
        specific_neurons : list, optional
            neurons to be considered, index starts from 1, by default `[]`
        style : dict, optional
            style of the plot, e.g., color, line weight, etc., by default `None`
        label : dict, optional
            label of the plot, e.g., axis label, legend, etc., by default `None`
        scale : dict, optional
            scale of the plot, e.g., x range, log scale, etc., by default `None`
        graph : object of Grapher, optional
            supply a Grapher object here for overlaying plots, by default `None`
        
        **options
        
        textbox : str, optional
            plot information to be displayed at top-left corner, show basic info by default
        fitgaussian : bool, optional
            fit the data with a Gaussian curve, by default `False`
        filelabel : str, optional
            label attached at the end of the file name, by default `''`
        fileextension : str, optional
            file extension, by default `'png'`
        subplotindex : int, optional
            index of sub-plot to be drawn into, by default `0`
        """
        print('Drawing graph: distribution of inter-spike intervals (ISI)')
        interspike_interval = self._init_interspike_interval(specific_neurons)

        _xlabel = 'Inter-spike intervals ISI (s)'
        _filename = 'Distribution of inter-spike intervals'

        self._graph_distribution_subroutine(interspike_interval, binsize, style, label, scale, graph, xlogscale=xlogscale, _xlabel=_xlabel, _filename=_filename, **options)

    def plot_spike_raster(self, trim=(None, None), graph=None, **options):
        """Plot a spike raster plot.

        Parameters
        ----------
        trim : tuple, optional
            the range of time to be plotted, by default `(None, None)`
        graph : object of Grapher, optional
            supply a Grapher object here for overlaying plots, by default `None`
        
        **options
        
        color : str, optional
            marker color, by default `'b'`
        filelabel : str, optional
            label attached at the end of the file name, by default `''`
        fileextension : str, optional
            file extension, by default `'png'`
        subplotindex : int, optional
            index of sub-plot to be drawn into, by default `0`
        """
        print('Drawing graph: spike raster plot')
        if graph == None: fig, ax = plt.subplots(figsize=(14,7))

        color = 'k'
        # plotlabel = ''
        filelabel = ''
        fileextension = 'png'
        subplotindex = None
        for key, value in options.items():
            if key == 'color' or key == 'c': color = value
            # elif key == 'plotlabel': plotlabel = value
            elif key == 'filelabel': filelabel = value
            elif key == 'fileextension': fileextension = value
            elif key == 'subplotindex': subplotindex = value
            else: print('Warning: the optional argument [{}] is not supported'.format(key))

        if trim[0] == None: trim=(0, trim[1])
        if trim[1] == None: trim=(trim[0], self._config[2]/1000)
        neuron_index = 0
        if graph == None:
            for neuronal_spike_times in (self._spike_times / 1000):
                neuron_index += 1
                if trim[0] > 0: neuronal_spike_times = neuronal_spike_times[neuronal_spike_times > float(trim[0])]
                if trim[1] < self._config[2]/1000: neuronal_spike_times = neuronal_spike_times[neuronal_spike_times < float(trim[1])]
                lc = mcol.EventCollection(neuronal_spike_times, lineoffset=neuron_index, linestyle='-',
                                          linelength=20, linewidth=1.5, color=color)
                ax.add_collection(lc)
            ax.set(xlabel='Time (s)', ylabel='Neuron index')
            ax.set_xlim(trim[0], trim[1])               # config[2] = T
            start_node, end_node = 0, self._config[0]   # config[0] = N
            ax.set_ylim(start_node-2, end_node+1)
            # ax.grid(True, which='major')
            ax.grid(False)
            plt.tight_layout()

            if filelabel == '': filename = '{}.{}'.format('Spike raster plot', fileextension)
            else: filename = '{}_{}.{}'.format('Spike raster plot', filelabel, fileextension)
            fig.savefig(os.path.join(self._output_path, filename))
        else:
            if subplotindex == None: ax = graph.ax
            else: ax = graph.axes[subplotindex]
            for neuronal_spike_times in (self._spike_times / 1000):
                neuron_index += 1
                if trim[0] > 0: neuronal_spike_times = neuronal_spike_times[neuronal_spike_times > float(trim[0])]
                if trim[1] < self._config[2]/1000: neuronal_spike_times = neuronal_spike_times[neuronal_spike_times < float(trim[1])]
                lc = mcol.EventCollection(neuronal_spike_times, lineoffset=neuron_index, linestyle='-',
                                          linelength=20, linewidth=1.5, color=color)
                ax.add_collection(lc)
            ax.set(ylim=(0, self._config[0]), xlim=trim)
            ax.minorticks_off()
            # ax.grid(False, which='minor')
            # ax.grid(True, which='major')
            ax.grid(False)

    def plot_membrane_potential_time_series(self, neuron_index: int, trim=(None, None), style=None, label=None, scale=None, graph=None, **options):
        print('Drawing graph: membrane potential time series of neuron-{}'.format(neuron_index))
        voltage = self._init_membrane_potential(neuron_index)

        _ylabel = 'Membrane potential (mV)'
        _filename = 'Membrane potential of neuron-{:d}'.format(neuron_index)

        self._graph_timeseries_subroutine(voltage, neuron_index, trim, style, label, scale, graph, _ylabel=_ylabel, _filename=_filename, **options)

    def plot_recovery_variable_time_series(self, neuron_index: int, trim=(None, None), style=None, label=None, scale=None, graph=None, **options):
        print('Drawing graph: recovery variable time series of neuron-{}'.format(neuron_index))
        recovery = self._init_recovery_variable(neuron_index)

        _ylabel = 'Recovery variable'
        _filename = 'Recovery variable of neuron-{:d}'.format(neuron_index)

        self._graph_timeseries_subroutine(recovery, neuron_index, trim, style, label, scale, graph, _ylabel=_ylabel, _filename=_filename, **options)

    def plot_synaptic_current_time_series(self, neuron_index: int, trim=(None, None), style=None, label=None, scale=None, graph=None, **options):
        print('Drawing graph: current time series of neuron-{}'.format(neuron_index))
        current = self._init_synaptic_current(neuron_index)

        _ylabel = 'Synaptic current'
        _filename = 'Synaptic current of neuron-{:d}'.format(neuron_index)

        self._graph_timeseries_subroutine(current, neuron_index, trim, style, label, scale, graph, _ylabel=_ylabel, _filename=_filename, **options)

    def plot_phase_diagram_voltage_recovery(self, neuron_index: int, trim=(None, None), label=None, graph=None, **options):
        print('Drawing graph: phase diagram of membrane potential and recovery variable')
        voltage = self._init_membrane_potential(neuron_index)
        recovery = self._init_recovery_variable(neuron_index)

        _axislabel = ['Membrane potential (mV)', 'Recovery variable']
        _filename = 'Phase diagram voltage-recovery of neuron-{:d}'.format(neuron_index)

        self._graph_phasediagram_subroutine(voltage, recovery, neuron_index, trim, label, graph, _axislabel=_axislabel, _filename=_filename, **options)

    def plot_phase_diagram_voltage_current(self, neuron_index: int, trim=(None, None), label=None, graph=None, **options):
        print('Drawing graph: phase diagram of membrane potential and synaptic current')
        voltage = self._init_membrane_potential(neuron_index)
        current = self._init_synaptic_current(neuron_index)

        _axislabel = ['Membrane potential (mV)', 'Synaptic current']
        _filename = 'Phase diagram voltage-current of neuron-{:d}'.format(neuron_index)

        self._graph_phasediagram_subroutine(voltage, current, neuron_index, trim, label, graph, _axislabel=_axislabel, _filename=_filename, **options)


# Other tools
def quick_preview_density_distribution(data, binsize, logdata=False, saveplot=False):
    graph = GraphDensityDistribution()
    graph.create_plot()
    graph.add_data(data, binsize, logdata=logdata)
    if logdata: graph.set_scale(dict(xlogscale=True))
    graph.make_plot()
    if saveplot: graph.save_plot('temp_preview')
    graph.show_plot()

def _plot_distribution_deprecated(data, bin_size=0.15, color='b', marker_style='^', line_style='-',
                      plot_type='line', marker_size=8, marker_fill=True, line_width=2,
                      figsize=(9, 6), dpi=150, plot_label='', xlabel='', textbox='',
                      xlim=(None, None), ylim=(None, None), show_norm=False,
                      x_logscale=False, y_logscale=False, remove_zero_density=False,
                      file_name='plot_dist', file_label='', file_type=['svg','png'],
                      output_path='', save_fig=True, show_fig=False, mpl_ax=None,
                      return_plot_data=False, return_area_under_graph=False):

    # Return ax
    def plot_subroutine(x, y, ax, plot_type: str, plot_label: str, plot_norm=False):
        if plot_norm: c='r'; marker='None'; ls='--'
        else: c=color; marker=marker_style; ls=line_style
        if marker_fill == True:
            if plot_type == 'line': ax.plot(x, y, c=c, marker=marker, ms=marker_size, ls=ls,
                                            lw=line_width, label=plot_label)
            else: ax.scatter(x, y, c=c, marker=marker, s=marker_size, label=plot_label)
        else:
            if plot_type == 'line': ax.plot(x, y, c=c, marker=marker, ms=marker_size, ls=ls,
                                            lw=line_width, mfc='none', label=plot_label)
            else: ax.scatter(x, y, c=c, marker=marker, s=marker_size, facecolors='none', label=plot_label)
        return ax
    
    # Return ax, plot_data, area_under_graph
    def make_plot(ax):
        min_datum = np.amin(data); max_datum = np.amax(data)
        if show_norm == True:
            mu = np.mean(data); sigma = np.std(data)
            x_value_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 150)
            ax = plot_subroutine(x_value_norm, stats.norm.pdf(x_value_norm, mu, sigma), ax,
                                 'line', 'Normal distribution', plot_norm=True)
        if x_logscale == False:
            number_of_bins = math.ceil((max_datum - min_datum) / bin_size)
            hist_density, bin_edges = np.histogram(data, bins=np.linspace(min_datum, min_datum+number_of_bins*bin_size,
                                                                          number_of_bins), density=True)
        elif x_logscale == True:
            min_datum = np.amin(np.array(data)[np.array(data) > 0])
            number_of_bins = math.ceil((math.log10(max_datum) - math.log10(min_datum)) / bin_size)
            hist_density, bin_edges = np.histogram(data, bins=np.logspace(math.log10(min_datum),
                                                    math.log10(min_datum)+number_of_bins*bin_size, number_of_bins))
        hist_density = np.array(hist_density, dtype=float)
        hist_density /= np.dot(hist_density, np.diff(bin_edges)) # normalization
        x_value = (bin_edges[1:] + bin_edges[:-1]) / 2
        area_under_graph = np.dot(hist_density, np.diff(bin_edges))
        if remove_zero_density == True:
            non_zero_points = np.argwhere(hist_density == 0)
            hist_density = np.delete(hist_density, non_zero_points, 0)
            x_value = np.delete(x_value, non_zero_points, 0)
        return plot_subroutine(x_value, hist_density, ax,
                               plot_type, plot_label), np.array(list(zip(x_value, hist_density))), area_under_graph

    # Return ax
    def format_plot(ax):
        if x_logscale == True: ax.set_xscale('log')
        if y_logscale == True: ax.set_yscale('log')
        ax.set(xlabel=xlabel, ylabel='Probability density')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels))#, title='Distribution')
        # textbox
        if textbox != '':
            props = dict(boxstyle='round', pad=0.1, facecolor='white', edgecolor='none', alpha=0.75)
            ax.text(0.00001, 1.05, textbox, fontsize=10, verticalalignment='top', transform=ax.transAxes, bbox=props)
        return ax

    if mpl_ax == None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax, plot_data, area_under_graph = make_plot(ax)
        ax = format_plot(ax)
        for ext in file_type:
            if file_label == '': file_path = file_name+'.'+ext
            else: file_path = file_name+'_'+file_label+'.'+ext
            if save_fig: fig.savefig(os.path.join(output_path, file_path))
        if show_fig: plt.show()
        plt.clf()
        if return_plot_data and not return_area_under_graph: return plot_data
        elif not return_plot_data and return_area_under_graph: return area_under_graph
        else: return plot_data, area_under_graph
    else:
        ax, plot_data, _ = make_plot(mpl_ax)
        if return_plot_data: return ax, plot_data
        else: return ax

def plot_mean_of_average_synaptic_weights_vs_spike_count(network: object, dynamics: object,
                        in_or_out: str, inh_or_exc: str, bins=16, mpl_ax=None, color='b', marker='^',
                        marker_fill=True, marker_size=8, line_style='', line_width=2,
                        xlim=(None, None), ylim=(None, None), plot_label='',
                        return_bin_info=False, return_plot_data=False, custom_bins=[]):
    """Plot the dependence of mean values of average synaptic weights on the number of spikes.
    
    Currently do not use without supplying `mpl_ax`.

    Parameters
    ----------
    network : object
        object of class Network
    dynamics : object
        object of class NeuralDynamics
    in_or_out : str
        the incoming / outgoing links to be plotted, options: 'in' or 'out'
    inh_or_exc : str
        the inhibitory (negative) / excitatory (positive) links to be plotted, options: 'inh' or 'exc'
    bins : int, optional
        number of bins, the resulting number of bins may be smaller due to duplicated bin values, by default 16
    mpl_ax : matplotlib.axes.Axes, optional
        if a matplotlib Axes is given, append the plot to the Axes, by default None
    return_bin_info : bool, optional
        also return the details on bin size, number of data in each bin, etc. if enabled, by default False
    return_plot_data : bool, optional
        also return the data points of the plot if enabled, by default False
    custom_bins : list, optional
        customized bin edges, override `bins` if used, by default []

    Returns
    -------
    matplotlib.axes.Axes
        if `mpl_ax` is provided, return the Axes with the plot appended
    numpy.ndarray
        if `return_bin_info` is True, also return the details on bin size, number of data in each bin, etc.
    numpy.ndarray
        if `return_plot_data` is True, also return the data points of the plot
    """
    
    '''Settings'''
    print_info = False
    save_info = False

    spike_count = dynamics.spike_counts()
    if in_or_out == 'in': s_in_out = network.average_synaptic_weights_of_incoming_links(inh_or_exc)
    elif in_or_out == 'out': s_in_out = network.average_synaptic_weights_of_outgoing_links(inh_or_exc)
    s_in_out = np.array([x if x != None else 0 for x in s_in_out]) # If a neuron has no outgoing/incoming links, assign 0

    spike_count_idxsort = np.flip(spike_count.argsort())
    spike_count = spike_count[spike_count_idxsort[::-1]]
    s_in_out = s_in_out[spike_count_idxsort[::-1]]

    # Calculate bin edges
    bin_edges = [0, ] # You can set the first element set to 0.1 to exclude neurons with no spiking activity
    num_of_data_each_bin = int(len(spike_count) / bins)
    for b in range(bins-1):
        bin_edges.append(spike_count[num_of_data_each_bin + b * num_of_data_each_bin])
    bin_edges.append(spike_count[-1]+1) # +1 as the bin does not include its right edge
    if print_info:
        print('> Number of bins: {}'.format(bins)); print('> Bin edges: {}'.format(bin_edges))
    if save_info:
        fout = open('bin_info.txt')
        fout.write('> Number of bins: {}\n'.format(bins)); fout.write('> Bin edges: {} '.format(bin_edges))
    # Custom bin edges
    if len(custom_bins) != 0:
        bin_edges = custom_bins
        bins = len(bin_edges) - 1
    # Managing duplicates
    bin_edges, duplicate_count = np.unique(bin_edges, return_counts=True)
    duplicate_count = np.sum(duplicate_count-1)
    bins -= duplicate_count
    if print_info:
        print('>> Removed {} duplicated bin.'.format(duplicate_count))
        print('>> Number of bins: {}'.format(bins))
        print('>> Bin edges: {}'.format(list(bin_edges)))
    if save_info:
        fout.write('Removed {} duplicated bin.\n'.format(duplicate_count))
        fout.write('Number of bins: {}\n'.format(bins))
        fout.write('Bin edges: {}\n'.format(str(bin_edges)))

    # Calculate number of data, average spike count for each bin
    bin_number = np.zeros(len(spike_count))
    num_of_data_in_bin = np.zeros(bins)
    avg_spike_count = np.zeros(bins)
    avg_s_in_out = np.zeros(bins)
    for i in range(len(spike_count)):
        for b in range(bins):
            if bin_edges[b] <= spike_count[i] < bin_edges[b+1]:
                bin_number[i] = b
                num_of_data_in_bin[b] += 1
                avg_spike_count[b] += spike_count[i]
                avg_s_in_out[b] += s_in_out[i]
                break
    for b in range(len(avg_spike_count)):
        avg_spike_count[b] /= num_of_data_in_bin[b]
        avg_s_in_out[b] /= num_of_data_in_bin[b]
    if print_info:
        print('>> Number of data in each in: {}'.format(num_of_data_in_bin))
        print('>> S.D. of No. of data: {:f}'.format(np.std(num_of_data_in_bin)))
    if save_info:
        fout.write('Number of data in each in: {}\n'.format(str(num_of_data_in_bin)))
        fout.write('S.D. of No. of data: {:f}'.format(np.std(num_of_data_in_bin)))
    bin_details = (bins, bin_edges, num_of_data_in_bin)

    del spike_count_idxsort
    del spike_count; del s_in_out
    del bin_edges; del bin_number
    del num_of_data_in_bin

    if not mpl_ax == None:
        ax = mpl_ax
        if marker_fill:
            ax.plot(avg_spike_count, avg_s_in_out, marker=marker, c=color, ms=marker_size,
                    ls=line_style, lw=line_width, label=plot_label)
        else:
            ax.plot(avg_spike_count, avg_s_in_out, mfc='none', marker=marker, c=color,
                    ms=marker_size, ls=line_style, lw=line_width, label=plot_label)
        if return_bin_info == True and return_plot_data == True:
            return ax, bin_details, np.array(list(zip(avg_spike_count, avg_s_in_out)))
        elif return_bin_info == True and return_plot_data == False:
            return ax, bin_details
        elif return_bin_info == False and return_plot_data == True:
            return ax, np.array(list(zip(avg_spike_count, avg_s_in_out)))
        else: return ax
    del avg_spike_count, avg_s_in_out

def fread_array(filename: str, dtype=float)->np.ndarray:
    try:
        return np.array(list(csv.reader(open(filename, 'r')))).astype(dtype).flatten()
    except FileNotFoundError:
        err = 'FileNotFoundError: matrix file "{}" cannot be found.'.format(filename)
        print(err); exit(1)

def fread_dense_matrix(filename: str, delim='\t')->np.ndarray:
    """Read a matrix from a file.
    
    The file should store every elements with each row elements separated by `delim`.

    Parameters
    ----------
    filename : str
        name or path of the file
    delim : str, optional
        delimiter of the input matrix file, by default `'\\t'`

    Returns
    -------
    np.ndarray
        the matrix in the file
    """
    try:
        return np.array(list(csv.reader(open(filename, 'r'), delimiter=delim))).astype(float)
    except FileNotFoundError:
        err = 'matrix file "{}" cannot be found'.format(filename)
        raise FileNotFoundError(err)

def fread_sparse_matrix(filename: str, delim=' ', start_idx=1)->np.ndarray:
    """Read a matrix from a file.
    
    The file should store only nonzero elements in each row, with format: j i w_ji, separated by `delim`.

    Parameters
    ----------
    filename : str
        name or path of the file
    delim : str or chr, optional
        delimiter of the matrix file, by default 'whitespace'
    start_idx : int, optional
        the index i and j start from, usually it's 0 or 1, by default 1

    Returns
    -------
    numpy.ndarray
        the matrix in the file
    """
    try:
        with open(filename, 'r', newline='') as fp:
            content = list(csv.reader(fp, delimiter=delim))
            A_size = int(content[0][0])
            for i in range(1, len(content)):
                content[i] = remove_all_occurrences('', content[i])
                content[i][0] = int(content[i][0])-start_idx # j
                content[i][1] = int(content[i][1])-start_idx # i
                content[i][2] = float(content[i][2]) # w_ij
            matrix = np.zeros((A_size, A_size))
            for item in content[1:]: matrix[item[1]][item[0]] = item[2]
            return np.array(matrix).astype(float)
    except FileNotFoundError:
        err = 'matrix file "{}" cannot be found'.format(filename)
        raise FileNotFoundError(err)

def fread_ragged_2darray(filename: str, delim='\t', dtype=float) -> np.ndarray:
    """Read a 2d array with varying row length from a file.

    Parameters
    ----------
    filename : str
        name or path of the file
    delim : str or chr, optional
        delimiter of the file, by default `'\\t'`
    dtype : datatype, optional
        datatype of elements, by default `float`

    Returns
    -------
    np.ndarray
        the 2d ragged array in the file
    """
    try:
        return np.array([np.array(
            list(filter(None, x))).astype(dtype) for x in
                list(csv.reader(open(filename, 'r'), delimiter=delim)
        )], dtype=object)
    except FileNotFoundError:
        err = 'matrix file "{}" cannot be found'.format(filename)
        raise FileNotFoundError(err)

def fwrite_dense_matrix(A: iter, filename: str, delim='\t', dtype=float):
    """Write a matrix into a file.
    
    Write the full matrix, that is, every element including 0. File size will be large.

    Parameters
    ----------
    A : iter
        matrix to be written
    filename : str
        name or path of the file
    delim : str, optional
        delimiter of the output matrix file, by default `'\\t'`
    dtype : datatype, optional
        datatype of elements, by default `float`
    """
    try:
        with open(filename, 'w') as fp:
            if dtype == int:
                for row in A:
                    if row[0] == 0: fp.write('{:d}'.format(row[0])) # To reduce file size
                    else:           fp.write( '{:d}'.format(row[0]))
                    for element in row[1:]:
                        if element == 0:    fp.write('{}{:d}'.format(delim, element)) # To reduce file size
                        else:               fp.write( '{}{:d}'.format(delim, element))
                    fp.write('\n')      
            else:
                for row in A:
                    if row[0] == 0: fp.write('{:.0f}'.format(row[0])) # To reduce file size
                    else:           fp.write( '{:.8}'.format(row[0]))
                    for element in row[1:]:
                        if element == 0:    fp.write('{}{:.0f}'.format(delim, element)) # To reduce file size
                        else:               fp.write( '{}{:.8}'.format(delim, element))
                    fp.write('\n')
    except FileNotFoundError:
        err = 'cannot write to file "{}", e.g., the path to the directory does not exist'.format(filename)
        raise FileNotFoundError(err)

def fwrite_sparse_matrix(A: iter, filename: str, delim=' ', start_idx=1, dtype=float):
    """Write a matrix into a file.
    
    Write only nonzero elements into each row with format: index j, index i, value (separated by `delim`).
    Efficient for sparse matrix with large amount of zero elements.

    Parameters
    ----------
    A : iter
        matrix to be written
    filename : str
        name or path of the file
    delim : str, optional
        delimiter of the output matrix file, by default 'whitespace'
    start_idx : int, optional
        the index i and j start from, usually it's 0 or 1, by default 1
    dtype : datatype, optional
        datatype of elements, by default `float`
    """
    try:
        with open(filename, 'w') as fp:
            fp.write(str(len(A)))
            if dtype == int:
                for j in range(len(A)):
                    for i in range(len(A[j])):
                        if A[i][j] != 0:
                            fp.write('\n{:d}{}{:d}{}{:d}'.format(j+start_idx, delim, i+start_idx, delim, A[i][j]))
            else:
                for j in range(len(A)):
                    for i in range(len(A[j])):
                        if A[i][j] != 0:
                            fp.write('\n{:d}{}{:d}{}{:.10f}'.format(j+start_idx, delim, i+start_idx, delim, A[i][j]))
    except FileNotFoundError:
        err = 'cannot write to file "{}", e.g., the path to the directory does not exist'.format(filename)
        raise FileNotFoundError(err)

def remove_diag(A: np.ndarray)->np.ndarray:
    """Remove diagonal elements from a numpy.ndarray square matrix."""
    if A.shape[0] == A.shape[1]:
        return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)
    else:
        err = 'ArgumentValueError: the input matrix must be square.'
        raise ValueError(err)

def remove_rowcol(A: np.ndarray, rm_idx)->np.ndarray:
    return np.delete(np.delete(A, rm_idx, axis=0), rm_idx, axis=1)

def is_strictly_increasing(seq: iter)->bool:
    """Return True if the input sequence is strictly increasing, else, return False."""
    return all(x > y for x, y in zip(seq, seq[1:]))

def is_strictly_decreasing(seq: iter)->bool:
    """Return True if the input sequence is strictly decreasing, else, return False."""
    return all(x > y for x, y in zip(seq, seq[1:]))

def is_non_increasing(seq: iter)->bool:
    """Return True if the input sequence is non increasing, else, return False."""
    return all(x >= y for x, y in zip(seq, seq[1:]))

def is_non_decreasing(seq: iter)->bool:
    """Return True if the input sequence is non decreasing, else, return False."""
    return all(x <= y for x, y in zip(seq, seq[1:]))

def is_monotonic(seq: iter)->bool:
    """Return True if the input sequence is monotonically increasing or decreasing, else, return False."""
    return is_non_decreasing(seq) or is_non_increasing(seq)

def remove_all_occurrences(x, a: list)->list:
    """Remove all occurences of an element from a list or string."""
    return list(filter((x).__ne__, a))

round_to_n = lambda x, n: x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
"""Round x to n sigificant figures"""


# Help
def ask_for_help():
    """Evoke this function to show help and advanced instructions."""
    print('\n  HOW TO FORMAT PLOTS AND GRAPHS')
    print('  ==============================')
    print(help_graph_formatting.__doc__)

def help_graph_formatting():
    """
    Below shows all optional parameters for the graphing tool.


    stylize_plot()
    --------------
    color | c : str
        color of markers and lines, e.g., `'b', 'C1', 'darkorange'`\n
    marker | m : str
        marker style, e.g., `'o', '^', 'D', ','`\n
    markersize | ms : float
        marker size\n
    markerfacecolor | mfc : str
        marker face color, default: same as `color`
    markerfilled | mf : bool
        solid marker or hollow marker, default `True`
    linestyle | ls : str
        line style, e.g., `'-', '--', '-.', ':'`\n
    lineweight | lw : float
        line weight\n

    label_plot()
    ------------
    title : str
        figure title\n
    plotlabel : str
        plot label to be displayed in legend\n
    axislabel : list of str
        x-axis and y-axis labels, e.g., `['time', 'voltage']`\n
    xlabel : str
        x-axis label\n
    ylabel : str
        y-axis label\n
    xlabelnplot : list of int | int
        show y-axis label in selected subplots, otherwise disable\n
    ylabelnplot : list of int | int
        show y-axis label in selected subplots, otherwise disable\n
    legend : bool | list
        - bool: legend on/off, default `False`
        - list: list of plotlabels (for single plot)
        - list: `[(subplotindex, plotlabels ...), ...]` (for subplots)\n
    legendcombine : bool
        combine legends if there are multiple subplots, default `False`\n
    textbox : str | list
        information to be displayed at top-left corner
        - list: `[(subplotindex, text ...), ...]` (for subplots)\n

    set_scale()
    -----------
    nplot : int
        set the scale of the n-th sub-plot if there are multiple subplots\n
    grid : bool
        grid on/off, default `True`\n
    gridnplot : list of int
        the indexes of subplot whose grid is on, default: all grids on\n
    minortick : bool
        minortick on/off, default: `True`\n
    xlim : tuple
        horizontal plot range, default: fitted to data\n
    ylim : tuple
        vertical plot range, default: fitted to data\n
    xlogscale : bool
        use log scale in x-axis, default `False`\n
    ylogscale : bool
        use log scale in y-axis, default `False`\n
    xsymlogscale : bool
        use symmetric log scale in x-axis (should be used with `xlinthresh`), default `False`\n
    xlinthresh : float
        the threshold of linear range when using `xsymlogscale`, default `1`\n
    ysymlogscale : bool
        use symmetric log scale in y-axis (should be used with `ylinthresh`), default `False`\n
    ylinthresh : float
        the threshold of linear range when using `ysymlogscale`, default `1`\n

    other arguments
    ---------------
    fitgaussian : bool
        fit the data with a Gaussian curve, by default `False`\n
    filelabel : str
        label attached at the end of the file name, by default `''`\n
    fileextension : str
        file extension, by default `'png'`\n
    subplotindex : int
        index of sub-plot to be drawn into, by default `0`\n
    saveplotinfo : bool
        return 1: the data points; 2: the details on bin size, number of data in each bin, etc., default `False`\n
    density : bool
        use probability density instead of histogram, default `True`\n
    histogram : bool
        use histogram instead of probability density, default `False`\n
    normalize_to_factor : float
        for probability density graphs, normalize the area under graph to a factor, default `1`\n
    """
    print('\n  HOW TO FORMAT PLOTS AND GRAPHS')
    print('  ==============================')
    print(help_graph_formatting.__doc__)


# Miscellaneous
def _colorline(x, y, z=None, cmap='hsv', norm=plt.Normalize(0.0, 1.0), linewidth=1, alpha=1, ax=None):
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    segments = _make_segments(x, y)
    lc = mcol.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    if ax==None: ax = plt.gca()
    ax.add_collection(lc)
    return lc

def _make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
