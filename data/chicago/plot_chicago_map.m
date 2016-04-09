function [ax_label, ax_im ] = plot_chicago_map(varargin)
%PLOT_CHICAGO_MAP Plot a map of chicago

% FONTS: Use consistent fonts for the axes labels and titles.
font_name  = 'Helvetica';
font_size  = 8;

% FIGURE SIZE: Adjust this as necessary to match your data.
fig_width  = 9; 
fig_height = 6;

% POSITION: Matlab doesn't use the figure space well by default.
if nargin < 1
    position   = [0.1 0.1 0.8 0.8];
else
    position = varargin{1};
end

line_width = 1;

% Setup some defaults.
set(0, 'DefaultTextInterpreter', 'tex', ...
       'DefaultTextFontName',    font_name, ...
       'DefaultTextFontSize',    font_size, ...
       'DefaultAxesFontSize',    font_size);

% Create the figure.
fig = figure('Units', 'inches', ...
       'Position', [0 0 fig_width fig_height], ...
       'PaperPositionMode', 'auto');

% This is only necessary if you're making one plot.  Which you
% should, by the way.  If you have subplots, use subfig in LaTeX.
ax_im = subplot('Position', position);

map = imread('streetmap_41.6_42.1_-87.9_-87.5_2.png');
da = [1,0.75,1];

% Since imagesc puts origin in upper left and plot puts origin in lower
% left, we need to reverse the y axis (latitude)
xb = [-87.9, -87.5];
yb = [41.6, 42.1];

imagesc(xb,yb,map)
set(ax_im,'Position', position,'XLim',xb,'YLim',yb,'DataAspectRatio',da,'Visible','off');
hold on

% Now overlay a new set of axes with the correct ordering
ax_label = axes();
% Correct the tick labels
set(ax_label,'Position',position,'XLim',xb,'YLim',yb,'DataAspectRatio',da,'color','none');
 
% Set labels
ylabel(ax_label,'Latitude');
xlabel(ax_label,'Longitude');

% Set hold
hold on

% Make sure that the axes remain aligned upon resize
resize_fn = @(src,event) set(ax_im,'Position',position);

set(fig,'ResizeFcn',resize_fn);

end

