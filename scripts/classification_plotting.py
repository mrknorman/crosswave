import bokeh.plotting as bk
import bokeh.palettes  as palettes

from bokeh.models import ColorBar
from bokeh.transform import linear_cmap
import pandas as pd

from bokeh.plotting import figure, save
from bokeh.layouts import gridplot, layout
from bokeh.models import ColumnDataSource, Select
from bokeh.resources import CDN
from bokeh.embed import file_html

import pandas as pd
import numpy as np

input_file_name = "skywarp_validation_scores_regression"
output_file_name = "skywarp_validation_scores_regression"

# Load dataframe:
df = pd.read_pickle(f"{input_file_name}.pkl")

print(df['spin1x_signal_a'].min(), df['spin1x_signal_a'].max())

# Set output file:
bk.output_file(f"{output_file_name}.html")

df['difference_a'] = (df['H1_time_signal_a_'] - df['results_A']).abs()
df['difference_b'] = (df['H1_time_signal_b_'] - df['results_B']).abs()

print(np.mean(df['difference_a']))
print(np.mean(df['difference_b']))

#df['difference'] = (df['model_prediction'] - df['overlap_present']).abs()

x_axis = 'H1_SNR_signal_a'
y_axis = 'H1_SNR_signal_b'

figure_single = bk.figure(        
    x_axis_label = x_axis,
    y_axis_label = "Score difference"
    )

figure_single.circle(df[x_axis], df['difference_a'], size=1, color="navy", alpha=0.5)

figure_twin = bk.figure(        
    x_axis_label = x_axis,
    y_axis_label = y_axis
    )

pallet = palettes.Plasma[11]

mapper = linear_cmap(field_name='difference_a', palette=pallet ,low=min(df['difference_a']) ,high=max(df['difference_a']))
figure_twin.circle(x_axis, y_axis, source=df, size=3, color=mapper, alpha=0.5)

color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0),title="Difference")
figure_twin.add_layout(color_bar, 'right')

bk.save(figure_twin)





