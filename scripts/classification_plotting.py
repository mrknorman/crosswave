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

def plot_classification_scores(
        input_file_name : str = "./validation_results/validation_scores_classification",
        output_file_name : str = "./validation_results/validation_scores_classification"
    ):

    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    # Set output file:
    bk.output_file(f"{output_file_name}.html")

    data["difference"] = (data["model_prediction"] - data["overlap_present"]).abs()
    data["Network SNR Signal A"] = np.sqrt(data["H1_SNR_signal_a"]**2 + data["L1_SNR_signal_a"]**2)
    data["Network SNR Signal B"] = np.sqrt(data["H1_SNR_signal_b"]**2 + data["L1_SNR_signal_b"]**2)

    x_axis = "Network SNR Signal A"
    y_axis = "Network SNR Signal B"

    figure_twin = bk.figure(        
        x_axis_label = x_axis,
        y_axis_label = y_axis
    )

    pallet = palettes.Plasma[11]

    mapper = linear_cmap(
        field_name="difference", 
        palette=pallet,
        low=min(data["difference"]),
        high=max(data["difference"])
    )

    figure_twin.circle(
        x_axis, 
        y_axis, 
        source=data, 
        size=3, 
        color=mapper, 
        alpha=0.5
    )

    color_bar = ColorBar(
        color_mapper=mapper['transform'], 
        width=8, 
        location=(0,0), 
        title="Classification Score Error"
    )
    figure_twin.add_layout(color_bar, 'right')

    bk.save(figure_twin)

def plot_regression_scores(
        input_file_name : str = "./validation_results/validation_scores_regression",
        output_file_name : str = "./validation_results/validation_scores_regression"
    ):

    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    data['Error Signal Time A'] = (data['H1_time_signal_a_'] - data['results_A']).abs()
    data['Error Signal Time B'] = (data['H1_time_signal_b_'] - data['results_B']).abs()

    data["Network SNR Signal A"] = np.sqrt(data["H1_SNR_signal_a"]**2 + data["L1_SNR_signal_a"]**2)
    data["Network SNR Signal B"] = np.sqrt(data["H1_SNR_signal_b"]**2 + data["L1_SNR_signal_b"]**2)

    x_axis = "Network SNR Signal A"
    y_axis = "Network SNR Signal B"

    figure_a_time = bk.figure(        
        x_axis_label = x_axis,
        y_axis_label = y_axis
    )

    pallet = palettes.Plasma[11]

    common_low = 0.0
    common_high = max(max(data["Error Signal Time A"]), min(data["Error Signal Time A"]))

    mapper = linear_cmap(
        field_name="Error Signal Time A", 
        palette=pallet,
        low=common_low,
        high=common_high
    )

    figure_a_time.circle(
        x_axis, 
        y_axis, 
        source=data, 
        size=3, 
        color=mapper, 
        alpha=0.5
    )

    color_bar = ColorBar(
        color_mapper=mapper['transform'], 
        width=8, 
        location=(0,0), 
        title="Error Signal Time A"
    )
    figure_a_time.add_layout(color_bar, 'right')
    
    # Set output file:
    bk.output_file(f"{output_file_name}_A.html")
    bk.save(figure_a_time)

    figure_b_time = bk.figure(        
        x_axis_label = x_axis,
        y_axis_label = y_axis
    )

    mapper = linear_cmap(
        field_name="Error Signal Time B", 
        palette=pallet,
        low=common_low,
        high=common_high
    )

    figure_b_time.circle(
        x_axis, 
        y_axis, 
        source=data, 
        size=3, 
        color=mapper, 
        alpha=0.5
    )

    color_bar = ColorBar(
        color_mapper=mapper['transform'], 
        width=8, 
        location=(0,0), 
        title="Error Signal Time B"
    )
    figure_b_time.add_layout(color_bar, 'right')
    
    # Set output file:
    bk.output_file(f"{output_file_name}_B.html")
    bk.save(figure_b_time)

if __name__ == "__main__":
    plot_classification_scores()
    plot_regression_scores()



