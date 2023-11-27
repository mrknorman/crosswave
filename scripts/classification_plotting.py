import bokeh.plotting as bk
import bokeh.palettes  as palettes

from bokeh.models import ColorBar
from bokeh.transform import linear_cmap
import pandas as pd

from bokeh.plotting import figure, save
from bokeh.layouts import gridplot, layout
from bokeh.models import ColumnDataSource, Select, Range1d, LinearAxis, CustomJSTickFormatter
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

def difference_plot(
        input_file_name : str = "./validation_results/validation_scores_regression",
        output_file_name: str = "./validation_results/validation_scores_regression",
    ):

    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    data['H1_time_signal_a_'] -= 10
    data['H1_time_signal_b_'] -= 10

    data['results_A'] -= 10
    data['results_B'] -= 10

    data['Error Signal Time A'] = (data['H1_time_signal_a_'] - data['results_A']).abs()
    data['Error Signal Time B'] = (data['H1_time_signal_b_'] - data['results_B']).abs()
    data["Time Differnece"] = (data['H1_time_signal_b_'] - data['H1_time_signal_a_']).abs() 

    # Create a figure
    p = figure(x_axis_label='Arrival Time Difference (s)', y_axis_label='Prediction Error (s)')

    # Finding the maximum absolute value for y-axis range
    max_range = max(data['Error Signal Time A'].abs().max(), data['Error Signal Time B'].abs().max())

    # Setting the y-axis range to be symmetric around zero
    p.y_range = Range1d(-max_range, max_range)

    # Adding lines to the plot
    p.circle(data['Time Differnece'], data['Error Signal Time A'], legend_label="Error Signal Time A", color="blue")
    p.circle(data['Time Differnece'], -data['Error Signal Time B'], legend_label="Error Signal Time B", color="red")  # Inverting B

    # Custom tick formatter to display absolute values
    p.yaxis.formatter = CustomJSTickFormatter(code="""
        return Math.abs(tick).toString();
    """)

    # Set output file:
    bk.output_file(f"difference_plot.html")
    bk.save(p)

def difference_plot_best(
        input_file_name : str = "./validation_results/crosswave_validation_scores_regression_best",
    ):
    
    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    data['H1_time_signal_a'] -= 60
    data['H1_time_signal_b'] -= 60 

    data['H1_time_signal_a_pred'] *= 4
    data['H1_time_signal_b_pred'] *= 4

    data['Error Signal Time A'] = (data['H1_time_signal_a'] - data['H1_time_signal_a_pred']).abs()
    data['Error Signal Time B'] = (data['H1_time_signal_b'] - data['H1_time_signal_b_pred']).abs()
    data["Time Differnece"] = (data['H1_time_signal_b'] - data['H1_time_signal_a']).abs()

    # Create a figure
    p = figure(x_axis_label='Arrival Time Difference (s)', y_axis_label='Prediction Error (s)')

    # Finding the maximum absolute value for y-axis range
    max_range = max(data['Error Signal Time A'].abs().max(), data['Error Signal Time B'].abs().max())

    # Setting the y-axis range to be symmetric around zero
    p.y_range = Range1d(-max_range, max_range)

    # Adding lines to the plot
    p.circle(data['Time Differnece'], data['Error Signal Time A'], legend_label="Error Signal Time A", color="blue")
    p.circle(data['Time Differnece'], -data['Error Signal Time B'], legend_label="Error Signal Time B", color="red")  # Inverting B

    # Custom tick formatter to display absolute values
    p.yaxis.formatter = CustomJSTickFormatter(code="""
        return Math.abs(tick).toString();
    """)

    # Set output file:
    bk.output_file(f"difference_plot_best.html")
    bk.save(p)

def difference_plot_best_difference(
        input_file_name : str = "./validation_results/crosswave_validation_scores_regression_best",
    ):
    
    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    data['H1_time_signal_a'] -= 60
    data['H1_time_signal_b'] -= 60 

    data['H1_time_signal_a_pred'] *= 4
    data['H1_time_signal_b_pred'] *= 4

    data['Error Signal Time A'] = (data['H1_time_signal_a'] - data['H1_time_signal_a_pred']).abs()
    data['Error Signal Time B'] = (data['H1_time_signal_b'] - data['H1_time_signal_b_pred']).abs()
    data["Time Difference"] = (data['H1_time_signal_b'] - data['H1_time_signal_a']).abs()
    data["Predicted Time Difference"] = (data['H1_time_signal_a_pred'] - data['H1_time_signal_b_pred']).abs()

    # Create a figure
    p = figure(x_axis_label='Arrival Time Difference (s)', y_axis_label='Predicted Time Difference (s)')

    # Finding the maximum absolute value for y-axis range
    max_range = max(data['Error Signal Time A'].abs().max(), data['Error Signal Time B'].abs().max())

    # Setting the y-axis range to be symmetric around zero
    p.y_range = Range1d(0, max_range)

    # Adding lines to the plot
    p.circle(data['Time Difference'], data["Predicted Time Difference"], color="blue")

    # Set output file:
    bk.output_file(f"difference_plot_best_difference.html")
    bk.save(p)

def difference_plot_best_difference_error(
        input_file_name : str = "./validation_results/crosswave_validation_scores_regression_best",
    ):
    
    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    data['H1_time_signal_a'] -= 60
    data['H1_time_signal_b'] -= 60 

    data['H1_time_signal_a_pred'] *= 4
    data['H1_time_signal_b_pred'] *= 4

    data['Error Signal Time A'] = (data['H1_time_signal_a'] - data['H1_time_signal_a_pred']).abs()
    data['Error Signal Time B'] = (data['H1_time_signal_b'] - data['H1_time_signal_b_pred']).abs()
    data["Time Difference"] = (data['H1_time_signal_b'] - data['H1_time_signal_a']).abs()
    data["Predicted Time Difference"] = (data['H1_time_signal_a_pred'] - data['H1_time_signal_b_pred']).abs()

    # Create a figure
    p = figure(x_axis_label='Arrival Time Difference (s)', y_axis_label='Predicted Time Difference Error (s)')

    # Finding the maximum absolute value for y-axis range
    max_range = max(data['Error Signal Time A'].abs().max(), data['Error Signal Time B'].abs().max())

    # Setting the y-axis range to be symmetric around zero
    p.y_range = Range1d(0, max_range)

    # Adding lines to the plot
    p.circle(data['Time Difference'], (data["Time Difference"] - data["Predicted Time Difference"]).abs(), color="blue")

    # Set output file:
    bk.output_file(f"difference_plot_best_difference_error.html")
    bk.save(p)

def difference_plot_best_difference_snr(
        input_file_name : str = "./validation_results/crosswave_validation_scores_regression_best",
    ):
    
    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    data['H1_time_signal_a'] -= 60
    data['H1_time_signal_b'] -= 60 

    data['H1_time_signal_a_pred'] *= 4
    data['H1_time_signal_b_pred'] *= 4

    data['Error Signal Time A'] = (data['H1_time_signal_a'] - data['H1_time_signal_a_pred']).abs()
    data['Error Signal Time B'] = (data['H1_time_signal_b'] - data['H1_time_signal_b_pred']).abs()
    data["Time Difference"] = (data['H1_time_signal_b'] - data['H1_time_signal_a']).abs()
    data["Predicted Time Difference"] = (data['H1_time_signal_a_pred'] - data['H1_time_signal_b_pred']).abs()

    # Create a figure
    p = figure(x_axis_label='Optimal SNR Signal B', y_axis_label='Predicted Time Difference Error (s)')

    # Finding the maximum absolute value for y-axis range
    max_range = max(data['Error Signal Time A'].abs().max(), data['Error Signal Time B'].abs().max())

    # Setting the y-axis range to be symmetric around zero
    p.y_range = Range1d(0, max_range)

    data["Network SNR Signal B"] = np.sqrt(data["H1_SNR_signal_b"]**2 + data["L1_SNR_signal_b"]**2)


    # Adding lines to the plot
    p.circle(data["Network SNR Signal B"], (data["Time Difference"] - data["Predicted Time Difference"]).abs(), legend_label="Error Signal Time A", color="blue")

    # Set output file:
    bk.output_file(f"difference_plot_best_difference_snr.html")
    bk.save(p)

if __name__ == "__main__":
    difference_plot()
    difference_plot_best()
    difference_plot_best_difference()
    difference_plot_best_difference_error()
    difference_plot_best_difference_snr()
    plot_classification_scores()
    plot_regression_scores()



