import bokeh.plotting as bk
import bokeh.palettes  as palettes

from bokeh.models import ColorBar
from bokeh.transform import linear_cmap
import pandas as pd

from bokeh.plotting import figure, save
from bokeh.layouts import gridplot, layout
from bokeh.models import ColumnDataSource, Select, Range1d, LinearAxis, CustomJSTickFormatter
from bokeh.models import ColumnDataSource, Select, CustomJS, LinearColorMapper, ColorBar
from bokeh.resources import CDN
from bokeh.embed import file_html

import pandas as pd
import numpy as np

def reorder(data):
    data = data.sort_values(by='original_index')
    preseverd_cols = data[["model_prediction", "overlap_present"]]
    preseverd_cols = preseverd_cols.reset_index(drop=True)

    columns_a = [col for col in data.columns if col.endswith('_a')]
    columns_a_data = data[columns_a][:10000]
    a_doubled = pd.concat([columns_a_data, columns_a_data], ignore_index=True)

    columns_b_data = data[columns_a][10000:20000]
    b_doubled = pd.concat([columns_b_data, columns_b_data], ignore_index=True)
    b_doubled = b_doubled.rename(columns={col: col.replace('_a', '_b') for col in b_doubled.columns})

    new_doubles = a_doubled.join(b_doubled)
    new_doubles = new_doubles.join(preseverd_cols[20000:].reset_index(drop=True))
    
    data = pd.concat([data[:20000], new_doubles])
    
    return data

def plot_classification_scores_separation(
    input_file_name: str = "./validation_results/validation_scores_classification",
    output_file_name: str = "./validation_results/validation_scores_classification_separation",
    rolling_window_size: int = 500 # You can adjust this value as needed
):
    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    # Set output file:
    bk.output_file(f"{output_file_name}.html")
    
    # Assume reorder is a function defined elsewhere that reorders the data
    data = reorder(data)
    data = data[data['overlap_present'] == 1]

    # Calculate the absolute difference between model predictions and actual overlap
    data["model_prediction_error"] = (data["model_prediction"] - data["overlap_present"]).abs()
    
    # Calculate the absolute time separation between signal a and signal b
    data["time_separation"] = (data["geocent_time_signal_a"] - data["geocent_time_signal_b"]).abs()

    # Sort data by time_separation to calculate rolling average
    data_sorted = data.sort_values('time_separation')

    # Calculate rolling average of model_prediction_error
    data_sorted["rolling_avg_error"] = data_sorted["model_prediction_error"].rolling(window=rolling_window_size, center=True).mean()

    # Create a new figure with appropriate axis labels
    figure_separation = bk.figure(
        title="Time Separation vs Model Prediction Error",
        x_axis_label="Time Separation (s)",
        y_axis_label="Model Prediction Error"
    )

    # Choose a palette for the color mapper
    palette = palettes.Plasma[11]

    # Create a linear color mapper for the points based on model prediction error
    mapper = linear_cmap(
        field_name="model_prediction_error", 
        palette=palette,
        low=min(data["model_prediction_error"]),
        high=max(data["model_prediction_error"])
    )

    # Add circle glyphs to the figure for the scatter points
    figure_separation.circle(
        "time_separation", 
        "model_prediction_error", 
        source=data, 
        size=3, 
        color=mapper, 
        alpha=0.5
    )
    
    df_nth = data_sorted.iloc[::rolling_window_size, :]
    # Add a line glyph for the rolling average
    figure_separation.line(
        df_nth['time_separation'], 
        df_nth['rolling_avg_error'], 
        line_width=4, 
        color='red'
    )

    # Create a color bar to show the model prediction error scale
    color_bar = ColorBar(
        color_mapper=mapper['transform'], 
        width=8, 
        location=(0,0),
        title="Model Prediction Error"
    )
    
    # Add the color bar to the figure
    figure_separation.add_layout(color_bar, 'right')

    # Save the figure to an HTML file
    bk.save(figure_separation)

def plot_classification_scores(
        input_file_name : str = "./validation_results/validation_scores_classification",
        output_file_name : str = "./validation_results/validation_scores_classification"
    ):

    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    # Set output file:
    bk.output_file(f"{output_file_name}.html")
    
    data = reorder(data)

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
    
def plot_classification_scores_mass(
    input_file_name: str = "./validation_results/validation_scores_classification",
    output_file_name: str = "./validation_results/validation_scores_classification_chirp_mass"
):
    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    # Set output file:
    bk.output_file(f"{output_file_name}.html")
    
    # Assume reorder is a function defined elsewhere that reorders the data
    data = reorder(data)

    data["difference"] = (data["model_prediction"] - data["overlap_present"]).abs()
    
    x_axis_data = "chirp_mass_signal_a"  # column name for x-axis data
    y_axis_data = "chirp_mass_signal_b"  # column name for y-axis data

    # New titles for the axes
    new_x_axis_title = "Chirp Mass of Signal A"
    new_y_axis_title = "Chirp Mass of Signal B"

    figure_twin = bk.figure(        
        x_axis_label = new_x_axis_title,  # Use the new x-axis title
        y_axis_label = new_y_axis_title,  # Use the new y-axis title
    )

    pallet = palettes.Plasma[11]

    mapper = linear_cmap(
        field_name="difference", 
        palette=pallet,
        low=min(data["difference"]),
        high=max(data["difference"])
    )

    figure_twin.circle(
        x=x_axis_data,  # use the actual column name for the x-axis data
        y=y_axis_data,  # use the actual column name for the y-axis data
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
    
def plot_classification_scores_ratio(
    input_file_name: str = "./validation_results/validation_scores_classification",
    output_file_name: str = "./validation_results/validation_scores_classification_mass_ratio"
):
    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    # Set output file:
    bk.output_file(f"{output_file_name}.html")
    
    # Assume reorder is a function defined elsewhere that reorders the data
    data = reorder(data)

    data["difference"] = (data["model_prediction"] - data["overlap_present"]).abs()
    
    x_axis = "mass_ratio_signal_a"  # column name for x-axis data
    y_axis = "mass_ratio_signal_b"  # column name for y-axis data

    # Set new titles for the axes here
    new_x_axis_title = "Mass Ratio of Signal A"
    new_y_axis_title = "Mass Ratio of Signal B"

    figure_twin = bk.figure(        
        x_axis_label = new_x_axis_title,  # Use the new x-axis title
        y_axis_label = new_y_axis_title,  # Use the new y-axis title
    )

    pallet = palettes.Plasma[11]

    mapper = linear_cmap(
        field_name="difference", 
        palette=pallet,
        low=min(data["difference"]),
        high=max(data["difference"])
    )

    figure_twin.circle(
        x=x_axis,  # use the actual column name for the x-axis data
        y=y_axis,  # use the actual column name for the y-axis data
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
    
def plot_classification_efficiencies(
        input_file_name : str = "./validation_results/validation_scores_classification",
        output_file_name : str = "./validation_results/validation_scores_efficiency",
        rolling_window_size: int = 1000, # You can adjust the window size as needed
        nth_value: int = 100  # Plot every nth value
    ):
    
    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    # Set output file:
    bk.output_file(f"{output_file_name}.html")
    
    df = reorder(data)
    
    df_single = df[df['overlap_present'] == 0]
    
    df = df[df['overlap_present'] == 1]
    
    df["Network SNR Signal A"] = np.sqrt(df["H1_SNR_signal_a"]**2 + df["L1_SNR_signal_a"]**2)
    df["Network SNR Signal B"] = np.sqrt(df["H1_SNR_signal_b"]**2 + df["L1_SNR_signal_b"]**2)
    
    # Calculate the minimum Network SNR
    df["Minimum Network SNR"] = np.minimum(df["Network SNR Signal A"], df["Network SNR Signal B"])
    df["Maximum Network SNR"] = np.maximum(df["Network SNR Signal A"], df["Network SNR Signal B"])
    
    min_snr = 150
    # Filter for Minimum Network SNR < 30 and Sort by Minimum Network SNR
    df = df[df["Minimum Network SNR"] < min_snr].sort_values(by="Minimum Network SNR")
    df["Rolling Avg Model Prediction"] = df['model_prediction'].rolling(window=rolling_window_size).mean()
    # Calculate rolling average of model_prediction

    # Select every nth value
    df_nth = df.iloc[::nth_value, :]
    # Create a new plot with a title and axis labels
    p = figure(
        x_axis_label='Minimum Network SNR', 
        y_axis_label='Rolling Average Model Prediction',
        width = 800
    )
    # Add a scatter renderer with a size, color, and alpha for the rolling average
    p.line(
        df_nth['Minimum Network SNR'], 
        df_nth['Rolling Avg Model Prediction'], 
        width=2, 
        color=palettes.Bright[7][0],
        legend_label="Minimum Signal SNR vs Model Prediction"
    )
    
    df = df[df["Maximum Network SNR"] < min_snr].sort_values(by="Maximum Network SNR")
    df["Rolling Avg Model Prediction Max"] = df['model_prediction'].rolling(window=rolling_window_size).mean()
    df_nth = df.iloc[::nth_value, :]
    p.line(
        df_nth['Maximum Network SNR'], 
        df_nth['Rolling Avg Model Prediction Max'], 
        width=2, 
        color=palettes.Bright[7][3],
        legend_label="Maximum Signal SNR vs Model Prediction"
    )
    
    df = df[df["Network SNR Signal A"] < min_snr].sort_values(by="Network SNR Signal A")
    df["Rolling Avg Model Prediction A"] = df['model_prediction'].rolling(window=rolling_window_size).mean()
    df_nth = df.iloc[::nth_value, :]
    p.line(
        df_nth['Network SNR Signal A'], 
        df_nth['Rolling Avg Model Prediction A'], 
        width=2, 
        color=palettes.Bright[7][1],
        legend_label="Signal A SNR vs Model Prediction"
    )
    
    df = df[df["Network SNR Signal B"] < min_snr].sort_values(by="Network SNR Signal B")
    df["Rolling Avg Model Prediction B"] = df['model_prediction'].rolling(window=rolling_window_size).mean()
    df_nth = df.iloc[::nth_value, :]
    p.line(
        df_nth['Network SNR Signal B'], 
        df_nth['Rolling Avg Model Prediction B'], 
        width=2, 
        color=palettes.Bright[7][2],
        legend_label="Signal B SNR vs Model Prediction"
    )
    
    df_single["Network SNR Signal A"] = np.sqrt(df_single["H1_SNR_signal_a"]**2 + df_single["L1_SNR_signal_a"]**2)
    df_single = df_single[df_single["Network SNR Signal A"] < min_snr].sort_values(by="Network SNR Signal A")
    df_single["Rolling Avg Model Prediction A"] = df_single['model_prediction'].rolling(window=rolling_window_size).mean()
    df_nth = df_single.iloc[::nth_value, :]
    p.line(
        df_nth['Network SNR Signal A'], 
        df_nth['Rolling Avg Model Prediction A'], 
        width=2, 
        color=palettes.Bright[7][2],
        legend_label="Single Signal SNR vs Model Prediction"
    )
    
    p.legend.title = 'Legend'
    p.legend.location = "center_right"
    p.legend.margin = 5

    # Specify the name of the output file and save the plot
    save(p)

def plot_regression_efficiencies(
        input_file_name : str = "./validation_results/crosswave_validation_scores_regression_best",
        output_file_name : str = "./validation_results/validation_scores_regression_efficiency",
        rolling_window_size: int = 1000, # You can adjust the window size as needed
        nth_value: int = 100  # Plot every nth value
    ):
    
    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    data['H1_time_signal_a_pred'] *= 4
    data['H1_time_signal_b_pred'] *= 4
    data['H1_time_signal_a'] -= 60
    data['H1_time_signal_b'] -= 60 

    df = data[data['H1_time_signal_b'] > 0].copy()

    df['Error Signal Time A'] = (df['H1_time_signal_a'] - df['H1_time_signal_a_pred']).abs()
    df['Error Signal Time B'] = (df['H1_time_signal_b'] - df['H1_time_signal_b_pred']).abs()
    
    df["Network SNR Signal A"] = np.sqrt(df["H1_SNR_signal_a"]**2 + df["L1_SNR_signal_a"]**2)
    df["Network SNR Signal B"] = np.sqrt(df["H1_SNR_signal_b"]**2 + df["L1_SNR_signal_b"]**2)
    
    # Calculate the minimum Network SNR
    df["Minimum Network SNR"] = np.minimum(df["Network SNR Signal A"], df["Network SNR Signal B"])
    df["Maximum Network SNR"] = np.maximum(df["Network SNR Signal A"], df["Network SNR Signal B"])
    
    for error_signal, name in zip(['Error Signal Time A', 'Error Signal Time B'], ["A", "B"]):

        # Set output file:
        bk.output_file(f"{output_file_name}_{name}.html")

        min_snr = 150
        
        # Create a new plot with a title and axis labels
        p = figure(
            x_axis_label='Network SNR',
            y_axis_label=f'Rolling Average Signal {name} Merger Time Error (s)',
            width = 800
        )

        # Filter for Minimum Network SNR < 30 and Sort by Minimum Network SNR
        df = df[df["Minimum Network SNR"] < min_snr].sort_values(by="Minimum Network SNR")
        df["Rolling Average Prediction Error (s)"] = df[error_signal].rolling(window=rolling_window_size).mean()
        # Select every nth value
        df_nth = df.iloc[::nth_value, :]
        # Add a scatter renderer with a size, color, and alpha for the rolling average
        p.line(
            df_nth['Minimum Network SNR'], 
            df_nth['Rolling Average Prediction Error (s)'], 
            width=2, 
            color=palettes.Bright[7][0],
            legend_label="Minimum Signal SNR vs Model Prediction"
        )
        
        df = df[df["Maximum Network SNR"] < min_snr].sort_values(by="Maximum Network SNR")
        df["Rolling Average Prediction Error (s)"] = df[error_signal].rolling(window=rolling_window_size).mean()

        df_nth = df.iloc[::nth_value, :]
        p.line(
            df_nth['Maximum Network SNR'], 
            df_nth['Rolling Average Prediction Error (s)'], 
            width=2, 
            color=palettes.Bright[7][3],
            legend_label="Maximum Signal SNR vs Model Prediction"
        )
        
        df = df[df["Network SNR Signal A"] < min_snr].sort_values(by="Network SNR Signal A")
        df["Rolling Average Prediction Error (s)"] = df[error_signal].rolling(window=rolling_window_size).mean()

        df_nth = df.iloc[::nth_value, :]
        p.line(
            df_nth['Network SNR Signal A'], 
            df_nth['Rolling Average Prediction Error (s)'], 
            width=2, 
            color=palettes.Bright[7][1],
            legend_label="Signal A SNR vs Model Prediction"
        )
        
        df = df[df["Network SNR Signal B"] < min_snr].sort_values(by="Network SNR Signal B")
        df["Rolling Average Prediction Error (s)"] = df[error_signal].rolling(window=rolling_window_size).mean()
        df_nth = df.iloc[::nth_value, :]
        p.line(
            df_nth['Network SNR Signal B'], 
            df_nth['Rolling Average Prediction Error (s)'], 
            width=2, 
            color=palettes.Bright[7][2],
            legend_label="Signal B SNR vs Model Prediction"
        )
        
        p.legend.title = 'Legend'
        p.legend.location = "center_right"
        p.legend.margin = 5

        # Specify the name of the output file and save the plot
        save(p)
    
def plot_regression_scores(
        input_file_name : str = "./validation_results/crosswave_validation_scores_regression_best",
        output_file_name : str = "./validation_results/validation_scores_regression"
    ):

    # Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")

    data['H1_time_signal_a_pred'] *= 4
    data['H1_time_signal_b_pred'] *= 4
    data['H1_time_signal_a'] -= 60
    data['H1_time_signal_b'] -= 60 

    data = data[data['H1_time_signal_b'] > 0]

    data['Error Signal Time A'] = data['H1_time_signal_a'] - data['H1_time_signal_a_pred']
    data['Error Signal Time B'] = data['H1_time_signal_b'] - data['H1_time_signal_b_pred']

    data = data[data['Error Signal Time A'] < 0.25]
    data = data[data['Error Signal Time B'] < 0.25]

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
    common_high = max(
        max(data["Error Signal Time A"]), 
        min(data["Error Signal Time A"])
    )

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
        input_file_name : str = "./validation_results/crosswave_validation_scores_regression_best",
        output_file_name: str = "./validation_results/crosswave_validation_scores_regression_best",
        rolling_window_size=500
    ):

    #Load dataframe:
    data = pd.read_pickle(f"{input_file_name}.pkl")
    
    data['H1_time_signal_a_pred'] *= 4
    data['H1_time_signal_b_pred'] *= 4
    data['H1_time_signal_a'] -= 60
    data['H1_time_signal_b'] -= 60 

    data = data[data["H1_time_signal_b"] > 0]
    data = data[data["H1_time_signal_b_pred"] > 0]

    data['Error Signal Time A'] = data['H1_time_signal_a_pred'] - data['H1_time_signal_a']
    data['Error Signal Time B'] = data['H1_time_signal_b_pred'] - data['H1_time_signal_b']
    data["Time Difference"] = data['H1_time_signal_a'] - data['H1_time_signal_b']

    for error_signal, title in zip(['Error Signal Time A', 'Error Signal Time B'], ['A', 'B']):
        # Sort data based on 'Time Difference' for meaningful rolling average
        sorted_data = data.sort_values(by='Time Difference')
        
        sorted_data[f"abs_{error_signal}"] = sorted_data[error_signal].abs()

        # Calculate rolling average
        sorted_data[f"{error_signal}_rolling_avg"] = sorted_data[error_signal].abs().rolling(window=rolling_window_size).mean()

        # Create a color mapper
        color_mapper = LinearColorMapper(palette=palettes.Plasma[256], low=sorted_data[f"abs_{error_signal}"].min(), high=sorted_data[f"abs_{error_signal}"].max())

        # Create a figure
        p = figure(x_axis_label='Arrival Time Difference (s)', y_axis_label=f'Prediction Error Signal {title} (s)')

        # Finding the maximum absolute value for y-axis range
        max_range = sorted_data[error_signal].max()

        # Setting the y-axis range to be symmetric around zero
        p.y_range = Range1d(-max_range, max_range)

        # Add circles with color mapped to absolute error
        source = ColumnDataSource(sorted_data)
        p.circle('Time Difference', error_signal, source=source, color={'field': f"abs_{error_signal}", 'transform': color_mapper}, line_color=None)

        # Add rolling average line
        df_nth = sorted_data.iloc[::rolling_window_size, :]
        source = ColumnDataSource(df_nth)
        p.line('Time Difference', f"{error_signal}_rolling_avg", source=source, line_width=2, color="red")

        # Add a color bar
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0,0), title=f"Absolute Error Signal {title} (s)")
        p.add_layout(color_bar, 'right')

        # Set output file and save
        bk.output_file(f"{output_file_name}_difference_plot_{title}.html")
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
    
    plot_classification_scores()
    plot_classification_scores_separation()
    plot_classification_scores_mass()
    plot_classification_scores_ratio()    
    plot_classification_efficiencies()

    plot_regression_efficiencies()
    difference_plot()
    difference_plot_best()
    difference_plot_best_difference()
    difference_plot_best_difference_error()
    difference_plot_best_difference_snr()
    plot_regression_scores()