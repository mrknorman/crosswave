import bokeh.plotting as bk
import pandas as pd

from bokeh.models import ColumnDataSource, Select, CustomJS, LinearColorMapper, ColorBar
from bokeh.plotting import figure
from bokeh.io import show
from bokeh import palettes
from bokeh.layouts import column

from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

input_file_name = "./validation_results/crosswave_validation_scores_regression_best"
output_file_name = "./validation_results/crosswave_validation_scores_regression_best"

# Load dataframe:
df = pd.read_pickle(f"{input_file_name}.pkl")

# Update the DataFrame according to the specified rules
for col in df.columns:
    if 'time' in col:
        if col.endswith("_pred"):  # If the column is a prediction
            df[col] = df[col] * 4
        else:  # If the column is ground truth
            df[col] = df[col] - 60
    if 'mass' in col:
        if col.endswith("_pred"):  # If the column is a prediction
            df[col] = df[col] * 100
    if 'distance' in col:
        if col.endswith("_pred"):  # If the column is a prediction
            df[col] = df[col] * 100 + 500
    if 'spin' in col:
        if col.endswith("_pred"):  # If the column is a prediction
            df[col] = df[col] * 2 - 1

# Filter columns for the scatter plot
true_cols = [col for col in df.columns if not col.endswith("_pred") and col + "_pred" in df.columns]

# Filter columns for the scatter plot
true_cols = [col for col in df.columns if not col.endswith("_pred") and col + "_pred" in df.columns]

# Calculate the difference and add it as a new column
for col in true_cols:
    df[f"{col}_diff"] = abs(df[col] - df[f"{col}_pred"])

# Create a list to hold all the plots
plots = []

r_values = []
mae_values = []
# Create a separate plot for each parameter
for col in true_cols:
    if "time" in col:
        df = df[df[col] > 0]
    
    source = ColumnDataSource(data=dict(
        x=df[col].tolist(),
        y=df[f"{col}_pred"].tolist(),
        diff=df[f"{col}_diff"].tolist()
    ))
    

    mapper = LinearColorMapper(palette=palettes.Plasma[11], low=min(df[f"{col}_diff"]), high=max(df[f"{col}_diff"]))
    
    # Calculate R-squared for each parameter
    r_values.append(r2_score(df[col], df[f"{col}_pred"]))
    mae_values.append(mean_absolute_error(df[col], df[f"{col}_pred"]))
    
    p = figure(height=600, width=675,  title=f"True vs. Predicted Values: {col} (R²={r_values[-1]:.2f})", # Doubled the height
               x_axis_label="True Value", y_axis_label="Predicted Value")
    p.scatter("x", "y", source=source, color={'field': 'diff', 'transform': mapper}, line_color=None)
    
    # Add a dotted line at x=y
    min_val = min(min(df[col]), min(df[f"{col}"]))
    max_val = max(max(df[col]), max(df[f"{col}"]))
    p.line([min_val, max_val], [min_val, max_val], line_dash="dotted", line_width=2, color="silver")
    
    # Create a color bar for each plot
    color_bar = ColorBar(color_mapper=mapper, width=8, location=(0,0), title="Prediction Error")
    p.add_layout(color_bar, 'right')

    plots.append(p)
    
results_df = pd.DataFrame({
    'Column': true_cols,
    'R²': r_values,
    'MAE': mae_values
})
print(results_df)

# Stack the plots vertically
layout = column(*plots)

# Output file and save the layout
bk.output_file("bokeh_dashboard_stacked.html")
bk.save(layout)