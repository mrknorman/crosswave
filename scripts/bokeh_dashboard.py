import bokeh.plotting as bk
import pandas as pd

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Select, CustomJS
from bokeh.plotting import figure

import pandas as pd
import numpy as np

input_file_name = "skywarp_validation_scores_regression_best"
output_file_name = "skywarp_validation_scores_regression_best"

# Load dataframe:
df = pd.read_pickle(f"{input_file_name}.pkl")

true_cols = [col for col in df.columns if not col.endswith("_pred") and col + "_pred" in df.columns]

source = ColumnDataSource(data=dict(x=df[true_cols[0]].tolist(), y=df[true_cols[0] + "_pred"].tolist()))

select = Select(title="Choose Parameter:", value=true_cols[0], options=true_cols)

p = figure(height=1000, width=1000, title="True vs. Predicted Values",
           x_axis_label="True Value", y_axis_label="Predicted Value")
p.scatter("x", "y", source=source)

# Convert the DataFrame to a dictionary
df_dict = df.to_dict(orient='list')

# JavaScript code for updating the data source when the selected parameter changes
callback = CustomJS(args=dict(source=source, df=df_dict), code="""
    var true_col = cb_obj.value;
    var pred_col = true_col + "_pred";
    var new_data = {
        x: df[true_col],
        y: df[pred_col]
    };
    source.data = new_data;
    source.change.emit();
""")

select.js_on_change("value", callback)

layout = column(select, p)

bk.output_file("bokeh_dashboard.html")
bk.save(layout)