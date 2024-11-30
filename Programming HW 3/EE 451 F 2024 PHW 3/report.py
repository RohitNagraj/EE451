import plotly.graph_objs as go
import plotly.offline as pyo

# Data from the three tables
threads = [1, 4, 16, 64, 256]

time1 = [1366.138831, 375.451539, 394.123210, 390.727256, 386.786432]
time2 = [319.371676, 101.459692, 95.469828, 101.700691, 90.865101]
time3 = [277.867787, 86.697244, 91.441576, 91.454943, 77.599605]

# Creating traces for each table
trace1 = go.Scatter(x=threads, y=time1, mode='lines+markers', name='Table 1')
trace2 = go.Scatter(x=threads, y=time2, mode='lines+markers', name='Table 2')
trace3 = go.Scatter(x=threads, y=time3, mode='lines+markers', name='Table 3')

# Combine all traces into a data list
data = [trace1, trace2, trace3]

# Setting up the layout
layout = go.Layout(
    title='Execution Time vs No. of Threads',
    xaxis=dict(title='Number of Threads'),
    yaxis=dict(title='Execution Time (sec)'),
    legend=dict(title='Tables')
)

# Creating the figure
fig = go.Figure(data=data, layout=layout)

# Plotting the figure
pyo.plot(fig, filename='execution_time_vs_threads.html')