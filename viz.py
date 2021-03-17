import streamlit as st
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import bokeh
from bokeh.plotting import figure
from bokeh.palettes import Category10
from bokeh.models import ColorBar, ColumnDataSource
from bokeh.transform import linear_cmap, log_cmap
from colour import Color
from collections import defaultdict

class dotted:
    def __init__(self, d):
        self._d = d
    def __getattr__(self, item):
        return self._d[item]

path = st.text_input("Folder", value='remote_out/')
files = sorted([join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'pk'])

attr_lookup = {}
all_attrs = set([])
for file in files:
    attr_lookup[file] = defaultdict(lambda :None)
    id = '.'.join([x for x in file.split("/")[-1][:-3].split(".") if "seed" not in x])
    attr_lookup[file]['id'] = id
    attrs = id.split('.')
    attr_lookup[file]['train'] = attrs[-1] == 'train'
    attr_lookup[file]['test'] = attrs[-1] == 'test'
    for attr, val in [a.split("_") for a in attrs[:-1]]:
        try: val = int(val)
        except:
            try: val = float(val)
            except: pass
        attr_lookup[file][attr] = val
    for attr in attr_lookup[file].keys(): all_attrs.add(attr)

run_criteria = st.text_input(f"Filter expression (available attrs: {', '.join(all_attrs)})")
filter_expression = run_criteria if run_criteria != "" else "False"
runs_to_plot = [file for file in files if eval(filter_expression, {}, attr_lookup[file])]
runs_to_plot.sort(key=lambda x: (len(attr_lookup[x]),'|'.join(sorted(attr_lookup[x].keys())),) +
                                 tuple([attr_lookup[x][attr] for attr in sorted(all_attrs) if attr in attr_lookup[x] and attr != 'seed']))

## construct sets of seeds
run_sets_to_plot = []
for run in runs_to_plot:
    home_found = False
    for common_attrs, runs_list in run_sets_to_plot:
        if len({attr for attr in common_attrs if attr != 'seed' and common_attrs[attr] != attr_lookup[run][attr]}) == 0:
            runs_list.append(run)
            home_found = True
            break
    if not home_found:
        run_sets_to_plot.append([attr_lookup[run], [run]])
run_sets_to_plot = [runs_list for common_attrs, runs_list in run_sets_to_plot]

@st.cache
def load_data(filepath):
    data = pd.read_pickle(filepath)
    steps = data['step']
    del data['step']
    return steps, data

all_cols = sorted({col for runs_to_plot in run_sets_to_plot for filepath in runs_to_plot for col in load_data(filepath)[1].columns})

mode = st.radio("Plot mode:", ["Training curves", "Heatmap"])

st.markdown("---")

if mode == "Training curves":

    if st.checkbox("Plot multiple y series"):
        cols_to_plot = st.multiselect("Columns to plot", all_cols, default=all_cols)
    else:
        cols_to_plot = [st.selectbox("Columns to plot", all_cols)]
    log_axis = st.checkbox("Log scale y-axis")
    show_indiv_runs = st.checkbox("Show individual runs")
    st.markdown("---")

    p = figure(x_axis_label='steps', y_axis_type="log" if log_axis else "linear")

    colors = Category10[max(len(run_sets_to_plot), 3)]
    styles = ['solid', 'dashed', 'dotted', 'dotdash']
    assert len(run_sets_to_plot) <= len(colors) and len(cols_to_plot) <= len(styles)

    for i, runs_list in enumerate(run_sets_to_plot):
        if len(runs_list) > 1:
            max_len = -1
            for filepath in runs_list:
                steps, serieses = load_data(filepath)
                max_len = max(max_len, steps.shape[0])
                if show_indiv_runs:
                    for j, col in enumerate(sorted({col for col in serieses.columns if col in cols_to_plot})):
                        p.line(steps, serieses[col],
                               line_width=1,
                               color=colors[i],
                               line_dash=styles[j],
                               alpha=.5)

            def pad_to_max_len(x):
                c = np.empty(max_len)
                c.fill(np.nan)
                c[:x.shape[0]] = x
                return c
            steps = np.nanmean([pad_to_max_len(load_data(filepath)[0]) for filepath in runs_list], 0)
            serieses = {col: np.nanmean([pad_to_max_len(load_data(filepath)[1][col]) for filepath in runs_list], 0) for col in cols_to_plot}
            columns = list(serieses.keys())
        else:
            filepath = runs_list[0]
            steps, serieses = load_data(filepath)
            columns = serieses.columns

        title = attr_lookup[runs_list[0]]['id']
        for j, col in enumerate(sorted({col for col in columns if col in cols_to_plot})):
            p.line(steps, serieses[col],
                   legend=title + ' ' + str(col).lower(),
                   line_width=2,
                   color=colors[i],
                   line_dash=styles[j])

    p.yaxis.formatter = bokeh.models.BasicTickFormatter(use_scientific=False)


    st.bokeh_chart(p, use_container_width=True)

elif mode == "Heatmap":
    X1_formula = st.text_input(f"X1 formula")
    X1_label = st.text_input(f"X1 label")
    X1_log = st.checkbox("Log scale X1")
    X2_formula = st.text_input(f"X2 formula")
    X2_label = st.text_input(f"X2 label")
    X2_log = st.checkbox("Log scale X2")
    Y = st.selectbox("Column to plot", all_cols)
    Y_as_perc = st.checkbox("Write as perc")
    Y_log = st.checkbox("Log scale colors")

    st.markdown("---")

    p = figure(x_axis_label=X1_label, x_axis_type="log" if X1_log else "linear",
               y_axis_label=X2_label, y_axis_type="log" if X2_log else "linear",
               )

    points_to_plot = []
    for runs_list in run_sets_to_plot:
        x1 = eval(X1_formula, {}, attr_lookup[runs_list[0]])
        x2 = eval(X2_formula, {}, attr_lookup[runs_list[0]])
        y = np.mean([load_data(filepath)[1][Y].iloc[-1] for filepath in runs_list
                     if len(load_data(filepath)[1][Y]) > 0])
        points_to_plot.append((x1,x2,y))

    source = ColumnDataSource(dict(X1=[p[0] for p in points_to_plot],
                               X2=[p[1] for p in points_to_plot],
                               Y=[p[2] for p in points_to_plot],
                               label= [f"{100*p[2]:.1f}%" for p in points_to_plot] if Y_as_perc else
                                      [f"{p[2]:.2E}" for p in points_to_plot]))

    cmap = log_cmap if Y_log else linear_cmap
    color_mapper = cmap(palette=tuple(c.get_hex_l() for c in Color("green").range_to(Color("red"),256)),
                        low=min([p[2] for p in points_to_plot]),
                        high=max([p[2] for p in points_to_plot]),
                        field_name="Y")

    p.text(x='X1', y='X2', text='label', text_align='center', text_font_size={'value': '12px'}, color=color_mapper, text_baseline='bottom', source=source)
    p.circle(x='X1', y='X2', color='black', fill_alpha=1., size=3, source=source)

    # p.circle(x='X1', y='X2', color=color_mapper, line_alpha=0.0, fill_alpha=.3, size=75, source=source)
    # color_bar = ColorBar(title=Y, color_mapper=color_mapper['transform'], margin=-10)
    # p.add_layout(color_bar, 'right')


    st.bokeh_chart(p, use_container_width=True)