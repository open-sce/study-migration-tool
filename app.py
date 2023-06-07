import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import html, dcc, callback, no_update
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template

from config import AppConfiguration
from logger import logger
from src import data_processing

load_figure_template("sandstone")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE, dbc.icons.BOOTSTRAP])
server = app.server

app.title = "Study Migration Tool"
version = "1.1.0"  # Release . Feature . Bugfix

app_config = AppConfiguration()
app_data = data_processing.Data(app_config)


# COMPONENTS -----------------------------------------------------------------------------------------------------------

def no_data_figure():
    return {
        "layout": {
            "xaxis": {"visible": False}, "yaxis": {"visible": False},
            "annotations": [{
                "text": 'No Data',
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 28}
            }],
            "margin": {
                "l": 5, "r": 5, "t": 5, "b": 5,
            }
        }
    }


def one_or_more_figure(selection_name: str):
    return {
        "layout": {
            "xaxis": {"visible": False}, "yaxis": {"visible": False},
            "annotations": [{
                "text": f'Please Select One or More {selection_name}',
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 28}
            }],
            "margin": {
                "l": 5, "r": 5, "t": 5, "b": 5,
            }
        }
    }


# WIDGETS --------------------------------------------------------------------------------------------------------------

navbar = dbc.Navbar([
    html.A(
        dbc.Row([
            dbc.Col(html.Img(src="/assets/logo.png", height="30px")),
            dbc.Col(dbc.NavbarBrand("Migration Tool", className="ml-2"))
        ], className="g-0", align="center"),
        href="https://www.achieveintelligence.com/"
    ),
    dbc.NavItem(
        dbc.NavLink("General Overview", href="/#general-overview-section", className="bg-secondary",
                    external_link=True)
    ),
    dbc.NavItem(
        dbc.NavLink("Study Specific", href="/#study-specific-section", className="bg-secondary",
                    external_link=True),
    ),
    dbc.NavItem(
        dbc.NavLink("Compound Specific", href="/#compound-specific-section", className="bg-secondary",
                    external_link=True)
    ),
    dbc.Row(children=[
        dbc.Col(dbc.Badge(version))
    ], style={"marginLeft": "auto"})
], sticky="top", color="primary", dark=True, style={"height": "4rem", "flexWrap": "nowrap", "marginBottom": "1rem",
                                                    "gap": "10px"})

date_picker = html.Div([
    dbc.Label("Timeframe", className="title-label"),
    dcc.DatePickerRange(
        id='sidebar-date-picker',
        min_date_allowed=datetime.date(1960, 1, 1),
        max_date_allowed=datetime.date(2200, 1, 1),
        initial_visible_month=datetime.date.today(),
        start_date=app_config.timeframe_start,
        end_date=app_config.timeframe_end,
        display_format="DD-MMM-YY"
    )
])

outside_timeframe_filter = html.Div([
    dbc.Label("Hide Milestones Outside Timeframe?", className="title-label"),
    dbc.RadioItems(
        id="timeframe-filter",
        options=[
            {"label": "Yes", "value": True},
            {"label": "No", "value": False},
        ],
        value=True, inline=True
    )
])

offset_widget = html.Div([
    dbc.Label("Milestone Offsets", className="title-label"),
    dbc.Row([
        dbc.Col([
            html.Span(className="sidebar-offset-header", children=["Offset (days)"])
        ], width={"size": 5, "offset": 0}),
        dbc.Col([], width={"size": 1, "offset": 0}),
        dbc.Col([
            html.Span(className="sidebar-offset-header", children=["Before"])
        ], width={"size": 3, "offset": 0}),
        dbc.Col([
            html.Span(className="sidebar-offset-header", children=["After"])
        ], width={"size": 3, "offset": 0})
    ]),
    html.Div(className="milestone-container", children=[
        dbc.Row(children=[
            dbc.Col([
                html.Span(label)
            ], className="my-auto", width={"size": 5, "offset": 0}),
            dbc.Col([
                dbc.Checkbox(id={"type": "offset-checkbox", "index": label}, value=milestone.active)
            ], className="my-auto", width={"size": 1, "offset": 0}),
            dbc.Col([
                dbc.Input(id={"type": "offset-before-input", "index": label}, type="number",
                          value=milestone.offset_before)
            ], className="my-auto", width={"size": 3, "offset": 0}),
            dbc.Col([
                dbc.Input(id={"type": "offset-after-input", "index": label}, type="number",
                          value=milestone.offset_after)
            ], className="my-auto", width={"size": 3, "offset": 0})
        ]) for label, milestone in app_config.milestone_definitions.items()
    ])
])

study_graph_filters = html.Div(className="filter-box", children=[
    html.Div(className="search-input", children=[
        html.Div([
            dbc.Label("Search 'Studies'", html_for="study-search-field", className="title-label"),
            html.P("Search the data for studies that you wish to display."),
            dbc.Row([
                dbc.Col([
                    dbc.Input(id="study-search-field"),
                ]),
                dbc.Col([
                    dbc.Button("Search", id="study-search-button"),
                ])
            ]),
        ]),
    ]),
    html.Div(id="study-search-output", className="search-output", children=[
        dcc.Loading(parent_className="loading-wrapper", children=[
            html.Div(id="study-search-output-formgroup", children=[
                dbc.Checklist(id="study-checklist-found", className="checklist-found")
            ])
        ])
    ])
])

compound_graph_filters = html.Div(className="filter-box", children=[
    html.Div(className="search-input", children=[
        html.Div([
            dbc.Label("Search 'Compounds'", html_for="compound-search-field", className="title-label"),
            html.P("Search the data for compounds that you wish to display."),
            dbc.Row([
                dbc.Col([
                    dbc.Input(id="compound-search-field")
                ]),
                dbc.Col([
                    dbc.Button("Search", id="compound-search-button")
                ])
            ]),
        ])
    ]),
    html.Div(id="compound-search-output", className="search-output", children=[
        dcc.Loading(parent_className="loading-wrapper", children=[
            html.Div(id="compound-search-output-formgroup", children=[
                dbc.Checklist(id="compound-checklist-found", className="checklist-found")
            ])
        ])
    ])
])

# APP LAYOUT -----------------------------------------------------------------------------------------------------------
app.layout = html.Div([
    navbar,
    dbc.Container(className="content", fluid=True, children=[
        html.Div(className="content-left", children=[
            html.Div(id="sidebar", children=[
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(children=[
                            dbc.Row([
                                dbc.Col(children=[
                                    html.Span("Global Filters")
                                ], width={"size": 10, "offset": 0}),
                                dbc.Col(children=[
                                    dbc.Button(id="collapse-sidebar",
                                               children=[html.I(className="bi bi-arrow-bar-left")])
                                ], width={"size": 2, "offset": 0})
                            ])
                        ], style={"fontWeight": "bold", "fontSize": 26}),
                        dbc.CardBody(className="sidebar-contents", children=[
                            date_picker,
                            html.Hr(),
                            offset_widget,
                            html.Div(style={"display": "flex", "flexDirection": "row"}, children=[
                                dbc.Button("refresh graphs", id="refresh-graphs-button"),
                                dbc.Spinner(children=[
                                    html.Div(id="button-spinner-div", style={"marginLeft": 50})
                                ])
                            ]),
                            html.Hr(),
                            outside_timeframe_filter
                        ])
                    ], color="light")
                ], className='dash-bootstrap')
            ]),
            html.Div(className="bg-light", id="collapsed-sidebar", children=[
                dbc.Button(id="open-sidebar", children=[html.I(className="bi bi-arrow-bar-right")])
            ], style={"display": "none"})
        ]),
        html.Div(className="content-right", id="content-right", children=[
            html.Section([
                html.A(id="general-overview-section"),
                html.Div(className="graph-container", id="study-graph-container", children=[
                    dbc.Spinner(children=[
                        dcc.Graph(
                            id="study-graph", responsive=True,
                            figure={
                                "layout": {
                                    "xaxis": {"visible": False}, "yaxis": {"visible": False},
                                    "annotations": [{
                                        "text": 'No Data',
                                        "xref": "paper",
                                        "yref": "paper",
                                        "showarrow": False,
                                        "font": {"size": 28}
                                    }],
                                    "margin": {
                                        "l": 5, "r": 5, "t": 5, "b": 5,
                                    }
                                }
                            }
                        )
                    ])
                ])
            ]),
            html.Section(className="section 2", children=[
                html.A(id="study-specific-section"),
                html.H2("Study Specific"),
                html.Div(className="graph-options", children=[
                    study_graph_filters
                ]),
                dbc.Spinner(children=[
                    dcc.Graph(
                        id="filter-study-graph", responsive=True,
                        figure=one_or_more_figure("Studies")
                    )
                ])
            ]),
            html.Section(className="section-3", children=[
                html.A(id="compound-specific-section"),
                html.H2("Compound Specific"),
                html.Div(className="graph-options", children=[
                    compound_graph_filters
                ]),
                dbc.Spinner(children=[
                    dcc.Graph(
                        id="filter-compound-graph", responsive=True,
                        figure=one_or_more_figure("Compounds")
                    )
                ])
            ])
        ])
    ]),

    dcc.Store(id="app-session-store", storage_type='session', data={
        'timeframe_start': app_config.timeframe_start.isoformat(),
        'timeframe_end': app_config.timeframe_end.isoformat(),
        'milestones': {
            label: milestone.__dict__ for label, milestone in app_config.milestone_definitions.items()
        },
        'first_start': True
    }),
    dcc.Store(id="prev-study-selected-store", storage_type='session'),
    dcc.Store(id="prev-compound-selected-store", storage_type='session'),
    dcc.Store(id="data-plot-df-store", storage_type='session'),
    dcc.Store(id="app-mode-store", data={"test_mode": False, "test_data": None})
])


# APP CALLBACKS --------------------------------------------------------------------------------------------------------


@app.callback(
    Output("sidebar", "style"),
    Output("collapsed-sidebar", "style"),
    Input("collapse-sidebar", "n_clicks"),
    Input("open-sidebar", "n_clicks"),
    prevent_initial_call=True
)
def collapse_sidebar(collapse_clicks, open_clicks):
    ctx = dash.callback_context
    activation_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if activation_id == 'collapse-sidebar' and collapse_clicks is not None:
        return {"display": "none"}, {"display": "block"}
    elif activation_id == 'open-sidebar' and open_clicks is not None:
        return {"display": "block"}, {"display": "none"}

    raise PreventUpdate


# Study Graph Callback
@app.callback(
    Output("study-graph", "figure"),
    Input("data-plot-df-store", "data"),
    Input("timeframe-filter", "value"),
    prevent_initial_call=True
)
def set_study_graph_figure(frozen, timeframe_filter):
    df = pd.read_json(frozen)
    fig = no_data_figure()

    if not df.empty:
        df["start"] = df["start"].apply(lambda x: pd.Timestamp(x, unit="ms"))
        df["end"] = df["end"].apply(lambda x: pd.Timestamp(x, unit="ms"))

        if timeframe_filter:
            df = df.loc[df["inside timeframe"] == "Yes"]

        fig = px.timeline(df, x_start="start", x_end="end", y=app_data.unique_identity_label,
                          color="type", opacity=0.8, range_x=[df["start"].min(), df["end"].max()])
        fig.update_layout(font={"size": 14}, xaxis_title="Time Period")
        fig.update_xaxes({"side": "top"})
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))

    return fig


# Filtered Study Graph Callback
@callback(Output("filter-study-graph", "figure"),
          Output("prev-study-selected-store", "data"),
          Input("study-checklist-found", "value"),
          Input("data-plot-df-store", "data"),
          Input("timeframe-filter", "value"),
          State("prev-study-selected-store", "data"),
          prevent_initial_call=True)
def set_filter_study_graph_figure(study_lst, frozen, timeframe_bool, prev_selected):
    if study_lst is None:
        # Catch initialisation cases
        raise PreventUpdate

    if not study_lst:
        return one_or_more_figure("Studies"), no_update

    if prev_selected == study_lst:
        raise PreventUpdate

    fig = no_data_figure()
    df = pd.read_json(frozen)

    df["start"] = df["start"].apply(lambda x: pd.Timestamp(x, unit="ms"))
    df["end"] = df["end"].apply(lambda x: pd.Timestamp(x, unit="ms"))

    df = df.loc[df[app_data.study_label].isin(study_lst)]

    if timeframe_bool:
        df = df.loc[df["inside timeframe"] == "Yes"]

    if not df.empty:
        fig = px.timeline(df, x_start="start", x_end="end", y=app_data.unique_identity_label,
                          opacity=0.8, color="type", range_x=[df["start"].min(), df["end"].max()])
        fig.update_layout(font={"size": 14}, xaxis_title="Time Period")
        fig.update_xaxes({"side": "top"})
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))

    return fig, study_lst


# Filtered Compound Graph Callback
@callback(Output("filter-compound-graph", "figure"),
          Output("prev-compound-selected-store", "data"),
          Input("compound-checklist-found", "value"),
          Input("data-plot-df-store", "data"),
          Input("timeframe-filter", "value"),
          State("prev-compound-selected-store", "data"),
          prevent_initial_call=True)
def set_filter_compound_graph_figure(compound_lst, frozen, timeframe_bool, prev_selected):
    if compound_lst is None:
        # Catch initialisation cases
        raise PreventUpdate

    if not compound_lst:
        return one_or_more_figure("Compounds"), no_update

    if prev_selected == compound_lst:
        raise PreventUpdate

    fig = no_data_figure()
    df = pd.read_json(frozen)

    df["start"] = df["start"].apply(lambda x: pd.Timestamp(x, unit="ms"))
    df["end"] = df["end"].apply(lambda x: pd.Timestamp(x, unit="ms"))

    df = df.loc[df[app_data.compound_label].isin(compound_lst)]

    if timeframe_bool:
        df = df.loc[df["inside timeframe"] == "Yes"]

    if not df.empty:
        fig = px.timeline(df, x_start="start", x_end="end", y=app_data.unique_identity_label,
                          opacity=0.8, color="type", range_x=[df["start"].min(), df["end"].max()])
        fig.update_layout(font={"size": 14},
                          xaxis_title="Time Period")
        fig.update_xaxes({"side": "top"})
        fig.update_layout(
            margin=dict(l=5, r=5, t=5, b=5),
        )

    return fig, compound_lst


# Update the data in the dcc.store. Including converting the data object to json and back
@callback(
    {
        "app_session_store": Output("app-session-store", "data"),
        "plot_df": Output("data-plot-df-store", "data"),
        "spinner_div": Output("button-spinner-div", "style")
    },
    Input("refresh-graphs-button", "n_clicks"),
    State("app-session-store", "data"),
    State("sidebar-date-picker", "start_date"),
    State("sidebar-date-picker", "end_date"),
    State({"type": "offset-checkbox", "index": ALL}, "value"),
    State({"type": "offset-before-input", "index": ALL}, "value"),
    State({"type": "offset-after-input", "index": ALL}, "value"),
    State("app-mode-store", "data"),
    prevent_inital_call=True
)
def update_graphs_and_stores(n_clicks, app_session_store, start_date_input, end_date_input,
                             checkbox_values, offset_before_values, offset_after_values, app_mode_store):

    start_date = pd.Timestamp(start_date_input)
    end_date = pd.Timestamp(end_date_input)

    if app_mode_store.get('test_mode'):
        app_data = app_mode_store['test_data']
    else:
        app_data = globals().get('app_data')

    if app_session_store['first_start']:
        app_data.create_timeblock(app_session_store)
        app_data.create_plotting_df(app_session_store)
        app_session_store['first_start'] = False
        return {
            "app_session_store": app_session_store,
            "plot_df": app_data.plot_df.to_json(),
            "spinner_div": {"marginLeft": 50}
        }
    else:
        app_session_store['timeframe_start'] = start_date.isoformat()
        app_session_store['timeframe_end'] = end_date.isoformat()

        active_milestone_list = []

        for milestone_label, active, offset_before, offset_after in zip(app_session_store['milestones'].keys(),
                                                                        checkbox_values, offset_before_values,
                                                                        offset_after_values):

            app_session_store['milestones'][milestone_label]['offset_before'] = offset_before
            app_session_store['milestones'][milestone_label]['offset_after'] = offset_after
            app_session_store['milestones'][milestone_label]['active'] = active
            active_milestone_list.append(active)

        if any(active_milestone_list):
            app_data.create_timeblock(app_session_store)
            app_data.create_plotting_df(app_session_store)
            return {
                "app_session_store": app_session_store,
                "plot_df": app_data.plot_df.to_json(),
                "spinner_div": {"marginLeft": 50}
            }
        else:
            logger.info(f"STORE UPDATE: no active milestones")
            return {
                "app_session_store": app_session_store,
                "plot_df": pd.DataFrame().to_json(),
                "spinner_div": {"marginLeft": 50}
            }


# Study Search and filter callbacks
@callback(Output("study-search-output-formgroup", "children"),
          Input('study-search-button', "n_clicks"),
          State("study-search-field", "value"),
          State("study-checklist-found", "value"),
          State("app-mode-store", "data"),
          prevent_initial_call=True)
def return_study_checklist_options(n_clicks, search_value, selected_items, app_mode_store):
    if n_clicks is None or search_value is None or search_value == '':
        raise PreventUpdate

    if app_mode_store.get('test_mode'):
        df = app_mode_store['test_data'].df
    else:
        df = app_data.df

    if search_value.isalnum():
        results = df.loc[df[app_data.study_label].astype(str).str.contains(search_value, case=False)]
        unique_studies = results[app_data.study_label].unique()
        unique_studies = set(selected_items + list(unique_studies)) if selected_items else unique_studies

        options_dict = [{'label': item, 'value': item} for item in sorted(unique_studies)]
        return dbc.Checklist(
            id="study-checklist-found", className="checklist-found", options=options_dict, inline=True,
            labelClassName="study-checklist-found-label", value=selected_items if selected_items else []
        )
    else:
        return dbc.Checklist(id="study-checklist-found", className="checklist-found")


# Compound Search and filter callbacks
@callback(
    Output("compound-search-output-formgroup", "children"),
    Input("compound-search-button", "n_clicks"),
    State("compound-search-field", "value"),
    State("compound-checklist-found", "value"),
    State("app-mode-store", "data"),
    prevent_initial_call=True
)
def return_compound_checklist_options(n_clicks, search_value, selected_items, app_mode_store):
    if n_clicks is None or search_value is None or search_value == "":
        raise PreventUpdate

    if app_mode_store.get('test_mode'):
        df = app_mode_store['test_data'].df
    else:
        df = app_data.df

    if search_value.isalnum():
        results = df.loc[df[app_data.compound_label].astype(str).str.contains(search_value, case=False)]
        unique_compounds = results[app_data.compound_label].unique()
        unique_compounds = set(selected_items + list(unique_compounds)) if selected_items else unique_compounds

        options_dict = [{'label': item, 'value': item} for item in unique_compounds]
        return dbc.Checklist(
            id="compound-checklist-found", className="checklist-found", options=options_dict, inline=True,
            labelClassName="compound-checklist-found-label", value=selected_items if selected_items else []
        )
    else:
        return dbc.Checklist(id="compound-checklist-found", className="checklist-found")


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run()
