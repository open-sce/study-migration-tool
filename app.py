import datetime
import math
import time
import uuid
from typing import List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import html, dcc, callback, no_update, dash_table
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template

from config import AppConfiguration
from logger import logger
from src import data_processing

load_figure_template("sandstone")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE, dbc.icons.BOOTSTRAP])
server = app.server

app.title = "Study Migration Tool"
version = "2.0.1"  # Release . Feature . Bugfix

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


def custom_filter(identifier: str):
    return html.Div([
        html.Div(className="filter-box", children=[
            dcc.Dropdown(
                options=[col for col in app_data.df.columns if col not in app_config.milestone_definitions.keys()],
                id={'type': 'column-label', 'index': identifier}, placeholder='Select column filter', multi=False,
                style={'flexGrow': 1}
            ),
            dbc.Button('Remove Filter', color='danger', id={'type': 'remove-filter-button', 'index': identifier})
        ]),
        dcc.Dropdown(id={'type': 'column-values', 'index': identifier}, style={"marginTop": 10},
                     options=[], multi=True, placeholder='Select column value(s)')
    ], id=identifier)


def generate_period_lengths_section(period_start_end: List[tuple]):
    return html.Div(className="period-info", children=[
        html.Div(className="period-box", children=[
            html.H4(f"Period {period_number}"),
            html.Span(f"Start: {period[0]}"),
            html.Hr(style={"marginTop": 5, "marginBottom": 5}),
            html.Span(f"End: {period[1]}"),
        ]) for period_number, period in enumerate(period_start_end, start=1)
    ])


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
    dbc.NavItem(
        dbc.NavLink("Migration Table", href="/#migration-section", className="bg-secondary",
                    external_link=True)
    ),
    dbc.Row(children=[
        dbc.Col(dbc.Badge(version))
    ], style={"marginLeft": "auto"})
], sticky="top", color="primary", dark=True, style={"height": "4rem", "flexWrap": "nowrap", "marginBottom": "1rem",
                                                    "gap": "10px"})

date_picker = html.Div([
    dbc.Label("Timeframe", className="title-label", style={'display': 'block'}),
    dcc.DatePickerRange(
        id='sidebar-date-picker',
        min_date_allowed=datetime.date(1960, 1, 1),
        max_date_allowed=datetime.date(2200, 1, 1),
        initial_visible_month=datetime.date.today(),
        start_date=app_config.timeframe_start,
        end_date=app_config.timeframe_end,
        display_format="DD-MMM-YY"
    )
], style={"marginBottom": 10})

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
                          value=milestone.offset_before, min=0)
            ], className="my-auto", width={"size": 3, "offset": 0}),
            dbc.Col([
                dbc.Input(id={"type": "offset-after-input", "index": label}, type="number",
                          value=milestone.offset_after, min=0)
            ], className="my-auto", width={"size": 3, "offset": 0})
        ]) for label, milestone in app_config.milestone_definitions.items()
    ])
], style={"marginBottom": 10})

custom_filter_widget = html.Div([
    html.Div(className="filter-box", style={"justifyContent": "space-between"}, children=[
        dbc.Label("Custom Filters", className="title-label"),
        dbc.Button('Add new filter', color='success', id='add-custom-filter-button'),
    ]),
    html.Div(id='custom-filter-wrapper', children=[])
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

migration_options = html.Div(children=[
    dbc.Card(className="migration-card", children=[
        dbc.CardHeader([
            html.Span('Migration Table Configuration', style={'fontSize': 18, 'fontWeight': 'bold'})
        ]),
        dbc.CardBody([
            dbc.Accordion(start_collapsed=True, children=[
                dbc.AccordionItem(title='Study transfer calculator', children=[
                    html.P("Use the inputs to calculate how many studies you are able to transfer. The resulting "
                           "number will be displayed (and editable) below."),
                    html.Div(className="filter-box", style={"flexWrap": "wrap"}, children=[
                        html.Div([
                            dbc.Label('Working hours in a day'),
                            dbc.Input(id='migration-day-length', type='number', value=2, min=0.5, step=0.5, max=24),
                        ]),
                        html.Div([
                            dbc.Label('Working days in a week'),
                            dbc.Input(id='migration-days-in-week', type='number', value=5, min=0.5, max=7, step=0.5,
                                      disabled=True)
                        ]),
                        html.Div([
                            dbc.Label('Average size of study'),
                            html.Div(className="select-with-unit", children=[
                                dbc.Input(id='migration-average-study-size', type='number', value=1, min=0.5, step=0.5,
                                          max=1000),
                                dbc.Select(["KB", "MB", "GB", "TB"], id="migration-study-size-unit", value="GB"),
                            ])
                        ]),
                        html.Div([
                            dbc.Label('Average transfer rate'),
                            html.Div(className="select-with-unit", children=[
                                dbc.Input(id='migration-transfer-rate', type='number', value=5, min=0.5, step=0.5,
                                          max=1000),
                                dbc.Select(["KB/s", "MB/s", "GB/s"], id="migration-transfer-rate-unit", value="MB/s"),
                            ])
                        ])
                    ])
                ])
            ]),
            html.Div(className="filter-box", style={"flexWrap": "wrap"}, children=[
                html.Div([
                    dbc.Label('Frequency'),
                    dbc.Select(["Days", "Weeks"], id="migration-frequency", value="Days"),
                ]),
                html.Div([
                    dbc.Label('Studies transferred per day', id='study-transfer-label'),
                    dbc.Input(id='migration-study-rate', type='number', value=20, min=1),
                ]),
                html.Div([
                    dbc.Label('Optional Grouping Column'),
                    dbc.Select(
                        ['Overall'] + [col for col in app_data.df.columns if
                                       col not in app_config.milestone_definitions.keys()],
                        value='Overall', id="migration-grouping-col")
                ]),
                html.Div([
                    dbc.Label('Length of period (days)', id='migration-period-length-label'),
                    dbc.Input(id='migration-period-length', type='number', value=610, min=1),
                ]),
            ])
        ]),
        dbc.CardFooter(children=[
            dcc.Loading(
                parent_className="filter-box",
                parent_style={"marginTop": 0, "justifyContent": "flex-end", "alignItems": "center"},
                children=[
                    dbc.Button("Generate Table", id="migration-generate-table-button"),
                    dbc.Button("Download migration", id="migration-download-button"),
                    dcc.Dropdown(options=['CSV'], value='CSV', id="migration-download-filetype", multi=False,
                                 searchable=False, clearable=False, style={"minWidth": 70}),
                    dcc.Download(id="migration-download"),
                    dcc.Download(id="migration-config-download")
                ]
            )
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
                                dbc.Col([
                                    html.Div(style={"display": "flex", "flexDirection": "row",
                                                    "justifyContent": "space-between", "marginBottom": 10}, children=[
                                        html.Span("Global Filters"),
                                        dbc.Button(id="collapse-sidebar", children=[
                                            html.I(className="bi bi-arrow-bar-left")
                                        ])
                                    ]),
                                    html.Div(style={"display": "flex", "flexDirection": "row"}, children=[
                                        dbc.Button("Update Global Filters", id="update-global-filters"),
                                        dbc.Spinner(children=[
                                            html.Div(id="button-spinner-div", style={"marginLeft": 50})
                                        ])
                                    ])
                                ])
                            ])
                        ], style={"fontWeight": "bold", "fontSize": 26}),
                        dbc.CardBody(className="sidebar-contents", children=[
                            date_picker,
                            outside_timeframe_filter,
                            html.Hr(),
                            offset_widget,
                            html.Hr(),
                            custom_filter_widget
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
                html.A(id="general-overview-section", style={'position': 'relative', 'top': '-70px'}),
                html.H2("General Overview", style={'paddingTop': 0}),
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
            html.Section(className="section 3", children=[
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
            ]),
            html.Section(className="section4", children=[
                html.A(id="migration-section"),
                html.H2("Migration Table"),
                html.Div(className="graph-options", children=[
                    migration_options,
                ]),
                dbc.Spinner(children=[
                    html.Div(id="period-lengths-wrapper"),
                    html.Div(id="migration-table", children=[
                        html.Div(className="no-migration-table-created", children=[
                            html.Span("Modify the configuration above and press generate table.")
                        ])
                    ])
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
        'active_filters': [],
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
    State("app-session-store", "data"),
    prevent_initial_call=True
)
def set_study_graph_figure(frozen, timeframe_filter, app_session_store):
    df = pd.read_json(frozen)
    fig = no_data_figure()

    if not df.empty:
        df["start"] = df["start"].apply(lambda x: pd.Timestamp(x, unit="ms"))
        df["end"] = df["end"].apply(lambda x: pd.Timestamp(x, unit="ms"))

        if timeframe_filter:
            df = df.loc[df["inside timeframe"] == "Yes"]

        if not df.empty:
            fig = px.timeline(df, x_start="start", x_end="end", y=app_data.unique_identity_label,
                              color="type", opacity=0.8, range_x=[df["start"].min(), df["end"].max()])
            fig.update_layout(font={"size": 14}, xaxis_title="Time Period")
            fig.update_xaxes({"side": "top"})
            fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
            fig.add_vline(x=pd.Timestamp(app_session_store['timeframe_start']), line_color='gray', opacity=0.6)
            fig.add_vline(x=pd.Timestamp(app_session_store['timeframe_end']), line_color='gray', opacity=0.6)

    return fig


# Filtered Study Graph Callback
@callback(Output("filter-study-graph", "figure"),
          Output("prev-study-selected-store", "data"),
          Input("study-checklist-found", "value"),
          Input("data-plot-df-store", "data"),
          Input("timeframe-filter", "value"),
          State("prev-study-selected-store", "data"),
          State("app-session-store", "data"),
          prevent_initial_call=True)
def set_filter_study_graph_figure(study_lst, frozen, timeframe_bool, prev_selected, app_session_store):
    if study_lst is None:
        # Catch initialisation cases
        raise PreventUpdate

    if not study_lst:
        return one_or_more_figure("Studies"), no_update

    activation_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if (prev_selected == study_lst) and not (activation_id == "data-plot-df-store"):
        raise PreventUpdate

    fig = no_data_figure()
    df = pd.read_json(frozen)

    if not df.empty:

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
            fig.add_vline(x=pd.Timestamp(app_session_store['timeframe_start']), line_color='gray', opacity=0.6)
            fig.add_vline(x=pd.Timestamp(app_session_store['timeframe_end']), line_color='gray', opacity=0.6)

    return fig, study_lst


# Filtered Compound Graph Callback
@callback(Output("filter-compound-graph", "figure"),
          Output("prev-compound-selected-store", "data"),
          Input("compound-checklist-found", "value"),
          Input("data-plot-df-store", "data"),
          Input("timeframe-filter", "value"),
          State("prev-compound-selected-store", "data"),
          State("app-session-store", "data"),
          prevent_initial_call=True)
def set_filter_compound_graph_figure(compound_lst, frozen, timeframe_bool, prev_selected, app_session_store):
    if compound_lst is None:
        # Catch initialisation cases
        raise PreventUpdate

    if not compound_lst:
        return one_or_more_figure("Compounds"), no_update

    activation_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if (prev_selected == compound_lst) and not (activation_id == "data-plot-df-store"):
        raise PreventUpdate

    fig = no_data_figure()
    df = pd.read_json(frozen)

    if not df.empty:

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
            fig.add_vline(x=pd.Timestamp(app_session_store['timeframe_start']), line_color='gray', opacity=0.6)
            fig.add_vline(x=pd.Timestamp(app_session_store['timeframe_end']), line_color='gray', opacity=0.6)

    return fig, compound_lst


# Migration Table Callback
@callback(Output("migration-table", "children"),
          Output("period-lengths-wrapper", "children"),
          Input("migration-generate-table-button", "n_clicks"),
          State("app-session-store", "data"),
          State("migration-study-rate", "value"),
          State("migration-study-rate", "invalid"),
          State("migration-frequency", "value"),
          State("migration-period-length", "value"),
          State("migration-period-length", "invalid"),
          State("migration-grouping-col", "value"),
          State("app-mode-store", "data"),
          prevent_initial_call=True)
def update_migration_table(n_clicks, app_session_store, migration_study_rate, migration_study_rate_error,
                           migration_frequency, migration_period_length, migration_period_length_error, group_col,
                           app_mode_store):

    if any(i is None for i in [n_clicks, migration_study_rate, migration_period_length]):
        raise PreventUpdate

    if any([migration_period_length_error, migration_study_rate_error]):
        raise PreventUpdate

    if all(not milestone["active"] for milestone in app_session_store['milestones'].values()):
        raise PreventUpdate

    if app_mode_store.get('test_mode'):
        active_app_data = app_mode_store['test_data']
    else:
        active_app_data = app_data

    timeblock_df = active_app_data.create_timeblock(app_session_store)

    if app_session_store.get("active_filters", False):
        for [col, values] in app_session_store["active_filters"]:
            timeblock_df = timeblock_df.loc[timeblock_df[col].isin(values)]

    merged_df = active_app_data.merge_timeblocks(timeblock_df)
    gap_info_df = active_app_data.generate_gap_information(merged_df, app_session_store)
    active_app_data.compute_weights(gap_info_df)
    migration_df, period_start_end = active_app_data.migration_table_processing(
        gap_info_df, app_session_store, active_studies_per=migration_study_rate, period_length=migration_period_length,
        transfer_window_type='D' if migration_frequency == 'Days' else 'W'
    )

    page_size = 19
    table_df = active_app_data.table_formatting(migration_df, group_col)

    migration_table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in table_df.columns],
        data=table_df.to_dict('records'),
        style_table={'overflowX': 'scroll'},
        style_header={'fontWeight': 'bold', 'fontSize': 20, 'position': 'sticky', 'top': 2},
        style_cell={'padding': 7, 'fontSize': 16},
        page_size=page_size,
        sort_action="native"
        # fixed_columns={'headers': True, 'data': 1}  # breaks column width currently
    )

    period_lengths_section = html.Div([
        html.P("Note periods with no migrated studies do not appear in the migration table below.",
               style={"fontWeight": 500}),
        generate_period_lengths_section(period_start_end)
    ])

    return migration_table, period_lengths_section


# Update the data in the dcc.store. Including converting the data object to json and back
@callback(
    {
        "app_session_store": Output("app-session-store", "data"),
        "plot_df": Output("data-plot-df-store", "data"),
        "spinner_div": Output("button-spinner-div", "style")
    },
    Input("update-global-filters", "n_clicks"),
    State("sidebar-date-picker", "start_date"),
    State("sidebar-date-picker", "end_date"),
    State("app-session-store", "data"),
    State({"type": "offset-checkbox", "index": ALL}, "value"),
    State({"type": "offset-before-input", "index": ALL}, "value"),
    State({"type": "offset-after-input", "index": ALL}, "value"),
    State({"type": "column-label", "index": ALL}, "value"),
    State({"type": "column-values", "index": ALL}, "value"),
    State("app-mode-store", "data")
)
def update_graphs_and_stores(n_clicks, start_date_input, end_date_input, app_session_store,
                             checkbox_values, offset_before_values, offset_after_values, column_filters,
                             column_filter_values, app_mode_store):
    start_date = pd.Timestamp(start_date_input)
    end_date = pd.Timestamp(end_date_input)

    if app_mode_store.get('test_mode'):
        active_app_data = app_mode_store['test_data']
    else:
        active_app_data = app_data

    if app_session_store['first_start']:
        timeblock_dataframe = active_app_data.create_timeblock(app_session_store)
        plotting_dataframe = active_app_data.create_plotting_df(timeblock_dataframe, app_session_store)
        app_session_store['first_start'] = False
        return {
            "app_session_store": app_session_store,
            "plot_df": plotting_dataframe.to_json(),
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
            timeblock_dataframe = active_app_data.create_timeblock(app_session_store)
            active_filters = []
            for col, values in zip(column_filters, column_filter_values):
                if (col is not None) and values:
                    active_filters.append([col, values])
                    timeblock_dataframe = timeblock_dataframe.loc[timeblock_dataframe[col].isin(values)]

            app_session_store["active_filters"] = active_filters
            plotting_dataframe = active_app_data.create_plotting_df(timeblock_dataframe, app_session_store)

            return {
                "app_session_store": app_session_store,
                "plot_df": plotting_dataframe.to_json(),
                "spinner_div": {"marginLeft": 50}
            }
        else:
            logger.info(f"STORE UPDATE: no active milestones")
            app_session_store["active_filters"] = []
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


# Migration table inputs callback
@callback(
    Output('migration-study-rate', 'value'),
    Output('study-transfer-label', 'children'),
    Output('migration-days-in-week', 'disabled'),
    Output('migration-period-length', 'value'),
    Output('migration-period-length-label', 'children'),
    Input('migration-frequency', 'value'),
    Input('migration-day-length', 'value'),
    Input('migration-days-in-week', 'value'),
    Input('migration-average-study-size', 'value'),
    Input('migration-study-size-unit', 'value'),
    Input('migration-transfer-rate', 'value'),
    Input('migration-transfer-rate-unit', 'value'),
    State('migration-period-length', 'value'),
    State('study-transfer-label', 'children'),
    prevent_initial_call=True
)
def calculate_study_transfer_count(mode: str, day_length: int, days_in_week: int, study_size: int,
                                   study_size_unit: str, transfer_rate, transfer_rate_unit: str,
                                   migration_period_length: int, current_study_transfer_label: str):
    if any(i is None for i in [mode, day_length, days_in_week, study_size, study_size_unit, transfer_rate,
                               transfer_rate_unit, migration_period_length]):
        raise PreventUpdate

    # turn sizes into bytes
    if study_size_unit == 'KB':
        study_size = study_size * 1000
    elif study_size_unit == 'MB':
        study_size = study_size * (1000 ** 2)
    elif study_size_unit == 'GB':
        study_size = study_size * (1000 ** 3)
    else:  # TB
        study_size = study_size * (1000 ** 4)

    if transfer_rate_unit == 'KB/s':
        transfer_rate = transfer_rate * 1000
    elif transfer_rate_unit == 'MB/s':
        transfer_rate = transfer_rate * (1000 ** 2)
    else:  # GB/s
        transfer_rate = transfer_rate * (1000 ** 3)

    if mode == "Days":
        transfer_seconds = 60 * 60 * day_length
        days_per_week_disabled = True

        if current_study_transfer_label != "Studies transferred per day":
            label = "Studies transferred per day"
            migration_period_label = 'Length of period (days)'
            migration_period_length = migration_period_length * 7
        else:
            label = no_update
            migration_period_label = no_update
            migration_period_length = no_update

    else:
        transfer_seconds = 60 * 60 * day_length * days_in_week
        days_per_week_disabled = False

        if current_study_transfer_label != "Studies transferred per week":
            label = "Studies transferred per week"
            migration_period_length = migration_period_length // 7
            migration_period_label = 'Length of period (weeks)'
        else:
            label = no_update
            migration_period_length = no_update
            migration_period_label = no_update

    seconds_per_study = study_size / transfer_rate
    return math.floor(transfer_seconds / seconds_per_study), label, days_per_week_disabled, migration_period_length, \
        migration_period_label


@callback(
    Output('custom-filter-wrapper', 'children'),
    Input('add-custom-filter-button', 'n_clicks'),
    Input({'type': 'remove-filter-button', 'index': ALL}, 'n_clicks'),
    State('custom-filter-wrapper', 'children'),
    prevent_initial_call=True
)
def update_custom_filters(add_filter_n_clicks, remove_filter_n_clicks, filter_wrapper_children):
    ctx = dash.callback_context
    activation_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if activation_id == 'add-custom-filter-button':
        # adding new filter
        filter_wrapper_children.append(custom_filter(identifier=str(uuid.uuid4())))
    else:
        # removing filter
        to_remove = None
        for index_, widget_ in enumerate(filter_wrapper_children):
            remove_button_id = activation_id.split('","type"')[0].split(':"')[-1]
            if widget_['props']['id'] == remove_button_id:
                to_remove = index_
                break

        if to_remove is not None:
            del filter_wrapper_children[to_remove]
        else:
            logger.error("To remove element failed with activation id: {activation_id}!")
            raise PreventUpdate

    return filter_wrapper_children


# get unique column values callback
@callback(
    Output({'type': 'column-values', 'index': MATCH}, 'options'),
    Input({'type': 'column-label', 'index': MATCH}, 'value'),
    State("app-mode-store", "data"),
    prevent_initial_call=True
)
def get_unique_column_values(column_label, app_mode_store):
    if column_label is None:
        raise PreventUpdate

    if app_mode_store.get('test_mode'):
        df = app_mode_store['test_data'].df
    else:
        df = app_data.df

    return list(df[column_label].unique())


@callback(
    Output("migration-download", "data"),
    Output("migration-config-download", "data"),
    Input("migration-download-button", "n_clicks"),
    State("migration-download-filetype", "value"),
    State("app-session-store", "data"),
    State("migration-study-rate", "value"),
    State("migration-study-rate", "invalid"),
    State("migration-frequency", "value"),
    State("migration-period-length", "value"),
    State("migration-period-length", "invalid"),
    State("app-mode-store", "data"),
    prevent_initial_call=True)
def export_migration_to_download(n_clicks, download_filetype, app_session_store, migration_study_rate,
                                 migration_study_rate_error, migration_frequency, migration_period_length,
                                 migration_period_length_error, app_mode_store):

    if any(i is None for i in [n_clicks, download_filetype, migration_study_rate, migration_period_length]):
        raise PreventUpdate

    if any([migration_period_length_error, migration_study_rate_error]):
        raise PreventUpdate

    if app_mode_store.get('test_mode'):
        active_app_data = app_mode_store['test_data']
    else:
        active_app_data = app_data

    timeblock_df = active_app_data.create_timeblock(app_session_store)

    if app_session_store.get("active_filters", False):
        for [col, values] in app_session_store["active_filters"]:
            timeblock_df = timeblock_df.loc[timeblock_df[col].isin(values)]

    merged_df = active_app_data.merge_timeblocks(timeblock_df)
    gap_info_df = active_app_data.generate_gap_information(merged_df, app_session_store)
    active_app_data.compute_weights(gap_info_df)
    migration_df, period_start_end = active_app_data.migration_table_processing(
        gap_info_df, app_session_store, active_studies_per=migration_study_rate, period_length=migration_period_length,
        transfer_window_type='D' if migration_frequency == 'Days' else 'W'
    )
    spreadsheet_df = active_app_data.format_for_export(migration_df)
    config_df = active_app_data.format_config_sheet(app_session_store=app_session_store,
                                                    period_start_end=period_start_end,
                                                    period_length=migration_period_length,
                                                    transfer_window_type='D' if migration_frequency == 'Days' else 'W',
                                                    active_studies_per=migration_study_rate)

    current_time = time.strftime('%Y%m%d-%H%M%S')
    data_csv = dcc.send_data_frame(spreadsheet_df.to_csv, f"migration_export_{current_time}.csv", index=False)
    config_csv = dcc.send_data_frame(config_df.to_csv, f"migration_configuration_export_{current_time}.csv",
                                     index=False)
    return data_csv, config_csv


@callback(
    Output('migration-period-length', 'max'),
    Input('app-session-store', 'data'),
    State('migration-frequency', 'value')
)
def set_maximum_period_length(app_session_store, transfer_window_type):

    start_date = pd.Timestamp(app_session_store['timeframe_start']).date()
    end_date = pd.Timestamp(app_session_store['timeframe_end']).date()

    if transfer_window_type == 'Days':
        return len(pd.date_range(start=start_date, end=end_date, freq='D'))
    else:
        week_sundays = list(pd.date_range(start=start_date, end=end_date, freq='W').date)
        if (end_date - week_sundays[-1]).days < 7:
            week_sundays = week_sundays[:-1]
        return len(week_sundays)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=8051, dev_tools_hot_reload=False)
