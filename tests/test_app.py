from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

import os
import dash_bootstrap_components as dbc
from config import AppConfiguration

import utils
from src.data_processing import Data
from dash import no_update
from dash.exceptions import PreventUpdate
from app import collapse_sidebar, set_study_graph_figure, no_data_figure, set_filter_study_graph_figure, \
    one_or_more_figure, set_filter_compound_graph_figure, update_graphs_and_stores, return_study_checklist_options, \
    return_compound_checklist_options
import pytest
import pandas as pd
import plotly.graph_objs as go


@pytest.fixture
def app_test_configuration():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    app_config = AppConfiguration()
    app_config.data_path = os.path.join(dir_path, "data/test-data-1-eu.csv")
    app_config.timeframe_start = pd.Timestamp(year=2016, month=1, day=1)
    app_config.timeframe_end = pd.Timestamp(year=2026, month=1, day=1)
    app_config.day_first_dates = True
    app_config.study_label = "Study"
    app_config.compound_label = "Compound"
    app_config.unique_identity_label = "DPN(Compound-Study)"
    app_config.milestone_definitions = {
        "Milestone 1": utils.Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
        "Milestone 2": utils.Milestone("Milestone 2", offset_before=14, offset_after=14, active=True),
        "Milestone 3": utils.Milestone("Milestone 3", offset_before=14, offset_after=14, active=True),
        "Milestone 4": utils.Milestone("Milestone 4", offset_before=7, offset_after=7, active=False),
    }
    return app_config


@pytest.fixture
def app_test_dataclass(app_test_configuration):
    return Data(app_test_configuration)


@pytest.fixture
def app_test_mode_store(app_test_dataclass):
    return {
        "test_mode": True,
        "test_data": app_test_dataclass
    }


@pytest.fixture
def empty_plot_df():
    return pd.DataFrame(
        columns=["DPN(Compound-Study)", "Study", "Compound", "start", "end", "type", "inside timeframe"]
    )


@pytest.fixture
def plot_df():
    return pd.DataFrame(
        columns=["DPN(Compound-Study)", "Study", "Compound", "start", "end", "type", "inside timeframe"],
        data={
            "DPN(Compound-Study)": ['DN054-007'] * 4 + ['DN053-038'] * 2 + ['DN055-022'],
            "Study": [7] * 4 + [38] * 2 + [22],
            "Compound": [54] * 4 + [53] * 2 + [55],
            "start": [
                pd.Timestamp(2019, 5, 7), pd.Timestamp(2020, 9, 1), pd.Timestamp(2026, 7, 2),
                pd.Timestamp(2026, 10, 13), pd.Timestamp(2016, 11, 16), pd.Timestamp(2018, 6, 20),
                pd.Timestamp(2028, 10, 13)
            ],
            "end": [
                pd.Timestamp(2019, 6, 4), pd.Timestamp(2020, 9, 29), pd.Timestamp(2026, 7, 30),
                pd.Timestamp(2026, 11, 10), pd.Timestamp(2016, 12, 14), pd.Timestamp(2018, 7, 18),
                pd.Timestamp(2028, 10, 13)
            ],
            "type": ["Milestone 1", "Milestone 2", "Milestone 3", "Milestone 4", "Milestone 1", "Milestone 2",
                     "Milestone 1"],
            "inside timeframe": ["Yes", "Yes", "No", "No", "Yes", "Yes", "No"]
        }
    )


def test_collapse_sidebar_callback():
    def run_callback(collapse_clicks, open_clicks, prop_id):
        context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": prop_id}]}))
        return collapse_sidebar(collapse_clicks, open_clicks)

    # None clicks event caused by element creation
    ctx = copy_context()
    with pytest.raises(PreventUpdate):
        ctx.run(run_callback, collapse_clicks=None, open_clicks=None, prop_id="collapse-sidebar.n_clicks")
        ctx.run(run_callback, collapse_clicks=1, open_clicks=None, prop_id="open-sidebar.n_clicks")

    # Invalid element id
    with pytest.raises(PreventUpdate):
        ctx.run(run_callback, collapse_clicks=1, open_clicks=1, prop_id="unexpected.n_clicks")

    # Valid collapse click
    output_1 = ctx.run(run_callback, collapse_clicks=1, open_clicks=None, prop_id="collapse-sidebar.n_clicks")
    expected_1 = {"display": "none"}, {"display": "block"}
    assert output_1 == expected_1

    # Valid open click
    output_2 = ctx.run(run_callback, collapse_clicks=2, open_clicks=1, prop_id="open-sidebar.n_clicks")
    expected_2 = {"display": "block"}, {"display": "none"}
    assert output_2 == expected_2


def test_set_study_graph_figure(plot_df, empty_plot_df):

    output_1 = set_study_graph_figure(empty_plot_df.to_json(), True)
    output_2 = set_study_graph_figure(plot_df.to_json(), True)
    output_3 = set_study_graph_figure(plot_df.to_json(), False)

    # Empty dataframe case for no selected milestones
    output_4 = set_study_graph_figure(pd.DataFrame().to_json(), True)

    assert output_1 == no_data_figure()
    assert isinstance(output_2, go.Figure)
    assert isinstance(output_3, go.Figure)
    assert output_4 == no_data_figure()

    # Ensure timeframe is filtered
    assert (output_2.data[0].y == ['DN054-007', 'DN053-038']).all()
    assert (output_3.data[0].y == ['DN054-007', 'DN053-038', 'DN055-022']).all()

    # Ensure margins are correct
    assert output_1['layout']['margin'] == dict(l=5, r=5, t=5, b=5)
    assert output_2.layout.margin == go.layout.Margin(l=5, r=5, t=5, b=5)
    assert output_3.layout.margin == go.layout.Margin(l=5, r=5, t=5, b=5)

    # Ensure the content of the graph is initially visible
    df = plot_df.loc[plot_df["inside timeframe"] == "Yes"]
    output_2_x_start, output_2_x_end = output_2.layout.xaxis.range
    assert all(start_date >= output_2_x_start for start_date in df['start'])
    assert all(end_date <= output_2_x_end for end_date in df['end'])

    output_3_x_start, output_3_x_end = output_3.layout.xaxis.range
    assert all(start_date >= output_3_x_start for start_date in plot_df['start'])
    assert all(end_date <= output_3_x_end for end_date in plot_df['end'])


def test_set_filter_study_graph_figure(plot_df, empty_plot_df):

    # None clicks event caused by element creation
    with pytest.raises(PreventUpdate):
        set_filter_study_graph_figure(None, plot_df.to_json(), True, [])

    # No studies selected
    output_1_figure, output_1_studies = set_filter_study_graph_figure([], plot_df.to_json(), True, [])
    assert output_1_figure == one_or_more_figure("Studies")
    assert output_1_studies == no_update

    # Same studies selected
    with pytest.raises(PreventUpdate):
        set_filter_study_graph_figure([123], plot_df.to_json(), True, [123])

    # No study within timeframe
    output_2 = set_filter_study_graph_figure([22], plot_df.to_json(), True, [])
    assert output_2 == (no_data_figure(), [22])

    # Study does not exist
    output_2 = set_filter_study_graph_figure([123], plot_df.to_json(), True, [])
    assert output_2 == (no_data_figure(), [123])

    # Timeframe filtering
    output_3_figure, output_3_study_list = set_filter_study_graph_figure([22], plot_df.to_json(), False, [])
    assert isinstance(output_3_figure, go.Figure)
    assert output_3_study_list == [22]
    assert (output_3_figure.data[0].y == ['DN055-022']).all()

    # No study data
    output_4 = set_filter_study_graph_figure([22], empty_plot_df.to_json(), True, [])
    assert output_4 == (no_data_figure(), [22])

    # Study filtering
    output_5_figure, output_5_study_list = set_filter_study_graph_figure([38, 7], plot_df.to_json(), True, [])
    assert isinstance(output_5_figure, go.Figure)
    assert output_5_study_list == [38, 7]
    assert (output_5_figure.data[0].y == ['DN054-007', 'DN053-038']).all()

    # Margins are set
    assert output_1_figure['layout']['margin'] == dict(l=5, r=5, t=5, b=5)
    assert output_3_figure.layout.margin == go.layout.Margin(l=5, r=5, t=5, b=5)
    assert output_5_figure.layout.margin == go.layout.Margin(l=5, r=5, t=5, b=5)

    # Data is visible
    filtered_df_3 = plot_df.loc[plot_df["Study"] == 22]
    output_3_x_start, output_3_x_end = output_3_figure.layout.xaxis.range
    assert all(start_date >= output_3_x_start for start_date in filtered_df_3['start'])
    assert all(end_date <= output_3_x_end for end_date in filtered_df_3['end'])

    filtered_df_5 = plot_df.loc[plot_df["Study"].isin([38, 7]) & plot_df["inside timeframe"] == "Yes"]
    output_5_x_start, output_5_x_end = output_5_figure.layout.xaxis.range
    assert all(start_date >= output_5_x_start for start_date in filtered_df_5['start'])
    assert all(end_date <= output_5_x_end for end_date in filtered_df_5['end'])


def test_set_filter_compound_graph_figure(plot_df, empty_plot_df):

    # None clicks event caused by element creation
    with pytest.raises(PreventUpdate):
        set_filter_compound_graph_figure(None, plot_df.to_json(), True, [])

    # No compounds selected
    output_1_figure, output_1_compounds = set_filter_compound_graph_figure([], plot_df.to_json(), True, [])
    assert output_1_figure == one_or_more_figure("Compounds")
    assert output_1_compounds == no_update

    # Same Compounds selected
    with pytest.raises(PreventUpdate):
        set_filter_compound_graph_figure([321], plot_df.to_json(), True, [321])

    # No compound within timeframe
    output_2 = set_filter_compound_graph_figure([55], plot_df.to_json(), True, [])
    assert output_2 == (no_data_figure(), [55])

    # Compound does not exist
    output_3 = set_filter_compound_graph_figure([123], plot_df.to_json(), True, [])
    assert output_3 == (no_data_figure(), [123])

    # Timeframe filtering
    output_4_figure, output_4_compounds = set_filter_compound_graph_figure([55], plot_df.to_json(), False, [])
    assert isinstance(output_4_figure, go.Figure)
    assert (output_4_figure.data[0].y == ['DN055-022']).all()
    assert output_4_compounds == [55]

    # No compound data
    output_5 = set_filter_compound_graph_figure([55], empty_plot_df.to_json(), True, [])
    assert output_5 == (no_data_figure(), [55])

    # Compound filtering
    output_6_figure, output_6_compounds = set_filter_compound_graph_figure([54, 53], plot_df.to_json(), True, [])
    assert isinstance(output_6_figure, go.Figure)
    assert (output_6_figure.data[0].y == ['DN054-007', 'DN053-038']).all()
    assert output_6_compounds == [54, 53]

    # Margins are set
    assert output_1_figure['layout']['margin'] == dict(l=5, r=5, t=5, b=5)
    assert output_4_figure.layout.margin == go.layout.Margin(l=5, r=5, t=5, b=5)
    assert output_6_figure.layout.margin == go.layout.Margin(l=5, r=5, t=5, b=5)

    # Data is visible
    filtered_df_4 = plot_df.loc[plot_df["Compound"] == 55]
    output_4_x_start, output_4_x_end = output_4_figure.layout.xaxis.range
    assert all(start_date >= output_4_x_start for start_date in filtered_df_4['start'])
    assert all(end_date <= output_4_x_end for end_date in filtered_df_4['end'])

    filtered_df_6 = plot_df.loc[plot_df["Study"].isin([54, 53]) & plot_df["inside timeframe"] == "Yes"]
    output_6_x_start, output_6_x_end = output_6_figure.layout.xaxis.range
    assert all(start_date >= output_6_x_start for start_date in filtered_df_6['start'])
    assert all(end_date <= output_6_x_end for end_date in filtered_df_6['end'])


def test_return_study_checklist_options(app_test_mode_store):

    # None n_clicks trigger from button generation
    with pytest.raises(PreventUpdate):
        return_study_checklist_options(None, 22, [], app_test_mode_store)

    # None search input button press
    with pytest.raises(PreventUpdate):
        return_study_checklist_options(1, None, [], app_test_mode_store)

    # Invalid search value
    output_1 = return_study_checklist_options(1, "White Space", [], app_test_mode_store)
    assert not hasattr(output_1, 'options')
    assert isinstance(output_1, dbc.Checklist)

    # Exact search value & no duplicates
    output_2 = return_study_checklist_options(1, "661", [], app_test_mode_store)
    assert output_2.options == [{'label': 661, 'value': 661}]
    assert isinstance(output_2, dbc.Checklist)

    # Close search value
    output_3 = return_study_checklist_options(1, "61", [], app_test_mode_store)
    assert isinstance(output_3, dbc.Checklist)
    assert sorted(output_3.options, key=lambda option: option['label']) == [
        {'label': 61, 'value': 61},
        {'label': 461, 'value': 461},
        {'label': 661, 'value': 661}
    ]

    # No matching study
    output_4 = return_study_checklist_options(1, "123456", [], app_test_mode_store)
    assert output_4.options == []
    assert isinstance(output_4, dbc.Checklist)

    # Already selected study included in values
    output_5 = return_study_checklist_options(1, "661", [661], app_test_mode_store)
    assert output_5.options == [{'label': 661, 'value': 661}]
    assert output_5.value == [661]

    # Ensure ID & className matches expected
    for output in [output_1, output_2, output_3, output_4, output_5]:
        assert output.id == 'study-checklist-found'
        assert output.className == 'checklist-found'


def test_return_compound_checklist_options(app_test_mode_store):

    # None n_clicks trigger from button generation
    with pytest.raises(PreventUpdate):
        return_compound_checklist_options(None, 22, [], app_test_mode_store)

    # None search input button press
    with pytest.raises(PreventUpdate):
        return_compound_checklist_options(1, None, [], app_test_mode_store)

    # Invalid search value
    output_1 = return_compound_checklist_options(1, "White Space", [], app_test_mode_store)
    assert not hasattr(output_1, 'options')
    assert isinstance(output_1, dbc.Checklist)

    # Exact search value & no duplicates
    output_2 = return_compound_checklist_options(1, "760", [], app_test_mode_store)
    assert output_2.options == [{'label': 760, 'value': 760}]
    assert isinstance(output_2, dbc.Checklist)

    # Close search value
    output_3 = return_compound_checklist_options(1, "32", [], app_test_mode_store)
    assert isinstance(output_3, dbc.Checklist)
    assert sorted(output_3.options, key=lambda option: option['label']) == [
        {'label': 132, 'value': 132},
        {'label': 323, 'value': 323},
        {'label': 832, 'value': 832}
    ]

    # No matching compound
    output_4 = return_compound_checklist_options(1, "123456", [], app_test_mode_store)
    assert output_4.options == []
    assert isinstance(output_4, dbc.Checklist)

    # Already selected compound included in values
    output_5 = return_compound_checklist_options(1, "760", [760], app_test_mode_store)
    assert output_5.options == [{'label': 760, 'value': 760}]
    assert output_5.value == [760]

    # Ensure ID & classname matches expected
    for output in [output_1, output_2, output_3, output_4, output_5]:
        assert output.id == 'compound-checklist-found'
        assert output.className == 'checklist-found'


def test_update_graphs_and_stores_first_start(app_test_mode_store, app_test_configuration):

    output_1 = update_graphs_and_stores(
        None,
        {
            'timeframe_start': app_test_configuration.timeframe_start.isoformat(),
            'timeframe_end': app_test_configuration.timeframe_end.isoformat(),
            'milestones': {
                label: milestone.__dict__ for label, milestone in app_test_configuration.milestone_definitions.items()
            },
            'first_start': True
        },
        "2016-01-01T00:00:00", "2026-01-01T00:00:00",
        [True, True, True, False], [14, 14, 14, 7], [14, 14, 14, 7],
        app_test_mode_store
    )

    assert list(output_1.keys()) == ["app_session_store", "plot_df", "spinner_div"]
    assert output_1.get("app_session_store") == {
        'timeframe_start': app_test_configuration.timeframe_start.isoformat(),
        'timeframe_end': app_test_configuration.timeframe_end.isoformat(),
        'milestones': {
            label: milestone.__dict__ for label, milestone in app_test_configuration.milestone_definitions.items()
        },
        'first_start': False
    }
    assert output_1.get("plot_df") == app_test_mode_store['test_data'].plot_df.to_json()
    assert output_1.get("spinner_div") == {"marginLeft": 50}


def test_update_graphs_and_stores_recurring_start(app_test_mode_store, app_test_configuration):

    # Same timeframe & milestones
    output_1 = update_graphs_and_stores(
        None,
        {
            'timeframe_start': app_test_configuration.timeframe_start.isoformat(),
            'timeframe_end': app_test_configuration.timeframe_end.isoformat(),
            'milestones': {
                "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7, "active": False},
            },
            'first_start': False
        },
        "2016-01-01T00:00:00", "2026-01-01T00:00:00",
        [True, True, True, False], [14, 14, 14, 7], [14, 14, 14, 7],
        app_test_mode_store
    )

    assert list(output_1.keys()) == ["app_session_store", "plot_df", "spinner_div"]
    assert output_1.get("app_session_store") == {
        'timeframe_start': "2016-01-01T00:00:00",
        'timeframe_end': "2026-01-01T00:00:00",
        'milestones': {
            "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7, "active": False},
        },
        'first_start': False
    }
    assert output_1.get("plot_df") == app_test_mode_store['test_data'].plot_df.to_json()
    assert output_1.get("spinner_div") == {"marginLeft": 50}

    # Updated timeframe start and end
    output_2 = update_graphs_and_stores(
        None,
        {
            'timeframe_start': app_test_configuration.timeframe_start.isoformat(),
            'timeframe_end': app_test_configuration.timeframe_end.isoformat(),
            'milestones': {
                "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7, "active": False},
            },
            'first_start': False
        },
        "2010-01-01T00:00:00", "2012-01-01T00:00:00",
        [True, True, True, False], [14, 14, 14, 7], [14, 14, 14, 7],
        app_test_mode_store
    )
    assert list(output_2.keys()) == ["app_session_store", "plot_df", "spinner_div"]
    assert output_2.get("app_session_store") == {
        'timeframe_start': "2010-01-01T00:00:00",
        'timeframe_end': "2012-01-01T00:00:00",
        'milestones': {
            "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7, "active": False},
        },
        'first_start': False
    }
    assert output_2.get("plot_df") == app_test_mode_store['test_data'].plot_df.to_json()
    assert output_2.get("spinner_div") == {"marginLeft": 50}

    # Updated milestones
    output_3 = update_graphs_and_stores(
        None,
        {
            'timeframe_start': app_test_configuration.timeframe_start.isoformat(),
            'timeframe_end': app_test_configuration.timeframe_end.isoformat(),
            'milestones': {
                "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7, "active": False},
            },
            'first_start': False
        },
        "2016-01-01T00:00:00", "2026-01-01T00:00:00",
        [False, False, False, True], [1, 2, 3, 4], [5, 6, 7, 8],
        app_test_mode_store
    )
    assert list(output_3.keys()) == ["app_session_store", "plot_df", "spinner_div"]
    assert output_3.get("app_session_store") == {
        'timeframe_start': "2016-01-01T00:00:00",
        'timeframe_end': "2026-01-01T00:00:00",
        'milestones': {
            "Milestone 1": {"label": "Milestone 1", "offset_before": 1, "offset_after": 5, "active": False},
            "Milestone 2": {"label": "Milestone 2", "offset_before": 2, "offset_after": 6, "active": False},
            "Milestone 3": {"label": "Milestone 3", "offset_before": 3, "offset_after": 7, "active": False},
            "Milestone 4": {"label": "Milestone 4", "offset_before": 4, "offset_after": 8, "active": True},
        },
        'first_start': False
    }
    assert output_3.get("plot_df") == app_test_mode_store['test_data'].plot_df.to_json()
    assert output_3.get("spinner_div") == {"marginLeft": 50}

    # No active milestones
    output_4 = update_graphs_and_stores(
        None,
        {
            'timeframe_start': app_test_configuration.timeframe_start.isoformat(),
            'timeframe_end': app_test_configuration.timeframe_end.isoformat(),
            'milestones': {
                "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
                "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7, "active": False},
            },
            'first_start': False
        },
        "2016-01-01T00:00:00", "2026-01-01T00:00:00",
        [False, False, False, False], [1, 2, 3, 4], [5, 6, 7, 8],
        app_test_mode_store
    )
    assert list(output_4.keys()) == ["app_session_store", "plot_df", "spinner_div"]
    assert output_4.get("app_session_store") == {
        'timeframe_start': "2016-01-01T00:00:00",
        'timeframe_end': "2026-01-01T00:00:00",
        'milestones': {
            "Milestone 1": {"label": "Milestone 1", "offset_before": 1, "offset_after": 5, "active": False},
            "Milestone 2": {"label": "Milestone 2", "offset_before": 2, "offset_after": 6, "active": False},
            "Milestone 3": {"label": "Milestone 3", "offset_before": 3, "offset_after": 7, "active": False},
            "Milestone 4": {"label": "Milestone 4", "offset_before": 4, "offset_after": 8, "active": False},
        },
        'first_start': False
    }
    assert output_4.get("plot_df") == pd.DataFrame().to_json()
    assert output_4.get("spinner_div") == {"marginLeft": 50}
