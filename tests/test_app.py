from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

import os
import json
import platform
import pytest
import pandas as pd
import dash_bootstrap_components as dbc
from config import AppConfiguration

import utils
from src.data_processing import Data
from dash import no_update
from dash.exceptions import PreventUpdate
from app import collapse_sidebar, set_study_graph_figure, no_data_figure, set_filter_study_graph_figure, \
    one_or_more_figure, set_filter_compound_graph_figure, update_graphs_and_stores, return_study_checklist_options, \
    return_compound_checklist_options, update_custom_filters, custom_filter, get_unique_column_values, \
    calculate_study_transfer_count, export_migration_to_download, update_migration_table, set_maximum_period_length
import plotly.graph_objs as go
from test_utils import NullContext

does_not_raise = NullContext()


@pytest.fixture(scope="session", autouse=True)
def log_global_env_facts(record_testsuite_property):
    record_testsuite_property("operating_system", platform.system())
    record_testsuite_property("operating_system_version", platform.version())
    record_testsuite_property("operating_system_release", platform.release())
    record_testsuite_property("python_version", platform.python_version())


def load_json_test_dataframe(filename) -> pd.DataFrame:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return pd.read_json(path_or_buf=os.path.join(dir_path, "data", "dataframes", f"{filename}.json"))


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
        "Milestone 5": utils.Milestone("Milestone 5", offset_before=7, offset_after=7, active=True)
    }
    app_config.day_weight_coefficient = 5
    app_config.gap_weight_coefficient = 0.1
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
def app_test_session_store(app_test_configuration):
    return {
        'timeframe_start': app_test_configuration.timeframe_start.isoformat(),
        'timeframe_end': app_test_configuration.timeframe_end.isoformat(),
        'milestones': {
            "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 4": {"label": "Milestone 4", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 5": {"label": "Milestone 5", "offset_before": 7, "offset_after": 7, "active": True},
        },
        'first_start': False
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


@pytest.mark.parametrize(
    "example_input, expected, exception_context, test_id",
    [
        # None clicks triggered by element creation
        (
            {"collapse_clicks": None, "open_clicks": None, "prop_id": "collapse-sidebar.n_clicks"},
            None,
            pytest.raises(PreventUpdate),
            'PYT1[Raise PreventUpdate 1]',
        ),
        # Invalid prop id
        (
            {"collapse_clicks": 1, "open_clicks": 1, "prop_id": "unexpected.n_clicks"},
            None,
            pytest.raises(PreventUpdate),
            'PYT1[Raise PreventUpdate 2]',
        ),
        # Valid collapse click
        (
            {"collapse_clicks": 1, "open_clicks": None, "prop_id": "collapse-sidebar.n_clicks"},
            ({"display": "none"}, {"display": "block"}),
            does_not_raise,
            'PYT1[Close Sidebar]',
        ),
        # Valid open click
        (
            {"collapse_clicks": 2, "open_clicks": 1, "prop_id": "open-sidebar.n_clicks"},
            ({"display": "block"}, {"display": "none"}),
            does_not_raise,
            'PYT1[Open Sidebar]',
        )
    ],
    ids=['PYT1[Raise PreventUpdate 1]', 'PYT1[Raise PreventUpdate 2]', 'PYT1[Close Sidebar]', 'PYT1[Open Sidebar]']
)
def test_collapse_sidebar_callback(example_input, expected, exception_context, test_id,
                                   record_xml_attribute):
    record_xml_attribute("qualification", "dq")
    record_xml_attribute("test_id", test_id)
    record_xml_attribute("srs_requirement", "SRSREQ6")
    record_xml_attribute("frs_requirement", "N/A")
    record_xml_attribute("scenario", "DQSCE1")

    record_xml_attribute("purpose", "Test that the sidebar correctly handles open and close callback events.")
    record_xml_attribute("description", "Invokes the callback within a copy_context with example input arguments.")
    record_xml_attribute("acceptance_criteria", "Raises PreventUpdate Exception or return correct display attributes.")

    def run_callback(collapse_clicks, open_clicks, prop_id):
        context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": prop_id}]}))
        return collapse_sidebar(collapse_clicks, open_clicks)

    ctx = copy_context()

    with exception_context:
        assert ctx.run(run_callback, **example_input) == expected


def test_set_study_graph_figure(plot_df, empty_plot_df, app_test_session_store, record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", "PYT2")
    record_xml_attribute("frs_requirement", "FRSREQ1")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("scenario", "OQSCE4")

    record_xml_attribute("purpose", "Verify that the study graph figures are correctly formatted.")
    record_xml_attribute(
        "description",
        "Create 4 study graph figures of different configurations and compare against expected values."
    )
    record_xml_attribute(
        "acceptance_criteria",
        "Assert that figure contents, margins, and presentation match expected values."
    )

    output_1 = set_study_graph_figure(empty_plot_df.to_json(), True, app_test_session_store)
    output_2 = set_study_graph_figure(plot_df.to_json(), True, app_test_session_store)
    output_3 = set_study_graph_figure(plot_df.to_json(), False, app_test_session_store)

    # Empty dataframe case for no selected milestones
    output_4 = set_study_graph_figure(pd.DataFrame().to_json(), True, app_test_session_store)

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


@pytest.mark.parametrize(
    "example_input, expected, exception_context, prop_id, test_id",
    [
        # None clicks caused by element creation
        (
            {"study_lst": None, "timeframe_bool": True, "prev_selected": []},
            None,
            pytest.raises(PreventUpdate),
            "study-checklist-found",
            'PYT3[Raise PreventUpdate 1]',
        ),
        # No studies selected
        (
            {"study_lst": [], "timeframe_bool": True, "prev_selected": []},
            (one_or_more_figure("Studies"), no_update),
            does_not_raise,
            "study-checklist-found",
            'PYT3[No Studies Selected]',
        ),
        # Same studies selected
        (
            {"study_lst": [123], "timeframe_bool": True, "prev_selected": [123]},
            None,
            pytest.raises(PreventUpdate),
            "study-checklist-found",
            'PYT3[Raise PreventUpdate 2]',
        ),
        # No study within timeframe
        (
            {"study_lst": [22], "timeframe_bool": True, "prev_selected": []},
            (no_data_figure(), [22]),
            does_not_raise,
            "study-checklist-found",
            'PYT3[No Study Within Timeframe]',
        ),
        # Study does not exist
        (
            {"study_lst": [123], "timeframe_bool": True, "prev_selected": []},
            (no_data_figure(), [123]),
            does_not_raise,
            "study-checklist-found",
            'PYT3[Study Does Not Exist]',
        )
    ],
    ids=['PYT3[Raise PreventUpdate 1]', 'PYT3[No Studies Selected]', 'PYT3[Raise PreventUpdate 2]',
         'PYT3[No Study Within Timeframe]', 'PYT3[Study Does Not Exist]']
)
def test_set_filter_study_graph_figure_error_handling(example_input, expected, exception_context, prop_id, test_id,
                                                      plot_df, app_test_session_store, record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", test_id)
    record_xml_attribute("frs_requirement", "FRSREQ3")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("scenario", "OQSCE5")

    record_xml_attribute("purpose", "Test the filtered study graph handles bad data and requests correctly.")
    record_xml_attribute(
        "description",
        "Run the set_filter_study_graph_figure callback with various bad inputs."
    )
    record_xml_attribute(
        "acceptance_criteria",
        "Raises PreventUpdate exception or returns a tuple of the correct figure and either selected studies or "
        "noupdate."
    )

    def run_callback(example_input, prop_id):
        context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": prop_id}]}))

        with exception_context:
            return set_filter_study_graph_figure(**example_input, frozen=plot_df.to_json(),
                                                 app_session_store=app_test_session_store)

    ctx = copy_context()

    assert ctx.run(run_callback, example_input, prop_id) == expected


def test_set_filter_study_graph_figure(empty_plot_df, plot_df, app_test_session_store, record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", "PYT4")
    record_xml_attribute("frs_requirement", "FRSREQ3")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("scenario", "OQSCE5")

    record_xml_attribute("purpose", "Test the filtered study graph output under expected conditions.")
    record_xml_attribute(
        "description",
        "Run the set_filter_study_graph_figure callback in 5 different situations."
    )
    record_xml_attribute(
        "acceptance_criteria",
        "Assert that correctly formatted figures of expected formats are returned for each situation."
    )

    def run_callback(example_input, prop_id):
        context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": prop_id}]}))
        return set_filter_study_graph_figure(*example_input)

    ctx = copy_context()

    output_1_figure, _ = ctx.run(run_callback, ([], plot_df.to_json(), True, [], app_test_session_store),
                                 prop_id="study-checklist-found")

    # No study data
    output_4 = ctx.run(run_callback, ([22], empty_plot_df.to_json(), True, [], app_test_session_store),
                       prop_id="study-checklist-found")
    assert output_4 == (no_data_figure(), [22])

    # Timeframe filtering
    output_3_figure, output_3_study_list = ctx.run(run_callback,
                                                   ([22], plot_df.to_json(), False, [], app_test_session_store),
                                                   prop_id="study-checklist-found")
    assert isinstance(output_3_figure, go.Figure)
    assert output_3_study_list == [22]
    assert (output_3_figure.data[0].y == ['DN055-022']).all()

    # Study filtering
    output_5_figure, output_5_study_list = ctx.run(run_callback,
                                                   ([38, 7], plot_df.to_json(), True, [], app_test_session_store),
                                                   prop_id="study-checklist-found")
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

    # Global filters applied despite search terms being the same
    output_6_figure, output_6_study_list = ctx.run(run_callback,
                                                   ([38, 7], plot_df.to_json(), True, [[38, 7]], app_test_session_store),
                                                   prop_id="data-plot-df-store")

    assert isinstance(output_6_figure, go.Figure)
    assert output_6_study_list == [38, 7]
    assert (output_6_figure.data[0].y == ['DN054-007', 'DN053-038']).all()


@pytest.mark.parametrize(
    "example_input, expected, exception_context, prop_id, test_id",
    [
        # None clicks caused by element creation
        (
            {"compound_lst": None, "timeframe_bool": True, "prev_selected": []},
            None,
            pytest.raises(PreventUpdate),
            "compound-checklist-found",
            'PYT5[Raise PreventUpdate 1]'
        ),
        # No compounds selected
        (
            {"compound_lst": [], "timeframe_bool": True, "prev_selected": []},
            (one_or_more_figure("Compounds"), no_update),
            does_not_raise,
            "compound-checklist-found",
            'PYT5[No Compound Selected]'
        ),
        # Same compounds selected
        (
            {"compound_lst": [321], "timeframe_bool": True, "prev_selected": [321]},
            None,
            pytest.raises(PreventUpdate),
            "compound-checklist-found",
            'PYT5[Raise PreventUpdate 2]'
        ),
        # No compound within timeframe
        (
            {"compound_lst": [55], "timeframe_bool": True, "prev_selected": []},
            (no_data_figure(), [55]),
            does_not_raise,
            "compound-checklist-found",
            'PYT5[No Compound Within Timeframe]'
        ),
        # Compound does not exist
        (
            {"compound_lst": [123], "timeframe_bool": True, "prev_selected": []},
            (no_data_figure(), [123]),
            does_not_raise,
            "compound-checklist-found",
            'PYT5[Compound Does Not Exist]'
        )
    ],
    ids=['PYT5[Raise PreventUpdate 1]', 'PYT5[No Studies Selected]', 'PYT5[Raise PreventUpdate 2]',
         'PYT5[No Compound Within Timeframe]', 'PYT5[Compound Does Not Exist]']
)
def test_set_filter_compound_graph_figure_error_handling(example_input, expected, exception_context, prop_id, test_id,
                                                         plot_df, app_test_session_store, record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", test_id)
    record_xml_attribute("frs_requirement", "FRSREQ4")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("scenario", "OQSCE6")

    record_xml_attribute("purpose", "Test the filtered compound graph handles bad data and requests correctly.")
    record_xml_attribute(
        "description",
        "Run the set_filter_compound_graph_figure callback with various bad inputs."
    )
    record_xml_attribute(
        "acceptance_criteria",
        "Raises PreventUpdate exception or returns a tuple of the correct figure and either selected compounds or no "
        "update."
    )

    def run_callback(example_input, prop_id):
        context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": prop_id}]}))

        with exception_context:
            return set_filter_compound_graph_figure(**example_input, frozen=plot_df.to_json(),
                                                    app_session_store=app_test_session_store)

    ctx = copy_context()

    assert ctx.run(run_callback, example_input, prop_id) == expected


def test_set_filter_compound_graph_figure(plot_df, empty_plot_df, app_test_session_store, record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", "PYT6")
    record_xml_attribute("frs_requirement", "FRSREQ4")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("scenario", "OQSCE6")

    record_xml_attribute("purpose", "Test the filtered compound graph output under expected conditions.")
    record_xml_attribute(
        "description",
        "Run the set_filter_compound_graph_figure callback in 5 different situations."
    )
    record_xml_attribute(
        "acceptance_criteria",
        "Assert that correctly formatted figures of expected formats are returned for each situation."
    )

    def run_callback(example_input, prop_id):
        context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": prop_id}]}))
        return set_filter_compound_graph_figure(*example_input)

    ctx = copy_context()

    # No compound data
    output_5 = ctx.run(run_callback, ([55], empty_plot_df.to_json(), True, [], app_test_session_store),
                       prop_id="compound-checklist-found")
    assert output_5 == (no_data_figure(), [55])

    # Timeframe filtering
    output_4_figure, output_4_compounds = ctx.run(run_callback, ([55], plot_df.to_json(), False, [], app_test_session_store),
                                                  prop_id="compound-checklist-found")
    assert isinstance(output_4_figure, go.Figure)
    assert (output_4_figure.data[0].y == ['DN055-022']).all()
    assert output_4_compounds == [55]

    # Compound filtering
    output_6_figure, output_6_compounds = ctx.run(run_callback,
                                                  ([54, 53], plot_df.to_json(), True, [], app_test_session_store),
                                                  prop_id="compound-checklist-found")
    assert isinstance(output_6_figure, go.Figure)
    assert (output_6_figure.data[0].y == ['DN054-007', 'DN053-038']).all()
    assert output_6_compounds == [54, 53]

    output_1_figure, _ = ctx.run(run_callback, ([], plot_df.to_json(), True, [], app_test_session_store),
                                 prop_id="compound-checklist-found")

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

    # Global filters applied despite search terms being the same
    output_6_figure, output_6_compound_list = ctx.run(run_callback,
                                                   ([54, 53], plot_df.to_json(), True, [[54, 53]], app_test_session_store),
                                                   prop_id="data-plot-df-store")

    assert isinstance(output_6_figure, go.Figure)
    assert output_6_compound_list == [54, 53]
    assert (output_6_figure.data[0].y == ['DN054-007', 'DN053-038']).all()


@pytest.mark.parametrize(
    "example_input, expected_options, expected_type, expected_value, exception_context, test_id",
    [
        # None n_clicks trigger from button generation
        (
            {"n_clicks": None, "search_value": 22, "selected_items": []},
            None,
            None,
            None,
            pytest.raises(PreventUpdate),
            'PYT7[Raise PreventUpdate 1]',
        ),
        # None search input button press
        (
            {"n_clicks": 1, "search_value": None, "selected_items": []},
            None,
            None,
            None,
            pytest.raises(PreventUpdate),
            'PYT7[Raise PreventUpdate 2]'
        ),
        # Invalid search value
        (
            {"n_clicks": 1, "search_value": "White Space", "selected_items": []},
            None,
            dbc.Checklist,
            None,
            pytest.raises(AttributeError),
            'PYT7[Raise AttributeError 1]'
        ),
        # Exact search value and no duplicates
        (
            {"n_clicks": 1, "search_value": "661", "selected_items": []},
            [{'label': 661, 'value': 661}],
            dbc.Checklist,
            [],
            does_not_raise,
            'PYT7[Exact Search Value And No Duplicates]'
        ),
        # Close search value
        (
            {"n_clicks": 1, "search_value": "61", "selected_items": []},
            [
                {'label': 61, 'value': 61},
                {'label': 461, 'value': 461},
                {'label': 661, 'value': 661}
            ],
            dbc.Checklist,
            [],
            does_not_raise,
            'PYT7[Close Search Value]'
        ),
        # No matching study
        (
            {"n_clicks": 1, "search_value": "123456", "selected_items": []},
            [],
            dbc.Checklist,
            [],
            does_not_raise,
            'PYT7[No Matching Study]'
        ),
        # Already selected study not duplicated in values
        (
            {"n_clicks": 1, "search_value": "661", "selected_items": [661]},
            [{'label': 661, 'value': 661}],
            dbc.Checklist,
            [661],
            does_not_raise,
            'PYT7[Already Selected Study Not Duplicated In Values]'
        ),
        # Already selected study always included in values
        (
            {"n_clicks": 2, "search_value": "123456", "selected_items": [661]},
            [{'label': 661, 'value': 661}],
            dbc.Checklist,
            [661],
            does_not_raise,
            'PYT7[Already Selected Study Included In Values]'
        ),
    ],
    ids=['PYT7[Raise PreventUpdate 1]', 'PYT7[Raise PreventUpdate 2]', 'PYT7[Raise AttributeError 1]',
         'PYT7[Exact Search Value And No Duplicates]', 'PYT7[Close Search Value]', 'PYT7[No Matching Study]',
         'PYT7[Already Selected Study Not Duplicated In Values]', 'PYT7[Already Selected Study Included In Values]']
)
def test_return_study_checklist_options(example_input, expected_options, expected_type, expected_value,
                                        exception_context, test_id, app_test_mode_store, record_xml_attribute):

    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", test_id)
    record_xml_attribute("frs_requirement", "FRSREQ3")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("scenario", "OQSCE5")

    record_xml_attribute("purpose", "Test study graph search checklist callback return options.")
    record_xml_attribute(
        "description",
        "Run the return_study_checklist_options callback in 8 different conditions."
    )
    record_xml_attribute(
        "acceptance_criteria",
        "Raises AttributeError exception, PreventUpdate exception or returns the correct search values and options."
    )

    with exception_context:
        checklist = return_study_checklist_options(**example_input, app_mode_store=app_test_mode_store)
        assert isinstance(checklist, expected_type)
        assert checklist.id == 'study-checklist-found'
        assert checklist.className == 'checklist-found'
        assert checklist.value == expected_value
        assert checklist.options == expected_options


@pytest.mark.parametrize(
    "example_input, expected_options, expected_type, expected_value, exception_context, test_id",
    [
        # None n_clicks trigger from button generation
        (
            {"n_clicks": None, "search_value": 22, "selected_items": []},
            None,
            None,
            None,
            pytest.raises(PreventUpdate),
            'PYT8[Raise PreventUpdate 1]'
        ),
        # None search input button press
        (
            {"n_clicks": 1, "search_value": None, "selected_items": []},
            None,
            None,
            None,
            pytest.raises(PreventUpdate),
            'PYT8[Raise PreventUpdate 2]'
        ),
        # Invalid search value
        (
            {"n_clicks": 1, "search_value": "White Space", "selected_items": []},
            None,
            dbc.Checklist,
            None,
            pytest.raises(AttributeError),
            'PYT8[Raise AttributeError 1]'
        ),
        # Exact search value and no duplicates
        (
            {"n_clicks": 1, "search_value": "760", "selected_items": []},
            [{'label': 760, 'value': 760}],
            dbc.Checklist,
            [],
            does_not_raise,
            'PYT8[Exact Search Value And No Duplicates]'
        ),
        # Close search value
        (
            {"n_clicks": 1, "search_value": "32", "selected_items": []},
            [
                {'label': 132, 'value': 132},
                {'label': 323, 'value': 323},
                {'label': 832, 'value': 832}
            ],
            dbc.Checklist,
            [],
            does_not_raise,
            'PYT8[Close Search Value]'
        ),
        # No matching compound
        (
            {"n_clicks": 1, "search_value": "123456", "selected_items": []},
            [],
            dbc.Checklist,
            [],
            does_not_raise,
            'PYT8[No Matching Compound]'
        ),
        # Already selected compound not duplicated in values
        (
            {"n_clicks": 1, "search_value": "760", "selected_items": [760]},
            [{'label': 760, 'value': 760}],
            dbc.Checklist,
            [760],
            does_not_raise,
            'PYT8[Already Selected Compound Not Duplicated In Values]'
        ),
        # Already selected compound always included in values
        (
            {"n_clicks": 2, "search_value": "123456", "selected_items": [760]},
            [{'label': 760, 'value': 760}],
            dbc.Checklist,
            [760],
            does_not_raise,
            'PYT8[Already Selected Compound Included In Values]'
        ),
    ],
    ids=['PYT8[Raise PreventUpdate 1]', 'PYT8[Raise PreventUpdate 2]', 'PYT8[Raise AttributeError 1]',
         'PYT8[Exact Search Value And No Duplicates]', 'PYT8[Close Search Value]', 'PYT8[No Matching Compound]',
         'PYT8[Already Selected Compound Not Duplicated In Values]', 'PYT8[Already Selected Compound Included In Values]']
)
def test_return_compound_checklist_options(example_input, expected_options, expected_type, expected_value,
                                           exception_context, test_id, app_test_mode_store, record_xml_attribute):

    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", test_id)
    record_xml_attribute("frs_requirement", "FRSREQ4")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("scenario", "OQSCE6")

    record_xml_attribute("purpose", "Test compound graph search checklist callback return options.")
    record_xml_attribute(
        "description",
        "Run the return_compound_checklist_options callback in 8 different conditions."
    )
    record_xml_attribute(
        "acceptance_criteria",
        "Raises AttributeError exception, PreventUpdate exception or returns the correct search values and options."
    )

    with exception_context:
        checklist = return_compound_checklist_options(**example_input, app_mode_store=app_test_mode_store)
        assert isinstance(checklist, expected_type)
        assert checklist.id == 'compound-checklist-found'
        assert checklist.className == 'checklist-found'
        assert checklist.value == expected_value
        assert sorted(checklist.options, key=lambda option: option['label']) == expected_options


def test_update_graphs_and_stores_first_start(app_test_mode_store, app_test_configuration, record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", "PYT9")
    record_xml_attribute("frs_requirement", "FRSREQ5")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("scenario", "OQSCE7")

    record_xml_attribute("purpose", "Validate session store widget contents are as expected on first start.")
    record_xml_attribute(
        "description",
        "Run the update_graphs_and_stores callback under the first start condition and validate outputs."
    )
    record_xml_attribute(
        "acceptance_criteria",
        "Assert that the app_session store contains the correct configuration and that the plot_df json is valid "
        "on first start."
    )

    test_session_store = {
        'timeframe_start': app_test_configuration.timeframe_start.isoformat(),
        'timeframe_end': app_test_configuration.timeframe_end.isoformat(),
        'milestones': {
            label: milestone.__dict__ for label, milestone in app_test_configuration.milestone_definitions.items()
        },
        'first_start': True
    }

    output_1 = update_graphs_and_stores(
        None,
        "2016-01-01T00:00:00", "2026-01-01T00:00:00",
        test_session_store,
        [True, True, True, False, True], [14, 14, 14, 7, 7], [14, 14, 14, 7, 7],
        [], [],
        app_test_mode_store
    )

    test_session_store['first_start'] = False

    assert list(output_1.keys()) == ["app_session_store", "plot_df", "spinner_div"]
    assert output_1.get("app_session_store") == test_session_store

    assert pd.read_json(output_1.get("plot_df")).equals(load_json_test_dataframe("PYT9"))
    assert output_1.get("spinner_div") == {"marginLeft": 50}


@pytest.mark.parametrize(
    "example_input, expected_keys, expected_app_session_store, expected_spinner_div, test_id",
    [
        # Same timeframe & milestones with no filters
        (
            {
                "n_clicks": None,
                "app_session_store": {
                    'milestones': {
                        "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7,
                                        "active": False},
                    },
                    'first_start': False,
                    'active_filters': [],
                },
                "start_date_input": "2016-01-01T00:00:00",
                "end_date_input": "2026-01-01T00:00:00",
                "checkbox_values": [True, True, True, False],
                "offset_before_values": [14, 14, 14, 7],
                "offset_after_values": [14, 14, 14, 7],
                "column_filters": [],
                "column_filter_values": []
            },
            ["app_session_store", "plot_df", "spinner_div"],
            {
                'timeframe_start': "2016-01-01T00:00:00",
                'timeframe_end': "2026-01-01T00:00:00",
                'milestones': {
                    "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
                    "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
                    "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
                    "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7, "active": False},
                },
                'first_start': False,
                'active_filters': [],
            },
            {"marginLeft": 50},
            'PYT10[Same Timeframe And Milestones]'
        ),
        # Updated timeframe start and end with no filters
        (
            {
                "n_clicks": None,
                "app_session_store": {
                    'milestones': {
                        "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7,
                                        "active": False},
                    },
                    'first_start': False,
                    'active_filters': [],
                },
                "start_date_input": "2010-01-01T00:00:00",
                "end_date_input": "2012-01-01T00:00:00",
                "checkbox_values": [True, True, True, False],
                "offset_before_values": [14, 14, 14, 7],
                "offset_after_values": [14, 14, 14, 7],
                "column_filters": [],
                "column_filter_values": []
            },
            ["app_session_store", "plot_df", "spinner_div"],
            {
                'timeframe_start': "2010-01-01T00:00:00",
                'timeframe_end': "2012-01-01T00:00:00",
                'milestones': {
                    "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
                    "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
                    "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
                    "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7, "active": False},
                },
                'first_start': False,
                'active_filters': []
            },
            {"marginLeft": 50},
            'PYT10[Updated Timeframe Start And End]'
        ),
        # Updated milestones with no filters
        (
            {
                "n_clicks": None,
                "app_session_store": {
                    'milestones': {
                        "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7,
                                        "active": False},
                    },
                    'first_start': False,
                    'active_filters': [],
                },
                "start_date_input": "2016-01-01T00:00:00",
                "end_date_input": "2026-01-01T00:00:00",
                "checkbox_values": [False, False, False, True],
                "offset_before_values": [1, 2, 3, 4],
                "offset_after_values": [5, 6, 7, 8],
                "column_filters": [],
                "column_filter_values": []
            },
            ["app_session_store", "plot_df", "spinner_div"],
            {
                'timeframe_start': "2016-01-01T00:00:00",
                'timeframe_end': "2026-01-01T00:00:00",
                'milestones': {
                    "Milestone 1": {"label": "Milestone 1", "offset_before": 1, "offset_after": 5, "active": False},
                    "Milestone 2": {"label": "Milestone 2", "offset_before": 2, "offset_after": 6, "active": False},
                    "Milestone 3": {"label": "Milestone 3", "offset_before": 3, "offset_after": 7, "active": False},
                    "Milestone 4": {"label": "Milestone 4", "offset_before": 4, "offset_after": 8, "active": True},
                },
                'first_start': False,
                'active_filters': [],
            },
            {"marginLeft": 50},
            'PYT10[Updated Milestones]'
        ),
        # No active milestones and no filters
        (
            {
                "n_clicks": None,
                "app_session_store": {
                    'milestones': {
                        "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7,
                                        "active": False},
                    },
                    'active_filters': [],
                    'first_start': False
                },
                "start_date_input": "2016-01-01T00:00:00",
                "end_date_input": "2026-01-01T00:00:00",
                "checkbox_values": [False, False, False, False],
                "offset_before_values": [1, 2, 3, 4],
                "offset_after_values": [5, 6, 7, 8],
                "column_filters": [],
                "column_filter_values": []
            },
            ["app_session_store", "plot_df", "spinner_div"],
            {
                'timeframe_start': "2016-01-01T00:00:00",
                'timeframe_end': "2026-01-01T00:00:00",
                'milestones': {
                    "Milestone 1": {"label": "Milestone 1", "offset_before": 1, "offset_after": 5, "active": False},
                    "Milestone 2": {"label": "Milestone 2", "offset_before": 2, "offset_after": 6, "active": False},
                    "Milestone 3": {"label": "Milestone 3", "offset_before": 3, "offset_after": 7, "active": False},
                    "Milestone 4": {"label": "Milestone 4", "offset_before": 4, "offset_after": 8, "active": False},
                },
                'first_start': False,
                'active_filters': [],
            },
            {"marginLeft": 50},
            'PYT10[No Active Milestones]'
        ),
        # Active milestones with single custom column filter
        (
            {
                "n_clicks": None,
                "start_date_input": "2016-01-01T00:00:00",
                "end_date_input": "2026-01-01T00:00:00",
                "app_session_store": {
                    'milestones': {
                        "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7,
                                        "active": False},
                    },
                    'first_start': False,
                    'active_filters': []
                },
                "checkbox_values": [True, True, True, True],
                "offset_before_values": [1, 2, 3, 4],
                "offset_after_values": [5, 6, 7, 8],
                "column_filters": ['Compound'],
                "column_filter_values": [[233, 846]]
            },
            ["app_session_store", "plot_df", "spinner_div"],
            {
                'timeframe_start': "2016-01-01T00:00:00",
                'timeframe_end': "2026-01-01T00:00:00",
                'milestones': {
                    "Milestone 1": {"label": "Milestone 1", "offset_before": 1, "offset_after": 5, "active": True},
                    "Milestone 2": {"label": "Milestone 2", "offset_before": 2, "offset_after": 6, "active": True},
                    "Milestone 3": {"label": "Milestone 3", "offset_before": 3, "offset_after": 7, "active": True},
                    "Milestone 4": {"label": "Milestone 4", "offset_before": 4, "offset_after": 8, "active": True},
                },
                'first_start': False,
                'active_filters': [['Compound', [233, 846]]],
            },
            {"marginLeft": 50},
            'PYT10[Single Custom Filter]'
        ),
        # Active milestones with multiple custom column filter
        (
            {
                "n_clicks": None,
                "start_date_input": "2016-01-01T00:00:00",
                "end_date_input": "2026-01-01T00:00:00",
                "app_session_store": {
                    'milestones': {
                        "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7,
                                        "active": False},
                    },
                    'first_start': False,
                    'active_filters': [],
                },
                "checkbox_values": [True, True, True, True],
                "offset_before_values": [1, 2, 3, 4],
                "offset_after_values": [5, 6, 7, 8],
                "column_filters": ['Compound', 'TA Code in Directory'],
                "column_filter_values": [[233, 846], ['gt', 'DN']]
            },
            ["app_session_store", "plot_df", "spinner_div"],
            {
                'timeframe_start': "2016-01-01T00:00:00",
                'timeframe_end': "2026-01-01T00:00:00",
                'milestones': {
                    "Milestone 1": {"label": "Milestone 1", "offset_before": 1, "offset_after": 5, "active": True},
                    "Milestone 2": {"label": "Milestone 2", "offset_before": 2, "offset_after": 6, "active": True},
                    "Milestone 3": {"label": "Milestone 3", "offset_before": 3, "offset_after": 7, "active": True},
                    "Milestone 4": {"label": "Milestone 4", "offset_before": 4, "offset_after": 8, "active": True},
                },
                'first_start': False,
                'active_filters': [
                    ['Compound', [233, 846]],
                    ['TA Code in Directory', ['gt', 'DN']]
                ],
            },
            {"marginLeft": 50},
            'PYT10[Multiple Custom Filters]'
        ),
        # No active milestones with multiple custom column filter
        (
            {
                "n_clicks": None,
                "start_date_input": "2016-01-01T00:00:00",
                "end_date_input": "2026-01-01T00:00:00",
                "app_session_store": {
                    'milestones': {
                        "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14,
                                        "active": False},
                        "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14,
                                        "active": True},
                        "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14,
                                        "active": False},
                        "Milestone 4": {"label": "Milestone 4", "offset_before": 7, "offset_after": 7,
                                        "active": False},
                    },
                    'first_start': False,
                    'active_filters': [],
                },
                "checkbox_values": [False, False, False, False],
                "offset_before_values": [1, 2, 3, 4],
                "offset_after_values": [5, 6, 7, 8],
                "column_filters": ['Compound', 'TA Code in Directory'],
                "column_filter_values": [[233, 846], ['gt', 'DN']]
            },
            ["app_session_store", "plot_df", "spinner_div"],
            {
                'timeframe_start': "2016-01-01T00:00:00",
                'timeframe_end': "2026-01-01T00:00:00",
                'milestones': {
                    "Milestone 1": {"label": "Milestone 1", "offset_before": 1, "offset_after": 5, "active": False},
                    "Milestone 2": {"label": "Milestone 2", "offset_before": 2, "offset_after": 6, "active": False},
                    "Milestone 3": {"label": "Milestone 3", "offset_before": 3, "offset_after": 7, "active": False},
                    "Milestone 4": {"label": "Milestone 4", "offset_before": 4, "offset_after": 8, "active": False},
                },
                'first_start': False,
                'active_filters': [],
            },
            {"marginLeft": 50},
            'PYT10[No Milestones And Multiple Custom Filters]'
        )
    ],
    ids=['PYT10[Same Timeframe And Milestones]', 'PYT10[Updated Timeframe Start And End]', 'PYT10[Updated Milestones]',
         'PYT10[No Active Milestones]', 'PYT10[Single Custom Filter]', 'PYT10[Multiple Custom Filters]',
         'PYT10[No Milestones And Multiple Custom Filters]']
)
def test_update_graphs_and_stores_recurring(example_input, expected_keys, expected_app_session_store,
                                            expected_spinner_div, test_id, app_test_mode_store, app_test_configuration,
                                            record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", test_id)
    record_xml_attribute("frs_requirement", "FRSREQ5")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("scenario", "OQSCE8")

    record_xml_attribute("purpose", "Test session store widget contents on recurring triggers.")
    record_xml_attribute(
        "description",
        "Run the update_graphs_and_stores callback under the first start condition and validate outputs."
    )
    record_xml_attribute(
        "acceptance_criteria",
        "Assert session store contents, app_session_div, plot df and output keys match expected values."
    )

    output = update_graphs_and_stores(**example_input, app_mode_store=app_test_mode_store)
    expected_plot_df = load_json_test_dataframe(test_id)

    assert list(output.keys()) == expected_keys
    assert output.get("app_session_store") == expected_app_session_store
    assert pd.read_json(output.get("plot_df")).equals(expected_plot_df)
    assert output.get("spinner_div") == expected_spinner_div


@pytest.mark.parametrize(
    "example_input, should_prevent_update, ids_to_remove, test_id",
    [
        # Add filter from empty
        (
            {
                'add_clicks': 1,
                'remove_clicks': None,
                'filter_children': [],
                'prop_id_dict': {'prop_id': 'add-custom-filter-button'}
            },
            False,
            [],
            "PYT11[Add Filter]"
        ),
        # Remove filter when empty
        (
            {
                'add_clicks': 1,
                'remove_clicks': 2,
                'filter_children': [],
                'prop_id_dict': {'prop_id': '{"index":"cant-remove","type":"remove-filter-button"}'}
            },
            True,
            [],
            "PYT11[Remove Filter From Empty]"
        ),
        # Correct element removed
        (
            {
                'add_clicks': 1,
                'remove_clicks': 2,
                'filter_children': [
                    custom_filter("should-not-remove-1").to_plotly_json(),
                    custom_filter("should-remove").to_plotly_json(),
                    custom_filter("should-not-remove-2").to_plotly_json()
                ],
                'prop_id_dict': {'prop_id': '{"index":"should-remove","type":"remove-filter-button"}'}
            },
            False,
            ["should-remove"],
            "PYT11[Remove Filter]"
        )
    ],
    ids=["PYT11[Add Filter From Empty]", "PYT11[Remove Filter From Empty]", "PYT11[Remove Filter]"]
)
def test_update_custom_filters(example_input, should_prevent_update, ids_to_remove, test_id, record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", test_id)
    record_xml_attribute("srs_requirement", "")
    record_xml_attribute("frs_requirement", "")
    record_xml_attribute("scenario", "")

    record_xml_attribute("purpose", "Test adding and removing custom filters on the UI.")
    record_xml_attribute("description",
                         "Verify the behavior of adding and removing filters in the UI, by checking the expected IDs "
                         "and properties of the UI components.")
    record_xml_attribute("acceptance_criteria",
                         "Either raise PreventUpdate or return specific sets of values and IDs when adding and "
                         "removing filters.")

    def run_callback(add_clicks, remove_clicks, filter_children, prop_id_dict):
        context_value.set(AttributeDict(**{"triggered_inputs": [prop_id_dict]}))
        return update_custom_filters(add_clicks, remove_clicks, filter_children)

    ctx = copy_context()

    if should_prevent_update:
        with pytest.raises(PreventUpdate):
            ctx.run(run_callback, **example_input)
    else:
        returned_parents = ctx.run(run_callback, **example_input)

        for returned_parent in returned_parents:
            if isinstance(returned_parent, dict):
                parent_id = returned_parent["props"]["id"]
                sub_container = returned_parent["props"]["children"][0]

                col_values_dropdown_id = returned_parent["props"]["children"][1].id
                col_label_dropdown_id = sub_container.children[0].id
                remove_button_id = sub_container.children[1].id
            else:
                parent_id = returned_parent.id
                sub_container = returned_parent.children[0]

                col_values_dropdown_id = returned_parent.children[1].id
                col_label_dropdown_id = sub_container.children[0].id
                remove_button_id = sub_container.children[1].id

            assert parent_id not in ids_to_remove
            assert col_values_dropdown_id == {
                "type": "column-values",
                "index": parent_id
            }
            assert col_label_dropdown_id == {
                "type": "column-label",
                "index": parent_id
            }
            assert remove_button_id == {
                "type": "remove-filter-button",
                "index": parent_id
            }


def test_get_unique_column_values(app_test_mode_store, record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", "PYT12")
    record_xml_attribute("srs_requirement", "")
    record_xml_attribute("frs_requirement", "")
    record_xml_attribute("scenario", "")

    record_xml_attribute("purpose", "Validate that unique column values can be retrieved from the dataset.")
    record_xml_attribute("description", "Run the get_unique_column_values callback with a test column.")
    record_xml_attribute("acceptance_criteria", "Either raise PreventUpdate or assert that the returned values are expected.")

    with pytest.raises(PreventUpdate):
        get_unique_column_values(None, app_test_mode_store)

    assert get_unique_column_values("TA Code in Directory", app_test_mode_store) == ['jN', 'gt', 'lo', 'JB', 'DN', 'Tb']


@pytest.mark.parametrize(
    "mode, day_length, days_in_week, study_size, study_size_unit, transfer_rate, transfer_rate_unit, "
    "migration_period_length, current_label, should_prevent_update, expected_result, test_id",
    [
        (
            "Days",
            None,
            5,
            50, 'MB',
            6, 'MB/s',
            100,
            'Studies transferred per day',
            True,
            (),
            "PYT13[None Prop Prevents Update]"
        ),
        (
            "Days",
            8,
            0,
            10000, "KB",
            0.2, "MB/s",
            80,
            'Studies transferred per day',
            False,
            (576, no_update, True, no_update, no_update),
            "PYT13[Days calculation KB and MB/s]"
        ),
        (
            "Days",
            8,
            5,
            50, "MB",
            400, "KB/s",
            100,
            'Studies transferred per day',
            False,
            (230, no_update, True, no_update, no_update),
            "PYT13[Days calculation MB and KB/s]"
        ),
        (
            "Days",
            3,
            2,
            0.4, "TB",
            0.01, "GB/s",
            20,
            'Studies transferred per day',
            False,
            (0, no_update, True, no_update, no_update),
            "PYT13[Days calculation TB and GB/s]"
        ),
        (
            "Weeks",
            8,
            5,
            0.05, "GB",
            400, "KB/s",
            14,
            'Studies transferred per week',
            False,
            (1152, no_update, False, no_update, no_update),
            "PYT13[Weeks calculation GB and KB/s]"
        ),
        (
            "Weeks",
            8,
            5,
            50, "MB",
            400, "KB/s",
            14,
            'Studies transferred per day',
            False,
            (1152, 'Studies transferred per week', False, 2, 'Length of period (weeks)'),
            "PYT13[Transition from days to weeks]"
        ),
        (
            "Days",
            8,
            5,
            50, "MB",
            400, "KB/s",
            14,
            'Studies transferred per week',
            False,
            (230, 'Studies transferred per day', True, 98, 'Length of period (days)'),
            "PYT13[Transition from weeks to days]"
        )
    ],
    ids=["PYT13[None Prop Prevents Update]", "PYT13[Days Calculation KB and MB/s]",
         "PYT13[Days Calculation MB and KB/s]", "PYT13[Days Calculation TB and GB/s]",
         "PYT13[Weeks calculation GB and KB/s]", "PYT13[Transition from Days to weeks]",
         "PYT13[Transition from weeks to days]"]
)
def test_calculate_study_transfer_count(mode, day_length, days_in_week, study_size, study_size_unit, transfer_rate,
                                        transfer_rate_unit, migration_period_length, current_label,
                                        should_prevent_update, expected_result, test_id, record_xml_attribute):

    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", test_id)
    record_xml_attribute("srs_requirement", "")
    record_xml_attribute("frs_requirement", "")
    record_xml_attribute("scenario", "")

    record_xml_attribute("purpose", "Test the calculate_study_transfer_count callback and its outputs in different situations.")
    record_xml_attribute("description",
                         "The test uses different parameter combinations to cover various scenarios, such as different "
                         "modes ('Days' or 'Weeks'), study sizes and transfer rates in different units, and migration " 
                         "period lengths.")
    record_xml_attribute("acceptance_criteria", "Verify that the expected results match the actual output of the "
                         "calculate_study_transfer_count callback, and test whether the callback raises a "
                         "PreventUpdate exception when required.")

    if should_prevent_update:
        with pytest.raises(PreventUpdate):
            calculate_study_transfer_count(mode, day_length, days_in_week, study_size, study_size_unit, transfer_rate,
                                           transfer_rate_unit, migration_period_length, current_label)
    else:
        assert expected_result == calculate_study_transfer_count(
            mode, day_length, days_in_week, study_size, study_size_unit, transfer_rate, transfer_rate_unit,
            migration_period_length, current_label)


@pytest.mark.parametrize(
    "kwargs_input, should_prevent_update, active_filters, test_id",
    [
        (
            {
                "n_clicks": None,
                "download_filetype": 'CSV',
                "migration_study_rate": 10,
                "migration_study_rate_error": False,
                "migration_frequency": "Days",
                "migration_period_length": 100,
                "migration_period_length_error": False
            },
            True,
            [],
            "PYT14[Raise PreventUpdate None Inputs]"
        ),
        (
            {
                "n_clicks": 1,
                "download_filetype": 'CSV',
                "migration_study_rate": 10,
                "migration_study_rate_error": False,
                "migration_frequency": "Days",
                "migration_period_length": 100,
                "migration_period_length_error": False
            },
            False,
            [],
            "PYT14[Daily Without Filters]"
        ),
        (
            {
                "n_clicks": 1,
                "download_filetype": 'CSV',
                "migration_study_rate": 10,
                "migration_study_rate_error": False,
                "migration_frequency": "Days",
                "migration_period_length": -10,
                "migration_period_length_error": True
            },
            True,
            [],
            "PYT14[Raise PreventUpdate Invalid Period Length]"
        ),
        (
            {
                "n_clicks": 1,
                "download_filetype": 'CSV',
                "migration_study_rate": -10,
                "migration_study_rate_error": True,
                "migration_frequency": "Days",
                "migration_period_length": 100,
                "migration_period_length_error": False
            },
            True,
            [],
            "PYT14[Raise PreventUpdate Invalid Study Rate]"
        ),
        (
            {
                "n_clicks": 1,
                "download_filetype": "CSV",
                "migration_study_rate": 10,
                "migration_study_rate_error": False,
                "migration_frequency": "Days",
                "migration_period_length": 100,
                "migration_period_length_error": False
            },
            False,
            [['Compound', [233, 846]], ['TA Code in Directory', ['gt', 'DN']]],
            "PYT14[Daily With Filters]"
        ),
        (
            {
                "n_clicks": 1,
                "download_filetype": 'CSV',
                "migration_study_rate": 10,
                "migration_study_rate_error": False,
                "migration_frequency": "Weeks",
                "migration_period_length": 100,
                "migration_period_length_error": False
            },
            False,
            [],
            "PYT14[Weekly Without Filters]"
        ),
        (
            {
                "n_clicks": 1,
                "download_filetype": 'CSV',
                "migration_study_rate": 10,
                "migration_study_rate_error": False,
                "migration_frequency": "Weeks",
                "migration_period_length": 100,
                "migration_period_length_error": False
            },
            False,
            [['Compound', [233, 846]], ['TA Code in Directory', ['gt', 'DN']]],
            "PYT14[Weekly With Filters]"
        )
    ],
    ids=["PYT14[Raise PreventUpdate None Filters]", "PYT14[Daily Without Filters]",
         "PYT14[Raise PreventUpdate Invalid Period Length]", "PYT14[Raise PreventUpdate Invalid Study Rate]",
         "PYT14[Daily With Filters]", "PYT14[Weekly Without Filters]", "PYT14[Weekly With Filters]"]
)
def test_export_migration_to_download(kwargs_input: dict, should_prevent_update, active_filters, test_id,
                                      app_test_configuration, app_test_mode_store, record_xml_attribute):

    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", test_id)
    record_xml_attribute("srs_requirement", "")
    record_xml_attribute("frs_requirement", "")
    record_xml_attribute("scenario", "")

    record_xml_attribute("purpose", "Verify the behavior of the export_migration_to_download callback in different scenarios, including cases where certain conditions should prevent the update.")
    record_xml_attribute("description", "Tests different scenarios by providing input arguments and checks whether the callback raises a PreventUpdate exception or returns the expected migration and configuration data.")
    record_xml_attribute("acceptance_criteria", "Raise a PreventUpdate exception if required, otherwise assert that the migration and configuration data is as expected (disregarding filenames).")

    app_test_session_store = {
        'timeframe_start': app_test_configuration.timeframe_start.isoformat(),
        'timeframe_end': app_test_configuration.timeframe_end.isoformat(),
        'milestones': {
            "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 4": {"label": "Milestone 4", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 5": {"label": "Milestone 5", "offset_before": 7, "offset_after": 7, "active": True},
        },
        'first_start': False,
        'active_filters': active_filters
    }

    kwargs_input["app_session_store"] = app_test_session_store
    kwargs_input["app_mode_store"] = app_test_mode_store

    if should_prevent_update:
        with pytest.raises(PreventUpdate):
            export_migration_to_download(**kwargs_input)
    else:

        migration_data, config_data = export_migration_to_download(**kwargs_input)
        dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(dir_path, 'data', 'dataframes', f'{test_id}_migration.json'), 'r') as source:
            expected_migration_data = json.load(source)

        with open(os.path.join(dir_path, 'data', 'dataframes', f'{test_id}_config.json'), 'r') as source:
            expected_config_data = json.load(source)

        del expected_migration_data['filename']
        del migration_data['filename']
        del expected_config_data['filename']
        del config_data['filename']

        migration_data['content'] = migration_data['content'].replace('\r\n', '\n')
        config_data['content'] = config_data['content'].replace('\r\n', '\n')

        assert expected_migration_data == migration_data
        assert expected_config_data == config_data


@pytest.mark.parametrize(
    "kwargs_input, expected_table_values, expected_period_lengths, should_prevent_update, active_filters, test_id",
    [
        (
            {
                "n_clicks": None,
                "migration_study_rate": None,
                "migration_study_rate_error": False,
                "migration_frequency": None,
                "migration_period_length": None,
                "migration_period_length_error": False,
                "group_col": None
            },
            {},
            [],
            True,
            [],
            "PYT15[Raise PreventUpdate None Inputs]"
        ),
        (
            {
                "n_clicks": 1,
                "migration_study_rate": -1,
                "migration_study_rate_error": True,
                "migration_frequency": 'Days',
                "migration_period_length": 100,
                "migration_period_length_error": False,
                "group_col": 'Overall'
            },
            {},
            [],
            True,
            [],
            "PYT15[Raise PreventUpdate Invalid Study Rate]"
        ),
        (
                {
                    "n_clicks": 1,
                    "migration_study_rate": 14,
                    "migration_study_rate_error": False,
                    "migration_frequency": 'Days',
                    "migration_period_length": -1,
                    "migration_period_length_error": True,
                    "group_col": 'Overall'
                },
                {},
                [],
                True,
                [],
                "PYT15[Raise PreventUpdate Invalid Period Length]"
        ),
        (
            {
                "n_clicks": 1,
                "migration_study_rate": 2,
                "migration_study_rate_error": False,
                "migration_frequency": 'Days',
                "migration_period_length": 100,
                "migration_period_length_error": False,
                "group_col": 'Overall'
            },
            {
                "expected_cols": [
                    'Total Moved', 'Total Not Moved', 'Period 1', 'Period 2', 'Period 3', 'Period 4'
                ],
                "expected_rows": 1
            },
            [
                ('Start: 2020-01-01', 'End: 2020-04-09'), ('Start: 2020-04-10', 'End: 2020-07-18'),
                ('Start: 2020-07-19', 'End: 2020-10-26'), ('Start: 2020-10-27', 'End: 2021-01-01')
            ],
            False,
            [],
            "PYT15[Daily Without Grouping Column]"
        ),
        (
            {
                "n_clicks": 1,
                "migration_study_rate": 2,
                "migration_study_rate_error": False,
                "migration_frequency": 'Days',
                "migration_period_length": 100,
                "migration_period_length_error": False,
                "group_col": 'TA Code in Directory'
            },
            {
                "expected_cols": [
                    'TA Code in Directory', 'Total Moved', 'Total Not Moved', 'Period 1', 'Period 2', 'Period 3',
                    'Period 4'
                ],
                "expected_rows": 6
            },
            [
                ('Start: 2020-01-01', 'End: 2020-04-09'), ('Start: 2020-04-10', 'End: 2020-07-18'),
                ('Start: 2020-07-19', 'End: 2020-10-26'), ('Start: 2020-10-27', 'End: 2021-01-01')
            ],
            False,
            [],
            "PYT15[Daily With Grouping Column]"
        ),
        (
            {
                "n_clicks": 1,
                "migration_study_rate": 1,
                "migration_study_rate_error": False,
                "migration_frequency": 'Weeks',
                "migration_period_length": 12,
                "migration_period_length_error": False,
                "group_col": 'Overall'
            },
            {
                "expected_cols": [
                    'Total Moved', 'Total Not Moved', 'Period 1', 'Period 2', 'Period 3', 'Period 4',
                    'Period 5'
                ],
                "expected_rows": 1
            },
            [
                ('Start: 2020-01-05', 'End: 2020-03-28'), ('Start: 2020-03-29', 'End: 2020-06-20'),
                ('Start: 2020-06-21', 'End: 2020-09-12'), ('Start: 2020-09-13', 'End: 2020-12-05'),
                ('Start: 2020-12-06', 'End: 2020-12-26')
            ],
            False,
            [],
            "PYT15[Weekly Without Grouping Column]"
        ),
        (
            {
                "n_clicks": 1,
                "migration_study_rate": 1,
                "migration_study_rate_error": False,
                "migration_frequency": 'Weeks',
                "migration_period_length": 12,
                "migration_period_length_error": False,
                "group_col": 'TA Code in Directory'
            },
            {
                "expected_cols": [
                    'TA Code in Directory', 'Total Moved', 'Total Not Moved', 'Period 1', 'Period 2', 'Period 3',
                    'Period 4', 'Period 5'
                ],
                "expected_rows": 6
            },
            [
                ('Start: 2020-01-05', 'End: 2020-03-28'), ('Start: 2020-03-29', 'End: 2020-06-20'),
                ('Start: 2020-06-21', 'End: 2020-09-12'), ('Start: 2020-09-13', 'End: 2020-12-05'),
                ('Start: 2020-12-06', 'End: 2020-12-26')
            ],
            False,
            [],
            "PYT15[Weekly With Grouping Column]"
        ),
    ],
    ids=["PYT15[Raise PreventUpdate None Inputs]", "PYT15[Raise PreventUpdate Invalid Study Rate]",
         "PYT15[Raise PreventUpdate Invalid Period Length]", "PYT15[Daily Without Grouping Column]",
         "PYT15[Daily With Grouping Column]", "PYT15[Weekly Without Grouping Column]",
         "PYT15[Weekly With Grouping Column]"]
)
def test_update_migration_table(kwargs_input: dict, expected_table_values, expected_period_lengths,
                                should_prevent_update, active_filters, test_id, app_test_mode_store,
                                app_test_configuration, record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", test_id)
    record_xml_attribute("srs_requirement", "")
    record_xml_attribute("frs_requirement", "")
    record_xml_attribute("scenario", "")

    record_xml_attribute("purpose",
                         "Ensure that the update_migration_table callback produces the expected results in terms of "
                         "table values, period lengths, and preventing updates under certain conditions.")
    record_xml_attribute("description",
                         "Verify the behavior of the update_migration_table callback with different input scenarios.")
    record_xml_attribute("acceptance_criteria",
                         "Raise a PreventUpdate exception when should_prevent_update is True. "
                         "Otherwise, verify that the generated migration table has the expected number of "
                         "columns and rows, the column names match the expected values, and the period lengths are "
                         "correctly represented in the period lengths section.")


    app_test_session_store = {
        'timeframe_start': pd.Timestamp(year=2020, month=1, day=1).isoformat(),
        'timeframe_end': pd.Timestamp(year=2021, month=1, day=1).isoformat(),
        'milestones': {
            "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 4": {"label": "Milestone 4", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 5": {"label": "Milestone 5", "offset_before": 7, "offset_after": 7, "active": True},
        },
        'first_start': False,
        'active_filters': active_filters
    }

    kwargs_input["app_session_store"] = app_test_session_store
    kwargs_input["app_mode_store"] = app_test_mode_store

    if should_prevent_update:
        with pytest.raises(PreventUpdate):
            update_migration_table(**kwargs_input)
    else:
        migration_table, period_lengths_section = update_migration_table(**kwargs_input)

        assert len(migration_table.columns) == len(expected_table_values['expected_cols'])
        assert all([col['name'] in expected_table_values['expected_cols'] for col in migration_table.columns])
        assert len(migration_table.data) == expected_table_values['expected_rows']

        assert len(period_lengths_section.children[1].children) == len(expected_period_lengths)

        for i, div_ in enumerate(period_lengths_section.children[1]):
            start_date_span = div_.children[1]
            end_date_span = div_.children[3]

            assert (start_date_span.children[0], end_date_span.children[0]) == expected_period_lengths[i]


def test_update_migration_table_no_active_milestones(app_test_mode_store, record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", "PYT16")
    record_xml_attribute("srs_requirement", "")
    record_xml_attribute("frs_requirement", "")
    record_xml_attribute("scenario", "")

    record_xml_attribute("purpose",
                         "Ensure that the update_migration_table correctly raises a Prevent Update exception when "
                         "no milestones are active.")
    record_xml_attribute("description",
                         "Verify the behavior of the update_migration_table callback under a no active milestones "
                         "scenario.")
    record_xml_attribute("acceptance_criteria",
                         "Assert that a PreventUpdate exception is raised when all milestones are marked active = "
                         "false in the test_app_session_store ")

    app_test_session_store = {
        'timeframe_start': pd.Timestamp(year=2020, month=1, day=1).isoformat(),
        'timeframe_end': pd.Timestamp(year=2021, month=1, day=1).isoformat(),
        'milestones': {
            "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": False},
            "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": False},
            "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": False},
            "Milestone 4": {"label": "Milestone 4", "offset_before": 14, "offset_after": 14, "active": False},
            "Milestone 5": {"label": "Milestone 5", "offset_before": 7, "offset_after": 7, "active": False},
        },
        'first_start': False,
        'active_filters': []
    }

    kwargs_input = {
        "n_clicks": None,
        "migration_study_rate": None,
        "migration_study_rate_error": False,
        "migration_frequency": None,
        "migration_period_length": None,
        "migration_period_length_error": False,
        "group_col": None,
        "app_session_store": app_test_session_store,
        "app_mode_store": app_test_mode_store
    }

    with pytest.raises(PreventUpdate):
        update_migration_table(**kwargs_input)


def test_set_maximum_period_length(record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", "PYT17")
    record_xml_attribute("srs_requirement", "")
    record_xml_attribute("frs_requirement", "")
    record_xml_attribute("scenario", "")

    record_xml_attribute("purpose",
                         "The purpose of this unit test is to ensure that the set_maximum_period_length "
                         "callback correctly calculates the maximum period length based on the given "
                         "session store and transfer window type.")
    record_xml_attribute("description",
                         "This unit test verifies the correctness of the set_maximum_period_length "
                         "callback by checking the expected maximum period lengths for both daily "
                         "and weekly transfer windows.")
    record_xml_attribute("acceptance_criteria",
                         "Assert that the expected maximum period length for daily transfer windows is 32 and the "
                         "expected maximum period length for weekly transfer windows is 3 when using the "
                         "provided session store.")

    app_test_session_store = {
        'timeframe_start': pd.Timestamp(year=2020, month=1, day=1).isoformat(),
        'timeframe_end': pd.Timestamp(year=2020, month=2, day=1).isoformat(),
        'milestones': {
            "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 4": {"label": "Milestone 4", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 5": {"label": "Milestone 5", "offset_before": 7, "offset_after": 7, "active": True},
        },
        'first_start': False
    }

    expected_daily_max = set_maximum_period_length(app_test_session_store, 'Days')
    assert expected_daily_max == 32

    expected_weekly_max = set_maximum_period_length(app_test_session_store, 'Weeks')
    assert expected_weekly_max == 3
