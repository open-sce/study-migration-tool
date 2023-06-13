from enum import Enum, unique, auto

import pandas as pd

from config import AppConfiguration
from logger import logger
from utils import Milestone


class Timeblock:
    def __init__(self, start, end, milestone):
        self.start: pd.Timestamp = start
        self.end: pd.Timestamp = end
        self.milestone = milestone

    def __eq__(self, other):
        if isinstance(other, Timeblock):
            return self.start == other.start and \
                self.end == other.end and \
                self.milestone == other.milestone

    def __repr__(self):
        return f"Timeblock(" \
               f"start={repr(self.start)}," \
               f"end={repr(self.end)}," \
               f"milestone={repr(self.milestone)}" \
               f")"


@unique
class ItemStatus(Enum):
    """
    Used to define the status' an item can hold.
    """
    CLOSING_BEFORE_TIMEFRAME = auto()
    STARTING_AFTER_TIMEFRAME = auto()
    NO_ACTIVITY_OVER_TIMEFRAME = auto()

    CLOSING_AFTER_TIMEFRAME = auto()
    CLOSING_DURING_TIMEFRAME = auto()

    NO_STATUS_GIVEN = auto()


class Data:
    """
    Holds the dataframe used for the Dash app displays.
    Functions required to compute the read in data into a suitable df.
    """

    def __init__(self, app_configuration: AppConfiguration):

        self.unique_identity_label = app_configuration.unique_identity_label
        self.study_label = app_configuration.study_label
        self.compound_label = app_configuration.compound_label
        self.milestone_labels = list(app_configuration.milestone_definitions.keys())

        self.df = pd.read_csv(app_configuration.data_path, parse_dates=self.milestone_labels,
                              dayfirst=app_configuration.day_first_dates)
        self.plot_df = pd.DataFrame(columns=[
            self.unique_identity_label, self.study_label, self.compound_label,
            "start", "end", "type", "inside timeframe"
        ])

    def create_timeblock_apply(self, series, milestones):
        """
        Function that is applied to each milestone column in the main df. Used to compute base 'timeblock' for each
        row/item.
        A Timeblock is an object with start date, end date and milestone type attributes. Representing periods of
        activity.
        """

        timeblock_lst = []

        for col in series.index:
            timestamp = series[col]
            if pd.isna(timestamp):
                continue

            milestone = milestones[col]
            start_timestamp, end_timestamp = milestone.apply_offsets(timestamp)

            timeblock_lst.append(Timeblock(start=start_timestamp, end=end_timestamp, milestone=milestone))

        return timeblock_lst  # for a single study

    def create_plotting_df_apply(self, df, timeframe_start: pd.Timestamp, timeframe_end: pd.Timestamp):

        id_lst, study_lst, compound_lst = [], [], []
        start_date_lst, end_date_lst, type_lst, inside_timeframe_flag_lst = [], [], [], []
        id_num, study, compound = None, None, None

        for col in df.columns:
            if col == self.study_label:
                study = df[col].values[0]
            elif col == self.compound_label:
                compound = df[col].values[0]
            elif col == self.unique_identity_label:
                id_num = df[col].values[0]
            elif col == "Time Block":
                timeblock_lst = df[col].values[0]
                for timeblock in timeblock_lst:
                    id_lst.append(id_num)
                    study_lst.append(study)
                    compound_lst.append(compound)

                    start_date_lst.append(timeblock.start)
                    end_date_lst.append(timeblock.end)
                    type_lst.append(timeblock.milestone.label)

                    latest_start = max(timeframe_start, timeblock.start)
                    earliest_end = min(timeframe_end, timeblock.end)
                    delta = (earliest_end - latest_start).days + 1
                    overlap = max(0, delta)
                    inside_timeframe_flag_lst.append("Yes" if overlap > 0 else "No")

        data_dict = {
            self.unique_identity_label: id_lst,
            self.study_label: study_lst,
            self.compound_label: compound_lst,
            "start": start_date_lst,
            "end": end_date_lst,
            "type": type_lst,
            "inside timeframe": inside_timeframe_flag_lst
        }

        return pd.DataFrame.from_dict(data=data_dict)

    def create_timeblock(self, app_session_store: dict) -> None:
        logger.info(f"Creating timeblocks")

        milestones = {label: Milestone(**milestone_dict) for label, milestone_dict in app_session_store['milestones'].items()}

        active_milestones = [milestone.label for milestone in milestones.values() if milestone.active]
        if active_milestones:
            self.df["Time Block"] = self.df[active_milestones].apply(
                lambda series: self.create_timeblock_apply(series, milestones), axis=1
            )

    def create_plotting_df(self, app_session_store: dict) -> None:
        logger.info(f"Creating plotting_df")

        timeframe_start = pd.Timestamp(app_session_store['timeframe_start'])
        timeframe_end = pd.Timestamp(app_session_store['timeframe_end'])

        if "Time Block" not in self.df.columns:
            return

        plot_loc = self.df[[self.study_label, self.compound_label, self.unique_identity_label, "Time Block"]]
        plot_group = plot_loc.groupby(self.unique_identity_label, group_keys=False)
        self.plot_df = plot_group.apply(
            self.create_plotting_df_apply, timeframe_start, timeframe_end
        ).reset_index(drop=True)
