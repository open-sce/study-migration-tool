import datetime
import json

import pandas as pd
import numpy as np
import re
from typing import List, Union
from cryptography.fernet import Fernet

from config import AppConfiguration
from logger import logger
from utils import Milestone, Timeblock, MergedTimeblock, Gap, ItemStatus, ItemInformation, pairwise


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
        self.day_weight_coefficient = app_configuration.day_weight_coefficient
        self.gap_weight_coefficient = app_configuration.gap_weight_coefficient
        self.fernet_instance = Fernet(Fernet.generate_key())

        source_name = app_configuration.data_path.split("/")[-1]

        if ".csv" in source_name:
            self.df = pd.read_csv(app_configuration.data_path, parse_dates=self.milestone_labels,
                                  dayfirst=app_configuration.day_first_dates)
        elif any(file_type in source_name for file_type in ["xlsx"]):
            self.df = pd.read_excel(app_configuration.data_path, parse_dates=self.milestone_labels)
        else:
            raise ValueError(f"Specified source path points to file: {source_name}, of unsupported type! \n "
                             "Supported types are: csv and xlsx.")

    def encrypt_item(self, item: Union[str, list, dict, pd.DataFrame]) -> str:
        if isinstance(item, pd.DataFrame):
            json_str = item.to_json()
        else:
            json_str = json.dumps(item)
        return self.fernet_instance.encrypt(json_str.encode()).decode()

    def decrypt_item(self, item: str, expect_dataframe: bool = False) -> Union[str, list, dict, pd.DataFrame]:
        if expect_dataframe:
            return pd.read_json(self.fernet_instance.decrypt(item.encode()).decode())
        else:
            return json.loads(self.fernet_instance.decrypt(item.encode()).decode())

    @staticmethod
    def _create_timeblock_apply(series: pd.Series, milestones: dict) -> list:
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

            timeblock_lst.append(
                Timeblock(start=start_timestamp, end=end_timestamp, milestone=milestone, timestamp=timestamp)
            )

        return timeblock_lst

    def create_timeblock(self, app_session_store: dict) -> pd.DataFrame:
        """
        Create timeblocks based on active milestones.

        Parameters:
            app_session_store: A dictionary containing session information, including milestones.

        Returns:
            pd.DataFrame: A DataFrame with a 'Time Block' column created based on the active milestones.
        """

        logger.debug(f"Creating timeblocks")
        timeblock_df = self.df.copy()

        milestones = {label: Milestone(**milestone_dict) for label, milestone_dict in
                      app_session_store['milestones'].items()}
        active_milestones = [milestone.label for milestone in milestones.values() if milestone.active]

        if active_milestones:
            timeblock_df["Time Block"] = timeblock_df[active_milestones].apply(
                lambda series: self._create_timeblock_apply(series, milestones), axis=1
            )
        return timeblock_df

    def _create_plotting_df_apply(self, df, timeframe_start: pd.Timestamp, timeframe_end: pd.Timestamp) -> pd.DataFrame:
        """
        Generates a pandas DataFrame containing data for plotting based on the given dataframe,
        start and end timestamps.

        Parameters:
            self: An instance of the class.
            df (pd.DataFrame): The input dataframe containing relevant columns.
            timeframe_start (pd.Timestamp): The start timestamp of the desired timeframe.
            timeframe_end (pd.Timestamp): The end timestamp of the desired timeframe.

        Returns:
            pd.DataFrame: The resulting dataframe with columns 'unique_identity_label',
            'study_label', 'compound_label', 'start', 'end', 'type', 'inside timeframe'.

        Notes:
            - The function iterates over the columns of the input dataframe and extracts relevant information.
            - The function checks if the column matches specific labels and assigns corresponding values to variables.
            - It then iterates over the 'Time Block' column values and appends data to respective lists based on
              matching conditions.
            - It calculates the overlap between the desired timeframe and each time block.
            - Finally, it constructs a dictionary from the collected data and converts it into a pandas DataFrame.

        """

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

    def create_plotting_df(self, timeblock_df: pd.DataFrame, app_session_store: dict) -> pd.DataFrame:
        """
        Creates a pandas DataFrame for plotting based on the provided time block DataFrame and application session
        store containing UI configurations.

        Parameters:
            timeblock_df (pd.DataFrame): A DataFrame containing time block data.
            app_session_store (dict): A dictionary containing the application session store.

        Returns:
            pd.DataFrame: A DataFrame containing the data for plotting.

        Description:
            This method generates a pandas DataFrame for plotting based on the given time block DataFrame and
            application session store. It extracts relevant information from the time block DataFrame and calculates
            the overlap between the desired timeframe and each time block.

            The method performs the following steps:

            1. If the "Time Block" column is not present in the time block DataFrame, an empty DataFrame is returned.
            2. The start and end of the active timeframe are obtained from the application session store.
            4. The DataFrame is grouped by the unique identity label.
            5. The `_create_plotting_df_apply` function is applied to the groupby
            6. The resulting DataFrame's index is reset and returned.
        """

        logger.debug(f"Creating plotting_df")

        if "Time Block" not in timeblock_df.columns:
            return pd.DataFrame()

        timeframe_start = pd.Timestamp(app_session_store['timeframe_start'])
        timeframe_end = pd.Timestamp(app_session_store['timeframe_end'])

        plot_loc = timeblock_df[[self.study_label, self.compound_label, self.unique_identity_label, "Time Block"]]
        plot_group = plot_loc.groupby(self.unique_identity_label, group_keys=False)
        return plot_group.apply(
            self._create_plotting_df_apply, timeframe_start, timeframe_end
        ).reset_index(drop=True)

    @staticmethod
    def _merge_timeblock_apply(timeblock_lst: List[Timeblock]) -> list:
        """
        Combines overlapping time blocks into single time blocks.

        Parameters:
            timeblock_lst (List[Timeblock]): A list of Timeblock objects to be merged.

        Returns:
            list: A list of merged time blocks.

        Description:
            This method takes a list of Timeblock objects and merges any overlapping time blocks into single time
            blocks.
            If multiple time blocks overlap, the milestone/timestamp attribute is transformed into a list containing
            all the milestones/timestamps of the intersecting time blocks.

            The method iterates through the time blocks and combines them as follows:

            1. If there is only one time block in the list, it is returned as is.
            2. For each pair of adjacent time blocks, the method checks if they overlap.
            3. If the time blocks overlap, their milestones and timestamps are appended to the current merging block.
               The merging block's end time is updated if necessary.
            4. If the time blocks do not overlap, the current merging block is appended to the merged time block list,
               and a new merging block is started.
            5. The process continues until all time blocks have been processed.

        Example:
            Input: [
                Timeblock(
                    start=pd.Timestamp('2023-01-5'), end=pd.Timestamp('2023-01-15'),
                    milestone=Milestone('A', 5, 5), timestamp=pd.Timestamp('2023-01-10')
                ),

                Timeblock(
                    start=pd.Timestamp('2023-01-7'), end=pd.Timestamp('2023-01-17'),
                    milestone=Milestone('B', 5, 5), timestamp=pd.Timestamp('2023-01-12')
                ),

                Timeblock(
                    start=pd.Timestamp('2023-02-01'), end=pd.Timestamp('2023-02-11'),
                    milestone=Milestone('C', 5, 5), timestamp=pd.Timestamp('2023-02-06')
                )
            ]

            Output: [
                MergedTimeblock(
                    start=pd.Timestamp('2023-01-5'), end=pd.Timestamp('2023-01-17'),
                    milestones=['A', 'B'], timestamps=[
                        pd.Timestamp('2023-01-10'), pd.Timestamp('2023-01-12')
                    ]
                ),

                Timeblock(
                    start=pd.Timestamp('2023-02-01'), end=pd.Timestamp('2023-02-11'),
                    milestone=Milestone('C', 5, 5), timestamp=pd.Timestamp('2023-02-06')
                )
            ]
        """

        merged_timeblock_lst = []
        block_type_lst = []
        block_timestamp_lst = []
        block_start, block_end = None, None

        def append_to_typeblock_list(start: pd.Timestamp, end: pd.Timestamp, type_lst: list, timestamp_lst: list,
                                     timeblocks: list) -> None:
            if len(timestamp_lst) == 1:
                timeblocks.append(Timeblock(start=start, end=end, milestone=type_lst[0], timestamp=timestamp_lst[0]))
            else:
                timeblocks.append(MergedTimeblock(start=start, end=end, milestones=type_lst, timestamps=timestamp_lst))

        if not timeblock_lst:
            return []

        if len(timeblock_lst) < 2:  # single block case
            return timeblock_lst

        for tb_now, tb_future in pairwise(timeblock_lst):

            if block_start is None:  # if active timeblock is empty
                block_start = tb_now.start
                block_end = tb_now.end
                block_type_lst.append(tb_now.milestone)
                block_timestamp_lst.append(tb_now.timestamp)

            if tb_future is None:  # last pair
                if tb_now.end > block_end:
                    # if pending merged block end is before this timeblock, we end the block with the
                    # current timeblock end
                    block_end = tb_now.end
                append_to_typeblock_list(block_start, block_end, block_type_lst, block_timestamp_lst,
                                         merged_timeblock_lst)
            elif tb_future.start <= tb_now.end:
                block_type_lst.append(tb_future.milestone)
                block_timestamp_lst.append(tb_future.timestamp)
                if not block_end > tb_future.end:
                    block_end = tb_future.end
            else:  # not overlapping, so end block
                append_to_typeblock_list(block_start, block_end, block_type_lst, block_timestamp_lst,
                                         merged_timeblock_lst)
                block_type_lst = []
                block_timestamp_lst = []
                block_start, block_end = None, None

        return merged_timeblock_lst

    def merge_timeblocks(self, timeblock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges overlapping time block objects in a pandas DataFrame into MergedTimeblock instances.

        Parameters:
            timeblock_df (pd.Dataframe): Contains the timeblock objects to be merged in a column called "Time Block".

        Returns:
            pd.Dataframe: Containing MergedTimeblocks in a column called "Merged Time Block"
        """
        logger.debug("Merging timeblocks")

        if "Time Block" in timeblock_df.columns:
            timeblock_df["Merged Time Block"] = timeblock_df["Time Block"].apply(
                lambda x: self._merge_timeblock_apply(x))

        return timeblock_df

    @staticmethod
    def _generate_gap_information_apply(study_id: str, merged_timeblock_lst: list, start_date: pd.Timestamp,
                                        end_date: pd.Timestamp) -> ItemInformation:

        """
        Generate gap information for a specific study.

        This function uses a list of time block objects to crate summary information about the gaps between
        the time blocks for a specific study. The generated gap information is returned as an `ItemInformation`
        object.

        Parameters:
            study_id (str): The identifier of the study.
            merged_timeblock_lst (list): The list of merged time block objects.
            start_date (pd.Timestamp): The start date of the time frame.
            end_date (pd.Timestamp): The end date of the time frame.

        Returns:
            ItemInformation: An `ItemInformation` object containing the generated gap information.

        Raises:
            AssertionError: If an item status is not set.

        Notes:
            The primary logical loop iterates over pairs of adjacent time block objects to identify gaps in activity.

            If a study is found to contain no activity throughout the entire timeframe. A single Gap object is created
            representing the entire timeframe with an INCLUSIVE start and end date.

            If a gap is found to intersect with the start or end of the timeframe, the start or end of the gap will
            INCLUSIVELY be set to the start or end of the timeframe.

            Gaps with a start or end date that do not intersect the timeframe start or end will have a start or end date
            that is EXCLUSIVE of the neighbouring timeblocks

        Todo:
            - Potentially remove timeblocks before and after the timeframe here (optimisation)
        """

        gap_number, gap_day_total = 0, 0
        gap_lst = []
        found_timeframe_start_flag = False
        status = None

        tb_lst = sorted(merged_timeblock_lst, key=lambda x: x.start)  # ensure timeblock list is in chronological order
        timeframe_date_range = pd.date_range(start=start_date, end=end_date)

        if not merged_timeblock_lst:
            return ItemInformation(
                study_id=study_id,
                gap_number=1,
                gap_day_total=len(timeframe_date_range),
                gap_lst=[Gap(
                    start=start_date,
                    end=end_date,
                    timestamp_lst=set(timeframe_date_range.date)
                )],
                active_during_timeframe=False,
                status=ItemStatus.NO_STATUS_GIVEN
            )

        if tb_lst[0].start > end_date:
            # fully after timeframe
            return ItemInformation(
                study_id=study_id,
                gap_number=1,
                gap_day_total=len(timeframe_date_range),
                gap_lst=[Gap(
                    start=start_date,
                    end=end_date,
                    timestamp_lst=set(timeframe_date_range.date)
                )],
                active_during_timeframe=False,
                status=ItemStatus.STARTING_AFTER_TIMEFRAME
            )
        elif tb_lst[-1].end < start_date:
            # fully before timeframe
            return ItemInformation(
                study_id=study_id,
                gap_number=1,
                gap_day_total=len(timeframe_date_range),
                gap_lst=[Gap(
                    start=start_date,
                    end=end_date,
                    timestamp_lst=set(timeframe_date_range.date)
                )],
                active_during_timeframe=False,
                status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
            )
        else:
            active_during_timeframe = True
            timeblock_contains_timeframe_start_flag = False
            timeblock_contains_timeframe_end_flag = False

            for index, timeblock in enumerate(tb_lst):

                if (timeblock.start == start_date) and (timeblock.end == end_date):
                    # the timeblock is exactly equal to the timeframe. We deal with this separately as the logic below
                    # does not handle this specific case.
                    return ItemInformation(
                        study_id=study_id,
                        gap_number=1,
                        gap_day_total=len(timeframe_date_range),
                        gap_lst=[Gap(
                            start=start_date,
                            end=end_date,
                            timestamp_lst=set(timeframe_date_range.date)
                        )],
                        active_during_timeframe=True,
                        status=ItemStatus.CLOSING_AFTER_TIMEFRAME
                    )

                if (timeblock.start <= start_date) and (timeblock.end >= start_date):
                    # if the timeframe start is inside one of the timeblocks
                    timeblock.is_timeframe_start = True
                    timeblock.start = start_date
                    tb_lst[index] = timeblock
                    timeblock_contains_timeframe_start_flag = True

                if (timeblock.end >= end_date) and (timeblock.start <= end_date):
                    # if one of the timeblock ends is after the timeframe
                    timeblock.is_timeframe_end = True
                    timeblock.end = end_date
                    tb_lst[index] = timeblock
                    status = ItemStatus.CLOSING_AFTER_TIMEFRAME
                    timeblock_contains_timeframe_end_flag = True

                elif index == len(tb_lst) - 1:
                    if timeblock.end <= end_date:
                        status = ItemStatus.CLOSING_DURING_TIMEFRAME
                    else:
                        status = ItemStatus.CLOSING_AFTER_TIMEFRAME

            if not timeblock_contains_timeframe_start_flag:
                timeblock_timeframe_start = Timeblock(
                    start=start_date, end=start_date,
                    milestone=Milestone(label='Timeframe Start', offset_before=0, offset_after=0),
                    timestamp=start_date)
                timeblock_timeframe_start.is_timeframe_start = True

                tb_lst.append(timeblock_timeframe_start)

            if not timeblock_contains_timeframe_end_flag:
                timeblock_timeframe_end = Timeblock(
                    start=end_date, end=end_date,
                    milestone=Milestone(label='Timeframe End', offset_before=0, offset_after=0),
                    timestamp=end_date)
                timeblock_timeframe_end.is_timeframe_end = True

                tb_lst.append(timeblock_timeframe_end)

            tb_lst = sorted(tb_lst, key=lambda x: x.start)  # ensure timeblock list is in chronological order

            for tb_now, tb_next in pairwise(tb_lst):

                if tb_now.is_timeframe_end:
                    break

                elif tb_now.is_timeframe_start:
                    found_timeframe_start_flag = True

                    if (tb_now.end <= start_date) and (tb_next.start >= end_date):
                        # the gap is over the total length of the timeframe

                        status = ItemStatus.NO_ACTIVITY_OVER_TIMEFRAME
                        gap_date_range = pd.date_range(start=start_date, end=end_date)
                        gap = Gap(start=start_date, end=end_date, timestamp_lst=set(gap_date_range.date))
                        gap_number = 1
                        gap_day_total += len(gap_date_range)
                        gap_lst = [gap]
                        break

                if found_timeframe_start_flag:
                    # Inclusive for timeframe start end but not timeblocks

                    # tagged as timeframe end BUT that timeblock start is not necessarily = timeframe end
                    if tb_next.start == end_date:
                        # we do include the first day of the timeblock when it is the timeframe end, but... (-> else)
                        gap_end = tb_next.start
                    else:
                        # we do not want to include the first day/week of a timeblock when calculating gap lengths
                        gap_end = tb_next.start - pd.to_timedelta(1, unit='d')

                    if tb_now.end == start_date:
                        gap_start = tb_now.end
                    else:
                        gap_start = tb_now.end + pd.to_timedelta(1, unit='d')

                    gap_date_range = pd.date_range(start=gap_start, end=gap_end)
                    gap = Gap(start=gap_start, end=gap_end, timestamp_lst=set(gap_date_range.date))
                    gap_number += 1
                    gap_day_total += len(gap_date_range)
                    gap_lst.append(gap)

            assert status is not None, "Status must be set!"

            item_information = ItemInformation(
                study_id=study_id,
                gap_number=gap_number,
                gap_day_total=gap_day_total,
                gap_lst=gap_lst,
                active_during_timeframe=active_during_timeframe,
                status=status
            )

            return item_information

    def generate_gap_information(self, merged_timeblock_df: pd.DataFrame, app_session_store: dict) -> pd.DataFrame:
        """
        Generates gap information for each study in the merged time block dataframe.
        The generated gap information is added as a new column named "Gap Information" in the dataframe.

        Parameters:
            merged_timeblock_df (pd.DataFrame): The merged time block dataframe. See merge_timeblocks for more
            information.
            app_session_store (dict): A dictionary containing the users current application configuration.

        Returns:
            pd.DataFrame: The modified merged time block dataframe with the added "Gap Information" column.
        """

        logger.debug(f"Generating gap information")

        timeframe_start = pd.Timestamp(app_session_store['timeframe_start'])
        timeframe_end = pd.Timestamp(app_session_store['timeframe_end'])

        if "Merged Time Block" in merged_timeblock_df.columns:
            merged_timeblock_df["Gap Information"] = list(
                map(
                    self._generate_gap_information_apply,
                    merged_timeblock_df[self.unique_identity_label],
                    merged_timeblock_df["Merged Time Block"],
                    [timeframe_start] * merged_timeblock_df.shape[0],
                    [timeframe_end] * merged_timeblock_df.shape[0]
                )
            )

        return merged_timeblock_df

    @staticmethod
    def _compute_weights_apply(item_info: ItemInformation) -> pd.Series:
        """
        Compute weights for studies/compounds based on their time frame status.

        This function sets the weight for all studies/compounds based on whether they are inside or outside
        the time frame. Studies/compounds outside the time frame are assigned a weight of 0, while those inside
        the time frame are assigned weights in the range (0, 1).

        Parameters:
            item_info (ItemInformation): The information of a study/compound.

        Returns:
            pd.Series: A pandas Series containing the computed day weight and gap weight.
        """

        day_weight = 0
        gap_weight = 0

        if item_info.status not in [ItemStatus.CLOSING_BEFORE_TIMEFRAME, ItemStatus.STARTING_AFTER_TIMEFRAME,
                                    ItemStatus.NO_STATUS_GIVEN]:
            day_weight = 1 if item_info.gap_day_total == 0 else 1 / item_info.gap_day_total
            gap_weight = 1 if item_info.gap_number == 0 else 1 / item_info.gap_number

        return pd.Series([day_weight, gap_weight])

    def compute_weights(self, gap_information_df: pd.DataFrame) -> None:
        """
        Compute weights for project transfers based on their priority and modify the DataFrame inplace.

        This function computes the weight required to order the project transfers in the DataFrame
        according to their priority. The computed weights are attached as a new column at the end
        of the DataFrame, and the DataFrame is sorted by descending weight.

        Parameters:
            gap_information_df (pd.DataFrame): The DataFrame containing study gap information.

        Returns:
            None.

        ToDo:
            - Include ability to influence weights from UI?
            - Allow for custom scaling, currently we set a 5x scaling on gap days and a 1/10 scaling on the gap number
            which prioritises gap days heavily over number of gaps
        """

        logger.debug("Generating weight values for studies")

        gap_information_df[['Day Weights', 'Gap Weights']] = gap_information_df["Gap Information"].apply(
            lambda item_info: self._compute_weights_apply(item_info))

        # Normalize the two types of weights we have
        gap_information_df['Day Weights'] = \
            (gap_information_df['Day Weights'] - gap_information_df['Day Weights'].min()) / \
            (gap_information_df['Day Weights'].max() - gap_information_df['Day Weights'].min())
        gap_information_df['Gap Weights'] = \
            (gap_information_df['Gap Weights'] - gap_information_df['Gap Weights'].min()) / \
            (gap_information_df['Gap Weights'].max() - gap_information_df['Gap Weights'].min())

        gap_information_df['Weights'] = (gap_information_df['Day Weights'] * self.day_weight_coefficient) + \
                                        (gap_information_df['Gap Weights'] * self.gap_weight_coefficient)

        # Remove the intermediate columns
        gap_information_df.drop(['Day Weights', 'Gap Weights'], axis=1, inplace=True)

        # scale the weights back to range(0,1)
        gap_information_df['Weights'] = \
            (gap_information_df['Weights'] - gap_information_df['Weights'].min()) / \
            (gap_information_df['Weights'].max() - gap_information_df['Weights'].min())

        gap_information_df.sort_values(by=['Weights'], inplace=True, ascending=False)

    @staticmethod
    def _timeframe_date_range(transfer_window_type: str, timeframe_start: pd.Timestamp,
                              timeframe_end: pd.Timestamp) -> list:
        """
        Returns a list of dates within the specified timeframe.

        Parameters:
            transfer_window_type (str): The type of transfer window ('D' for daily or other).
            timeframe_start (pd.Timestamp): The start date of the timeframe.
            timeframe_end (pd.Timestamp): The end date of the timeframe.

        Returns:
            A list dates within the specified timeframe.
        """

        if transfer_window_type == 'D':
            timeframe_date_list = pd.date_range(start=timeframe_start, end=timeframe_end,
                                                freq=transfer_window_type).date.tolist()
        else:
            # For the weekly view, we want the time frame to only contain whole weeks.
            # Therefore, we collapse the start and end to the nearest week (each week being anchored on its Sunday)
            if timeframe_start.weekday() != 6:
                starting_sunday = timeframe_start + pd.DateOffset(6 - timeframe_start.weekday())
            else:
                starting_sunday = timeframe_start
            if timeframe_end.weekday() == 6:  # if it lands on a sunday, need to go back to last week
                ending_sunday = timeframe_end - pd.DateOffset(7)
            elif timeframe_end.weekday() == 5:
                ending_sunday = timeframe_end - pd.DateOffset(6)
            else:
                ending_sunday = timeframe_end - pd.DateOffset(8 + timeframe_end.weekday())

            timeframe_date_list = pd.date_range(start=starting_sunday, end=ending_sunday,
                                                freq=transfer_window_type).date.tolist()

        return timeframe_date_list

    @staticmethod
    def _migrate_study_apply(study_info: ItemInformation, period_moving_dates: list, transfer_window_type: str) \
            -> pd.Series:
        """
        Applies migration logic to a study to locate and assign a transfer date (where possible).

        Parameters:
            study_info (ItemInformation): Information about the study.
            period_moving_dates (list): List of dictionaries representing the periods and their moving dates.
            transfer_window_type (str): The type of transfer window ('W' for weekly or other).

        Returns:
            pandas.Series: A series containing transfer details including transfer flag, period moved,
                           transfer start date, and transfer range.
        """

        transfer_flag = False
        period_moved = "Not Moved"
        transfer_start_date = pd.NaT
        transfer_range = ""

        for period_int, period_dates_dict in enumerate(period_moving_dates, 1):

            if not period_dates_dict:
                # If period is fully allocated (empty) continue to next one
                continue

            for gap in study_info.gap_lst:
                # if it is the weekly view, we want gaps to contain at least one whole week. So we round any gaps to
                # the nearest week (each week being anchored on its Sunday)

                if transfer_window_type == 'W':
                    if gap.start.weekday() != 6:
                        starting_sunday = gap.start + pd.DateOffset(6 - gap.start.weekday())
                    else:
                        starting_sunday = gap.start
                    if gap.end.weekday() == 6:
                        ending_sunday = gap.end - pd.DateOffset(7)
                    elif gap.end.weekday() == 5:
                        ending_sunday = gap.end - pd.DateOffset(6)
                    else:
                        ending_sunday = gap.end - pd.DateOffset(8 + gap.end.weekday())

                    if starting_sunday > ending_sunday:
                        # if the gap does not contain a full week then it does not count as a gap, so it is ignored
                        continue

                    study_gap_dates_set = set(
                        pd.date_range(start=starting_sunday, end=ending_sunday, freq=transfer_window_type).date
                    )
                else:
                    study_gap_dates_set = gap.timestamp_lst

                for date_ in period_dates_dict:
                    if date_ in study_gap_dates_set:
                        # found a transfer date
                        transfer_start_date = date_
                        transfer_flag = True
                        period_moved = f'Period {period_int}'
                        if transfer_window_type == 'W':
                            transfer_range = str(transfer_start_date) + ' to ' + str(
                                (transfer_start_date + pd.DateOffset(6)).date())

                        period_dates_dict[transfer_start_date] -= 1
                        break  # break from comparing gap dates with period

                if not pd.isna(transfer_start_date):
                    if period_dates_dict[transfer_start_date] == 0:
                        del period_dates_dict[transfer_start_date]
                    break  # from iterating over gap_list for study

            if not pd.isna(transfer_start_date):
                break  # break from iterating over period in period moving dates

        return pd.Series([transfer_flag, period_moved, transfer_start_date, transfer_range])

    def migration_table_processing(self, input_df: pd.DataFrame, app_session_store: dict, studies_per: int,
                                   period_length: int, transfer_window_type: str) -> tuple:
        """
        Takes in a dataframe of studies sorted by weights (See compute_weights) and attempts to assign each study a
        moving date on its first GAP DATE that aligns with an available date in period_moving_dates. Period moving dates
        have a capacity (specified by studies_per) which when filled will no longer accept additional studies.

        Parameters:
            input_df (Dataframe): Contains and has been sorted by computed weights
            app_session_store (dict): User-specific app configuration settings from UI
            studies_per (int): Active studies we can transfer per day/week
            period_length (int): Number of days/weeks per period
            transfer_window_type (str): 'W' or 'D' for day/week transfer type. Treat days or weeks as smallest unit.

        Returns:
            pandas.DataFrame: The final migration DataFrame with the following additional columns:
                'Transfer Flag': bool,
                'Transfer Start Date': timestamp or pd.NaT,
                'Transfer Range': string of from date - to date used when the transfer window is weekly,
                'Period Moved': string of period moved eg "Period 1" or "Not Moved".
            List[tuple]: List of tuples containing the start and end dates of the periods.
        """

        timeframe_start = pd.Timestamp(app_session_store['timeframe_start'])
        timeframe_end = pd.Timestamp(app_session_store['timeframe_end'])

        timeframe_date_list = self._timeframe_date_range(transfer_window_type, timeframe_start, timeframe_end)

        period_moving_dates = [
            {k: studies_per for k in timeframe_date_list[i:i + period_length]}
            for i in range(0, len(timeframe_date_list), period_length)
        ]

        if transfer_window_type == 'D':
            periods_start_end = [(list(period.keys())[0], list(period.keys())[-1]) for period in period_moving_dates]
        else:
            periods_start_end = [(list(period.keys())[0], (list(period.keys())[-1] + pd.DateOffset(6)).date()) for
                                 period in period_moving_dates]
        # [ ( start_date, end_date ), ( start_date, end_date ), ... ]

        input_df['Transfer Flag'] = False
        input_df['Transfer Start Date'] = pd.NaT
        input_df['Transfer Range'] = ''
        input_df['Period Moved'] = 'Not Moved'

        if studies_per == 0:
            return input_df, periods_start_end

        input_df[['Transfer Flag', 'Period Moved', 'Transfer Start Date', 'Transfer Range']] = \
            input_df["Gap Information"].apply(
                lambda study_info: self._migrate_study_apply(study_info, period_moving_dates, transfer_window_type))

        if transfer_window_type == 'D':
            input_df.drop(columns='Transfer Range', inplace=True)

        return input_df, periods_start_end

    @staticmethod
    def table_formatting(input_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Formats the migration DataFrame for presentation in a dash datatable.
        See migration_table_processing for more information on the format of this dataframe.

        Parameters:
            input_df (pandas.DataFrame): The DataFrame to be formatted.
            group_col (str): Column label to group by.

        Returns:
            pandas.DataFrame: The formatted DataFrame.
        """

        by_cols = ['Period Moved'] if group_col == 'Overall' else [group_col, 'Period Moved']
        grouped_df = input_df.groupby(by=by_cols, sort=True).size().reset_index(name='Transfer Dates Found')
        if group_col == 'Overall':
            grouped_df['Group'] = 'Overall'

        grouped_df = pd.pivot(
            grouped_df,
            index=group_col if group_col != 'Overall' else 'Group',
            columns='Period Moved',
            values='Transfer Dates Found'
        ).reset_index().rename(columns={grouped_df.index.name: group_col, **{'Not Moved': 'Transfer Dates Not Found'}})

        if 'Transfer Dates Not Found' not in grouped_df.columns:
            grouped_df['Transfer Dates Not Found'] = 0

        grouped_df['Transfer Dates Found'] = \
            grouped_df[grouped_df.columns[grouped_df.columns.str.startswith('Period')]].sum(axis=1)

        grouped_df['Total'] = grouped_df['Transfer Dates Found'] + grouped_df['Transfer Dates Not Found']

        df_cols = [group_col, 'Total', 'Transfer Dates Found', 'Transfer Dates Not Found'] \
            if group_col != 'Overall' \
            else ['Total', 'Transfer Dates Found', 'Transfer Dates Not Found']

        df_cols.extend(sorted([col for col in grouped_df.columns if re.match(r'\bPeriod\s+(\d+)', col)],
                              key=lambda x: int(x.replace('Period ', ''))))

        grouped_df = grouped_df.reindex(columns=df_cols)\
            .sort_values(by=['Transfer Dates Found'], ascending=False).fillna(0)

        float_cols = grouped_df.select_dtypes(include='float').columns
        grouped_df[float_cols] = grouped_df[float_cols].astype(np.int64)

        return grouped_df

    @staticmethod
    def format_for_export(migration_df: pd.DataFrame):
        """
        Formats the migration DataFrame for export.
        See migration_table_processing for more information on the format of this dataframe.

        Parameters:
            migration_df (pandas.DataFrame): The DataFrame containing imported data.

        Returns:
            pandas.DataFrame: The formatted DataFrame ready for export.
        """

        # Prevent column from transitioning to float dtype
        migration_df['Gap Day Total'] = 0

        for index, row in migration_df.iterrows():

            migration_df.loc[index, 'Closing Status'] = row['Gap Information'].status.value
            migration_df.loc[index, 'Gap Day Total'] = row['Gap Information'].gap_day_total

            for i, gap in enumerate(row['Gap Information'].gap_lst, 1):
                string_start = f'Gap {i} Start'
                string_end = f'Gap {i} End'
                string_len = f'Gap {i} Length'

                if string_start not in migration_df.columns:
                    migration_df[string_start] = None
                    migration_df[string_end] = None
                    migration_df[string_len] = None

                migration_df.loc[index, string_start] = str(gap.start.date())
                migration_df.loc[index, string_end] = str(gap.end.date())
                migration_df.loc[index, string_len] = len(gap.timestamp_lst)

        migration_df.drop(['Gap Information', 'Time Block', 'Merged Time Block'], axis=1, inplace=True)
        migration_df.sort_values(['Transfer Flag', 'Weights'], ascending=[False, False], inplace=True)
        migration_df.rename(columns={'Transfer Flag': 'Transfer Status'}, inplace=True)
        return migration_df

    @staticmethod
    def format_config_sheet(app_session_store: dict, period_start_end: List[tuple], period_length: int,
                            transfer_window_type: str, studies_per: int):
        """
        Creates a data frame containing a complete summary of the configuration used when exporting the migration table.

        Parameters:
            app_session_store (dict): The application session store containing UI information.
            period_start_end (list): A list of tuples representing the start and end dates of each period.
            period_length (int): The length of each period in days or weeks.
            transfer_window_type (str): The type of transfer window ('D' for daily or 'W' for weekly).
            studies_per (int): The number of studies per day or week.

        Returns:
            pd.DataFrame: A data frame containing the configuration information.
        """

        milestone_list = []
        for label, values in app_session_store['milestones'].items():
            milestone_list.append(
                {
                    'Type': label,
                    'Offset Before (Days)': values['offset_before'],
                    'Offset After (Days)': values['offset_after'],
                    'Active Status': values['active']
                }
            )

        df_config = pd.DataFrame(milestone_list)

        df_config[' '] = ''
        df_config['Time Frame Start'] = ''
        df_config.loc[0, 'Time Frame Start'] = str(pd.Timestamp(app_session_store['timeframe_start']).date())
        df_config['Time Frame End'] = ''
        df_config.loc[0, 'Time Frame End'] = str(pd.Timestamp(app_session_store['timeframe_end']).date())
        df_config['  '] = ''

        df_config['Migration Schedule Type'] = ''
        if transfer_window_type == 'D':
            df_config.loc[0, 'Migration Schedule Type'] = 'Daily'
            df_config.loc[1, 'Migration Schedule Type'] = str(studies_per) + ' studies per day'
        else:
            df_config.loc[0, 'Migration Schedule Type'] = 'Weekly'
            df_config.loc[1, 'Migration Schedule Type'] = str(studies_per) + ' studies per week'

        if app_session_store['active_filters']:
            df_config['    '] = ''
            for active_filter in app_session_store['active_filters']:
                label = active_filter[0]
                filtered_values = active_filter[1]
                for i, filter_value in enumerate(filtered_values):
                    df_config.loc[i, f'Filter - {label}'] = filter_value

        df_config['      '] = ''
        df_config['Period'] = ''
        df_config['Period - Start'] = ''
        df_config['Period - End'] = ''

        if transfer_window_type == 'D':
            df_config[f'Period - Length (Days)'] = ''
        else:
            df_config[f'Period - Length (Weeks)'] = ''

        for i, (period_start, period_end) in enumerate(period_start_end, 1):
            df_config.loc[i - 1, f'Period'] = f'Period {i}'

            if transfer_window_type == 'D':
                df_config.loc[i - 1, f'Period - Start'] = str(period_start)
                df_config.loc[i - 1, f'Period - End'] = str(period_end)
                df_config.loc[i - 1, f'Period - Length (Days)'] = len(pd.date_range(start=period_start, end=period_end, freq='D'))

            else:
                df_config.loc[i - 1, f'Period - Start'] = str(period_start)
                df_config.loc[i - 1, f'Period - End'] = str(period_end)
                df_config.loc[i - 1, f'Period - Length (Weeks)'] = len(pd.date_range(start=period_start, end=period_end, freq='W'))

        return df_config
