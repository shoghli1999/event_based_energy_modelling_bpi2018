import pickle

import pm4py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional



class EventObject:
    """Class representing an event with specified attributes."""

    def __init__(self, e_id: int, case_id: int, description: str,
                 case_history: List[int], event_type: str,
                 timestamp: datetime, duration: float):
        self.e_id = e_id
        self.case_id = case_id
        self.description = description
        self.case_history = case_history
        self.event_type = event_type
        self.timestamp = timestamp
        self.duration = duration

    def __repr__(self):
        return (f"EventObject(e_id={self.e_id}, case_id={self.case_id}, "
                f"event_type='{self.event_type}', "
                f"timestamp='{self.timestamp}', duration={self.duration:.2f})")

    #Looks up the base cost for this event_type in the global dict BASECOSTS and multiplies by global COST_SCALE.
    def get_base_cost(self):
        return __BASECOSTS__.get(self.event_type) * __COST_SCALE__

    #Returns True if a given event_id appears in this event’s case_history and case_id equals provided case_identifier.
    def is_subsequent_event(self, event_id, case_identifier):
        return event_id in self.case_history and self.case_id == case_identifier

    #This adds cost every 900 seconds (15 minutes) if the event is late.
    def get_time_diff_cost(self, end_timestamp):
        time_diff = self.timestamp - end_timestamp
        time_diff_cost = 0
        interval = 900
        while interval < time_diff.total_seconds():
            time_diff_cost += self.get_base_cost() * __DURATION_SCALE__
            interval += 900

        return time_diff_cost

    #Total cost of ending event
    def determine_end_event_cost(self, timestamp):
        return self.get_base_cost() + self.get_time_diff_cost(timestamp) + self.get_base_cost() * __END_SCALE__

    def get_event_duration(self):
        return self.duration


def process_xes_events(xes_path: str, interval_length: int,
                       event_processor: Callable[[EventObject], Any],
                       start_timestamp: Optional[datetime] = None,
                       end_timestamp: Optional[datetime] = None):
    """
    Processes an XES file and creates EventObject instances for each event,
    then calls the provided event_processor function on each object.

    Parameters:
    -----------
    xes_path : str
        Path to the XES file
    interval_length : int
        Length of each interval in seconds (used for grouping/processing)
    event_processor : Callable[[EventObject], Any]
        Function that will be called for each event object created
    start_timestamp : datetime, optional
        Start timestamp for analysis (if None, uses all events)
    end_timestamp : datetime, optional
        End timestamp for analysis (if None, uses all events)

    Returns:
    --------
    Dict[str, List[EventObject]]
        Dictionary mapping event types to lists of EventObject instances
    """
    # Import the XES file
    log = pm4py.read_xes(xes_path)

    # Convert to DataFrame for easier manipulation
    df = pm4py.convert_to_dataframe(log)

    # Check for required columns
    required_columns = ['case:concept:name', 'concept:name', 'time:timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"XES file missing required columns: {missing_columns}")

    # Ensure timestamp is in datetime format
    if not isinstance(df['time:timestamp'].iloc[0], datetime):
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    # Filter events within the specified time range if provided
    if start_timestamp is not None:
        if isinstance(start_timestamp, str):
            start_timestamp = pd.to_datetime(start_timestamp)
        df = df[df['time:timestamp'] >= start_timestamp]

    if end_timestamp is not None:
        if isinstance(end_timestamp, str):
            end_timestamp = pd.to_datetime(end_timestamp)
        df = df[df['time:timestamp'] <= end_timestamp]

    if df.empty:
        raise ValueError("No events found in the specified time range")

    # Get unique event types
    event_types = df['concept:name'].unique()

    # Create a dictionary to store event objects by type
    events_by_type = {event_type: [] for event_type in event_types}

    # Track case histories
    case_histories = {}

    # Calculate the standard deviation for the normal distribution
    # For 99% of values to fall between 2 and 4 minutes, we need:
    # mean = 3 minutes = 180 seconds
    # The range (2 mins to 4 mins) = 120 seconds, which is 6 standard deviations
    # So one standard deviation = 20 seconds
    mean_duration = 180  # 3 minutes in seconds
    std_deviation = 20  # Standard deviation in seconds

    # Create event objects
    for idx, row in df.iterrows():
        case_id = row['case:concept:name']
        event_type = row['concept:name']
        timestamp = row['time:timestamp']

        # Get or initialize case history
        if case_id not in case_histories:
            case_histories[case_id] = []

        # Get event ID (using row index if not available)
        e_id = idx
        if 'event_id' in row:
            e_id = row['event_id']

        # Get description if available, otherwise use event type
        description = event_type
        if 'description' in row:
            description = row['description']

        # Generate random duration following normal distribution
        # Ensure duration is between 2 and 4 minutes (120-240 seconds)
        duration = np.random.normal(mean_duration, std_deviation)
        duration = max(120, min(240, duration))  # Clip to ensure values are within desired range

        # Create a copy of the case history before adding the current event
        current_case_history = case_histories[case_id].copy()

        # Create the event object
        event_obj = EventObject(
            e_id=e_id,
            case_id=int(case_id) if str(case_id).isdigit() else hash(str(case_id)),
            description=description,
            case_history=current_case_history,
            event_type=event_type,
            timestamp=timestamp,
            duration=duration
        )

        # Update case history with this event
        case_histories[case_id].append(e_id)

        # Store the event by type
        events_by_type[event_type].append(event_obj)

        # Process the event with the provided function
        event_processor(event_obj)

    return events_by_type


# Example usage:
def example_event_processor(event: EventObject):
    """Example event processor function that prints the event and its duration in minutes"""
    duration_min = event.duration / 60
    print(f"Event: {event.event_type}, Duration: {duration_min:.2f} minutes")

# Read EventObjects from Pickle file
def read_event_objects_from_pickle(file_path: str) -> Dict[str, List[EventObject]]:
    """Reads event objects from a pickle file."""
    with open(file_path, "rb") as f:
        event_objects = pickle.load(f)
    return event_objects



#events = process_xes_events(f"X:\\PHD\\lehner_phd_overview\\signal_generator\\filtered_log.xes",
#                                interval_length=900,  # 15-mins intervals
#                                event_processor=example_event_processor
#)
# Write the event objects to a binary file
#with open(f"event_objects_all_variants.pkl", "wb") as f:
#    pickle.dump(events, f)


__BASECOSTS__ = {
    "Record Goods Receipt": 99,
    "Create Purchase Order Item": 98,
    "Record Invoice Receipt": 97,
    "Vendor creates invoice": 54,
    "Clear invoice": 56,
    "Record Service Entry Sheet": 81,
    "Remove Payment Block": 59,
    "Create Purchase Requisition Item": 11,
    "Receive Order Confirmation": 66,
    "Change Quantity": 2,
    "Change Price": 1,
    "Delete Purchase Order item": 8,
    "Cancel Invoice Receipt": 9,
    "Change Approval for Purchase Order": 21,
    "Vendor creates debit memo": 20,
    "Change Delivery Indicator": 52,
    "Cancel Goods Receipt": 10,
    "Release Purchase Order": 60,
    "SRM: In Transfer to Execution Syst.": 87,
    "SRM: Created": 78,
    "SRM: Complete": 77,
    "SRM: Awaiting Approval": 63,
    "SRM Document Completed": 70,
    "SRM: Ordered": 44,
    "SRM Change was Transmitted": 92,
    "Reactivate Purchase Order Item": 57,
    "Block Purchase Order Item": 31,
    "Cancel Subsequent Invoice": 32,
    "Change Storage Location": 36,
    "Update Order Confirmation": 22,
    "Record Subsequent Invoice": 40,
    "Release Purchase Requisition": 62,
    "Set Payment Block": 26,
    "SRM: Deleted": 5,
    "Change Currency": 150,
    "Change Final Invoice Indicator": 144,
    "SRM: Transaction Completed": 121,
    "SRM: Incomplete": 111,
    "SRM: Held": 88,
    "Change payment term": 4,
    "Change Rejection Indicator": 200
}

__COST_SCALE__ = 1.0
__DURATION_SCALE__ = 0.01
__END_SCALE__ = 0.1
