import numpy as np


def load_data(path:str):
    """Reads each row of the train dataset, converts it to numbers, and returns comleted list of utterances"""
    completed_data = []
    row_current = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            values = [float(val) for val in line.split()]

            if len(values) == 12 and all(val == 1.0 for val in values):
                if row_current:
                    completed_data.append(np.array(row_current))
                    row_current = []
            else:
                row_current.append(values)
    if row_current:
        completed_data.append(np.array(row_current))


    return completed_data


def create_labels(utterance_per_person):
    """Assigns each utterance to a concrete person"""
    labels = []

    for person, utter_current_person in enumerate(utterance_per_person, start=1):
        for _ in range(utter_current_person):
            labels.append(person)

    return np.array(labels)
