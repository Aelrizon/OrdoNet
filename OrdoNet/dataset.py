import csv

def load_csv(filepath, delimiter=',', has_header=True):
    """
    Loads data from a CSV file.
    
    Each row contains features and the last column is the label.
    Returns two lists:
      - values: list of feature lists
      - labels: list of labels (number or string)
    """
    values = []  # features go here
    labels = []  # labels go here
    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        if has_header:
            next(reader)  # skip header
        for row in reader:
            if not row:
                continue  # skip empty rows
            try:
                # convert all but last column to float
                current_values = [float(x) for x in row[:-1]]
            except ValueError:
                continue  # skip if conversion fails
            current_label = row[-1]  # last column is the label
            try:
                # convert label to float if possible
                current_label = float(current_label)
            except ValueError:
                pass  # keep as string if not a number
            values.append(current_values)
            labels.append(current_label)
    return values, labels

def normalize(data):
    """
    Normalizes data columns to range [0, 1].
    
    For each column:
      1. Find min and max
      2. Scale each value: (x - min) / (max - min)
      3. If all values are the same, set to 0.5
    Returns a list of normalized rows.
    """
    if not data:
        return data
    # transpose rows to columns
    columns = list(zip(*data))
    norm_columns = []
    for col in columns:
        min_val = min(col)  # find min
        max_val = max(col)  # find max
        if max_val == min_val:
            norm_col = [0.5] * len(col)  # if all values same, set 0.5
        else:
            norm_col = [(x - min_val) / (max_val - min_val) for x in col]
        norm_columns.append(norm_col)
    # transpose columns back to rows
    normalized_data = [list(row) for row in zip(*norm_columns)]
    return normalized_data

def batches(data, labels, batch_size):
    """
    Splits data and labels into mini-batches of a given size.
    
    data: list of inputs (each input is a feature list)
    labels: list of labels for each input
    batch_size: how big each batch should be
    
    Returns:
      - batch_data: list of feature batches
      - batch_labels: list of label batches
    """
    batch_data = []    # batches of inputs
    batch_labels = []  # batches of labels
    for i in range(0, len(data), batch_size):
        # take data from i to i + batch_size
        batch_data.append(data[i:i+batch_size])
        batch_labels.append(labels[i:i+batch_size])
    return batch_data, batch_labels