import csv

def load_csv(filepath, delimiter=',', has_header=True):
    """
    Loads data from a CSV file.
    
    Each row should contain features (all columns except the last) and the last column is the label.
    Returns two lists:
      - values: list of feature lists (floats)
      - labels: list of labels (float if possible, else string)
    """
    values = []  # features go here
    labels = []  # labels go here
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            if has_header:
                next(reader)  # skip header
            for row in reader:
                if not row or all(cell.strip() == '' for cell in row):
                    continue  # skip empty rows
                try:
                    # Convert all but the last column to floats
                    current_values = [float(x) for x in row[:-1]]
                except ValueError:
                    # Skip row if conversion fails
                    continue
                current_label = row[-1]  # last column is the label
                try:
                    # Convert label to float if possible
                    current_label = float(current_label)
                except ValueError:
                    pass  # keep label as string if conversion fails
                values.append(current_values)
                labels.append(current_label)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return values, labels

def normalize(data):
    """
    Normalizes data columns to range [0, 1].
    
    For each column:
      1. Find the minimum and maximum.
      2. Scale each value: (x - min) / (max - min).
      3. If all values are the same, set to 0.5.
    Returns a list of normalized rows.
    """
    if not data:
        return data
    try:
        # Transpose rows to columns
        columns = list(zip(*data))
        norm_columns = []
        for col in columns:
            try:
                min_val = min(col)
                max_val = max(col)
            except Exception as e:
                print("Error computing min/max for a column:", e)
                min_val, max_val = 0, 0
            if max_val == min_val:
                norm_col = [0.5] * len(col)  # if all values are the same, use 0.5
            else:
                norm_col = [(x - min_val) / (max_val - min_val) for x in col]
            norm_columns.append(norm_col)
        # Transpose columns back to rows
        normalized_data = [list(row) for row in zip(*norm_columns)]
    except Exception as e:
        print("Error during normalization:", e)
        normalized_data = data
    return normalized_data

def batches(data, labels, batch_size):
    """
    Splits data and labels into mini-batches of a given size.
    
    data: list of input vectors (each input is a feature list)
    labels: list of labels for each input
    batch_size: desired size for each mini-batch (must be > 0)
    
    Returns:
      - batch_data: list of feature batches
      - batch_labels: list of label batches
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    batch_data = []    # batches of inputs
    batch_labels = []  # batches of labels
    for i in range(0, len(data), batch_size):
        batch_data.append(data[i:i+batch_size])
        batch_labels.append(labels[i:i+batch_size])
    return batch_data, batch_labels