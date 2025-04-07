import random

class Matrix:
    def __init__(self, cols, rows, data=None):
        self.cols = cols  # number of columns
        self.rows = rows  # number of rows

        if data:
            # Use provided data (ensure it has the correct dimensions)
            if len(data) != rows or any(len(row) != cols for row in data):
                raise ValueError("Data dimensions do not match given rows and cols.")
            self.data = data
        else:
            # If no data, fill the matrix with zeros
            self.data = [[0 for _ in range(cols)] for _ in range(rows)]

    def randomize(self):
        # Fill the matrix with random numbers from -1 to 1
        self.data = [
            [random.uniform(-1, 1) for _ in range(self.cols)]  # for each column
            for _ in range(self.rows)                           # for each row
        ]

    def show(self):
        # Print the matrix row by row
        for row in self.data:
            print(row)

    def __str__(self):
        # Create a neat string representation of the matrix
        return "\n".join(str(row) for row in self.data)

    def transpose(self):
        # Flip rows and columns
        flipped = list(zip(*self.data))               # Convert columns to rows
        new_data = [list(row) for row in flipped]     # Convert tuples to lists
        return Matrix(self.cols, self.rows, new_data)   # Note: new shape is (cols, rows)

    def dot(self, other):
        # Standard matrix multiplication: self (rows x cols) dot other (other.rows x other.cols)
        if self.cols != other.rows:
            raise ValueError("Matrix dot product dimension mismatch: self.cols must equal other.rows")
        
        result = []
        for row in self.data:  # For each row in self
            new_row = []
            # For each column in other (via transposition)
            for col in zip(*other.data):
                total = 0
                for a, b in zip(row, col):  # Multiply corresponding elements and sum
                    total += a * b
                new_row.append(total)
            result.append(new_row)
        return Matrix(self.rows, other.cols, result)

    def add(self, other):
        # Element-wise addition: both matrices must have the same dimensions
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix addition dimension mismatch.")
        
        new_data = [
            [a + b for a, b in zip(row1, row2)]  # Add each corresponding pair
            for row1, row2 in zip(self.data, other.data)
        ]
        return Matrix(self.cols, self.rows, new_data)

    def subtract(self, other):
        # Element-wise subtraction: both matrices must have the same dimensions
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix subtraction dimension mismatch.")
        
        new_data = [
            [a - b for a, b in zip(row1, row2)]  # Subtract each pair
            for row1, row2 in zip(self.data, other.data)
        ]
        return Matrix(self.cols, self.rows, new_data)

    def multiply(self, other):
        # Element-wise multiplication (Hadamard product): matrices must have the same dimensions
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix element-wise multiplication dimension mismatch.")
        
        new_data = [
            [a * b for a, b in zip(row1, row2)]  # Multiply each pair
            for row1, row2 in zip(self.data, other.data)
        ]
        return Matrix(self.cols, self.rows, new_data)

    def scalar(self, number):
        # Multiply each element by a scalar number
        new_data = [
            [a * number for a in row]  # Scale each value
            for row in self.data
        ]
        return Matrix(self.cols, self.rows, new_data)
