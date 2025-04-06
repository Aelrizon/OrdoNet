import random

class Matrix:
    def __init__(self, cols, rows, data=None):
        self.cols = cols  # number of columns
        self.rows = rows  # number of rows

        if data:
            self.data = data  # use given data if provided
        else:
            # if no data, fill matrix with zeros
            self.data = [[0 for _ in range(cols)] for _ in range(rows)]

    def randomize(self):
        # fill the matrix with random numbers from -1 to 1
        self.data = [
            [random.uniform(-1, 1) for _ in range(self.cols)]  # for each column
            for _ in range(self.rows)                          # for each row
        ]

    def show(self):
        # print the matrix row by row
        for row in self.data:
            print(row)

    def transpose(self):
        # flip rows and columns
        flipped = list(zip(*self.data))               # convert columns to rows
        new_data = [list(row) for row in flipped]     # convert to list of lists
        return Matrix(self.rows, self.cols, new_data) # return new transposed matrix

    def dot(self, other):
        # standard matrix multiplication: row * column
        result = []

        for row in self.data:  # go through each row in this matrix
            new_row = []
            for col in zip(*other.data):  # go through each column in other matrix
                total = 0
                for a, b in zip(row, col):  # multiply and sum pairs
                    total += a * b
                new_row.append(total)  # add to result row
            result.append(new_row)  # add row to final result

        return Matrix(self.rows, other.cols, result)

    def add(self, other):
        # add two matrices element by element
        new = [
            [a + b for a, b in zip(row1, row2)]  # add each pair
            for row1, row2 in zip(self.data, other.data)
        ]
        return Matrix(self.rows, self.cols, new)

    def subtract(self, other):
        # subtract one matrix from another element by element
        new = [
            [a - b for a, b in zip(row1, row2)]  # subtract each pair
            for row1, row2 in zip(self.data, other.data)
        ]
        return Matrix(self.rows, self.cols, new)

    def multiply(self, other):
        # element-wise multiplication (Hadamard product)
        new = [
            [a * b for a, b in zip(row1, row2)]  # multiply each pair
            for row1, row2 in zip(self.data, other.data)
        ]
        return Matrix(self.rows, self.cols, new)

    def scalar(self, number):
        # multiply each element by a number
        new = [
            [a * number for a in row]  # scale each value
            for row in self.data
        ]
        return Matrix(self.rows, self.cols, new)