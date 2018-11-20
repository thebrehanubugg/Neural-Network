"""Matrix Class."""
from random import randint, seed
seed(1)


class Matrix():
    """Matrix Class."""

    def __init__(self, i, j):
        """Matrix Initialization Method."""
        self.rows = i
        self.cols = j

        self.n_rows = range(self.rows)
        self.n_cols = range(self.cols)

        self.data = [[0 for j in self.n_cols] for i in self.n_rows]

    def randomize(self):
        """Set matrix data to random numbers."""
        for i in self.n_rows:
            for j in self.n_cols:
                self.data[i][j] = randint(0, 10)

    def print(self):
        """Print out the matrix a in human-readable format."""
        for row in self.data:
            print(row)
        print("\n")

    @staticmethod
    def raise_compatibility_error(a, b, dot=False):
        """Matrix dimensions do not match."""
        a_val = "{0}x{1}".format(a.rows, a.cols)
        b_val = "{0}x{1}".format(b.rows, b.cols)

        if dot:
            b_reverse = "{0}x{1}".format(b.cols, b.rows)
            error = "Should be " + b_reverse + " for second Matrix."
        else:
            error = a_val + " does not match " + b_val

        raise IndexError(error)

    @staticmethod
    def compatibility_error(a, b, dot=False):
        """Check size compatibility between matricies."""
        if dot:
            if a.rows != b.cols and a.cols != b.rows:
                return True
            else:
                return False
        else:
            if a.rows != b.rows and a.cols != b.cols:
                return True
            else:
                return False

    def map(self, func):
        """Apply a function on the data."""
        result = Matrix(self.rows, self.cols)

        for i in result.n_rows:
            for j in result.n_cols:
                value = self.data[i][j]
                result.data[i][j] = func(value)

        self.data = result.data

    def transpose(self):
        """Convert columns to rows and vice-versa."""
        result = Matrix(self.cols, self.rows)

        for i in result.n_rows:
            for j in result.n_cols:
                result.data[i][j] = self.data[j][i]

        result.data.reverse()
        return result

    @staticmethod
    def to_matrix(arr):
        """Convert array to Matrix object."""
        result = Matrix(len(arr), 1)

        for i in enumerate(arr):
            result.data[i[0]][0] = arr[i[0]]

        return result

    @staticmethod
    def to_array(matrix):
        """Convert Matrix object to array."""
        result = []

        for i in matrix.n_rows:
            for j in matrix.n_cols:
                result.append(matrix.data[i][j])

        return result

    def add(self, b):
        """Element-wise addition of two matricies."""
        if Matrix.compatibility_error(self, b):
            Matrix.raise_compatibility_error(self, b)
        else:
            result = Matrix(self.rows, b.cols)

            for i in range(self.rows):
                for j in range(b.cols):
                    result.data[i][j] = self.data[i][j] + b.data[i][j]

            self.data = result.data

    def subtract(self, b):
        """Element-wise subtraction of two matricies."""
        if Matrix.compatibility_error(self, b):
            Matrix.raise_compatibility_error(self, b)
        else:
            result = Matrix(self.rows, b.cols)

            for i in self.n_rows:
                for j in b.n_cols:
                    difference = self.data[i][j] - b.data[i][j]
                    result.data[i][j] = difference

            return result

    def dot(self, b):
        """Matrix multiplication between two matricies."""
        if Matrix.compatibility_error(self, b, dot=True):
            Matrix.raise_compatibility_error(self, b, dot=True)
        else:
            result = Matrix(self.rows, b.cols)
            b = b.transpose()

            for i in enumerate(self.data):
                for j in enumerate(b.data):
                    products = []
                    for k in enumerate(i[1]):
                        product = k[1] * j[1][k[0]]
                        products.append(product)

                    value = sum(products)
                    result.data[i[0]][j[0]] = value

            [i.reverse() for i in result.data]
            return result
