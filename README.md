# Simple Neural Network
**(c) 2018, Brehanu Bugg**

This is a simple Neural Network written in Python. The purpose of this project was to understand the inner workings of a Neural Network and how it works. If you are interested, keep reading.

This project is divided into 4 simple files. There's the *Matrix.py* file, the *Network.py* file, the *compute.py* file. and the *activation_functions.py* file. Here's how they work together:

  *Matrix.py*: This file is the blueprint for what a Matrix datastructure is and how it acts. It is created by using an n-dimensional array based on the specified rows and columns that are fed into it. Here are the operations you can perform on it:
  
    - Addition: element-wise addition of two matricies
    
    - Subtraction: element-wise subtraction of two matricies
    
    - Sub: subtracts a scalar from a Matrix object (used for Gradient Descent: 1.0 - final_outputs)
    
    - Scale: scales a Matrix by a scalar number (used for learning rate)
    
    - Map: apply a function that iterates through all of the data points in the Matrix
    
    - Apply Function: similar to map, but returns a Matrix object
    
    - Activate: pass in activation function, and it applies it to given Matrix
    
    - Transpose: convert the current Matrix to the "sideways" version of it
    
    - Random: returns a normalized random number between a -1000 to +1000
    
    - Randomize: initialize the Matrix with random values using the Random function
    
    - Print: prints out the matrix in a human-readable format (similar to Numpy)
    
    - Compatibility Error: error handling to determine if two matricies can be operated on or not
    
    - Raise Compatibility Error: raises the error only if Compatibility Error returns True
    
    - To Matrix: takes an array and creates an **n**x1 Matrix where **n** is the length of the array
    
    - To Array: converts the Matrix to an array by making it one dimensional
    
    - Dot: performs the dot product between two matricies (performs Transpose for you)

  *Network.py*: This file is the blueprint for what a Neural Network is and how it works. It is dependent on the *Matrix.py* file in order for it to work. Here are how it works:
  
    - Initialization Method: creates the weights and biases based on the Network size the user passes in (number of input, hidden, and output nodes)
    
    - Feedforward Method: Neural Network feedforward algorithm (read more about it below)
    
    - Train Method: Neural Network training algorithm (algorithm description coming later)
    
  *compute.py*: This file is where you initialize your data and pass it through the Neural Network. It is NOT dependent of the *Matrix.py* file, but it IS dependent of the *Network.py* file.
  
 *activation_functions.py*: This file have multiple different activation functions including sigmoid, tan, arctan, and gaussian. In the top of the *compute.py* file, include `from activation_function import sigmoid` or whichever activation function you'd like. Then include that same function name as the last argument of `BRAIN = NeuralNetwork(x, y, z, [activation_function, derivative_function])`. 

*Feedforward Algorithm Walkthrough*
The method ``NeuralNetwork.predict(x)`` expects a parameter for the inputs. This is a one-dimensional array. Here is the code walkthrough.

``inputs = Matrix.to_matrix(inputs_arr)`` converts the given one-dimensional array and creates a Nx1 Matrix where N is the amount of items in the array. If passed in ``[0.275, 0.382, 0.231]``, it would make a 3x1 Matrix.

``hidden_inputs = self.weights_ih.dot(inputs)`` performs the dot product between the weights between the input and hidden, and the given data. The dot product method in the Matrix library returns a Matrix, so it is stored in ``hidden_inputs``.

``hidden_outputs = hidden_inputs.activate(self.activation_function)`` applies the activation function (which is passed in the Neural Network constructor function) on a specified Matrix. This returns a Matrix, so it is stored in ``hidden_outputs``.

``final_inputs = self.weights_ho.dot(hidden_outputs)`` performs the dot product between the weights between the hidden and output, and the hidden output. The dot product method in the Matrix library returns a Matrix, so it is stored in ``final_inputs``.

``final_outputs = final_inputs.activate(self.activation_function)`` applies the activation function (which is passed in the Neural Network constructor function) on a specified Matrix. This returns a Matrix, so it is stored in ``final_outputs``.

``return final_outputs`` returns the Network's guess! To see the output, do the following:

```python
Y = NeuralNetwork.predict([0, 0])
Y.print()  # [0.002738291]
```

This is not meant to be an advanced project, whatsoever. It's just to learn about the inner workings of a Neural Network and how it works. 
