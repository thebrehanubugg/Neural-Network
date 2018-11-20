# Simple Neural Network
**(c) 2018, Brehanu Bugg**

This is a simple Neural Network written in Python. The purpose of this project was to understand the inner workings of a Neural Network and how it works. If you are interested, keep reading.

This project is divided into 4 simple files. There's the *Matrix.py* file, the *Network.py* file, the *compute.py* file. and the *activation_functions.py* file. Here's how they work together:

  *Matrix.py*: This file is the blueprint for what a Matrix datastructure is and how it acts. It is created by using an n-dimensional array based on the specified rows and columns that are fed into it. Here are the operations you can perform on it:
  
    - Addition: element-wise addition of two matricies
    
    - Subtraction: element-wise subtraction of two matricies
    
    - Map: apply a function that iterates through all of the data points in the Matrix
    
    - Transpose: convert the current Matrix to the "sideways" version of it
    
    - Randomize: initialize the Matrix with random values between 0-10 instead of the 0 default
    
    - Print: prints out the matrix in a human-readable format
    
    - Compatibility Error: error handling to determine if two matricies can be operated on or not
    
    - Raise Compatibility Error: raises the error only if Compatibility Error returns True
    
    - To Matrix: takes an array and creates an **n**x1 Matrix where **n** is the length of the array
    
    - To Array: converts the Matrix to an array by making it one dimensional
    
    - Dot: performs the dot product between two matricies (performs Transpose for you)

  *Network.py*: This file is the blueprint for what a Neural Network is and how it works. It is dependent on the *Matrix.py* file in order for it to work. Here are how it works:
  
    - Initialization Method: creates the weights and biases based on the Network size the user passes in (number of input, hidden, and output nodes)
    
    - Feedforward Method: "feeds" the given input (an array) and flows it through the Network. First it converts the input to an **n**x1 Matrix where **n** is the number of elements in the array. Then it performs the dot product between the hidden weights and the inputs, applies the sigmoid function, and adds the bias. It performs the same operations for the hidden to output, then returns the output.
    
    - Train Method: **UPDATE LATER**
    
  *compute.py*: This file is where you initialize your data and pass it through the Neural Network. It is NOT dependent of the *Matrix.py* file, but it IS dependent of the *Network.py* file.
  
 *activation_functions.py*: This file have multiple different activation functions including sigmoid, tan, arctan, and gaussian. In the top of the *compute.py* file, include `from activation_function import sigmoid` or whichever activation function you'd like. Then include that same function name as the last argument of `BRAIN = NeuralNetwork(x, y, z, function_name)`. 
  
This is not meant to be an advanced project, whatsoever. It's just to learn about the inner workings of a Neural Network and how it works. 
