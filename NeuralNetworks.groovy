Random rand = new Random(1)
e = 2.718281828
def fSigmoid = { x -> 1 / (1 + e.power(-x)) }
def negFactor = { (new Random()).nextInt(10) < 5 ? -1 : 1 }
def max = 1000

//neural network related settings
final int inputParamsCount = 5
final int midLayerMaxElementsCount = 3
final int outputParamsCount = 1
final int innerLayer = 4

//learning constant
alpha = 0.4

def xSize = innerLayer + 1
def yzSize = [inputParamsCount, midLayerMaxElementsCount, outputParamsCount].max()

//define weight matrix
def w = new double[xSize][yzSize][yzSize]

//define intermediate output matrix
def out = new double[xSize + 1][yzSize]

//define error
def err = new double[xSize + 1][yzSize]

//initialize first set of inputs
def x = [1.5, 2.02, 0.6, 0.4, 0.3, 0.5, 0.8]

//initialize first set of outputs
def y = [0.5]

//display neural network weights
def displayMatrix3d = { x1 ->
    for (int i = 0; i < x1.size(); i++) {
        for (int j = 0; j < x1[0].size(); j++) {
            for (int k = 0; k < x1[1].size(); k++) {
                print x1[i][j][k] + " "
            }
            println()
        }
        println()
    }
}

//display neural network weights
def displayMatrix2d = { x1 ->
    for (int i = 0; i < x1.size(); i++) {
        for (int j = 0; j < x1[0].size(); j++) {
            print x1[i][j] + " "
        }
        println()
    }
}

//initialize weight matrix
for (int i = 0; i < xSize; i++) {
    for (int j = 0; j < yzSize; j++) {
        for (int k = 0; k < yzSize; k++) {
            w[i][j][k] = negFactor() * rand.nextInt(max) / max
            //Adjusting for input layer and output layer counts
            if ((i == 0 && j > inputParamsCount - 1) || (i == innerLayer && k >= outputParamsCount)) {
                w[i][j][k] = 0
            }
        }
    }
}

//w = [[[0.5, 0.4, 0.1], [0.2, 0.6, 0.2], [0, 0, 0]], [[0.10, 0.55, 0.35], [0.2, 0.45, 0.35], [0.25, 0.15, 0.60]], [[0.3, 0.35, 0], [0.35, 0.25, 0], [0.45, 0.3, 0]]]


def outCalculation(fSigmoid, x, out, xSize, inputParamsCount, midLayerMaxElementsCount, outputParamsCount, yzSize, w) {
//Start out matrix calculation
//initialize first element of out matrix
    (0..x.size() - 1).each {
        out[0][it] = x[it]
    }

//Initializing remaining elements of out matrix
    def sum
    for (int i = 1; i < xSize + 1; i++) {

        //The last row of the out elements is the final output

        // find the maximum number of elements in the next layer
        //for input parameters i is inputParamsCount
        kmax = i == 1 ? inputParamsCount : midLayerMaxElementsCount
        //for output parameters
        yzSize1 = i == xSize ? outputParamsCount : yzSize

        for (int j = 0; j < yzSize1; j++) {
            sum = 0
            for (int k = 0; k < kmax; k++) {
                sum += w[i - 1][k][j] * out[i - 1][k]
                //print('w['+(i-1)+']['+k+']['+j+']*out['+(i-1)+']['+k+']  + ')
                //print(w[i - 1][k][j] + ' * ' + out[i - 1][k] + ' + ')
            }
            out[i][j] = fSigmoid(sum)
            //print(' = ' + out[i][j])
            //println()
        }
    }
/*
for (int i = 0; i < xSize + 1; i++) {
    for (int j = 0; j < yzSize; j++) {
        print out[i][j] + " "
    }
    println()
}
*/
//end out matrix calculation
    return out
}

def backpropagation(fSigmoid, inputParamsCount, midLayerMaxElementsCount, outputParamsCount, innerLayer, alpha, xSize, yzSize, w, out, err, x, y) {

    //println("Calculating out matrix...")
    out = outCalculation(fSigmoid, x, out, xSize, inputParamsCount, midLayerMaxElementsCount, outputParamsCount, yzSize, w)

    //println("Calculating error matrix...")
    //initialize first element of error matrix
    (0..y.size() - 1).each {
        err[xSize][it] = y[it] - out[xSize][it]
    }

    //Initializing remaining elements of err matrix
    //looping backwards in the error matrix (since the length of array is xSize+1, we will start with xSize-1
    //the intent is to start with last inner layer in the neural network
    for (int i = xSize - 1; i >= 1; i--) {

        //find the maximum number of elements in the next layer
        kmax = i == xSize - 1 ? outputParamsCount : midLayerMaxElementsCount

        //looping over all the elements in the current layer.
        for (int j = 0; j < yzSize; j++) {
            sum = 0
            for (int k = 0; k < kmax; k++) {
                sum += w[i][j][k] * err[i + 1][k]
                //println('w['+(i)+']['+j+']['+k+']*err['+(i+1)+']['+k+']')
            }
            //println()
            err[i][j] = out[i][j] * (1 - out[i][j]) * sum
        }
    }
    /*
    for (int i = 0; i < xSize + 1; i++) {
        for (int j = 0; j < yzSize; j++) {
            print err[i][j] + " "
        }
        println()
    }
    */
    //println('adjusting weights')
    //Adjust weights for each layer
    for (int i = 0; i < xSize; i++) {
        for (int j = 0; j < yzSize; j++) {
            for (int k = 0; k < yzSize; k++) {

                if ((i == 0 && j > inputParamsCount - 1) || (i == innerLayer && k >= outputParamsCount)) {
                    w[i][j][k] = 0
                } else {
                    //print('w'+ '[' + i + ',' + j + ',' + k + '] ->> ')
                    //println(w[i][j][k] + ' + ' + alpha +  '*' + err[i + 1][k] + ' * ' + out[i][k] + ' = ' + w[i][j][k])
                    w[i][j][k] = w[i][j][k] + alpha * err[i + 1][k] * out[i][k]

                }
            }
        }
    }
    //println('done')
    return w
}

xInput = [[0.4, 0.7, 0.6, 0.9, 0.5],
          [0.5, 0.5, 0.8, 0.5, 1],
          [0.7, 0.9, 0.3, 0.2, 0.5],
          [0.9, 0.5, 1, 0.7, 0.9],
          [0.5, 0.1, 0.6, 0.3, 0.9],
          [0.5, 0.6, 0.4, 0.7, 1],
          [0.2, 0.8, 0.9, 0.8, 1],
          [0.4, 0.5, 0.2, 0.3, 0.7],
          [0.6, 0.7, 1, 0.1, 0.3],
          [0.4, 0.4, 0.5, 0.4, 0.5],
          [0.2, 0.6, 0.4, 0.2, 0.5],
          [0.2, 0.7, 0.2, 0.6, 0.8],
          [0.6, 0.8, 0.3, 0.2, 0.1],
          [0.8, 0.4, 0.9, 0.7, 0.3],
          [0.3, 0.2, 1, 0.8, 0.4],
          [0.3, 0.3, 0.9, 1, 0.9],
          [0.5, 0.1, 0.3, 0.6, 0.2],
          [0.3, 0.2, 0.8, 0.4, 0.9],
          [0.7, 0.4, 0.4, 0.5, 1]]

yInput = [[0.729333333],
          [0.656],
          [0.516666667],
          [0.5517333333],
          [0.428666667],
          [0.514666667],
          [0.754],
          [0.344666667],
          [0.65],
          [0.396666667],
          [0.366],
          [0.436666667],
          [0.478],
          [0.650666667],
          [0.728],
          [0.784666667],
          [0.412666667],
          [0.434666667],
          [0.397333333],
          [0.516]]


100.times {
    println("iteration number:: $it")
    for (int i = 0; i < xInput.size(); i++) {
        x = xInput[i]
        y = yInput[i]
        //println("training the neural network using $x as input and $y as output")
        //displayMatrix3d(w)
        100.times {
            w = backpropagation(fSigmoid, inputParamsCount, midLayerMaxElementsCount, outputParamsCount, innerLayer, alpha, xSize, yzSize, w, out, err, x, y)
            // displayMatrix3d(w)
        }
    }
//    x = xInput[5]
//    out = outCalculation(fSigmoid, x, out, xSize, inputParamsCount, midLayerMaxElementsCount, outputParamsCount, yzSize, w)
//displayMatrix2d(out)

//    println("predicted output for this input is: " + out[out.size()-1][0])
}

//error value calculation
def errsum = 0
for (int i = 0; i < xInput.size(); i++) {
    x = xInput[i]
    y = yInput[i]
    out = outCalculation(fSigmoid, x, out, xSize, inputParamsCount, midLayerMaxElementsCount, outputParamsCount, yzSize, w)
//displayMatrix2d(out)
    err = (y[0] - out[out.size() - 1][0])
    println("value of $i input variable is: $err")
    errsum += Math.pow(err, 2)

}
mse = errsum / xInput.size()
println(Math.pow(mse, 0.5))