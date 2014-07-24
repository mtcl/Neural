Random rand = new Random(1)
e = 2.718281828
def fSigmoid = { x -> 1 / (1 + e.power(-x)) }
def negFactor = { (new Random()).nextInt(10) < 5 ? -1 : 1 }
def max = 1000

//neural network related settings
final int inputParamsCount = 2
final int midLayerMaxElementsCount = 3
final int outputParamsCount = 2
final int innerLayer = 2

def xSize = innerLayer + 1
def yzSize = [inputParamsCount, midLayerMaxElementsCount, outputParamsCount].max()

//define weight matrix
def w = new double[xSize][yzSize][yzSize]

//define intermediate output matrix
def out = new double[xSize + 1][yzSize]

//define error
def err = new double[xSize + 1][yzSize]

//initialize first set of inputs
def x = [0.05, 0.02]

//initialize first set of outputs
def y = [1, 0]

//display neural network weights
def displayMatrix = { x1 ->
    for (int i = 0; i < xSize; i++) {
        for (int j = 0; j < yzSize; j++) {
            for (int k = 0; k < yzSize; k++) {
                print x1[i][j][k] + " "
            }
            println()
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

w=[[[0.5, 0.4, 0.1], [0.2, 0.6, 0.2], [0,0,0]], [[0.10, 0.55, 0.35], [0.2, 0.45, 0.35], [0.25, 0.15, 0.60]], [[0.3, 0.35, 0], [0.35, 0.25, 0], [0.45, 0.3,0]]]
displayMatrix(w)

//initialize first element of out matrix
(0..x.size() - 1).each {
    out[0][it] = x[it]
}

//Initializing remaining elements of out matrix
def sum
for (int i = 1; i < xSize + 1; i++) {

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

for (int i = 0; i < xSize + 1; i++) {
    for (int j = 0; j < yzSize; j++) {
        print out[i][j] + " "
    }
    println()
}

//initialize first element of error matrix
(0..y.size() - 1).each {
    err[xSize][it] = y[it]-out[xSize][it]
}
println()
//Initializing remaining elements of err matrix
//looping backwards in the error matrix (since the length of array is xSize+1, we will start with xSize-1
//the intent is to start with last inner layer in the neural network
for (int i = xSize-1; i >= 1; i--) {

    //find the maximum number of elements in the next layer
    kmax = i == xSize-1 ? outputParamsCount : midLayerMaxElementsCount

    //looping over all the elements in the current layer.
    for (int j = 0; j < yzSize; j++) {
        sum=0
        for (int k = 0; k < kmax; k++) {
            sum+= w[i][j][k]*err[i+1][k]
            //println('w['+(i)+']['+j+']['+k+']*err['+(i+1)+']['+k+']')
        }
        //println()
        err[i][j] = out[i][j]*(1-out[i][j])*sum
    }
}

for (int i = 0; i < xSize + 1; i++) {
    for (int j = 0; j < yzSize; j++) {
        print err[i][j] + " "
    }
    println()
}
