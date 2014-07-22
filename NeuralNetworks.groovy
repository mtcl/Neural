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
def out = new double[xSize+1][yzSize]

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
displayMatrix(w)

//initialize first set of inputs
def x = [0.05, 0.02]

//initialize first element of output matrix
(0..x.size() - 1).each {
    out[0][it] = x[it]
}
def sum
for (int i = 1; i < xSize+1; i++) {
    //for input parameters
    kmax = i == 1 ? inputParamsCount : midLayerMaxElementsCount
    //for output parameters
    yzSize1 = i == xSize ? outputParamsCount : yzSize

    for (int j = 0; j < yzSize1; j++) {
        sum=0
        for (int k=0; k<kmax; k++){
            sum+= w[i-1][k][j]*out[i-1][k]
            //println('w['+(i-1)+']['+k+']['+j+']*out['+(i-1)+']['+k+']')
        }
       //println()
        out[i][j]=fSigmoid(sum)
    }
}

for (int i = 0; i < xSize+1; i++) {
    for (int j = 0; j < yzSize; j++) {
        print out[i][j]+ " "
    }
println()
}
