import math
import random
import sys

NUM_INPUTS = 9 # Input nodes, plus the bias input.
NUM_PATTERNS = 500 # Input patterns for XOR experiment.
NUM_HIDDEN = 7
LR_IH = 0.9 # Learning rate, input to hidden weights.
LR_HO = 0.09 # Learning rate, hidden to output weights.

# The data here is the XOR data which has been rescaled to the range -1 to 1.
# An extra input value of 1 is also added to act as the bias.
# e.g: [Value 1][Value 2][Bias]
TRAINING_INPUT = []

# The output must lie in the range -1 to 1.
TRAINING_OUTPUT = []


class Backpropagation1:
    def __init__(self, numInputs, numPatterns, numHidden,  i2hLearningRate, h2oLearningRate, inputValues, outputValues):
        self.mNumInputs = numInputs
        self.mNumPatterns = numPatterns
        self.mNumHidden = numHidden
        self.mI2HLearningRate = i2hLearningRate
        self.mH2OLearningRate = h2oLearningRate
        self.hiddenVal = [] # Hidden node outputs.
        self.weightsIH = [] # Input to Hidden weights.
        self.weightsHO = [] # Hidden to Output weights.
        self.trainInputs = inputValues
        self.trainOutput = outputValues # "Actual" output values.
        self.errThisPat = 0.0
        self.outPred = 0.0 # "Expected" output values.
        self.RMSerror = 1.0 # Root Mean Squared error.
        return
    def init(self):
	#print self.trainOutput
	f=open("train8.txt")
	count=0
	for i in range(self.mNumPatterns):
	    st=f.readline()
	    res=1
	    ar=[]
	    st=st.split('\t')
	    if st[-1] == 'benign\n':
		count=count+1
	    	for j in range(self.mNumInputs):
		    ar.append(int(st[j]))
	    	TRAINING_INPUT.append(ar)
	    	TRAINING_OUTPUT.append(res)
        self.mNumPatterns=count
	print self.mNumPatterns	
	f.close()
    def init_test(self):
        # Initialize weights to random values.
	f=open("weight8.txt")
        for j in range(self.mNumInputs):
            newRow = []
            for i in range(self.mNumHidden):
                self.weightsHO.append(0)
                newRow.append(0)
	    self.weightsIH.append(newRow)
 
        self.hiddenVal = [0.0] * self.mNumHidden
        f.close()
        return
    def initialize_arrays_test(self):
        # Initialize weights to random values.
        for j in range(self.mNumInputs):
            newRow = []
            for i in range(self.mNumHidden):
                self.weightsHO.append((random.random() - 0.5) / 2.0)
                weightValue = (random.random() - 0.5) / 5.0
                newRow.append(weightValue)
                #sys.stdout.write("Weight = " + str(weightValue) + "\n")
            self.weightsIH.append(newRow)
        
        self.hiddenVal = [0.0] * self.mNumHidden
        
        return
    def initialize_arrays(self):
        # Initialize weights to random values.
	f=open("weight8.txt")
        for j in range(self.mNumInputs):
            newRow = []
            for i in range(self.mNumHidden):
                self.weightsHO[i]=(float(f.readline()))
                self.weightsIH[j][i]=(float(f.readline()))
 
        self.hiddenVal = [0.0] * self.mNumHidden
        f.close()
        return

    def weight_save(self):
        f=open("weight8.txt",'w')
        for j in range(self.mNumInputs):
            newRow = []
            for i in range(self.mNumHidden):
                f.write(str(self.weightsHO[i])+"\n")
            	f.write(str(self.weightsIH[j][i])+"\n")
	self.hiddenVal = [0.0] * self.mNumHidden        
	f.close()
        
        return
    
    def calc_net(self, patNum):
        # Calculates values for Hidden and Output nodes.
        for i in range(self.mNumHidden):
            self.hiddenVal[i] = 0.0
            for j in range(self.mNumInputs):
                self.hiddenVal[i] += (self.trainInputs[patNum][j] * self.weightsIH[j][i])
            
            self.hiddenVal[i] = math.tanh(self.hiddenVal[i])
        
        self.outPred = 0.0
        
        for i in range(self.mNumHidden):
            self.outPred += self.hiddenVal[i] * self.weightsHO[i]
        
        self.errThisPat = self.outPred - self.trainOutput[patNum] # Error = "Expected" - "Actual"
        return
    
    def adjust_hidden_to_output_weights(self):
        for i in range(self.mNumHidden):
            weightChange = self.mH2OLearningRate * self.errThisPat * self.hiddenVal[i]
            self.weightsHO[i] -= weightChange
            
            # Regularization of the output weights.
            if self.weightsHO[i] < -5.0:
                self.weightsHO[i] = -5.0
            elif self.weightsHO[i] > 5.0:
                self.weightsHO[i] = 5.0
        
        return
    
    def adjust_input_to_hidden_weights(self, patNum):
        for i in range(self.mNumHidden):
            for j in range(self.mNumInputs):
                x = 1 - math.pow(self.hiddenVal[i], 2)
                x = x * self.weightsHO[i] * self.errThisPat * self.mI2HLearningRate
                x = x * self.trainInputs[patNum][j]
                
                weightChange = x
                self.weightsIH[j][i] -= weightChange
        
        return
    
    def calculate_overall_error(self):
        errorValue = 0.0
        
        for i in range(self.mNumPatterns):
            self.calc_net(i)
            errorValue += math.pow(self.errThisPat, 2)
        
        errorValue /= self.mNumPatterns
        
        return math.sqrt(errorValue)
    
    def train_network(self):
        patNum = 0
        c=0
        while self.RMSerror >0.0001:
            for i in range(self.mNumPatterns):
                # Select a pattern at random.
                patNum = random.randrange(0, self.mNumPatterns)
                
                # Calculate the output and error for this pattern.
                self.calc_net(patNum)
                
                # Adjust network weights.
                self.adjust_hidden_to_output_weights()
                self.adjust_input_to_hidden_weights(patNum)
            
            self.RMSerror = self.calculate_overall_error()
            c+=1
            sys.stdout.write( "iteration: "+str(c)+" RMS Error = " + str(self.RMSerror) + "\n")
        
        return
    
    def display_results(self):
        for i in range(self.mNumPatterns):
            self.calc_net(i)
            sys.stdout.write("pat = " + str(i + 1) + " actual = " + str(self.trainOutput[i]) + " neural model = " + str(self.outPred) + "\n")
        return    

if __name__ == '__main__':
    bp1 = Backpropagation1(NUM_INPUTS, NUM_PATTERNS, NUM_HIDDEN, LR_IH, LR_HO, TRAINING_INPUT, TRAINING_OUTPUT)
    bp1.init()
   # bp1.init_test()
    bp1.initialize_arrays_test()
    bp1.train_network()
    bp1.display_results()
    bp1.weight_save()
    
    

