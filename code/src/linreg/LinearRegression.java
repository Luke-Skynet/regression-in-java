package linreg;

import interfaces.Model;

import math.Vector;

import java.io.*;

/**
 * This class provides multilinear regression. R^n -> R. 
 * Instead of using least squares to optimize, it uses gradient descent.
 */
public class LinearRegression implements Model<Vector, Float, LinRegData>{
	
	private Vector weights;
	private float bias;
	
	/**
	 * Constructor where only the dimension is given and all values are set to default 0. Good for when model will be trained.
	 * @param dimensions - the number of features the model takes in and transforms linearly (Y = wx1 + wx2 + wxn + b)
	 */
	public LinearRegression(int dimensions) {
		this.weights = new Vector(dimensions);
		this.bias = 0;
	}

	/**
	 * Constructor where all the weights and bias are initialized to specific input. Good for when parameters are already estimated.
	 * @param weights - the weights that are deeply copied and used as parameters 
	 * @param bias - the bias / intecept that is copied
	 */
	public LinearRegression(Vector weights, float bias) {
		this.weights = weights.deepCopy();
		this.bias = bias;
	}

	/**
	 * The main inference / computation of the linear function
	 * @param x the input vector X
	 * @return scalar Y = W*X + b
	 */
	@Override
	public Float compute(Vector x) {
		return weights.dot(x) + bias;
	}
	
	/**
	 * Training method that looks at every data point in the training set before updating weights during steps (nonstochastic).
	 * @param training - an array of LinRegData (vector, float) objects that the model uses for weight updating
	 * @param testing - an array of LinRegData (vector, float) objects that is used to display loss when verbose is true 
	 * @param learningRate - a single precision float used to scale gradients for training steps. good values are usually between .01 and .1
	 * @param epochs - number of times the model goes through the training data array, (also number of training steps as this method is not stochastic)
	 * @param verbose - display toggle for viewing training process, (setting to false will disable testing data passes / loss computation)
	 */
	@Override
	public void train(LinRegData[] training, LinRegData[] testing, float learningRate, int epochs, boolean verbose){

		for (int e = 0; e < epochs; e++) {
			
			this.updateWB(training, learningRate);
			
			if (verbose) {
				if (e % 100 == 0)
					System.out.println("Iteration: " + e + " Loss: " + this.getLoss(testing));
			}
		}
		
	}

	/**
	 * Training method that uses batches of data samples to update weights at every step (stochastic gradient descent).
	 * @param training - an array of LinRegData (vector, float) objects that the model uses for weight updating
	 * @param testing - an array of LinRegData (vector, float) objects that is used to display loss when verbose is true 
	 * @param learningRate - a single precision float used to scale gradients for training steps. good values are usually between .01 and .1
	 * @param iterations - the number of times that the model uses a batch to update its parameters (Epochs = iterations * batchsize / training length)
	 * @param batchSize - the number of samples the model uses to update its weights during a training step
	 * @param verbose - display toggle for viewing training process, (setting to false will disable testing data passes / loss computation)
	 */
	public void train(LinRegData[] training, LinRegData[] testing, float learningRate, int iterations, int batchSize, boolean verbose){
		
		if (batchSize > training.length)
			throw new IllegalArgumentException("Batch size must not exceed data size");
		
		int batchCount = training.length / batchSize;
		int spareCount = training.length % batchSize;
		
		int place = 0;
		
		LinRegData[][] batches = new LinRegData[batchCount][batchSize];
		for (int i = 0; i < batchCount; i++) {
			for (int j = 0; j < batchSize; j++) {
				batches[i][j] = new LinRegData(training[place].getData(), training[place].getLabel());
				place++;
			}
		}
		
		LinRegData[] spareBatch = new LinRegData[spareCount];
		if (spareCount > 0) {
			spareBatch = new LinRegData[spareCount];
			for (int i = 0; i < spareBatch.length; i++) {
				spareBatch[i] = new LinRegData(training[place].getData(), training[place].getLabel());
				place++;
			}
		}
	
		int iteration = 0;
		
		outerloop:
		while (true) {
			for (int i = 0; i < batchCount; i++) {
				
				this.updateWB(batches[i], learningRate);
				
				if (verbose) {
					if (iteration % 100 == 0)
						System.out.println("Iteration: " + iteration + " Loss: " + this.getLoss(testing));
				}
				
				if (iteration++ >= iterations) 
					break outerloop;
			}
			if (spareCount > 0) {
				
				this.updateWB(spareBatch, learningRate);
				
				if (verbose) {
					if (iteration % 100 == 0)
						System.out.println("Iteration: " + iteration + " Loss: " + this.getLoss(testing));
				}
				
				if (iteration++ >= iterations) 
					break outerloop;
			}
		}
	}

	/**
	 * This is an internal method for taking single training step based off of a batch of samples.
	 * @param training - array of training samples to calculate gradients
	 * @param learningRate - floating point scalar multiplier used to scale gradient before adding them to wieghts and bias
	 */
	private void updateWB(LinRegData[] training, float learningRate){
		
		Vector deltaWeights = new Vector(weights.getLength());
		float deltaBias = 0;
		
		for (int i = 0; i < training.length; i++) {
			
			Vector xi = training[i].getData();
			float yi = training[i].getLabel();
				
			float error = yi - this.compute(xi);

			Vector dwi = xi.scaled(-2 * error);
			float dbi = (-2 * error);
				
			deltaWeights = deltaWeights.plus(dwi);
			deltaBias += dbi;
		}
		
		deltaWeights.scale((float) (1.0 / training.length));
		deltaBias /= (float) training.length;
		
		weights = weights.minus(deltaWeights.scaled(learningRate));
		bias = bias - (deltaBias * learningRate);
	}
	
	/**
	 * This calculates the mean square error of the predicted values yhat = mx+b and ground truth.
	 * @param examples - LinRegData array of samples (Vector, float)
	 * @return loss - double precision floating point number
	 */
	@Override
	public double getLoss(LinRegData[] examples) {
	
		double loss = 0.0;

		for (int i = 0; i < examples.length; i++) {
			
			Vector xi = examples[i].getData();
			float yi = examples[i].getLabel();
			
			loss += Math.pow(yi - this.compute(xi), 2);
		}
		
		loss /= examples.length;
		
		return loss;
	}

	/**
	 * This returns a defensive copy of the model's weights.
	 * @return Vector - the weights of the model
	 */
	public Vector getWeights() {
		
		return this.weights.deepCopy();
	}

	/**
	 * This returns a single floating point value of a specific weight by index.
	 * @param i the index of the weight
	 * @return float - the weight value
	 */
	public float getWeightValue(int i) {
		return weights.getValue(i);
	}
	
	/**
	 * This returns the bias / intecept of the model's linear function.
	 * @return float - the b part of y = W*X + b
	 */
	public float getBias() {
		return bias;
	}
	
	/**
	 * This records the weights and bias of the model in a .txt file.
	 * @param name - the directory / file name
	 */
	public void save(String name){
		
		try {
		
			File file = new File(name);
			
			if (file.exists()) {
				throw new Exception();
			}
		
			file.createNewFile();
		
			PrintWriter writer = new PrintWriter(file, "utf-8");
		
			writer.println(this.weights.toString());
			writer.print(this.bias);
		
			writer.close();
		
		} catch (Exception e) {
			System.out.println("Saving failed");
		}
	}

	/**
	 * This forces a weight to a specified value for if you ever need it.
	 * @param i the index of the weight
	 * @param value - that value that the weight is forced to
	 */
	public void forceWeightValue(int i, float value) {
		weights.setValue(i, value);
	}

	/**
	 * This forces the bias to a specified value if you ever need it.
	 * @param value - the value that the weight is forced to
	 */
	public void forceBiasValue(float value) {
		bias = value;
	}

	/**
	 * used for testing
	 * @param args
	 */
	public static void main(String[] args) {

	}
}

