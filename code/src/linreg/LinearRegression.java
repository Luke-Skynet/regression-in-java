package linreg;

import interfaces.Model;

import math.Vector;

import java.io.*;

/**
 * This class provides multilinear regression. R^n -> R. 
 * Instead of using least squares to optimize, it uses gradient descent.
 */
public class LinearRegression implements Model<Vector, Double, LinRegData>{
	
	private Vector weights;
	private double bias;
	
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
	public LinearRegression(Vector weights, double bias) {
		this.weights = weights.deepCopy();
		this.bias = bias;
	}

	/**
	 * The main inference / computation of the linear function
	 * @param x the input vector X
	 * @return scalar Y = W*X + b
	 */
	@Override
	public Double compute(Vector x) {
		return weights.dot(x) + bias;
	}

	@Override
	public void train(LinRegData[] training, LinRegData[] testing, int batchSize, double learningRate, int epochs, boolean verbose){
		
		if (verbose){
			System.out.println("Creating Batches");
		}
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
	
		if (verbose){
			System.out.println("Starting Training:");
		}
	
        for(int e = 1; e <= epochs; e++){
            for (int i = 0; i < batchCount; i++) {
                this.updateWB(training, learningRate);
			}
        
			if (spareCount > 0) {
                this.updateWB(spareBatch, learningRate);
			}

            if (verbose) {
				System.out.println("Epoch: " + e + " Loss: " + this.getLoss(testing));
            }
		}
	}

	/**
	 * This is an internal method for taking single training step based off of a batch of samples.
	 * @param training - array of training samples to calculate gradients
	 * @param learningRate - floating point scalar multiplier used to scale gradient before adding them to wieghts and bias
	 */
	private void updateWB(LinRegData[] training, double learningRate){
		
		Vector deltaWeights = new Vector(weights.getLength());
		double deltaBias = 0;
		
		for (int i = 0; i < training.length; i++) {
			
			Vector xi = training[i].getData();
			double yi = training[i].getLabel();
				
			double error = yi - this.compute(xi);

			Vector dwi = xi.scaled(-2 * error);
			double dbi = (-2 * error);
				
			deltaWeights = deltaWeights.plus(dwi);
			deltaBias += dbi;
		}
		
		deltaWeights.scale((1.0 / training.length));
		deltaBias /= training.length;
		
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
			double yi = examples[i].getLabel();
			
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
	 * @return double - the weight value
	 */
	public double getWeightValue(int i) {
		return weights.getValue(i);
	}
	
	/**
	 * This returns the bias / intecept of the model's linear function.
	 * @return double - the b part of y = W*X + b
	 */
	public double getBias() {
		return bias;
	}
	
	/**
	 * This records the weights and bias of the model in a .txt file.
	 * @param filePath - the absolute path for the new file.
	 * @throws IOException - is thrown if a file already exists at filePath.
	 */
	public void save(String filePath) throws IOException {

		File file = new File(filePath);

		//try to create file and error out if not
		if (!file.createNewFile()){
			throw new IOException("Cannot save model. File already exists at filePath.");
		}

		PrintWriter writer = new PrintWriter(file, "utf-8");

		writer.println(this.weights.toString());
		writer.print(this.bias);

		writer.close();
	}

	/**
	 * This creates a LinearRegression model object using the values from a .txt file
	 * @param filePath - the absolute path location of the file.
	 * @throws IOException - is thrown if the file does not exist, or has trouble reading it.
	 */
	public static LinearRegression load(String filePath) throws IOException {

		File file = new File(filePath);

		if (!file.exists()){
			throw new IOException("Cannot load model. File does not exists at filePath.");
		}

		BufferedReader br = new BufferedReader(new FileReader(file));
	
		String weightString = br.readLine();
		String biasString = br.readLine();

		br.close();

		try{	

			double bias = Float.parseFloat(biasString);

			String[] tokens = weightString.split("\\s+");
			Vector weights = new Vector(tokens.length);
			for (int i = 0; i < tokens.length; i++){
				weights.setValue(i, Float.parseFloat(tokens[i]));
			}

			return new LinearRegression(weights, bias);

		} catch (Exception e) {
			throw new IOException("Could not read file contents.");
		}
	}

	/**
	 * This forces a weight to a specified value for if you ever need it.
	 * @param i - the index of the weight
	 * @param value - the value that the weight is forced to
	 */
	public void forceWeightValue(int i, double value) {
		this.weights.setValue(i, value);
	}

	/**
	 * This forces the bias to a specified value for if you ever need it.
	 * @param value - the value that the weight is forced to
	 */
	public void forceBiasValue(double value) {
		this.bias = value;
	}

	/**
	 * used for testing
	 * @param args
	 */
	public static void main(String[] args) {

	}
}

