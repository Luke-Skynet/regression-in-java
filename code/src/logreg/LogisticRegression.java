package logreg;

import interfaces.Model;

import java.io.*;
import math.Vector;

/**
 * This provides multifeature logistic regresion R^N -> (0,1),
 * uses SGD to optimize parameters.
 */
public class LogisticRegression implements Model<Vector, Double, LogRegData>{
	
	private Vector weights;
	private double bias;

	/**
	 * Constructor where only the dimension is given and all values are set to default 0. Good for when model will be trained.
	 * @param features - the number of features the model takes in and transforms linearly (Y = sigmoid (wx1 + wx2 + wxn + b))
	 */
	public LogisticRegression(int features) {
		this.weights = new Vector(features);
		this.bias = 0.0;
	}
	/**
	 * Constructor where all the weights and bias are initialized to specific input. This is good for when parameters are already estimated.
	 * @param weights - the weights that are deeply copied and used as parameters 
	 * @param bias - the bias / intecept that is copied
	 */
	public LogisticRegression(Vector weights, double bias) {
		this.weights = weights.deepCopy();
		this.bias = bias;
	}
	/**
	* This is the main inference / computation of the model.
	* @param x this is the input vector X
	* @return scalar Y = sigmoid(W*X + b)
	*/
	@Override
	public Double compute(Vector x) {
		return 1 / (1 + Math.exp( -(weights.dot(x) + bias) ));
	}

	@Override
	public void train(LogRegData[] training, LogRegData[] testing, int batchSize, double learningRate, int epochs, boolean verbose){
		
		if (verbose){
			System.out.println("Creating Batches");
		}
		if (batchSize > training.length)
			throw new IllegalArgumentException("Batch size must not exceed data size");

		int batchCount = training.length / batchSize;
		int spareCount = training.length % batchSize;
		
		int place = 0;
		
		LogRegData[][] batches = new LogRegData[batchCount][batchSize];
		for (int i = 0; i < batchCount; i++) {
			for (int j = 0; j < batchSize; j++) {
				batches[i][j] = new LogRegData(training[place].getData(), training[place].getLabel());
				place++;
			}
		}
		
		LogRegData[] spareBatch = new LogRegData[spareCount];
		if (spareCount > 0) {
			spareBatch = new LogRegData[spareCount];
			for (int i = 0; i < spareBatch.length; i++) {
				spareBatch[i] = new LogRegData(training[place].getData(), training[place].getLabel());
				place++;
			}
		}
		
		if(verbose){
			System.out.println("Starting Training: ");
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
	 * @param learningRate - doubleing point scalar multiplier used to scale gradient before adding them to wieghts and bias
	 */
	private void updateWB(LogRegData[] training, double learningRate) {
		
		Vector deltaWeights = new Vector(weights.getLength());
		double deltaBias = 0;
		
		for (int i = 0; i < training.length; i++) {
			
			Vector xi = training[i].getData();
			double yi = training[i].getLabelVal();
			
			double error = yi - this.compute(xi);

			Vector dwi = xi.scaled(-1 * error);
			double dbi = -1 * error;
				
			deltaWeights = deltaWeights.plus(dwi);
			deltaBias += dbi;
		}
		
		deltaWeights.scale( (1.0 / training.length));
		deltaBias /= training.length;
		
		weights = weights.minus(deltaWeights.scaled(learningRate));
		bias = bias - (deltaBias * learningRate);
	}
	
	/**
	 * This calculates the cross entropy/log loss between the predicted values y' = sigmoid(W*X+b) and ground truth (y).
	 * @param examples - LogRegData array of samples (Vector, double)
	 * @return double - Cross entropy between y' and y
	 */
	@Override
	public double getLoss(LogRegData[] examples) {
		
		double loss = 0.0;
		
		for(int i = 0; i < examples.length; i++) {
			Vector xi = examples[i].getData();
			boolean yi = examples[i].getLabel();
			
			if(yi){
				loss += -1 * Math.log(this.compute(xi));
			} else {
				loss += -1 * Math.log(1.0f - this.compute(xi));
			}
		}
		
		return loss;
	}
	/**
	 * This returns a defensive copy of the model's weights,
	 * @return Vector - the weights of the model
	 */
	public Vector getWeights() {
		return this.weights.deepCopy();
	}

	/**
	 * This returns a single doubleing point value of a specific weight by index.
	 * @param i the index of the weight
	 * @return double - the weight value
	 */
	public double getWeightValue(int i) {
		return weights.getValue(i);
	}
	
	/**
	 * This returns the bias / shift of the model's logistic function.
	 * @return double - the b part of y = sigmoid(W*X + b)
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
	public static LogisticRegression load(String filePath) throws IOException {

		File file = new File(filePath);

		if (!file.exists()){
			throw new IOException("Cannot load model. File does not exists at filePath.");
		}

		BufferedReader br = new BufferedReader(new FileReader(file));
	
		String weightString = br.readLine();
		String biasString = br.readLine();

		br.close();

		try{	

			double bias = Double.parseDouble(biasString);

			String[] tokens = weightString.split("\\s+");
			Vector weights = new Vector(tokens.length);
			for (int i = 0; i < tokens.length; i++){
				weights.setValue(i, Double.parseDouble(tokens[i]));
			}

			return new LogisticRegression(weights, bias);

		} catch (Exception e) {
			throw new IOException("Could not read file contents.");
		}
	}

	/**
	 * This forces a weight to a specified value for if you ever need it.
	 * @param i - the index of the weight
	 * @param value - that value that the weight is forced to
	 */
	public void forceWeightValue(int i, double value) {
		weights.setValue(i, value);
	}

	/**
	 * This forces the bias to a specified value if you ever need it.
	 * @param value - the value that the weight is forced to
	 */
	public void forceBiasValue(double value) {
		bias = value;
	}
	
	public static void main(String[] args) {
		
	}
}
