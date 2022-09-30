package logreg;

import interfaces.Model;

import java.io.*;
import math.Vector;
import math.Functions;

public class LogisticRegression implements Model<Vector, Float, LogRegData>{
	
	private Vector weights;
	private float bias;
	
	public LogisticRegression(int features) {
		this.weights = new Vector(features);
		this.bias = 0.0f;
	}
	
	public LogisticRegression(Vector thoseWeights, float bias) {
		this.weights = thoseWeights;
		this.bias = bias;
	}

	@Override
	public Float compute(Vector x) {
		return Functions.sigmoid(weights.dot(x) + bias);
	}
	
	@Override
	public void train(LogRegData[] training, LogRegData[] testing, float learningRate, int epochs, boolean verbose){
		
		for (int e = 0; e < epochs; e++) {
			
			this.updateWB(training, learningRate);
			
			if (verbose) {
				if (e % 100 == 0)
					System.out.println("Iteration: " + e + " Loss: " + this.getLoss(testing));
			}
		}
		
	}
	
	public void train(LogRegData[] training, LogRegData[] testing, float learningRate, int iterations, int batchSize, boolean verbose){
		
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
	
	private void updateWB(LogRegData[] training, float learningRate) {
		
		Vector deltaWeights = new Vector(weights.getLength());
		float deltaBias = 0;
		
		for (int i = 0; i < training.length; i++) {
			
			Vector xi = training[i].getData();
			float yi = training[i].getLabelVal();
			
			float error = yi - this.compute(xi);

			Vector dwi = xi.scaled(-1 * error);
			float dbi = -1 * error;
				
			deltaWeights = deltaWeights.plus(dwi);
			deltaBias += dbi;
		}
		
		deltaWeights.scale((float) (1.0 / training.length));
		deltaBias /= (float) training.length;
		
		weights = weights.minus(deltaWeights.scaled(learningRate));
		bias = bias - (deltaBias * learningRate);
	}
	
	@Override
	public double getLoss(LogRegData[] examples) {
		
		double loss = 0.0;
		
		for(int i = 0; i < examples.length; i++) {
			Vector xi = examples[i].getData();
			float yi = examples[i].getLabelVal();
			
			if(yi == 1.0f){
				loss += -1 * Math.log(this.compute(xi));
			} else {
				loss += -1 * Math.log(1.0f - this.compute(xi));
			}
		}
		
		return loss;
	}
	
	public Vector getWeights() {
		
		Vector result = new Vector(weights.getLength());
		
		for (int i = 0; i < result.getLength(); i++)
			result.setValue(i, weights.getValue(i));
		
		return result;
	}
	
	public float getWeightValue(int i) {
		return weights.getValue(i);
	}
	
	public float getBias() {
		return bias;
	}
	
	public void save(String name){
		
		try {
		
			File file = new File(name);
			
			if (file.exists()) {
				throw new Exception();
			}
		
			file.createNewFile();
		
			PrintWriter writer = new PrintWriter(file, "utf-8");
		
			writer.println(this.weights.asString());
			writer.print(this.bias);
		
			writer.close();
		
		} catch (Exception e) {
			System.out.println("Saving failed");
		}
	}
	
	public void forceWeightValue(int i, float value) {
		weights.setValue(i, value);
	}
	
	public void forceBiasValue(float value) {
		bias = value;
	}
	
	public static void main(String[] args) {
		
	}
}
