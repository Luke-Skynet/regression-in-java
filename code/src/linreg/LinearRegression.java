package linreg;

import interfaces.Model;

import java.io.*;
import math.Vector;

public class LinearRegression implements Model<Vector, Float, LinRegData>{
	
	private Vector weights;
	private float bias;
	
	public LinearRegression(int dimensions) {
		this.weights = new Vector(dimensions);
		this.bias = 0;
	}
	
	public LinearRegression(Vector thoseWeights, float bias) {
		this.weights = thoseWeights;
		this.bias = bias;
	}
	
	public Float compute(Vector x) {
		return weights.dot(x) + bias;
	}

	public void train(LinRegData[] training, LinRegData[] testing, float learningRate, int epochs, boolean verbose){

		for (int e = 0; e < epochs; e++) {
			
			this.updateWB(training, learningRate);
			
			if (verbose) {
				if (e % 100 == 0)
					System.out.println("Iteration: " + e + " Loss: " + this.getLoss(testing));
			}
		}
		
	}
	
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

