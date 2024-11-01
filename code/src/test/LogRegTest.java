package test;

import math.Vector;
import logreg.LogisticRegression;

import java.io.IOException;

import logreg.LogRegData;

public class LogRegTest {

	public static void main(String[] args) throws IOException{

		boolean exportTest = false;
		String exportPath = "model.txt";

		//HyperParameters
		
		int dimensions = 100;
		
		int epochs = 1000;
		int batchsize = 500;
		double learningRate = 0.1;
		
		int trainingSize = 5000;
		int testingSize = 1000;
		
		double noiseScale = .1;

		boolean verbose = true;
		
		//Generate Target Parameters
		
		Vector targetWeights = new Vector(dimensions);
		for (int i = 0; i < targetWeights.getLength(); i++) 
			targetWeights.setValue(i, (2.0 * (Math.random() -.5)));
		
		double targetBias = 5; 
		
		//Generate Training Data Based off Target Parameters
		
		Vector[] trainingData = new Vector[trainingSize];
		double[] trainingLabels = new double[trainingSize];
		
		for(int i = 0; i < trainingSize; i++) {
			trainingData[i] = new Vector(dimensions);
			trainingData[i].setValuesRandom();
			trainingData[i].scale(10);
			trainingLabels[i] = computeTestLabel(trainingData[i], targetWeights, targetBias, noiseScale);
		}
		LogRegData[] trainingExamples = LogRegData.format(trainingData, trainingLabels);
		
		//Generate Testing Data that is different from the Training Data
		
		Vector[] testingData = new Vector[testingSize];
		double[] testingLabels = new double[testingSize];
		
		for(int i = 0; i < testingSize; i++) {
			testingData[i] = new Vector(dimensions);
			testingData[i].setValuesRandom();
			testingData[i].scale(10);
			testingLabels[i] = computeTestLabel(testingData[i], targetWeights, targetBias, noiseScale);
		}
		LogRegData[] testingExamples = LogRegData.format(testingData, testingLabels);
		
		//Model Creation and Testing
		
		LogisticRegression model = new LogisticRegression(dimensions);
		model.train(trainingExamples, testingExamples, batchsize, learningRate, epochs,  verbose);

		//Show Target Parameters compared to Model Parameters
		
		Vector weights = model.getWeights();
		double bias = model.getBias();
		
		System.out.println("\nTarget - Learned");

		for (int i = 0; i < dimensions; i++)
			System.out.println(targetWeights.getValue(i) + "  " + weights.getValue(i));

		System.out.println("\n" + targetBias + "  " + bias + "\n");

		if (exportTest){

			model.save(exportPath);
		
			LogisticRegression model2 = LogisticRegression.load(exportPath);
		
			System.out.println(model2.getLoss(testingExamples));
			System.out.println(model2.getLoss(trainingExamples));
		}
	}

	public static double computeTestLabel(Vector input, Vector targetWeights, double targetBias, double noiseScale) {
		
		double mxb = targetWeights.dot(input) + targetBias;
		double noise = 1.0 - 2 * noiseScale * Math.random();

		return 1 / (1 + Math.exp(mxb * noise));

	}

}
