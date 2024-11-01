package test;

import math.Vector;
import linreg.LinearRegression;
import linreg.LinRegData;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

public class LinRegTest {

	public static void main(String[] args) throws IOException {

		boolean exportTest = false;
		String exportPath = "model.txt";

		//HyperParameters
		
		int dimensions = 100;
		
		int epochs = 10;
		int batchSize = 100;
		double learningRate = 0.01f;
		
		int trainingSize = 5000;
		int testingSize = 1000;
		
		double noiseScale = .01f;

		boolean verbose = true;
		
		//Generate Target Parameters
		
		Vector targetWeights = new Vector(dimensions);
		for (int i = 0; i < targetWeights.getLength(); i++) 
			targetWeights.setValue(i, (20.0 * (Math.random() -.5)));
		
		double targetBias = 100; 
		
		//Generate Training Data Based off Target Parameters
		
		Vector[] trainingData = new Vector[trainingSize];
		double[] trainingLabels = new double[trainingSize];
		
		for(int i = 0; i < trainingSize; i++) {
			trainingData[i] = new Vector(dimensions);
			trainingData[i].setValuesRandom();
			trainingData[i].scale(10);
			trainingLabels[i] = computeTestLabel(trainingData[i], targetWeights, targetBias, noiseScale);
		}
		LinRegData[] trainingExamples = LinRegData.format(trainingData, trainingLabels);
		
		//Generate Testing Data that is different from the Training Data
		
		Vector[] testingData = new Vector[testingSize];
		double[] testingLabels = new double[testingSize];
		
		for(int i = 0; i < testingSize; i++) {
			testingData[i] = new Vector(dimensions);
			testingData[i].setValuesRandom();
			testingData[i].scale(10);
			testingLabels[i] = computeTestLabel(testingData[i], targetWeights, targetBias, noiseScale);
		}
		LinRegData[] testingExamples = LinRegData.format(testingData, testingLabels);
		
		//Model Creation and Testing
		
		LinearRegression model = new LinearRegression(dimensions);
		model.train(trainingExamples, testingExamples, batchSize, learningRate, epochs, verbose);
		
		//Show Target Parameters compared to Model Parameters
		
		Vector weights = model.getWeights();
		double bias = model.getBias();
		
		System.out.println("\nTarget - Learned");
		for (int i = 0; i < dimensions; i++)
			System.out.println(targetWeights.getValue(i) + "  " + weights.getValue(i));

		System.out.println("\n" + targetBias + "  " + bias + "\n");
		
		if (exportTest){

			model.save(exportPath);
		
			LinearRegression model2 = LinearRegression.load(exportPath);
		
			System.out.println(model2.getLoss(testingExamples));
			System.out.println(model2.getLoss(trainingExamples));
		}
	}

	public static double computeTestLabel(Vector input, Vector targetWeights, double targetBias, double noiseScale) {
	
		double result = targetWeights.dot(input) + targetBias;
		result += ( result * 2 * ( Math.random() - .5 ) * noiseScale);
	
		return result;

	}

}
