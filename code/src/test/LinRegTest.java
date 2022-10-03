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

		//HyperParameters
		
		int dimensions = 100;
		
		int iterations = 1000;
		int batchSize = 100;
		float learningRate = 0.01f;
		
		int trainingSize = 5000;
		int testingSize = 1000;
		
		float noiseScale = .01f;

		boolean verbose = true;
		
		//Generate Target Parameters
		
		Vector targetWeights = new Vector(dimensions);
		for (int i = 0; i < targetWeights.getLength(); i++) 
			targetWeights.setValue(i, (float) (20.0 * (Math.random() -.5)));
		
		float targetBias = 100; 
		
		//Generate Training Data Based off Target Parameters
		
		Vector[] trainingData = new Vector[trainingSize];
		float[] trainingLabels = new float[trainingSize];
		
		for(int i = 0; i < trainingSize; i++) {
			trainingData[i] = new Vector(dimensions);
			trainingData[i].setValuesRandom();
			trainingData[i].scale(10);
			trainingLabels[i] = computeTestLabel(trainingData[i], targetWeights, targetBias, noiseScale);
		}
		LinRegData[] trainingExamples = LinRegData.format(trainingData, trainingLabels);
		
		//Generate Testing Data that is different from the Training Data
		
		Vector[] testingData = new Vector[testingSize];
		float[] testingLabels = new float[testingSize];
		
		for(int i = 0; i < testingSize; i++) {
			testingData[i] = new Vector(dimensions);
			testingData[i].setValuesRandom();
			testingData[i].scale(10);
			testingLabels[i] = computeTestLabel(testingData[i], targetWeights, targetBias, noiseScale);
		}
		LinRegData[] testingExamples = LinRegData.format(testingData, testingLabels);
		
		//Model Creation and Testing
		
		LinearRegression model = new LinearRegression(dimensions);
		model.train(trainingExamples, testingExamples, learningRate, iterations, batchSize, verbose);
		
		//Show Target Parameters compared to Model Parameters
		
		Vector weights = model.getWeights();
		float bias = model.getBias();
		
		for (int i = 0; i < dimensions; i++)
			System.out.println(targetWeights.getValue(i) + "  " + weights.getValue(i));

		System.out.println("\n" + targetBias + "  " + bias);
		
		//Data Export test
		
		/*
		String filename = "ModelSaveTest" + ".txt";
		
		model.save(filename);
		
		//Data Import test
		
		File savedParameters = new File(filename);
		Scanner reader = new Scanner(savedParameters);
		
		LinearRegression model2 = new LinearRegression(dimensions);
		
		for(int i = 0; i < dimensions; i++) {
			model2.forceWeightValue(i, reader.nextFloat());
		}
		
		model2.forceBiasValue(reader.nextFloat());
		
		reader.close();
		
		System.out.println(model2.getLoss(testingExamples));
		System.out.println(model2.getLoss(trainingExamples));
		
		*/
	}

	public static float computeTestLabel(Vector input, Vector targetWeights, float targetBias, float noiseScale) {
	
		float result = targetWeights.dot(input) + targetBias;
		result += (float) ( result * 2 * ( Math.random() - .5 ) * noiseScale);
	
		return result;

	}

}
