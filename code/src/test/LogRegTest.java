package test;

import math.Functions;
import math.Vector;
import logreg.LogisticRegression;
import logreg.LogRegData;

public class LogRegTest {

	public static void main(String[] args){

		//HyperParameters
		
		int dimensions = 100;
		
		int iterations = 10000;
		int batchsize = 500;
		float learningRate = 0.1f;
		
		int trainingSize = 5000;
		int testingSize = 1000;
		
		float noiseScale = .1f;

		boolean verbose = true;
		
		//Generate Target Parameters
		
		Vector targetWeights = new Vector(dimensions);
		for (int i = 0; i < targetWeights.getLength(); i++) 
			targetWeights.setValue(i, (float) (2.0 * (Math.random() -.5)));
		
		float targetBias = 5; 
		
		//Generate Training Data Based off Target Parameters
		
		Vector[] trainingData = new Vector[trainingSize];
		float[] trainingLabels = new float[trainingSize];
		
		for(int i = 0; i < trainingSize; i++) {
			trainingData[i] = new Vector(dimensions);
			trainingData[i].setValuesRandom();
			trainingData[i].scale(10);
			trainingLabels[i] = computeTestLabel(trainingData[i], targetWeights, targetBias, noiseScale);
		}
		LogRegData[] trainingExamples = LogRegData.format(trainingData, trainingLabels);
		
		//Generate Testing Data that is different from the Training Data
		
		Vector[] testingData = new Vector[testingSize];
		float[] testingLabels = new float[testingSize];
		
		for(int i = 0; i < testingSize; i++) {
			testingData[i] = new Vector(dimensions);
			testingData[i].setValuesRandom();
			testingData[i].scale(10);
			testingLabels[i] = computeTestLabel(testingData[i], targetWeights, targetBias, noiseScale);
		}
		LogRegData[] testingExamples = LogRegData.format(testingData, testingLabels);
		
		//Model Creation and Testing
		
		LogisticRegression model = new LogisticRegression(dimensions);
		model.train(trainingExamples, testingExamples, learningRate, iterations, batchsize, verbose);

		//Show Target Parameters compared to Model Parameters
		
		Vector weights = model.getWeights();
		float bias = model.getBias();
		
		System.out.println("\nTarget - Learned");

		for (int i = 0; i < dimensions; i++)
			System.out.println(targetWeights.getValue(i) + "  " + weights.getValue(i));

		System.out.println("\n" + targetBias + "  " + bias);
	}

	public static float computeTestLabel(Vector input, Vector targetWeights, float targetBias, float noiseScale) {
		
		float mxb = targetWeights.dot(input) + targetBias;
		float noise = 1.0f - 2 * noiseScale * (float) Math.random();

		return Functions.sigmoid(mxb * noise);

	}

}
