package test;

import math.Functions;
import math.Matrix;
import math.Vector;

public class LayerTest{

	public static void main(String[] args) {
		
		// I/O Layer Dimensions
		
		int inputLayerSize = 5;
		
		int outputLayerSize = 3;
		
		// Create input vector x
		
		Vector vectorInput = new Vector(inputLayerSize);
		vectorInput.setValuesRandom();
		
		// Create Matrix M
		
		Matrix matrix = new Matrix(outputLayerSize, inputLayerSize);
		matrix.setValuesRandom();
		
		// Create b
		
		Vector biasLayer = new Vector(outputLayerSize);
		biasLayer.setValuesRandom();
		
		// Compute M*x
		
		Vector matrixOutput = matrix.dot(vectorInput);
		
		// Compute Z = M*x + b
		
		Vector result = matrixOutput.plus(biasLayer);
		
		// Compute a = g(Z)
		
		Vector activatedResult = Functions.sigmoid(result);
		
		// Print Everything
		
		vectorInput.print();
		System.out.println();
		
		matrix.print();
		System.out.println();
		
		matrixOutput.print();
		System.out.println();
		
		biasLayer.print();
		System.out.println();
		
		result.print();
		System.out.println();
		
		activatedResult.print();
		System.out.println();
		
		// Full a = g(M*x + b) because it looks dope
		
		Vector a = Functions.sigmoid(matrix.dot(vectorInput).plus(biasLayer));
		a.print();
	}
}
