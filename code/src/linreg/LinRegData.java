package linreg;

import interfaces.Sample;

import math.Vector;

/**
 * This class defines the data type for Linear Regression.
 * It is an object that bundles together an input vector (data) and
 * an output scalar double value representing the ground truth. 
 */
public class LinRegData implements Sample<Vector, Double>{
	
	private final Vector data;
	private final double label;

	/**
	 * basic constructor with the data and label
	 * @param data - Vector (X)
	 * @param label - double (Y)
	 */
	public LinRegData(Vector data, double label) {
		this.data = data;
		this.label = label;
	}
	
	/**
	 * @return Vector - Data (X)
	 */
	@Override
	public Vector getData() {
		return data;
	}

	/**
	 * @return double - scalar (Y)
	 */
	@Override
	public Double getLabel(){
		return label;
	}
	/**
	 * This is a static method that reorganizes arrays into linreg data.
	 * @param xValues two dimensional array where the first index denotes each vector
	 * @param yValues one dimensional array representing the labels
	 * @return LinRegData array
	 */
	public static LinRegData[] format(double[][] xValues, double[] yValues) {
		
		Vector[] vectors = new Vector[xValues.length];
		
		for (int i = 0; i < xValues.length; i++) {
			vectors[i] = new Vector(xValues[i]);
		}
		
		LinRegData[] data = new LinRegData[vectors.length];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new LinRegData(vectors[i], yValues[i]);
		}
		
		return data;
	}
	/**
	 * This is a static method that bundles an array of vectors and an array of labels into an array of linregdata samples,
	 * each index is buldled together (v[i[,l[i]]).
	 * @param vectors - the array of vectors
	 * @param labels - the array of labels
	 * @return
	 */
	public static LinRegData[] format(Vector[] vectors, double[] labels) {
		
		LinRegData[] data = new LinRegData[vectors.length];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new LinRegData(vectors[i], labels[i]);
		}
		
		return data;
	}
}