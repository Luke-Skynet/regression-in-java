package logreg;

import interfaces.Sample;

import math.Vector;
/**
 * This class defines the data type for Logistic Regression.
 * It is an object that bundles together an input vector (data) and
 * an output boolean value representing the ground truth positice (1) and negative (0) label. 
 */

public class LogRegData implements Sample<Vector, Boolean>{
	
	private final Vector data;
	private final boolean label;
	
	/**
	 * Basic constructor with the data and label.
	 * @param data - Vector (X)
	 * @param label - boolean (Y)
	 */
	public LogRegData(Vector data, boolean label) {
		this.data = data;
		this.label = label;
	}

	/**
	 * Another constructor with the data and label as a float.
	 * @param data - Vector (X)
	 * @param label - float to boolean true if > .f
	 */
	public LogRegData(Vector data, float label) {
		this.data = data;
		this.label = label > .5f;
	}

	/**
	* @return Vector - Data (X)
	*/
	@Override
	public Vector getData() {
		return data;
	}
	
	/**
	 * @return boolean - label (Y)
	 */
	@Override
	public Boolean getLabel(){
		return label;
	}
	
	/**
	 * @return int - label (Y)
	 */
	public int getLabelNum(){
		return label ? 1 : 0;
	}
	
	/**
	 * @return float - label (Y)
	 */
	public float getLabelVal(){
		return label ? 1f : 0f;
	}

	/**
	 * This is a static method that reorganizes arrays into logreg data.
	 * @param xValues two dimensional array where the first index denotes each vector
	 * @param yValues one dimensional array representing the labels
	 * @return
	 */
	public static LogRegData[] format(float[][] xValues, boolean[] yValues) {
		
		Vector[] vectors = new Vector[xValues.length];
		
		for (int i = 0; i < xValues.length; i++) {
			vectors[i] = new Vector(xValues[i]);
		}
		
		LogRegData[] data = new LogRegData[vectors.length];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new LogRegData(vectors[i], yValues[i]);
		}
		
		return data;
	}
	
	/**
	 * This is a static method that bundles an array of vectors and an array of labels into an array of logregdata samples,
	 * each index is buldled together (v[i],l[i]).
	 * @param vectors - the array of vectors
	 * @param labels - the array of labels
	 * @return LogRegData array
	 */
	public static LogRegData[] format(Vector[] vectors, boolean[] labels) {
		
		LogRegData[] data = new LogRegData[vectors.length];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new LogRegData(vectors[i], labels[i]);
		}
		
		return data;
	}
	
	/**
	 * This is a static method that reorganizes arrays into logreg data.
	 * @param xValues two dimensional array where the first index denotes each vector
	 * @param yValues one dimensional array representing the labels
	 * @return LogRegData array
	 */
	public static LogRegData[] format(float[][] xValues, float[] yValues) {
		
		Vector[] vectors = new Vector[xValues.length];
		
		for (int i = 0; i < xValues.length; i++) {
			vectors[i] = new Vector(xValues[i]);
		}
		
		LogRegData[] data = new LogRegData[vectors.length];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new LogRegData(vectors[i], yValues[i]);
		}
		
		return data;
	}
	
	/**
	 * This is a static method that bundles an array of vectors and an array of labels into an array of logregdata samples,
	 * each index is buldled together (v[i[,l[i]]).
	 * @param vectors - the array of vectors
	 * @param labels - the array of labels
	 * @return LogRegData array
	 */
	public static LogRegData[] format(Vector[] vectors, float[] labels) {
		
		LogRegData[] data = new LogRegData[vectors.length];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new LogRegData(vectors[i], labels[i]);
		}
		
		return data;
	}
}

