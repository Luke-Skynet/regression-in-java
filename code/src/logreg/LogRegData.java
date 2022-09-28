package logreg;

import interfaces.Sample;

import math.Vector;

public class LogRegData implements Sample{
	
	private final Vector data;
	private final boolean label;
	
	public LogRegData(Vector data, boolean label) {
		this.data = data;
		this.label = label;
	}
	public LogRegData(Vector data, float label) {
		this.data = data;
		this.label = label > (float) Math.random();
	}
	public Vector getData() {
		return data;
	}
	
	public float getDataValue(int i) {
		return data.getValue(i);
	}
	
	public boolean getLabel(){
		return label;
	}
	
	public int getLabelNum(){
		return label ? 1 : 0;
	}
	
	public float getLabelVal(){
		return label ? 1f : 0f;
	}
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
	
	public static LogRegData[] format(Vector[] vectors, boolean[] labels) {
		
		LogRegData[] data = new LogRegData[vectors.length];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new LogRegData(vectors[i], labels[i]);
		}
		
		return data;
	}
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
	
	public static LogRegData[] format(Vector[] vectors, float[] labels) {
		
		LogRegData[] data = new LogRegData[vectors.length];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new LogRegData(vectors[i], labels[i]);
		}
		
		return data;
	}
}

