package linreg;

import interfaces.Sample;

import math.Vector;

public class LinRegData implements Sample<Vector, Float>{
	
	private final Vector data;
	private final float label;
	
	public LinRegData(Vector data, float label) {
		this.data = data;
		this.label = label;
	}
	
	public Vector getData() {
		return data;
	}
	
	public int getDim() {
		return data.getLength();
	}
	
	public float getDataValue(int i) {
		return data.getValue(i);
	}
	
	public Float getLabel(){
		return label;
	}

	public static LinRegData[] format(float[][] xValues, float[] yValues) {
		
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
	
	public static LinRegData[] format(Vector[] vectors, float[] labels) {
		
		LinRegData[] data = new LinRegData[vectors.length];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new LinRegData(vectors[i], labels[i]);
		}
		
		return data;
	}
}