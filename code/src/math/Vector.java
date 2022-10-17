package math;

import java.util.Arrays;

public class Vector {

	private final float[] arr;
	
	//Constructors
	
	public Vector(float[] arr) {
		this.arr = new float[arr.length];
		for(int i = 0; i < arr.length; i++) {
			this.arr[i] = arr[i];
		}
	}
	
	public Vector(int length) {
		arr = new float[length];
	}
	
	//Accessors
	
	public int getLength() {
		return arr.length;
	}
	
	public float getValue(int i) {
		return arr[i];
	}
	public Vector deepCopy(){
		return new Vector(this.arr);
	}

	//Mutators
	
	public void setValue(int i, float value) {
		arr[i] = value;
	}
	
	public void setValues(float value) {
		for(int i = 0; i < arr.length; i++) {
			this.arr[i] = value;
		}
	}
	
	public void setValuesRandom() {
		for(int i = 0; i < arr.length; i++) {
			this.arr[i] = (float) ( 2 * ( Math.random() - .5 ) );
		}
	}
	
	//Printers
	
	public void print() {
		for(int i = 0; i < arr.length; i++) {
			System.out.println(getValue(i));
		}
	}
	
	public String toString() {
		
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < this.getLength(); i++)
			sb.append(arr[i] + " ");
	
		sb.deleteCharAt(sb.length() - 1);
		
		return sb.toString();
	}
	
	//Math Functions
	
	public Vector plus(Vector that) {
		
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException();
		
		Vector result = new Vector(this.getLength());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(i) + that.getValue(i));
		}
		
		return result;
	}
	
	public Vector minus(Vector that) {
		
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException();
		
		Vector result = new Vector(this.getLength());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(i) - that.getValue(i));
		}
		
		return result;
	}
	
	public Vector times(Vector that) {
		
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException();
		
		Vector result = new Vector(this.getLength());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(i) * that.getValue(i));
		}
		
		return result;
	}
	
	public Vector divide(Vector that) {
		
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException();
		
		Vector result = new Vector(this.getLength());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(i) / that.getValue(i));
		}
		
		return result;
	}
	
	public float dot(Vector that) {
		
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException();
		
		float result = 0;
		
		for (int i = 0; i < this.getLength(); i++) {
			result += this.getValue(i) * that.getValue(i);
		}
		
		return result;
	}
	
	public Vector dot(Matrix that) {
		
		int x = this.getLength();
		
		if(x != that.getColumnSize())
			throw new IllegalArgumentException();
		
		Vector result = new Vector(that.getRowSize());
		
		for (int i = 0; i < result.getLength(); i++) {
			float total = 0;
			for (int k = 0; k < x; k++) {
				total += this.getValue(k) * that.getValue(k, i);
			}
			result.setValue(i, total);
		}
		
		return result;
	}
	
	public void scale(float scalar) {
		
		for (int i = 0; i < arr.length; i++) {
			arr[i] = arr[i] * scalar;
		}
	}
	
	public Vector scaled(float scalar) {
		
		Vector result = new Vector(arr.length);
		
		for (int i = 0; i < arr.length; i++) {
			result.setValue(i, arr[i] * scalar);
		}
		
		return result;
	}
	
	public static void main(String[] args) {
		
	}
}
