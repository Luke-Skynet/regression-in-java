package math;

public class Functions {
	
	//Sigmoid Function (individual and Vector)
	
	public static float sigmoid(float x) {
		return 1 / (float) (1 + Math.exp(-x));
	}
	
	public static Vector sigmoid(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, sigmoid(that.getValue(i)));
		}
		
		return vector;
	}
	
	//Tanh Function (individual and Vector)
	
	public static float tanh(float x) {
		return (float) Math.tanh(x);
	}
	
	public static Vector tanh(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, tanh(that.getValue(i)));
		}
		
		return vector;
	}
	
	//ReLU Function (individual and Vector)
	
	public static float ReLU(float x) {
		return Math.max(x, 0);
	}
	
	public static Vector ReLU(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, ReLU(that.getValue(i)));
		}
		
		return vector;
	}
	
	//Swish Function (individual and Vector)
	
	public static float swish(float x) {
		return x / (float) (1 + Math.exp(-x));
	}
	
	public static Vector swish(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, swish(that.getValue(i)));
		}
		
		return vector;
	}
	
	//Softmax Function (just Vector)
	
	public static Vector softMax(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		//compute sum (1 -> z) e^z
		
		float sum = 0f;
		for (int i = 0; i < that.getLength(); i++) {
			sum += (float) Math.exp(that.getValue(i));
		}
		
		//compute e^z / sum
		
		for (int i = 0; i < that.getLength(); i++) {
			vector.setValue(i, (float) Math.exp(that.getValue(i)) / sum);
		}
		
		return vector;
	}

	public static void main(String[] args) {
		
	}
}
