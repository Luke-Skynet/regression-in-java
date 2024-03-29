package math;

/**
 * This class provides static functions of those commonly used in Neural Nets, along with their derivatives.
 * Each function has a scalar version and a vector version, (except for softmax).
 * Vectors computations are always performed element - element wise.
 */
public class Functions {
	
	//Sigmoid Function (individual and Vector)
	
	/**
	 * 
	 * @param x - input variable
	 * @return 1 / (1 + e^-x)
	 */
	public static float sigmoid(float x) {
		return 1 / (float) (1 + Math.exp(-x));
	}
	/**
	 * 
	 * @param that - input vector(x^n)
	 * @return that(xn) = 1 / (1 + e^-xn)
	 */
	public static Vector sigmoid(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, sigmoid(that.getValue(i)));
		}
		
		return vector;
	}
	/**
	 * 
	 * @param x - input variable
	 * @return sigmoid(x)*(1-sigmoid(x))
	 */
	public static float derSigmoid(float x){
		return sigmoid(x) * (1.0f - sigmoid(x));
	}
	/**
	 * 
	 * @param that - input vector(x^n)
	 * @return that(xn) = sigmoid(x)*(1-sigmoid(x))
	 */
	public static Vector derSigmoid(Vector that){
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, derSigmoid(that.getValue(i)));
		}
		
		return vector;
	}

	//Tanh Function (individual and Vector)
	/**
	 * 
	 * @param x - input variable
	 * @return exp(x) - exp(-x) / exp(x) + exp(-x)
	 */
	public static float tanh(float x) {
		return (float) Math.tanh(x);
	}
	/**
	 * 
	 * @param that - input vector(x^n)
	 * @return that(xn) = exp(x) - exp(-x) / exp(x) + exp(-x)
	 */
	public static Vector tanh(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, tanh(that.getValue(i)));
		}
		
		return vector;
	}
	/**
	 * 
	 * @param x - input variable
	 * @return 1 - tanh^2(x)
	 */
	public static float derTanh(float x) {
		return (float) (1.0 - Math.pow(Math.tanh(x), 2));
	}
	/**
	 * 
	 * @param that - input vector(x^n)
	 * @return that(xn) = 1 - tanh^2(x)
	 */
	public static Vector derTanh(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, derTanh(that.getValue(i)));
		}
		
		return vector;
	}

	//ReLU Function (individual and Vector)
	/**
	 * 
	 * @param x - input variable
	 * @return max(0,x)
	 */
	public static float ReLU(float x) {
		return Math.max(x, 0);
	}
	/**
	 * 
	 * @param that - input vector(x^n)
	 * @return that(xn) = max(0,x)
	 */
	public static Vector ReLU(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, ReLU(that.getValue(i)));
		}
		
		return vector;
	}
	/**
	 * 
	 * @param x - input vector(x^n)
	 * @return .5 sign(x) + .5 -- (0,1)
	 */
	public static float derReLU(float x){
		if (x >= 0){
			return 1.0f;
		} else {
			return 0.0f;
		}
	}
	/**
	 * 
	 * @param that - input vector(x^n)
	 * @return that(xn) = .5 sign(x) + .5 -- (0,1)
	 */
	public static Vector derReLU(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, derReLU(that.getValue(i)));
		}
		
		return vector;
	}

	//Swish Function (individual and Vector)
	/**
	 * 
	 * @param x - input variable
	 * @return x / (1 + e^-x)
	 */
	public static float swish(float x) {
		return x / (float) (1 + Math.exp(-x));
	}
	/**
	 * 
	 * @param that - input vector(x^n)
	 * @return that(xn) = x / (1 + e^-x)
	 */
	public static Vector swish(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, swish(that.getValue(i)));
		}
		
		return vector;
	}
	/**
	 * 
	 * @param x - input variable
	 * @return sig(x) + swish(x)*(1-sig(x))
	 */
	public static float derSwish(float x) {
		return sigmoid(x) + swish(x)*(1.0f - sigmoid(x));
	}
	/**
	 * 
	 * @param that - input vector(x^n)
	 * @return that(xn) = sig(x) + swish(x)*(1-sig(x))
	 */
	public static Vector derSwish(Vector that) {
		
		Vector vector = new Vector(that.getLength());
		
		for (int i = 0; i < vector.getLength(); i++) {
			vector.setValue(i, derSwish(that.getValue(i)));
		}
		
		return vector;
	}

	//Softmax Function (just Vector)
	
	/**
	 * 
	 * @param that - input vector(x^n)
	 * @return that(zi) = e^zi / sum(e^zi) for i in vector
	 */
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

	/**
	 * 
	 * @param that - input vector(x^n)
	 * @return matrix( (i, j = i) -> sm(i)(1-sm(i)), else (i,j) -> -sm(i)sm(j) )
	 */
	public static Matrix derSoftMax(Vector that){

		int dims = that.getLength();
		Matrix matrix = new Matrix(dims, dims);

		Vector sm = softMax(that);

		for(int i = 0; i < dims; i++){
			for (int j = 0; j < dims; j++){
				if (i == j){
					matrix.setValue(i, i, sm.getValue(i) * (1.0f - sm.getValue(i)));
				} else {
					matrix.setValue(i, j, -1.0f * sm.getValue(i) * sm.getValue(j));
				}
			}
		}
		return matrix;
	}

	public static void main(String[] args) {
		
	}
}
