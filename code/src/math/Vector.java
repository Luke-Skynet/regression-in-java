package math;


/**
 * This class provides a vector datatype of floats. V = f^n
 * NOTE: Although these vectors interact with matrices a lot, they do NOT have a definitive orientation.
 * Each member function specifies when needed whether the vector acts as a column or a row.
 */
public class Vector {

	private final float[] arr;
	
	//Constructors
	
	/**
	 * This constructor creates a vector object using the values from a float array.
	 * @param arr - the float array that is deeply copied.
	 */
	public Vector(float[] arr) {
		this.arr = new float[arr.length];
		for(int i = 0; i < arr.length; i++) {
			this.arr[i] = arr[i];
		}
	}
	
	/**
	 * This constructore creates a vector object with a specified length and all values set to default (0).
	 * @param length int - the length / dimension of the vector
	 */
	public Vector(int length) {
		arr = new float[length];
	}
	
	//Accessors
	
	/**
	 * This returns the dimension of the vector float^n -> n.
	 * @return int - natural number
	 */
	public int getLength() {
		return arr.length;
	}
	
	/**
	 * This returns the scalar value at the specified index.
	 * @param i int - the index
	 * @return float - value at index i
	 */
	public float getValue(int i) {
		return arr[i];
	}
	/**
	 * This returns a new Vector with the same values as the vector
	 * @return vector - replica of the copied vector
	 */
	public Vector deepCopy(){
		return new Vector(this.arr);
	}

	//Mutators
	
	/**
	 * This mutates specific element in the vector.
	 * @param i int - the index of the mutated value
	 * @param value float - the new value at the index
	 */
	public void setValue(int i, float value) {
		arr[i] = value;
	}
	
	/**
	 * This sets all the elements in the vector to a single value.
	 * @param value float - the value to set all elements to
	 */
	public void setValues(float value) {
		for(int i = 0; i < arr.length; i++) {
			this.arr[i] = value;
		}
	}

	/**
	 * This sets all the elements to random numbers between (-1,1)
	 */
	public void setValuesRandom() {
		for(int i = 0; i < arr.length; i++) {
			this.arr[i] = (float) ( 2 * ( Math.random() - .5 ) );
		}
	}
	
	//Printers
	/**
	 * This prints out all the values of the vector to console.
	 */
	public void print() {
		for(int i = 0; i < arr.length; i++) {
			System.out.print(this.getValue(i) + " ");
		}
	}
	
	/**
	 * This returns all the elements of the vector with spaces in between them.
	 */
	public String toString() {
		
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < this.getLength(); i++)
			sb.append(arr[i] + " ");
	
		sb.deleteCharAt(sb.length() - 1);
		
		return sb.toString();
	}
	
	//Math Functions
	/**
	 * Element-wise vector addition.
	 * @param that - another vector of the same dimension
	 * @return vector C = A + B
	 */
	public Vector plus(Vector that) {
		
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException();
		
		Vector result = new Vector(this.getLength());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(i) + that.getValue(i));
		}
		
		return result;
	}
	
	/**
	 * Element-wise vector subtraction.
	 * @param that - another vector of the same dimension
	 * @return vector C = A - B
	 */
	public Vector minus(Vector that) {
		
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException();
		
		Vector result = new Vector(this.getLength());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(i) - that.getValue(i));
		}
		
		return result;
	}
	
	/**
	 * Element-wise vector multiplication.
	 * @param that - another vector of the same dimension
	 * @return Ci = Ai * Bi for i in vector
	 */
	public Vector times(Vector that) {
		
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException();
		
		Vector result = new Vector(this.getLength());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(i) * that.getValue(i));
		}
		
		return result;
	}
	/**
	 * Element-wise vector division.
	 * @param that - another vector of the same dimension
	 * @return Ci = Ai / Bi for i in vector
	 */
	public Vector divide(Vector that) {
		
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException();
		
		Vector result = new Vector(this.getLength());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(i) / that.getValue(i));
		}
		
		return result;
	}
	
	/**
	 * Typical Vector dot product.
	 * @param that - aother vector of the same dimension
	 * @return float y = sum(Ai * bi) for i in vectors A,B
	 */
	public float dot(Vector that) {
		
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException();
		
		float result = 0;
		
		for (int i = 0; i < this.getLength(); i++) {
			result += this.getValue(i) * that.getValue(i);
		}
		
		return result;
	}
	
	/**
	 * Inplace scaling of each element by a scalar.
	 * @param scalar - float scalar k in V -> kV
	 */
	public void scale(float scalar) {
		
		for (int i = 0; i < arr.length; i++) {
			arr[i] = arr[i] * scalar;
		}
	}
	
	/**
	 * Non-inplace scaling of each element by a float scalar.
	 * @param scalar - float scalar k in kV
	 * @return vector W = kV
	 */
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
