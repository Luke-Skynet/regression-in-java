package math;

/**
 * This class provides a matrix data type (mat = float(m,n)). 
 */
public class Matrix {

	private final float[][] arr;

	//Constructors

	/**
	 * This constructs a matrix object by defensively copying the contents of a 2d rectangular float array.
	 * @param arr - a non jagged 2d float array
	 */
	public Matrix(float[][] arr) {
	
		for(int i = 0; i < arr.length; i++){
			if (arr[0].length != arr[i].length)
				throw new IllegalArgumentException("Cannot create Matrix - 2D array is jagged");
		}
		
		this.arr = new float[arr.length][arr[0].length];
		for(int i = 0; i < arr.length; i++) {
			for(int j = 0; j < arr[0].length; j++) {
				this.arr[i][j] = arr[i][j];
			}
		}
	}
	
	/**
	 * This constructs a matrix object full of default values (0) given the dimensions (m,n).
	 * @param columnSize - number of rows in the matrix (m)
	 * @param rowSize - number of colums in the matrix (n)
	 */
	public Matrix(int columnSize, int rowSize) {
		arr = new float[columnSize][rowSize];
	}
	
	//Accessors and Mutators

	/**
	 * @return int - row count / column size (m) of matrix
	 */
	public int getColumnSize(){
		return arr.length;
	}
	
	/**
	 * @return int - column count / row size (n) of matrix
	 */
	public int getRowSize() {
		return arr[0].length;
	}
	
	/**
	 * This modifies a single value in the Matrix, given its index (i,j) or (m,n).
	 * @param i - first index (m)
	 * @param j - second index (n)
	 * @param value - the new floating point value in Mat(i,j)
	 */
	public void setValue(int i, int j, float value) {
		arr[i][j] = value;
	}
	
	/**
	 * This sets all of the values in the matrix to a random number between (-1, 1).
	 */
	public void setValuesRandom() {
		for (int i = 0; i < this.getColumnSize(); i++) {
			for (int j = 0; j < this.getRowSize(); j++) {
				this.setValue(i, j, (float) ( 2 * (Math.random() - .5) ) );
			}
		}
	}
	
	/**
	 * This accesses a single value in the matrix, given its index (i,j) or (m,n).
	 * @param i - first index (m)
	 * @param j - second index (n)
	 * @return float - value at mat(i,j)
	 */
	public float getValue(int i, int j) {
		return arr[i][j];
	}
	
	/**
	 * This returns the (jth) column in the Matrix as a 1D Vector.
	 * @param column - the index of the column
	 * @return Vector - defensively copied Vector from the column array in the matrix
	 */
	public Vector getColumnVector(int column) {
		
		Vector result = new Vector(this.getColumnSize());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(i, column));
		}
		
		return result;
	}
	
	/**
	 * This returns the (ith) row in the Matrix as a 1D Vector.
	 * @param row - the index of the row
	 * @return Vector - Defensively copied Vector from the row array in the matrix
	 */
	public Vector getRowVector(int row) {
		
		Vector result = new Vector(this.getRowSize());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(row, i));
		}
		
		return result;
	}

	/**
	 * This returns a whole new deep copy of the matrix, seperate from the original.
	 * @return Matrix - a replica of the matrix with the same values
	 */
	public Matrix deepCopy(){
		return new Matrix(this.arr);
	}
	//printer
	
	/**
	 * This prints out the array with each line as a row.
	 */
	public void print() {
		for (int i = 0; i < this.getColumnSize(); i++) {
			for (int j = 0; j < this.getRowSize(); j++) {
				System.out.print(this.getValue(i, j) + " ");
			}
			System.out.println();;
		}
	}
	
	//Math Functions
	/**
	 * Element wise matrix addition.
	 * @param that - another matrix of the same dimension
	 * @return matrix C = A + B
	 */
	public Matrix plus(Matrix that) {
		
		if(this.getColumnSize() != that.getColumnSize() || this.getRowSize() != that.getRowSize())
			throw new IllegalArgumentException("Matrix dimensions do not match.");
		
		int columnsize = this.getColumnSize();
		int rowsize = this.getRowSize();
		
		Matrix result = new Matrix(columnsize, rowsize);
		
		for (int i = 0; i < columnsize; i++) {
			for (int j = 0; j < rowsize; j++) {
				result.setValue(i, j, this.getValue(i, j) + that.getValue(i, j));
			}
		}
		
		return result;
	}
	
	/**
	 * Element wise matrix subtraction.
	 * @param that - another matrix of the same dimension
	 * @return Matrix C = A - B
	 */
	public Matrix minus(Matrix that) {
		
		if(this.getColumnSize() != that.getColumnSize() || this.getRowSize() != that.getRowSize())
			throw new IllegalArgumentException();
		
		int columnsize = this.getColumnSize();
		int rowsize = this.getRowSize();
		
		Matrix result = new Matrix(columnsize, rowsize);
		
		for (int i = 0; i < columnsize; i++) {
			for (int j = 0; j < rowsize; j++) {
				result.setValue(i, j, this.getValue(i, j) - that.getValue(i, j));
			}
		}
		
		return result;
	}
	
	/**
	 * Linear Algebra dot product/composition operation. (non-commutative)
	 * @param that - another matrix where ||this(j)|| = ||that(i)||
	 * @return matrix C = A * B
	 */
	public Matrix dot(Matrix that){

		if (this.getRowSize() != that.getColumnSize())
			throw new IllegalArgumentException();
		
		int columnsize = this.getColumnSize();
		int rowsize = that.getRowSize();
		
		Matrix result = new Matrix(columnsize, rowsize);

		for (int i = 0; i < columnsize; i++) {
			for (int j = 0; j < rowsize; j++) {
				result.setValue(i,j, this.getRowVector(i).dot(that.getColumnVector(j)));
			}
		}
		
		return result;

	}
	
	/**
	 * Matrix transformation of a Vector
	 * @param that vector input where dim(vec) = dim(mat(n))
	 * @return vector B = Ax
	 */
	public Vector dot(Vector that) {
		
		int x = this.getRowSize();
		
		if(x != that.getLength())
			throw new IllegalArgumentException();
		
		Vector result = new Vector(this.getColumnSize());
		
		for (int i = 0; i < result.getLength(); i++) {
			float total = 0f;
			for (int k = 0; k < x; k++) {
				total += this.getValue(i, k) * that.getValue(k);
			}
			result.setValue(i, total);
		}
		
		return result;
	}
	
	/**
	 * Element wise multiplication between two matrices.
	 * @param that - another matrix of the same dimension
	 * @return - matrix C = A ox B
	 */
	public Matrix eleMult(Matrix that) {
		
		if(this.getColumnSize() != that.getColumnSize() || this.getRowSize() != that.getRowSize())
			throw new IllegalArgumentException();
		
		int columnsize = this.getColumnSize();
		int rowsize = this.getRowSize();
		
		Matrix result = new Matrix(columnsize, rowsize);
		
		for (int i = 0; i < columnsize; i++) {
			for (int j = 0; j < rowsize; j++) {
				result.setValue(i, j, this.getValue(i, j) * that.getValue(i, j));
			}
		}
		
		return result;
	}
	
	/**
	 * Element wise division between two matrices.
	 * @param that - another matrix of the same dimension
	 * @return - matrix C = A o/ B
	 */
	public Matrix eleDiv(Matrix that) {
		
		if(this.getColumnSize() != that.getColumnSize() || this.getRowSize() != that.getRowSize())
			throw new IllegalArgumentException();
		
		int columnsize = this.getColumnSize();
		int rowsize = this.getRowSize();
		
		Matrix result = new Matrix(columnsize, rowsize);
		
		for (int i = 0; i < columnsize; i++) {
			for (int j = 0; j < rowsize; j++) {
				result.setValue(i, j, this.getValue(i, j) / that.getValue(i, j));
			}
		}
		
		return result;
	}
	
	/**
	 * Inplace scaling of the matrix by a float scalar.
	 * @param scalar - float scalar k in A <- kA
	 */
	public void scale(float scalar) {
		
		for (int i = 0; i < this.getColumnSize(); i++) {
			for (int j = 0; j < this.getRowSize(); j++) {
				arr[i][j] *= scalar;
			}
		}
	}
	
	/**
	 * Non-inplace scaling of a matrix by a float scalar.
	 * @param scalar - float scalar k in kA
	 * @return Matrix B = kA
	 */
	public Matrix scaled(float scalar) {
		
		int columnsize = this.getColumnSize();
		int rowsize = this.getRowSize();
		
		Matrix result = new Matrix(columnsize, rowsize);
		
		for (int i = 0; i < columnsize; i++) {
			for (int j = 0; j < rowsize; j++) {
				result.setValue(i, j, arr[i][j] * scalar);
			}
		}
		
		return result;
	}
	
	/**
	 * This method returns a new matrix with the columns and rows flipped
	 * @return matrix B = A^T
	 */
	public Matrix transpose() {
		
		int columnsize = this.getRowSize();
		int rowsize = this.getColumnSize();
		
		Matrix result = new Matrix(columnsize, rowsize);
		
		for (int i = 0; i < columnsize; i++) {
			for (int j = 0; j < rowsize; j++) {
				result.setValue(j, i, arr[i][j]);
			}
		}
		
		return result;		
	}
	
	public static void main(String[] args) {
		
	}
}
