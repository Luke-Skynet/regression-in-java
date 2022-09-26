package math;

public class Matrix {

	private final float[][] arr;

	//Constructors
	
	public Matrix(float[][] arr) {
	
		for(int i = 0; i < arr.length; i++){
			if (arr[0].length != arr[i].length)
				throw new IllegalArgumentException();
		}
		
		this.arr = new float[arr.length][arr[0].length];
		for(int i = 0; i < arr.length; i++) {
			for(int j = 0; j < arr[0].length; j++) {
				this.arr[i][j] = arr[i][j];
			}
		}
	}
	
	public Matrix(int columnSize, int RowSize) {
		arr = new float[columnSize][RowSize];
	}
	
	//Accessors and Mutators
	
	public int getColumnSize(){
		return arr.length;
	}
	
	public int getRowSize() {
		return arr[0].length;
	}
	
	public void setValue(int i, int j, float value) {
		arr[i][j] = value;
	}
	
	public void setValuesRandom() {
		for (int i = 0; i < this.getColumnSize(); i++) {
			for (int j = 0; j < this.getRowSize(); j++) {
				this.setValue(i, j, (float) ( 2 * (Math.random() - .5) ) );
			}
		}
	}
	
	public float getValue(int i, int j) {
		return arr[i][j];
	}
	
	public Vector getColumnVector(int column) {
		
		Vector result = new Vector(this.getColumnSize());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(i, column));
		}
		
		return result;
	}
	
	public Vector getRowVector(int row) {
		
		Vector result = new Vector(this.getRowSize());
		
		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, this.getValue(row, i));
		}
		
		return result;
	}
	
	//printer
	
	public void print() {
		for (int i = 0; i < this.getColumnSize(); i++) {
			for (int j = 0; j < this.getRowSize(); j++) {
				System.out.print(this.getValue(i, j) + " ");
			}
			System.out.println();;
		}
	}
	
	//Math Functions
	
	public Matrix plus(Matrix that) {
		
		if(this.getColumnSize() != that.getColumnSize() || this.getRowSize() != that.getRowSize())
			throw new IllegalArgumentException();
		
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
	
	public void scale(float scalar) {
		
		for (int i = 0; i < this.getColumnSize(); i++) {
			for (int j = 0; j < this.getRowSize(); j++) {
				arr[i][j] *= scalar;
			}
		}
	}
	
	public Matrix timesScalar(float scalar) {
		
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
	
	public Matrix transpose() {
		
		int columnsize = this.getColumnSize();
		int rowsize = this.getRowSize();
		
		Matrix result = new Matrix(rowsize, columnsize);
		
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
