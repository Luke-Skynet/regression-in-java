package test;

import java.util.*;
import math.Matrix;

public class TensorTesterOne {

	public static void main(String[] args) {
		
		Random random = new Random();
		
		//make matrix 1
		Matrix matrix1 = new Matrix(4,2);
		
		for (int i = 0; i < matrix1.getColumnSize(); i++) {
			for (int j = 0; j < matrix1.getRowSize(); j++) {
				matrix1.setValue(i, j, random.nextInt(10));
			}
		}

		//make matrix 2
		Matrix matrix2 = new Matrix(2,4);
		
		for (int i = 0; i < matrix2.getColumnSize(); i++) {
			for (int j = 0; j < matrix2.getRowSize(); j++) {
				matrix2.setValue(i, j, random.nextInt(10));
			}
		}
		
		//make matrices as a function of the first two
		Matrix matrix3 = matrix1.dot(matrix2);
		
		Matrix matrix4 = matrix2.dot(matrix1);
		
		matrix1.print();
		System.out.println();
		
		matrix2.print();
		System.out.println();
		
		matrix3.print();
		System.out.println();
		
		matrix4.print();
		System.out.println();
		
	}
}
