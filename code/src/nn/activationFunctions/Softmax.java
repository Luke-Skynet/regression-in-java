package nn.activationFunctions;
import interfaces.ActivationFunction;
import math.*;

public class Softmax implements ActivationFunction<Vector>{
    
    Vector input;

    public Softmax(){}

	@Override
    public Vector forward(Vector input){

        double normalization = Math.pow(input.dot(input) / input.getLength(), .5);

		double[] vector = new double[input.getLength()];
		double sum = 0;

		for (int i = 0; i < input.getLength(); i++) {
			vector[i] = Math.exp(input.getValue(i) - normalization);
			sum += vector[i];
		}

		Vector result = new Vector(input.getLength());

		for (int i = 0; i < result.getLength(); i++) {
			result.setValue(i, (vector[i] / sum) );
		}

		return result;
    }

	@Override
    public Vector backward(Vector gradient){
        return gradient;
    }

	@Override
	public void zeroGrad(){};

	@Override
    public void update(double learningRate, int t, int batchSize){};
}
