package nn.activationFunctions;
import interfaces.ActivationFunction;
import math.Vector;

public class ReLU implements ActivationFunction<Vector>{
    
    Vector input;

    public ReLU(int inDimension){
        this.input = new Vector(inDimension);
    }

    public Vector forward(Vector input){

        this.input = input;

        Vector result = new Vector(input.getLength());
        
        for(int i = 0; i < result.getLength(); i++){
            result.setValue(i, Math.max(0.0, input.getValue(i)));
        }
        
        return result;
    }

    public Vector backward(Vector gradient){

        Vector result = new Vector(gradient.getLength());

        for(int i = 0; i < result.getLength(); i++){
            result.setValue(i, this.input.getValue(i) >= 0.0 ? gradient.getValue(i) : 0.0);
        }

        return result;
    }
}
