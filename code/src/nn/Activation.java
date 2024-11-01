package nn;

import interfaces.ActivationFunction;
import interfaces.Layer;
import math.Vector;

public class Activation implements Layer<Vector,Vector> {

    private ActivationFunction<Vector> actFunct;

    public Activation(ActivationFunction<Vector> actFunct){
        this.actFunct = actFunct;
    }

    public Vector forward(Vector input){
        return this.actFunct.forward(input);
    }

    public Vector backward(Vector gradient){
        return this.actFunct.backward(gradient);
    }

    public void zeroGrad(){
        return;
    }
    public void update(double learningRate, int t, int batchSize){
        return;
    }
}
