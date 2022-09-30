package interfaces;

public interface Model<I, O, D>{

    public O compute(I input);

    public void train(D[] training, D[] testing, float learningRate, int epochs, boolean verbose);
    
    public double getLoss(D[] examples);
}