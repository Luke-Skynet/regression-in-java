package interfaces;


/**
 * This interface defines the basic functionality for models in this package.   
 * It needs 3 abstract types:    
 * @param I - Input (usually a vector) 
 * @param O - Output (either a scalar or a vector) 
 * @param D - DataType (this is usually an implementation of the sample interface) 
 * I and O are abstract data types to accomodate for any n tensor of any data type
 */
public interface Model<I, O, D>{
    /**
     * This is the method for performing inference.
     * @param input - data to be inferenced on (X)
     * @return O - the prediction (Y)
     */
    public O compute(I input);

    /**
	 * Training method that uses batches of data samples to update weights at every step (stochastic gradient descent).
     * @param training - an array of Sample objects that the model uses for weight updating
     * @param testing - an array of Sample objects that is used to display loss when verbose is true
     * @param batchSize - the number of samples the model uses to update its weights during a training step
     * @param learningRate - double precision flaot used to scale gradients for training steps
     * @param epochs - number of times the model goes through the training data array
     * @param verbose - display toggle for viewing training process, (setting to false will disable testing data passes / loss computation)
     */
    public void train(D[] training, D[] testing, int batchSize, double learningRate, int epochs, boolean verbose);
    
    /**
     * This method returns the loss of the models predictions over the imput dataset.
     * @param examples - array of Sample objects that is used to compute loss in whatever way the model does it (MSE, LL, etc)
     * @return - scalar representing representing error of prediction over samples
     */
    public double getLoss(D[] examples);
}