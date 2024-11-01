package interfaces;

/**
 * This interface defines the fundamental operations for a neural network layer.
 * It includes methods for the forward and backward passes, gradient management, and weight updates.
 * @param I - the input type, typically a vector or tensor
 * @param O - the output type, typically a vector or tensor
 */
public interface Layer<I, O> {

    /**
     * Performs the forward pass through the layer using the input data.
     * @param input - the input data to the layer
     * @return O - the output data after processing through the layer
     */
    public O forward(I input);

    /**
     * Computes the backward pass, applying the layer's gradient to the provided gradient from the next layer.
     * @param gradient - the gradient received from the next layer or loss function
     * @return I - the gradient to propagate to the previous layer
     */
    public I backward(O gradient);

    /**
     * Resets the gradient values in the layer to zero.
     * Used to clear accumulated gradients before a new training iteration.
     */
    public void zeroGrad();

    /**
     * Updates the layer's weights based on the current gradient and learning parameters.
     * @param learningRate - the learning rate that scales the gradient update
     * @param t - the current time step in training. Used for Adam
     * @param batchSize - the size of the mini-batch for gradient scaling, also used for Adam
     */
    public void update(double learningRate, int t, int batchSize);
}