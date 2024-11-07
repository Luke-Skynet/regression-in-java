package interfaces;

/**
 * This interface defines the essential operations for an activation function used in neural network models.
 * It contains methods for the forward pass (activation function application) and backward pass (gradient of activation).
 * @param <T> - the input type, typically a vector or tensor
 */
public interface ActivationFunction<T> extends Layer<T,T>{

    /**
     * Applies the activation function on the input data.
     * @param input - the input vector to which the activation function is applied
     * @return Vector - the output vector after applying the activation function
     */
    public T forward(T input);

    /**
     * Computes the gradient of the activation function with respect to its input,
     * used during the backpropagation phase in neural networks.
     * @param gradient - the gradient vector from the subsequent layer or loss function
     * @return Vector - the gradient vector after applying the derivative of the activation function
     */
    public T backward(T gradient);

    /**
     * Not used for activation functions, but is here to simplify the NN training method.
     */
    public void zeroGrad();

    /**
     * Not used for activation functions, but is here to simplify the NN training method.
     */
    public void update(double learningRate, int t, int batchSize);

}