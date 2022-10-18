package interfaces;
/**
 * This interface outlines the data sample objects used by the models for learning.
 * There are two key components:
 * D - Data (the X) input and it is usually a vector
 * L - Label (the Y) ground truth output and it is either a scalar or a vector
 * both of these fields are abstract data types to accomodate for any n tensor of any data type
 */
public interface Sample <D, L>{

    /**
     * This gives you the (X) to do inference on
     * @return Data - implemented type
     */
    public D getData();

    /**
     * This gives you the (Y) ground truth
     * @return Label - implemented type
     */
    public L getLabel();
}