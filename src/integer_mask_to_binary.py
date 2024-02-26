import numpy as np

def integer_mask_to_binary(arr, num_bits):
    """Converts input array to binary with a fixed number of bits, and reshapes the output.
    
    Args:
        arr (numpy.ndarray): Input numpy array.
        num_bits (int): The fixed number of bits for the binary representation.
    
    Returns:
        numpy.ndarray: A new array with shape (num_bits, *arr.shape) containing the binary representation.
    """
    bin_nums = ((arr.reshape(arr.shape + (-1,)) & (2 ** np.arange(num_bits))) != 0).astype(int)

    return bin_nums