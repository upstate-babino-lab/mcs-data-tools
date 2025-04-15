import numpy as np

def centered_moving_average(data, window_size):
  """
  Calculates the centered moving average of a 1D NumPy array.

  Args:
    data (np.ndarray): The input 1D NumPy array.
    window_size (int): The size of the moving average window (must be odd for perfect centering).

  Returns:
    np.ndarray: The centered moving average of the input data.
                Values at the edges where the window is incomplete will be affected.
  """
  if window_size % 2 == 0:
    raise ValueError("Window size must be odd for a perfectly centered moving average.")

  # Create a window of equal weights (simple moving average)
  weights = np.ones(window_size) / window_size

  # Use convolution to calculate the moving average with 'same' mode for centered output
  cma = np.convolve(data, weights, mode='same')
  return cma



def find_square_wave_steps(signal, threshold=0.5):
  """
  Finds the indices where steps occur in a square wave.

  Args:
    signal (np.ndarray): A 1D NumPy array representing the square wave.
    threshold (float): The minimum absolute difference between consecutive
                       points to be considered a step. Adjust this based on
                       the noise and amplitude of your square wave.

  Returns:
    np.ndarray: A 1D NumPy array containing the indices where steps occur.
  """
  # Calculate the difference between consecutive data points
  differences = np.diff(signal)

  # Find the indices where the absolute difference is greater than the threshold
  step_indices = np.where(np.abs(differences) > threshold)[0] + 1
  # We add 1 to the indices because np.diff reduces the array length by one,
  # and the step occurs at the index of the *second* point in the difference.

  return step_indices
