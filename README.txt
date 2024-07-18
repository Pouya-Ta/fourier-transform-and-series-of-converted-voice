# Fourier Transform and Series of Converted Voice

`main.py` is a Python script that performs various Fourier Transform operations on audio signals to investigate their frequency domain characteristics. The script loads multiple audio files, computes their Fourier Transforms, and applies different phase and magnitude adjustments to analyze and manipulate the signals.

## Features

- **Fourier Transform Computation**: Calculates the complex Fourier Transform of input audio signals.
- **Phase Manipulation**: Includes operations to negate the phase, set the phase to zero, and add specific values to the phase.
- **Fourier Transform Resizing**: Adjusts the size of the Fourier Transform to analyze effects on the time-domain signal.
- **Interpolation and Average Sizing**: Interpolates Fourier Transform values to a common average size for comparative analysis.
- **Phase and Magnitude Adjustment**: Combines magnitude and phase information from different signals for further analysis.
- **Audio File Handling**: Loads and saves audio files with modified frequency domain characteristics.

## How It Works

1. **Loading Audio Files**: The script loads audio files using `scipy.io.wavfile.read`.
2. **Fourier Transform Calculation**: Computes the Fourier Transform of the audio signals using `numpy.fft.fft`.
3. **Phase and Magnitude Adjustments**:
   - **Negate Phase**: Negates the phase of the Fourier Transform.
   - **Zero Phase**: Sets the phase to zero while retaining the magnitude.
   - **Add Specific Values to Phase**: Adds specific values to the phase for selected frequencies.
4. **Resizing Fourier Transform**: Doubles the size of the Fourier Transform and applies inverse Fourier Transform to obtain time-domain signals.
5. **Interpolation**: Interpolates Fourier Transform values to a common average size for comparative analysis.
6. **Combining Phase and Magnitude**: Adjusts the phase and magnitude by combining information from different signals.
7. **Saving Modified Audio Files**: Saves the resulting audio files after inverse Fourier Transform using `scipy.io.wavfile.write`.

## Usage

1. Ensure you have the required libraries installed:
    ```bash
    pip install numpy scipy
    ```
2. Place your audio files in the appropriate directory and update the file paths in the script.
3. Run the script:
    ```bash
    python main.py
    ```
4. The modified audio files will be saved to the specified paths.

## Example Output

### Fourier Transform Phase Manipulation
- **Negated Phase**: Adjusts the phase of the Fourier Transform by negating it.
- **Zero Phase**: Sets the phase to zero and retains the magnitude.

### Fourier Transform Resizing
- **Doubled Size**: Adjusts the size of the Fourier Transform to analyze the effect on the time-domain signal.

### Interpolation and Average Sizing
- **Interpolated Values**: Interpolates Fourier Transform values to a common average size for comparison.

### Combined Phase and Magnitude
- **Adjusted Phase and Magnitude**: Combines magnitude and phase information from different signals.

## Code Overview

### Loading and Fourier Transform Calculation
```python
import numpy as np
from scipy.io import wavfile

# Load the first audio file
sr_1, audio_1 = wavfile.read('path/to/your/audio_file1.wav')

# Calculate the complex Fourier transform of the first audio signal
sp_1 = np.fft.fft(audio_1)

# Load the second audio file
sr_2, audio_2 = wavfile.read('path/to/your/audio_file2.wav')

# Calculate the complex Fourier transform of the second audio signal
sp_2 = np.fft.fft(audio_2)
```

### Phase Manipulation
```python
# Negate the phase for the first signal
sp_mag_1 = np.abs(sp_1) # magnitude
sp_phase_1 = np.angle(sp_1) # phase
sp_phase_neg_1 = -sp_phase_1 # negate phase
sp_adj1_1 = sp_mag_1 * np.exp(1j * sp_phase_neg_1) # adjust phase

# Negate the phase for the second signal
sp_mag_2 = np.abs(sp_2) # magnitude
sp_phase_2 = np.angle(sp_2) # phase
sp_phase_neg_2 = -sp_phase_2 # negate phase
sp_adj1_2 = sp_mag_2 * np.exp(1j * sp_phase_neg_2) # adjust phase
```

### Resizing Fourier Transform
```python
# Double the size of the Fourier transform for the first signal
sp_big_1 = np.zeros_like(sp_1, dtype=np.complex128) # initialize new array with double the size of original array
sp_big_1[:N_1//2] = sp_adj3_1[:N_1//2] # copy values from first half of the original array
sp_big_1[-N_1//2:] = sp_adj3_1[-N_1//2:] # copy values from second half of the original array
sp_adj4_1 = np.fft.ifft(sp_big_1) # inverse Fourier transform to obtain time-domain signal with double the size
```

### Interpolation and Average Sizing
```python
# Set the size of the Fourier transform at all frequencies the same and equal to the average size
avg_size = (sp_1.shape[0] + sp_2.shape[0]) // 2 # average size of the Fourier transforms
sp_adj5_1 = np.zeros(avg_size, dtype=sp_1.dtype) # initialize new array with the average size
sp_adj5_2 = np.zeros(avg_size, dtype=sp_2.dtype) # initialize new array with the average size
index_1 = np.linspace(0, sp_1.shape[0] - 1, avg_size, endpoint=True, dtype=int) # indices for interpolation of the first signal
index_2 = np.linspace(0, sp_2.shape[0] - 1, avg_size, endpoint=True, dtype=int) # indices for interpolation of the second signal
sp_adj5_1 = np.interp(index_1, np.arange(sp_1.shape[0]), sp_1) # interpolate values for the first signal
sp_adj5_2 = np.interp(index_2, np.arange(sp_2.shape[0]), sp_2) # interpolate values for the second signal
```
