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

# Section 1: Negate the phase for the first signal
sp_mag_1 = np.abs(sp_1) # magnitude
sp_phase_1 = np.angle(sp_1) # phase
sp_phase_neg_1 = -sp_phase_1 # negate phase
sp_adj1_1 = sp_mag_1 * np.exp(1j * sp_phase_neg_1) # adjust phase
# Negate the phase for the second signal
sp_mag_2 = np.abs(sp_2) # magnitude
sp_phase_2 = np.angle(sp_2) # phase
sp_phase_neg_2 = -sp_phase_2 # negate phase
sp_adj1_2 = sp_mag_2 * np.exp(1j * sp_phase_neg_2) # adjust phase

# Section 2: Set the phase to zero for the first signal
sp_adj2_1 = sp_mag_1 * np.exp(1j * 0) # adjust phase
# Set the phase to zero for the second signal
sp_adj2_2 = sp_mag_2 * np.exp(1j * 0) # adjust phase

# Section 3: Add n*omega0 to the phase for n/N = 1/4, 1/2, 3/4 for the first signal
N_1 = sp_1.shape[0] # number of samples for the first signal
n1_1 = round(N_1/4) # n = N/4
n2_1 = round(N_1/2) # n = N/2
n3_1 = round(3*N_1/4) # n = 3N/4
omega0_1 = 2*np.pi/N_1 # (rad/sample), normalized frequency
sp_phase_new_1 = np.zeros_like(sp_phase_1, dtype=np.float64) # initialize new phase for the first signal
sp_phase_new_1[n1_1] = n1_1*omega0_1 # add n1*omega0
sp_phase_new_1[n2_1] = n2_1*omega0_1 # add n2*omega0
sp_phase_new_1[n3_1] = n3_1*omega0_1 # add n3*omega0
sp_adj3_1 = sp_mag_1 * np.exp(1j * sp_phase_new_1) # adjust phase for the first signal
# Add n*omega0 to the phase for n/N = 1/4, 1/2, 3/4 for the second signal
N_2 = sp_2.shape[0] # number of samples for the second signal
n1_2 = round(N_2/4) # n = N/4
n2_2 = round(N_2/2) # n = N/2
n3_2 = round(3*N_2/4) # n = 3N/4
omega0_2 = 2*np.pi/N_2 # (rad/sample), normalized frequency
sp_phase_new_2 = np.zeros_like(sp_phase_2, dtype=np.float64) # initialize new phase for the second signal
sp_phase_new_2[n1_2] = n1_2*omega0_2 # add n1*omega0
sp_phase_new_2[n2_2] = n2_2*omega0_2 # add n2*omega0
sp_phase_new_2[n3_2] = n3_2*omega0_2 # add n3*omega0
sp_adj3_2 = sp_mag_2 * np.exp(1j * sp_phase_new_2) # adjust phase for the second signal

# Section 4: Double the size of the Fourier transform for the first signal
sp_big_1 = np.zeros_like(sp_1, dtype=np.complex128) # initialize new array with double the size of original array
sp_big_1[:N_1//2] = sp_adj3_1[:N_1//2] # copy values from first half of the original array
sp_big_1[-N_1//2:] = sp_adj3_1[-N_1//2:] # copy values from second half of the original array
sp_adj4_1 = np.fft.ifft(sp_big_1) # inverse Fourier transform to obtain time-domain signal with double the size
# Double the size of the Fourier transform for the second signal
sp_big_2 = np.zeros_like(sp_2, dtype=np.complex128) # initialize new array with double the size of original array
sp_big_2[:N_2//2] = sp_adj3_2[:N_2//2] # copy values from first half of the original array
sp_big_2[-N_2//2:] = sp_adj3_2[-N_2//2:] # copy values from second half of the original array
sp_adj4_2 = np.fft.ifft(sp_big_2) # inverse Fourier transform to obtain time-domain signal with double the size

# Print numpy arrays
np.set_printoptions(precision=3, suppress=True)
print("Section 1")
print(sp_adj1_1)
print(sp_adj1_2)
print("Section 2")
print(sp_adj2_1)
print(sp_adj2_2)
print("Section 3")
print(sp_adj3_1)
print(sp_adj3_2)
print("Section 4")
print(sp_adj4_1)
print(sp_adj4_2)

# Section 5: Set the size of the Fourier transform at all frequencies the same and equal to the average size of the
# Fourier transform
avg_size = (sp_1.shape[0] + sp_2.shape[0]) // 2 # average size of the Fourier transforms
sp_adj5_1 = np.zeros(avg_size, dtype=sp_1.dtype) # initialize new array with the average size
sp_adj5_2 = np.zeros(avg_size, dtype=sp_2.dtype) # initialize new array with the average size
index_1 = np.linspace(0, sp_1.shape[0] - 1, avg_size, endpoint=True, dtype=int) # indices for interpolation of the first signal
index_2 = np.linspace(0, sp_2.shape[0] - 1, avg_size, endpoint=True, dtype=int) # indices for interpolation of the second signal
sp_adj5_1 = np.interp(index_1, np.arange(sp_1.shape[0]), sp_1) # interpolate values for the first signal
sp_adj5_2 = np.interp(index_2, np.arange(sp_2.shape[0]), sp_2) # interpolate values for the second signal

# Print numpy arrays
np.set_printoptions(precision=3, suppress=True)
print("Section 5")
print(sp_adj5_1)
print(sp_adj5_2)

# Section 6: Record another audio file of the same length. Move the phase and magnitude of the Fourier transform for
# the two signals
sr_3, audio_3 = wavfile.read('path/to/your/audio_file3.wav')
sp_3 = np.fft.fft(audio_3)

# Move the phase and magnitude of the Fourier transform for the two signals
sp_mag_3 = np.abs(sp_3)
sp_phase_3 = np.angle(sp_3)
sp_phase_diff_1 = sp_phase_2 - sp_phase_1 # calculate phase difference between signals 1 and 2
sp_adj6_1 = sp_mag_3 * np.exp(1j * (sp_phase_3 + sp_phase_diff_1)) # combine magnitude of signal 3 with phase difference between signals 1 and 2 for signal 1
sp_adj6_2 = sp_mag_3 * np.exp(1j * sp_phase_3) # combine magnitude of signal 3 with phase of signal 2 for signal 2

# Print numpy arrays
np.set_printoptions(precision=3, suppress=True)
print("Section 6")
print(sp_adj6_1)
print(sp_adj6_2)

# Save the resulting audio files
wavfile.write('path/to/your/new_audio_file1.wav', sr_1, np.real(np.fft.ifft(sp_adj6_1)))
wavfile.write('path/to/your/new_audio_file2.wav', sr_2, np.real(np.fft.ifft(sp_adj6_2)))
