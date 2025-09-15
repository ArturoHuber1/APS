import numpy as np
import matplotlib.pyplot as plt


# =========== (a) PSD de senoidales =============
# Parámetros
N = 1000
fs = 1000
df = fs / N
amp = np.sqrt(2) 
t = np.arange(N) / fs  # vector de tiempo

# Valores de k0
k0_values = [N/4, N/4 + 0.25, N/4 + 0.5]

# Crear figura para PSDs
plt.figure(figsize=(12, 6))

for i, k0 in enumerate(k0_values):
    
    f0 = k0 * df                                 # frecuencia en Hz
    x = amp* np.sin(2 * np.pi * f0 * t)          # señal
    
    # FFT y PSD
    X = np.fft.fft(x, N)                         # FFT
    psd = (np.abs(X)**2) / N                     # densidad espectral
    
    # Eje de frecuencias
    freqs = np.fft.fftfreq(N, 1/fs)
    # Grafico
    plt.subplot(3, 1, i+1)
    plt.plot(freqs[:N//2], 10*np.log10(psd[:N//2] + 1e-12),'x')
    plt.title(f"PSD para k0 = {k0:.2f} (f0 = {f0:.2f} Hz)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.grid()

plt.tight_layout()
plt.show()


# ====== (b) Verificación de Parseval ======
for k0 in k0_values:
    f0 = k0 * df
    x = amp * np.sin(2 * np.pi * f0 * t)
   

    X = np.fft.fft(x, N)
    
    energia_tiempo = np.sum(np.abs(x)**2)/N
    energia_freq = np.sum(np.abs(X)**2) / N**2
    
    print(f"f = {f0:.2f} : Potencia temporal = {energia_tiempo:.2f}, Potencia espectral = {energia_freq:.2f}")


# ====== (c) Experimento con zero padding ======
N_pad = 10 * N
t_pad = np.arange(N_pad) / fs

eps = 1e-12  # para evitar log(0)


for i, k0 in enumerate(k0_values):
    f0 = k0 * df
    x = amp * np.sin(2 * np.pi * f0 * t)
    #x = x / np.sqrt(np.mean(x**2))
    
    # Agregar ceros
    x_pad = np.concatenate([x, np.zeros(9*N)])
    
    # FFT con padding
    X_pad = np.fft.fft(x_pad, N_pad)
    psd_pad = (np.abs(X_pad)**2) / N_pad
    psd_pad_db = 10 * np.log10(psd_pad + eps)  
    
    freqs_pad = np.fft.fftfreq(N_pad, 1/fs)
    
    plt.figure(figsize=(12, 6))
    plt.plot(freqs_pad[:N_pad//2], psd_pad_db[:N_pad//2],'.')
    plt.title(f"PSD (dB) con Zero Padding para k0 = {k0:.2f} (f0 = {f0:.2f} Hz)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD [dB]")
    plt.grid()
    plt.show()

print(f"la resolucion de la frecuencia es: {fs/N_pad}Hz")