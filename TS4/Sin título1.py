import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc

N = 1000          # número de muestras
fs = 1000        # frecuencia de muestreo [Hz]
df = fs / N      # resolución espectral
a0 = np.sqrt(2)         # amplitud seno
SNR_dB = 3            # SNR en dB

# Potencia de la señal
P_signal = a0**2 / 2

# Potencia de ruido a partir de SNR en dB
SNR_linear = 10**(SNR_dB/10)
pot_ruido = P_signal / SNR_linear

# %% Matriz de senos
R = 200
flattop = sc.windows.flattop(N).reshape((-1,1))
U = np.sum(flattop)/N                           # Correccion de la ventana      

tt_vector = np.arange(N)/fs
ff_vector = np.random.uniform(-2, 2, R)         # Frecuencias random

tt_columnas = tt_vector.reshape((-1,1))         # Tamaño N (vector COLUMNA)
ff_filas = ff_vector.reshape((1,-1))            # Tamaño R (vector FILA)
TT_sen = np.tile(tt_columnas, (1, R))           # Tamaño NxR (matriz)
FF_sen = np.tile(ff_filas, (N, 1))              # Tamaño RxN (matriz)
ruido = np.random.normal(loc = 0, scale = np.sqrt(pot_ruido), size = (N,R))

xx_sen = a0 * np.sin(2 * np.pi * (N/4+FF_sen) * df * TT_sen)
xx_sen_ruido = a0 * np.sin(2 * np.pi * (N/4+FF_sen) * df * TT_sen) + ruido
xx_vent = xx_sen_ruido * flattop

XX_sen = np.fft.fft(xx_vent, n = N, axis = 0)/(N*U)

freqs = np.fft.fftfreq(N, d=1/fs)


# # Señales en el tiempo
# plt.figure(figsize=(12,6))
# plt.plot(tt_vector, xx_sen_ruido)
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Amplitud")
# plt.title("Señales senoidales con ruido")
# plt.grid(True)
# #plt.legend()
# plt.show()

# # Señales en frecuencia
# plt.figure(figsize=(12,6))
# plt.plot(freqs, 10*np.log10(XX_sen**2 + 1e-12),'.')
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("Amplitud[dB]")
# plt.title(" FFT de Señales senoidales con ruido")
# plt.xlim(0,N/2)
# plt.grid(True)
# #plt.legend()
# plt.show()


# Estimador de amplitud (en f0)
a1_est = np.abs(XX_sen[N//4, :])

# Estimador de frecuencia (máximo del espectro)
idx_max = np.argmax(np.abs(XX_sen), axis=0)
Omega1 = freqs[idx_max]
Omega1_est = Omega1[(Omega1 >= 0) & (Omega1 <= N//2)]

# Estadisticos
bias_a1 = np.median(a1_est) - a0
variance_a1 = np.var(a1_est, ddof=1)

bias_Omega1 = np.median(Omega1_est) - N/4
variance_Omega1 = np.var(Omega1_est, ddof=1)

print(f"Sesgo: {bias_a1:.2f}")
print(f"Varianza: {variance_a1:.2f}\n")

print(f"Sesgo: {bias_Omega1:.2f}")
print(f"Varianza: {variance_Omega1:.2f}")


# Histograma de amplitud
plt.figure(figsize=(10,5))
plt.hist(a1_est, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Amplitud estimada")
plt.ylabel("Cantidad de señales")
plt.title("Histograma de estimaciones de amplitud a1")
plt.grid(True)
plt.show()

# Histograma de frecuencia
plt.figure(figsize=(10,5))
plt.hist(Omega1_est, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Frecuencia estimada [Hz]")
plt.ylabel("Cantidad de señales")
plt.title("Histograma de estimaciones de frecuencia Ω1")
plt.grid(True)
plt.show()


