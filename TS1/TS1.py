import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generar_seno(amplitud, frecuencia, fase, fs, duracion):
    """
    Devuele los vectores t, y
    """
    Ts = 1/fs                                           # Tiempo de muestreo
    N = int(fs*duracion)                                # Numero de muestras
    t = np.arange(N) * Ts                               # Vector tiempo
    y = amplitud * np.sin(2*np.pi*frecuencia*t + fase)  # Vector seno

    print(f"Tiempo entre muestras Ts = {Ts:.2e} s")
    print(f"Número de muestras N = {N}")
    print(f"Potencia = {np.mean(y**2):.3f}")
    return t, y


def ortogonalidad(f, g):
    
    tol = 1*np.exp(-5)
    """
    Verifica la ortogonalidad de dos funciones discretas usando producto interno.

    Parámetros:
        f, g : arrays de NumPy
            Valores de las funciones evaluadas en los mismos puntos.
        tol : float
            Tolerancia para considerar el producto interno como cero.
    
    """
    # Producto interno discretizado
    producto_interno = np.sum(f * g)
    print (f"{producto_interno}")
    
    if producto_interno < tol:
        print(f"las funciones son ortogonales")
    
    
######################## Punto 1 y 2 ###################
# Señal original
AmplitudOriginal = 1
FrecuenciaOriginal = 2000

# Datos generales
fs = 50000
duracion = 0.01

t1, Original = generar_seno(amplitud = AmplitudOriginal, frecuencia = FrecuenciaOriginal, 
                      fase=0, fs=fs, duracion=duracion)

# Señal amplificada y desfazada

t2, y2 = generar_seno(amplitud = AmplitudOriginal * 2, frecuencia = FrecuenciaOriginal, 
                      fase = np.pi/2, fs=fs, duracion=duracion)

# Graficar ambas en la misma figura
plt.figure(figsize=(10,5))
plt.plot(t1, Original, label="Original")
plt.plot(t2, y2, label="Amplificada y desfazada")
plt.title("Comparación de señales")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.show()

######################## Punto 3 ######################

# Señal moduladora
t, moduladora = generar_seno(amplitud = AmplitudOriginal, frecuencia = FrecuenciaOriginal/2, 
                             fase=0, fs=fs, duracion=duracion)

# Señal AM: (1 + moduladora) * portadora
s_am = (1 + moduladora) * Original

print(f"Potencia señal AM = {np.mean(s_am**2):.3f}")

# Graficar
plt.figure(figsize=(10,5))
plt.plot(t, s_am, label="Señal AM")
plt.plot(t, Original, "--", alpha = 0.8,label=f"Portadora {FrecuenciaOriginal} Hz")
plt.plot(t, moduladora, "--",alpha = 0.8, label=f"Moduladora {FrecuenciaOriginal/2} Hz")
plt.title("Señal modulada en amplitud (AM)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

##################### Punto 4 ######################

s_clip = np.clip(s_am, - (AmplitudOriginal *0.75), AmplitudOriginal *0.75)

# Graficar
plt.figure(figsize=(10,5))
plt.plot(t, s_clip)
plt.title("Señal recortada al 75% de la amplitud original")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

################### Punto 5 ######################

FrecuenciaCuadrada = 4000

sq = AmplitudOriginal * signal.square(2*np.pi*FrecuenciaCuadrada*t)   # Señal cuadrada (valores +1 / -1)

# Graficar
plt.figure(figsize=(10,5))
plt.plot(t, sq)
plt.title("Señal recortada al 75% de la amplitud original")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

################### Punto 6 #######################

# Parámetros

duracion = 0.05    
pulso_duracion = 0.01  # 10 ms

# Vector de tiempo
t = np.linspace(0, duracion, 500, endpoint=False)

# Pulso rectangular: 1 durante 10 ms, luego 0
pulso = np.where(t < pulso_duracion, 1.0, 0.0)

# Graficar
plt.figure(figsize=(10,4))
plt.plot(t, pulso, drawstyle="steps-post")
plt.title("Pulso rectangular de 10 ms")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.ylim(-0.2, 1.2)
plt.grid(True)
plt.show()


ortogonalidad(Original, y2)