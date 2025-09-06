import numpy as np
import matplotlib.pyplot as plt
from TS1 import Original, y2, s_am, s_clip, sq, pulso, fs

def sistema_lti(x, nombre):
    """
    Resuelve
    y[n] = 0.03 x[n] + 0.05 x[n-1] + 0.03 x[n-2] + 1.5 y[n-1] - 0.5 y[n-2]

    Parámetros
    ----------
    x :  Vector
        Señal de entrada x[n] 
    
    Returns
    -------
    y : Vector
        Señal de salida y[n]
    """
    Ts = 1/fs
    N = len(x)
    t = np.arange(N) * Ts
    y = np.zeros(N, dtype=float)

    for n in range(N):
        
        # Si los coeficientes son negativos, el resultado es 0        
        if n-1 >= 0:
            x1 = x[n-1]
            y1 = y[n-1]
        else:
            x1 = 0.0
            y1 = 0.0

        if n-2 >= 0:
            x2 = x[n-2]
            y2 = y[n-2]
        else:
            x2 = 0.0
            y2 = 0.0

        # Calculo
        y[n] = 0.03*x[n] + 0.05*x1 + 0.03*x2 + 1.5*y1 - 0.5*y2
        
    # Grafico
    plt.figure(figsize=(10,5))
    plt.plot(t, y, label="Salida")
    plt.title(f"Respuesta a la señal {nombre}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()
    plt.show()
    

    print(f"Potencia de la señal {nombre} = {np.mean(y**2):.3f}")

    return y

def convolucion(x, h, nombre):
    """
    Calcula la respuesta a traves de la convolucion con h

    Parameters
    ----------
    x : Vector
        Señal de entrada
    h : Vector
        Respuesta al impulso

    Returns
    -------
    None.

    """
    y = np.convolve(x, h, mode ="same")
    
    Ts = 1/fs
    N = len(y)
    t = np.arange(N) * Ts
    
        
    # Grafico
    plt.figure(figsize=(10,5))
    plt.plot(t, y, label="Salida")
    plt.axvline(x=0.01, color = 'black', linestyle='--', label='s = 0.01')
    plt.title(f"Salida de la señal {nombre} mediante convolución")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()
    plt.show()



def punto2(x, impulso):
    """   
    Hallar la respuesta al impulso y la salida correspondiente a una señal de entrada senoidal
    en los sistemas definidos mediante las siguientes ecuaciones en diferencias:
     primera ecuacion : y[n] =  x[n] + 3 x[n-10]
     segunda ecuacion : y[n] =  x[n] + 3 y[n-10]
        
    Parameters
    ----------
    x : Vector
        Señal de entrada (senoidal)
    impulso : Vector
        Señal del impulso

    Returns
    -------
    None.

    """
    
    N = len(impulso)
    h1 = np.zeros(N, dtype=float)       # h1 : respuesta al impulso de la primera ecuacion
    h2 = np.zeros(N, dtype=float)       # h1 : respuesta al impulso de la segunda ecuacion
    y1 = np.zeros(N, dtype=float)       # h1 : respuesta de la señal senoidal de la primera ecuacion
    y2 = np.zeros(N, dtype=float)       # h1 : respuesta de la señal senoidal de la segunda ecuacion
    
    
    # Calcular las salidas del impulso (h1) y de la senoidal (y1) de la primera ecuacion
    for n in range (N):
    
        h1[n] = impulso[n]
        y1[n] = x[n]
        
        if (n-10) >= 0:
            h1[n] += 3 * impulso[n-10]
            y1[n] += 3 * x[n-10]
    
    # Calcular las salidas del impulso (h2) y de la senoidal (y2) de la segunda ecuacion
    for n in range (N):
    
        h2[n] = impulso[n]
        y2[n] = x[n]
        
        if (n-10) >= 0:
            h2[n] += 3 * h2[n-10]
            y2[n] += 3 * y2[n-10]


    # Grafico
    plt.figure(figsize=(10,5))
    plt.plot(h1, label="Respuesta al impulso de la primer ecuacion")
    plt.plot(y1, label="Respuesta de la señal senoidal la primer ecuacion")
    plt.title("Respuestas de la primera ecuacion")
    plt.xlabel("Muestra [n]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()
    plt.show()        

    # Grafico
    plt.figure(figsize=(10,5))
    plt.plot(h2, label="Respuesta al impulso de la segunda ecuacion")
    plt.plot(y2, label="Respuesta de la señal senoidal la segunda ecuacion")
    plt.title("Respuestas de la segunda ecuacion")
    plt.xlabel("Muestra [n]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()
    plt.show()      
            
N = 500          
print(f"Para las simulaciones se toma un fs = {fs} y una duracion de {N/fs:.2f} \n")
sistema_lti(Original, "Original")                      # Senoidal f: 2KHz A:1 fase :0
sistema_lti(y2, "Desplazada y Amplificada")            # Senoidal f: 2KHz A:2 fase :p1/2
sistema_lti(s_am, "Modulada")                          # Señal original modulada
sistema_lti(s_clip, "Modulada y recortada")            # Señal anterior recortada
sistema_lti(sq, "Cuadrada")                            # Cuadrada f: 4KHz A:1
sistema_lti(pulso, "Pulso")                            # Pulso de 10ms

# Generar señal impulso

impulso = np.zeros(N)
impulso[0] = 1 

h = sistema_lti(impulso, "impulso")                    # h: respuesta al impulso


convolucion(Original, h, "Original")                   # Senoidal f: 2KHz A:1 fase :0
convolucion(y2, h, "Desplazada y Amplificada")         # Senoidal f: 2KHz A:2 fase :p1/2
convolucion(s_am, h, "Modulada")                       # Señal original modulada
convolucion(s_clip, h, "Modulada y recortada")         # Señal anterior recortada
convolucion(sq, h, "Cuadrada")                         # Cuadrada f: 4KHz A:1
convolucion(pulso, h, "Pulso")                         # Pulso de 10ms


punto2(Original, impulso)                              # Original: Senoidal f: 2KHz A:1 fase :0





