import healpy as hp
import numpy as np
import os

# =========================================================
# CONFIGURACIÓN PARA CIB (o la que se quiera)
# =========================================================
ruta_in = "ruta_(la_que_sea)" 
ruta_out = "cib_n512.fits"
nside_out = 512

if not os.path.exists(ruta_in):
    print(f"ERROR: No encuentro el archivo en {ruta_in}")
    print("Verifica la ruta en la carpeta .pysm_data o .astropy")
else:
    print(f"Reduciendo CIB (1.6 GB) por promediado de píxeles...")
    
    # Leer el mapa original (NSIDE 4096 generalmente en WebSky)
    # memmap=True para el tamaño del archivo
    mapa_grande = hp.read_map(ruta_in, memmap=True, verbose=False)
    nside_in = hp.get_nside(mapa_grande)
    npix_out = hp.nside2npix(nside_out)
    ratio = (nside_in // nside_out)**2
    print(f"Detectado NSIDE {nside_in}. Reduciendo a {nside_out} (Ratio: {ratio}:1)...")

    # Crear el mapa de salida
    mapa_pequeno = np.zeros(npix_out, dtype=np.float32)
    # Bucle de promediado
    for i in range(npix_out):
        idx = np.arange(i * ratio, (i + 1) * ratio)
        mapa_pequeno[i] = np.mean(mapa_grande[idx])
        if i % 100000 == 0:
            print(f"Progreso: {100 * i / npix_out:.1f}%")

    # Guardar resultados
    hp.write_map(ruta_out, mapa_pequeno, overwrite=True)
    print(f"¡LOGRADO! Archivo guardado como: {ruta_out}")