import torch
import numpy as np
import os
from scipy.ndimage import gaussian_filter
import scipy.stats
from cmbnncs import unet
from cmbnncs.simulator import sim_CMB

# =========================================================
# CONFIGURACIÓN Y FUNCIONES CIENTÍFICAS
# =========================================================
nside = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pix_size_arcmin = 1.5
beam_efectivo_arcmin = 10

def calcular_psd_2d(imagen, pix_size_arcmin=pix_size_arcmin):
    ny, nx = imagen.shape
    # Resolución exacta en radianes
    res_rad = np.radians(pix_size_arcmin / 60.0)
    # Factor de corrección de la máscara
    w2 = np.mean(mask_apodized**2)
    # Transformada de Fourier normalizada
    f_coef = np.fft.fftshift(np.fft.fft2(imagen))
    psd_2d = (np.abs(f_coef)**2) / (nx * ny) * (res_rad**2)
    psd_2d /= w2
    # Crear rejilla radial de distancias en píxeles
    y, x = np.indices(psd_2d.shape)
    center = (nx // 2, ny // 2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    # Binning radial (Promedio de círculos)
    r_int = r.astype(int)
    tbin = np.bincount(r_int.ravel(), psd_2d.ravel())
    nr = np.bincount(r_int.ravel())
    psd_1d = tbin / np.where(nr == 0, 1, nr)
    # Eje L Físico
    k_step = (2 * np.pi) / (nx * res_rad)
    l_axis_radial = np.arange(len(psd_1d)) * k_step
    # Conversión a Dl (uK^2)
    dl_1d = l_axis_radial * (l_axis_radial + 1) * psd_1d / (2 * np.pi)
    return l_axis_radial, dl_1d

def binear_esfera(l, dl, bin_size=30):
    l_bins = np.arange(l.min(), l.max(), bin_size)
    # Calcular el promedio de Dl en cada bin
    dl_binned, bin_edges, _ = scipy.stats.binned_statistic(l, dl, statistic='mean', bins=l_bins)
    # Calcular el centro de cada bin para el eje l
    l_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return l_centers, dl_binned

def binear_espectro_para_chi2(l, dl, bin_size=50):
    l_bins = np.arange(2, 1000, bin_size)
    dl_binned, bin_edges, _ = scipy.stats.binned_statistic(l, dl, statistic='mean', bins=l_bins)
    l_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return l_centers, dl_binned

# Cargar el modelo
if nside == 64:
    model = unet.UNet5(channels=[32, 64, 128, 256, 512], channel_in=15, channel_out=1, kernels_size=[4]*10, strides=[2]*10, extra_pads=[0]*10, sides=nside).to(device)
else:
    model = unet.UNet8(channels=[32, 64, 128, 256, 512, 512, 512, 512], channel_in=15, channel_out=1, kernels_size=[4]*16, strides=[2]*16, extra_pads=[0]*16, sides=nside).to(device)
model.load_state_dict(torch.load("mejor_modelo_final.pth", map_location=device, weights_only=False))
model.eval()

# Máscara 
mask_base = np.zeros((nside, nside))
margin = int(nside * 0.1) 
mask_base[margin:-margin, margin:-margin] = 1.0
mask_apodized = gaussian_filter(mask_base, sigma=5.0)
f_sky_corr = np.mean(mask_apodized**2)

# =========================================================
# ANÁLISIS ESTADÍSTICO DE VALIDACIÓN (MONTE CARLO)
# =========================================================
print("--- Iniciando validación científica ---")
ruta_datos = "./dataset_tfg_final/val"
archivos_val = sorted([f for f in os.listdir(ruta_datos) if f.endswith('.pt')])
lista_dls_ia, lista_dls_target, residuos_acumulados, lista_r_pearson = [], [], [], []
mapas_para_ruido = []

with torch.no_grad():
    for i, archivo in enumerate(archivos_val):
        try:
            partes = archivo.split('_')
            seed_actual = int(partes[2].split('.')[0])
        except (IndexError, ValueError):
            print(f"Error procesando nombre de archivo: {archivo}")
            continue
        data = torch.load(os.path.join(ruta_datos, archivo), map_location=device, weights_only=False)
        t_in = data['input'].unsqueeze(0).to(device)
        target_np = data['target'].cpu().numpy()[0]
        # Recuperar el factor de escala físico original usando la seed
        _, _, _, std_target = sim_CMB(seed=seed_actual)
        # Inferencia de la IA y des-normalización
        pred_ia = model(t_in).cpu().numpy()[0, 0]
        pred_ia_phys = pred_ia * std_target
        target_phys = target_np * std_target
        # Cálculo de correlación por mapa
        mask_c = mask_apodized > 0.99
        r_mapa = np.corrcoef(target_np[mask_c].flatten(), pred_ia[mask_c].flatten())[0, 1]
        lista_r_pearson.append(r_mapa)
        # PSD y cálculos
        l_axis, dl_ia = calcular_psd_2d(pred_ia_phys * mask_apodized, pix_size_arcmin=pix_size_arcmin)
        _, dl_target = calcular_psd_2d(target_phys * mask_apodized, pix_size_arcmin=pix_size_arcmin)
        lista_dls_ia.append(dl_ia / f_sky_corr)
        lista_dls_target.append(dl_target / f_sky_corr)
        residuos_acumulados.append(target_np - pred_ia)
        # Guardar ejemplo para la Figura 1 (en uK)
        if i == 0: 
            mapas_finales = {
                'in': data['input'].cpu().numpy() * std_target, 
                'target': target_phys, 
                'ia': pred_ia_phys, 
                'res': target_phys - pred_ia_phys
            }

# =========================================================
# GUARDADO DE RESULTADOS DE VALIDACIÓN EN DISCO
# =========================================================
print("\n--- Guardando resultados en formato comprimido de NumPy ---")

# Convertir las listas a arrays de NumPy para poder guardarlos eficientemente
np.savez_compressed(
    "resultados_evaluacion_tfg.npz",
    lista_r_pearson=np.array(lista_r_pearson),
    lista_dls_ia=np.array(lista_dls_ia),
    lista_dls_target=np.array(lista_dls_target),
    residuos_acumulados=np.array(residuos_acumulados),
    # Guardar el eje L y los mapas del ejemplo para no perderlos
    l_axis=l_axis,
    mapa_ejemplo_in=mapas_finales['in'],
    mapa_ejemplo_target=mapas_finales['target'],
    mapa_ejemplo_ia=mapas_finales['ia'],
    mapa_ejemplo_res=mapas_finales['res']
)

print("¡Hecho! Todo el análisis de los 220 parches se ha guardado en 'resultados_evaluacion_tfg.npz'")