import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import camb
import pandas as pd
import scipy.stats
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

# =========================================================
# CARGA DE DATOS DESDE EL CACHÉ Y VARIABLES BASE
# =========================================================
print("--- Cargando resultados cacheados ---")
datos = np.load("resultados_evaluacion_tfg.npz")

lista_r_pearson = datos['lista_r_pearson'].tolist()
lista_dls_ia = datos['lista_dls_ia'].tolist()
lista_dls_target = datos['lista_dls_target'].tolist()
residuos_acumulados = datos['residuos_acumulados'].tolist()
l_axis = datos['l_axis']

# Reconstrucción del diccionario de los mapas de la Figura 1
mapas_finales = {
    'in': datos['mapa_ejemplo_in'],
    'target': datos['mapa_ejemplo_target'],
    'ia': datos['mapa_ejemplo_ia'],
    'res': datos['mapa_ejemplo_res']
}

# Variables de configuración necesarias para las gráficas
nside = 512
indice_mapa = 0
pix_size_arcmin = 1.5
ruta_datos = "./dataset_tfg_final/val"
archivos_val = sorted([f for f in os.listdir(ruta_datos) if f.endswith('.pt')])

mask_base = np.zeros((nside, nside))
margin = int(nside * 0.1) 
mask_base[margin:-margin, margin:-margin] = 1.0
mask_apodized = gaussian_filter(mask_base, sigma=5.0)
f_sky_corr = np.mean(mask_apodized**2)

# =========================================================
# FUNCIONES AUXILIARES NECESARIAS
# =========================================================
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
    # Creación de rejilla radial de distancias en píxeles
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
    dl_binned, bin_edges, _ = scipy.stats.binned_statistic(l, dl, statistic='mean', bins=l_bins)
    l_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return l_centers, dl_binned

def obtener_teoria_camb(lmax=2000):
    pars_t = camb.CAMBparams()
    pars_t.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, omk=0, tau=0.0544)
    pars_t.InitPower.set_params(As=2.1e-9, ns=0.9649)
    pars_t.set_for_lmax(lmax)
    results_t = camb.get_results(pars_t)
    powers_t = results_t.get_cmb_power_spectra(pars_t, CMB_unit='muK')
    return np.arange(len(powers_t['total'][:, 0])), powers_t['total'][:, 0]

# =========================================================
# CÁLCULO DE MÉTRICAS GLOBALES
# =========================================================
dl_ia_media = np.mean(lista_dls_ia, axis=0)
dl_ia_std = np.std(lista_dls_ia, axis=0)
dl_target_media = np.mean(lista_dls_target, axis=0)
cov_matrix = np.cov(np.array(lista_dls_ia).T)
r_medio = np.mean(lista_r_pearson)
r_std = np.std(lista_r_pearson)
mapa_snr = np.abs(mapas_finales['ia']) / (np.std(residuos_acumulados, axis=0) + 1e-6)
bias_relativo = 100 * (dl_ia_media - dl_target_media) / np.where(dl_target_media==0, 1, dl_target_media)
print(f"-> Correlación de Pearson media (r): {r_medio:.4f} ± {r_std:.4f}")

# =========================================================
#  VALIDACIÓN TEÓRICA (CAMB)
# =========================================================
# Definición de los parámetros base del modelo Lambda-CDM
H0_val = 67.36
ombh2_val = 0.02237
omch2_val = 0.1200
ns_val = 0.9649
As_val = 2.1e-9
tau_val = 0.0544

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=0.06, omk=0, tau=tau_val)
pars.InitPower.set_params(As=As_val, ns=ns_val)
pars.set_for_lmax(int(l_axis.max()))
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
dl_teorico = powers['total'][:, 0]; l_teorico = np.arange(len(dl_teorico))

# =========================================================
# GRÁFICAS
# =========================================================

# FIGURA 1: COMPARATIVA DE MAPAS
parche_pt = torch.load(os.path.join(ruta_datos, archivos_val[indice_mapa]), map_location='cpu', weights_only=True)
mapas_list = [
    mapas_finales['in'][7],     # El mapa input
    mapas_finales['target'],    # El target CMB real
    mapas_finales['ia'],        # La reconstrucción
    mapas_finales['res']        # El residuo
]
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
titulos = ["Input (Canal 7)", "Target Real (CMB)", "IA Reconstruido", "Residuos (Error)"]
for i in range(4):
    if i == 0:
        v = 3 * np.std(mapas_list[0])
    elif i == 3:
        v = 3 * np.std(mapas_list[3])
    else:
        # El Target y la red deben compartir la misma escala de color para ser comparables
        v = 3 * np.std(mapas_finales['target']) 
    im = axes[i].imshow(mapas_list[i], cmap='RdYlBu_r', origin='lower', vmin=-v, vmax=v)
    axes[i].set_title(titulos[i], fontweight='bold')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
plt.savefig("1_comparativa_mapas.png")

# FIGURA 2: ESPECTRO
mask_l = (l_axis > 10) & (l_axis < 2000)
l_filtrado = l_axis[mask_l]
dl_ia_filtrado = dl_ia_media[mask_l]
dl_std_filtrado = dl_ia_std[mask_l]
l_bin, dl_bin = binear_esfera(l_filtrado, dl_ia_filtrado, bin_size=25)
_, std_bin = binear_esfera(l_filtrado, dl_std_filtrado, bin_size=25)
plt.figure(figsize=(10, 6))
plt.plot(l_teorico, dl_teorico, 'k--', label='Espectro Planck', alpha=0.3)
plt.errorbar(l_bin, dl_bin, yerr=std_bin, fmt='o', ms=4, color='blue', label='Reconstrucción Red', capsize=3)
plt.xscale('linear')
plt.yscale('linear')
plt.ylim(0, 6500)
plt.xlim(0, 2000)
plt.xlabel(r'Multipolo $l$')
plt.ylabel(r'$D_l \equiv l(l+1)C_l / 2\pi$ [$\mu K^2$]')
plt.title('Espectro de Potencias Angular')
plt.legend(loc='upper right', frameon=True)
plt.grid(True, which="major", alpha=0.5, linestyle='--')
plt.tight_layout()
plt.savefig("2_espectro.png", dpi=300)

# FIGURA 3: CORRELACIÓN T-T (SCATTER)
lim = 500
plt.figure(figsize=(8, 8))
mask_central = mask_apodized > 0.99  # Solo el centro limpio
x = mapas_finales['target'][mask_central].flatten()
y = mapas_finales['ia'][mask_central].flatten()
# Calcular correlación lineal (R)
r_pearson = np.corrcoef(x, y)[0, 1]
# Dibujar con hexbin
targets_flat = mapas_finales['target'][mask_central].flatten()
outputs_flat = mapas_finales['ia'][mask_central].flatten()
hb = plt.hexbin(x, y, gridsize=100, cmap='inferno', bins='log', extent=[-lim, lim, -lim, lim])
cb = plt.colorbar(hb, label='log10(N píxeles)')
plt.plot([-lim, lim], [-lim, lim], 'w--', alpha=0.7, label=f'Ideal (1:1)| r = {r_pearson:.3f}')
plt.xlabel(r'CMB Target ($\mu K$)')
plt.ylabel(r'CMB Recuperado red ($\mu K$)')
plt.title('Correlación Píxel a Píxel (TT)')
plt.legend()
plt.grid()
plt.savefig("3_correlacion_tt.png")

# FIGURA 4: ZOOM AL CENTRO GALÁCTICO
c = nside // 2; z = nside // 4
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(mapas_finales['in'][7, c-z:c+z, c-z:c+z], cmap='RdYlBu_r')
axes[1].imshow(mapas_finales['target'][c-z:c+z, c-z:c+z], cmap='RdYlBu_r')
axes[2].imshow(mapas_finales['ia'][c-z:c+z, c-z:c+z], cmap='RdYlBu_r')
for ax in axes: ax.axis('off')
plt.savefig("4_zoom_galactico.png")

# FIGURA 5: MATRIZ DE COVARIANZA
plt.figure(figsize=(8, 6))
limite_cov = min(40, cov_matrix.shape[0])
corr_matrix = np.corrcoef(np.array(lista_dls_ia).T)
plt.imshow(cov_matrix[2:limite_cov, 2:limite_cov], cmap='inferno')
plt.colorbar(label=r"Covarianza [$\mu K^4$]")
plt.title("Matriz de Covarianza entre Multipolos")
plt.savefig("5_matriz_covarianza.png")

# FIGURA 6 y 8: ANÁLISIS DE SESGO SISTEMÁTICO y FUNCIÓN DE TRANSFERENCIA
# Configuramos la figura
fig, ax1 = plt.subplots(figsize=(10, 6))
l_bin, dl_ia_bin = binear_esfera(l_filtrado, dl_ia_media[mask_l], bin_size=25)
_, dl_target_bin = binear_esfera(l_filtrado, dl_target_media[mask_l], bin_size=25)
t_func_bin = dl_ia_bin / dl_target_bin
mask_l_plot = (l_axis > 2) & (l_axis < 1500) 
# Eje izquierdo: Función de Transferencia (dl_ia_bin / dl_target_bin)
ax1.plot(l_bin, t_func_bin, 'ro-', lw=2, ms=5, label='Transferencia red/Target')
ax1.axhline(1, color='black', lw=1, linestyle='--')
ax1.fill_between(l_bin, 0.9, 1.1, color='green', alpha=0.1, label='Margen ±10%')
ax1.set_xscale('log')
ax1.set_ylim(0, 1.5)
ax1.set_xlabel('Multipolo $l$')
ax1.set_ylabel(r'Función de Transferencia $T(l)$', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True, which="both", alpha=0.2)
# Eje derecho: Sesgo Relativo
ax2 = ax1.twinx() 
ax2.plot(l_axis[mask_l_plot], bias_relativo[mask_l_plot], color='purple', label='Sesgo Sistemático', alpha=0.7)
ax2.axhline(0, color='purple', lw=1, linestyle=':', alpha=0.5)
ax2.set_ylabel('Sesgo [%]', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
plt.title('Caracterización Espectral: Transferencia y Sesgo')
# Para unir las leyendas de ambos ejes en una sola caja
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')
ax1.set_ylim(0, 1.2)
ax2.set_ylim(-100, 20)
plt.tight_layout()
plt.savefig("6_y_8_analisis_espectral_unificado.png")

# FIGURA 7: MAPA DE FIABILIDAD (SNR)
plt.figure(figsize=(8, 6))
plt.imshow(mapa_snr * mask_apodized, cmap='viridis', origin='lower', vmin=0, vmax=5)
plt.colorbar(label='SNR')
plt.axis('off')
plt.title("Mapa de Fiabilidad Local (SNR)")
plt.savefig("7_resultado_snr.png")

# FIGURA 9: ESPECTRO DE RESIDUOS (Ruido de la red)
plt.figure(figsize=(10, 6))
l_res, dl_res = calcular_psd_2d(mapas_finales['res'] * mask_apodized)
mask_res_plot = (l_res > 10) & (l_res < 2000)
plt.plot(l_res[mask_res_plot], dl_res[mask_res_plot] / f_sky_corr, color='blue', label='Potencia de Residuos (Ruido red)')
plt.plot(l_axis[mask_l], dl_target_media[mask_l], 'r--', alpha=0.5, label='Señal Target')
plt.xlim(10, 2000)
plt.ylim(0, np.max(dl_target_media[mask_l]) * 1.2) 
plt.xlabel('Multipolo $l$')
plt.ylabel('$D_l$')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right')
plt.title("Análisis del Ruido Residual")
plt.savefig("9_psd_residuos.png")

# Figura 10: ESTADÍSTICA DE PÍXELES (HISTOGRAMA)
plt.figure(figsize=(9, 6))
# Cálculo del error residual solo en la zona central (sin bordes)
mask_central = mask_apodized > 0.99
residuos_pixel = (mapas_finales['target'] - mapas_finales['ia'])[mask_central].flatten()
# Histograma de los datos
n, bins, patches = plt.hist(residuos_pixel, bins=80, density=True, 
                            alpha=0.6, color='skyblue', label='Residuos red')
# Ajuste de una Gaussiana teórica para comparar
mu, std = scipy.stats.norm.fit(residuos_pixel)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = scipy.stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2, label=fr'Ajuste Gaussiano' + '\n' + fr'($\mu={mu:.3f}$, $\sigma={std:.3f}$)')
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.title('Distribución Estadística del Error en los Píxeles', fontweight='bold')
plt.xlabel(r'Diferencia Temperatura [$\mu K$]')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig("10_histograma_residuos.png")

# =========================================================
# EXPORTACIÓN DE RESULTADOS Y PARÁMETROS
# =========================================================
beam_efectivo_arcmin = 10.0
def encontrar_pico(l_arr, dl_arr, l_min, l_max):
    mask = (l_arr > l_min) & (l_arr < l_max)
    if not any(mask): return np.nan
    return l_arr[mask][np.argmax(dl_arr[mask])]

# Picos y chi2
print("\n--- Analizando estructura de picos y bondad de ajuste ---")
l_pico1 = encontrar_pico(l_axis, dl_ia_media, 180, 250)
l_pico2 = encontrar_pico(l_axis, dl_ia_media, 500, 600)
l_pico3 = encontrar_pico(l_axis, dl_ia_media, 750, 850)

l_max_u = min(len(dl_ia_media), len(dl_teorico))
r_u = slice(2, l_max_u)

# Error estadístico: Combinación de Varianza de la red y Varianza Cósmica
c_var = np.sqrt(2 / (2 * l_axis[r_u] + 1)) * dl_teorico[r_u]
err_t = np.sqrt(dl_ia_std[r_u]**2 + c_var**2 + 1e-2)
chi2_global_red = np.mean((dl_ia_media[r_u] - dl_teorico[r_u])**2 / err_t**2)

# Ajuste para hallar los 6 parámetros
def objetivo_completo(p, l_data, dl_data, beam_val):
    # p = [H0, ombh2, omch2, ns, As_escalada, tau]
    h0_p, ob_p, oc_p, ns_p, as_p, tau_p = p
    try:
        pars_t = camb.CAMBparams()
        pars_t.set_cosmology(H0=h0_p, ombh2=ob_p, omch2=oc_p, mnu=0.06, tau=tau_p)
        pars_t.InitPower.set_params(As=as_p*1e-9, ns=ns_p)
        pars_t.set_for_lmax(int(l_data.max()) + 50)
        results_t = camb.get_results(pars_t)
        dl_teorico_t = results_t.get_cmb_power_spectra(pars_t, CMB_unit='muK')['total'][:, 0]
        # Aplicación del beam efectivo (realmente no hay uno solo)
        l_arr = np.arange(len(dl_teorico_t))
        sigma = np.radians(beam_efectivo_arcmin/60) / 2.355
        bl = np.exp(-l_arr * (l_arr + 1) * (sigma**2))
        dl_modelo = dl_teorico_t * bl
        # Comparación en espacio logarítmico: 
        # Esto hace que la caída del beam sea una diferencia lineal, mucho más fácil de optimizar
        idx = l_data.astype(int)
        diff = np.log10(dl_data + 1) - np.log10(dl_modelo[idx] + 1)
        # Peso l*(l+1) o l^2 para dar importancia a las escalas pequeñas
        pesos = (l_data / 100)**2 
        return np.sum(pesos * (diff**2))
    except:
        return 1e12

# Gráfica de diagnóstico de Beam
plt.figure(figsize=(10, 5))
l_eval = l_axis[mask_l].astype(int)
ratio = dl_ia_media[mask_l] / dl_teorico[l_eval] 
# Beam teórico para comparar
bl_15 = np.exp(-l_filtrado*(l_filtrado+1)*(np.radians(1.5/60)/2.355)**2)
bl_12 = np.exp(-l_filtrado*(l_filtrado+1)*(np.radians(5/60)/2.355)**2)
bl_10 = np.exp(-l_filtrado*(l_filtrado+1)*(np.radians(10/60)/2.355)**2)
plt.plot(l_filtrado, ratio, label='Ratio Datos red / Teoría Pura')
plt.plot(l_filtrado, bl_15, 'r:', label="Curva esperada (Beam 1.5')")
plt.plot(l_filtrado, bl_12, color='purple', linestyle='--', linewidth=1.0, label="Curva esperada (Beam 5')")
plt.plot(l_filtrado, bl_10, 'k--', label="Curva esperada (Beam 10')")
plt.yscale('log')
plt.ylim(0.015, 1.2)
plt.xlabel(r'Multipolo $l$')
plt.ylabel(r'$D_l^{red} / D_l^{Teórico}$')
plt.title("Análisis de Sensibilidad al Beam")
plt.legend()
plt.grid(True, which="both", alpha=0.2)
plt.savefig("extra_sensibilidad_beam.png", dpi=300)

def aplicar_beam_a_teoria(l_axis, dl_vals, fwhm_arcmin):
    fwhm_rad = np.radians(fwhm_arcmin / 60.0)
    sigma = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    # Calculamos el factor de atenuación gaussiano
    w_l = np.exp(-l_axis * (l_axis + 1) * sigma**2)
    return dl_vals * w_l
# Generar la teoría
l_camb, dl_camb = obtener_teoria_camb(lmax=2000)
# Aplicar el Beam a esa teoría
dl_teorica_con_beam = aplicar_beam_a_teoria(l_camb, dl_camb, fwhm_arcmin=10.0)
# Calcular la media de los mapas
dl_ia_media = np.mean(lista_dls_ia, axis=0)
dl_ia_std = np.std(lista_dls_ia, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(l_camb, dl_teorica_con_beam, color='red', label=r'Teoría $\Lambda$CDM (con Beam 10\')', lw=2)
plt.plot(l_camb, dl_camb, color='red', linestyle='--', alpha=0.3, label='Teoría pura (sin beam)')
plt.plot(l_axis, dl_ia_media, color='blue', label='red (Media 20 mapas)', lw=1.5)

plt.fill_between(l_axis, dl_ia_media - dl_ia_std, dl_ia_media + dl_ia_std, 
                 color='blue', alpha=0.2, label=r'Desviación estándar (1$\sigma$)')
plt.ylim(-1000, 10000)
plt.xlabel(r'Multipolo $\ell$')
plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
plt.title('Validación Científica: Red vs Modelo Teórico LiteBIRD')
plt.xlim(2, 2000)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("2b_validacion_teorica_litebird.png", dpi=300)

print("--- Iniciando optimización multivariable (Hallando Cosmología) ---")
# Punto de partida (Planck) y límites físicos
x0 = [67.3, 0.022, 0.12, 0.96, 2.1, 0.05]
bounds = [
    (20.0, 90.0),       # H0
    (0.015, 0.045),     # Omega_b h2
    (0.05, 0.60),       # Omega_c h2
    (0.90, 3.0),        # n_s
    (0.5, 2.0),         # A_s
    (0.00, 1.00)        # tau
]

res_6params = minimize(objetivo_completo, x0=x0, bounds=bounds,
                       args=(l_filtrado, dl_ia_media[mask_l], beam_efectivo_arcmin), 
                       method='L-BFGS-B')

h_ia, ob_ia, oc_ia, ns_ia, as_ia, tau_ia = res_6params.x

# =========================================================
# EXPORTACIÓN DE RESULTADOS Y PARÁMETROS COSMOLÓGICOS
# =========================================================
# Parámetros base y objetivos de referencia
params_nombres = ['H0', 'Omega_b h2', 'Omega_c h2', 'n_s', 'A_s (x1e-9)', 'tau']
valores_target = [H0_val, ombh2_val, omch2_val, ns_val, As_val*1e9, tau_val]
valores_ia = [h_ia, ob_ia, oc_ia, ns_ia, as_ia, tau_ia]

# =========================================================
# CÁLCULO DE INCERTIDUMBRES MEDIANTE DIFERENCIAS FINITAS (HESSIANO)
# =========================================================
print("\n--- Calculando barras de error asociadas a los parámetros ---")
errores_ia = []

# Incrementos característicos para la derivada numérica (se adaptan a la escala de cada parámetro)
eps = [0.1, 1e-4, 1e-3, 1e-3, 0.01, 1e-3] 
f_zero = res_6params.fun

for i in range(6):
    p_plus = valores_ia.copy()
    p_minus = valores_ia.copy()
    p_plus[i] += eps[i]
    p_minus[i] -= eps[i]
    # Evaluar la función a la derecha y a la izquierda
    f_plus = objetivo_completo(p_plus, l_filtrado, dl_ia_media[mask_l], beam_efectivo_arcmin)
    f_minus = objetivo_completo(p_minus, l_filtrado, dl_ia_media[mask_l], beam_efectivo_arcmin)
    # Segunda derivada parcial aproximada: d2f/dx2 = (f(x+e) - 2f(x) + f(x-e)) / e^2
    d2f_dx2 = (f_plus - 2 * f_zero + f_minus) / (eps[i] ** 2)
    # En una función de tipo Chi-cuadrado, la incertidumbre 1-sigma es sqrt(2 / (d2f/dx2))
    if d2f_dx2 > 0:
        sigma = np.sqrt(2.0 / d2f_dx2)
    else:
        # Salvaguarda estadística en caso de inestabilidad numérica en la región plana de la frontera
        sigma = valores_ia[i] * 0.02 
    errores_ia.append(sigma)

# =========================================================
# CONSTRUCCIÓN DEL DATAFRAME DE RESULTADOS (PANDAS)
# =========================================================
# Creación del DataFrame principal con los parámetros físicos y sus incertidumbres 1-Sigma
df_parametros = pd.DataFrame({
    'Parametro': params_nombres,
    'Target_Planck': valores_target,
    'Hallado_IA': valores_ia,
    'Error_1Sigma': errores_ia
})

# Cálculo del error relativo porcentual basado en el valor central
df_parametros['Error_Relativo_%'] = (np.abs(df_parametros['Hallado_IA'] - df_parametros['Target_Planck']) / df_parametros['Target_Planck']) * 100

# Creación de las filas correspondientes a las métricas de validación adicionales
metricas_extra = pd.DataFrame({
    'Parametro': ['--- Metricas de Validacion ---', 'Chi2_Reducido', 'Pearson_r', 'Pico_1_l', 'Pico_2_l', 'Pico_3_l'],
    'Target_Planck': [np.nan, 0.0, 1.0, 220, 540, 800],
    'Hallado_IA': [np.nan, chi2_global_red, r_medio, l_pico1, l_pico2, l_pico3],
    'Error_1Sigma': [np.nan]*6,
    'Error_Relativo_%': [np.nan]*6
})

# Concatenar ambos DataFrames de forma limpia asegurando que coincidan todas las columnas
df_final = pd.concat([df_parametros, metricas_extra], ignore_index=True)
# Exportación final a disco en formato CSV estructurado
df_final.to_csv("parametros_cosmologicos_extraidos.csv", index=False)

# =========================================================
# IMPRESIÓN REPORTE TERMINAL
# =========================================================
print("\n" + "="*75)
print("   INFERENCIA CIENTÍFICA: PARÁMETROS RECUPERADOS POR LA IA (1-SIGMA)")
print("="*75)
# Tabla limpia con los datos comparables
columnas_mostrar = ['Parametro', 'Target_Planck', 'Hallado_IA', 'Error_1Sigma', 'Error_Relativo_%']
print(df_final[columnas_mostrar].dropna(subset=['Target_Planck', 'Error_Relativo_%']).to_string(index=False))
print("-" * 75)
print(f"RESUMEN ESTADÍSTICO DE VALIDACIÓN:")
print(f" > Bondad de ajuste global (Chi2 Red): {chi2_global_red:.4f}")
print(f" > Fidelidad de imagen píxel (Pearson r medio): {r_medio:.4f}")
print(f" > Posición del Primer Pico Acústico: {l_pico1:.1f} (Modelo Ideal: ~220)")
print("="*75)
print("Análisis terminado con éxito. Se han generado las gráficas y el archivo 'parametros_cosmologicos_extraidos.csv'.")