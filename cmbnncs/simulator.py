import numpy as np
import pysm3
import pysm3.units as u
import healpy as hp
import camb
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from cmbnncs import unet, simulator
import time
import gc
import pysm3.models.template as template

nside=512

# =============================================================================
# PARÁMETROS COSMOLÓGICOS (Actualizados a Planck 2018 - TT,TE,EE+lowE+lensing)
# =============================================================================
# Valores basados en la Tabla 2 de Aghanim et al. 2018 (Planck VI)
hubble = [67.36, 0.54]        # H0: constante de Hubble
ombh2 = [0.02237, 0.00015]       # Densidad de bariones
omch2 = [0.1200, 0.0012]         # Densidad de materia oscura fría
re_optical_depth = [0.0544, 0.0073] # Tau: reionización (Bajó mucho respecto a 2015)
scalar_amp_1 = [2.100e-9, 0.030e-9] # As: Amplitud de fluctuaciones
scalar_spectral_index_1 = [0.9649, 0.0042] # ns: Índice espectral

def sim_power_spectra(sim_H0, sim_ombh2, sim_omch2, sim_tau, sim_As, sim_ns, spectra_type='lensed_scalar'):
    pars = camb.model.CAMBparams()
    pars.set_cosmology(H0=sim_H0, ombh2=sim_ombh2, omch2=sim_omch2, tau=sim_tau)
    pars.InitPower.set_params(As=sim_As, ns=sim_ns)
    pars.set_for_lmax(2150, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars)
    unlensedtotCls = powers[spectra_type] * 7.42835025e12
    return unlensedtotCls[2:, :]

def get_spectra(random='Normal'):
    if random == 'Normal':
        sim_H0 = np.random.randn() * hubble[1] + hubble[0]
        sim_ombh2 = np.random.randn() * ombh2[1] + ombh2[0]
        sim_omch2 = np.random.randn() * omch2[1] + omch2[0]
        sim_tau = np.random.randn() * re_optical_depth[1] + re_optical_depth[0] 
        sim_As = np.random.randn() * scalar_amp_1[1] + scalar_amp_1[0]
        sim_ns = np.random.randn() * scalar_spectral_index_1[1] + scalar_spectral_index_1[0]
    return sim_power_spectra(sim_H0, sim_ombh2, sim_omch2, sim_tau, sim_As, sim_ns)

# =============================================================================
# CONFIGURACIÓN LITEBIRD (Instrumento)
# =============================================================================
LiteBIRD_Instrument = {
    'frequencies' : np.array([40, 50, 60, 68, 78, 89, 100, 119, 140, 166, 195, 235, 280, 337, 402]), 
    'use_smoothing' : False,
    'beams' : np.array([70.5, 58.5, 51.1, 41.6, 36.9, 33.0, 37.8, 33.6, 30.8, 28.9, 28.0, 24.7, 22.5, 20.9, 17.9]),
    'add_noise' : True,
    'sens_I' : np.array([37.42, 33.46, 21.31, 16.87, 12.07, 11.30, 6.56, 4.58, 4.79, 5.57, 5.85, 10.79, 13.80, 21.95, 47.45]), 
    'sens_P' : np.array([37.42, 33.46, 21.31, 16.87, 12.07, 11.30, 6.56, 4.58, 4.79, 5.57, 5.85, 10.79, 13.80, 21.95, 47.45])* np.sqrt(2), 
    'nside' : nside,
    'noise_seed' : 1111,
    'use_bandpass' : False,
    'output_directory' : './',
    'output_prefix' : '',
    'output_units' : 'uK_CMB'
}

# =============================================================================
# GENERAR LOS ARCHIVOS PARA EL ENTRENAMIENTO
# =============================================================================

class Spectra:
    def __init__(self, Cls = None, isCl = True, Name = '', Checkl = True):
        self.Name = Name
        self.Cls = Cls
        self.isCl = isCl
        self.isDl = not self.isCl
        if Checkl:
            self.lcheck()
    def lcheck(self, lmax = None):
        if type(self.Cls) != type(None):
            lmin = int(min(self.Cls[0]))
            if lmin > 0:
                tmp = np.zeros((np.shape(self.Cls)[0] - 1 , lmin)) 
                tmp1 = np.array([range(lmin)])
                tmp = np.concatenate((tmp1, tmp))
                self.Cls = np.concatenate((tmp, self.Cls), axis = 1)
            if lmax == None:
                lmax = max(self.Cls[0])
            lmax = int(lmax)
            self.Cls = self.Cls.transpose()[:lmax + 1].transpose()
    def Cl2Dl(self):
        if self.isDl:
            raise ValueError('Spectra have already been multiplied by factor l(l+1)/2/pi')
        data = self.Cls.transpose()
        for i in range(len(data)):
            data[i][1:] = data[i][1:] * data[i][0] * (data[i][0] + 1.) / 2. / np.pi
        self.Cls = data.transpose()
        self.isDl = True
        self.isCl = False
    def Dl2Cl(self):
        if self.isCl:
            raise ValueError('Spectra have already been divided by factor l(l+1)/2/pi')
        data = self.Cls.transpose()
        for i in range(len(data)):
            if data[i][0] == 0.:
                continue
            data[i][1:] = data[i][1:] / data[i][0] / (data[i][0] + 1.) * 2. * np.pi
        self.Cls = data.transpose()
        self.isDl = False
        self.isCl = True

# Definir nuestra propia clase para cargar los FITS
class MapaLocalEstandard:
    def __init__(self, ruta_fits, nside):
        self.nside = nside
        # Leemos el mapa directamente con healpy
        self.mapa = hp.read_map(ruta_fits, verbose=False)
        
    def get_emission(self, freq, weights=None):
        # Esta es la función que PySM3 llama internamente.
        # Devuelve el mapa con la unidad uK_CMB.
        # Añade una dimensión extra al principio porque PySM espera [I, Q, U]
        # y solo tenemos Intensidad (I).
        return np.array([self.mapa]) * u.uK_CMB

# Crear los objetos con la ruta a los archivos reducidos previamente
rg1_final  = MapaLocalEstandard("radio_n512.fits", nside=nside)
tsz1_final = MapaLocalEstandard("tsz_n512.fits", nside=nside)
ksz1_final = MapaLocalEstandard("ksz_n512.fits", nside=nside)
cib1_final  = MapaLocalEstandard("cib_n512.fits", nside=nside)

#pysm_names = ["d1", "s1", "a1", "f1", "co2", cib1, "rg1", "tsz1", "ksz1"]
# Configuramos el Sky
pysm_names_base = ["d1", "s1", "a1", "f1", "co2"]
sky_config = pysm3.Sky(nside=nside, 
                       preset_strings=pysm_names_base,
                       component_objects=[rg1_final, tsz1_final, ksz1_final, cib1_final])

def sim_CMB(seed, nside=nside, sky=sky_config, rot_custom=None):
    """
    - Usa get_spectra() para CMB físico (Planck 2018).
    - Usa las 15 frecuencias de LiteBIRD.
    - Aplica rotación aleatoria para cubrir todo el cielo.
    """
    # Configuración de frecuencias y parámetros de parche
    np.random.seed(seed)
    frecuencias = LiteBIRD_Instrument['frequencies']
    n_freqs = len(frecuencias)

    # Rotación aleatoria para no aprender siempre la misma zona de la galaxia
    # Opción a elegir el punto en el que se ejecuta
    if rot_custom is not None:
        rot_random = rot_custom
    else:
        rot_random = [np.random.uniform(0, 360), np.random.uniform(-90, 90)]
    reso_arcmin = 1.5
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600

    # CMB Físico (Target)
    # get_spectra() devuelve Dl = l(l+1)Cl / 2pi. synfast necesita Cl.
    dl_planck = get_spectra() # Dl en uK^2
    ls = np.arange(2, len(dl_planck) + 2)
    # Convertir Dl a Cl: Cl = Dl * 2pi / (l*(l+1))
    cl_planck = dl_planck[:, 0] * 2 * np.pi / (ls * (ls + 1))
    
    # Añadir los multipolos 0 y 1 (monopolo y dipolo) como 0
    cl_input = np.concatenate(([0, 0], cl_planck))

    # Generar el mapa esférico en uK (Valores típicos +- 300 uK)
    mapa_cmb_esf = hp.synfast(cl_input, nside, pixwin=True, verbose=False)

    # Inicializar cubos de componentes 
    comps = {name: np.zeros((n_freqs, nside, nside), dtype=np.float32) 
         for name in ['cmb', 'dust', 'sync', 'ame', 'free', 'cib', 'co', 'radio', 'tsz', 'ksz', 'noise']}
    # Bucle de Frecuencias
    for i, f in enumerate(frecuencias):
        f_ghz = f * u.GHz
        equiv = u.cmb_equivalencies(f_ghz)
        fwhm_rad = np.radians(LiteBIRD_Instrument['beams'][i]/60)

        # Proyectar CMB con el beam de esta frecuencia
        cmb_suave = hp.smoothing(mapa_cmb_esf, fwhm=fwhm_rad, verbose=False)
        comps['cmb'][i] = hp.gnomview(cmb_suave, rot=rot_random, xsize=nside, 
                                      reso=reso_arcmin, no_plot=True, return_projected_map=True)
        
        # Proyectar cada Foreground por separado
        # Mapear cada preset de PySM a su nombre en el diccionario comps
        nombres_mapeo = ['dust', 'sync', 'ame', 'free', 'co', 'radio', 'tsz', 'ksz', 'cib']
        for j, comp_obj in enumerate(sky.components):
            name = nombres_mapeo[j]
            # Extraer la emisión de la esfera
            esfera_fg = comp_obj.get_emission(f_ghz).to(u.uK_CMB, equiv)[0].value
            # Aplicar suavizado
            esfera_suave = hp.smoothing(esfera_fg, fwhm=fwhm_rad, verbose=False)
            # Proyectar
            comps[name][i] = hp.gnomview(esfera_suave, rot=rot_random, xsize=nside, 
                                         reso=reso_arcmin, no_plot=True, return_projected_map=True)
        
        # Ruido instrumental
        sigma_noise = LiteBIRD_Instrument['sens_I'][i] / np.sqrt(pixel_area_arcmin2)
        comps['noise'][i] = np.random.normal(0, sigma_noise, size=(nside, nside))
    
    # Generar el Target (El CMB puro sin ruido ni galaxia)
    target_suave_esf = hp.smoothing(mapa_cmb_esf, fwhm=fwhm_rad, verbose=False)
    m_pure_target = hp.gnomview(target_suave_esf, rot=rot_random, xsize=nside, 
                                reso=reso_arcmin, no_plot=True, 
                                return_projected_map=True).astype(np.float32)
    m_pure_target -= np.mean(m_pure_target)
    # Añadir la dimensión de canal (1, nside, nside)
    m_pure_target = m_pure_target[np.newaxis, ...]
    # Suma Total de componentes (Input X)
    m_in = np.sum(list(comps.values()), axis=0).astype(np.float32)
    # Normalizar cada uno de los 15 canales del input
    std_target = np.std(m_pure_target)
    m_pure_target /= (std_target + 1e-8) # Ahora el target tiene std=1
    for i in range(m_in.shape[0]):
        m_in[i] -= np.mean(m_in[i])
        m_in[i] /= (np.std(m_in[i]) + 1e-8)
    del mapa_cmb_esf, target_suave_esf
    
    return m_in, m_pure_target, comps, std_target
