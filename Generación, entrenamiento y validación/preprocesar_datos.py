import os
import torch
from cmbnncs import simulator
from tqdm import tqdm
import gc

# Configuración
nside = 512
num_mapas_train = 1000 # Si no es 1000 ajustar a los que se quiera todo
ruta_base = "./dataset_tfg"
ruta_train = os.path.join(ruta_base, "train")
ruta_val = os.path.join(ruta_base, "val")

# Crear estructura de carpetas
os.makedirs(ruta_train, exist_ok=True)
os.makedirs(ruta_val, exist_ok=True)

print(f"--- Generando {num_mapas_train} mapas de entrenamiento ---")
for i in tqdm(range(num_mapas_train)):
    seed = 2000 + i
    if i < 800:
        nombre_archivo = os.path.join(ruta_train, f"mapa_train_{seed}.pt")
    else:
        nombre_archivo = os.path.join(ruta_val, f"mapa_val_{seed}.pt")
    if os.path.exists(nombre_archivo): continue

    m_in, m_pure, _, std_target = simulator.sim_CMB(seed=seed, nside=nside)
    # Guardar como un diccionario de tensores
    data = {
        'input': torch.from_numpy(m_in).float(),
        'target': torch.from_numpy(m_pure).float(),
        'std_target': float(std_target)
    }
    torch.save(data, nombre_archivo)

    # Limpieza 
    del m_in, m_pure, data
    gc.collect()

print(f"\n--- Proceso completado ---")
print(f"Total Train: {len(os.listdir(ruta_train))}")
print(f"Total Val: {len(os.listdir(ruta_val))}")