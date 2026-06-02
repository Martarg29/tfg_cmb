import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from cmbnncs import unet, simulator
import gc
import os
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import pandas as pd

# =========================================================
# PROTOCOLO DE SEGURIDAD PARA MULTIPROCESAMIENTO
# =========================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Definición del Dataset fuera para que sea serializable
class CMBDataset(torch.utils.data.Dataset):
    def __init__(self, ruta):
        self.ruta = ruta
        # Listar archivos y los ordenar para que sea reproducible
        self.archivos = sorted([f for f in os.listdir(ruta) if f.endswith('.pt')])
    def __len__(self):
        return len(self.archivos)
    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.ruta, self.archivos[idx]), map_location='cpu')
        t_in = data['input']
        t_target = data['target']
        # Rotación aleatoria (0, 90, 180, 270 grados)
        k = np.random.randint(0, 4)
        if k > 0:
            t_in = torch.rot90(t_in, k, dims=[-2, -1])
            t_target = torch.rot90(t_target, k, dims=[-2, -1])
        return t_in, t_target

# =========================================================
# CONFIGURACIÓN Y HARDWARE
# =========================================================
# Parámetros LiteBIRD
freqs_litebird = [40, 50, 60, 68, 78, 89, 100, 119, 140, 166, 195, 235, 280, 337, 402]
nside = 512
lr = 5e-5

# =========================================================
# BLOQUE DE ANÁLISIS DE DATOS 
# =========================================================
print("\n--- Generando galerías de componentes para la memoria del TFG ---")

# Definición de la Función de Galería
def generar_galeria(lista_mapas, freqs, titulo_base, nombre_archivo):
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(titulo_base, fontsize=20, fontweight='bold', y=0.98)
    # Si viene de GPU, se pasa a CPU y numpy
    if torch.is_tensor(lista_mapas):
        lista_mapas = lista_mapas.detach().cpu().numpy()
    # Lista plana de ejes para iterar fácilmente
    axes_flat = axes.flatten()
    for i in range(15): 
        ax = axes_flat[i]
        v_min = np.percentile(lista_mapas[i], 1)
        v_max = np.percentile(lista_mapas[i], 99)
        im = ax.imshow(lista_mapas[i], cmap='RdYlBu_r', origin='lower', vmin=v_min, vmax=v_max)
        ax.set_title(f"Canal: {freqs[i]} GHz", fontsize=12)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(nombre_archivo)
    plt.close(fig)
    del fig, axes

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Ejecutando en: {device} ---")
    m_in, m_pure, comps, std_target = simulator.sim_CMB(seed=42, nside=nside)
    generar_galeria(m_in, freqs_litebird, "Input Total (CMB + Foregrounds)", "galeria_input_total.png")
    # Crear el target repetido solo para la foto y borrarlo
    m_pure_gal = np.repeat(m_pure, 15, axis=0) 
    generar_galeria(m_pure_gal, freqs_litebird, "CMB Puro", "galeria_cmb_puro.png")
    # Generación dinámica de componentes desde el diccionario comps
    claves_visualizar = {
        'dust': 'Polvo Galáctico', 
        'sync': 'Sincrotrón', 
        'ame': 'AME (Polvo Anómalo)', 
        'free': 'Emisión Free-Free', 
        'cib': 'Fondo Infrarrojo Cósmico (CIB)', 
        'co': 'Líneas de Monóxido de Carbono (CO)', 
        'radio': 'Radio Galaxias Puntuales (WebSky)',
        'tsz': 'Efecto tSZ (WebSky)',
        'ksz': 'Efecto kSZ (WebSky)',
        'noise': 'Ruido Instrumental LiteBIRD'
    }

    # =========================================================
    # GALERÍA ESPECÍFICA: TODOS LOS CONTAMINANTES A 119 GHz
    # =========================================================
    print("--- Generando galería de contaminantes a 119 GHz (Canal 8) ---")
    idx_119 = 7  # Índice de 119 GHz en freqs_litebird
    # Recopilar los nombres y los mapas del canal 8
    nombres_119 = []
    mapas_119 = []
    # Añadir el CMB puro y el Input Total
    nombres_119.extend(["Input Total (Mezcla)", "Target (CMB Puro)"])
    mapas_119.extend([m_in[idx_119], m_pure[0]])
    for c, nombre in claves_visualizar.items():
        if c in comps:
            nombres_119.append(nombre)
            mapas_119.append(comps[c][idx_119])
    # Configuración de la cuadrícula
    n_mapas = len(nombres_119)
    cols = 4
    rows = int(np.ceil(n_mapas / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Desglose de Emisión a 119 GHz (Canal 8)", fontsize=20, fontweight='bold', y=0.98)
    axes_flat = axes.flatten()
    
    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n_mapas:
            mapa_actual = mapas_119[i]
            # Asegurar que sea numpy array
            if torch.is_tensor(mapa_actual):
                mapa_actual = mapa_actual.detach().cpu().numpy() 
            # Calcular percentiles para no saturar el color con píxeles extremos
            v_min = np.percentile(mapa_actual, 1)
            v_max = np.percentile(mapa_actual, 99)
            im = ax.imshow(mapa_actual, cmap='RdYlBu_r', origin='lower', vmin=v_min, vmax=v_max)
            ax.set_title(nombres_119[i], fontsize=12, fontweight='bold')
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # Ocultar los ejes vacíos si la cuadrícula no está llena
            ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("galeria_resumen_119GHz.png")
    plt.close(fig)
    print("--- Galería de 119 GHz guardada con éxito ---")

    # Limpieza de memoria RAM
    for c, nombre in claves_visualizar.items():
        if c in comps:
            generar_galeria(comps[c], freqs_litebird, f"Componente: {nombre}", f"galeria_{c}.png")
    del comps, m_in, m_pure, m_pure_gal
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("--- Galerías guardadas y RAM liberada ---")

    # =========================================================
    # DEFINICIÓN DEL MODELO
    # =========================================================
    if nside == 512:
        channels_final = [32, 64, 128, 256, 512, 512, 512, 512]
        k_size = [4] * 16
        s_tri = [2] * 16 
        e_pad = [0] * 16
        model = unet.UNet8(channels=channels_final,channel_in=15, channel_out=1,kernels_size=k_size,strides=s_tri,extra_pads=e_pad,sides=nside).to(device)
    elif nside == 64:
        channels_final = [32, 64, 128, 256, 512]
        k_size = [4] * 10
        s_tri = [2] * 10 
        e_pad = [0] * 10
        model = unet.UNet5(channels=channels_final,channel_in=15, channel_out=1,kernels_size=k_size,strides=s_tri,extra_pads=e_pad,sides=nside).to(device)

    # =========================================================
    # GENERACIÓN DE LA MÁSCARA (Apodizada)
    # =========================================================
    size = nside
    mask_base = np.zeros((size, size))
    margin = int(size * 0.1) # Dejamos un 10% de margen
    mask_base[margin:-margin, margin:-margin] = 1.0
    # Suavizado directo en 2D (Apodización)
    sigma_mask = 3.0 if nside == 64 else 15.0
    mask_apodized = gaussian_filter(mask_base, sigma=sigma_mask)
    # Convertir a tensor para PyTorch
    mask_tensor = torch.from_numpy(mask_apodized).float().to(device).unsqueeze(0).unsqueeze(0)

    # =========================================================
    # BUCLE DE ENTRENAMIENTO CIENTÍFICO
    # =========================================================
    # Parámetros de control
    batch_size = 2
    ruta_base_datos = "./dataset_tfg_final"
    ruta_train = os.path.join(ruta_base_datos, "train")
    ruta_val = os.path.join(ruta_base_datos, "val")
    if not os.path.exists(ruta_train):
        print(f"ERROR: No existe la carpeta {ruta_train}")
        exit()
    train_dataset = CMBDataset(ruta=ruta_train)
    val_dataset = CMBDataset(ruta=ruta_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # Entrenamiento
    model.load_state_dict(torch.load("mejor_modelo_final.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    num_epochs = 0 # Veces que se entrena a la red con los mapas
    print(f"--- Iniciando entrenamiento: {len(train_dataset) + len(val_dataset)} mapas ---")
    best_val_loss = float('inf')

    def loss_espectro(pred, target, nside):
        # FFT y PSD
        psd_p = torch.abs(torch.fft.fft2(pred))**2
        psd_t = torch.abs(torch.fft.fft2(target))**2
        # Mapa de multipolos l
        h, w = pred.shape[-2:]
        yc, xc = torch.meshgrid(torch.fft.fftfreq(h).to(pred.device), 
                            torch.fft.fftfreq(w).to(pred.device), indexing='ij')
        l_map = torch.sqrt(xc**2 + yc**2) * nside # Conversión aproximada a multipolo l
        # Multiplicación por l(l+1) antes de aplicar el logaritmo
        # Esto resalta los picos acústicos antes de comprimir el rango dinámico
        weight = l_map * (l_map + 1)
        log_p = torch.log1p(weight * psd_p)
        log_t = torch.log1p(weight * psd_t)
        # Loss en escala logarítmica para que el error en multipolos altos (l>200) pese tanto como el primer pico
        return torch.mean((log_p - log_t)**2)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs}")
        for t_in, t_target in pbar:
            t_in, t_target = t_in.to(device), t_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=('cuda' if torch.cuda.is_available() else 'cpu')):
                outputs = model(t_in)
                # Aplicar la máscara para que la pérdida ignore los bordes
                outputs_m = outputs * mask_tensor
                target_m = t_target * mask_tensor
                # Pérdida de Píxel (L1); texturas
                loss_l1 = torch.sum(torch.abs(outputs_m - target_m)) / torch.sum(mask_tensor)
                # Pérdida de Píxel (MSE); estabilidad global
                loss_mse = torch.mean((outputs_m - target_m)**2)
                # Pérdida de Correlación (Pearson); similitud morfológica
                # r = cov(x,y) / (std(x)*std(y))
                vx = outputs_m - torch.mean(outputs_m)
                vy = target_m - torch.mean(target_m)
                loss_corr = 1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2)) + 1e-8)
                # Pérdida de Potencia (Espectro); Preserva los picos acústicos
                loss_pow = loss_espectro(outputs_m, target_m, nside)
                # Loss total
                loss = (0.05 * loss_l1) + (0.70 * loss_mse) + (0.20 * loss_pow) + (0.05 * loss_corr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})
            # Limpieza post batch
            del t_in, t_target, outputs, loss
        # Validación
        model.eval()
        v_loss_total = 0
        with torch.no_grad():
            for t_in_v, t_target_v in val_loader:
                t_in_v, t_target_v = t_in_v.to(device), t_target_v.to(device)
                with torch.amp.autocast(device_type=('cuda' if torch.cuda.is_available() else 'cpu')):
                    out_v = model(t_in_v)
                    # Aplicar máscara
                    out_vm = out_v * mask_tensor
                    target_vm = t_target_v * mask_tensor
                    # Pérdida de Píxel (L1)
                    v_l1 = torch.sum(torch.abs(out_vm - target_vm)) / torch.sum(mask_tensor)
                    # Pérdida de Píxel (MSE)
                    v_mse = torch.mean((out_vm - target_vm)**2)
                    # Pérdida de Correlación (Pearson)
                    vx_v = out_vm - torch.mean(out_vm)
                    vy_v = target_vm - torch.mean(target_vm)
                    v_corr = 1 - torch.sum(vx_v * vy_v) / (torch.sqrt(torch.sum(vx_v**2) * torch.sum(vy_v**2)) + 1e-8)
                    # Pérdida de Potencia (Espectro)
                    v_pow = loss_espectro(out_vm, target_vm, nside)
                    # Loss total de validación (Mismos pesos que en Train)
                    v_loss = (0.05 * v_l1) + (0.70 * v_mse) + (0.20 * v_pow) + (0.05 * v_corr)
                v_loss_total += v_loss.item()
                del t_in_v, t_target_v, out_v
        # Métricas
        avg_train = train_loss / len(train_loader)
        avg_val = v_loss_total / len(val_loader)
        scheduler.step(avg_val)
        print(f"Resumen Época {epoch+1}: Train={avg_train:.6f} | Val={avg_val:.6f}")
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train) 
        history["val_loss"].append(avg_val)
        # Guardado y limpieza de época
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "mejor_modelo_final.pth")
            print("--- Modelo guardado (mejora en validación) ---")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    # Finalización
    pd.DataFrame(history).to_csv("log_entrenamiento.csv", index=False)
    print("--- Proceso completado con éxito ---")
    
    # Representación de la curva de aprendizaje (solo se guarda el último entreno)
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Entrenamiento")
    plt.plot(history["epoch"], history["val_loss"], label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curva de Aprendizaje - Reconstrucción CMB")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("curva_aprendizaje.png")
    
    print("\n--- Proceso completado con éxito ---")
    print("1. Modelo guardado como 'mejor_modelo_final.pth'")
    print("2. Historial en 'log_entrenamiento.csv'")
    print("3. Gráfica en 'curva_aprendizaje.png'")