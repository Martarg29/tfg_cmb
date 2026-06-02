import os
import sys
import shutil
import numpy as np
from pathlib import Path


def mkdir(path):
    '''
    Crea un directorio si no existe de forma segura.
    '''
    if not path:
        return
    # Usamos Path para manejar espacios y barras automáticamente según el OS
    p = Path(path.strip().replace(' ', ''))
    p.mkdir(parents=True, exist_ok=True)

def rmdir(path):
    '''
    remove a folder in a particular location if it is exists, otherwise, do nothing 
    '''
    p = Path(path)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
        print(f'Carpeta "{path}" eliminada correctamente.')

def savetxt(path, file_name, data):
    '''
    save the .txt files using np.savetxt() funtion
    '''
    mkdir(path)
    full_path = Path(path) / f"{file_name}.txt"
    np.savetxt(full_path, data)

def savedat(path, file_name, data):
    '''
    save the .dat files using np.savetxt() funtion
    '''
    mkdir(path)
    full_path = Path(path) / f"{file_name}.dat"
    np.savetxt(full_path, data)

def savenpy(path, file_name, data, dtype=np.float64):
    '''
    save an array to a binary file in NumPy .npy format using np.save() functiond
    '''
    mkdir(path)
    if isinstance(data, np.ndarray):
        data = data.astype(dtype)
    full_path = Path(path) / f"{file_name}.npy"
    np.save(full_path, data)

class Logger(object):
    ''' Redirige la consola a un archivo .log para guardar el historial. '''
    def __init__(self, path='logs', file_name="log", stream=sys.stdout):
        self.terminal = stream
        mkdir(path)
        self.log = open(Path(path) / f"{file_name}.log", "a", encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logger(path='logs', file_name='training'):
    ''' Activa el guardado automático de todo lo que salga por pantalla. '''
    sys.stdout = Logger(path=path, file_name=file_name, stream=sys.stdout)
    sys.stderr = Logger(path=path, file_name=file_name, stream=sys.stderr)
