import os
import sys
import subprocess
import shutil
import time

def ejecutar_comando(comando, descripcion):
    print(f"\n [DEVOPS] Ejecutando: {descripcion}...")
    try:
        resultado = subprocess.run(comando, shell=True, check=True, text=True, capture_output=True, encoding='utf-8')
        print(f" {descripcion} finalizado con éxito.")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Alerta/Log en {descripcion}: {e.stderr or e.output}")
        return False

def main():
    print("======================================================================")
    print("   ORQUESTADOR BLINDADO DE ACTUALIZACIÓN - TRADUCTOR VISUAL PRO   ")
    print("======================================================================")

    print("\n Paso 1: Terminando instancias huérfanas de Python y ejecutables...")
    mi_pid = os.getpid()
    try:
        cmd_pids = "wmic process where \"name='python.exe'\" get ProcessId"
        pids_raw = subprocess.check_output(cmd_pids, shell=True, text=True)
        for linea in pids_raw.splitlines():
            linea = linea.strip()
            if linea.isdigit():
                pid_detectado = int(linea)
                if pid_detectado != mi_pid:
                    os.system(f"taskkill /F /PID {pid_detectado} >nul 2>&1")
                    print(f"Correcto: se terminó el proceso 'python.exe' fantasma con PID {pid_detectado}.")
    except Exception:
        pass
        
    os.system("taskkill /F /IM main.exe >nul 2>&1")
    time.sleep(1)

    print("\n Paso 2: Purgando bases de datos y caché envenenada del disco duro...")
    rutas_cache = ["translation_cache", "cache.db", "server_run.log"]
    for ruta in rutas_cache:
        target = os.path.join(os.getcwd(), ruta)
        if os.path.exists(target):
            try:
                if os.path.isdir(target):
                    shutil.rmtree(target)
                else:
                    os.remove(target)
                print(f" Eliminado con éxito: '{ruta}'")
            except Exception as e:
                print(f" No se pudo eliminar '{ruta}': {e}")

    print("\n Paso 3: Reactivando el Pipeline Especializado de Manga (MIT)...")
    os.environ["SKIP_MIT_INIT"] = ""
    os.environ["PYTHONIOENCODING"] = "utf-8"

    spec_path = "main.spec"
    if os.path.exists(spec_path):
        print(f"\n Paso 4: Ajustando el archivo '{spec_path}' para incluir pesos y binarios de IA...")
        with open(spec_path, "r", encoding="utf-8") as f:
            contenido_spec = f.read()

        parches_datos = [
            ("('manga-image-translator', 'manga-image-translator')", "manga-image-translator"),
            ("('ocr_models', 'ocr_models')", "ocr_models")
        ]
        
        modificado = False
        for tupla_str, validacion in parches_datos:
            if validacion not in contenido_spec:
                contenido_spec = contenido_spec.replace("datas=[", f"datas=[{tupla_str}, ")
                modificado = True
                
        if modificado:
            with open(spec_path, "w", encoding="utf-8") as f:
                f.write(contenido_spec)
            print(" Estructura del archivo .spec actualizada con dependencias nativas de visión artificial.")
        else:
            print(" El archivo .spec ya cuenta con los mapeos de carpetas de modelos correctos.")

    pyinstaller_cmd = r".\env\Scripts\python.exe -m PyInstaller --noconfirm --clean main.spec"
    exito_compilacion = ejecutar_comando(pyinstaller_cmd, "Compilación limpia con PyInstaller (Utilizando 'env/')")

    if exito_compilacion:
        print("\n======================================================================")
        print(" ¡SISTEMA REPARADO COMPLETO! El ejecutable definitivo fue creado con éxito.")
        print("======================================================================")
        
        exe_final = os.path.join("dist", "main", "main.exe")
        if os.path.exists(exe_final):
            print("\n Paso 6: Lanzando el Traductor Visual Pro automáticamente en modo producción...")
            subprocess.Popen([exe_final], shell=True)
        else:
            exe_unico = os.path.join("dist", "main.exe")
            if os.path.exists(exe_unico):
                print("\n Paso 6: Lanzando el ejecutable único de Traductor Visual Pro...")
                subprocess.Popen([exe_unico], shell=True)
    else:
        print("\n Ocurrió un inconveniente durante el empaquetado. Revisa las trazas de logs de arriba.")

if __name__ == "__main__":
    main()
