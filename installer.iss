; Traductor Visual Pro — Inno Setup Installer
; Para compilar: ISCC.exe installer.iss
; Descarga Inno Setup: https://jrsoftware.org/isdl.php
;
; Características:
; - Instalación básica: ~1GB (ejecutable + runtime Python completo)
; - Modelos CT2 y OCR descargados bajo demanda en primer uso (ahorra ~500MB)
; - RapidOCR (ONNX) incluido vía pip, sin modelos adicionales
; - CTD eliminado (dependencia externa frágil, reemplazado por EasyOCR GPU + RapidOCR)
; - Desinstalación limpia

#define MyAppName "Traductor Visual Pro"
#define MyAppVersion "0.1.44"
#define MyAppPublisher "Traductor Visual"
#define MyAppURL "http://127.0.0.1:5174"
#define MyAppExeName "main.exe"

[Setup]
AppId={{E8F4A23B-8F2C-4C3B-9A1D-7F2E8C4D5B6A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=.\dist\installer
OutputBaseFilename=TraductorVisual_Setup_{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
; Tamaño mínimo: ~1GB (main.exe 360MB + _internal/ runtime 607MB)
; Modelos CT2 y OCR se descargan bajo demanda (~500MB extra)
ExtraDiskSpaceRequired=1000000000

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "&Crear acceso directo en el escritorio"; GroupDescription: "Accesos directos:"
Name: "downloadmodels"; Description: "&Descargar modelos de traducción ahora (~400MB)"; GroupDescription: "Modelos:"

[Files]
; .exe principal y runtime
Source: "dist\main.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\main\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

; Script de instalación
Source: "setup.ps1"; DestDir: "{app}"; Flags: ignoreversion

; RapidOCR y EasyOCR se instalan vía pip (no requieren modelos adicionales en bundle)
; CTD eliminado en Julio 2026 — reemplazado por EasyOCR GPU + RapidOCR (ONNX)

[Dirs]
Name: "{app}\env"; Permissions: users-modify
Name: "{app}\models\ct2"; Permissions: users-modify
Name: "{app}\ocr_models"; Permissions: users-modify
; RapidOCR usa modelos ONNX desde site-packages (pip), no requiere directorio propio

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Desinstalar {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Configurar entorno virtual
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File ""{app}\setup.ps1"" -InstallDir ""{app}"""; Flags: runhidden waituntilterminated; StatusMsg: "Configurando entorno virtual Python e instalando dependencias (~10 min)..."; AfterInstall: AfterSetup

; Iniciar servidor
Filename: "{app}\{#MyAppExeName}"; Flags: runhidden nowait; Description: "Iniciar {#MyAppName}"; StatusMsg: "Iniciando servidor..."

[UninstallRun]
Filename: "taskkill"; Parameters: "/F /IM main.exe"; Flags: runhidden skipifdoesntexist
Filename: "taskkill"; Parameters: "/F /IM python.exe"; Flags: runhidden skipifdoesntexist

[Code]
var
  DownloadPage: TDownloadWizardPage;

function InitializeSetup: Boolean;
begin
  Result := True;
end;

procedure AfterSetup;
var
  ResultCode: Integer;
  DownloadModels: Boolean;
begin
  DownloadModels := WizardIsTaskSelected('downloadmodels');
  if DownloadModels then
  begin
    // Descargar modelos CT2 y OCR bajo demanda
    // RapidOCR no requiere descarga — modelos ONNX se incluyen en el paquete pip
    Exec('powershell.exe', 
         '-ExecutionPolicy Bypass -Command "& { ' +
         'Write-Host \"Descargando modelos CT2 y EasyOCR...\"; ' +
         '& \"' + ExpandConstant('{app}') + '\env\Scripts\python.exe\" -c \"' +
         'import sys; sys.path.insert(0, ''' + ExpandConstant('{app}') + '''); ' +
         'from translator import _get_ct2_translator; ' +
         't, tk = _get_ct2_translator(''es'', ''en''); ' +
         'print(f\"CT2 es→en listo: {t is not None}\"); ' +
         't2, tk2 = _get_ct2_translator(''en'', ''es''); ' +
         'print(f\"CT2 en→es listo: {t2 is not None}\"); ' +
         'print(f\"RapidOCR se carga bajo demanda (no requiere pre-descarga)\"); ' +
         '}\" }"',
         '', SW_SHOW, ewWaitUntilTerminated, ResultCode);
  end;
end;
