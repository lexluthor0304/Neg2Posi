# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

project_root = Path(__file__).resolve().parents[1]


def _relocate_binaries(entries, destination):
    relocated = []
    for src, _ in entries:
        basename = os.path.basename(src)
        relocated.append((src, os.path.join(destination, basename)))
    return relocated


def _relocate_datas(entries, destination):
    relocated = []
    for src, dest in entries:
        relocated.append((src, os.path.join(destination, dest)))
    return relocated


datas = []
binaries = []
hiddenimports = []

lensfun_binaries = collect_dynamic_libs("lensfunpy")
lensfun_datas = collect_data_files("lensfunpy")

binaries += _relocate_binaries(
    lensfun_binaries, os.path.join("Contents", "Frameworks")
)
datas += _relocate_datas(lensfun_datas, os.path.join("Contents", "Resources"))


a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="Neg2Posi",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name="Neg2Posi.app",
    icon=None,
    bundle_identifier=None,
)
