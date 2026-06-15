from PyInstaller.utils.hooks import collect_data_files


datas = [
    ("PIC_CON", "PIC_CON"),
    ("PIC_MANU", "PIC_MANU"),
    ("PIC_RUN", "PIC_RUN"),
    ("PIC_SURFACE", "PIC_SURFACE"),
    ("PIC_VF", "PIC_VF"),
]
binaries = []
hiddenimports = [
    "meshio.gmsh",
    "meshio.gmsh._gmsh22",
    "meshio.gmsh._gmsh40",
    "meshio.gmsh._gmsh41",
    "shapely.geometry",
    "shapely.ops",
    "shapely.prepared",
]

datas += collect_data_files(
    "pyvista",
    excludes=["examples/**", "jupyter/**", "trame/**"],
)
datas += collect_data_files("pyvistaqt")
datas += collect_data_files("shapely", excludes=["tests/**"])


a = Analysis(
    ["main_run.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tensorflow",
        "torch",
        "torchvision",
        "jax",
        "jaxlib",
        "cupy",
        "cupyx",
        "cupy_backends",
        "cuda",
        "numba.cuda",
        "nvidia",
        "dask",
        "distributed",
        "pandas",
        "bokeh",
        "imageio",
        "imageio_ffmpeg",
        "sklearn",
        "skimage",
        "tables",
        "h5py",
        "datasets",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "cv2",
        "pyarrow",
        "tensorstore",
        "kaleido",
        "plotly",
        "trame",
        "IPython",
        "notebook",
        "pytest",
    ],
    noarchive=False,
    optimize=0,
)

# Some scientific-package hooks discover an installed CUDA toolkit even when
# the application only uses Numba's CPU backend. Keep the release CPU-only.
cuda_binary_names = (
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvjitlink",
    "nvrtc",
)
a.binaries = [
    entry
    for entry in a.binaries
    if not any(name in entry[0].lower() for name in cuda_binary_names)
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MES-Acoustic",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="PIC_MANU/MES.png",
    contents_directory=".",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MES-Acoustic",
)
