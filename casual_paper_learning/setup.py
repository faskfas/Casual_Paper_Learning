from setuptools import setup, find_packages


setup(
    name="casual_paper_learning",
    version="0.1.2",
    py_modules=[
        "c_ddpm", "c_iddpm", "c_cfg", "c_ldm",  # DDPM series
        "c_vae", "c_vqvae", "c_mae",  # Auto Encoder series
        "c_fm",  # Flow matching series
        "c_kv_cache"
    ]
)
