from setuptools import setup, find_packages


setup(
    name="casual_paper_learning",
    version="0.1.1",
    py_modules=[
        "c_ddpm", "c_iddpm", "c_cfg",  # DDPM series
        "c_vae"  # Auto Encoder series
    ]
)
