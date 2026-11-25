from setuptools import setup, find_packages

setup(
    name="pongrl-pro",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Dependencies are listed in requirements.txt
    ],
    author="PongRL Team",
    description="Intelligent Game Agent Using Deep Reinforcement Learning",
)
