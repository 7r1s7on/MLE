from setuptools import setup, find_packages

setup(
    name="calorie_predictor",
    version="0.1.0",
    packages=find_packages(include=['calorie_predictor', 'calorie_predictor.*']),
    install_requires=[
        "pandas",
        "scikit-learn",
        "flask",
        "gunicorn",
        "gdown",
        "joblib"
    ],
    entry_points={
        'console_scripts': [
            'run_batch_pipeline=calorie_predictor.batch_pipeline:main_batch_pipeline',
            'run_api_server=calorie_predictor.api:app.run', # basic run, gunicorn preferred
        ],
    },
    python_requires='>=3.8',
    description="A simple calorie predictor model deployment assignment.",
    author="Bakhrom",
    author_email="areszeus@gmail.com",
)
