# TFM Redes neuronales adversarias en seguridad informática

```
.
├── datasets/
│   ├── CASIA-Multi-Spectral-PalmprintV1/
│   ├── CASIA-PalmprintV1/
|   └── SOCOFing/
│  
├── Project/
│   ├── Latex/
│   ├── LatexSlide/
│   ├── UML/
│   ├── manual/
│   └── research/
│  
├── docs/
│   └── ...
│  
├── manual/
│   └── ...
│  
├── models/
│   └── ...
│  
├── scripts/
│   └── ...
│  
├── README.md
├── Makefile
├── LICENSE
├── Containerfile
├── MANIFEST.in
├── mkdocs.yml
├── coverage.xml
├── requirements-test.txt
├── requirements.txt
├── setup.py
├── tests/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_base.py
│   └── test_resizer.py
└── rna_si/
    ├── __pycache__/
    ├── VERSION
    ├── __init__.py
    ├── __main__.py
    ├── base.py
    ├── cli.py
    ├── resizer.py
    └── utils.py
```

## Instalar Pytorch y CUDA

```bash

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

```


## Generar virtualenv

```bash
make virtualenv
make install
```

## Generar documentación

```bash
make docs-serve
make docs
```

## Makefile

```bash
# Source project
make virtualenv
make install

# Info package
make help
make show

# DEV
make fmt
make lint
make watch
make clean

# Run test with pytest and unittest
make test

# Release
make release

make docs
make docs-serve

make manual-build
make manual-install
make manual-read
```

## Instalar dependencias

```bash
# Before make
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-test.txt
```

## Extra

<details>
<summary> Desactivar la generación de `__pycache__` </summary>

```bash
export PYTHONDONTWRITEBYTECODE=1
```

</details>

---

<details>

<summary>Run pytest</summary>

```bash
pytest tests/pytest/
pytest tests/pytest/test_base.py
```

</details>

---

<details>

<summary>Run unittest</summary>

```bash
python -m unittest tests/utests/__main__.py

python -m unittest tests/utests/mock.py
python -m unittest tests/utests/module1.py
```

</details>

<details>

<summary>Coverage</summary>

```bash
# With parallel = true in .coveragerc
coverage run -m pytest -v tests/pytest/*
coverage run -m unittest -v tests/utests/*
coverage combine
# With parallel = false in .coveragerc
coverage run -a -m pytest -v tests/pytest/*
coverage run -a -m unittest -v tests/utests/*

# Generate report
coverage xml
coverage json
coverage html

python -m coverage run -m pytest -v tests/pytest/test_base.py
python -m coverage run -m unittest -v tests/utests/test_coverage.py
```

</details>

### Other

```bash

# For seaborn
sudo apt-get install msttcorefonts -q
```
