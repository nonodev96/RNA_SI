TFM
===


```
.
├── datasets/
│   ├── CASIA-Multi-Spectral-PalmprintV1/
│   └── CASIA-PalmprintV1/
├── Project/
│   ├── Latex/
│   ├── LatexSlide/
│   ├── UML/
│   ├── manual/
│   └── research/
├── docs/
│   └── ...
├── examples/
│   └── ...
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
└── tfm_sai/
│   ├── __pycache__/
    ├── VERSION
    ├── __init__.py
    ├── __main__.py
    ├── base.py
    ├── cli.py
    ├── resizer.py
    └── utils.py
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
# Info package
make help
make show

# Source project
make virtualenv
make install

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

# make switch-to-poetry # TODO
# make init             # REMOVED
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
pytest tests/
pytest tests/test_base.py
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


