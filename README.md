TFM
===


```ansi
.
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
│   ├── __init__.py
│   ├── __pycache__
│   ├── conftest.py
│   ├── test_base.py
│   └── test_resizer.py
└── tfm_sai/
    ├── VERSION
    ├── __init__.py
    ├── __main__.py
    ├── __pycache__
    ├── base.py
    ├── cli.py
    ├── resizer.py
    └── utils.py
```


```bash
make clean
make docs
make docs-serve
make fmt
make help
make init
make install
make lint
make manual
make manual-read
make release
make show
make switch-to-poetry
make test
make virtualenv
make watch
```