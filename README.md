#N3

###dev notes

the package can be installed via pip:

```bash
pip install -e .
```

if the cuda version of jax is causing issues, you can instead install the cpu version by editing out the `[cuda12]` from jax in the pyproject.toml file.

Basic scripts with argparsing are in the `scripts` directory.
The notebooks in the `notebooks` directory are illustrate what the scripts do.
