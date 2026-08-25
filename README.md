# chomik-grad

Minimalny framework tensorowy w Pythonie: lazy graf, autograd, wtyczkowe
kompilatory i mały zestaw narzędzi do uczenia sieci. Runtime wymaga tylko NumPy.

## Architektura

Cały IR ma pięć operacji:

1. `ELEMENTWISE` — m.in. add, mul, exp, log i ReLU jako warianty jednej operacji,
2. `REDUCE` — sum/max,
3. `RESHAPE`,
4. `PERMUTE`,
5. `MATMUL`.

`softmax` i `log_softmax` są stabilnymi numerycznie kompozycjami tych
prymitywów. Nie dodają kolejnej instrukcji ani specjalnego przypadku w backendach.

Operacje na `Tensor` wyłącznie budują graf. `numpy()`, `item()`, `realize()` albo
`SGD.step()` przekazują cały potrzebny graf do wybranego kompilatora. Domyślna
wtyczka `cpu` generuje prostoliniową funkcję Pythona z wywołaniami NumPy, a
następnie ją wykonuje.

```python
import numpy as np
from chomikgrad import Linear, SGD, Tensor, cross_entropy

model = Linear(4, 3)
optimizer = SGD(model.parameters(), lr=0.1)
x = Tensor(np.random.randn(8, 4).astype(np.float32))
y = np.array([0, 1, 2, 0, 1, 2, 0, 1])

loss = cross_entropy(model(x), y)  # nadal nic nie zostało policzone
loss.backward()                    # powstaje lazy graf gradientów
optimizer.step()                   # kompilacja i wykonanie na CPU
```

## Wtyczka kompilatora

Wtyczka implementuje jedną metodę. Dostaje wyjściowe `LazyNode` i zwraca
wywoływalny `CompiledProgram`:

```python
from chomikgrad import Compiler, register_compiler, set_default_compiler

class MyCompiler(Compiler):
    def compile(self, outputs):
        # Przejdź inputs każdego LazyNode i obsłuż pięć wartości Op.
        # Zwrócony program ma zwracać tuple[np.ndarray, ...].
        ...

register_compiler("my-device", MyCompiler)
set_default_compiler("my-device")
```

Kompilator można też wskazać dla pojedynczej realizacji:
`tensor.numpy(compiler="my-device")`.

## GPU Apple Silicon przez MLX

Opcjonalna wtyczka `mlx` tłumaczy dokładnie ten sam pięciooperacyjny IR na
`mlx.core` i jawnie wykonuje graf na urządzeniu Metal GPU. Brak MLX albo Metal
kończy się czytelnym błędem — backend nie przechodzi po cichu na CPU.

Parametry i gradienty pozostają jako natywne tablice MLX na GPU pomiędzy
krokami. Strukturalnie identyczne grafy korzystają z cache oraz `mx.compile`, a
SGD aktualizuje parametry bez kopiowania ich przez NumPy. Transfer do RAM-u
następuje dopiero po jawnym `numpy()`, `item()` albo użyciu kompilatora `cpu`.

MLX wymaga Apple Silicon, macOS 14+ i natywnego Pythona 3.10+. Przykładowa
instalacja, gdy systemowy Python jest starszy:

```bash
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/python -m pip install '.[demo,mlx]'
.venv/bin/python examples/train_digits.py --compiler mlx
```

## Uruchomienie

```bash
python -m unittest discover -s tests -v
python -m pip install '.[demo]'
python examples/train_digits.py
```

Demo trenuje MLP `64 -> 48 -> 10` na wbudowanym w scikit-learn darmowym
zbiorze cyfr 8×8. Skrypt kończy się błędem, jeśli test accuracy nie osiągnie 90%.

## Transformer

`MATMUL` obsługuje także batch dimensions, dlatego ten sam pięciooperacyjny IR
pokrywa wielogłowe attention bez specjalnej instrukcji. Pakiet zawiera
`LayerNorm`, `MultiHeadSelfAttention` i pre-norm `TransformerEncoderBlock`.

Drugi przykład traktuje osiem wierszy obrazu cyfry jako osiem tokenów. Używa
embeddingu 32, dwóch bloków encodera, czterech głów, MLP 64 i mean poolingu:

```bash
python examples/train_digits_transformer.py --compiler cpu
.venv/bin/python examples/train_digits_transformer.py --compiler mlx
```

## Benchmark transformera względem tinygrad

Skrypt buduje w obu frameworkach tę samą architekturę i zaczyna od tych samych
wag. Mierzy dziesięć epok treningu bez synchronizacji wyniku loss po każdym
batchu, a następnie sprawdza accuracy:

```bash
.venv/bin/python -m pip install tinygrad
DEV=METAL PYTHONPATH=. .venv/bin/python \
  benchmarks/transformer_vs_tinygrad.py \
  --frameworks chomik-cpu chomik-mlx tinygrad --trials 3
```

Przykładowy wynik na Apple M1 Max (`tinygrad 0.14.0`, `mlx 0.32.1`):

| backend | mediana 10 epok | test accuracy |
|---|---:|---:|
| chomik CPU / NumPy | 2,208 s | 77,78% |
| chomik GPU / MLX Metal | 1,268 s | 77,78% |
| tinygrad / Metal | 1,444 s | 77,78% |

To mały model, więc wynik mierzy również narzut kompilacji i Pythona. Na innych
wersjach bibliotek oraz układach Apple proporcje mogą być inne.
