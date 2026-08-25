# chomik-grad

Minimalny framework tensorowy w Pythonie: lazy graf, autograd, wtyczkowe
kompilatory i mały zestaw narzędzi do uczenia sieci. Runtime wymaga tylko NumPy.

## Architektura

Cały IR ma pięć operacji:

1. `ELEMENTWISE` — m.in. add, mul, exp, log, sqrt i ReLU jako warianty jednej operacji,
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

Domyślnie `Tensor(np_array)` posiada własną kopię danych. Dla świeżych lub
niemutowanych tablic można jawnie użyć `Tensor(np_array, copy=False)`, aby
pominąć dodatkową kopię w RAM. Backend MLX nie cache'uje wtedy wejścia i
zauważa późniejsze zmiany źródłowej tablicy.

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
SGD oblicza gradienty i aktualizuje parametry przy jednej synchronizacji GPU,
bez kopiowania ich przez NumPy. Transfer do RAM-u
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

## Benchmark względem tinygrad

Główny benchmark uruchamia Chomika i tinygrad w osobnych procesach, aby ich
runtime'y Metal nie wpływały na siebie. Obejmuje osiem operacji tensorowych,
20 epok MLP i 10 epok transformera. Kontroluje zgodność wyników oraz accuracy;
nie zawiera niestabilnych progów czasowych:

```bash
.venv/bin/python -m pip install '.[benchmark]'
.venv/bin/python benchmarks/compare_tinygrad_10_cases.py --trials 3
.venv/bin/python benchmarks/compare_tinygrad_10_cases.py --json
```

Skrypt `benchmarks/transformer_vs_tinygrad.py` pozostaje krótszym benchmarkiem
samego transformera.

Przykładowy wynik głównego benchmarku na Apple M1 Max (`tinygrad 0.14.0`,
`mlx 0.32.1`):

| przypadek | Chomik | tinygrad |
|---|---:|---:|
| elementwise, 1M | **0,75 ms** | 1,82 ms |
| reduce sum, 4M | **0,74 ms** | 1,60 ms |
| softmax, 1024×1024 | **0,65 ms** | 1,95 ms |
| matmul, 64×64 | **0,29 ms** | 2,41 ms |
| matmul, 256×256 | **0,33 ms** | 2,46 ms |
| matmul, 1024×1024 | **1,57 ms** | 3,24 ms |
| matmul, 2048×2048 | **5,35 ms** | 6,59 ms |
| batched matmul, 16×4×64 | **0,40 ms** | 2,67 ms |
| trening MLP, 20 epok | **0,37 s** | 1,06 s |
| trening transformera, 10 epok | **1,46 s** | 1,63 s |

To mały model, więc wynik mierzy również narzut kompilacji i Pythona. Na innych
wersjach bibliotek oraz układach Apple proporcje mogą być inne.

### Inference rdzenia LLM około 1B

Drugi benchmark buduje decoder-only transformer core bez embeddingu,
tokenizera i LM headu. Domyślna konfiguracja ma 20 bloków, szerokość 2048,
16 głów, FFN 8192, sekwencję 32 i dokładnie 1 007 169 536 parametrów:

```bash
.venv/bin/python benchmarks/llm_1b_inference.py
.venv/bin/python benchmarks/llm_1b_inference.py --json
```

Na M1 Max, FP32 i `batch=1` mediana dziesięciu rozgrzanych forwardów wyniosła
32,83 ms dla Chomika oraz 40,38 ms dla tinygrad. Pierwszy forward trwał
odpowiednio 0,62 s i 2,63 s. Jest to prefill syntetycznych hidden states, a nie
autoregresywne generowanie z KV cache.

## Pełne generowanie realnym modelem 1.1B

Przykład `generate_tinyllama.py` uruchamia prawdziwy
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`: pobiera przypiętą rewizję wag, renderuje
chat template, tokenizuje prompt, wykonuje embedding, 22 bloki Llama, LM head,
greedy decoding albo sampling i dekoduje odpowiedź. Wagi mają 1 100 048 384
parametry BF16 i pozostają w natywnej pamięci MLX.

```bash
.venv/bin/python -m pip install '.[llm,mlx]'
.venv/bin/python examples/generate_tinyllama.py
.venv/bin/python examples/generate_tinyllama.py \
  --prompt 'Explain lazy execution in one sentence.' \
  --temperature 0.7 --top-k 50 --max-new-tokens 32
```

Pierwsze uruchomienie pobiera około 2,2 GB do standardowego cache Hugging Face;
wagi nie trafiają do repozytorium. Rewizja modelu jest przypięta do
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`, aby przykład był powtarzalny.

Prefill tworzy cache K/V, a każdy kolejny krok aktualizuje go maską. Embedding,
RoPE, grouped-query attention, RMSNorm, SiLU, cache K/V i wybór ostatniej pozycji
nadal składają się wyłącznie z pięciu instrukcji IR opisanych wyżej.

Dla domyślnego promptu model generuje:

```text
The capital of France is Paris.
```

Na Apple M1 Max mediana pięciu osobnych uruchomień wyniosła 0,19 s do pierwszego
tokenu i 46,2 tokenu/s dla rozgrzanych kroków decode; szczyt użycia GPU wyniósł
około 2,16 GiB. Osiem identyfikatorów wygenerowanych tokenów jest identycznych
z niezależnym wynikiem MLX-LM 0.31.3. MLX-LM jest wyspecjalizowanym runtime'em i
osiągał w tym samym teście około 144 tokenów/s po rozgrzaniu; Chomik pozostaje
celowo małym, czytelnym frameworkiem ogólnego przeznaczenia.
