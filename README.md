# chomik-grad

Minimalny framework tensorowy w Pythonie: lazy graf, autograd, wtyczkowe
kompilatory i mały zestaw narzędzi do uczenia sieci. Runtime wymaga tylko NumPy.

## Architektura

Cały IR ma sześć operacji:

1. `ELEMENTWISE` — m.in. add, mul, exp, log, sqrt i ReLU jako warianty jednej operacji,
2. `REDUCE` — sum/max,
3. `RESHAPE`,
4. `PERMUTE`,
5. `MATMUL`,
6. `GATHER` — indeksowanie pierwszej osi, używane m.in. przez embeddingi.

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

## Wtyczka kompilatora i urządzenia

Kompilator dostaje wyjściowe `LazyNode` i zwraca `CompiledProgram`. Każdy
program udostępnia ten sam kontrakt `run(bindings, synchronize=False)`, więc
można podmieniać wartości liści bez budowania grafu od nowa:

```python
native_outputs = program.run(
    {input_node: program.device.array(new_value)},
    synchronize=False,
)
```

Dla inferencji można dodatkowo wskazać liście, które rzeczywiście zmieniają
się między wywołaniami. Backend może wtedy przechwycić wagi i pozostałe stałe:

```python
program = compile_graph(
    output,
    compiler="mlx",
    dynamic_inputs=(tokens, position, key_cache, value_cache),
)
```

MLX używa tej informacji wyłącznie do specjalizacji programu inferencyjnego.
Zwykłe `compile_graph(...)`, autograd i trening nadal przekazują wszystkie
liście dynamicznie, więc aktualizacja parametrów nie wymaga rekompilacji.

Wtyczka ma parę małych elementów:

```python
from chomikgrad import Compiler, DeviceAdapter, register_compiler

class MyDevice(DeviceAdapter):
    # array, evaluate, synchronize, to_numpy, argmax i dtype
    ...

class MyCompiler(Compiler):
    device = MyDevice()

    def compile(self, outputs):
        # Przejdź inputs każdego LazyNode i obsłuż sześć wartości Op.
        # Zwróć CompiledProgram korzystający z tego samego device.
        ...

register_compiler("my-device", MyCompiler)
```

`DeviceAdapter` oddziela tworzenie natywnych tablic, synchronizację, odczyt do
NumPy, `argmax`, mapowanie dtype i opcjonalne ładowanie safetensors od kompilacji
IR. Dzięki temu przyszły backend CUDA albo Vulkan może użyć tego samego runtime'u
inferencji; nadal musi dostarczyć własne sześć loweringów i ewentualne szybkie
kernele RMSNorm/RoPE/attention. Kompilator można wskazać dla pojedynczej
realizacji:
`tensor.numpy(compiler="my-device")`.

## GPU Apple Silicon przez MLX

Opcjonalna wtyczka `mlx` tłumaczy dokładnie ten sam sześciooperacyjny IR na
`mlx.core` i jawnie wykonuje graf na urządzeniu Metal GPU. Brak MLX albo Metal
kończy się czytelnym błędem — backend nie przechodzi po cichu na CPU.

Parametry i gradienty pozostają jako natywne tablice MLX na GPU pomiędzy
krokami. Strukturalnie identyczne grafy korzystają z cache oraz `mx.compile`, a
SGD oblicza gradienty i aktualizuje parametry przy jednej synchronizacji GPU,
bez kopiowania ich przez NumPy. Transfer do RAM-u
następuje dopiero po jawnym `numpy()`, `item()` albo użyciu kompilatora `cpu`.
MLX implementuje wspólne `CompiledProgram.run(...)` i `DeviceAdapter`, zamiast
wymagać specjalnego API od generatora. Pozwala to utrzymywać jeden program
autoregresywnego decode i wiązać do niego nowe tokeny oraz cache K/V. Po
wybraniu szybkiego loweringu kompilator odcina jego nieużywane przenośne
rozwinięcie; pełny fallback pozostaje dostępny dla CPU i innych backendów.

MLX wymaga Apple Silicon, macOS 14+ i natywnego Pythona 3.10+. Przykładowa
instalacja, gdy systemowy Python jest starszy:

```bash
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/python -m pip install '.[demo,mlx]'
.venv/bin/python examples/train_digits.py --compiler mlx
```

## NVIDIA GPU przez CUDA

Opcjonalna wtyczka `cuda` wykonuje ten sam sześciooperacyjny IR przez CuPy.
Nie przechodzi po cichu na CPU, a parametry i gradienty SGD pozostają w pamięci
GPU pomiędzy krokami. Wariant zależności `ctk` dołącza potrzebne składniki
CUDA, dlatego wystarczy zgodny sterownik NVIDIA:

```bash
python -m pip install '.[benchmark,cuda]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python benchmarks/compare_tinygrad_10_cases.py --device cuda --trials 3
```

Na natywnym Windows wariant `torch-compile` używa oficjalnego backendu
`cudagraphs`, ponieważ wheel PyTorch nie zawiera Tritona. Na systemie z
działającym Tritonem można wybrać pełny Inductor przez
`--torch-compile-backend inductor`.

## Neural Engine Apple Silicon przez Core ML

Opcjonalna wtyczka `coreml` kompiluje ten sam sześciooperacyjny IR do
`ML Program` w FP16 i ładuje go z `CPU_AND_NE`. Apple nie udostępnia
bezpośredniego API obliczeniowego ANE ani trybu `NE_ONLY`: Core ML podejmuje
ostateczną decyzję osobno dla każdej operacji. `CoreMLProgram.compute_plan_summary()`
odczytuje plan skompilowanego modelu, dlatego testy nie traktują samej flagi
`CPU_AND_NE` jako dowodu użycia Neural Engine.

Backend jest przeznaczony wyłącznie do inferencji. Przechwytuje stałe wagi,
rozpoznaje `x @ weight.T` jako Core ML `linear` i dzieli modele przekraczające
1 GiB stałych na segmenty. Bez segmentacji monolityczny TinyLlama 1.1B działał
poprawnie, ale Core ML na M1 przypisał cały graf do CPU. Trzy segmenty zachowują
identyczny wynik i przywracają wykonanie większości grafu na ANE.

```bash
.venv/bin/python -m pip install '.[coreml,llm]'
.venv/bin/python examples/generate_tinyllama.py \
  --compiler coreml --dtype float16 --max-new-tokens 8
```

Ograniczenia są celowo jawne:

- tylko FP16, macOS 15+ i Apple Silicon,
- inference ze stałymi kształtami; brak autogradu i treningu,
- wywołanie `MLModel.predict` jest synchroniczne i przechodzi przez tablice
  NumPy na granicach segmentów,
- pojedynczy token przy `batch=1` nie wykorzystuje ANE tak efektywnie jak GPU.

Sportowe porównanie używa dokładnie tych samych wag, aktywacji, cache K/V,
implementacji Chomika, promptu i tokenów w FP16:

```bash
.venv/bin/python benchmarks/tinyllama_coreml_vs_mlx.py --trials 3
```

Przykładowy wynik na Apple M1 Max (`coremltools 9.0`, `MLX 0.32.1`, 28 tokenów
promptu i osiem tokenów odpowiedzi):

| TinyLlama 1.1B FP16 | Core ML / ANE | MLX / GPU |
|---|---:|---:|
| pierwszy token, wraz z kompilacją | 42,879 s | **0,061 s** |
| pełna odpowiedź, wraz z kompilacją | 71,836 s | **0,140 s** |
| pierwszy token po rozgrzaniu | 0,793 s | **0,021 s** |
| rozgrzany decode | 15,0 tokenu/s | **114,5 tokenu/s** |
| pełna odpowiedź po rozgrzaniu | 1,424 s | **0,083 s** |

Compute Plan wskazał dla prefillu 1377 operacji preferujących Neural Engine i
185 CPU, a dla decode odpowiednio 1235 i 218. Wszystkie osiem identyfikatorów
tokenów pozostało identycznych. Na M1 ANE jest więc działającym backendem
badawczym, ale nie jest konkurencyjny czasowo wobec
Metal GPU dla autoregresywnego LLM `batch=1`.

Dokumentacja Apple: [wybór CPU i Neural Engine](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine),
[Compute Plan](https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html)
i [wykonanie FP16](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html).

## Uruchomienie

```bash
python -m unittest discover -s tests -v
python -m pip install '.[demo]'
python examples/train_digits.py
```

Demo trenuje MLP `64 -> 48 -> 10` na wbudowanym w scikit-learn darmowym
zbiorze cyfr 8×8. Skrypt kończy się błędem, jeśli test accuracy nie osiągnie 90%.

## Transformer

`MATMUL` obsługuje także batch dimensions, dlatego ten sam sześciooperacyjny IR
pokrywa wielogłowe attention bez specjalnej instrukcji. Pakiet zawiera
`LayerNorm`, `MultiHeadSelfAttention` i pre-norm `TransformerEncoderBlock`.

Drugi przykład traktuje osiem wierszy obrazu cyfry jako osiem tokenów. Używa
embeddingu 32, dwóch bloków encodera, czterech głów, MLP 64 i mean poolingu:

```bash
python examples/train_digits_transformer.py --compiler cpu
.venv/bin/python examples/train_digits_transformer.py --compiler mlx
```

## Benchmark względem tinygrad

Główny benchmark uruchamia Chomika, tinygrad i opcjonalne warianty PyTorch w
osobnych procesach, aby ich runtime'y GPU nie wpływały na siebie. Obejmuje osiem
operacji tensorowych, 20 epok MLP i 10 epok transformera. Kontroluje zgodność
wyników oraz accuracy; nie zawiera niestabilnych progów czasowych:

```bash
.venv/bin/python -m pip install '.[benchmark]'
.venv/bin/python benchmarks/compare_tinygrad_10_cases.py --trials 3
.venv/bin/python benchmarks/compare_tinygrad_10_cases.py --json
```

Na NVIDIA/CUDA użyj `--device cuda`; domyślny tryb `metal` zachowuje
dotychczasowe zachowanie na Apple Silicon.

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

Na NVIDIA GeForce RTX 5070 Ti (`CuPy 14.2.0`, `tinygrad 0.14.0`,
`PyTorch 2.13.0+cu130`, Python 3.14.2) pełny przebieg
`--device cuda --trials 3` dał następujące mediany:

| przypadek | Chomik CUDA | tinygrad CUDA | PyTorch eager | PyTorch compile/CUDA Graphs |
|---|---:|---:|---:|---:|
| elementwise, 1M | 1,523 ms | 2,911 ms | 1,113 ms | **1,002 ms** |
| reduce sum, 4M | 1,261 ms | 1,926 ms | **0,927 ms** | 1,000 ms |
| softmax, 1024×1024 | 1,257 ms | 2,663 ms | 0,791 ms | **0,756 ms** |
| matmul, 64×64 | 0,125 ms | 2,259 ms | **0,112 ms** | 0,177 ms |
| matmul, 256×256 | 0,227 ms | 2,384 ms | **0,187 ms** | 0,318 ms |
| matmul, 1024×1024 | 1,497 ms | 3,985 ms | **0,995 ms** | 1,096 ms |
| matmul, 2048×2048 | 6,202 ms | 9,956 ms | **4,045 ms** | 4,269 ms |
| batched matmul, 16×4×64 | 0,614 ms | 3,046 ms | **0,362 ms** | 0,445 ms |
| trening MLP, 20 epok | 0,498 s | 1,239 s | **0,262 s** | 0,363 s |
| trening transformera, 10 epok | 2,321 s | 1,727 s | **0,987 s** | 1,207 s |

PyTorch eager wygrał osiem przypadków, a CUDA Graphs dwa. Kontrole fingerprintów
i accuracy przeszły dla wszystkich frameworków. Mikrobenchmarki obejmują
transfer wejścia z NumPy na GPU oraz odczyt wyniku z powrotem do NumPy.

### Inference rdzenia LLM około 1B

Drugi benchmark buduje decoder-only transformer core bez embeddingu,
tokenizera i LM headu. Domyślna konfiguracja ma 20 bloków, szerokość 2048,
16 głów, FFN 8192, sekwencję 32 i dokładnie 1 007 169 536 parametrów:

```bash
.venv/bin/python benchmarks/llm_1b_inference.py
.venv/bin/python benchmarks/llm_1b_inference.py --json
```

Na NVIDIA/CUDA uruchom ten sam model przez
`python benchmarks/llm_1b_inference.py --device cuda`.

Na M1 Max, FP32 i `batch=1` mediana dziesięciu rozgrzanych forwardów wyniosła
32,83 ms dla Chomika oraz 40,38 ms dla tinygrad. Pierwszy forward trwał
odpowiednio 0,62 s i 2,63 s. Jest to prefill syntetycznych hidden states, a nie
autoregresywne generowanie z KV cache.

Na RTX 5070 Ti ten sam benchmark FP32 (`--device cuda --warm-runs 30`) dał
zgodne fingerprinty wyjścia i następujące wyniki:

| metryka | Chomik CUDA | tinygrad CUDA | PyTorch eager | PyTorch compile/CUDA Graphs |
|---|---:|---:|---:|---:|
| inicjalizacja modelu | 6,644 s | **6,052 s** | 6,516 s | 6,480 s |
| pierwszy forward | 0,542 s | 3,699 s | **0,156 s** | 3,372 s |
| mediana 30 rozgrzanych forwardów | **9,69 ms** | 61,61 ms | 10,79 ms | 12,26 ms |
| peak RAM procesu | 4583,1 MiB | 7917,4 MiB | **1048,3 MiB** | 1219,2 MiB |
| pamięć GPU raportowana przez runtime | 3990,3 MiB | **3844,4 MiB** | 3877,1 MiB | 3877,1 MiB |

Chomik wygrał rozgrzany forward: był 1,11× szybszy od PyTorch eager, 1,26× od
CUDA Graphs i 6,36× od tinygrad. Backend kompiluje graf raz, spłaszcza projekcje
liniowe do 2D GEMM i używa jednego kernela FP32 dla LayerNorm. Model ma
1 007 169 536 losowych parametrów FP32 (3,752 GiB samych wag); benchmark nie
zawiera embeddingu, tokenizera, LM headu, KV cache ani generowania tokenów.

## Pełne generowanie realnym modelem 1.1B

Przykład `generate_tinyllama.py` uruchamia prawdziwy
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`: pobiera przypiętą rewizję wag, renderuje
chat template, tokenizuje prompt, wykonuje embedding, 22 bloki Llama, LM head,
greedy decoding albo sampling i dekoduje odpowiedź. Wagi mają 1 100 048 384
parametry BF16 i pozostają w natywnej pamięci MLX.

```bash
.venv/bin/python -m pip install '.[llm,mlx]'
.venv/bin/python examples/generate_tinyllama.py --compiler mlx
.venv/bin/python examples/generate_tinyllama.py \
  --prompt 'Explain lazy execution in one sentence.' \
  --temperature 0.7 --top-k 50 --max-new-tokens 32
```

Pierwsze uruchomienie pobiera około 2,2 GB do standardowego cache Hugging Face;
wagi nie trafiają do repozytorium. Rewizja modelu jest przypięta do
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`, aby przykład był powtarzalny.

Prefill tworzy cache K/V, a każdy kolejny krok aktualizuje go maską. Embedding,
RoPE, grouped-query attention, RMSNorm, SiLU, cache K/V i wybór ostatniej pozycji
nadal składają się wyłącznie z sześciu instrukcji IR opisanych wyżej. Backend
MLX może rozpoznawać przenośne podgrafy RMSNorm, RoPE i attention i opuszczać
je do szybszych kerneli; inne backendy wykonują ich zwykłe rozwinięcia.
`TinyLlamaRuntime` materializuje wagi tylko raz i cache'uje programy prefill
według kształtu oraz decode według długości cache. Dane tokenów i K/V są nadal
wiązane osobno dla każdego żądania.

Dla domyślnego promptu model generuje:

```text
The capital of France is Paris.
```

Aktualne porównanie z MLX-LM uruchamia oba runtime'y w osobnych procesach,
materializuje wagi przed pomiarem i sprawdza identyczność tokenów:

```bash
.venv/bin/python -m pip install '.[benchmark,llm]'
.venv/bin/python benchmarks/tinyllama_vs_mlx_lm.py --trials 9
```

Przykładowy wynik na Apple M1 Max (`MLX 0.32.1`, `MLX-LM 0.31.3`, 28 tokenów
promptu i osiem tokenów odpowiedzi):

| TinyLlama 1.1B BF16 | Chomik | MLX-LM |
|---|---:|---:|
| pierwszy token, zimny graf | **0,077 s** | 0,121 s |
| pełna odpowiedź, zimny graf | **0,157 s** | 0,201 s |
| pierwszy token po rozgrzaniu | **0,024 s** | 0,041 s |
| rozgrzany decode | 118,9 tokenu/s | **130,4 tokenu/s** |
| pełna odpowiedź po rozgrzaniu | **0,084 s** | 0,094 s |
| szczyt pamięci GPU | 2,118 GiB | 2,120 GiB |

Chomik osiąga około 91% przepustowości decode natywnego MLX-LM, a dzięki cache
całych programów ma krótszy TTFT dla powtarzanego kształtu. Wszystkie osiem
identyfikatorów tokenów jest identyczne w obu implementacjach.

### Eksperyment TinyLlama względem tinygrad

Porównanie wykonano na Apple M1 Max, macOS 27.0 i Pythonie 3.11.14. Chomik
używał MLX 0.32.1. tinygrad pochodził z oficjalnego taga
[`v0.14.0`](https://github.com/tinygrad/tinygrad/tree/v0.14.0), commit
`6f87158`; użyto jego implementacji `tinygrad.llm.model.Transformer`, `TinyJit`
i cache K/V. Tag źródłowy był konieczny, ponieważ koło 0.14.0 z PyPI nie
zawierało w testowanym środowisku katalogu `tinygrad.llm.kernels`.

Oba frameworki dostały te same wagi, aktywacje i cache K/V w BF16, przypiętą
rewizję TinyLlama, ten sam 28-tokenowy chat prompt, `batch=1`, kontekst 36,
czysty greedy argmax i limit ośmiu nowych tokenów. Ponieważ oficjalny model
tinygrad konwertuje embedding do FP32 i cache do FP16, w tymczasowej kopii taga
te dwa casty ustawiono na BF16; Gumbel sampling zastąpiono równoważnym dla
temperatury zero bezpośrednim `argmax`. Wagi były już w lokalnym cache, a ich
ładowanie nie wchodziło do pomiaru. Dla obu frameworków wykonano po sześć
generacji. Wynik warm jest medianą ostatnich czterech prób; cache promptu był
resetowany, więc oba frameworki ponownie wykonywały prefill.

| TinyLlama 1.1B, Metal | Chomik | tinygrad 0.14.0 |
|---|---:|---:|
| pierwszy token, zimny JIT | **0,078 s** | 2,585 s |
| pełne osiem tokenów, zimny JIT | **0,174 s** | 4,335 s |
| pierwszy token po capture | **0,024 s** | 0,389 s |
| pełne osiem tokenów po capture | **0,085 s** | 0,550 s |
| rozgrzany decode | **117,9 tokenu/s** | 43,6 tokenu/s |
| pamięć urządzenia | 2,118 GiB | **2,052 GiB** |

W stałym decode Chomik był około 2,7 raza szybszy, a cała krótka odpowiedź po
rozgrzaniu zajmowała około 6,5 raza mniej czasu. Największa różnica wystąpiła
przy zimnym JIT: tinygrad potrzebował dwóch wolnych przebiegów na capture i
kompilację. tinygrad zużył około 3% mniej pamięci urządzenia.

W obu przypadkach powstały identyczne identyfikatory tokenów:

```text
1576, 7483, 310, 3444, 338, 3681, 29889, 2
The capital of France is Paris.
```

Porównanie wyrównuje dtype przechowywanych tensorów. Wewnętrzna precyzja
akumulacji pozostaje decyzją kernela każdego frameworka. Zweryfikowano
identyczne tokeny wynikowe, nie bitową identyczność wszystkich logits. Dla
dłuższych kontekstów i odpowiedzi proporcje mogą się zmienić.

### Eksperymentalne speculative decoding

`LlamaDecoderBlock` weryfikuje kilka pozycji w jednym grafie, korzystając z
tych samych sześciu operacji IR. Mechanizm nie należy do backendu MLX, więc
przyszły backend CUDA albo Vulkan może skompilować dokładnie ten sam blok.
Greedy runtime potrafi opcjonalnie użyć przypiętego
`Felladrin/Llama-68M-Chat-v1` jako draftu. Przy ładowaniu sprawdza pełną mapę
32 000 tokenów, a model docelowy akceptuje kandydatów tylko do pierwszej
różnicy:

```bash
PYTHONPATH=. .venv/bin/python examples/generate_tinyllama.py \
  --speculative-tokens 6 --temperature 0
```

Opcja jest domyślnie wyłączona. W BF16 blokowy target może użyć innego kernela
Metal niż pojedynczy decode, a więc zmienić kolejność akumulacji. To nie zmienia
dtype ani matematycznego algorytmu, ale po kilku zaakceptowanych tokenach
zaokrąglenia cache K/V mogą prowadzić do innego `argmax`. Z tego powodu ten
eksperyment nie spełnia jeszcze rygorystycznego wymagania identycznych tokenów.

Powtarzalny benchmark obejmuje dziesięć promptów, wykonuje warianty w osobnych
procesach i raportuje zarówno czas, jak i identyczność całej sekwencji:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/tinyllama_speculative.py \
  --trials 3 --speculative-tokens 6
```

Na Apple M1 Max osiem z dziesięciu przypadków zachowało identyczne tokeny, a
tylko trzy były szybsze. Speedup wynosił od 1,00 do 1,06 raza w wygranych
przypadkach, natomiast najgorszy przypadek był około 3,6 raza wolniejszy.
Wniosek: przenośna infrastruktura działa, ale niezależny draft 68M i obecne
kernele blokowe nie są jeszcze optymalizacją nadającą się do domyślnej ścieżki.
Do bezpiecznego włączenia potrzeba kernela weryfikującego o zgodnej kolejności
akumulacji oraz draftu wytrenowanego pod konkretny target.
