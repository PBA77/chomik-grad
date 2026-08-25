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

CUDA i OpenCL SGD mogą opcjonalnie aktualizować istniejący storage parametrów:

```python
optimizer = SGD(model.parameters(), lr=0.1, inplace=True)
optimizer.step(compiler="cuda")
```

Domyślne `inplace=False` zachowuje snapshot wag używany przez wcześniej
zbudowane lazy grafy. Tryb in-place zmniejsza peak pamięci, ale takie stare
grafy widzą już zaktualizowane wagi.

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
IR. Dzięki temu kolejny backend, np. Vulkan, może użyć tego samego runtime'u
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
python benchmarks/compare_tinygrad_10_cases.py --device cuda --trials 5 --chomik-jit
```

Opcjonalny `compile_train_step` przechwytuje stałokształtny forward i backward
tylko raz. CUDA scala łańcuchy elementwise, softmax/log-softmax oraz aktualizacje
SGD w grupy do 16 parametrów. Jest to replay wygenerowanego programu Chomika,
nie CUDA Graph: CuPy 14.2 nie pozwala obecnie przechwycić używanego tutaj
`cp.matmul`, ponieważ ustawianie streamu cuBLAS podczas capture jest niewspierane.

Na natywnym Windows wariant `torch-compile` używa oficjalnego backendu
`cudagraphs`, ponieważ wheel PyTorch nie zawiera Tritona. Na systemie z
działającym Tritonem można wybrać pełny Inductor przez
`--torch-compile-backend inductor`.

## GPU przez OpenCL

Opcjonalna wtyczka `opencl` wykonuje pełny IR FP32 przez PyOpenCL. Operacje
elementwise, redukcje, reshape, permute, gather i SGD korzystają z własnych
kerneli, a zwykły, strided-batched i offset-batched `MATMUL` z CLBlast. Forward
LayerNorm, softmax i log-softmax oraz ich backwardy mają scalone kernele. Łańcuchy
operacji elementwise o tym samym kształcie i jednym konsumencie są scalane bez
duplikowania obliczeń. `reshape` i `permute` używają lekkich widoków runtime,
a aktualizacje SGD są łączone w grupy do 16 parametrów na kernel. Parametry oraz
gradienty pozostają na GPU między krokami i nie ma cichego fallbacku na CPU.

PyOpenCL instaluje się z extra projektu, natomiast współdzieloną bibliotekę
CLBlast 1.7 trzeba zainstalować osobno. Jej ścieżkę można podać bezpośrednio:

```powershell
python -m pip install '.[benchmark,opencl]'
$env:CLBLAST_PATH='C:\ścieżka\do\clblast.dll'
python benchmarks\compare_tinygrad_10_cases.py --device opencl --trials 3
```

Dla powtarzanego kroku treningowego o stałym kształcie batcha można opcjonalnie
przechwycić forward, backward i SGD tylko raz:

```python
from chomikgrad import SGD, Tensor, compile_train_step

optimizer = SGD(model.parameters(), lr=0.03)

def loss_function(inputs, targets):
    logits = model(inputs)
    return -(logits.log_softmax(axis=1) * targets).sum() / inputs.shape[0]

step = compile_train_step(
    loss_function,
    optimizer,
    Tensor.zeros((64, 8, 8)),
    Tensor.zeros((64, 10)),
    compiler="opencl",
)
step(batch_inputs, batch_targets_one_hot)
```

Wejścia mogą zmieniać wartości, lecz muszą zachować kształt i dtype przykładów.
Dla krótszego ostatniego batcha trzeba utworzyć drugi krok. Domyślnie krok nie
realizuje wartości loss; `return_loss=True` włącza jej zwracanie do monitoringu.

Na Linuksie `CLBLAST_PATH` może wskazywać `libclblast.so`. Backend sprawdza też
systemowy loader oraz `Library/bin/clblast.dll` albo `lib/libclblast.so`
aktywnego virtualenv. Benchmark ustawia dla tinygrad urządzenie `CL`, więc obie
implementacje używają OpenCL na tym samym GPU.

Backend celowo odrzuca dtype inne niż FP32, z wyjątkiem int32/int64 dla indeksów
`GATHER`. Sterownik NVIDIA użyty w pomiarze udostępniał OpenCL 3.0, ale tylko
OpenCL C 1.2 i bez `cl_khr_fp16`, dlatego tryb FP16 nie jest oferowany pozornie
przez konwersję do FP32.

Na GeForce RTX 5070 Ti (`PyOpenCL 2026.1.3`, `CLBlast 1.7.0`,
`tinygrad 0.14.0`, Python 3.14.2) pojedynczy pełny przebieg z domyślną liczbą
powtórzeń mikrobenchmarków dał:

| przypadek | Chomik OpenCL | tinygrad CL |
|---|---:|---:|
| elementwise, 1M | **1,964 ms** | 2,886 ms |
| reduce sum, 4M | **1,327 ms** | 1,799 ms |
| softmax, 1024×1024 | **1,438 ms** | 2,874 ms |
| matmul, 64×64 | **0,277 ms** | 2,342 ms |
| matmul, 256×256 | **0,344 ms** | 2,393 ms |
| matmul, 1024×1024 | **2,370 ms** | 3,904 ms |
| matmul, 2048×2048 | **7,864 ms** | 9,556 ms |
| batched matmul, 16×4×64 | **0,734 ms** | 2,894 ms |
| trening MLP, 20 epok | 3,092 s | **2,262 s** |
| trening transformera, 10 epok | 7,776 s | **6,695 s** |

Scalone backwardy, pula buforów, przygotowywanie invokerów tylko raz,
offset-batched GEMM dla częściowego broadcastu i cache planów offsetów skróciły
trening transformera z 331,278 s do 7,776 s, czyli 42,6 raza. Chomik wygrał
wszystkie osiem mikrobenchmarków; tinygrad pozostał o 37% szybszy w treningu MLP
i o 16% w treningu transformera. Na tym samym GPU backend CUDA nadal pozostaje
właściwym wyborem dla maksymalnej wydajności.

Osobna seria pięciu pełnych prób treningowych porównała capture po fuzji z
TinyJit. `--repeat-scale 0.01` skracał wyłącznie mikrobenchmarki i nie zmieniał
liczby epok:

| wariant | MLP pierwszy | MLP mediana | Transformer pierwszy | Transformer mediana |
|---|---:|---:|---:|---:|
| Chomik `compile_train_step` | **1,327 s** | **0,626 s** | **4,250 s** | **2,237 s** |
| tinygrad `TinyJit` | 3,336 s | 2,217 s | 8,957 s | 2,816 s |

W porównaniu z capture sprzed fuzji mediana Chomika spadła z 1,023 s do 0,626 s
dla MLP i z 3,965 s do 2,237 s dla transformera. Fuzje softmax i elementwise
usunęły 28 operacji wykonawczych z grafu transformera, lekkie widoki ograniczyły
narzut Pythona, a zgrupowane SGD zmniejszyło liczbę kerneli aktualizacji 39 wag
z 39 do 3 na krok. Po rozgrzaniu Chomik jest 3,54 raza szybszy od tinygrad dla
MLP i 1,26 raza dla transformera; pierwszy przebieg transformera jest o 52,5%
krótszy.

## GPU przez Vulkan

Opcjonalna wtyczka `vulkan` używa `wgpu-native`, ale wybiera wyłącznie adapter,
którego `backend_type` jest równy `Vulkan`; nie może więc przejść po cichu na
D3D12 ani OpenGL. Każdy lazy graf jest kodowany jako jeden command buffer z
kernelami WGSL. Reshape i permute zachowują widoki przez stride'y, a tiled
matmul obsługuje batch i broadcasting bez pętli dispatchy po stronie Pythona.
Backend obejmuje pełny IR FP32, autograd oraz zwykły i in-place SGD.

`wgpu` wymaga Pythona 3.11 lub nowszego:

```powershell
python -m pip install '.[benchmark,vulkan]'
python -m pip install 'dawn-python==0.3.0'
python benchmarks\compare_tinygrad_10_cases.py --device vulkan --micro-only
```

Porównanie z tinygrad WEBGPU wymaga dodatkowo `dawn-python==0.3.0`. Skrypt
ustawia `WEBGPU_BACKEND=WGPUBackendType_Vulkan`, więc Dawn także nie wybiera
innego API. Indeksy int64 są po sprawdzeniu zakresu zawężane do int32, ponieważ
WGSL nie udostępnia przenośnego typu int64 dla buforów storage.

Na tej samej GeForce RTX 5070 Ti (`wgpu 0.31.1`, `dawn-python 0.3.0`,
`tinygrad 0.14.0`) pełny zestaw powtórzeń mikrobenchmarku dał:

| przypadek | Chomik Vulkan | tinygrad WEBGPU/Vulkan |
|---|---:|---:|
| elementwise, 1M | **5,111 ms** | 8,427 ms |
| reduce sum, 4M | **5,535 ms** | 42,152 ms |
| softmax, 1024×1024 | **3,755 ms** | 52,596 ms |
| matmul, 64×64 | **0,828 ms** | 3,347 ms |
| matmul, 256×256 | **0,929 ms** | 3,465 ms |
| matmul, 1024×1024 | **4,735 ms** | 9,807 ms |
| matmul, 2048×2048 | **20,517 ms** | 32,764 ms |
| batched matmul, 16×4×64 | **2,508 ms** | 4,721 ms |

Chomik wygrał wszystkie osiem przypadków. Jego pojedynczy pełny worker wykonał
20 epok MLP w 1,463 s i 10 epok transformera w 5,911 s. Worker treningowy
tinygrad/Dawn nie ukończył się w ciągu pięciu minut i został przerwany, dlatego
nie przypisano mu pozornego wyniku. Dla transformera Chomik Vulkan był 1,32×
szybszy od OpenCL (7,776 s), ale 2,8× wolniejszy od CUDA (2,095 s).

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

Na NVIDIA/CUDA użyj `--device cuda`, dla OpenCL `--device opencl`, a dla Vulkan
`--device vulkan`; domyślny tryb `metal` zachowuje dotychczasowe zachowanie na
Apple Silicon. Flaga `--micro-only` pomija dwa długie przypadki treningowe.
Opcja `--chomik-jit` przechwytuje oba treningi Chomika na CUDA albo OpenCL.

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
| elementwise, 1M | 1,554 ms | 2,767 ms | **0,971 ms** | 1,241 ms |
| reduce sum, 4M | 1,295 ms | 1,965 ms | **0,925 ms** | 1,422 ms |
| softmax, 1024×1024 | 1,248 ms | 2,689 ms | **0,656 ms** | 0,825 ms |
| matmul, 64×64 | 0,116 ms | 2,301 ms | **0,108 ms** | 0,187 ms |
| matmul, 256×256 | 0,215 ms | 2,385 ms | **0,186 ms** | 0,246 ms |
| matmul, 1024×1024 | 1,607 ms | 3,876 ms | **0,980 ms** | 1,281 ms |
| matmul, 2048×2048 | 6,689 ms | 9,948 ms | **3,981 ms** | 4,362 ms |
| batched matmul, 16×4×64 | 0,667 ms | 2,921 ms | **0,367 ms** | 0,541 ms |
| trening MLP, 20 epok | 0,563 s | 1,233 s | **0,253 s** | 0,375 s |
| trening transformera, 10 epok | 1,682 s | 1,722 s | **1,007 s** | 1,718 s |

PyTorch eager wygrał wszystkie dziesięć przypadków. Kontrole fingerprintów i
accuracy przeszły dla wszystkich frameworków. Mikrobenchmarki obejmują transfer
wejścia z NumPy na GPU oraz odczyt wyniku z powrotem do NumPy. Fused backward
softmax i LayerNorm skrócił trening transformera z 2,257 s do 1,682 s.

Po dodaniu CUDA JIT wykonano pięć naprzemiennych prób eager/JIT w jednym
procesie. Mediany dla aktualnej implementacji wyniosły:

| trening | Chomik CUDA | Chomik CUDA JIT | przyspieszenie |
|---|---:|---:|---:|
| MLP, 20 epok | 0,427 s | **0,233 s** | 1,83× |
| transformer, 10 epok | 1,640 s | **0,987 s** | 1,66× |

Accuracy pozostało w dotychczasowym zakresie. Te liczby są osobną serią A/B i
nie należy mieszać ich z wcześniejszą czteroframeworkową tabelą, wykonaną przy
innym obciążeniu GPU.

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

### Trening rdzenia LLM około 1B

Benchmark treningowy wykonuje na tym samym rdzeniu syntetyczny MSE oraz SGD
z `lr=1e-3`. Mierzy pierwszy krok i medianę kolejnych kroków, a zgodność Chomika
z PyTorch kontroluje przez fingerprint gradientu i zaktualizowanej wagi końcowej
normalizacji:

```bash
python benchmarks/llm_1b_training.py --steps 12
python benchmarks/llm_1b_training.py --steps 12 --inplace-sgd
python benchmarks/llm_1b_training.py --steps 12 --inplace-sgd --json
```

Na RTX 5070 Ti, FP32, `batch=1` i sekwencji 32 oba frameworki ukończyły 12
kroków bez OOM. Poniższy przebieg używa `--inplace-sgd` dla Chomika:

| metryka | Chomik CUDA | PyTorch eager |
|---|---:|---:|
| inicjalizacja modelu | 6,533 s | **6,485 s** |
| materializacja wag Chomika na GPU | 0,372 s | — |
| pierwszy `forward + backward + SGD` | **272,9 ms** | 293,6 ms |
| mediana 11 rozgrzanych kroków | 65,4 ms | **51,9 ms** |
| peak RAM procesu | 4588,1 MiB | **1186,6 MiB** |
| pamięć GPU raportowana przez runtime | 7795,4 MiB | **7750,4 MiB** |

Optymalizacja usuwa redukcje po osiach długości jeden, kieruje singleton-batch
do 2D GEMM, scala backward softmax do jednego kernela, scala trzy gradienty
LayerNorm we wspólne wywołanie CUDA oraz wykonuje aktualizację SGD jednym
kernelem. Względem poprzedniego pomiaru pierwszy krok Chomika skrócił się
z 335,8 ms do 272,9 ms, a warm step z 91,1 ms do 65,4 ms, czyli o 28,2%.
Analiza czasu życia buforów zwalnia wyniki po ostatnim użyciu i obniżyła peak
domyślnego trybu z 11817,8 MiB do 11635,7 MiB. Opcjonalny in-place SGD usuwa
kopię nowych wag i obniża peak dalej do 7795,4 MiB, tylko o 45,0 MiB (0,6%)
więcej od PyTorch. Warm step Chomika pozostaje o 26,1% wolniejszy.

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
