# openpilot.distill su NVIDIA Tesla V100 — handoff per Claude

Data: 17 ottobre 2026.

Questo documento contiene la risposta integrale di Codex e, in appendice, lo script Python completo a cui la risposta fa riferimento. I collegamenti allo script temporaneo sono stati adattati per puntare all'appendice; il testo della risposta è altrimenti conservato.

Stato delle verifiche: sono stati ispezionati sorgenti e metadati del torchpackage pubblico e il checkout locale di openpilot.distill. Lo script ha superato soltanto il controllo sintattico e della CLI. Conversione dei pesi, forward CPU/GPU, confronto numerico e benchmark restano da eseguire. Le stime non sono misure.

La consegna di questo documento non autorizza modifiche live, installazioni o riavvii sul comma, né l'accensione di Serverone-ai.

## Risposta integrale

**La strada più solida è dequantizzare su CPU, usare pesi densi FP16 e KV cache FP16, eseguire il modello in eager con SDPA.** Prima costruirei anche un riferimento denso FP32.

Ho ispezionato i sorgenti e i metadati del torchpackage pubblico tramite letture parziali, senza scaricare tutti i pesi. Ho preparato [uno script Python completo](#appendice-script-python-completo) per ispezione, conversione, caricamento e confronto. **Ho verificato sintassi e CLI; conversione e forward non sono stati eseguiti.** Nessuna modifica al repository o ai dispositivi.

1. **Dequantizzazione: nel pacchetto il formato è identificabile con precisione.**

   Nell’indice dello state dict compaiono effettivamente:

   | Parte | Rappresentazione |
   |---|---|
   | Attention | `torchao.quantization.Float8Tensor` |
   | MLP | `torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor` |
   | Dati FP4 | `uint8`, due valori per byte |
   | Blocchi NVFP4 | 16 elementi |
   | Scale NVFP4 | FP8 E4M3, con layout **swizzled** |
   | Scala globale | `per_tensor_scale` |
   | Dtype originale dichiarato | BF16 |

   Quindi **non basta `weight.half()`**: sui tensor subclass può cambiare il dtype dichiarato lasciando dati e comportamento quantizzati.

   Il nucleo della conversione è:

   ```python
   from torchao.quantization import Float8Tensor
   from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

   if isinstance(weight, NVFP4Tensor):
       working = weight.to(dtype=torch.float32)
       dense = working.dequantize(torch.float32)
   elif isinstance(weight, Float8Tensor):
       dense = weight.dequantize(torch.float32)
   else:
       raise TypeError(type(weight))

   if type(dense) is not torch.Tensor:
       raise TypeError("La dequantizzazione ha conservato un tensor subclass")

   if not torch.isfinite(dense).all():
       raise ValueError("Pesi dequantizzati non finiti")

   dense_fp16 = dense.to(torch.float16)

   if not torch.isfinite(dense_fp16).all():
       raise ValueError("Overflow nella conversione FP16")
   ```

   Il passaggio intermedio NVFP4 è intenzionale: in torchao 0.17, `get_hp_scales()` consulta `orig_dtype`. Portarlo a FP32 prima della dequantizzazione evita un passaggio delle scale in BF16. Il dequantizzatore gestisce anche trasposizioni e riordinamento delle scale. [Implementazione NVFP4 di torchao 0.17](https://github.com/pytorch/ao/blob/v0.17.0/torchao/prototype/mx_formats/nvfp4_tensor.py).

   Concettualmente:

   ```text
   peso = valore_FP4_decodificato × scala_del_blocco × scala_globale
   ```

   Ma ordine dei nibble, orientamento e swizzle devono seguire quella classe. **Non ricostruirei manualmente il formato quando il dequantizzatore è disponibile.**

   La conversione conserva l’approssimazione dei pesi quantizzati: non recupera i pesi BF16 precedenti alla quantizzazione.

2. **Nel modello ispezionato non serve sostituire indiscriminatamente i Linear.**

   La classe `torchtitan.models.common.nn_modules.Linear` eredita da `nn.Linear` senza ridefinirne il forward. La quantizzazione risiede nei pesi tensor subclass.

   Per questo pacchetto conviene:

   - mantenere nomi e struttura dei moduli;
   - caricare tensori ordinari con `strict=True, assign=True`;
   - selezionare SDPA;
   - cambiare il resolver del dtype della cache;
   - eliminare cache e maschere precedentemente costruite.

   Il resolver attuale riconosce FP8 e altrimenti restituisce BF16: **passare semplicemente `"float16"` oggi non produce una cache FP16**. Nel sorgente estratto lo cambierei così:

   ```python
   KVCacheDType = Literal[
       "bfloat16",
       "float8_e4m3fn",
       "float16",
       "float32",
   ]

   def _resolve_kv_cache_dtype(requested_dtype):
       if requested_dtype == "float16":
           return torch.float16
       elif requested_dtype == "float32":
           return torch.float32
       elif requested_dtype == "bfloat16":
           return torch.bfloat16
       elif requested_dtype == "float8_e4m3fn":
           return torch.float8_e4m3fn
       else:
           raise ValueError(requested_dtype)
   ```

   Poi:

   ```python
   output = model.generate(
       **inputs,
       dtype=torch.float16,
       kv_cache_dtype="float16",
       steps=15,
       num_prefill_frames=14,
       cfg=2.0,
   )
   ```

   Le cache K/V sono già registrate con `persistent=False`: **non introducono chiavi nello state dict e non compromettono `strict=True`**.

   Per la prima prova manterrei il torchpackage originale come contenitore del codice, caricando separatamente lo state dict denso. Lo script allegato usa questa soluzione e applica l’adattamento nel namespace privato del `PackageImporter`.

   Dopo la validazione renderei permanenti le modifiche nei sorgenti estratti. Un nuovo torchpackage richiederebbe anche metadati aggiornati: quelli originali dichiarano BF16, `fp8_nvfp4` e compute capability minima `(10, 0)`. Cambiare soltanto questi metadati non rende compatibile il modello.

3. **`compile_for_inference()` va inizialmente saltato, insieme al percorso FLEX.**

   Nel codice reale la funzione fa:

   ```python
   for block in self.blocks:
       block.compile(mode="max-autotune-no-cudagraphs")
   ```

   Non converte i pesi e non è necessaria al sampling.

   Esiste inoltre una compilazione separata per costruire le maschere FLEX. Per la prima versione V100 imposterei SDPA su configurazione e moduli attention, invalidando le maschere esistenti.

   **Saltare soltanto `compile_for_inference()` lasciando FLEX e pesi quantizzati non è sufficiente.** Partirei senza compilazione; eventuali ottimizzazioni arrivano dopo il confronto numerico.

4. **FP16 va validato: proteggere solo il softmax non basta.**

   Userei questa politica iniziale:

   | Operazione | Precisione |
   |---|---|
   | Pesi delle proiezioni e MLP | FP16 |
   | KV cache | FP16 |
   | RMSNorm, incluse Q/K norm | Calcolo FP32, uscita nel dtype richiesto |
   | Attention di riferimento | SDPA `MATH` |
   | Frequenze e funzioni sinusoidali dei timestep | FP32 |
   | CFG e aggiornamento Euler | Preferibilmente FP32 |
   | Residual e modulazione adattiva | Monitorare; promuovere a FP32 se necessario |

   Il modello mantiene già in FP32 una parte del calcolo dei timestep. Nel backend SDPA matematico PyTorch mantiene normalmente gli intermedi in FP32 per ingressi FP16/BF16. Questo migliora la stabilità, ma aumenta memoria e costo dell’attention. [Numerical accuracy di PyTorch](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

   La combinazione CFG va promossa **prima** delle operazioni:

   ```python
   unconditional = unconditional.float()
   conditional = conditional.float()

   velocity = unconditional + cfg * (conditional - unconditional)
   next_x = scheduler.step(velocity, step_idx, x.float())
   x = next_x.to(dtype=x.dtype)
   ```

   Analogamente, un residual FP32 deve effettuare somma e gating in FP32. Fare `.float()` dopo che un valore è diventato `inf` non lo recupera.

   Lo script fornito protegge RMSNorm e usa SDPA matematico; **è una baseline da validare**, non una garanzia che tutti i residual FP16 siano stabili.

   FP32 completo ha senso come riferimento e per diagnosticare overflow. Su V100 32 GB è plausibile per il world model isolato; su 16 GB pesi e memoria operativa rendono la configurazione impraticabile. GradScaler non risolve gli overflow del forward in inferenza.

   Per l’autoencoder partirei da FP32. I suoi circa 128 milioni di parametri complessivi occupano circa 0,48 GiB in FP32. [Model card dell’autoencoder](https://huggingface.co/commaai/vit-ae-2x-f8c32).

   Attenzione agli export: `.float()` sui parametri non elimina eventuali cast BF16 incorporati nel grafo. Per scoprirli:

   ```python
   import torch

   exported = torch.export.load("encoder.pt2")
   print(exported.graph_module.code)
   ```

   Se contiene cast espliciti o operazioni specializzate, va adattato il grafo oppure riesportato dal sorgente; non assumerei che `.to(torch.float16)` basti.

5. **Memoria: la 32 GB è il bersaglio ragionevole; la 16 GB richiede margini misurati.**

   Dai metadati della struttura ho ricavato **56 blocchi, dimensione 2304, 36 teste**, quindi head dimension 64.

   Questo sampling condivide il prefisso tra i due rami CFG:

   ```text
   prefisso:          14 × 128 = 1792 token
   suffix con CFG:     2 × 128 =  256 token
   capacità cache:                 2048 token
   ```

   La cache FP16 vale quindi:

   ```text
   2 × 56 × 2048 × 2304 × 2 byte = 0,984375 GiB
   ```

   **Non raddoppia tutta la cache per CFG.** Il funzionamento prefill/decode è descritto anche nella [model card del world model](https://huggingface.co/commaai/worldmodel-4B).

   Stime di pianificazione, non misure:

   | Componente | Ordine di grandezza |
   |---|---:|
   | Pesi 4B FP16 | 7,45 GiB |
   | KV FP16 con CFG | 0,98 GiB |
   | Una matrice score FP32 del prefill | 0,49 GiB |
   | World model FP16 con SDPA matematico, complessivo | circa 11–15+ GiB |
   | World model interamente FP32 | circa 19–24+ GiB |

   Gli intermedi dell’attention vivono per il blocco corrente: non bisogna moltiplicarne il picco per 56. Vanno però aggiunti allocator, workspace e gli altri modelli. **Queste stime non includono il learner con gradienti e optimizer.**

   SDPA matematico è sufficiente per una prima implementazione corretta. Non introdurrei subito xformers. Dopo, proverei il backend `EFFICIENT_ATTENTION` disponibile nella build, verificando supporto di GPU, shape e maschera. SDPA consente di selezionare esplicitamente i backend. [Documentazione SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

   Per la velocità, il solo conteggio delle proiezioni è circa:

   ```text
   2 × 4B × (1792 + 15 × 256) ≈ 45 TFLOP
   ```

   cui aggiungere attention e altri calcoli. Budgetterei **secondi o decine di secondi per predizione**, non tempo reale, finché non esiste un benchmark. PCIe/SXM, backend attention e precisione cambiano molto il risultato.

6. **La procedura riproducibile è nello script allegato.**

   Copia [worldmodel_v100_tools.py](#appendice-script-python-completo) nell’ambiente Linux. I comandi seguenti assumono che il torchpackage sia già disponibile e che l’ambiente contenga versioni compatibili di PyTorch e torchao. Nel lockfile locale ho trovato torchao **0.17.0**: partirei da quella versione, verificandone l’importazione.

   ```bash
   uv run --no-sync python worldmodel_v100_tools.py inspect \
     models/model.fp8_nvfp4.torchpackage \
     --sources worldmodel-inspected \
     > worldmodel-inspection.txt
   ```

   Questo stampa classi, shape, dtype e attributi di quantizzazione, oltre al codice dei dequantizzatori installati.

   Costruisci prima il riferimento FP32:

   ```bash
   uv run --no-sync python worldmodel_v100_tools.py convert \
     models/model.fp8_nvfp4.torchpackage \
     worldmodel-dense-fp32.pt \
     --dtype fp32
   ```

   Poi la versione FP16:

   ```bash
   uv run --no-sync python worldmodel_v100_tools.py convert \
     models/model.fp8_nvfp4.torchpackage \
     worldmodel-dense-fp16.pt \
     --dtype fp16
   ```

   Prevedi spazio per entrambi gli output, circa 24 GB decimali oltre all’originale. Per la conversione FP32 pianificherei almeno 32 GB di RAM libera, preferibilmente più margine.

   Genera un input sintetico comune:

   ```bash
   uv run --no-sync python worldmodel_v100_tools.py inputs \
     models/model.fp8_nvfp4.torchpackage \
     smoke-inputs.pt
   ```

   Primo confronto con un solo step:

   ```bash
   uv run --no-sync python worldmodel_v100_tools.py run \
     models/model.fp8_nvfp4.torchpackage \
     worldmodel-dense-fp32.pt smoke-inputs.pt cpu-output.pt \
     --device cpu --dtype fp32 --steps 1

   uv run --no-sync python worldmodel_v100_tools.py run \
     models/model.fp8_nvfp4.torchpackage \
     worldmodel-dense-fp16.pt smoke-inputs.pt gpu-output.pt \
     --device cuda:0 --dtype fp16 --steps 1

   uv run --no-sync python worldmodel_v100_tools.py compare \
     cpu-output.pt gpu-output.pt
   ```

   Il caricamento usa `strict=True, assign=True`; il confronto riporta errore massimo assoluto, RMSE e errore L2 relativo.

   Dopo il primo esito: ripeti con 15 step, quindi con ingressi reali e rollout. Il tempo stampato dallo script è una chiamata **a freddo**, non un benchmark stabilizzato. La memoria riportata copre le allocazioni PyTorch, non quelle di ORT.

   **Il riferimento CPU FP32 verifica il porting denso. Non dimostra equivalenza al modello quantizzato originale**, che quantizza anche le attivazioni e usa cache FP8. Per quella verifica serve l’originale su hardware supportato oppure un’emulazione completa delle quantizzazioni. Il semplice test sintetico non valida la qualità del simulatore.

7. **`onnxruntime-gpu 1.29` da PyPI non è la build giusta per la V100.**

   Qui la documentazione è esplicita:

   | Build ORT 1.29 Linux x64 | CUDA | `sm_70` |
   |---|---|---|
   | PyPI predefinita | 13.x | Assente |
   | Variante CUDA 12 | 12.8 | Presente |

   Lo dichiarano sia la [matrice CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) sia le [release notes 1.29](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0).

   **Torch `+cu126` e ORT CUDA 12.8 non sono automaticamente incompatibili, ma non basta che entrambi dicano “CUDA 12”.** ORT richiede librerie adeguate alla propria build; quelle caricate da Torch cu126 possono non bastare.

   Per verificare la disponibilità della variante 1.29 nel feed Microsoft, senza installare:

   ```bash
   uv pip install --dry-run --no-deps \
     --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
     "onnxruntime-gpu==1.29.0"
   ```

   Per mantenere un ambiente basato sulle librerie di Torch cu126, **ORT 1.20.1 CUDA 12/cuDNN 9 è una candidata conservativa**, ma soltanto se il modello è compatibile: ORT 1.20 supporta ONNX IR fino a 10 e opset standard fino a 21. [Compatibilità ORT](https://onnxruntime.ai/docs/reference/compatibility.html).

   Controlla prima:

   ```python
   import onnx

   model = onnx.load(
       "models/big_driving_supercombo.onnx",
       load_external_data=False,
   )

   print("IR:", model.ir_version)
   for entry in model.opset_import:
       print("DOMAIN:", entry.domain, "OPSET:", entry.version)
   ```

   Nel progetto vanno controllati anche i grafi prodotti da `split_supercombo()` e `make_dense()`, perché sono quelli effettivamente passati a ORT.

   Se serve ORT recente, userei la variante CUDA 12.8 con dipendenze verificate, oppure una build da sorgente mirata. **Non proporrei una build 1.29 con CUDA 12.6 come sicuramente supportata senza verificarne i requisiti di compilazione.**

   Verifica anche cuDNN effettivo: la [matrice NVIDIA di cuDNN 9.10.2](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html) include esplicitamente Volta.

   Infine, `get_available_providers()` non prova l’esecuzione sulla GPU. Servono una vera inferenza e il profiling dei nodi; il fallback CPU può mascherare una configurazione errata.

8. **Per l’insegnante partirei dall’ONNX originale; FP16 viene dopo il riferimento.**

   Mantieni inizialmente la precisione originale per verificare runtime e risultati. Poi prova FP16 o mixed precision: sulla V100 può ridurre memoria e accelerare le parti compatibili.

   Nel tuo progetto convertirei i grafi **dopo split e densificazione**, mantenendo inizialmente gli I/O FP32:

   ```python
   import onnx
   from onnxconverter_common import float16

   from supervised.onnx_helpers import make_dense, split_supercombo

   original = onnx.load("models/big_driving_supercombo.onnx")
   vision, policies = split_supercombo(original)
   policies = make_dense(policies)

   vision_fp16 = float16.convert_float_to_float16(
       vision,
       keep_io_types=True,
   )

   policies_fp16 = float16.convert_float_to_float16(
       policies,
       keep_io_types=True,
   )
   ```

   `keep_io_types=True` conserva l’interfaccia esterna. Il convertitore supporta inoltre esclusioni per operatori o nodi quando alcune parti devono restare FP32. [Conversione FP16 e mixed precision di ORT](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).

   Confronta FP32 e FP16 sulla **stessa sequenza registrata**, partendo dallo stesso stato delle code: un singolo frame non rileva la deriva temporale.

   Le sei varianti Hugging Face non sono intercambiabili solo perché hanno lo stesso filename. Prima di sceglierne una vanno confrontati I/O, metadati e compatibilità con lo split del progetto; un README vuoto non permette di stabilire quale sia migliore.

<oai-mem-citation>
<citation_entries>
MEMORY.md:476-476|note=[prior V100 compatibility context checked against current primary sources]
</citation_entries>
<rollout_ids>
01a1011b-f2bb-78f0-9942-96bd7781134d
</rollout_ids>
</oai-mem-citation>

## Appendice: script Python completo

Salvare il contenuto del seguente blocco come `worldmodel_v100_tools.py`. Il codice è quello del prototipo descritto nella risposta, senza modifiche aggiuntive. Richiede revisione e verifica nell'ambiente Linux di destinazione prima di considerarlo funzionante.

```python
"""Inspection and dense eager prototype for the commaai worldmodel package.

The inference and conversion paths require the package's compatible torchao.
This adapter was derived from the inspected package, but has not run on a GPU.
Only load pickle files from a trusted source.
"""

import argparse
import inspect
import json
from pathlib import Path
import time
import types
import zipfile


def load_quantized_state(path):
    import torch

    with zipfile.ZipFile(path) as archive:
        with archive.open("archive/assets/state_dict.pt") as stream:
            state = torch.load(stream, map_location="cpu", weights_only=False)
    return state


def describe_tensor(value):
    return {
        "class": type(value).__module__ + "." + type(value).__name__,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def inspect_package(args):
    import torch
    from torch.package import PackageImporter

    with zipfile.ZipFile(args.package) as archive:
        if args.sources is not None:
            root = args.sources.resolve()
            root.mkdir(parents=True, exist_ok=True)
            for name in archive.namelist():
                if not name.endswith(".py"):
                    continue
                target = (root / name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Unsafe archive path: " + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(archive.read(name))

    importer = PackageImporter(str(args.package))
    metadata = importer.load_pickle("meta", "meta.pkl", map_location="cpu")
    print("METADATA", metadata)
    state = load_quantized_state(args.package)
    counts = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError((name, type(value)))
        description = describe_tensor(value)
        kind = description["class"]
        counts[kind] = counts.get(kind, 0) + 1
        for attribute in (
            "qdata", "scale", "per_tensor_scale", "act_per_tensor_scale",
            "block_size", "orig_dtype", "is_swizzled_scales", "use_triton_kernel",
        ):
            if not hasattr(value, attribute):
                continue
            item = getattr(value, attribute)
            if isinstance(item, torch.Tensor):
                description[attribute] = describe_tensor(item)
            else:
                description[attribute] = str(item)
        print(name, json.dumps(description))
    print("COUNTS", counts)
    seen = set()
    for value in state.values():
        cls = type(value)
        if cls is torch.Tensor or cls in seen:
            continue
        seen.add(cls)
        print("CLASS SOURCE", inspect.getfile(cls))
        if hasattr(cls, "dequantize"):
            print(inspect.getsource(cls.dequantize))
        if hasattr(cls, "get_hp_scales"):
            print(inspect.getsource(cls.get_hp_scales))


def convert(args):
    import torch
    from torchao.quantization import Float8Tensor
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    if args.output.exists():
        raise FileExistsError(args.output)
    state = load_quantized_state(args.package)
    dense = {}
    total = 0
    for name in list(state.keys()):
        value = state.pop(name)
        if isinstance(value, NVFP4Tensor):
            # In torchao 0.17 get_hp_scales consults orig_dtype.
            # This changes wrapper metadata, then dequantizes in FP32.
            working = value.to(dtype=torch.float32)
            result = working.dequantize(torch.float32)
        elif isinstance(value, Float8Tensor):
            result = value.dequantize(torch.float32)
        elif type(value) is torch.Tensor:
            if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise TypeError("Raw FP8 needs an explicit scale mapping: " + name)
            result = value
        else:
            raise TypeError("Unsupported state entry: " + name + " " + str(type(value)))

        if type(result) is not torch.Tensor:
            raise TypeError("Dequantization retained a tensor subclass: " + name)
        if result.is_floating_point():
            result = result.float()
            if not torch.isfinite(result).all().item():
                raise ValueError("Nonfinite source: " + name)
            if args.dtype == "fp16":
                result = result.to(torch.float16)
            if not torch.isfinite(result).all().item():
                raise ValueError("Overflow after conversion: " + name)
        result = result.contiguous()
        dense[name] = result
        total += result.numel() * result.element_size()
        del value, result
    with args.output.open("xb") as stream:
        torch.save(dense, stream)
    print("SAVED", args.output, "tensor GiB", total / (1024 ** 3))


def resolve_dense_cache_dtype(name):
    import torch

    if name == "float16":
        return torch.float16
    elif name == "float32":
        return torch.float32
    else:
        raise ValueError("Dense prototype requires float16 or float32 KV cache")


def rmsnorm_fp32(self, x):
    import torch.nn.functional as functional

    weight = None
    if self.weight is not None:
        weight = self.weight.float()
    result = functional.rms_norm(x.float(), self.normalized_shape, weight, self.eps)
    return result.to(dtype=x.dtype)


def load_dense_model(package, checkpoint, device, dtype):
    import torch
    from torch.package import PackageImporter

    importer = PackageImporter(str(package))
    inference = importer.import_module("torchtitan.experiments.worldmodel.model_for_inference")
    architecture = importer.import_module("torchtitan.experiments.worldmodel.model")
    model = importer.load_pickle("model", "model.pkl", map_location="cpu")
    metadata = importer.load_pickle("meta", "meta.pkl", map_location="cpu")

    # Adapter confined to this PackageImporter's private module namespace.
    # A maintained fork should implement this resolver directly in source.
    inference._resolve_kv_cache_dtype = resolve_dense_cache_dtype
    model.cleanup_caches()
    model.clear_inference_masks()
    model.mask = None
    model.config.transformer.attention_impl = "SDPA"
    for module in model.modules():
        if isinstance(module, architecture.SelfAttention):
            module.config.attention_impl = "SDPA"
            module.flex_attention = None
            if module.sdpa is None:
                module.sdpa = architecture.ScaledDotProductAttention.Config().build()
        if isinstance(module, torch.nn.Linear):
            if type(module).forward is not torch.nn.Linear.forward:
                raise TypeError("Review custom Linear.forward: " + str(type(module)))

    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    for name, value in state.items():
        if type(value) is not torch.Tensor:
            raise TypeError("Dense checkpoint contains a subclass: " + name)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval()
    model.to(device=device, dtype=dtype)
    for module in model.modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.forward = types.MethodType(rmsnorm_fp32, module)
    for name, value in model.named_parameters():
        if value.is_meta:
            raise ValueError("Unmaterialized parameter: " + name)
    model.requires_grad_(False)
    # Deliberately no compile_for_inference() and no quantize_for_inference().
    return model, metadata


def make_inputs(args):
    import torch
    from torch.package import PackageImporter

    if args.output.exists():
        raise FileExistsError(args.output)
    importer = PackageImporter(str(args.package))
    metadata = importer.load_pickle("meta", "meta.pkl", map_location="cpu")
    shapes = metadata["model_io"]["in_shape"]
    torch.manual_seed(1234)
    inputs = {}
    for name, shape in shapes.items():
        if name == "latents":
            inputs[name] = torch.randn(tuple(shape), dtype=torch.float32)
        elif name in ("augments_pos_ref_augment", "ref_augment_from_augments_euler"):
            inputs[name] = torch.zeros(tuple(shape), dtype=torch.float32)
        elif name == "pose_mask":
            inputs[name] = torch.ones(tuple(shape), dtype=torch.int64)
            inputs[name][:, -1] = 0
        elif name == "fidxs":
            inputs[name] = torch.arange(shape[1], dtype=torch.int64).unsqueeze(0).expand(shape[0], -1).clone()
        else:
            raise ValueError("Unknown input: " + name)
    with args.output.open("xb") as stream:
        torch.save(inputs, stream)


def run(args):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if args.dtype == "fp16":
        dtype = torch.float16
        cache_name = "float16"
    else:
        dtype = torch.float32
        cache_name = "float32"
    if device.type == "cuda":
        print("GPU", torch.cuda.get_device_name(device))
        print("CAPABILITY", torch.cuda.get_device_capability(device))
        print("ARCH LIST", torch.cuda.get_arch_list())
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    model, metadata = load_dense_model(args.package, args.checkpoint, device, dtype)
    print("TRANSFORMER", model.config.transformer)
    source = torch.load(args.inputs, map_location="cpu", weights_only=True)
    inputs = {}
    for name, value in source.items():
        if value.is_floating_point():
            inputs[name] = value.to(device=device, dtype=dtype)
        else:
            inputs[name] = value.to(device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        outputs = model.generate(
            **inputs,
            steps=args.steps,
            num_prefill_frames=inputs["latents"].shape[1] - 1,
            dtype=dtype,
            inference_schedule="linear",
            cfg=2.0,
            kv_cache_dtype=cache_name,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print("PEAK TORCH GiB", torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    print("COLD GENERATE SECONDS", time.perf_counter() - started)
    saved = {}
    for name, value in outputs.items():
        value = value.detach().float().cpu()
        if not torch.isfinite(value).all().item():
            raise ValueError("Nonfinite output: " + name)
        saved[name] = value
    with args.output.open("xb") as stream:
        torch.save(saved, stream)


def compare(args):
    import torch

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    if reference.keys() != candidate.keys():
        raise ValueError("Output keys differ")
    for name in reference:
        a = reference[name].double()
        b = candidate[name].double()
        if a.shape != b.shape:
            raise ValueError("Shape mismatch: " + name)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite output: " + name)
        error = b - a
        rmse = error.square().mean().sqrt()
        reference_rms = a.square().mean().sqrt().clamp_min(1e-12)
        print(name, "max_abs", error.abs().max().item(), "rmse", rmse.item(),
              "relative_l2", (rmse / reference_rms).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("package", type=Path)
    inspect_parser.add_argument("--sources", type=Path)
    inspect_parser.set_defaults(function=inspect_package)
    convert_parser = commands.add_parser("convert")
    convert_parser.add_argument("package", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--dtype", choices=("fp16", "fp32"), required=True)
    convert_parser.set_defaults(function=convert)
    inputs_parser = commands.add_parser("inputs")
    inputs_parser.add_argument("package", type=Path)
    inputs_parser.add_argument("output", type=Path)
    inputs_parser.set_defaults(function=make_inputs)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("package", type=Path)
    run_parser.add_argument("checkpoint", type=Path)
    run_parser.add_argument("inputs", type=Path)
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("--device", required=True)
    run_parser.add_argument("--dtype", choices=("fp16", "fp32"), required=True)
    run_parser.add_argument("--steps", type=int, default=1)
    run_parser.set_defaults(function=run)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("reference", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
```
