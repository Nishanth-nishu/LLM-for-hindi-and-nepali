# Phase 1 — Reproducibility Report

| | |
|---|---|
| Generated (UTC) | 2026-08-25 09:38:50 |
| Git commit | `987ad30458abfbc6b83154a7717d846e4bc85abc` |
| Branch | `phase-1` |
| Working tree | **dirty** — uncommitted changes present |

> Values marked **ESTIMATE** are character-based proxies computed before a tokenizer existed. Final token counts are measured by encoding the final corpus with the final tokenizer (`count_corpus_tokens.py`).

## Environment

| | |
|---|---|
| Python | 3.12.13 |
| Platform | Linux-6.1.0-52-cloud-amd64-x86_64-with-glibc2.36 |
| Processor | unknown |
| Git commit | `987ad30458abfbc6b83154a7717d846e4bc85abc` |
| Git branch | `phase-1` |

### Installed packages

```
absl-py @ file:///home/conda/feedstock_root/build_artifacts/absl-py_1783083173550/work
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiohttp-cors==0.8.1
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.14.2
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
array_record==0.8.3
arrow==1.4.0
asttokens==3.0.2
astunparse==1.6.3
async-lru==2.3.0
atpublic==5.1
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.15.0
bigframes==2.48.0
bleach==6.4.0
blessed==1.48.0
bq_stats @ file:///tmp/environments/base/pip/packages/bq_stats
cachetools==5.5.2
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
cheroot==11.1.2
click==8.4.2
click-option-group==0.5.7
cloud-tpu-client==0.7
cloud-tpu-profiler==2.4.0
cloudpickle==3.1.2
colorama==0.4.6
colorful==0.5.8
comm==0.2.3
contourpy==1.3.3
courlan==1.4.0
cryptography==42.0.8
cuda-bindings==13.3.1
cuda-pathfinder==1.6.0
cuda-toolkit==13.0.2
cupy-cuda13x==14.1.1
cycler==0.12.1
dataproc-spark-connect==1.1.0
datasets==5.0.1
dateparser==1.4.2
db-dtypes==1.7.1
debugpy==1.8.21
decorator==5.3.1
defusedxml==0.7.1
dill==0.4.1
distlib==0.4.3
distro==1.9.0
dm-tree==0.1.10
docker==7.2.0
docstring_parser==0.18.0
einops==0.8.2
entrypoints==0.4
etils==1.14.0
executing==2.2.1
explainable-ai-sdk==1.3.3
Farama-Notifications==0.0.6
fastapi==0.141.1
fastjsonschema==2.22.2
filelock==3.32.3
fire==0.7.1
flatbuffers==25.12.19
fonttools==4.63.0
fqdn==1.5.1
frozenlist==1.8.0
fsspec==2026.6.0
gast==0.7.0
gcsfs==2026.8.0
geopandas==1.1.4
gitdb==4.0.12
GitPython==3.1.59
google-api-core==2.34.0
google-api-python-client==2.198.0
google-auth==2.37.0
google-auth-httplib2==0.4.1
google-auth-oauthlib==1.4.0
google-cloud-aiplatform==1.164.0
google-cloud-artifact-registry==1.22.0
google-cloud-bigquery==3.43.0
google-cloud-bigquery-connection==1.22.0
google-cloud-bigquery-storage==2.40.0
google-cloud-core==2.6.1
google-cloud-dataproc==5.30.0
google-cloud-datastore==2.26.0
google-cloud-functions==1.24.0
google-cloud-language==2.21.0
google-cloud-monitoring==2.31.0
google-cloud-resource-manager==1.18.0
google-cloud-storage==3.13.1
google-cloud-storage-control==1.13.0
google-crc32c==1.8.0
google-genai==2.18.1
google-pasta==0.2.0
google-resumable-media==2.10.1
googleapis-common-protos==1.75.1
gpustat==1.1.1
greenlet==3.5.5
grpc-google-iam-v1==0.14.5
grpcio==1.83.0
grpcio-status==1.83.0
gviz-api==1.10.0
gymnasium==1.2.2
h11==0.16.0
h5py==3.14.0
hf-xet==1.6.0
htmldate==1.10.0
htmlmin==0.1.12
httpcore==1.0.9
httplib2==0.32.0
httptools==0.8.0
httpx==0.28.1
huggingface_hub==1.28.0
humanize==4.16.0
idna==3.18
ImageHash==4.3.2
immutabledict==4.3.1
ipykernel==7.3.0
ipython==9.16.1
ipython-genutils==0.2.0
ipython-sql==0.5.0
ipython_pygments_lexers==1.1.1
ipywidgets==8.1.8
isoduration==20.11.0
jaraco.classes==3.4.0
jaraco.context==6.1.2
jaraco.functools==4.6.0
jedi==0.20.0
jeepney==0.9.0
Jinja2==3.1.6
jinxed==2.1.0
joblib==1.5.3
json5==0.15.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_builder==1.2.2
jupyter_client==8.9.1
jupyter_core==5.9.1
jupyter_server==2.20.0
jupyter_server_terminals==0.5.4
jupyterlab==4.6.3
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.0
jupyterlab_widgets==3.0.16
jusText==3.0.2
keras==3.15.1
keras-tuner==1.4.8
keyring==25.7.0
keyrings.google-artifactregistry-auth==1.1.2
kfp==2.17.0
kfp-pipeline-spec==2.17.0
kfp-server-api==2.17.0
kiwisolver==1.5.0
kt-legacy==1.0.5
kubernetes==30.1.0
lark==1.3.1
libclang==18.1.1
linkify-it-py==2.1.0
llvmlite==0.49.0
lxml==6.1.2
lxml_html_clean==0.4.5
lz4==4.4.5
Markdown==3.10.3
markdown-it-py==4.2.0
MarkupSafe==2.1.5
matplotlib==3.11.1
matplotlib-inline==0.2.2
mdit-py-plugins==0.6.1
mdurl==0.1.2
memray==1.20.0
missingno==0.5.2
mistune==3.3.4
ml_dtypes==0.6.0
mmh3==5.2.1
more-itertools==11.1.0
mpmath==1.3.0
msgpack==1.2.1
multidict==6.7.1
multimethod==2.1
multiprocess==0.70.19
namex==0.1.0
narwhals==2.24.0
nb_micromamba_kernels @ file:///tmp/nb_micromamba_kernels
nbclient==0.11.0
nbconvert==7.17.1
nbdime==4.0.4
nbformat==5.11.0
nest-asyncio2==1.7.2
networkx==3.6.1
notebook==7.6.2
notebook_shim==0.2.4
numba==0.67.0
numpy==2.5.2
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-ml-py==13.610.43
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
oauth2client==4.1.3
oauthlib==3.3.1
opencensus==0.11.4
opencensus-context==0.1.3
opentelemetry-api==1.44.0
opentelemetry-exporter-prometheus==0.65b0
opentelemetry-proto==1.44.0
opentelemetry-sdk==1.44.0
opentelemetry-semantic-conventions==0.65b0
opt_einsum==3.4.0
optree==0.19.1
ormsgpack==1.12.2
packaging @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_packaging_1785888127/work
pandas==3.0.5
pandas-gbq==0.35.1
pandas-profiling==3.2.0
pandocfilters==1.5.1
papermill==2.7.0
parso==0.8.7
pdf2image==1.17.0
pexpect==4.9.0
phik==0.12.5
pillow==12.3.0
platformdirs==4.11.3
plotly==6.9.0
pluggy==1.6.0
prettytable==3.18.0
prometheus_client==0.26.0
promise==2.3
prompt_toolkit==3.0.53
propcache==0.5.2
Protego==0.6.2
proto-plus==1.28.3
protobuf==6.33.6
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
py-spy==0.4.2
py4j==0.10.9.9
pyarrow==25.0.1
pyasn1==0.6.4
pyasn1_modules==0.4.2
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
pydata-google-auth==1.9.1
Pygments==2.20.0
pyiceberg==0.11.1
pyogrio==0.13.0
pyOpenSSL==24.0.0
pyparsing==3.3.2
pyproj==3.7.2
pyroaring==1.1.0
pyspark==4.0.4
pytesseract==0.3.13
python-dateutil==2.9.0.post0
python-discovery==1.5.2
python-dotenv==1.2.3
python-json-logger==4.2.0
pytz==2026.3.post1
PyWavelets==1.9.0
PyYAML==6.0.3
pyzmq==27.1.0
ray==2.57.0
ray-haproxy==2.8.25
referencing==0.37.0
regex==2026.7.19
requests==2.33.0
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
retrying==1.4.2
rfc333
```

## Configuration files

- `hindi/configs/data_config.yaml`
- `hindi/configs/pdf_sources.txt`
- `hindi/configs/seed_domains.txt`
- `hindi/configs/tokenizer_config.yaml`
- `nepali/configs/data_config.yaml`
- `nepali/configs/pdf_sources.txt`
- `nepali/configs/seed_domains.txt`
- `nepali/configs/tokenizer_config.yaml`

## GCS bucket structure

```
gs://lma-01-hi-ne-corpus/
└── raw/
    ├── hi/
    │   ├── wikipedia/data.jsonl
    │   ├── sangraha/verified/data.jsonl
    │   ├── sangraha/unverified/data.jsonl
    │   ├── opus/OpenSubtitles_en-hi.zip      (excluded — see configs)
    │   └── hi_ne_corpus.zip                  (excluded — see configs)
    └── ne/
        ├── wikipedia/data.jsonl
        ├── sangraha/verified/data.jsonl
        └── sangraha/unverified/data.jsonl
```

## Commands to reproduce

```bash
git clone <repo> && cd <repo>
git checkout 987ad30458abfbc6b83154a7717d846e4bc85abc
pip install -r requirements.txt
gcloud auth application-default login
gcloud config set project lma-01

for L in hindi nepali; do
  python run_phase1.py --lang $L --stage gcs-ingest
  python run_phase1.py --lang $L --stage discover,plan
  python run_phase1.py --lang $L --stage scrape --scrape-hours 6
  python -m pipeline.collect.pdf_harvest --lang $L --out-dir ~/${L}_pdfs \
      --depth 4 --user-agent 'corpus-research/1.0 (+you@example.com)'
  python -m pipeline.collect.ocr_collect --lang $L \
      --input-dir ~/${L}_pdfs --text-layer-only
  python -m pipeline.process.build_corpus --lang $L --repo-root . --workers 7
  python run_phase1.py --lang $L --stage tokenizer,count,report
done
python -m pipeline.process.verify_corpora --repo-root .
python tools/make_reports.py --repo-root .
```

### Determinism

- Split assignment: seeded (`--seed`, default 20260820)
- MinHash permutations: seeded from the same constant
- URL shuffle in the scraper: seeded
- SentencePiece: `shuffle_input_sentence` with a fixed sample size

Collection stages are **not** deterministic — the live web changes between runs. Reproducing the exact corpus requires the collected raw JSONL, not just the code.
