## Performance Benchmarking (TTFT, TPOT, E2E Latency)

### Optional, recommended for process memory monitoring
python3 -m pip install psutil

### Verify the runtime
llama-cli --version

python3 gguf_token_benchmark.py   --model "/llama-3.2-1B-Instruct-gguf/Llama-3.2-1B-Instruct.Q2_K.gguf"   --server-bin "$HOME/llama.cpp/build/bin/llama-server"   --runs 5   --warmup-runs 1   --n-predict 128  --ignore-eos   --ctx-size 512   --threads 4   --gpu-layers 0   --timeout 180





### Batch 

 
benchmark_results/
├── Llama-3.2-1B-Q4_K_M/
│   ├── benchmark_runs.csv
│   ├── benchmark_summary.csv
│   └── logs/
├── Llama-3.2-1B-Q8_0/
│   ├── benchmark_runs.csv
│   ├── benchmark_summary.csv
│   └── logs/
└── all_models_summary.csv


pip install llama-cpp-python psutil



python3 gguf_batch_benchmark.py \
  --models-dir "/llama-3.2-1B-Instruct-gguf" \
  --server-bin "$HOME/llama.cpp/build/bin/llama-server" \
  --output-dir "$HOME/gguf_batch_results" \
  --runs 5 --warmup-runs 1 --n-predict 128 --ignore-eos \
  --ctx-size 512 --threads 4 --gpu-layers 0 --timeout 180


If models are nested in subdirectories, add --recursive.
