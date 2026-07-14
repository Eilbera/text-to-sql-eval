A eval harness measuring how well an LLM trasnaltes English questions into SQL, using the BIRD mini-dev benchmark (~500 questions over 11 real SQLite databases).



**Metric: execution accuracy**

The model's query and gold quert are both executed against the real database; the model scores only if the results match. Comparing results instead of SQL text matters because many different queries are correct.



**Baseline result:** 25.4%  - qwen3.5:27b (Q4_K_M via Ollama), zero-shot, temperature 0



## Run it

1. Download BIRD mini-dev from [https://bird-bench.github.io](https://bird-bench.github.io) and update
     `DB_DIRQUESTIONS` in `eval_baseline.py` 
2. Have Ollama running with 'qwen3.5:27b' pulled
3. `pip install -r requirements.txt && python3 eval_baseline.py`



## Known limitations 

- Ignores BIRD's `evidence` hints (official eval includes them)

- Result comparison is order-sensitive

- Single generation per question



## Next 

Fine-tune a small model (SFT, then GRPO with excution-match as the reward) and re-run this eval as the treatment measurement. 



