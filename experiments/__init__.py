"""Faun ML experiments: inference / linear probe / k-NN benchmarks.

Каждый эксперимент — модуль exp_e<N>.py с функцией run(cfg) -> dict.
Запуск: python -m experiments.runner E1 E3  |  --all
Результат — строка в results.csv (model,dataset,metric,value,runtime_s,vram_mb,notes).
"""
