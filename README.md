# Projeto 2 — Reinforcement Learning no Taxi-v4: Q-learning vs SARSA

Insper · Inteligência Artificial · 4º semestre

Treino de agentes **Q-learning** e **SARSA** no ambiente `Taxi-v4` do Gymnasium, em sua versão determinística e na versão
não determinística (`is_rainy=True, rainy_probability=0.6`), com comparação de eficiência de aprendizado e desempenho final.

**Relatório (resultados, gráficos, tabelas e discussão): [`projeto2_taxi_qlearning_sarsa.ipynb`](projeto2_taxi_qlearning_sarsa.ipynb)**
(já executado, com todas as saídas).

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `projeto2_taxi_qlearning_sarsa.ipynb` | Relatório: experimentos, gráficos, tabelas, discussão e conclusões |
| `rl_taxi.py` | Código: ambiente, Q-learning/SARSA (`train`), avaliação (`evaluate`), value iteration (ótimo exato, só como referência), busca de hiperparâmetros |
| `figures/` | Gráficos do relatório (PNG) |
| `results/` | Curvas de treino e Q-tables (`train_*.npz`), tabelas-resumo (`resumo_*.csv`) e resultados da busca de hiperparâmetros (`tuning_*.csv`) |
| `requirements.txt` | Dependências |

## Como rodar

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace projeto2_taxi_qlearning_sarsa.ipynb   # ~5 min
```

Ou abra o notebook no Jupyter/VS Code e execute todas as células. Todas as *seeds* são fixas, então os resultados são reproduzíveis.
A busca de hiperparâmetros usa resultados salvos em `results/` (`RERUN_TUNING = True` no notebook refaz a busca; leva vários minutos).

## Hiperparâmetros usados (iguais para Q-learning e SARSA)

`alpha=0.3`, `gamma=0.99`, `epsilon: 1.0 → 0.01` (decaimento 0.99 por episódio), `5000` episódios, 5 execuções de treino por algoritmo/ambiente.
