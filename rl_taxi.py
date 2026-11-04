"""
Funções de apoio do Projeto 2 (Q-learning vs SARSA no Taxi-v4).

Tudo que é "lógica" fica aqui; o notebook só chama estas funções, roda os
experimentos e mostra gráficos/tabelas/discussão.

Conteúdo
--------
make_env            : cria o ambiente Taxi-v4 (determinístico ou chuvoso)
epsilon_greedy      : política de exploração (com desempate aleatório)
train               : treina um agente com Q-learning OU SARSA
evaluate            : executa a política gulosa (greedy) da Q-table N vezes
value_iteration     : política ótima exata (usa o modelo P do ambiente) -> só
                      como REFERÊNCIA/meta, os agentes não têm acesso a isso
run_many            : treina várias execuções (seeds) em paralelo
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict

import gymnasium as gym
import numpy as np

N_STATES = 500
N_ACTIONS = 6
ACTION_NAMES = ["Sul", "Norte", "Leste", "Oeste", "Pegar", "Deixar"]

# Parâmetros do ambiente não determinístico pedido no enunciado
RAINY_KWARGS = dict(is_rainy=True, rainy_probability=0.6)
DET_KWARGS = dict()


# --------------------------------------------------------------------------- #
# Ambiente
# --------------------------------------------------------------------------- #
def make_env(rainy: bool = False, **extra) -> gym.Env:
    """Taxi-v4. rainy=True -> is_rainy=True e rainy_probability=0.6."""
    kwargs = dict(RAINY_KWARGS if rainy else DET_KWARGS)
    kwargs.update(extra)
    return gym.make("Taxi-v4", disable_env_checker=True, **kwargs)


# --------------------------------------------------------------------------- #
# Hiperparâmetros
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HParams:
    n_episodes: int = 5000
    alpha: float = 0.1        # taxa de aprendizado inicial
    alpha_min: float = 0.0    # piso da taxa de aprendizado
    alpha_decay: float = 1.0  # alpha <- max(alpha_min, alpha * alpha_decay) a cada episódio (1.0 = constante)
    gamma: float = 0.99       # fator de desconto
    eps_start: float = 1.0    # exploração inicial (100% aleatório)
    eps_min: float = 0.01     # exploração mínima
    eps_decay: float = 0.995  # eps <- max(eps_min, eps * eps_decay) a cada episódio


# --------------------------------------------------------------------------- #
# Política epsilon-greedy
# --------------------------------------------------------------------------- #
def epsilon_greedy(Q: np.ndarray, s: int, eps: float, rng: np.random.Generator) -> int:
    """Com prob. eps age aleatoriamente; senão pega a melhor ação.

    Desempate aleatório: no início a Q-table é toda zero, e um argmax simples
    sempre devolveria a ação 0 (Sul), enviesando a exploração.
    """
    if rng.random() < eps:
        return int(rng.integers(N_ACTIONS))
    q = Q[s]
    best = int(q.argmax())
    ties = np.flatnonzero(q == q[best])
    if len(ties) > 1:
        return int(rng.choice(ties))
    return best


# --------------------------------------------------------------------------- #
# Treino (Q-learning e SARSA)
# --------------------------------------------------------------------------- #
def train(
    algo: str,
    hp: HParams,
    seed: int,
    rainy: bool = False,
) -> dict:
    """Treina um agente.

    algo : "qlearning" ou "sarsa"

    Regra de atualização (a ÚNICA diferença entre os dois é o alvo):
        Q-learning (off-policy):  alvo = r + gamma * max_a' Q(s', a')
        SARSA      (on-policy) :  alvo = r + gamma * Q(s', a')   com a' escolhida
                                  pela própria política epsilon-greedy
        Q(s,a) <- Q(s,a) + alpha * (alvo - Q(s,a))

    alpha e epsilon decaem a cada episódio (ver HParams).

    Se o episódio TERMINOU (passageiro entregue), não há futuro: alvo = r.
    Se foi apenas cortado pelo limite de 200 passos (truncation), o futuro
    existe, então continuamos "bootstrapando" normalmente.

    Retorna Q-table, reward acumulado por episódio e nº de passos por episódio
    (ambos durante o treino, com exploração).
    """
    if algo not in ("qlearning", "sarsa"):
        raise ValueError(algo)

    env = make_env(rainy)
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))

    ep_rewards = np.zeros(hp.n_episodes)
    ep_steps = np.zeros(hp.n_episodes, dtype=np.int32)
    eps = hp.eps_start
    alpha = hp.alpha

    for ep in range(hp.n_episodes):
        # seed só no 1º reset: o gerador do ambiente segue sozinho depois
        s, _ = env.reset(seed=seed) if ep == 0 else env.reset()
        a = epsilon_greedy(Q, s, eps, rng)
        total, steps, done = 0.0, 0, False

        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            a2 = epsilon_greedy(Q, s2, eps, rng)  # próxima ação (usada por SARSA e na sequência)

            if terminated:
                target = r
            elif algo == "qlearning":
                target = r + hp.gamma * Q[s2].max()
            else:  # sarsa
                target = r + hp.gamma * Q[s2, a2]

            Q[s, a] += alpha * (target - Q[s, a])

            s, a = s2, a2
            total += r
            steps += 1
            done = terminated or truncated

        ep_rewards[ep] = total
        ep_steps[ep] = steps
        eps = max(hp.eps_min, eps * hp.eps_decay)
        alpha = max(hp.alpha_min, alpha * hp.alpha_decay)

    env.close()
    return dict(Q=Q, rewards=ep_rewards, steps=ep_steps, algo=algo, seed=seed, rainy=rainy)


def _train_job(args):
    algo, hp, seed, rainy = args
    return train(algo, hp, seed, rainy)


def run_many(algo: str, hp: HParams, seeds, rainy: bool = False, workers: int | None = None):
    """Treina uma execução por seed (em paralelo) e devolve a lista de resultados."""
    jobs = [(algo, hp, s, rainy) for s in seeds]
    if workers == 1 or len(jobs) == 1:
        return [_train_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_train_job, jobs))


# --------------------------------------------------------------------------- #
# Avaliação (política gulosa, sem exploração e sem aprender)
# --------------------------------------------------------------------------- #
def run_policy(policy_fn, rainy: bool, n_episodes: int = 100, seed: int = 0) -> dict:
    """Executa policy_fn(estado)->ação por n_episodes episódios.

    Episódio i usa env.reset(seed=seed+i): para o mesmo `seed`, todos os agentes
    começam exatamente dos mesmos estados iniciais (comparação justa/pareada).
    """
    env = make_env(rainy)
    steps = np.zeros(n_episodes, dtype=np.int32)
    rewards = np.zeros(n_episodes)
    success = np.zeros(n_episodes, dtype=bool)
    for i in range(n_episodes):
        s, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            s, r, terminated, truncated, _ = env.step(policy_fn(s))
            rewards[i] += r
            steps[i] += 1
            done = terminated or truncated
        success[i] = terminated  # terminated=True só quando entregou certo
    env.close()
    return dict(steps=steps, rewards=rewards, success=success)


def evaluate(Q: np.ndarray, rainy: bool, n_episodes: int = 100, seed: int = 0) -> dict:
    """Executa a política gulosa da Q-table (argmax) n_episodes vezes."""
    greedy = Q.argmax(axis=1)
    return run_policy(lambda s: int(greedy[s]), rainy, n_episodes, seed)


# --------------------------------------------------------------------------- #
# Referência: política ótima exata (Value Iteration)
# --------------------------------------------------------------------------- #
def value_iteration(rainy: bool = False, gamma: float = 1.0, tol: float = 1e-10):
    """Resolve o MDP exatamente usando o modelo de transição P do gymnasium.

    NÃO é usado para treinar os agentes (eles aprendem só por tentativa e erro);
    serve para saber qual é o reward ótimo (a "meta" dos gráficos).

    Devolve (V, Q*, retorno ótimo esperado a partir da distribuição inicial).
    """
    env = make_env(rainy)
    base = env.unwrapped
    P = base.P
    V = np.zeros(N_STATES)
    Qs = np.zeros((N_STATES, N_ACTIONS))
    while True:
        for s in range(N_STATES):
            for a in range(N_ACTIONS):
                Qs[s, a] = sum(p * (r + gamma * V[s2] * (not t)) for p, s2, r, t in P[s][a])
        V_new = Qs.max(axis=1)
        if np.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new
    mean_return = float(base.initial_state_distrib @ V)
    env.close()
    return V, Qs, mean_return


def hp_dict(hp: HParams) -> dict:
    return asdict(hp)


# --------------------------------------------------------------------------- #
# Métricas de aprendizado + busca de hiperparâmetros
# --------------------------------------------------------------------------- #
def moving_average(x, w: int = 100) -> np.ndarray:
    """Média móvel de janela w (valor i = média dos episódios i-w+1..i)."""
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return (c[w:] - c[:-w]) / w


def episodes_to_goal(rewards, goal: float, w: int = 100) -> float:
    """1º episódio em que a média móvel (janela w) do reward atinge `goal`.
    NaN se nunca atingir."""
    hit = np.flatnonzero(moving_average(rewards, w) >= goal)
    return float(hit[0] + w) if len(hit) else float("nan")


def _tune_job(args):
    algo, hp, seed, rainy, goal, n_eval = args
    r = train(algo, hp, seed, rainy)
    ev = evaluate(r["Q"], rainy, n_eval, seed=20_000)
    row = asdict(hp)
    row.update(
        algo=algo, seed=seed, rainy=rainy,
        ep_to_goal=episodes_to_goal(r["rewards"], goal),
        reward_first500=r["rewards"][:500].mean(),
        reward_last500=r["rewards"][-500:].mean(),
        eval_reward=ev["rewards"].mean(),
        eval_steps=ev["steps"].mean(),
        eval_success=ev["success"].mean(),
    )
    return row


def grid_search(hps, algos, seeds, rainy: bool, goal: float, n_eval: int = 1000):
    """Treina todas as combinações (hps x algos x seeds) em paralelo e devolve
    um DataFrame com uma linha por treino."""
    import pandas as pd

    jobs = [(a, hp, s, rainy, goal, n_eval) for hp in hps for a in algos for s in seeds]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(_tune_job, jobs))
    return pd.DataFrame(rows)
