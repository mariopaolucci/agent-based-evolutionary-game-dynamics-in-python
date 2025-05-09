import random
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.batchrunner import batch_run
import pandas as pd

# ─────── Parametri ───────
STEPS = 751
POPULATIONS = [10, 100, 1000]
PAYOFF_MATRIX = [[1, 0], [0, 2]]
DECISION_RULE = "imitate-if-better"
PAYOFF_TO_USE = "play-with-one-rd-agent"
PROB_REVISION = 0.05
NOISE = 0.0
M_PARAM = 1.0

# Per il secondo plot 
DECISION_RULES = [
    "imitate-if-better",
    "imitative-pairwise-difference",
    "direct-best",
    "direct-pairwise-difference",
    "direct-positive-proportional-m"
]
PAYOFF_MODES = ["play-with-one-rd-agent", "use-strategy-expected-payoff"]

# ─── CLASSE AGENTE ───
class Player(Agent):
    def __init__(self, model, strategy):
        super().__init__(model)
        self.strategy = strategy
        self.strategy_after_revision = strategy
        self.payoff = 0

    def update_payoff(self):
        if self.model.payoff_to_use == "play-with-one-rd-agent":
            others = [a for a in self.model.agents if a is not self]
            mate = random.choice(others)
            self.payoff = self.model.payoff_matrix[self.strategy][mate.strategy]
        else:
            self.payoff = self.model.strategy_expected_payoffs[self.strategy]

    def update_strategy_after_revision(self):
        if random.random() < self.model.noise:
            self.strategy_after_revision = random.randint(0, self.model.n_of_strategies - 1)
        else:
            rule = getattr(self, f"{self.model.decision_rule.replace('-', '_')}_rule", None)
            if rule:
                rule()

    # Decision rules
    def imitate_if_better_rule(self):
        obs = random.choice([a for a in self.model.agents if a is not self])
        if obs.payoff > self.payoff:
            self.strategy_after_revision = obs.strategy

    def imitative_pairwise_difference_rule(self):
        obs = random.choice([a for a in self.model.agents if a is not self])
        diff = obs.payoff - self.payoff
        if diff > 0 and random.random() < (diff / self.model.max_payoff_difference):
            self.strategy_after_revision = obs.strategy

    def imitative_linear_attraction_rule(self):
        obs = random.choice([a for a in self.model.agents if a is not self])
        denom = self.model.max_of_payoff_matrix - self.model.min_of_payoff_matrix
        if denom > 0:
            p = (obs.payoff - self.model.min_of_payoff_matrix) / denom
            if random.random() < p:
                self.strategy_after_revision = obs.strategy

    def imitative_linear_dissatisfaction_rule(self):
        obs = random.choice([a for a in self.model.agents if a is not self])
        denom = self.model.max_of_payoff_matrix - self.model.min_of_payoff_matrix
        if denom > 0:
            p = (self.model.max_of_payoff_matrix - self.payoff) / denom
            if random.random() < p:
                self.strategy_after_revision = obs.strategy

    def direct_best_rule(self):
        pairs = [(s, self.model.payoff_for_strategy(s)) for s in range(self.model.n_of_strategies)]
        random.shuffle(pairs)
        best = max(pairs, key=lambda x: x[1])
        self.strategy_after_revision = best[0]

    def direct_pairwise_difference_rule(self):
        candidates = [s for s in range(self.model.n_of_strategies) if s != self.strategy]
        c = random.choice(candidates)
        diff = self.model.payoff_for_strategy(c) - self.payoff
        if diff > 0 and random.random() < (diff / self.model.max_payoff_difference):
            self.strategy_after_revision = c

    def direct_positive_proportional_m_rule(self):
        pairs = [(s, self.model.payoff_for_strategy(s) ** self.model.m)
                 for s in range(self.model.n_of_strategies)]
        tot = sum(p[1] for p in pairs)
        if tot == 0:
            self.strategy_after_revision = random.randint(0, self.model.n_of_strategies - 1)
        else:
            weights = [p[1] / tot for p in pairs]
            self.strategy_after_revision = random.choices([p[0] for p in pairs], weights=weights)[0]

    def update_strategy(self):
        self.strategy = self.strategy_after_revision

# ─── CLASSE MODELLO ───
class GameModel(Model):
    def __init__(
        self,
        n_of_players_for_each_strategy=[700, 300],
        payoff_matrix=PAYOFF_MATRIX,
        prob_revision=PROB_REVISION,
        decision_rule=DECISION_RULE,
        payoff_to_use=PAYOFF_TO_USE,
        noise=NOISE,
        m=M_PARAM,
        seed=None
    ):
        super().__init__(seed=seed)
        self.payoff_matrix = payoff_matrix
        self.n_of_strategies = len(payoff_matrix)
        self.n_of_players_for_each_strategy = n_of_players_for_each_strategy
        self.n_of_players = sum(n_of_players_for_each_strategy)
        self.prob_revision = prob_revision
        self.decision_rule = decision_rule
        self.payoff_to_use = payoff_to_use
        self.noise = noise
        self.m = m

        # Pre-calcoli
        self.min_of_payoff_matrix = min(min(r) for r in payoff_matrix)
        self.max_of_payoff_matrix = max(max(r) for r in payoff_matrix)
        self.max_payoff_difference = self.max_of_payoff_matrix - self.min_of_payoff_matrix

        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={"x_agent": lambda m: m.x_agent}
        )

        # Creazione agenti
        self.setup_players()
        self.strategy_expected_payoffs = [0.0] * self.n_of_strategies
        self.x_agent = self.count_strategy(1) / self.n_of_players
        self.update_strategy_expected_payoffs()
        self.datacollector.collect(self)

    def setup_players(self):
        for strat, count in enumerate(self.n_of_players_for_each_strategy):
            for _ in range(count):
                Player(self, strat)

    def count_strategy(self, strategy):
        return sum(1 for a in self.agents if a.strategy == strategy)

    def update_strategy_expected_payoffs(self):
        total = len(self.agents)
        freqs = [self.count_strategy(s)/total for s in range(self.n_of_strategies)] if total else [0]*self.n_of_strategies
        self.strategy_expected_payoffs = [
            sum(payoff*freq for payoff, freq in zip(payoffs, freqs))
            for payoffs in self.payoff_matrix
        ]

    # Mean-dynamics 
    def imitate_if_better_play_with_one_rd_agent_md(self, x):
        return x * (x**3 - 4*x**2 + 4*x - 1)

    def imitate_if_better_use_strategy_expected_payoff_md(self, x):
        val = 1 if x > 1/3 else (-1 if x < 1/3 else 0)
        return x * (1 - x) * val

    def replicator_dynamics_md(self, x):
        return x * (4*x - 3*x**2 - 1) / 2

    # Aliases MD
    imitative_pairwise_difference_play_with_one_rd_agent_md = replicator_dynamics_md
    imitative_pairwise_difference_use_strategy_expected_payoff_md = replicator_dynamics_md
    imitative_linear_attraction_play_with_one_rd_agent_md = replicator_dynamics_md
    imitative_linear_attraction_use_strategy_expected_payoff_md = replicator_dynamics_md
    imitative_linear_dissatisfaction_play_with_one_rd_agent_md = replicator_dynamics_md
    imitative_linear_dissatisfaction_use_strategy_expected_payoff_md = replicator_dynamics_md

    def direct_best_play_with_one_rd_agent_md(self, x):
        return x * (1 - x) / 2

    def direct_best_use_strategy_expected_payoff_md(self, x):
        return (1 - x) if x > 1/3 else (-x) if x < 1/3 else (0.5 - x)

    def direct_pairwise_difference_play_with_one_rd_agent_md(self, x):
        return (1 - x) * x**2

    def direct_pairwise_difference_use_strategy_expected_payoff_md(self, x):
        return 0.5*(3*x-1)*(1-x) if x >= 1/3 else 0.5*(3*x-1)*x

    def direct_positive_proportional_m_play_with_one_rd_agent_md(self, x):
        return 0.5*(2**self.m - 1)*(x*(1-x))/(2**self.m + 1)

    def direct_positive_proportional_m_use_strategy_expected_payoff_md(self, x):
        num = (1 - x)**self.m
        den = num + (2**self.m)*(x**self.m)
        return (1 - x) - num/den

    def payoff_for_strategy(self, strategy):
        if self.payoff_to_use == "play-with-one-rd-agent":
            others = [a for a in self.agents]
            opp = random.choice(others)
            return self.payoff_matrix[strategy][opp.strategy]
        else:
            return self.strategy_expected_payoffs[strategy]

    def step(self):
        agents = list(self.agents)
        # 1) aggiorna payoff
        for a in agents:
            a.update_payoff()
        # 2) revisione strategia
        for a in agents:
            if random.random() < self.prob_revision:
                a.update_strategy_after_revision()
        # 3) applica nuova strategia
        for a in agents:
            a.update_strategy()

        # aggiorna payoffs attesi e frazione x
        self.update_strategy_expected_payoffs()
        self.x_agent = self.count_strategy(1)/self.n_of_players

        # raccogli dati
        self.datacollector.collect(self)

# ----funzione md per la simulazione----
def simulate_md(md_func, x0, prob_revision, noise, steps):
    xs = np.empty(steps+1)
    Fs = np.empty(steps+1)
    x = x0
    for t in range(steps+1):
        xs[t] = x
        Fs[t] = md_func(x)
        x = np.clip(x + prob_revision*((1 - noise)*Fs[t] + noise*(0.5 - x)), 0.0, 1.0)
    return xs, Fs

# ─── helper batch ───
def batch_run_abm(populations, model_kwargs, runs_per_N=4):
    results = {}
    for N in populations:
        n0 = int(0.7 * N)
        params = {**model_kwargs, 'n_of_players_for_each_strategy': [n0, N-n0]}
        parameters = {k: [v] if not isinstance(v, list) else [v] for k, v in params.items()}
        parameters['n_of_players_for_each_strategy'] = [[n0, N-n0]]
        data = batch_run(GameModel, 
                         parameters=parameters,
                         iterations=runs_per_N, 
                         max_steps=STEPS,
                         number_processes= None,
                         data_collection_period=1, 
                         display_progress=False)
        results[N] = pd.DataFrame(data)
    return results

# Usiamo batch_run_abm per entrambi i plot
# ─── plot_abm ───
def plot_abm(populations, model_kwargs, runs_per_N=4):
    batch_results = batch_run_abm(populations, model_kwargs, runs_per_N)
    fig, axes = plt.subplots(1, len(populations), figsize=(12, 4))
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98, wspace=0.25)
    for ax, N in zip(axes, populations):
        # Calcola media ABM
        df = batch_results[N]
        df_grouped = df.groupby('Step')['x_agent'].mean()
        t = df_grouped.index
        xs_abm = df_grouped.values
        # Calcola curva MD analitica con simulate_md
        x0 = df_grouped.iloc[0]
        temp_model = GameModel([int(0.7*N), N-int(0.7*N)], **model_kwargs)
        md_func = getattr(temp_model, f"{model_kwargs.get('decision_rule', DECISION_RULE).replace('-', '_')}_{model_kwargs.get('payoff_to_use', PAYOFF_TO_USE).replace('-', '_')}_md", temp_model.replicator_dynamics_md)
        xs_md, _ = simulate_md(md_func, x0, model_kwargs['prob_revision'], model_kwargs['noise'], STEPS)

        # Plot
        ax.fill_between(t, 0, xs_abm, color='tab:green', alpha=0.6)
        ax.fill_between(t, xs_abm, 1, color='tab:orange', alpha=0.6)
        ax.plot(range(len(xs_md)), xs_md, '--k', lw=1.5, label='MD (Euler)')
        ax.set_title(f"$N={N}$", fontsize=10, pad=8)
        ax.set_xlim(0, STEPS)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel('Fraction $x$', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, linestyle=':', linewidth=0.5)
    handles = [plt.Line2D([], [], color='tab:green', alpha=0.6, lw=8),
               plt.Line2D([], [], color='tab:orange', alpha=0.6, lw=8),
               plt.Line2D([], [], color='black', ls='--', lw=1.5)]
    labels = ['Stratega B (300)', 'Stratega A (700)', 'Md (Euler)']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Simulazioni del gioco di coordinazione 1-2', y=0.95, fontsize=12)
    return fig

# ─── secondo plot con regole decisionali e payoff ───
def plot_md(decision_rules, payoff_modes, model_kwargs, runs_per_setting=4):
    """
    Plot combinato: curva MD analitica (linea tratteggiata) e ABM per ogni regola e payoff_mode.
    """
    fig, axes = plt.subplots(len(decision_rules), len(payoff_modes),
                             figsize=(4*len(payoff_modes), 2*len(decision_rules)))
    for i, rule in enumerate(decision_rules):
        for j, pm in enumerate(payoff_modes):
            # Configura modello per estrarre la funzione MD analitica
            temp_model = GameModel([700,300], **{**model_kwargs, 'decision_rule':rule, 'payoff_to_use':pm})
            md_func = getattr(temp_model, f"{rule.replace('-','_')}_{pm.replace('-','_')}_md", temp_model.replicator_dynamics_md)
            # Calcola MD analitica
            xs_md, _ = simulate_md(md_func, 300/1000, PROB_REVISION, NOISE, STEPS)
            # Esegue ABM batch e media
            params = {**model_kwargs, 'decision_rule':rule, 'payoff_to_use':pm, 'n_of_players_for_each_strategy':[700,300]}
            parameters = {k:[v] for k,v in params.items()}
            data = batch_run(GameModel, parameters=parameters,
                             iterations=runs_per_setting,
                             max_steps=STEPS,
                             data_collection_period=1,
                             display_progress=False)
            df = pd.DataFrame(data)
            xs_abm = df.groupby('Step')['x_agent'].mean().values
            t = np.arange(len(xs_abm))

            ax = axes[i][j]
            # ABM area
            ax.fill_between(t, 0, xs_abm, color="tab:green", alpha=0.6)
            ax.fill_between(t, xs_abm, 1, color="tab:orange", alpha=0.6)
            # MD analytic
            ax.plot(range(len(xs_md)), xs_md, "--k", linewidth=1)
            ax.set_title(f"{rule}({pm})", fontsize=8)
            ax.set_xlim(0, STEPS)
            ax.set_ylim(0, 1)
            if i == len(decision_rules)-1:
                ax.set_xlabel("Step", fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel("Fraction $x$", fontsize=8)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, linestyle=':', linewidth=0.5)
    handles = [plt.Line2D([], [], color='tab:green', alpha=0.6, lw=8),
               plt.Line2D([], [], color='tab:orange', alpha=0.6, lw=8),
               plt.Line2D([], [], color='black', ls='--', lw=1)]
    labels = ["Stratega B (300)", "Stratega A (700)", "Md (Euler)"]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.00),
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Strategy Distribution & Mean Dynamics", y=0.94, fontsize=14)
    return fig

# ─── MAIN ───
if __name__ == "__main__":
    model_kwargs = {
        'payoff_matrix': PAYOFF_MATRIX,
        'prob_revision': PROB_REVISION,
        'noise': NOISE,
        'm': M_PARAM
    }
    # primo grafico
    fig1 = plot_abm(POPULATIONS, model_kwargs)
    plt.show()
    # secondo grafico 
    fig2 = plot_md(DECISION_RULES, PAYOFF_MODES, model_kwargs)
    plt.show()
