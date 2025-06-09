import random
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from concurrent.futures import ProcessPoolExecutor

# ─────── Parametri ───────
STEPS = 751
POPULATIONS = [10, 100, 1000]
PAYOFF_MATRIX = [[1, 0], [0, 2]]
DECISION_RULE = "direct-positive-proportional-m"
PAYOFF_TO_USE = "play-with-one-rd-agent"
PROB_REVISION = 0.05
NOISE = 0.0
M_PARAM = 1.0

# Per il secondo plot MD-only
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
    def __init__(self, unique_id, model, strategy):
        super().__init__(unique_id, model)
        self.strategy = strategy
        self.strategy_after_revision = strategy
        self.payoff = 0

    def update_payoff(self):
        if self.model.payoff_to_use == "play-with-one-rd-agent":
            agents = self.model.schedule.agents
            mate = agents[random.randrange(len(agents))]
            if mate is self:
                mate = agents[(agents.index(self) + 1) % len(agents)]
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

    # Regole di decisione
    def imitate_if_better_rule(self):
        agents = self.model.schedule.agents
        obs = agents[random.randrange(len(agents))]
        if obs is not self and obs.payoff > self.payoff:
            self.strategy_after_revision = obs.strategy

    def imitative_pairwise_difference_rule(self):
        agents = self.model.schedule.agents
        obs = agents[random.randrange(len(agents))]
        diff = obs.payoff - self.payoff
        if obs is not self and diff > 0 and random.random() < (diff / self.model.max_payoff_difference):
            self.strategy_after_revision = obs.strategy

    def imitative_linear_attraction_rule(self):
        obs = random.choice(self.model.schedule.agents)
        denom = self.model.max_of_payoff_matrix - self.model.min_of_payoff_matrix
        if denom > 0:
            p = (obs.payoff - self.model.min_of_payoff_matrix) / denom
            if random.random() < p:
                self.strategy_after_revision = obs.strategy

    def imitative_linear_dissatisfaction_rule(self):
        obs = random.choice(self.model.schedule.agents)
        denom = self.model.max_of_payoff_matrix - self.model.min_of_payoff_matrix
        if denom > 0:
            p = (self.model.max_of_payoff_matrix - self.payoff) / denom
            if random.random() < p:
                self.strategy_after_revision = obs.strategy

    def direct_best_rule(self):
        pairs = [(s, self.model.payoff_for_strategy(s)) for s in range(self.model.n_of_strategies)]
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
        m=M_PARAM
    ):
        super().__init__()
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

        self.schedule = RandomActivation(self)
        self.history_agent = []
        self.history_md = []

        self.setup_players()
        self.strategy_expected_payoffs = [0.0] * self.n_of_strategies
        self.x_agent = self.count_strategy(1) / self.n_of_players
        self.x_md = self.x_agent
        self.update_strategy_expected_payoffs()

    def setup_players(self):
        aid = 0
        for strat, count in enumerate(self.n_of_players_for_each_strategy):
            for _ in range(count):
                p = Player(aid, self, strat)
                self.schedule.add(p)
                aid += 1

    def count_strategy(self, strategy):
        return sum(1 for a in self.schedule.agents if a.strategy == strategy)

    def update_strategy_expected_payoffs(self):
        total = len(self.schedule.agents)
        freqs = [self.count_strategy(s) / total for s in range(self.n_of_strategies)] if total else [0]*self.n_of_strategies
        self.strategy_expected_payoffs = [sum(payoff * freq for payoff, freq in zip(payoffs, freqs))
                                         for payoffs in self.payoff_matrix]

    def payoff_for_strategy(self, strategy):
        if self.payoff_to_use == "play-with-one-rd-agent":
            opp = random.choice(self.schedule.agents)
            return self.payoff_matrix[strategy][opp.strategy]
        else:
            return self.strategy_expected_payoffs[strategy]

    def step(self):
        agents = self.schedule.agents
        # ABM update
        for a in agents:
            a.update_payoff()
        for a in agents:
            if random.random() < self.prob_revision:
                a.update_strategy_after_revision()
        for a in agents:
            a.update_strategy()

        self.update_strategy_expected_payoffs()
        self.x_agent = self.count_strategy(1) / self.n_of_players

        # Mean-dynamics
        func = getattr(
            self,
            f"{self.decision_rule.replace('-', '_')}_{self.payoff_to_use.replace('-', '_')}_md",
            self.replicator_dynamics_md
        )
        F = func(self.x_md)
        self.x_md = np.clip(
            self.x_md + self.prob_revision * ((1 - self.noise) * F + self.noise * (0.5 - self.x_md)),
            0.0, 1.0
        )

        # salva storici
        self.history_agent.append(self.x_agent)
        self.history_md.append(self.x_md)
        self.schedule.step()

    # Mean-dynamics functions
    def imitate_if_better_play_with_one_rd_agent_md(self, x):
        return x * (x**3 - 4*x**2 + 4*x - 1)

    def imitate_if_better_use_strategy_expected_payoff_md(self, x):
        val = 1 if x > 1/3 else (-1 if x < 1/3 else 0)
        return x * (1 - x) * val

    def replicator_dynamics_md(self, x):
        return x * (4*x - 3*x**2 - 1) / 2

    # Alias per altre regole
    imitative_pairwise_difference_play_with_one_rd_agent_md = replicator_dynamics_md
    imitative_pairwise_difference_use_strategy_expected_payoff_md = replicator_dynamics_md
    imitative_linear_attraction_play_with_one_rd_agent_md = replicator_dynamics_md
    imitative_linear_attraction_use_strategy_expected_payoff_md = replicator_dynamics_md
    imitative_linear_dissatisfaction_play_with_one_rd_agent_md = replicator_dynamics_md
    imitative_linear_dissatisfaction_use_strategy_expected_payoff_md = replicator_dynamics_md

    def direct_best_play_with_one_rd_agent_md(self, x):
        return x * (1 - x) / 2

    def direct_best_use_strategy_expected_payoff_md(self, x):
        return (1 - x) if x > 1/3 else ((-x) if x < 1/3 else (0.5 - x))

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


# ─── helper per MD-only ───
def simulate_md(md_func, x0, prob_revision, noise, steps):
    xs = np.empty(steps+1)
    Fs = np.empty(steps+1)
    x = x0
    for t in range(steps+1):
        xs[t] = x
        Fs[t] = md_func(x)
        x = np.clip(x + prob_revision * ((1 - noise)*Fs[t] + noise*(0.5 - x)), 0.0, 1.0)
    return xs, Fs


# ─── Funzione di utility per ABM-only ───
def run_abm(N, model_kwargs):
    n0 = int(0.7 * N)
    n1 = N - n0
    m = GameModel([n0, n1], **model_kwargs)
    for _ in range(STEPS):
        m.step()
    return m.history_agent, m.history_md


# ─── Funzioni di plotting ───
def plot_abm(populations, model_kwargs):
    # creazione figura e assi
    fig, axes = plt.subplots(1, len(populations), figsize=(12, 4))
    # regolo i margini per spazio a legende e titoli
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98, wspace=0.25)

    for ax, N in zip(axes, populations):
        y_agent, y_md = run_abm(N, model_kwargs)
        t = range(len(y_agent))
        ax.fill_between(t, 0, y_agent, color="tab:green", alpha=0.6)
        ax.fill_between(t, y_agent, 1, color="tab:orange", alpha=0.6)
        ax.plot(t, y_md, "--k", lw=1.5)
        ax.set_title(f"$N={N}$", fontsize=10, pad=8)
        ax.set_xlim(0, len(y_agent)-1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Fraction $x$", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, linestyle=':', linewidth=0.5)

    # legenda generale
    handles = [plt.Line2D([], [], color='tab:green', alpha=0.6, lw=8),
               plt.Line2D([], [], color='tab:orange', alpha=0.6, lw=8),
               plt.Line2D([], [], color='black', ls='--', lw=1.5)]
    labels = ["Stratega B (300)", "Stratega A (700)", "Md (Euler)"]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Simulazioni del gioco di coordinazione 1-2", y=0.95, fontsize=12)
    return fig


def plot_md(decision_rules, payoff_modes, model_kwargs):
    fig, axes = plt.subplots(len(decision_rules), len(payoff_modes),
                             figsize=(4*len(payoff_modes), 2*len(decision_rules)))
    fig.subplots_adjust(top=0.88, hspace=0.4, wspace=0.25, left=0.08, right=0.98)

    for i, rule in enumerate(decision_rules):
        for j, pm in enumerate(payoff_modes):
            xs, _ = simulate_md(
                getattr(GameModel([700,300], **model_kwargs),
                        f"{rule.replace('-','_')}_{pm.replace('-','_')}_md",
                        GameModel.replicator_dynamics_md),
                300/1000, PROB_REVISION, NOISE, STEPS)
            ax = axes[i][j]
            ax.fill_between(range(STEPS+1), 0, xs, color="tab:green", alpha=0.6)
            ax.fill_between(range(STEPS+1), xs, 1, color="tab:orange", alpha=0.6)
            ax.plot(range(STEPS+1), xs, "--k", linewidth=1)
            ax.set_title(f"{rule}\n({pm})", fontsize=8, pad=6)
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
        'decision_rule': DECISION_RULE,
        'payoff_to_use': PAYOFF_TO_USE,
        'noise': NOISE,
        'm': M_PARAM
    }
    fig1 = plot_abm(POPULATIONS, model_kwargs)
    plt.show()
    fig2 = plot_md(DECISION_RULES, PAYOFF_MODES, model_kwargs)
    plt.show()
