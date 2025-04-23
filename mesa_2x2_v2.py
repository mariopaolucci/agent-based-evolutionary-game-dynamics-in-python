
# IMPORTIAMO TUTTE LE LIBRERIE NECESSARIE PER FAR FUNZIONARE IL MODELLO 
import numpy as np                              # NUMPY: per le operazioni matematiche
import random                                   # RANDOM: per generare numeri casuali
import matplotlib.pyplot as plt                 # MATPLOTLIB: per la visualizzazione dei dati
from matplotlib.animation import FuncAnimation  # ANIMAZIONE: per animare il modello
from mesa import Model                          # MESA: per la creazione del modello
from mesa.space import MultiGrid                # MESA: per la creazione della griglia



# ——— Parametri del modello ———             #  qui andiamo a definire i parametri BASE del modello
WIDTH = 80                                  # LARGHEZZA DELLA GRIGLIA
HEIGHT = 80                                 # ALTEZZA DELLA GRIGLIA
STEPS = 100                                 # NUMERO DI PASSI DELLA SIMULAZIONE
INITIAL_C_PERCENT = 0.9                     # PERCENTUALE INIZIALE DI COOPERANTI

# ——— Matrice dei pagamenti ———             # MATRICE DEI PAYOFF DELLA STRATEGIA
PAYOFF_CC = 1.0                             # PAYOFF PER COOPERANTE VS COOPERANTE
PAYOFF_CD = 0.0                             # PAYOFF PER COOPERANTE VS DEFEZIONISTA
PAYOFF_DC = 1.7                             # PAYOFF PER DEFEZIONISTA VS COOPERANTE
PAYOFF_DD = 0.0                             # PAYOFF PER DEFEZIONISTA VS DEFEZIONISTA

payoff_matrix = np.array([                  # QUI DEFINIAMO L'ARRAY DELLA MATRICE DEFINITA POCO SOPRA
    [PAYOFF_CC, PAYOFF_CD],                 # CREANDO LA GRIGLIA 2X2, DA CUI PRENDE NOME IL MODELLO 2X2
    [PAYOFF_DC, PAYOFF_DD]
])

class CustomRandomActivation:               # CLASSE PER LA GESTIONE DELLA SCHEDULAZIONE DEGLI AGENTI, comportamento e ordine degli agenti
    def __init__(self, model):              # init ci serve per inizializzare gli agenti. Init è il costruttore della libreria mesa
        self.model = model                  #   
        self.agents = []                    # self.model SERVE A SALVARE TUTTI GLI AGENTI DELLA SIMULAZIONE.
                                            # Inizializza la scheduler personalizzata, memorizzando il riferimento al modello
                                            # e una lista di agenti da gestire nella simulazione.
    def add(self, agent):                   # Aggiunge un agente alla lista degli agenti da gestire nella simulazione.
        self.agents.append(agent)           # Append inserisce nella lista i nuovi agenti nella simulazione

    def step(self):                                                     # questo è il cuore della scheduler e garantisce l'aggiornamento sincrono delle strategie degli agenti
        for agent in random.sample(self.agents, k=len(self.agents)):    # 1. Tutti gli agenti vengono attivati in ordine casuale per calcolare il proprio payoff (metodo step). 
            agent.step()                                                # 2. Tutti gli agenti aggiornano la loro strategia in base ai payoff calcolati (metodo advance).
        for agent in self.agents:                                       # Questo schema garantisce un aggiornamento sincrono.   
            agent.advance()

class ImitateAgent:                                                                          # CLASSE PER LA CREAZIONE DEGLI AGENTI
    def __init__(self, unique_id, model):                                                    # INIZIALIZZA L'AGENTE CON UN ID UNICO PER ''RICONOSCERLO'' E IL MODELLO A CUI APPARTIENE
        self.unique_id = unique_id                                                           # ID UNICO DELL'AGENTE
        self.model = model                                                                   # RIFERIMENTO AL MODELLO A CUI APPARTIENE
        self.strategy = 0 if random.random() < INITIAL_C_PERCENT else 1                      # STRATEGIA INIZIALE DELL'AGENTE (0 = cooperante, 1 = defezionista) + LA % DI C INIZIALI
        self.payoff = 0.0                                                                    # PAYOFF DELL'AGENTE INIZIALMENTE IMPOSTATO A 0
        self.pos = None                                                                      # POSIZIONE DELL'AGENTE NELLA GRIGLIA CHE Verrà assegnata quando l’agente viene piazzato sulla griglia

    def step(self):                                                                                     # CALCOLA IL PAYOFF DELL'AGENTE IN BASE ALLE STRATEGIE DEGLI AGENTI VICINI                  
        neighborhood = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True)         # Ottiene gli agenti vicini all'agente corrente (incluso se stesso) utilizzando la griglia
        total = 0.0                                                                                     # Inizializza il totale del payoff a 0
        for other in neighborhood:                                                                      # Itera attraverso gli agenti vicini con il ciclo for
            total += self.model.payoff_matrix[self.strategy, other.strategy]                            # Aggiorna il totale del payoff in base alla matrice dei payoff
        self.payoff = total                                                                             # Calcola il payoff totale dell'agente corrente in base alla matrice dei payoff e alle strategie degli agenti vicini

                                                                                                        # IN QUESTO BLOCCO IN SINTESI L'AGENTE DECIDE COSA FARE IN BASE A CIÒ CHE OSSERVA NELLO STEP PRECEDENTE
    def advance(self):                                                                                  # # AGGIORNA LA STRATEGIA DELL'AGENTE IN BASE AL PAYOFF DEGLI AGENTI VICINI
        neighborhood = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True)         # Ottiene gli agenti vicini all'agente corrente (incluso se stesso) utilizzando la griglia
        max_payoff = max(agent.payoff for agent in neighborhood)                                        # qui cerchiamo il payoff massimo per vedere chi ha guadagnato di più
        best = [agent for agent in neighborhood if agent.payoff == max_payoff]                          # raccoglie tutti i payoff massimi in una lista
        chosen = random.choice(best)                                                                    # sceglie casualmente uno degli agenti con il payoff massimo
        self.strategy = chosen.strategy                                                                 # aggiorna la strategia dell'agente corrente in base alla strategia dell'agente scelto

class ImitateBestModel(Model):                                                                          #  CLASSE PER LA CREAZIONE DEL MODELLO
    def __init__(self, width, height):                                                                  #  INIZIALIZZA IL MODELLO CON LARGHEZZA E ALTEZZA DELLA GRIGLIA
        super().__init__()                                                                              #  INIZIALIZZA IL MODELLO con il costruttore della libreria mesa
        self.grid = MultiGrid(width, height, torus=False)                                               #  CREAZIONE DELLA GRIGLIA MULTIAGENTE
        self.schedule = CustomRandomActivation(self)                                                    #  CREAZIONE DELLA SCHEDULER PERSONALIZZATA
        self.payoff_matrix = payoff_matrix                                                              #  MATRICE DEI PAYOFF DEFINITA SOPRA

        uid = 0                                                                                         # ID UNICO PER OGNI AGENTE
        for x in range(width):                                                                          # CICLO PER CREARE GLI AGENTI E POSIZIONARLI NELLA GRIGLIA
            for y in range(height):                                                                     # iteriamo su lunghezza e larghezza della griglia
                agent = ImitateAgent(uid, self)                                                         # CREIAMO UN NUOVO AGENTE CON ID UNICO E RIFERIMENTO AL MODELLO
                uid += 1                                                                                # incrementiamo l'ID unico per il prossimo agente
                self.grid.place_agent(agent, (x, y))                                                    # posizioniamo l'agente nella griglia
                agent.pos = (x, y)                                                                      # assegniamo la posizione all'agente
                self.schedule.add(agent)                                                                # aggiungiamo l'agente alla scheduler personalizzata

    def step(self):                                                                                     # ESEGUE UN PASSO DEL MODELLO
        self.schedule.step()                                                                            #  

    def get_grid_array(self):                                                                           # RESTITUISCE UNA MATRICE CHE RAPPRESENTA LA DISTRIBUZIONE DELLE STRATEGIE NELLA GRIGLIA
        grid_array = np.zeros((WIDTH, HEIGHT))                                                          # INIZIALIZZA UNA MATRICE DI ZERI DELLA DIMENSIONE DELLA GRIGLIA
        for x in range(WIDTH):                                                                          # CICLO PER POPOLARE LA MATRICE CON LE STRATEGIE DEGLI AGENTI
            for y in range(HEIGHT):                                                                     # ITERIAMO SU LARGHEZZA E ALTEZZA DELLA GRIGLIA
                agent = self.grid.get_cell_list_contents([(x, y)])                                      # OTTENIAMO GLI AGENTI NELLA POSIZIONE (X, Y)
                if agent:                                                                               # SE CI SONO AGENTI NELLA POSIZIONE (X, Y)
                    grid_array[x, y] = agent[0].strategy                                                # ASSEGNA LA STRATEGIA DELL'AGENTE ALLA MATRICE
        return grid_array                                                                               # RESTITUISCE LA MATRICE POPOLATA CON LE STRATEGIE DEGLI AGENTI

    def count_strategies(self):                                                                         # RESTITUISCE IL NUMERO E LA PERCENTUALE DI COOPERANTI E DEFEZIONISTI
        num_C = sum(1 for agent in self.schedule.agents if agent.strategy == 0)                         # CONTEGGIO DEL NUMERO DI COOPERANTI
        num_D = sum(1 for agent in self.schedule.agents if agent.strategy == 1)                         # CONTEGGIO DEL NUMERO DI DEFEZIONISTI
        total = len(self.schedule.agents)                                                               # CONTEGGIO DEL NUMERO TOTALE DI AGENTI
        return num_C, num_D, 100 * num_C / total, 100 * num_D / total                                   # RESTITUISCE IL NUMERO E LA PERCENTUALE DI COOPERANTI E DEFEZIONISTI CON UN SEMPLICE CALCOLO MATEMATICO 

# ——— Animazione ———
def update(frame):                                                                                      # FUNZIONE PER AGGIORNARE L'ANIMAZIONE
    model.step()                                                                                        # ESEGUE UN PASSO DEL MODELLO
    grid_array = model.get_grid_array()                                                                 # OTTIENE LA MATRICE DELLA DISTRIBUZIONE DELLE STRATEGIE
    im.set_array(grid_array.T)                                                                          # AGGIORNA L'IMMAGINE CON LA MATRICE DELLA DISTRIBUZIONE DELLE STRATEGIE
    ax.set_title(f"Distribuzione delle Strategie - Passo {frame + 1}")                                  # AGGIORNA IL TITOLO DELL'IMMAGINE CON IL NUMERO DEL PASSO

    num_C, num_D, perc_C, perc_D = model.count_strategies()                                             # CONTEGGIA IL NUMERO E LA PERCENTUALE DI COOPERANTI E DEFEZIONISTI

    counter_C.set_text(f"Cooperanti: {num_C} ({perc_C:.1f}%)")                                          # AGGIORNA IL TESTO DEL COUNTER DEI COOPERANTI
    counter_D.set_text(f"Defezionisti: {num_D} ({perc_D:.1f}%)")                                        # AGGIORNA IL TESTO DEL COUNTER DEI DEFEZIONISTI

    x_data.append(frame)                                                  # AGGIUNGE IL NUMERO DEL PASSO ALLA LISTA DEI DATI X                              
    y_data_C.append(perc_C)                                               # AGGIUNGE LA PERCENTUALE DEI COOPERANTI ALLA LISTA DEI DATI Y
    y_data_D.append(perc_D)                                               # AGGIUNGE LA PERCENTUALE DEI DEFEZIONISTI ALLA LISTA DEI DATI Y

    line_C.set_data(x_data, y_data_C)                                     # AGGIORNA I DATI DELLA LINEA DEI COOPERANTI
    line_D.set_data(x_data, y_data_D)                                     # AGGIORNA I DATI DELLA LINEA DEI DEFEZIONISTI

    return [im, counter_C, counter_D, line_C, line_D]                     # RESTITUISCE GLI OGGETTI DA AGGIORNARE NELL'ANIMAZIONE 

if __name__ == "__main__":                                                # BLOCCO PRINCIPALE DEL PROGRAMMA
    model = ImitateBestModel(WIDTH, HEIGHT)                               # INIZIALIZZA IL MODELLO CON LA LARGHEZZA E L'ALTEZZA DELLA GRIGLIA

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))                  # CREA UNA FINESTRA PER L'ANIMAZIONE


# IL BLOCCO SUCCESSIVO È PER LA CREAZIONE DELLA FINESTRA DELL'ANIMAZIONE E LA PARTE GRAFICA DEL CODICE
    # Griglia a sinistra
    grid_array = model.get_grid_array()                                             # OTTIENE LA MATRICE DELLA DISTRIBUZIONE DELLE STRATEGIE
    im = ax.imshow(grid_array.T, cmap="coolwarm", origin="lower", animated=True)    # CREA L'IMMAGINE DELLA DISTRIBUZIONE DELLE STRATEGIE
    ax.set_xticks(np.arange(0, WIDTH, 5))                                           # 
    ax.set_yticks(np.arange(0, HEIGHT, 5))                                          # 
    ax.set_xlabel("Posizione X")
    ax.set_ylabel("Posizione Y")
    ax.set_title("Distribuzione delle Strategie")

    # Counter affiancati con colori
    counter_C = ax.text(0.3, -0.12, "", transform=ax.transAxes,                     # IMPOSTIAMO IL COUNTER DEI COLLABORATORI
                        ha='center', fontsize=11, color='blue')
    counter_D = ax.text(0.7, -0.12, "", transform=ax.transAxes,                     # IMPOSTIAMO IL COUNTER DEI DEFETTORI
                        ha='center', fontsize=11, color='red')

    # Grafico a destra
    ax2.set_xlim(0, STEPS)                                                  # IMPOSTIAMO I LIMITI DELLA GRAFICA X
    ax2.set_ylim(0, 100)                                                    # IMPOSTIAMO I LIMITI DELLA GRAFICA E Y
    ax2.set_xlabel("Passi")                                                 # IMPOSTIAMO L'ETICHETTA DELL'ASSE X
    ax2.set_ylabel("Percentuale")                                           # IMPOSTIAMO L'ETICHETTA DELL'ASSE Y
    ax2.set_title("Evoluzione delle Percentuali di C e D")                  # IMPOSTIAMO IL TITOLO DELLA GRAFICA
    line_C, = ax2.plot([], [], label="C (Cooperante)", color='blue')        # IMPOSTIAMO LA LINEA DEI COOPERANTI
    line_D, = ax2.plot([], [], label="D (Difensore)", color='red')          # IMPOSTIAMO LA LINEA DEI DEFETTORI
    ax2.legend()

    x_data = []                                                             # INIZIALIZZA LA LISTA DEI DATI X
    y_data_C = []                                                           # INIZIALIZZA LA LISTA DEI DATI Y DEI COOPERANTI
    y_data_D = []                                                           # INIZIALIZZA LA LISTA DEI DATI Y DEI DEFETTORI

    ani = FuncAnimation(fig, update, frames=STEPS, interval=500, blit=False, repeat=False)      # CREA L'ANIMAZIONE CON LA FUNZIONE DI AGGIORNAMENTO DEFINITA SOPRA

    plt.tight_layout()                              # REGOLA IL LAYOUT DELLA FINESTRA
    plt.show()                                      # MOSTRA LA FINESTRA DELL'ANIMAZIONE
