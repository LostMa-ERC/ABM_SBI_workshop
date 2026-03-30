import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import birth_death_utils as bd
import os
from collections import Counter
from tqdm.notebook import tqdm
import pickle
from itertools import groupby
import pandas as pd
import yaml
import re
import multiprocessing as mp
import random
import seaborn as sns
from sbi.analysis import pairplot
from sbi.inference import NPE, simulate_for_sbi, NLE
from sbi.utils import BoxUniform
from torch import Tensor
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


## Date handling function

def convert_date(x):
    pattern1 = re.compile('[0-9][0-9][0-9][0-9]')
    pattern2 = re.compile('[0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]')

    if bool(pattern1.fullmatch(x)):
        return int(x)
    if bool(pattern2.fullmatch(x)):
        xx = x.split('-')
        return (int(xx[0]),int(xx[1]))

def top(x):
    '''
    Upper bound on witness date (single year or range)
    '''
    match x:
        case (x1, x2):
            return x2
        case _:
            return x

def bottom(x):
    '''
    Lower bound on witness date (single year or range)
    '''
    match x:
        case (x1, x2):
            return x1
        case _:
            return x


def expected_abs_diff_degenerate(c, a, b):
    '''
    Returns the expectation value of the timelapse between one date given as range
    and one other as a single year
    '''
    if a == b:
        return abs(a - c)
    if a <= c <= b:
        return ((c-a)**2 + (b-c)**2) / (2*(b-a))
    elif c < a:
        return (a+b)/2 - c
    else: 
        return c - (a+b)/2

def expected_abs_diff(a, b, c, d):
    '''
    Return the expectation value of the timelapse between two dates given as
    intervals
    '''
    if a == b:
        return expected_abs_diff_degenerate(a, c, d)
    if c == d:
        return expected_abs_diff_degenerate(c, a, b)
    elif a < b < c < d:
        return (c+d-b-a)/2
    elif a <= c <= d < b:
        return (1/(b-a)) * ((1/2)*((d-a)*(c-a)+(b-d)*(b-c)) + (1/3)*(d-c)**2)
    elif a <= c <= b <= d:
        return (1/(b-a)) * ((1/2)* ((c-a)*(d-a) + (b-c)*(d-b)) + (1/(3*(d-c)))*(b-c)**3 )
    else:
        print(f"{a},{b} -- {c},{d} ")
        raise ValueError("Must have a < b, c < d, and a < c")

## Data loader functions

def load_stemmata(path):
    wholeCorpus = {}
    corpus_dates = {}
    for work in tqdm(os.listdir(f'{path}/')):
        st = bd.load_from_OpenStemmata(f'corpus_stemmata/{work}/stemma.gv')
        with open(f"{path}/{work}/metadata.txt", 'r') as f:
            content = f.read()
        metadata = yaml.safe_load(content)
        if "wits" in metadata:
            dates = [wit["witOrigDate"] for wit in metadata['wits'] if wit["witOrigDate"] != '']
        else:
            dates = []

        dates_num = []
        for x in dates:
            if x != '':
                date_num = convert_date(x)
                if date_num != None:
                    dates_num.append(date_num)
    
        wholeCorpus[f"{work}"] = st
        corpus_dates[f"{work}"] = dates_num

    ranges_per_work = {}
    for work, t_dates in corpus_dates.items():
        if t_dates != []:
            lb = sorted(t_dates, key=bottom)[0]
            ub = sorted(t_dates, key=top)[-1]
            ranges_per_work[work] = (lb,ub)


    lifespans = {}
    for w,v in ranges_per_work.items():
        match v:
            case [(a,b), (c,d)]:
                lifespans[w] = expected_abs_diff(a,b,c,d)
            case [(a,b), c]:
                lifespans[w] = expected_abs_diff_degenerate(a,b,c)
            case [c,(a,b)]:
                lifespans[w] = expected_abs_diff_degenerate(a,b,c)
            case [a,b]:
                lifespans[w] = abs(a-b)
            case _:
                print('error')

    x_obs0 = {}
    sizes = {}
    for k in lifespans.keys():
        ## computation of observables
        g = wholeCorpus[k]

        n_living = list(nx.get_node_attributes(g, 'state').values()).count(True)
        sizes[k] = n_living
        degrees = []
        direct_filiation_nb = 0
        arch_dists = []

        if n_living >= 3:
            st = bd.generate_stemma(g)
            archetype = bd.root(st)
            for n in st.nodes():
                degrees.append(st.out_degree(n))

                if n != archetype:
                    father = list(st.predecessors(n))[0]
                    if st.nodes[n]['state'] and st.nodes[father]['state']:
                        direct_filiation_nb +=1
                arch_dists.append(len(nx.shortest_path(st, source=archetype, target=n)))
            
            timelapse = lifespans[k]
            deg_dist = Counter(degrees)
            deg1 = deg_dist[1]
            deg2 = deg_dist[2]
            deg3 = deg_dist[3]
            deg4 = deg_dist[4]
            depth = max(arch_dists)
            n_nodes = len(list(st.nodes()))

            x_obs0[k] = [
                n_living,
                4*int(timelapse),
                n_nodes,
                direct_filiation_nb,
                deg1,
                deg2,
                deg3,
                deg4,
                depth
            ]

    return list(x_obs0.values())

def load_2_wits_texts(file):
    df = pd.read_csv(file)
    f1_works = []
    f2_works = []
    works = set(list(df['text H-ID']))
    size_frags_d = []
    size_d = []
    for work in works:
        n_wit = len(df[(df['text H-ID'] == work) & (df['status'] != 'fragment')])
        n_frags = len(df[(df['text H-ID'] == work) & (df['status'] == 'fragment')])
        if n_wit !=0:
            size_d.append(n_wit)
        if n_frags !=0:
            size_frags_d.append(n_frags)

        if n_wit == 2:
            f2_works.append(work)
        if n_wit == 1:
            f1_works.append(work)

    size_dist = Counter(size_d)
    size_dist_frags = Counter(size_frags_d)
    f2_dates_0 = [df[(df["text H-ID"] == x) & (df['status'] != 'fragment')]["Date"].values.tolist() for x in f2_works]
    f2_dates = [list(map(convert_date, x)) for  x in f2_dates_0]

    ranges_per_work = []
    for t_dates in f2_dates:
        if t_dates != []:
            lb = sorted(t_dates, key=bottom)[0]
            ub = sorted(t_dates, key=top)[-1]
            ranges_per_work.append((lb,ub))


    lifespans_f2 = []
    for v in ranges_per_work:
        match v:
            case [(a,b), (c,d)]:
                lifespans_f2.append(expected_abs_diff(a,b,c,d))
            case [(a,b), c]:
                lifespans_f2.append(expected_abs_diff_degenerate(a,b,c))
            case [c,(a,b)]:
                lifespans_f2.append(expected_abs_diff_degenerate(a,b,c))
            case [a,b]:
                lifespans_f2.append(abs(a-b))
            case _:
                print('error')

    return [[2, int(4 * n), -1,-1,-1,-1,-1,-1,-1] for n in lifespans_f2]

# Compute summary stats on simulated data

def compute_summary_stats(g):
    """
    Generate a vector of summary statistics from a graph generated by birth-death simulation

    Parameters
    ----------
    g : nx.DiGraph()
        tree graph generated by birth_death_utils.ct_bd_tree

    Returns
    -------
    list
        vector of summary statistics characterizing a single tradition
    """
    n_living = list(nx.get_node_attributes(g, 'state').values()).count(True)

    if n_living == 0:
        return None
    
    if n_living == 1:
        return [1,0, -1,-1,-1,-1,-1,-1,-1]

    if n_living == 2:
        birth_times_trad = []
        for n in g.nodes():
            if g.nodes[n]['state']:
                birth_times_trad.append(g.nodes[n]['birth_time'])
        timelapse = int(max(birth_times_trad)-min(birth_times_trad))
        return [2, timelapse, -1,-1,-1,-1,-1,-1,-1]
    
    if n_living >= 3:
        birth_times_trad = []
        degrees = []
        direct_filiation_nb = 0
        arch_dists = []
        st = bd.generate_stemma(g)
        archetype = bd.root(st)

        for n in st.nodes():
            degrees.append(st.out_degree(n))

            if n != archetype:
                father = list(st.predecessors(n))[0]
                if st.nodes[n]['state'] and st.nodes[father]['state']:
                    direct_filiation_nb +=1
            if st.nodes[n]['state']:
                birth_times_trad.append(st.nodes[n]['birth_time'])
            arch_dists.append(len(nx.shortest_path(st, source=archetype, target=n)))
        
        timelapse = int(max(birth_times_trad)-min(birth_times_trad))
        deg_dist = Counter(degrees)
        deg1 = deg_dist[1]
        deg2 = deg_dist[2]
        deg3 = deg_dist[3]
        deg4 = deg_dist[4]
        depth = max(arch_dists)
        n_nodes = len(list(st.nodes()))

        return [
            n_living,
            timelapse,
            n_nodes,
            direct_filiation_nb,
            deg1,
            deg2,
            deg3,
            deg4,
            depth
        ]

# Model simulator

def generate_tree_crbd(lda, mu, Tact, Tinact):
    """
    Generate a tree (arbre réel) according to birth death model.

    Parameters
    ----------
    lda : float
        birth rate of new node per node per iteration
    mu : float
        death rate of nodes per node per per iteration
    Nact : int
        number of iterations of the active reproduction phase
    Ninact : int
        number of iterations of the pure death phase (lda is set to 0)

    Returns
    -------
    G : nx.DiGraph()
        networkx graph object of the generated tree with following node attributes:
            'state' : boolean, True if node living at the end of simulation
            'birth_time' : int
            'death_time' : int

    """
    currentID = 0
    G = nx.DiGraph()
    G.add_node(currentID)
    living_nodes = set([0])

    birth_time = {0:0}
    death_time = {}

    pop = 1
    prob_birth = lda / (lda + mu)
    prob_death =  mu / (lda + mu)
    prob_event = lda + mu

    t = 0

    while t < Tact:
        if pop == 0:
            t = Tact
            break
        next_event = np.random.exponential(scale = 1. / (prob_event * pop))
        if next_event > Tact:
            t = Tact
            break

        t += next_event
        r = np.random.rand()
        current_node = np.random.choice(list(living_nodes))
        if r < prob_birth:
            currentID += 1
            G.add_node(currentID)
            G.add_edge(current_node, currentID)
            living_nodes.add(currentID)
            pop += 1
            birth_time[currentID] = t
        else:
            living_nodes.remove(current_node)
            pop -= 1
            death_time[current_node] = t
    
    while t < Tact + Tinact:
        if pop == 0:
            t = Tact + Tinact
            break
        next_event = np.random.exponential(scale = 1. / (mu * pop))
        if next_event > Tact + Tinact:
            t = Tact + Tinact
            break
        t += next_event
        current_node = np.random.choice(list(living_nodes))
        living_nodes.remove(current_node)
        pop -= 1
        death_time[current_node] = t
    
    living = {n:(n in living_nodes) for n in G.nodes()}
    nx.set_node_attributes(G, living, 'state')
    nx.set_node_attributes(G, birth_time, 'birth_time')
    nx.set_node_attributes(G, death_time, 'death_time')

    return G

def generate_tree_bd_decay(lda0, mu, Tact, Tinact):
    currentID = 0
    G = nx.DiGraph()
    G.add_node(currentID)
    living_nodes = set([0])

    birth_time = {0:0}
    death_time = {}

    pop = 1

    t = 0

    while t < Tact:
        lda1 = (2 * lda0 / Tact) * (Tact-t)
        prob_event = lda1 + mu
        prob_birth = lda1 / (lda1 + mu)
        prob_death = mu / (lda1 + mu)

        if pop == 0:
            t = Tact
            break
        next_event = np.random.exponential(scale = 1. / (prob_event * pop))
        if next_event > Tact:
            t = Tact
            break

        t += next_event
        r = np.random.rand()
        current_node = np.random.choice(list(living_nodes))
        if r < prob_birth:
            currentID += 1
            G.add_node(currentID)
            G.add_edge(current_node, currentID)
            living_nodes.add(currentID)
            pop += 1
            birth_time[currentID] = t
        else:
            living_nodes.remove(current_node)
            pop -= 1
            death_time[current_node] = t
    
    while t < Tact + Tinact:
        if pop == 0:
            t = Tact + Tinact
            break
        next_event = np.random.exponential(scale = 1. / (mu * pop))
        if next_event > Tact + Tinact:
            t = Tact + Tinact
            break
        t += next_event
        current_node = np.random.choice(list(living_nodes))
        living_nodes.remove(current_node)
        pop -= 1
        death_time[current_node] = t
    
    living = {n:(n in living_nodes) for n in G.nodes()}
    nx.set_node_attributes(G, living, 'state')
    nx.set_node_attributes(G, birth_time, 'birth_time')
    nx.set_node_attributes(G, death_time, 'death_time')

    return G

def generate_tree_bd_decim(lda, mu, Tact, Tinact, Tcrisis, decim_rate):
    currentID = 0
    G = nx.DiGraph()
    G.add_node(currentID)
    living_nodes = set([0])

    birth_time = {0:0}
    death_time = {}

    pop = 1
    prob_birth = lda / (lda + mu)
    prob_death =  mu / (lda + mu)
    prob_event = lda + mu

    t = 0
    crisis_happened = False

    while t < Tact:
        if pop == 0:
            t = Tact
            break
        next_event = np.random.exponential(scale = 1. / (prob_event * pop))
        if next_event > Tact:
            t = Tact
            break

        t += next_event
        r = np.random.rand()
        current_node = np.random.choice(list(living_nodes))

        if r < prob_birth:
            currentID += 1
            G.add_node(currentID)
            G.add_edge(current_node, currentID)
            living_nodes.add(currentID)
            pop += 1
            birth_time[currentID] = t
        else:
            living_nodes.remove(current_node)
            pop -= 1
            death_time[current_node] = t
        
        if t > Tcrisis and not crisis_happened:
            decimated_nodes = random.sample(list(living_nodes), int(decim_rate*pop))
            for n in decimated_nodes:
                living_nodes.remove(int(n))
                death_time[n] = t
                pop -= 1
            crisis_happened = True

    while t < Tact + Tinact:
        if pop == 0:
            t = Tact + Tinact
            break
        next_event = np.random.exponential(scale = 1. / (mu * pop))
        if next_event > Tact + Tinact:
            t = Tact + Tinact
            break
        t += next_event
        current_node = np.random.choice(list(living_nodes))
        living_nodes.remove(current_node)
        pop -= 1
        death_time[current_node] = t
    
    living = {n:(n in living_nodes) for n in G.nodes()}
    nx.set_node_attributes(G, living, 'state')
    nx.set_node_attributes(G, birth_time, 'birth_time')
    nx.set_node_attributes(G, death_time, 'death_time')

    return G