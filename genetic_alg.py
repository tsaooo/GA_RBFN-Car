from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread
import numpy as np
import random
import math
from IPython import embed

class GA(QThread):
    sig_train_detail = pyqtSignal(int, float, float)
    def __init__(self, train_data, iter_times, population_len, mean_range, p_mutation, p_crossover, rbfn):
        super().__init__()
        self.train_data = train_data
        self.iter_times = iter_times
        self.population_len = population_len
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.dev_max = rbfn.dev_max
        self.optimized = rbfn

        self.m_range = mean_range
        self.num_rbfn_neuron = self.optimized.num_neuron
        self.data_dim = self.optimized.data_dim
        self.population = []

        self.exit = False

    def run(self):
        for _ in range(self.population_len):
            self.population.append(self.init_gene())
        self.min_err = math.inf
        for t in range(self.iter_times):
            if self.exit:
                break
            errs = self.__get_err_detail()
            self.sig_train_detail.emit(t+1, self.min_err, self.avg_err)

            scores = self.get_scores(errs)
            gene_score_pairs = list(zip(self.population, scores))
            #start shuffle whole population
            self.roulette_choice(gene_score_pairs, sum(scores))
            self.crossover()
            self.mutation()

        self.__get_err_detail()
        self.sig_train_detail.emit(t+1, self.min_err, self.avg_err)
        
        self.optimized.update_parameters(self.best_gene)

    #for better excution efficiency, calculate min, avg error and get best_gene at the same time 
    def __get_err_detail(self):
        errs = []
        sum_err = 0
        for gene in self.population:
            err = fitting_func(gene, self.optimized, self.train_data)
            if err < self.min_err:
                self.min_err = err
                self.best_gene = gene
            sum_err += err
            errs.append(err)
        
        self.avg_err = sum_err/len(errs)
        return errs
    def get_scores(self, errs, amplifier = 1.5):
        scores = np.full_like(errs, max(errs) + self.min_err) - errs
        scores = np.power(scores, amplifier)
        return scores

    def init_gene(self):
        gene = np.random.uniform(-1,1,self.num_rbfn_neuron)
        gene = np.append(gene, np.random.uniform(*self.m_range, self.data_dim*(self.num_rbfn_neuron-1)))
        gene = np.append(gene, np.random.uniform(0, self.dev_max ,self.num_rbfn_neuron-1))
        return gene    
    def roulette_choice(self, gene_score_pairs, sum_scores):
        def choose(maximum):
            magic_num = random.uniform(0, maximum)
            accu = 0
            for pair in gene_score_pairs:
                accu += pair[1]
                if accu >= magic_num:        
                    return pair[0]

        chosen = []
        for _ in range(self.population_len):
            chosen.append(choose(sum_scores))
        self.population = chosen
        self.population[1] = self.best_gene
    def crossover(self):
        #iterate for [n/2] time 
        for t in range(self.population_len//2):
            chosen = t*2
            pair = []
            random.shuffle(self.population)
            if random.uniform(0,1) < self.p_crossover:
                #sample 2 gene in population 
                idx_1 = random.randint(0, len(self.population)-1 - chosen)
                pair.append(self.population[idx_1])

                idx_2 = random.randint(0, len(self.population)-1 - chosen - 1)
                pair.append(self.population[idx_2])
                # direction = 1 : further, = -1 : closer
                direction = 1 if bool(random.getrandbits(1)) else -1
                self.population[idx_1] = self.__gene_clip(pair[0] + direction*random.uniform(0, 1)*(pair[0] - pair[1]))
                self.population[idx_2] = self.__gene_clip(pair[1] + direction*random.uniform(0, 1)*(pair[1] - pair[0]))
                #swap crossover pair to the end of population
                self.population[-1-chosen], self.population[idx_1] = self.population[idx_1], self.population[-1-chosen]
                self.population[-2-chosen], self.population[idx_2] = self.population[idx_2], self.population[-2-chosen]

    def mutation(self):
        s = 0.1
        for gene in self.population:
            if random.uniform(0,1)< self.p_mutation:
                noise = s if bool(random.getrandbits(1)) else -s
                gene = self.__gene_clip(gene + noise*self.init_gene())

    def __gene_clip(self, gene):
        np.clip(gene[:self.num_rbfn_neuron],-1,1, out=gene[:self.num_rbfn_neuron])
        np.clip(gene[self.num_rbfn_neuron:-(self.num_rbfn_neuron-1)],*self.m_range, out=gene[self.num_rbfn_neuron:-(self.num_rbfn_neuron-1)])
        np.clip(gene[-(self.num_rbfn_neuron-1):],0,self.dev_max, out=gene[-(self.num_rbfn_neuron-1):])
        return gene
    @pyqtSlot()
    def stop(self):
        self.exit = True

def fitting_func(gene, rbfn, train_data):
    rbfn.update_parameters(gene)
    sum_val = sum(abs(data["label"] - rbfn.output(data["data"])) for data in train_data)
    return sum_val/len(train_data)