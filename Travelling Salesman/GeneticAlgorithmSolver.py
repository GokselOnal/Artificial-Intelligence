from RouteManager import RouteManager
from Route import Route

import numpy as np


class GeneticAlgorithmSolver:
    def __init__(self, cities, population_size=50, mutation_rate=0.2, tournament_size=10, elitism=False):
        self.cities = cities
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

    def solve(self, rm):
        rm = self.evolve(rm)
        for i in range(100):
            rm = self.evolve(rm)
        return rm

    def evolve(self, routes):
        new_generation = RouteManager(self.cities,self.population_size)

        if self.elitism:
            best_route = routes.find_best_route()
            new_generation.set_route(0,best_route)
            k = 1
            for i in range(len(new_generation) - 1):
                parent_1 = self.tournament(routes)
                parent_2 = self.tournament(routes)
                child = self.crossover(parent_1,parent_2)
                new_generation.set_route(k,child)
                k += 1
        else:
            for i in range(len(new_generation)):
                parent_1 = self.tournament(routes)
                parent_2 = self.tournament(routes)
                child = self.crossover(parent_1,parent_2)
                mutated_child = self.mutate(child)
                new_generation.set_route(i,mutated_child)
        return new_generation

    def crossover(self, route_1, route_2):
        child = Route(self.cities)
        start = np.random.randint(len(route_1))
        if start < len(route_1) - 1:
            end = np.random.randint(start + 1,len(route_2))
            s = end - start
            k = 0
            while 0 < (end - start):
                random_city = route_1.get_city(start)
                child.assign_city(k,random_city)
                start += 1
                k += 1
            for i in range(len(route_2)):
                random_city = route_2.get_city(i)
                if random_city not in child.route:
                    child.assign_city(s,random_city)
                    s += 1
        else:
            k = 1
            random_city = route_1.get_city(start)
            child.assign_city(0,random_city)
            for i in range(len(route_2)):
                random_city = route_2.get_city(i)
                if random_city not in child.route:
                    child.assign_city(k, random_city)
                    k += 1
        return child

    def mutate(self, route):
        for i in range(len(route)):
            if np.random.random() < self.mutation_rate:
                rand_1 = np.random.randint(len(route))
                rand_2 = np.random.randint(len(route))
                dna_1 = route.get_city(rand_1)
                dna_2 = route.get_city(rand_2)
                temp = dna_1
                route.assign_city(rand_1,dna_2)
                route.assign_city(rand_2,temp)
        return route

    def tournament(self, routes):
        selection = RouteManager(self.cities,self.tournament_size)
        selection_idx = 0
        for _ in range(len(selection)):
            sum_fitness = 0
            for i in routes.routes:
                fitness_val = i.calc_fitness()
                fitness_val *= 100
                sum_fitness += fitness_val
            probability = list()
            for i in routes.routes:
                normalized_fitness = (i.calc_fitness() * 100) / sum_fitness
                probability.append(normalized_fitness)
            index = 0
            random_num = np.random.random()
            while random_num > 0:
                random_num = random_num - probability[index]
                index += 1
            index -= 1
            selected_route = routes.get_route(index)
            selection.set_route(selection_idx,selected_route)
            selection_idx += 1
        winner = selection.find_best_route()

        return winner

