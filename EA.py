import random
import os
import sys
import subprocess
import xml.etree.ElementTree as ET

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'.")

import traci

sumoBinary = "sumo" 
sumoCmd = [sumoBinary, "-c", "final.sumocfg"]
    
def calculate_average_delay():
    total_time = 0
    vehicle_count = len(traci.vehicle.getIDList())
    print ("vehicles" ,vehicle_count)
    for vehicle_id in traci.vehicle.getIDList():
        total_time += traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
    
    print("Time", total_time)
    if vehicle_count > 0:
        average_delay = total_time / vehicle_count
    else:
        average_delay = 0
    
    return average_delay
# print(calculate_average_delay())

def generate_population(population_size):
    min_green_time = 20
    max_green_time = 90
    initial_population = []
    for _ in range(population_size):
        ns_green_time = random.randint(min_green_time, max_green_time)
        ew_green_time = random.randint(min_green_time, max_green_time)
        yellow_time = 3
        configuration = [ns_green_time, yellow_time, ew_green_time, yellow_time]
        initial_population.append(configuration)
    return initial_population


def write_to(individual):
    tree = ET.parse('final.net.xml')
    root = tree.getroot()
    for tlLogic in root.findall('.//tlLogic[@id="{}"]'.format("J3")):
        i=0
        for phase in tlLogic.findall('phase'):
            phase.set('duration', str(individual[i]))
            i += 1
    
    tree.write('final.net.xml')
    
def select_parents(results, N_OFFSPRING):
    parents = []
    for _ in range(N_OFFSPRING):
        tournament = random.sample(results, 2)
        # compare based on average_delay
        winner = min(tournament, key=lambda x: x[1])
        parents.append(winner[0])  
    return parents

def crossover(parent1, parent2, crossover_prob):
    if random.random() < crossover_prob:
        # crossover point
        point = random.randint(1, len(parent1) - 2)
        # create new child by combination
        child = parent1[:point] + parent2[point:]
    else:
        # no crossover
        child = parent1.copy()
    return child

def mutate(child, mutation_prob):
    min_green_time = 20
    max_green_time = 90
    if random.random() < mutation_prob:
        child[0] = random.randint(min_green_time, max_green_time)
    if random.random() < mutation_prob:
        child[2] = random.randint(min_green_time, max_green_time)
    return child


# print("Population",generate_population(population_size))
indi = [42,3,42,3]
write_to(indi)
traci.start(sumoCmd)
while traci.simulation.getTime() <3600:
    if traci.simulation.getTime() == 1500:
        delay= calculate_average_delay()
    traci.simulationStep()
traci.close()
print("INITIAL DELAY: " ,delay)
graph = []
# print (population)
def ea(population_size, crossover_prob, mutation_prob, population, best_fitness, generations):
    generation = 0 
    
    while generation < generations:
        ev_population=[]
        for individual in population:
            ns_green, ns_yellow, ew_green, ew_yellow = individual
            # print (individual)
            write_to(individual)
            traci.start(sumoCmd)
            while traci.simulation.getTime() < 3600 :
                if traci.simulation.getTime() == 1500:
                    average_delay = calculate_average_delay()
                traci.simulationStep()
            # average_delay = calculate_average_delay()
            ev_population.append((individual,average_delay))
            traci.close()
            
            
        parents = select_parents(ev_population,10)
        print ("parents", parents)
        offspring = [crossover(parents[i], parents[i+1], crossover_prob) for i in range(0,population_size,2) ]
        print ("offspring: ", offspring)

        for child in offspring:
            mutate(child, mutation_prob)
        print ("MUTATED offspring: ", offspring) 
        ev_offspring = []
        for individual in offspring:
            ns_green, ns_yellow, ew_green, ew_yellow = individual
            # print (individual)
            write_to(individual)
            traci.start(sumoCmd)
            while traci.simulation.getTime() < 3600 :
                traci.simulationStep()
            average_delay = calculate_average_delay()

            ev_offspring.append((individual, average_delay))
            traci.close()

        combined_population = ev_population + ev_offspring
        print("COMBINED POP", combined_population)
        combined_population.sort(key=lambda x: x[1])
        new_population = [config for config in combined_population[:population_size]]
        current_best = new_population[0]
        if current_best[1] < best_fitness:
            best_solution = current_best
            best_fitness = best_solution[1]
        population = [individual[0] for individual in new_population]
        print("generation", generation)
        generation += 1
        graph.append(best_solution[1])
    return best_solution

best_fitness = 100000000000000000000
population_size = 10
crossover_prob = 0.8
mutation_prob = 0.5
generations = 20

ipopulation = generate_population(population_size)
solution = ea(population_size, crossover_prob, mutation_prob, ipopulation, best_fitness, generations)
print("Best configuration: ", solution[0])
print ("Average delay: ", solution[1])
print("for graph", graph)

# print(best_solution)
# def create_tlLogic(individual, tl_id="C2"):
#     ns_green, ns_yellow, ew_green, ew_yellow = individual

#     tlLogic_xml = f'''
#     <tlLogic id="{tl_id}" type="static" programID="0" offset="0">
#         <phase duration="{ns_green}" state="GGGrrrGGGrrr"/>
#         <phase duration="{ns_yellow}" state="yyyrrryyyrrr"/>
#         <phase duration="{ew_green}" state="rrrGGGrrrGGG"/>
#         <phase duration="{ew_yellow}" state="rrryyyrrryyy"/>
#     </tlLogic>
#     '''
#     return tlLogic_xml
# 
# def select_parents(population, calculate_average_delay, N_OFFSPRING):
#     parents = []
#     for _ in range (N_OFFSPRING):
#         tournament = random.sample(population, 2)
#         winner = min(tournament, key = calculate_average_delay)
#         parents.append(winner)
#     return parents

# def roulette_wheel_selection(population, evaluate, pop_size):
#     selected_parents = []

#     while len(selected_parents) < pop_size:
#         r = random.uniform(0, 1)
#         cumulative_fitness = 0
#         total_fitness = sum(1 / evaluate(adjacency_matrix, route) for route in population)
#         cumulative_probability = 0

#         for i, route in enumerate(population):
#             cumulative_fitness += 1 / evaluate(adjacency_matrix, route)
#             cumulative_probability = cumulative_fitness / total_fitness

#             if cumulative_probability >= r:
#                 selected_parents.append(route)
#                 break

#     return selected_parents