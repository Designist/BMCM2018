import networkx

import gerrychain.partition

from gerrychain.partition import Partition
from gerrychain.partition import GeographicPartition
from gerrychain.proposals import propose_random_flip
from gerrychain.defaults import Grid

from gerrychain import MarkovChain
from gerrychain.constraints import Validator, single_flip_contiguous, new_constraint
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import math
from termcolor import colored
import random


def partitionfunc(n, k, l = 1):
    if k < 1:
        raise StopIteration
    if k == 1:
        if n >= l:
            yield (n,)
        raise StopIteration
    for i in range(l,n+1):
        for result in partitionfunc(n-i,k-1,i):
            yield (i,)+result

def iter_sample_fast(iterable, samplesize):
    results = []
    for i, v in enumerate(iterable):
        r = random.randint(0, i)
        if r < samplesize:
            if i < samplesize:
                results.insert(r, v) # add first samplesize items in random order
            else:
                results[r] = v # at a decreasing rate, replace random items

    if len(results) < samplesize:
        raise ValueError("Sample larger than population.")
 
    return results

def bubble_leanings(size, proportion, number_bubbles):

    side_1 = size[0]
    side_2 = size[1]
    total_blue = int(math.floor(side_1 * side_2 * proportion))

    city_sizes = iter_sample_fast(partitionfunc(total_blue, number_bubbles), 1)[0]

    biggest_bubble = max(city_sizes)

    leaning_matrix = np.zeros([side_1, side_2]).flatten() # zero = red

    positions = np.random.choice(side_1*side_2 - biggest_bubble, number_bubbles) 

    # one = blue
    for i in range(number_bubbles):
        size = city_sizes[i]
        start = positions[i]

        for j in range(size):
            leaning_matrix[start + j] = 1

    leaning_matrix = np.reshape(leaning_matrix[:side_1*side_2], (side_1, side_2))
    leaning_matrix[1::2, :] = leaning_matrix[1::2, ::-1]

    return leaning_matrix.astype(int)




def get_political_leanings(size, num_classes=2):
    # return np.random.randint(0, num_classes, size)
    return np.random.choice([0, 1], size=size, p=[.45, .55])

def get_polarity(grid, num_classes=2):
    areas = np.zeros(num_classes)
    for row in grid:
        for value in row:
            areas[value] += 1
    if (num_classes == 2):
        return areas[0]/(np.sum(areas))

# partition = districts
# grid = political leanings
def get_district_vote(assignment, leanings_grid, num_classes=2, num_districts=4):
    district_votes = np.zeros((num_districts, num_classes))
    for node in assignment:
        district = assignment[node]
        leaning = leanings_grid[node]
        district_votes[district][leaning] += 1
    return district_votes

def get_district_polarities(district_votes):
    polarities = [areas[0]/np.sum(areas) for areas in district_votes]
    return polarities

def delta_dvec(polarities, polarity):
    running_sum = 0
    for p in polarities:
        running_sum += abs(p - polarity) ** 2
    running_sum = sqrt(running_sum)
    return running_sum/len(polarities)

def get_vote(district_votes):
    return [areas[0] > areas[1] for areas in district_votes]

def delta_d(votes, polarity):
    return abs(sum(votes)/len(votes) - polarity)

def flipped_vote(votes, polarity):
    # Returns true if district and popular votes are different:
    if ((polarity < 0.5) and (sum(votes)/len(votes) > 0.5)) or ((polarity > 0.5) and (sum(votes)/len(votes) < 0.5)):
        return True 
    else:
        return False

def district_assignment(size, districts):

    side_1 = size[0]
    side_2 = size[1]

    population = side_1 * side_2
    district_size = math.floor(population/districts)

    assignment_matrix = np.zeros([side_1, side_2]).flatten()
    bonus = 1

    for i in range(population):
        assignment = math.floor(i/district_size) + 1 

        if assignment > districts:

            assignment_matrix = np.insert(assignment_matrix, (bonus)*(district_size + 1) - 1, bonus)
            bonus = (bonus + 1) % districts

        else:
            assignment_matrix[i] = assignment

    assignment_matrix = np.reshape(assignment_matrix[:side_1*side_2], (side_1, side_2))
    assignment_matrix[1::2, :] = assignment_matrix[1::2, ::-1]
    assignment_matrix -= np.ones(size)

    assignment_dict = dict()

    for i in range(side_1):
        for j in range(side_2):
            assignment_dict[(i, j)] = int(assignment_matrix[(i, j)])

    return assignment_dict

def print_partition(partition, leanings, size):
    matrix = np.zeros(size)
    for key in partition.assignment:
        matrix[key] = partition.assignment[key]
    output = ""
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            leaning = leanings[i, j]
            if leaning == 0:
                output += colored(int(value), 'red')
            else:
                output += colored(int(value), 'blue')
        output += "\n"
    print(output)



if __name__ == '__main__':
    
    size = (10,10)
    districts = 9
 
    # grid = Grid(size, assignment = )
    
    flipped_count = 0
    flipped_value_sum = 0
    flipped_values = []
    nonflipped_count = 0
    nonflipped_value_sum = 0
    nonflipped_values = []
    lowest_min = .99
    highest_max = 0

    for i in range(1):
        # leanings_grid = get_political_leanings(size)
        leanings_grid = bubble_leanings(size, 0.45, 1)
        polarity = get_polarity(leanings_grid)

        # print(district_assignment(size, districts))
        grid = Grid(size, assignment=district_assignment(size, districts))
        # print("Results\n--------")
        

        chain = MarkovChain(
        proposal=propose_random_flip,
        is_valid=Validator([new_constraint]),
        accept=always_accept,
        initial_state=grid,
        total_steps=100005
        )

        
        for idx, partition in enumerate(chain):
            if (idx % 1000) == 0:
                district_votes = get_district_vote(partition.assignment, leanings_grid, num_districts=districts) # counts
                polarities = get_district_polarities(district_votes)
                # print(delta_dvec(polarities, polarity))
                votes = get_vote(district_votes) # booleans
                # print(delta_d(votes, polarity))
                # print(partition)
                
                
                value = min(delta_dvec(polarities, polarity), delta_d(votes, polarity))
                if flipped_vote(votes, polarity):
                    flipped_count += 1
                    flipped_value_sum += value
                    flipped_values.append(value)
                    if (value < lowest_min):
                        lowest_min = value
                        print("-----\nNew lowest flipped value:")
                        print(value)
                        print_partition(partition, leanings_grid, size)
                        print("Polarity: {}".format(polarity))
                        # Get winners:
                        print("Votes: {}".format(votes))
                        print(district_votes)
                else:
                    nonflipped_count += 1
                    nonflipped_value_sum += value
                    nonflipped_values.append(value)
                    if (value > highest_max):
                        highest_max = value
                        print("-----\nNew highest non-flipped value:")
                        print(value)
                        print_partition(partition, leanings_grid, size)
                        print("Polarity: {}".format(polarity))
                        # Get winners:
                        print("Votes: {}".format(votes))
                        print(district_votes)
                    # print("Vote flipped!")
                    # print("Delta D_vec: {}".format(delta_dvec(polarities, polarity)))
                    # print("Delta D: {}".format(delta_d(votes, polarity)))
                    # print("\n")
            if (idx % 100000) == 0 and idx > 0:
                print("-----------")
                # print("Number samples: {}".format(idx/1000))
                print("Iteration: {}".format(i))
                print("Number flipped elections: {}".format(flipped_count))
                if flipped_count != 0:
                    print("Flipped avg: {}".format(flipped_value_sum/flipped_count))
                print("Nonflipped avg: {}".format(nonflipped_value_sum/nonflipped_count))

        # print("Flipped avg: {}".format(flipped_value_sum/flipped_count))
        # print("Nonflipped avg: {}".format(nonflipped_value_sum/nonflipped_count))
        # print(lowest_min)
    # Print boxplot:
    box = plt.boxplot([flipped_values, nonflipped_values], showmeans=True, whis=99)
    plt.setp(box['boxes'][0], color='red')
    plt.setp(box['caps'][0], color='red')
    plt.setp(box['whiskers'][0], color='red')
    plt.setp(box['caps'][1], color='red')
    plt.setp(box['whiskers'][1], color='red')
    plt.setp(box['boxes'][1], color='blue')
    plt.setp(box['caps'][2], color='blue')
    plt.setp(box['whiskers'][2], color='blue')
    plt.setp(box['caps'][3], color='blue')
    plt.setp(box['whiskers'][3], color='blue')
    plt.ylim([0, 0.3]) # y axis gets more space at the extremes
    plt.grid(True, axis='y') # let's add a grid on y-axis
    plt.title('Metric values and election flipping', fontsize=18) # chart title
    plt.ylabel('') # y axis title
    plt.xticks([1,2,3], ['Flipped','Nonflipped'])
    plt.show()


    


# I/P: size, a tuple
#      num_classes, the number of political parties

# import numpy as np

# def generate_starting_partition(size, num_groups):
#     height = size[0]
#     width = size[1]
#     values = np.zeros(size)
#     if ((height * width) % num_groups) == 0:
#         starting_point = (0,0)
#         for group in range(num_groups):
#             values[] == group

#         print(values)
#     else:
#         raise ValueError('Invalid number of groups.')

# generate_starting_partition((5,3), 3)
