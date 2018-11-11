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


def example_partition():
    graph = networkx.complete_graph(4)
    assignment = {0: 1, 1: 1, 2: 2, 3:2}
    partition = Partition(graph, assignment)
    return (graph, partition)

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
    running_sum /= len(polarities)
    return sqrt(running_sum)

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



leanings_grid = get_political_leanings((10,10))
# print(get_polarity(grid))

if __name__ == '__main__':
    size = (10,10)
    flipped_count = 0
    flipped_value_sum = 0
    flipped_values = []
    nonflipped_count = 0
    nonflipped_value_sum = 0
    nonflipped_values = []
    lowest_min = .99
    for i in range(12):
        leanings_grid = get_political_leanings(size)
        polarity = get_polarity(leanings_grid)

        grid = Grid(size)
        print("Results\n--------")
        print(grid)

        chain = MarkovChain(
        proposal=propose_random_flip,
        is_valid=Validator([new_constraint]),
        accept=always_accept,
        initial_state=grid,
        total_steps=100005
        )

        
        for idx, partition in enumerate(chain):
            if (idx % 1000) == 0:
                district_votes = get_district_vote(partition.assignment, leanings_grid) # counts
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
                else:
                    nonflipped_count += 1
                    nonflipped_value_sum += value
                    nonflipped_values.append(value)
                    # print("Vote flipped!")
                    # print("Delta D_vec: {}".format(delta_dvec(polarities, polarity)))
                    # print("Delta D: {}".format(delta_d(votes, polarity)))
                    # print("\n")
            if (idx % 100000) == 0 and idx > 0:
                print("-----------")
                print("Number samples: {}".format(idx/1000))
                print("Number flipped elections: {}".format(flipped_count))
                if flipped_count != 0:
                    print("Flipped avg: {}".format(flipped_value_sum/flipped_count))
                print("Nonflipped avg: {}".format(nonflipped_value_sum/nonflipped_count))

        # print("Flipped avg: {}".format(flipped_value_sum/flipped_count))
        # print("Nonflipped avg: {}".format(nonflipped_value_sum/nonflipped_count))
        # print(lowest_min)
    # Print boxplot:
    box = plt.boxplot([flipped_values, nonflipped_values], showmeans=True, whis=99)
    plt.setp(box['boxes'][0], color='green')
    plt.setp(box['caps'][0], color='green')
    plt.setp(box['whiskers'][0], color='green')
    plt.setp(box['boxes'][1], color='blue')
    plt.setp(box['caps'][1], color='blue')
    plt.setp(box['whiskers'][1], color='blue')
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