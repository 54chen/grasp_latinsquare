#!/bin/python3
import random
import sys
from multiprocessing.connection import wait
from random import randint
from tokenize import group
import time
import numpy as np

##
alpha = 0.5
treatment = 0  # number of treatment
B = 0 # BLANK


def incremental_cost(m, candidate):
    matrix = np.copy(m)
    ic = 0
    matrix[candidate[0]][candidate[1]] = candidate[2]
    row = matrix[candidate[0]]
    col = matrix[:,candidate[1]]
    values, arr_counts = np.unique(row, return_counts=True)
    for i in range(0, len(values)):
        if values[i] == candidate[2] and arr_counts[i] > 1:
            ic += arr_counts[i]
    values, arr_counts = np.unique(col, return_counts=True)
    for i in range(0, len(values)):
        if values[i] == candidate[2] and arr_counts[i] > 1:
            ic += arr_counts[i]

    # c1 = cost(matrix)
    # matrix[candidate[0]][candidate[1]] = candidate[2]
    # c2 = cost(matrix)
    # return c2 - c1
    return ic

def cost(matrix):
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return sys.maxsize
    cost = 0  # cost of perfect latin square
    missing = (matrix == B).sum()  # missing value cost
    for i in range(0, matrix.shape[0]):
        values, arr_counts = np.unique(matrix[i], return_counts=True)
        # to count the col duplicated value except B
        for x in range(0, len(values)):
            if values[x] != B and arr_counts[x] > 1:
                cost += arr_counts[x]

    matrix = matrix.T # col to row

    for i in range(0, matrix.shape[0]):
        values, arr_counts = np.unique(matrix[i], return_counts=True)
        # to count the col duplicated value except B
        for x in range(0, len(values)):
            if values[x] != B and arr_counts[x] > 1:
                cost += arr_counts[x]

    cost += missing
    return cost


def generate_neighbourhoods(s, original_matrix):
    solution = np.copy(s)
    neighbors = []
    # find the position can be replaced
    can_swaped = []
    for i in range(0, original_matrix.shape[0]):
        for j in range(0, original_matrix.shape[1]):
            if original_matrix[i][j] == B:
                can_swaped.append([i, j])

    length = len(can_swaped)
    for i in range(int(length*0.5)):
        for j in range(int(length*0.5), length):
            temp = np.copy(solution)
            swap = temp[can_swaped[i][0]][can_swaped[i][1]]
            temp[can_swaped[i][0]][can_swaped[i][1]
                                          ] = temp[can_swaped[j][0]][can_swaped[j][1]]
            temp[can_swaped[j][0]][can_swaped[j][1]] = swap

            neighbors.append(temp)
    print("2.generate {} neighborhoods".format(len(neighbors)))
    return neighbors


def local_search(solution, matrix):
    solutions = generate_neighbourhoods(solution, matrix)
    best_solution = solution
    best_cost = cost(solution)
    for i in range(0, len(solutions)):
        c = cost(solutions[i])
        if (c < best_cost):
            best_cost = c
            best_solution = solutions[i]
            print("2.5.find best solution by local search score:{}".format(c))
    return best_solution


def rank(matrix, i, j):
    best = 0
    best_cost = 0
    for x in range(1, treatment):
        matrix[i][j] = x
        cost = cost(matrix)
        if cost < best_cost:
            best_cost = cost
            best = x
    return best


def generate_ground_set(m):
    matrix = np.copy(m)
    lst = []

    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            if matrix[i][j] != B:
                continue
            row = matrix[i]
            col = matrix[:, j]
            for k in range(1, treatment+1):
                if k in row:
                    continue
                if k in col:
                    continue
                next_treatment = k
                candidate = [i, j, next_treatment, 0]
                lst.append(candidate)
    return lst


def reevaluate(m, RCL):
    new_rcl = []
    matrix = np.copy(m)
    for i in RCL:
        candidate = i
        cost = incremental_cost(matrix, candidate)
        candidate[3] = cost
        new_rcl.append(candidate)
    return new_rcl


def cost_min(s):
    min = sys.maxsize
    for i in range(0, len(s)):
        if s[i][3] < min:
            min = s[i][3]
    return min


def cost_max(s):
    max = 0
    for i in range(0, len(s)):
        if s[i][3] > max:
            max = s[i][3]
    return max


def greedy_randmized_construction(m):
    max = m[m == B].sum()
    matrix = np.copy(m)
    ground_set = generate_ground_set(matrix)
    while len(ground_set) > 0:
        c_min = cost_min(ground_set)
        c_max = cost_max(ground_set)
        threshold = c_min + alpha * (c_max - c_min)
        RCL = []
        for i in range(0, len(ground_set)):
            if ground_set[i][3] <= threshold:
                RCL.append(ground_set[i])
        choosing = True
        cnt = 0
        while choosing and cnt <= max:
            cnt += 1
            s = random.choice(RCL)
            if matrix[s[0]][s[1]] == B:
                matrix[s[0]][s[1]] = s[2]
                choosing = False
            RCL.remove(s)
            ground_set = reevaluate(matrix, RCL)
    return matrix


def repair(solution):
    m = np.copy(solution)
    blank = (m == B).sum()
    if blank > 0:
        values, cnts = np.unique(m, return_counts=True)
        wait_list = []
        for i in range(0, len(values)):
            if values[i] == B:
                continue
            for x in range(cnts[i], treatment):
                wait_list.append(values[i])

        for i in range(0,  m.shape[0]):
            for j in range(0, m.shape[1]):
                if m[i][j] == B:
                    choose = random.choice(wait_list)
                    m[i][j] = choose
                    wait_list.remove(choose)

        print("\n1.repair blank:{} score:{}".format(blank, cost(m)))

    return m


def GRASP(iteration, matrix):
    best_solution = [[]]
    best_cost = sys.maxsize
    local_search_time = 0
    for k in range(0, iteration):
        solution = greedy_randmized_construction(matrix)
        solution = repair(solution)
        st = time.time()
        solution = local_search(solution, matrix)
        local_search_time += time.time() - st
        c = cost(solution)
        print("3.iter:{}, cost: {}".format(k, c))

        if (c < best_cost):
            best_cost = c
            best_solution = solution
            print("4.BEST!=>iter:{}, solution:\n{}, cost: {}".format(k, solution, c))
        if best_cost == 0:
            break

    return local_search_time, k, best_cost, best_solution


if __name__ == "__main__":
    start_time = time.time()
    alpha = 0.8
    treatment = 10
    # 5*5
    a = np.array([
        [1, 2, B, B, 3],
        [B, 3, B, 4, B],
        [3, 5, B, 1, B],
        [B, B, B, B, B],
        [B, B, 3, B, B]])
    # 8*8*50%
    b = np.array([
        [5, B, B, 2, 3, B, B, B],
        [4, 7, B, 5, 2, 6, B, 3],
        [B, B, B, B, B, 8, 1, B],
        [B, 8, B, 7, B, 1, B, B],
        [B, 1, 5, 6, 7, B, 2, 4],
        [3, B, 7, B, 8, B, B, 6],
        [2, 5, B, 3, B, 4, B, 8],
        [1, 3, 4, 8, B, B, 5, 7]
    ])
    # 8*8*25%
    c = np.array([
        [5, B, B, B, 3, B, B, B],
        [B, B, B, 5, B, B, B, 3],
        [B, B, B, B, B, 8, 1, B],
        [B, 8, B, 7, B, B, B, B],
        [B, B, 5, 6, B, B, B, B],
        [3, B, 7, B, B, B, B, B],
        [B, 5, B, B, B, B, B, 8],
        [B, 3, 4, B, B, B, B, B]
    ])

    # 10*10*20%
    d = np.array([
        [5, B, B, B, 3, B, B, B, B, B],
        [B, B, B, 1, B, B, B, 3, B, B],
        [B, B, B, B, B, B, 1, B, 2, B],
        [B, 8, B, B, B, 3, B, B, B, B],
        [B, B, B, 6, 4, B, B, B, B, B],
        [B, B, 7, B, B, B, 3, B, B, B],
        [B, B, B, B, B, B, B, 8, 1, B],
        [2, B, 4, B, B, B, B, B, B, B],
        [B, 5, B, B, B, B, B, B, B, 2],
        [B, B, B, B, B, 1, B, B, B, 9],
    ])

    # 10*10*10%
    e = np.array([
        [5, B, B, B, B, B, B, B, B, B],
        [B, B, B, B, B, B, B, 3, B, B],
        [B, B, B, B, B, B, 1, B, B, B],
        [B, 8, B, B, B, B, B, B, B, B],
        [B, B, B, 6, B, B, B, B, B, B],
        [B, B, 7, B, B, B, B, B, B, B],
        [B, B, B, B, B, B, B, 8, B, B],
        [B, B, 4, B, B, B, B, B, B, B],
        [B, 5, B, B, B, B, B, B, B, B],
        [B, B, B, B, B, 1, B, B, B, B],
    ])
    
    local_search_time, itr, best_score, best_solution = GRASP(100, e)
    print("[END]\n After {} iterations.\n The best result is \n {} \n The best score:{} \n Local search time:{}s \nTotal time: {}s".format(
        itr,best_solution, best_score, local_search_time, time.time() - start_time))