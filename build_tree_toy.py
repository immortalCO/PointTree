import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import random
import logging

global directions
PI = torch.acos(torch.tensor(-1.0))
directions = None
otho = []
basic_directions = set()
transforms = []

def init_directions(chaos_limit=4, calc_dmap=True):
    global directions
    if directions is not None:
        return num_directions()

    directions = []
    for xx in reversed(range(0, chaos_limit + 1)):
        for yy in reversed(range(0, chaos_limit + 1)):
            for zz in reversed(range(0, chaos_limit + 1)):

                x = 2 * xx - chaos_limit
                y = 2 * yy - chaos_limit
                z = 2 * zz - chaos_limit

                # print(x, y, z)

                # if x < 0:
                #     continue
                # if abs(x) < 1e-6 and y < 0:
                #     continue
                # if abs(x) < 1e-6 and abs(y) < 1e-6 and z < 0:
                #     continue

                d = torch.tensor([x, y, z]).float()

                if d.norm() < 1e-6:
                    continue

                d /= d.norm()

                for d2 in directions:
                    if (d - d2).norm() < 1e-6 or (d + d2).norm() < 1e-6:
                        d = None
                        break

                if d is None:
                    continue

                directions.append(d)
                otho.append(set())

                if abs(x) + abs(y) + abs(z) == 1:
                    basic_directions.add(len(directions) - 1)


    logging.info(f"init_directions: # = {len(directions)}")
    for i, d in enumerate(directions):
        
        for j, e in enumerate(directions):
            if d.dot(e).abs().item() < 1e-6:
                otho[i].add(j)

        x, y, z = d.numpy().tolist()

        # logging.debug(f"{i}: {' '.join(map(lambda x : '%.6lf' % x, d.cpu().numpy().tolist()))} otho = {otho[i]}")
        print(f"{{{x}, {y}, {z}}},")

init_directions(10)

