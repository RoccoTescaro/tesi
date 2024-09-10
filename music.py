__package__ = "music"

import numpy as np
from symusic import Score, TimeUnit

def midi2np(filepath):
    score = Score(filepath, ttype=TimeUnit.quarter)
    
    if len(score.tracks) > 1:
        raise ValueError("Only one track is supported")
    
    track = score.tracks[0]

    if len(track.pedals) != 0:
        raise ValueError("Pedals are not supported")
    
    resolution = 32
    instrument = 0 #symusic does not support instruments?
    cloud = [
                ( instrument, 
                  n.pitch, 
                  int(n.time*resolution), 
                  int(n.duration*resolution), 
                  n.velocity ) 
                for n in track.notes
            ]
    
    return np.array(cloud) #dtype=np.uint16

def nPitches(cloud):
    return len(set(cloud[:,1]))

def nNotes(cloud):
    return len(cloud)

def pitchClassHist(cloud):
    hist, _ = np.histogram(cloud[:,1] % 12, bins = 12)
    return hist / len(cloud)
    
def pitchClassTransMatrix(cloud, norm = 0):
    transition_matrix = np.zeros((12,12))
    for i in range(len(cloud)-1):
        transition_matrix[cloud[i][1] % 12, cloud[i+1][1] % 12] += 1
    transition_matrix /= len(cloud) 

    if norm == 0:
        return transition_matrix

    elif norm == 1:
        sums = np.sum(transition_matrix, axis=1)
        sums[sums == 0] = 1
        return transition_matrix / sums.reshape(-1, 1)

    elif norm == 2:
        return transition_matrix / np.sum(transition_matrix)

    else:
        print("invalid normalization mode, return unnormalized matrix")
        return transition_matrix
    
def pitchRange(cloud):
    return np.max(cloud[:,1]) - np.min(cloud[:,1])

def avgPitchShift(cloud):
    return np.mean(cloud[1:,1] - cloud[:-1,1])

def avgIOI(cloud):
    return np.mean(cloud[1:,2] - cloud[:-1,2])

def noteLengthHist(cloud, resolution):
    bins = 4 * resolution
    hist, _ = np.histogram(cloud[:,3], bins = bins)
    return hist / len(cloud)

def noteLengthTransMatrix(cloud, resolution, norm = 0):
    bins = 4 * resolution
    transition_matrix = np.zeros((bins,bins))
    for i in range(len(cloud)-1):
        transition_matrix[cloud[i][3], cloud[i+1][3]] += 1
    transition_matrix /= len(cloud) 

    if norm == 0:
        return transition_matrix

    elif norm == 1:
        sums = np.sum(transition_matrix, axis=1)
        sums[sums == 0] = 1
        return transition_matrix / sums.reshape(-1, 1)

    elif norm == 2:
        return transition_matrix / np.sum(transition_matrix)

    else:
        print("invalid normalization mode, return unnormalized matrix")
        return transition_matrix

