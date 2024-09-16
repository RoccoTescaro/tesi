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
    
    return np.array(cloud), score.time_signatures, resolution

def measures(cloud, time_signatures, resolution):
    measures = []
    
    measure_boundaries = []
    for i, ts in enumerate(time_signatures):
        start_time = ts.time*resolution
        nominator = ts.numerator
        denominator = ts.denominator

        # Calculate the ticks per measure: 
        # resolution gives us ticks per beat, and the nominator gives the number of beats in a measure
        ticks_per_measure = resolution * nominator * (4 / denominator)
        
        # Define the measure boundaries for this time signature
        if i + 1 < len(time_signatures):
            next_ts_time = time_signatures[i + 1].time*resolution
        else:
            next_ts_time = max([note[2] + note[3] for note in cloud])  # last note end time
        
        measure_time = start_time
        while measure_time < next_ts_time:
            measure_boundaries.append((measure_time, measure_time + ticks_per_measure))
            measure_time += ticks_per_measure
    
    # For each measure, extract the notes from the cloud that fall within the measure's boundaries
    for start, end in measure_boundaries:
        measure_notes = [note for note in cloud if start <= note[2] < end]
        measures.append(measure_notes)
    
    return measures

def nPitches(cloud):
    return len(set([note[1] for note in cloud]))

def nPitchesPerMeasure(path):
    cloud, time_signatures, resolution = midi2np(path)
    n_pitches = []
    measures_ = measures(cloud, time_signatures, resolution)
    for measure in measures_:
        n_pitches.append(nPitches(np.array(measure)))
    return n_pitches

def nNotes(cloud):
    return len(cloud)

def nNotesPerMeasure(path):
    cloud, time_signatures, resolution = midi2np(path)
    n_notes = []
    measures_ = measures(cloud, time_signatures, resolution)
    for measure in measures_:
        n_notes.append(nNotes(np.array(measure)))
    return n_notes

def pitchClassHist(path):
    cloud, _, _ = midi2np(path)
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

def noteLengthHist(path):
    cloud, _, _ = midi2np(path)
    bins = 24
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

