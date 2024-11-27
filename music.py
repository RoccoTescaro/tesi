__package__ = "music"

import numpy as np
from symusic import Score, TimeUnit, TimeSignature

n_measures = 8

def midi2np(filepath):
    score = Score(filepath, ttype=TimeUnit.quarter)
    
    if len(score.tracks) > 1:
        raise ValueError("Only one track is supported")
    
    track = score.tracks[0]

    if len(track.pedals) != 0:
        raise ValueError("Pedals are not supported")
    
    #n_measures = 8
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
    
    ts = score.time_signatures
    if len(ts) == 0:
        ts = [TimeSignature(0, 4, 4)]
        #raise ValueError("No time signature found")
    
    return np.array(cloud), ts, resolution

def measures(cloud, time_signatures, resolution):
    measures_ = []
    
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
        measures_.append(measure_notes)
    
    return measures_

#TODO add trans matrix by measure

def nPitches(cloud):
    return len(set([note[1] for note in cloud]))

def nPitchesPerMeasure(path):
    cloud, time_signatures, resolution = midi2np(path)
    n_pitches = []
    measures_ = measures(cloud, time_signatures, resolution)
    for measure in measures_:
        n_pitches.append(nPitches(np.array(measure)))

    if len(n_pitches) != n_measures:
        n_pitches += [0] * (n_measures - len(n_pitches))
    return n_pitches

def nNotes(cloud):
    return len(cloud)

def nNotesPerMeasure(path):
    cloud, time_signatures, resolution = midi2np(path)
    n_notes = []
    measures_ = measures(cloud, time_signatures, resolution)
    for measure in measures_:
        n_notes.append(nNotes(np.array(measure)))

    if len(n_notes) != n_measures:
        n_notes += [0] * (n_measures - len(n_notes))
    return n_notes

def pitchClassHist(path):
    cloud, _, _ = midi2np(path)
    hist, _ = np.histogram(cloud[:,1] % 12, bins = 12)
    return hist / len(cloud)

def pitchClassHistPerMeasure(path): 
    cloud, time_signatures, resolution = midi2np(path)
    hist = []
    measures_ = measures(cloud, time_signatures, resolution)
    for measure in measures_:
        pitches = [note[1] % 12 for note in measure]
        hist_, _ = np.histogram(pitches, bins = 12)
        if len(pitches) == 0:
            hist_ = np.zeros(12)
        else:
            hist_ = hist_ / len(pitches)
        hist.append(hist_)

    if len(hist) != n_measures:
        hist += [np.zeros(12)] * (n_measures - len(hist))

    hist = np.array(hist).flatten()
    return hist
    
def pitchClassTransMatrix(path): 
    #TODO fix this, should calculate the pitch shift checking if next note in the cloud is not played with current note
    #maybe by grouping notes by time and then calculating the pitch shift between the highest (to the lowest) notes in each group
    cloud, _, _ = midi2np(path)
    transition_matrix = np.zeros((12,12))
    for i in range(len(cloud)-1):
        transition_matrix[cloud[i][1] % 12, cloud[i+1][1] % 12] += 1
    transition_matrix /= len(cloud) 
    
    return transition_matrix.flatten()
'''
def pitchClassTransMatrixPerMeasure(path):
    cloud, time_signatures, resolution = midi2np(path)
    transition_matrix = []
    measures_ = measures(cloud, time_signatures, resolution)
    for measure in measures_:
        transition_matrix_ = np.zeros((12,12))
        for i in range(len(measure)-1):
            transition_matrix_[measure[i][1] % 12, measure[i+1][1] % 12] += 1
        
        if len(measure) == 0:
            transition_matrix_ = np.zeros((12,12))
        else:
            transition_matrix_ /= len(measure) 
        transition_matrix.append(transition_matrix_.flatten())
    
    if len(transition_matrix) != n_measures:
        transition_matrix += [np.zeros(144)] * (n_measures - len(transition_matrix))

    transition_matrix = np.array(transition_matrix).flatten()    
    return transition_matrix
'''
def pitchRange(path):
    cloud, _, _ = midi2np(path)
    return [np.max(cloud[:,1]) - np.min(cloud[:,1])]

def avgPitchShift(path):
    cloud, _, _ = midi2np(path)
    return [np.mean(cloud[1:,1] - cloud[:-1,1])]

def avgIOI(path):
    cloud, _, _ = midi2np(path)
    return [np.mean(cloud[1:,2] - cloud[:-1,2])]

def noteLengthHist(path):
    cloud, _, _ = midi2np(path)
    hist, _ = np.histogram(cloud[:3] % 24, bins = 24)
    return hist / len(cloud)

def noteLengthHistPerMeasure(path):
    cloud, time_signatures, resolution = midi2np(path)
    hist = []
    measures_ = measures(cloud, time_signatures, resolution)
    for measure in measures_:
        note_lengths = [note[3] % 24 for note in measure]
        hist_, _ = np.histogram(note_lengths, bins = 24)
        if len(note_lengths) == 0:
            hist_ = np.zeros(24)
        else:
            hist_ = hist_ / len(note_lengths)
        hist.append(hist_)

    if len(hist) != n_measures:
        hist += [np.zeros(24)] * (n_measures - len(hist))

    hist = np.array(hist).flatten()
    return hist

def noteLengthTransMatrix(path):
    #TODO fix this, should calculate the pitch shift checking if next note in the cloud is not played with current note
    # maybe by grouping notes by time and then calculating the pitch shift between the highest (to the lowest) notes in each group    
    cloud, _, _ = midi2np(path)
    bins = 24
    transition_matrix = np.zeros((bins,bins))

    for i in range(len(cloud)-1):
        transition_matrix[cloud[i][3] % bins, cloud[i+1][3] % bins] += 1
    transition_matrix /= len(cloud) 

    return transition_matrix.flatten()
'''
def noteLengthTransMatrixPerMeasure(path):
    cloud, time_signatures, resolution = midi2np(path)
    bins = 24
    transition_matrix = []
    measures_ = measures(cloud, time_signatures, resolution)
    for measure in measures_:
        transition_matrix_ = np.zeros((bins,bins))
        for i in range(len(measure)-1):
            transition_matrix_[measure[i][3] % bins, measure[i+1][3] % bins] += 1
        transition_matrix_ /= len(measure) 
        transition_matrix.append(transition_matrix_.flatten())

    if len(transition_matrix) != n_measures:
        transition_matrix += [np.zeros(576)] * (n_measures - len(transition_matrix))
    return transition_matrix
'''