import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import glob
import pdb

def PTBTIRDataset():
    return PTBTIRDatasetClass().get_sequence_list()


class PTBTIRDatasetClass(BaseDataset):
    """VOTRGBT2018 dataset

    Publication:
        The sixth Visual Object Tracking VOTRGBT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2019rgbt/dataset.html"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.ptbtir_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name        
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        img_dir = os.path.join(self.base_path, sequence_path, 'img')   
        img_list = glob.glob(img_dir + "/*.jpg")
        img_list.sort()
        frames = [os.path.join(img_dir, x) for x in img_list]        

#        return Sequence(sequence_name, framesv, ground_truth_rect)
        return Sequence(sequence_name, frames, 'PTBTIR', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list= [
        'airplane',
        'birds',
        'campus1',
        'campus2',
        'circle1',
        'circle2',
        'classroom1',
        'classroom2',
        'classroom3',
        'conversation',
        'crossing',
        'crossroad1',
        'crossroad2',
        'crouching',
        'crowd1',
        'crowd2',
        'crowd3',
        'crowd4',
        'distractor1',
        'distractor2',
        'fighting',
        'hiding',
        'jacket',
        'meetion1',
        'meetion2',
        'meetion3',
        'meetion4',
        'park1',
        'park2',
        'park3',
        'park4',
        'park5',
        'patrol1',
        'patrol2',
        'phone1',
        'phone2',
        'phone3',
        'road1',
        'road2',
        'road3',
        'room1',
        'room2',
        'room3',
        'sandbeach',
        'saturated',
        'school1',
        'school2',
        'sidewalk1',
        'sidewalk2',
        'sidewalk3',
        'soldier',
        'stranger1',
        'stranger2',
        'stranger3',
        'street1',
        'street2',
        'street3',
        'street4',
        'street5',
        'walking']
        return sequence_list
