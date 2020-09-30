# Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of the UFPA - ITU Artificial Intelligence/Machine Learning in 5G Challenge (http://ai5gchallenge.ufpa.br/).
# Author       		: Ailton Oliveira, Aldebaro Klautau, Arthur Nascimento, Diego Gomes, Jamelly Ferreira, Walter Frazão
# Email          	: ml5gphy@gmail.com
# License		: This script is distributed under "Public Domain" license.
###################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
from CSVHandler import CSVHandler
import os
from numpy import load


def processCoordinates(data_folder, dataset, rsu_coord, area_shp):
    print("Generating Beams ...")
    csvHand = CSVHandler()

    inputDataDir = data_folder
    coordFileName = "CoordVehiclesRxPerScene_s008"
    coordURL = dataset + coordFileName + ".csv"

    coordinates_train, context_train, coordinates_test, context_test = csvHand.getCoord(
        coordURL, 1564
    )

    area_shp = [
        area_shp[0] - rsu_coord[0],
        area_shp[1] - rsu_coord[1],
        area_shp[2] - rsu_coord[0],
        area_shp[3] - rsu_coord[1],
    ]

    coordinates_train = [
        [(float(a) - float(b)) / c for a, b, c in zip(x, rsu_coord, area_shp[2:])]
        for x in coordinates_train
    ]  # coordinates_train - rsu_coord

    coordinates_test = [
        [(float(a) - float(b) / c) for a, b, c in zip(x, rsu_coord, area_shp[2:])]
        for x in coordinates_test
    ]  # coordinates_test - rsu_coord

    train_channels = len(coordinates_train)

    # train
    np.savez(inputDataDir + "coord_train" + ".npz", coordinates=coordinates_train)
    np.savez(inputDataDir + "context_train" + ".npz", context=context_train)
    # test
    np.savez(inputDataDir + "coord_validation" + ".npz", coordinates=coordinates_test)
    np.savez(inputDataDir + "context_test" + ".npz", context=context_test)

    print("Coord npz files saved!")

    return train_channels
