# Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of system localization.
# Author       		: Camilo Gonçalves and Luan Gonçalves
# Email          	: {camilo.goncalves, luan.goncalves}@itec.ufpa.br
# License		: This script is distributed under "Public Domain" license.
###################################################################


import numpy as np
import os

# from builtins import print

from mimo_channels import getNarrowBandULAMIMOChannel, getDFTOperatedChannel
from processCoordinates import processCoordinates
import csv
import h5py
from math import ceil
import xml.etree.ElementTree as ET


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def dataset_generation(cfg_file):
    tree = ET.parse(cfg_file)
    root = tree.getroot()

    cfg = {}
    for item in root.iter("item"):
        # print(item.attrib['name'], item.text)
        if item.text.isdigit():
            cfg[item.attrib["name"]] = int(item.text)
        elif isfloat(item.text):
            cfg[item.attrib["name"]] = float(item.text)
        else:
            cfg[item.attrib["name"]] = item.text

    limit = processCoordinates(
        cfg["data_folder"],
        cfg["dataset"],
        [cfg["rsu_x"], cfg["rsu_y"], cfg["rsu_z"]],
        [cfg["area_x_l"], cfg["area_y_l"], cfg["area_x_r"], cfg["area_y_r"]],
    )

    if not os.path.exists(cfg["outputFolder"]):
        os.makedirs(cfg["outputFolder"])

    # initialize variables
    numOfValidChannels = 0
    numOfInvalidChannels = 0
    numLOS = 0
    numNLOS = 0
    count = 0

    """use dictionary taking the episode, scene and Rx number of file with rows e.g.:
    0,0,0,flow11.0,Car,753.83094753535,649.05232524135,1.59,D:/insitedata/noOverlappingTx5m/run00000,LOS=0
    0,0,2,flow2.0,Car,753.8198286576,507.38595866735,1.59,D:/insitedata/noOverlappingTx5m/run00000,LOS=1
    0,0,3,flow2.1,Car,749.7071175056,566.1905128583,1.59,D:/insitedata/noOverlappingTx5m/run00000,LOS=1"""

    with open(cfg["insiteCSVFile"], "r") as f:
        insiteReader = csv.DictReader(f)
        insiteDictionary = {}
        numExamples = 0
        for row in insiteReader:
            isValid = row["Val"]  # V or I are the first element of the list thisLine
            if isValid == "V":  # filter the valid channels
                numExamples += 1
                thisKey = (
                    str(row["EpisodeID"])
                    + ","
                    + str(row["SceneID"])
                    + ","
                    + str(row["VehicleArrayID"])
                )
                insiteDictionary[thisKey] = row
        lastEpisode = int(row["EpisodeID"])
    allOutputs = np.nan * np.ones(
        (numExamples, cfg["number_Rx_antennas"], cfg["number_Tx_antennas"]),
        np.complex128,
    )

    for e in range(cfg["numEpisodes"]):
        # print("Episode # ", e)
        b = h5py.File(cfg["inputPath"] + str(e) + ".hdf5", "r")
        allEpisodeData = b.get("allEpisodeData")
        numScenes = allEpisodeData.shape[0]
        numReceivers = allEpisodeData.shape[1]
        # store the position (x,y,z), 4 angles of strongest (first) ray and LOS or not
        receiverPositions = np.nan * np.ones((numScenes, numReceivers, 8), np.float32)
        # store two integers converted to 1
        episodeOutputs = np.nan * np.ones(
            (
                numScenes,
                numReceivers,
                cfg["number_Rx_antennas"],
                cfg["number_Tx_antennas"],
            ),
            np.float32,
        )

        for s in range(numScenes):
            for r in range(numReceivers):  # 10
                insiteData = allEpisodeData[s, r, :, :]

                # if insiteData corresponds to an invalid channel, all its values will be NaN.
                # We check for that below
                numNaNsInThisChannel = sum(np.isnan(insiteData.flatten()))
                if numNaNsInThisChannel == np.prod(insiteData.shape):
                    numOfInvalidChannels += 1
                    continue  # next Tx / Rx pair
                thisKey = str(e) + "," + str(s) + "," + str(r)

                try:
                    thisInSiteLine = list(
                        insiteDictionary[thisKey].items()
                    )  # recover from dic
                except KeyError:
                    print("Could not find in dictionary the key: ", thisKey)
                    print("Verify file", insiteCSVFile)
                    exit(-1)
                # tokens = thisInSiteLine.split(',')
                if numNaNsInThisChannel > 0:
                    numOfValidRays = int(
                        thisInSiteLine[8][1]
                    )  # number of rays is in 9-th position in CSV list
                    # I could simply use
                    # insiteData = insiteData[0:numOfValidRays]
                    # given the NaN are in the last rows, but to be safe given that did not check, I will go for a slower solution
                    insiteDataTemp = np.zeros((numOfValidRays, insiteData.shape[1]))
                    numMaxRays = insiteData.shape[0]
                    validRayCounter = 0
                    for itemp in range(numMaxRays):
                        if (
                            sum(np.isnan(insiteData[itemp].flatten())) == 1
                        ):  # if insite version 3.2, else use 0
                            insiteDataTemp[validRayCounter] = insiteData[itemp]
                            validRayCounter += 1
                    insiteData = insiteDataTemp  # replace by smaller array without NaN
                receiverPositions[s, r, 0:3] = np.array(
                    [thisInSiteLine[5][1], thisInSiteLine[6][1], thisInSiteLine[7][1]]
                )

                numOfValidChannels += 1
                gain_in_dB = insiteData[:, 0]
                timeOfArrival = insiteData[:, 1]
                # InSite provides angles in degrees. Convert to radians
                # This conversion is being done within the channel function
                AoD_el = insiteData[:, 2]
                AoD_az = insiteData[:, 3]
                AoA_el = insiteData[:, 4]
                AoA_az = insiteData[:, 5]
                RxAngle = insiteData[:, 8][0]
                RxAngle = RxAngle + 90.0
                if RxAngle > 360.0:
                    RxAngle = RxAngle - 360.0
                # Correct ULA with Rx orientation
                AoA_az = -RxAngle + AoA_az  # angle_new = - delta_axis + angle_wi;

                # first ray is the strongest, store its angles
                receiverPositions[s, r, 3] = AoD_el[0]
                receiverPositions[s, r, 4] = AoD_az[0]
                receiverPositions[s, r, 5] = AoA_el[0]
                receiverPositions[s, r, 6] = AoA_az[0]

                isLOSperRay = insiteData[:, 6]
                pathPhases = insiteData[:, 7]

                # in case any of the rays in LOS, then indicate that the output is 1
                isLOS = 0  # for the channel
                if np.sum(isLOSperRay) > 0:
                    isLOS = 1
                    numLOS += 1
                else:
                    numNLOS += 1
                receiverPositions[s, r, 7] = isLOS
                mimoChannel = getNarrowBandULAMIMOChannel(
                    AoD_az,
                    AoA_az,
                    gain_in_dB,
                    cfg["number_Tx_antennas"],
                    cfg["number_Rx_antennas"],
                    cfg["normalizedAntDistance"],
                    cfg["angleWithArrayNormal"],
                )
                # equivalentChannel = getDFTOperatedChannel(mimoChannel, number_Tx_antennas, number_Rx_antennas)
                # equivalentChannelMagnitude = np.abs(equivalentChannel)
                episodeOutputs[s, r] = np.abs(mimoChannel)
                allOutputs[count] = episodeOutputs[s, r]
                count += 1

            # finished processing this episode
        # Save beam per episode
        """
        npz_name = outputFolder + 'beams_output' + '_positions_e_' + str(e) + '.npz'
        np.savez(npz_name, episodeOutputs=episodeOutputs)
        np.savez(npz_name, receiverPositions=receiverPositions)
        print('Saved file ', npz_name) """

    # print("alloutputs shape = ", allOutputs.shape)
    mimoChannels_test = allOutputs[limit:]
    mimoChannels_train = allOutputs[:limit]
    print(
        "alloutputs shape = ",
        allOutputs.shape,
        "train = ",
        mimoChannels_train.shape,
        "test = ",
        mimoChannels_test.shape,
    )

    npz_name_train = cfg["outputFolder"] + "mimoChannels_train" + ".npz"
    np.savez(npz_name_train, output_classification=mimoChannels_train)
    print("Saved file ", npz_name_train)

    npz_name_validation = cfg["outputFolder"] + "mimoChannels_validation" + ".npz"
    np.savez(npz_name_validation, output_classification=mimoChannels_test)
    print("Saved file ", npz_name_validation)

    print(
        "Sanity check (must be 0 NaN) sum of isNaN = ", np.sum(np.isnan(allOutputs[:]))
    )
    print("total numOfInvalidChannels = ", numOfInvalidChannels)
    print("total numOfValidChannels = ", numOfValidChannels)
    print("Sum = ", numOfValidChannels + numOfInvalidChannels)

    print("total numNLOS = ", numNLOS)
    print("total numLOS = ", numLOS)
    print("Sum = ", numLOS + numNLOS)
