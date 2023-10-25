import sys

sys.path.append("../")
from TrajectoryStabilizationTVLQR import TVLQRTrajectoryStabilizer


def runTrajectoryStabilization():
    trajStabilizer = TVLQRTrajectoryStabilizer(paramsFile="detumbleConfig.toml", trajFile="detumbleTraj.p")
    # trajStabilizer.synthesizeTVLQRController()
    # trajStabilizer.runSimulation(visualize=False)
    trajStabilizer.readResultsFromFile("sim_results.p")
    trajStabilizer.plotResults(plotPositions=False, plotForces=True, plotVelocities=True)


if __name__ == "__main__":
    runTrajectoryStabilization()
