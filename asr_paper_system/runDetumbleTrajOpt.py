import sys

sys.path.append("../")
import numpy as np
from PostCapDetumbleTrajOpt import PostCapDetumbleTrajectory


def runTrajectoryOptimization():
    """
    Function to run an example trajectory optimization for detumbling the spacecraft given in the
    paper:

    Vyas, S., Maywald, L., Kumar, S., Jankovic, M., Mueller, A. and Kirchner, F., 2022.
    Post-capture detumble trajectory stabilization for robotic active debris removal.
    Advances in Space Research. 2022.

    """
    detumbleTraj = PostCapDetumbleTrajectory("detumbleConfigInitialGuess.toml", np.array([0, np.deg2rad(5), 0]))
    detumbleTraj.setupOptimizationProblem()
    detumbleTraj.solveOptimizationProblem(saveFileName="detumbleTraj.pkl")
    detumbleTraj.readResultsFromFile("detumbleTraj.pkl")
    detumbleTraj.plotResults(plotPositions=False, plotForces=True, plotVelocities=True)


def runASRPaperTrajectoryOptimization():
    """
    Function to (almost) reproduce the results of the paper. Due to the randomness in the initial
    conditions and non-deterministic nature of the optimization algorithm, the results will not
    match exactly. However, they are usually quite close.

    Vyas, S., Maywald, L., Kumar, S., Jankovic, M., Mueller, A. and Kirchner, F., 2022.
    Post-capture detumble trajectory stabilization for robotic active debris removal.
    Advances in Space Research. 2022.

    Initial trajectory optimization is done with a linear interpolation-based guess and relaxed
    constraints on quaternion unit length, dynamics equations, and final velocity. The results are
    then used as the initial guess for a second optimization with tighter constraints.
    """
    # # First optimization
    # print("Running first optimization to get initial guess...")
    # detumbleTrajInitGuess = PostCapDetumbleTrajectory("detumbleConfigInitialGuess.toml", np.array([0, np.deg2rad(5), 0]))
    # detumbleTrajInitGuess.setupOptimizationProblem()
    # detumbleTrajInitGuess.solveOptimizationProblem(saveFileName="detumbleTrajInitGuess.pkl")

    # Second optimization
    print("Running second optimization with initial guess from first optimization...")
    detumbleTraj = PostCapDetumbleTrajectory("detumbleConfig.toml", np.array([0, np.deg2rad(5), 0]))
    detumbleTraj.setupOptimizationProblem()
    detumbleTraj.solveOptimizationProblem(saveFileName="detumbleTraj.pkl")
    detumbleTraj.readResultsFromFile("detumbleTraj.pkl")
    detumbleTraj.plotResults(plotPositions=False, plotForces=True, plotVelocities=True)


if __name__ == "__main__":
    runASRPaperTrajectoryOptimization()
    # runTrajectoryOptimization()
