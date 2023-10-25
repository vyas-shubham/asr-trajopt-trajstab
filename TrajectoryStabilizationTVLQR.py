import os
import sys
import copy
import toml
import numpy as np
import pickle
import pydot
import matplotlib.pyplot as plt

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    PiecewisePolynomial,
    InitializeAutoDiff,
    BasicVector_,
    AutoDiffXd,
    ExtractGradient,
    TrajectoryAffineSystem,
    LinearQuadraticRegulator,
    FiniteHorizonLinearQuadraticRegulatorOptions,
    FiniteHorizonLinearQuadraticRegulator,
    TrajectorySource,
    Gain,
    Adder,
    LogVectorOutput,
    JacobianWrtVariable,
    Simulator,
)

from pydrake.geometry import (
    MeshcatVisualizer,
    StartMeshcat,
)


sys.path.append("../")
from utils import QuaternionLQRStateSelector


class TVLQRTrajectoryStabilizer:
    def __init__(self, paramsFile: "str", trajFile: "str"):
        assert os.path.isfile(paramsFile), "paramsFile must be a valid file path"
        self.config = toml.load(paramsFile)
        self.SCpath = self.config["paths"]["sc_path"]
        assert self.SCpath.endswith(".urdf"), "SC path must be a .urdf file"
        assert os.path.isfile(self.SCpath), "SC path must be a valid file path"
        traj_data = pickle.load(open(trajFile, "rb"))
        self.q_opt = traj_data["q"]
        self.qd_opt = traj_data["qd"]
        self.qdd_opt = traj_data["qdd"]
        self.dt_opt = traj_data["h"].T
        self.u_opt = traj_data["u"]
        self.N = self.q_opt.shape[0] - 1
        self.nq = self.q_opt.shape[1]
        self.armDoF = self.nq - 7
        self.nqd = self.qd_opt.shape[1]
        self.nu = self.u_opt.shape[1]
        assert self.N == self.u_opt.shape[0], "Number of knot points inconsistent between q and u"

        print("Num. Knot Points: ", self.N)
        print("Num. Positions: ", self.nq)
        print("Num. Velocities: ", self.nqd)
        print("Num. Gen Forces: ", self.nu)

        # Create a time array from the delta T array
        self.dt_opt = traj_data["h"]
        print("dt_opt shape: ", self.dt_opt.shape)
        self.time_array = np.array([sum(self.dt_opt[:t]) for t in range(len(self.dt_opt) + 1)])
        print("Time array shape: ", self.time_array.shape)
        # Make last control input zero
        self.u_opt = np.append(self.u_opt, np.zeros((1, self.u_opt.shape[1])), axis=0)
        print("Final U shape: ", self.u_opt.shape)

        # Setup Drake Multibody Plant
        self.builder = DiagramBuilder()
        self.SC_plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.0)
        Parser(self.SC_plant).AddModelFromFile(self.SCpath)
        # Remove gravity
        plantGravityField = self.SC_plant.gravity_field()
        plantGravityField.set_gravity_vector([0, 0, 0])
        self.SC_plant.Finalize()
        # Some Sanity Checks
        assert (
            self.SC_plant.num_positions() == self.nq
        ), "Number of positions in URDF model and trajectory data are inconsistent"
        assert (
            self.SC_plant.num_velocities() == self.nqd
        ), "Number of velocities in URDF model and trajectory data are inconsistent"

        # Create AutoDiff version of the plant for Linearization
        self.SC_plant_ad = self.SC_plant.ToAutoDiffXd()

    def synthesizeTVLQRController(self):
        print("Synthesizing TVLQR Controller...")
        self._generateLinearMatrixPolynomials()  # type: ignore
        # Create the Time-Varying Linear System that will be used for the controller synthesis.
        timeVaryingSCSystem = TrajectoryAffineSystem(
            self.ALinPoly, self.BLinPoly, self.f0LinPoly, self.CLinPoly, self.DLinPoly, self.y0LinPoly  # type: ignore
        )
        linSysContext = timeVaryingSCSystem.CreateDefaultContext()
        # Get Terminal Cost from the final Time-Invariant LQR Controller
        Q_tilqr = self._synthesizeFinalLQRController()  # type: ignore

        tvlqrOptions = FiniteHorizonLinearQuadraticRegulatorOptions()
        tvlqrOptions.x0 = self.x0Poly  # type: ignore
        tvlqrOptions.u0 = self.uPoly  # type: ignore
        tvlqrOptions.Qf = np.diag(np.diag(Q_tilqr))
        # tvlqrOptions.Qf = Q_tilqr
        tvlqrOptions.use_square_root_method = False  # We don't use it at the moment.
        # tvlqrOptions.simulator_config.max_step_size = 0.2
        tvlqrOptions.simulator_config.use_error_control = False
        tvlqrOptions.simulator_config.integration_scheme = "implicit_euler"
        # tvlqrOptions.simulator_config.accuracy = 1e-2
        tvlqrOptions.input_port_index = timeVaryingSCSystem.get_input_port().get_index()

        # Load the gain matrices from the config file
        Q_diag = np.hstack(
            (
                self.config["tvlqr_options"]["Q_tvlqr_base_rot_pos"],
                self.config["tvlqr_options"]["Q_tvlqr_base_lin_pos"],
                self.config["tvlqr_options"]["Q_tvlqr_joint_pos"],
                self.config["tvlqr_options"]["Q_tvlqr_base_rot_vel"],
                self.config["tvlqr_options"]["Q_tvlqr_base_lin_vel"],
                self.config["tvlqr_options"]["Q_tvlqr_joint_vel"],
            )
        )

        R_diag = np.hstack(
            (
                self.config["tvlqr_options"]["R_tvlqr_base_torque"],
                self.config["tvlqr_options"]["R_tvlqr_base_force"],
                self.config["tvlqr_options"]["R_tvlqr_joint_torque"],
            )
        )

        Q = np.diag(Q_diag)
        R = np.diag(R_diag)

        self.tvlqr_results = FiniteHorizonLinearQuadraticRegulator(
            system=timeVaryingSCSystem,
            context=linSysContext,
            t0=self.ALinPoly.start_time(),  # type: ignore
            tf=self.ALinPoly.end_time(),  # type: ignore
            Q=Q,
            R=R,
            options=tvlqrOptions,
        )

    def runSimulation(
        self,
        ControllerOn=True,
        tumbleRate=np.array([0, np.deg2rad(5), 0]),
        visualize=True,
        saveFileName="sim_results.p",
    ):
        """
        Setup the Drake simulation using Diagram Builder and connect the controller to the plant.
        If ControllerOn is False, the controller is not connected to the plant. Just the torques
        are applied as open-loop control inputs.
        The tumble rate is set to the tumbleRate value. This will be used in future for checking
        the robustness of the detumble controller.
        """
        # pylance: disable-reportUnboundVariable
        print("Setting up Blocks and Diagram for Simulation...")

        CustomTVLQRController = self._setCustomTVLQRDiagram()  # type: ignore
        if ControllerOn:
            # Add TVLQR and Quaternion State Selector Systems to the Diagram
            TVLQRController = self.builder.AddNamedSystem("CustomTVLQRController", CustomTVLQRController)
            stateSelector = self.builder.AddNamedSystem(
                "QuatStateSelector", QuaternionLQRStateSelector(self.SC_plant.num_multibody_states())
            )
            # Setup Wiring of the diagram
            self.builder.Connect(self.SC_plant.get_state_output_port(), stateSelector.get_input_port())
            self.builder.Connect(stateSelector.get_output_port(), TVLQRController.GetInputPort("xMinusX0"))
            self.builder.Connect(
                TVLQRController.get_output_port(), self.SC_plant.get_applied_generalized_force_input_port()
            )
            input_logger = LogVectorOutput(TVLQRController.get_output_port(0), self.builder)
        if visualize:
            meshcat = StartMeshcat()  # type: ignore
            # Setup Visualization
            meshcat.Delete()
            meshcat.DeleteAddedControls()
            # meshcat = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url, delete_prefix_on_load=True)
            visualizer = MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, meshcat)  # type: ignore

        # Setup Loggers for recording data during simulation
        state_logger = LogVectorOutput(self.SC_plant.get_state_output_port(), self.builder)
        acc_logger = LogVectorOutput(self.SC_plant.get_generalized_acceleration_output_port(), self.builder)

        # Build the final diagram for simulation
        diagram = self.builder.Build()

        # Save the built diagram for Debugging/Understanding.
        graph = pydot.graph_from_dot_data(diagram.GetGraphvizString())
        graph[0].write_svg("diagram.svg")

        # Run Simulation
        duration = self.uPoly.end_time()

        print("Starting Simulation with the following tumble rate (deg/s): ", np.rad2deg(tumbleRate))

        # Set the initial conditions for the simulation
        # init_q_dot = self._computeQDotFromGJM(self.q_opt[0], tumbleRate)
        init_q_dot = self.qd_opt[0]

        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        SC_plant_sim_context = self.SC_plant.GetMyMutableContextFromRoot(context)
        self.SC_plant.get_actuation_input_port().FixValue(SC_plant_sim_context, np.zeros(self.nu - 6))
        SC_plant_sim_context.SetContinuousState(np.append(self.q_opt[0], init_q_dot))
        context.SetTime(0.0)
        simulator.Initialize()
        if ControllerOn:
            inputLog_ctx = input_logger.GetMyMutableContextFromRoot(context)
            input_logger.GetLog(inputLog_ctx).Clear()
        stateLog_ctx = state_logger.GetMyMutableContextFromRoot(context)
        accLog_ctx = acc_logger.GetMyMutableContextFromRoot(context)
        state_logger.GetLog(stateLog_ctx).Clear()
        acc_logger.GetLog(accLog_ctx).Clear()
        if visualize:
            visualizer.StartRecording()
        simulator.AdvanceTo(duration)
        if visualize:
            visualizer.StopRecording()
            visualizer.PublishRecording()
            input("Visualization ready in the link above. Press Enter to exit visualization and save data...")
        # Collect Data
        state_log = copy.deepcopy(state_logger.GetLog(stateLog_ctx).data())
        acc_log = copy.deepcopy(acc_logger.GetLog(accLog_ctx).data())
        sim_time = copy.deepcopy(state_logger.GetLog(stateLog_ctx).sample_times())
        if ControllerOn:
            input_log = copy.deepcopy(input_logger.GetLog(inputLog_ctx).data())
        else:
            input_log = None
        self.simulation_results = {}
        self.simulation_results["state_log"] = state_log
        self.simulation_results["acc_log"] = acc_log
        self.simulation_results["sim_time"] = sim_time
        self.simulation_results["input_log"] = input_log
        self.simulation_results["tumble_rate"] = tumbleRate
        pickle.dump(self.simulation_results, open(saveFileName, "wb"))

    def plotResults(self, plotPositions=False, plotForces=True, plotVelocities=True):
        """
        Plot the results from the simulation.
        """
        state_log = self.simulation_results["state_log"]
        input_log = self.simulation_results["input_log"]
        sim_time = self.simulation_results["sim_time"]
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.size": 20,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "legend.fontsize": 16,
                "legend.handlelength": 1,
                "font.family": "serif",
            }
        )
        lineWidth = 0.5
        markerType = "."
        markerSize = 5

        if plotVelocities:
            # We only plot the angular/rotational velocities
            plt.figure()
            # Omega x
            plt.plot(
                self.time_array,
                self.qd_opt[:, 0],
                marker=markerType,
                markersize=markerSize,
                linewidth=0,
                color="red",
                label=r"$\omega_x^*$",
            )
            plt.plot(
                sim_time,
                state_log[7 + self.armDoF + 0, :].T,
                color="red",
                markersize=markerSize,
                label=r"$\omega_x$",
            )
            # Omega y
            plt.plot(
                self.time_array,
                self.qd_opt[:, 1],
                marker=markerType,
                markersize=markerSize,
                linewidth=0,
                color="green",
                label=r"$\omega_y^*$",
            )
            plt.plot(
                sim_time,
                state_log[7 + self.armDoF + 1, :].T,
                color="green",
                markersize=markerSize,
                label=r"$\omega_y$",
            )
            # Omega z
            plt.plot(
                self.time_array,
                self.qd_opt[:, 2],
                marker=markerType,
                markersize=markerSize,
                linewidth=0,
                color="blue",
                label=r"$\omega_z^*$",
            )
            plt.plot(
                sim_time,
                state_log[7 + self.armDoF + 2, :].T,
                color="blue",
                markersize=markerSize,
                label=r"$\omega_z$",
            )
            plt.legend()
            plt.title(r"Base Rotational Velocities")
            plt.xlabel(r"Time $(s)$")
            plt.ylabel(r"Rotation Velocity $(rad/s)$")
            plt.grid(True)

            plt.figure()
            for armIdx in range(self.armDoF):
                plt.plot(
                    self.time_array,
                    self.qd_opt[:, 6 + armIdx],
                    marker=markerType,
                    markersize=markerSize,
                    linewidth=0,
                    label=r"$\dot{q}^*_{%d}$" % (armIdx + 1),
                )
                plt.plot(
                    sim_time,
                    state_log[7 + 6 + self.armDoF + armIdx, :].T,
                    markersize=markerSize,
                    label=r"$\dot{q}_{%d}$" % (armIdx + 1),
                )
            plt.legend()
            plt.title(r"Joint Velocities")
            plt.xlabel(r"Time $(s)$")
            plt.ylabel(r"Rotation Velocity $(rad/s)$")
            plt.grid(True)

        if input_log is not None:
            if plotForces:
                plt.figure()
                # Base Torques
                plt.plot(
                    self.time_array,
                    self.u_opt[:, 0],
                    marker=markerType,
                    markersize=markerSize,
                    linewidth=0,
                    color="red",
                    label=r"$\tau^*_x$",
                )
                plt.plot(sim_time, input_log[0, :].T, markersize=markerSize, label=r"$\tau_x$", color="red")
                plt.plot(
                    self.time_array,
                    self.u_opt[:, 1],
                    marker=markerType,
                    markersize=markerSize,
                    linewidth=0,
                    color="green",
                    label=r"$\tau^*_y$",
                )
                plt.plot(sim_time, input_log[1, :].T, markersize=markerSize, label=r"$\tau_y$", color="green")
                plt.plot(
                    self.time_array,
                    self.u_opt[:, 2],
                    marker=markerType,
                    markersize=markerSize,
                    linewidth=0,
                    color="blue",
                    label=r"$\tau^*_z$",
                )
                plt.plot(sim_time, input_log[2, :].T, markersize=markerSize, label=r"$\tau_z$", color="blue")
                plt.legend()
                plt.title(r"Base Torques")
                plt.xlabel(r"Time $(s)$")
                plt.ylabel(r"Torque $(\tau)$")
                plt.grid(True)

                # Plot the joint torques
                plt.figure()
                for armIdx in range(self.armDoF):
                    plt.plot(
                        self.time_array,
                        self.u_opt[:, 6 + armIdx],
                        marker=markerType,
                        markersize=markerSize,
                        linewidth=0,
                        label=r"$\tau^*_{%d}$" % (armIdx + 1),
                    )
                    plt.plot(
                        sim_time,
                        input_log[6 + armIdx, :].T,
                        markersize=markerSize,
                        label=r"$\tau_{%d}$" % (armIdx + 1),
                    )
                plt.legend()
                plt.title(r"Joint Torques")
                plt.xlabel(r"Time $(s)$")
                plt.ylabel(r"Torque $(\tau)$")
                plt.grid(True)
        else:
            print("No input log found. Skipping plotting of forces.")

        if plotPositions:
            plt.figure()
            # Base Quaternions
            plt.plot(
                self.time_array,
                self.q_opt[:, 0],
                marker=markerType,
                markersize=markerSize,
                linewidth=0,
                color="red",
                label=r"$q_w^*$",
            )
            plt.plot(sim_time, state_log[0, :].T, markersize=markerSize, label=r"$q_w$", color="red")

            plt.plot(
                self.time_array,
                self.q_opt[:, 1],
                marker=markerType,
                markersize=markerSize,
                linewidth=0,
                color="blue",
                label=r"$q_x^*$",
            )
            plt.plot(sim_time, state_log[1, :].T, markersize=markerSize, label=r"$q_x$", color="blue")

            plt.plot(
                self.time_array,
                self.q_opt[:, 2],
                marker=markerType,
                markersize=markerSize,
                linewidth=0,
                color="green",
                label=r"$q_y^*$",
            )
            plt.plot(sim_time, state_log[2, :].T, markersize=markerSize, label=r"$q_y$", color="green")

            plt.plot(
                self.time_array,
                self.q_opt[:, 3],
                marker=markerType,
                markersize=markerSize,
                linewidth=0,
                color="orange",
                label=r"$q_z^*$",
            )
            plt.plot(sim_time, state_log[3, :].T, markersize=markerSize, label=r"$q_z$", color="orange")

            plt.legend()
            plt.title(r"Base Quaternions")
            plt.xlabel(r"Time $(s)$")
            plt.ylabel(r"Position $(m)$")
            plt.grid(True)

            # Plot the joint positions
            plt.figure()
            for armIdx in range(self.armDoF):
                plt.plot(
                    self.time_array,
                    self.q_opt[:, 7 + armIdx],
                    marker=markerType,
                    markersize=markerSize,
                    linewidth=0,
                    label=r"$q^*_{%d}$" % (armIdx + 1),
                )
                plt.plot(
                    sim_time,
                    state_log[7 + armIdx, :].T,
                    markersize=markerSize,
                    label=r"$q_{%d}$" % (armIdx + 1),
                )
            plt.legend()
            plt.title(r"Joint Positions")
            plt.xlabel(r"Time $(s)$")
            plt.ylabel(r"Position $(m)$")
            plt.grid(True)

        plt.show()

    def readResultsFromFile(self, fileName: "str"):
        """
        Read Results from file. Useful for only plotting without doing the simulation again.
        """
        assert fileName.endswith(".p"), "File must be a pickle .p file."
        assert os.path.isfile(fileName), "File does not exist."
        self.simulation_results = pickle.load(open(fileName, "rb"))

    def _synthesizeFinalLQRController(self):
        """
        Synthesize a final LQR controller for the final time step of the trajectory.
        We use the cost matrix from this controller as the final cost matrix for the TVLQR
        controller.
        """
        Q_diag = np.hstack(
            (
                self.config["tvlqr_options"]["Q_tilqr_base_rot_pos"],
                self.config["tvlqr_options"]["Q_tilqr_base_lin_pos"],
                self.config["tvlqr_options"]["Q_tilqr_joint_pos"],
                self.config["tvlqr_options"]["Q_tilqr_base_rot_vel"],
                self.config["tvlqr_options"]["Q_tilqr_base_lin_vel"],
                self.config["tvlqr_options"]["Q_tilqr_joint_vel"],
            )
        )

        R_diag = np.hstack(
            (
                self.config["tvlqr_options"]["R_tilqr_base_torque"],
                self.config["tvlqr_options"]["R_tilqr_base_force"],
                self.config["tvlqr_options"]["R_tilqr_joint_torque"],
            )
        )
        Q = np.diag(Q_diag)
        R = np.diag(R_diag)

        (K, S) = LinearQuadraticRegulator(self.ALinTraj[-1], self.BLinTraj[-1], Q, R)

        return S

    def _generateLinearMatrixPolynomials(self):
        """
        Generate the linear matrix polynomials for the TVLQR controller synthesis.
        """
        self.ALinTraj = []
        self.BLinTraj = []
        CLinTraj = []
        DLinTraj = []
        f0LinTraj = []
        y0LinTraj = []
        uTraj = []
        x0Traj = []

        for tStep in range(self.q_opt.shape[0]):
            Alin, Blin, Clin, Dlin = self._createMultiBodyLinMatrices(tStep)
            if np.isnan(Alin).any() or np.isnan(Blin).any() or np.isnan(Clin).any() or np.isnan(Dlin).any():
                print("Time Step with Nan", tStep)
                print("Optimal q", self.q_opt[tStep])
                print("Optimal q_dot", self.qd_opt[tStep])
                print("Optimal U", self.u_opt[tStep])
                assert False
            self.ALinTraj.append(Alin)
            self.BLinTraj.append(Blin)
            CLinTraj.append(Clin)
            DLinTraj.append(Dlin)
            f0LinTraj.append(np.zeros((Alin.shape[0], 1)))
            y0LinTraj.append(np.zeros((Alin.shape[0], 1)))
            uTraj.append(np.array([self.u_opt[tStep]]).T)
            x0Traj.append(np.array([np.append(self.q_opt[tStep][1:], self.qd_opt[tStep])]).T)

        self.ALinPoly = PiecewisePolynomial.ZeroOrderHold(self.time_array, self.ALinTraj)  # type: ignore
        self.BLinPoly = PiecewisePolynomial.ZeroOrderHold(self.time_array, self.BLinTraj)  # type: ignore
        self.CLinPoly = PiecewisePolynomial.ZeroOrderHold(self.time_array, CLinTraj)  # type: ignore
        self.DLinPoly = PiecewisePolynomial.ZeroOrderHold(self.time_array, DLinTraj)  # type: ignore
        self.f0LinPoly = PiecewisePolynomial.ZeroOrderHold(self.time_array, f0LinTraj)  # type: ignore
        self.y0LinPoly = PiecewisePolynomial.ZeroOrderHold(self.time_array, y0LinTraj)  # type: ignore
        self.uPoly = PiecewisePolynomial.ZeroOrderHold(self.time_array, uTraj)  # type: ignore
        self.x0Poly = PiecewisePolynomial.ZeroOrderHold(self.time_array, x0Traj)  # type: ignore

    def _createMultiBodyLinMatrices(self, tStep):
        # Compute EoM Partial Derivatives
        context_ad = self.SC_plant_ad.CreateDefaultContext()
        forceInputPort = self.SC_plant_ad.get_applied_generalized_force_input_port()
        forceInputPort.FixValue(context_ad, self.u_opt[tStep])
        actuationPort = self.SC_plant_ad.get_actuation_input_port()
        actuationPort.FixValue(context_ad, np.zeros((actuationPort.size(), 1)))
        x_val = np.hstack((self.q_opt[tStep], self.qd_opt[tStep]))
        u_drake = self.u_opt[tStep]
        A_lin, B_lin = self._get_dynamics_gradient_quat_constraint(
            self.SC_plant_ad, context_ad, forceInputPort.get_index(), x_val, u_drake
        )
        Clin = np.eye(A_lin.shape[0])
        Dlin = np.zeros([B_lin.shape[0], B_lin.shape[1]])

        return A_lin, B_lin, Clin, Dlin

    def _get_dynamics_gradient_quat_constraint(self, plant_ad, context_ad, input_port, x_val, u_val):
        nx = context_ad.num_continuous_states()
        quat_vec_val = x_val[1:4]
        x_non_quat_val = x_val[4:]
        x_variable = np.hstack((quat_vec_val, x_non_quat_val))
        xu_val = np.hstack((x_variable, u_val))
        # This is necessary for gradients w.r.t both x and u. Otherwise it is only w.r.t x.
        xu_ad = InitializeAutoDiff(xu_val)
        quat_scalar_ad = np.array([np.sqrt(1 - np.sum(xu_ad[0:3] ** 2))])
        xu_ad = np.vstack((quat_scalar_ad, xu_ad))
        x_ad = xu_ad[:nx]
        u_ad = xu_ad[nx:]
        context_ad.SetContinuousState(x_ad)
        plant_ad.get_input_port(input_port).FixValue(context_ad, BasicVector_[AutoDiffXd](u_ad))
        xdot_ad = plant_ad.EvalTimeDerivatives(context_ad).CopyToVector()
        # The xdot vector also contains the derivative of the scalar part of the quaternion q0. We do not need this gradient/derivative.
        # This gradient/derivative makes the A matrix non-square. We remove it here.
        xdot_ad = xdot_ad[1:]
        AB = ExtractGradient(xdot_ad)
        A = AB[:, : nx - 1]
        B = AB[:, nx - 1 :]
        return A, B

    def _setCustomTVLQRDiagram(self):
        """
        Create the custom TVLQR block of the diagram.
        """

        # Diagram Builder for a custom block for the tvlqr controller
        CustomTVLQRBuilder = DiagramBuilder()
        # Set up Trajectory source for time varying K matrix. Maybe not the nicest way to do this,
        # but it should work. We use the TrajectoryAffineSystem from Drake to create the
        # time-varying K matrix multiplication with the state error vector. -K(t)(x(t)-x0(t))
        ATraj = [np.zeros((self.nqd * 2, self.nqd * 2)) for _ in range(self.q_opt.shape[0])]
        BTraj = [np.zeros((self.nqd * 2, self.nqd * 2)) for _ in range(self.q_opt.shape[0])]
        CTraj = [np.zeros((self.nqd, self.nqd * 2)) for _ in range(self.q_opt.shape[0])]
        f0Traj = [np.zeros((self.nqd * 2, 1)) for _ in range(self.q_opt.shape[0])]
        y0Traj = [np.zeros((self.nqd, 1)) for _ in range(self.q_opt.shape[0])]

        APoly = PiecewisePolynomial.ZeroOrderHold(self.time_array, ATraj)  # type: ignore
        BPoly = PiecewisePolynomial.ZeroOrderHold(self.time_array, BTraj)  # type: ignore
        CPoly = PiecewisePolynomial.ZeroOrderHold(self.time_array, CTraj)  # type: ignore
        DPoly = self.tvlqr_results.K
        f0Poly = PiecewisePolynomial.ZeroOrderHold(self.time_array, f0Traj)  # type: ignore
        y0Poly = PiecewisePolynomial.ZeroOrderHold(self.time_array, y0Traj)  # type: ignore

        timeVaryingMatrixGain = TrajectoryAffineSystem(APoly, BPoly, f0Poly, CPoly, DPoly, y0Poly)

        # Calculate u0-K(x-x0) using these primitives
        x0_source = CustomTVLQRBuilder.AddSystem(TrajectorySource(self.tvlqr_results.x0))
        x0NegativeGain = CustomTVLQRBuilder.AddSystem(Gain(k=np.ones(self.nqd * 2) * -1))
        xMinusX0 = CustomTVLQRBuilder.AddNamedSystem(
            "xMinusX0", Adder(2, self.nqd * 2)
        )  # 2 inputs of state (pos-vel w/o q0) vectors

        kMatrixGain = CustomTVLQRBuilder.AddSystem(timeVaryingMatrixGain)
        kNegativeGain = CustomTVLQRBuilder.AddSystem(Gain(k=np.ones(self.nu) * -1))
        u0MinusK = CustomTVLQRBuilder.AddNamedSystem(
            "u0MinusK", Adder(2, self.nu)
        )  # 2 inputs of num gen forces vectors
        u0_source = CustomTVLQRBuilder.AddSystem(TrajectorySource(self.tvlqr_results.u0))

        CustomTVLQRBuilder.ExportInput(xMinusX0.get_input_port(0), "xMinusX0")
        CustomTVLQRBuilder.ExportOutput(u0MinusK.get_output_port())

        # x0 into neg. gain into sum1 (x-x0)
        CustomTVLQRBuilder.Connect(x0_source.get_output_port(), x0NegativeGain.get_input_port())
        CustomTVLQRBuilder.Connect(x0NegativeGain.get_output_port(), xMinusX0.get_input_port(1))

        # sum1 into K matrix gain
        CustomTVLQRBuilder.Connect(xMinusX0.get_output_port(), kMatrixGain.get_input_port())

        # K matrix gain into neg. gain into sum2
        CustomTVLQRBuilder.Connect(kMatrixGain.get_output_port(), kNegativeGain.get_input_port())
        CustomTVLQRBuilder.Connect(kNegativeGain.get_output_port(), u0MinusK.get_input_port(0))

        # u0 into  into sum2
        CustomTVLQRBuilder.Connect(u0_source.get_output_port(), u0MinusK.get_input_port(1))

        # Log Trajectory for Plotting
        # traj_logger = LogVectorOutput(x0_source.get_output_port(), CustomTVLQRBuilder)

        CustomTVLQRController = CustomTVLQRBuilder.Build()

        return CustomTVLQRController

    def _computeQDotFromGJM(self, q, tumble_rate):
        """
        Compute the initial velocities for the trajectory optimization problem using the
        Generalized Jacobian Matrix (GJM) from the paper:
        Umetani, Y. and Yoshida, K., 1989. Resolved motion rate control of space manipulators with
        generalized Jacobian matrix. IEEE Transactions on robotics and automation, 5(3), pp.303-314.

        The GJM is then used to compute the initial velocities of the arm using the given initial
        position of the system. Using conservation of momentum equations, the initial velocities
        of the base are computed to ensure zero linear momentum of the system as linear momentum is
        not important for the detumble maneuver. We only care about the angular momentum of the
        system.
        """
        GJMContext = self.SC_plant.CreateDefaultContext()
        targetSpatialVelDesired = np.zeros(6)
        targetSpatialVelDesired[:3] = tumble_rate
        self.SC_plant.SetPositions(GJMContext, q)
        world_frame = self.SC_plant.world_frame()
        targetSC_frame = self.SC_plant.GetFrameByName(self.config["target_name_in_urdf"])
        targetSCJacobian = self.SC_plant.CalcJacobianSpatialVelocity(
            context=GJMContext,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=targetSC_frame,
            p_BoBp_B=np.zeros(3),
            frame_A=world_frame,
            frame_E=world_frame,
        )
        massMatrix = self.SC_plant.CalcMassMatrix(GJMContext)
        totalDoF = self.nq - 1
        M_b = massMatrix[0:6, 0:6]
        M_bm = massMatrix[0:6, 6:totalDoF]
        M_m = massMatrix[6:totalDoF, 6:totalDoF]
        J_ChaserSC_TargetSC = targetSCJacobian[0:6, 0:6]
        J_ChaserManip_TargetSC = targetSCJacobian[0:6, 6:]
        GJM = J_ChaserManip_TargetSC - J_ChaserSC_TargetSC.dot(np.linalg.inv(M_b).dot(M_bm))
        xint_arm_vel = np.linalg.pinv(GJM).dot(targetSpatialVelDesired)
        # Momentum Calculations to find base velocities for zero linear momentum of
        # the system. Momentum Equations:
        # H0*u0 + H0m*qm = [L; P]
        # Assuming zero base-velocities:
        # [L; P] = H0m*qm
        LPInitMomentum = M_bm.dot(xint_arm_vel)
        # We only want to cancel out the linear momentum part.
        LPInitMomentum[0:3] = np.zeros(3)
        # Compute base velocities to cancel exactly the linear momentum.
        xint_base_vel = -np.linalg.inv(M_b).dot(LPInitMomentum)
        return np.concatenate((xint_base_vel, xint_arm_vel))
