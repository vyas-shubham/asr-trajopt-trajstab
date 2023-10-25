import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pickle
import toml

from pydrake.all import MultibodyPlant, Parser
from pydrake.all import (
    MathematicalProgram,
    eq,
    SnoptSolver,
    PiecewisePolynomial,
    JacobianWrtVariable,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Simulator,
)

from utils import convert_w_to_quatdot


class PostCapDetumbleTrajectory:
    def __init__(self, paramsFile: "str", targetRotRate: "np.array"):  # type: ignore
        assert os.path.isfile(paramsFile), "paramsFile must be a valid file path"
        self.config = toml.load(paramsFile)
        self.targetRotRate = targetRotRate
        self.SCpath = self.config["paths"]["sc_path"]
        print(os.path.abspath(paramsFile))
        print("Target Name: ", self.config["target_name_in_urdf"])
        assert self.SCpath.endswith(".urdf"), "SC path must be a .urdf file"
        assert os.path.isfile(self.SCpath), "SC path must be a valid file path"
        self.initChaserPos = np.array(self.config["initial_state"]["position"])

        self.SC_plant = MultibodyPlant(time_step=0.0)
        Parser(self.SC_plant).AddModelFromFile(self.SCpath)
        plantGravityField = self.SC_plant.gravity_field()
        plantGravityField.set_gravity_vector([0, 0, 0])  # type: ignore
        self.SC_plant.Finalize()

        # Get Autodiff MultibodyPlant
        self.SC_plant_autodiff = self.SC_plant.ToAutoDiffXd()

        # Model Positions: 4 Quaternion + 3 Position + n-Dof Arm Positions
        self.nq = self.SC_plant.num_positions()
        self.armDoF = self.nq - 7
        # Model Velocities: 3 Angular Velocity + 3 Linear Velocity + n-Dof Arm Velocities
        self.nqd = self.SC_plant.num_velocities()
        # Model Generalized Forces: 3 Torques + 3 Forces + n-Dof Arm Torques
        self.nu = self.SC_plant.get_applied_generalized_force_input_port().size()

        print("Num Positions: {}".format(self.nq))
        print("Num Velocities: {}".format(self.nqd))
        print("Num Gen Forces: {}".format(self.nu))

        # Init the NLP handler
        self.prog = MathematicalProgram()

    def setupOptimizationProblem(self):
        """
        Setup the optimization problem for the trajectory optimization.
        """
        # Declare all the optimization variables.
        self.N = self.config["trajopt_options"]["knot_points"]
        self.h = self.prog.NewContinuousVariables(self.N, name="h")
        # system configuration, generalized velocities, and accelerations
        self.q = self.prog.NewContinuousVariables(rows=self.N + 1, cols=self.nq, name="q")
        self.qd = self.prog.NewContinuousVariables(rows=self.N + 1, cols=self.nqd, name="qd")
        self.qdd = self.prog.NewContinuousVariables(rows=self.N, cols=self.nqd, name="qdd")
        # generalized force inputs
        self.u = self.prog.NewContinuousVariables(rows=self.N, cols=self.nu, name="u")

        # Add all the constraints
        # Add the min/max time constraints
        h_min = self.config["trajopt_options"]["delta_t_min"]
        h_max = self.config["trajopt_options"]["delta_t_max"]
        print("Trajectory Min Time: ", self.N * h_min, "Trajectory Max Time: ", self.N * h_max)
        h_c = self.prog.AddBoundingBoxConstraint([h_min] * self.N, [h_max] * self.N, self.h)
        h_c.evaluator().set_description("deltaT Limits Constraint")
        # In this version we separate integration and dynamics constraints.
        self._addQuaternionUnitLengthConstraints()
        # TODO: Combine Integration and Dynamics constraints with Hermite-Simpson Collocation
        self._addCollocationConstraintsEuler()
        self._addDynamicsConstraints()
        self._addEqualdTConstraint()
        if self.config["trajopt_options"]["use_actuation_limits"]:
            self._addGeneralizedForcesConstraints()
        self.__addKinematicBoundaryConstraints()

        # Add the costs for the optimization problem
        self._addDetumbleTimeCost()
        self._addActuationCosts()
        self._addFinalStateCosts()

    def solveOptimizationProblem(self, saveFileName: "str"):
        """
        Solve the Detumble optimization problem.
        """
        assert self.config["solver_options"]["solver_name"] == "snopt", "Only SNOPT solver is supported at this time."
        solver = SnoptSolver()
        snoptID = SnoptSolver().solver_id()

        # Setup Extra Solver Options
        if "total_integer_workspace" in self.config["solver_options"]:
            self.prog.SetSolverOption(
                snoptID, "Total integer workspace", self.config["solver_options"]["total_integer_workspace"]
            )

        if "total_real_workspace" in self.config["solver_options"]:
            self.prog.SetSolverOption(
                snoptID, "Total real workspace", self.config["solver_options"]["total_real_workspace"]
            )

        if "total_character_workspace" in self.config["solver_options"]:
            self.prog.SetSolverOption(
                snoptID, "Total character workspace", self.config["solver_options"]["total_character_workspace"]
            )

        if "scale_option" in self.config["solver_options"]:
            self.prog.SetSolverOption(snoptID, "Scale option", self.config["solver_options"]["scale_option"])

        if "major_iterations_limit" in self.config["solver_options"]:
            self.prog.SetSolverOption(
                snoptID, "Major iterations limit", self.config["solver_options"]["major_iterations_limit"]
            )

        if "iterations_limit" in self.config["solver_options"]:
            self.prog.SetSolverOption(snoptID, "Iterations limit", self.config["solver_options"]["iterations_limit"])

        if "superbasics_limit" in self.config["solver_options"]:
            self.prog.SetSolverOption(snoptID, "Superbasics limit", self.config["solver_options"]["superbasics_limit"])

        print("Starting Optimization solver................")
        currTime = time.localtime()
        current_time = time.strftime("%H:%M:%S", currTime)
        print(current_time)
        startTime = time.perf_counter()
        result = solver.Solve(self.prog, self._createInitialGuess())  # type: ignore
        solver_details = result.get_solver_details()
        endTime = time.perf_counter()
        print("Time taken for Optimization (minutes): {}".format(np.around((endTime - startTime) / 60.0), 2))
        print("Solver Exit Code: {}".format(solver_details.info))
        if not result.is_success():
            print("Solver failed to find a solution.")
            print("Use Solver Exit Code: {} to debug.".format(solver_details.info))
            print("Trying to print infeasible constraints:")
            infeasible_constraints = result.GetInfeasibleConstraints(self.prog)
            for c in infeasible_constraints:
                print(f"infeasible constraint: {c}")
        assert result.is_success(), "Optimization failed to find a solution."
        print("Optimization Successful.")
        dt_opt = result.GetSolution(self.h)
        u_opt = result.GetSolution(self.u)
        q_opt = result.GetSolution(self.q)
        qd_opt = result.GetSolution(self.qd)
        qdd_opt = result.GetSolution(self.qdd)
        time_traj = np.array([sum(dt_opt[:t]) for t in range(self.N + 1)])
        print("Detumble Trajectory Time: {}".format(time_traj[-1]))
        self.result_data = {}
        self.result_data["q"] = q_opt
        self.result_data["qd"] = qd_opt
        self.result_data["qdd"] = qdd_opt
        self.result_data["h"] = dt_opt
        self.result_data["u"] = u_opt
        pickle.dump(self.result_data, open(saveFileName, "wb"))

    def readResultsFromFile(self, fileName: "str"):
        """
        Read Results from file. Useful for only plotting without doing the optimization again.
        """
        assert fileName.endswith(".p"), "File must be a pickle .p file."
        assert os.path.isfile(fileName), "File does not exist."
        self.result_data = pickle.load(open(fileName, "rb"))

    def plotResults(self, plotPositions=False, plotForces=True, plotVelocities=True):
        """
        Plot the Detumble trajectory optimization results.
        """
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
        dt_opt = self.result_data["h"]
        u_opt = self.result_data["u"]
        q_opt = self.result_data["q"]
        qd_opt = self.result_data["qd"]
        qdd_opt = self.result_data["qdd"]
        N = dt_opt.shape[0]
        time_traj = np.array([sum(dt_opt[:t]) for t in range(N + 1)])
        if plotForces:
            # Plot base torques
            plt.figure(1)
            plt.plot(time_traj[:-1], u_opt[:, 0])
            plt.plot(time_traj[:-1], u_opt[:, 1])
            plt.plot(time_traj[:-1], u_opt[:, 2])
            plt.legend([r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"])
            plt.title("Base Torques")
            plt.xlabel("Time(s)")
            plt.ylabel("Base Torques (Nm)")
            plt.grid(True)

            # Plot base forces
            plt.figure(2)
            plt.plot(time_traj[:-1], u_opt[:, 3])
            plt.plot(time_traj[:-1], u_opt[:, 4])
            plt.plot(time_traj[:-1], u_opt[:, 5])
            plt.legend([r"$F_x$", r"$F_y$", r"$F_z$"])
            plt.title("Base Forces")
            plt.xlabel("Time(s)")
            plt.ylabel("Base Forces (N)")
            plt.grid(True)

            # Plot joint torques
            plt.figure(3)
            for i in range(self.armDoF):
                plt.plot(time_traj[:-1], u_opt[:, i + 6], label=r"$\tau_{}$".format(i))
            plt.title("Joint Torques (Nm)")
            plt.xlabel("Time(s)")
            plt.legend()
            plt.ylabel("Joint Torques (Nm)")
            plt.grid(True)

        if plotPositions:
            # Plot Base Quaternions
            plt.figure(4)
            plt.plot(time_traj, q_opt[:, 0])
            plt.plot(time_traj, q_opt[:, 1])
            plt.plot(time_traj, q_opt[:, 2])
            plt.plot(time_traj, q_opt[:, 3])
            plt.legend([r"$q_w$", r"$q_x$", r"$q_y$", r"$q_z$"])
            plt.xlabel("Time(s)")
            plt.ylabel("Quaternions")
            plt.title("Quaternions")
            plt.grid(True)

            # Plot Base Positions
            plt.figure(5)
            plt.plot(time_traj, q_opt[:, 4])
            plt.plot(time_traj, q_opt[:, 5])
            plt.plot(time_traj, q_opt[:, 6])
            plt.legend([r"$x$", r"$y$", r"$z$"])
            plt.title("Base positions")
            plt.xlabel("Time(s)")
            plt.ylabel("Base Positions (m)")
            plt.grid(True)

            # Plot Joint Positions
            plt.figure(6)
            for i in range(self.armDoF):
                plt.plot(time_traj, q_opt[:, i + 6], label=r"$q_{}$".format(i + 6))
            plt.xlabel("Time(s)")
            plt.ylabel("Joint Positions (rad)")
            plt.legend()
            plt.title("Joint positions")
            plt.grid(True)

        if plotVelocities:
            # Plot Base Angular Velocities
            plt.figure(7)
            plt.plot(time_traj, qd_opt[:, 0])
            plt.plot(time_traj, qd_opt[:, 1])
            plt.plot(time_traj, qd_opt[:, 2])
            plt.legend([r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"])
            plt.title("Base Angular Velocities")
            plt.xlabel("Time(s)")
            plt.ylabel("Base Angular Velocities (rad/s)")
            plt.grid(True)

            # Plot Base Linear Velocities
            plt.figure(8)
            plt.plot(time_traj, qd_opt[:, 3])
            plt.plot(time_traj, qd_opt[:, 4])
            plt.plot(time_traj, qd_opt[:, 5])
            plt.legend([r"$v_x$", r"$v_y$", r"$v_z$"])
            plt.xlabel("Time(s)")
            plt.ylabel("Base Linear Velocities (m/s)")
            plt.title("Base Linear Velocities")
            plt.grid(True)

            # Plot Joint Velocities
            plt.figure(9)
            for i in range(self.armDoF):
                plt.plot(time_traj, qd_opt[:, i + 6], label=r"$qd_{}$".format(i))
            plt.title("Joint Velocities")
            plt.xlabel("Time(s)")
            plt.ylabel("Joint Velocities (rad/s)")
            plt.legend()
            plt.grid(True)

        plt.show()

    def _computeInitVelocities(self):
        """
        If no initial velocities are provided in the config file, then compute the initial
        velocities for the trajectory optimization problem using the Generalized Jacobian
        Matrix (GJM) from the paper:
        Umetani, Y. and Yoshida, K., 1989. Resolved motion rate control of space manipulators with
        generalized Jacobian matrix. IEEE Transactions on robotics and automation, 5(3), pp.303-314.

        The GJM is then used to compute the initial velocities of the arm using the given initial
        position of the system. Using conservation of momentum equations, the initial velocities
        of the base are computed to ensure zero linear momentum of the system as linear momentum is
        not important for the detumble maneuver. We only care about the angular momentum of the
        system.
        """
        if self.config["initial_state"]["velocity"]:
            return self.config["initial_state"]["velocity"]
        else:
            GJMContext = self.SC_plant.CreateDefaultContext()
            targetSpatialVelDesired = np.zeros(6)
            targetSpatialVelDesired[:3] = self.targetRotRate
            self.SC_plant.SetPositions(GJMContext, self.initChaserPos)
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
            totalDoF = 6 + self.armDoF
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

    def _generateFinalPositionGuess(self, dt):
        """
        Generate a guess for the final position using the initial position and the given rotation
        rate. Integrate the initial position for dt using the given rotation rate to get the final
        position guess. We use Drake's simulator to integrate and get a physically consistent guess.
        """
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        Parser(plant).AddModelFromFile(self.SCpath)
        plantGravityField = plant.gravity_field()
        plantGravityField.set_gravity_vector([0, 0, 0])  # type: ignore
        plant.Finalize()
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        plant.SetPositions(plant_context, self.initChaserPos)
        plant.SetVelocities(plant_context, self._computeInitVelocities())
        plant.get_actuation_input_port().FixValue(plant_context, np.zeros(self.armDoF))
        simulator = Simulator(diagram, diagram_context)
        simulator.Initialize()
        simulator.AdvanceTo(dt)
        finalPosition = plant.GetPositions(plant_context)
        return finalPosition

    def _addQuaternionUnitLengthConstraints(self):
        """
        Add the quaternion constraints: unit quaternion constraint and quaternion integration.
        """
        unit_quat_eps = self.config["epsilons"]["unit_quaternion"]
        q1_unit_c = self.prog.AddBoundingBoxConstraint([-1.0] * (self.N + 1), [1.0] * (self.N + 1), self.q[:, 0])
        q1_unit_c.evaluator().set_description("q1 Unit Constraint")
        q2_unit_c = self.prog.AddBoundingBoxConstraint([-1.0] * (self.N + 1), [1.0] * (self.N + 1), self.q[:, 1])
        q2_unit_c.evaluator().set_description("q2 Unit Constraint")
        q3_unit_c = self.prog.AddBoundingBoxConstraint([-1.0] * (self.N + 1), [1.0] * (self.N + 1), self.q[:, 2])
        q3_unit_c.evaluator().set_description("q3 Unit Constraint")
        q4_unit_c = self.prog.AddBoundingBoxConstraint([-1.0] * (self.N + 1), [1.0] * (self.N + 1), self.q[:, 3])
        q4_unit_c.evaluator().set_description("q4 Unit Constraint")
        # Add the quaternion integration constraint
        for t in range(self.N):
            # Quaternion unit length constraint
            (
                self.prog.AddConstraint(lambda x: [x @ x], [1 - unit_quat_eps], [1 + unit_quat_eps], self.q[t + 1, 0:4])  # type: ignore
                .evaluator()
                .set_description(f"q[{t}] unit quaternion constraint")
            )

    def _addCollocationConstraintsEuler(self):
        """
        Add the backwards Euler integration constraints for the dynamics.
        """
        for t in range(self.N):
            # Quaternion integration using the Betts Book method.
            # Quaternion (q1, q2, q3) integration
            varsq = np.concatenate((self.q[t, 0:4], self.qd[t + 1, 0:3]))
            (
                self.prog.AddConstraint(
                    eq(self.q[t + 1, 1:4], self.q[t, 1:4] + self.h[t] * convert_w_to_quatdot(varsq))
                )
                .evaluator()
                .set_description("w to quat_dot")
            )
            # Position Integration
            self.prog.AddConstraint(eq(self.q[t + 1, 4:], self.q[t, 4:] + self.h[t] * self.qd[t + 1, 3:]))

            # Velocity Integration
            # From Betts Notebook and from Drake Compass gait exercise.
            self.prog.AddConstraint(
                eq(self.qd[t + 1], self.qd[t] + self.h[t] * self.qdd[t])
            ).evaluator().set_description("vel euler integration")

    def _addDynamicsConstraints(self):
        """
        Add manipulator/multibody dynamics equation as a constraint.
        """
        dyn_eqn_eps = self.config["epsilons"]["dynamics_equation"]
        for t in range(self.N):
            vars = np.concatenate((self.q[t + 1], self.qd[t + 1], self.qdd[t], self.u[t]))
            c = self.prog.AddConstraint(
                self._manipulator_equations, lb=[-dyn_eqn_eps] * self.nqd, ub=[dyn_eqn_eps] * self.nqd, vars=vars  # type: ignore
            )
            c.evaluator().set_description("dynamics eqn")

    def _addEqualdTConstraint(self):
        """
        If needed, equal time steps constraint.
        """
        equal_dt_active = self.config["trajopt_options"]["equal_dt"]
        if equal_dt_active:
            for t in range(self.N):
                # Equal deltaT= h
                self.prog.AddLinearConstraint(self.h[t] == self.h[t - 1])

    def _addGeneralizedForcesConstraints(self):
        """
        Add the generalized forces constraints. These are the force/torque limits.
        """
        base_torque_limit = np.array(self.config["joint_limits"]["base_torque_limit"])
        base_force_limit = np.array(self.config["joint_limits"]["base_force_limit"])
        joint_torque_limit = self.config["joint_limits"]["joint_torque_limit"]
        base_gen_force_limit = np.concatenate((base_torque_limit, base_force_limit))
        # Loop over all knots points to apply constraints to
        # TODO: Instead of this loop, use self.u[:, baseF] and self.u[:, jointT+6]
        for t in range(self.N):
            # Loop over 6 generalized base forces
            for baseF in range(6):
                self.prog.AddBoundingBoxConstraint(
                    -base_gen_force_limit[baseF], base_gen_force_limit[baseF], self.u[t, baseF]
                ).evaluator().set_description("u{} limit".format(baseF))
            # Loop over remaining n-6 joint torques.
            for jointT in range(self.u.shape[1] - 6):
                self.prog.AddBoundingBoxConstraint(
                    -joint_torque_limit, joint_torque_limit, self.u[t, jointT + 6]
                ).evaluator().set_description("u{} min".format(jointT + 6))

    def __addKinematicBoundaryConstraints(self):
        """
        Add the kinematic constraints. These are usually the boundary constraints for detumbling.
        While we have an initial position (capture configuration) and initial velocity from GJM,
        and zero initial acceleration, we do not have a final position or velocity.
        For detumbling, we want to constrain the final velocity to be as close to zero as possible.
        """
        xf_vel = np.zeros(self.nqd)
        final_vel_eps = self.config["epsilons"]["final_velocity"]
        # Initial Conditions
        c1 = self.prog.AddLinearConstraint(self.q[0, :], lb=self.initChaserPos, ub=self.initChaserPos)
        c1.evaluator().set_description("init pos")
        init_vel = self._computeInitVelocities()
        c2 = self.prog.AddLinearConstraint(self.qd[0, :], lb=init_vel, ub=init_vel)
        c2.evaluator().set_description("init vel")
        c3 = self.prog.AddLinearConstraint(self.qdd[0, :], lb=xf_vel, ub=xf_vel)
        c3.evaluator().set_description("init acc")

        # Final Conditions
        c4 = self.prog.AddLinearConstraint(self.qd[-1, :], lb=xf_vel - final_vel_eps, ub=xf_vel + final_vel_eps)
        c4.evaluator().set_description("final vel")

    def _addDetumbleTimeCost(self):
        """
        Add cost on the total time of detumble
        """
        timeCostWeight = self.config["trajopt_cost_weights"]["total_time"]
        # Time cost to find minimum time solution.
        totalTime = np.sum(self.h)
        self.prog.AddCost(totalTime * timeCostWeight).evaluator().set_description("Total time cost.")

    def _addActuationCosts(self):
        """
        Add the costs on the generalized forces/actuation: base forces, base torques, and joint
        torques. we use euler integration for these costs.
        """
        costWeightBaseTorque = self.config["trajopt_cost_weights"]["base_torques"]
        costWeightBaseForce = self.config["trajopt_cost_weights"]["base_forces"]
        costWeightJointTorque = self.config["trajopt_cost_weights"]["joint_torques"]
        # Cost on the base torques
        for rotIdx in range(3):
            totalQuadraticEffort = self.u[:, rotIdx] ** 2
            totalEffort = np.sum(np.multiply(totalQuadraticEffort, self.h))
            (self.prog.AddCost(totalEffort * costWeightBaseTorque).evaluator().set_description("Base Torque cost."))
        # Cost on the base forces
        for tranIdx in range(3, 6):
            totalQuadraticEffort = self.u[:, tranIdx] ** 2
            totalEffort = np.sum(np.multiply(totalQuadraticEffort, self.h))
            (self.prog.AddCost(totalEffort * costWeightBaseForce).evaluator().set_description("Base Force cost."))
        # Cost on the joint torques
        for jointIdx in range(6, 6 + self.armDoF):
            totalQuadraticEffort = self.u[:, jointIdx] ** 2
            totalEffort = np.sum(np.multiply(totalQuadraticEffort, self.h))
            (self.prog.AddCost(totalEffort * costWeightJointTorque).evaluator().set_description("Joint Torque cost."))

    def _addFinalStateCosts(self):
        """
        Add cost on the final state for detumbling.
        """
        finalBaseVelocityCostWeight = self.config["trajopt_cost_weights"]["final_base_velocity"]
        finalJointVelocityCostWeight = self.config["trajopt_cost_weights"]["final_joint_velocity"]
        # Cost on Base Final Velocity
        quadraticVelocityCost = np.sum(self.qd[-1, :6] ** 2)
        (
            self.prog.AddCost(quadraticVelocityCost * finalBaseVelocityCostWeight)
            .evaluator()
            .set_description("Final Base Velocity Costs.")
        )

        # Cost on Joint Final Velocity
        quadraticVelocityCost = np.sum(self.qd[-1, 6:] ** 2)
        (
            self.prog.AddCost(quadraticVelocityCost * finalJointVelocityCostWeight)
            .evaluator()
            .set_description("Final Base Velocity Costs.")
        )

    def _createInitialGuess(self):
        """
        Create an initial guess for the solver. If previous solution exists, use that from a
        python pickle (".p") file. Else, use linear interpolation to create an initial guess.

        TODO: Use PiecewisePolynomial to interpolate initial ggess from file. This allows the
        initial guess to have different number knot points than current optimization and still
        be useful.
        """
        initial_guess = np.empty(self.prog.num_vars())
        useInitialGuess = self.config["trajopt_options"]["use_initial_guess"]
        if useInitialGuess:
            # Initial guess using the results from the previous optimization
            initialGuessFile = self.config["paths"]["initial_guess"]
            assert initialGuessFile.endswith(".p"), "Initial guess file must be a pickle .p file."
            assert os.path.isfile(initialGuessFile), "Initial guess file does not exist."
            init_data = pickle.load(open(initialGuessFile, "rb"))
            q_guess = init_data["q"]
            qd_guess = init_data["qd"]
            qdd_guess = init_data["qdd"]
            h_guess = init_data["h"]
            u_guess = init_data["u"]
        else:
            # Initial guess using linear interpolation of states and max time step

            # initial guess for the time step
            h_guess = self.config["trajopt_options"]["delta_t_max"]

            finalPositionGuess = self._generateFinalPositionGuess(self.N * h_guess)

            # linear interpolation of the configuration
            q_guess_poly = PiecewisePolynomial.FirstOrderHold(
                [0, self.N * h_guess],  # type: ignore
                np.column_stack((self.initChaserPos, finalPositionGuess)),
            )

            qd_guess_poly = PiecewisePolynomial.FirstOrderHold(
                [0, self.N * h_guess],  # type: ignore
                np.column_stack((self._computeInitVelocities(), np.zeros(self.nqd))),
            )

            qdd_guess_poly = qd_guess_poly.derivative()

            # # set initial guess for configuration, velocity, and acceleration
            q_guess = np.hstack([q_guess_poly.value(t * h_guess) for t in range(self.N + 1)]).T
            qd_guess = np.hstack([qd_guess_poly.value(t * h_guess) for t in range(self.N + 1)]).T
            qdd_guess = np.hstack([qdd_guess_poly.value(t * h_guess) for t in range(self.N)]).T

            h_guess = [h_guess] * self.N

        # Assign Initial Guess to Mathematical Program
        self.prog.SetDecisionVariableValueInVector(self.h, h_guess, initial_guess)
        self.prog.SetDecisionVariableValueInVector(self.q, q_guess, initial_guess)
        self.prog.SetDecisionVariableValueInVector(self.qd, qd_guess, initial_guess)
        self.prog.SetDecisionVariableValueInVector(self.qdd, qdd_guess, initial_guess)

        return initial_guess

    def _manipulator_equations(self, vars):
        """
        Function that given the current configuration, velocity, and acceleration, evaluates the
        manipulator equations. The output of this function is a vector with dimensions equal to the
        number of configuration variables. If the output of this function is equal to zero then the
        given arguments verify the manipulator equations.

        Input: 1-D Numpy Array: [nq, nqd, nqdd, nu]
        Output: nqx1 1-D Array showing the deviations from satisfying the manipulator equations.
        """

        # split input vector in sub-variables
        # configuration, velocity, acceleration, stance-foot force
        assert vars.size == self.nq + self.nqd + self.nqd + self.nu
        split_at = [self.nq, self.nq + self.nqd, self.nq + self.nqd + self.nqd]
        q, qd, qdd, u = np.split(vars, split_at)
        SC_plant_eval = self.SC_plant_autodiff if self.q.dtype == np.object else self.SC_plant  # type: ignore

        # set SC_plant state
        context = SC_plant_eval.CreateDefaultContext()
        SC_plant_eval.SetPositions(context, q)
        SC_plant_eval.SetVelocities(context, qd)

        # matrices for the manipulator equations
        M = SC_plant_eval.CalcMassMatrixViaInverseDynamics(context)
        Cv = SC_plant_eval.CalcBiasTerm(context)
        tauG = SC_plant_eval.CalcGravityGeneralizedForces(context)

        # return violation of the manipulator equations
        return M.dot(qdd) + Cv - tauG - u
