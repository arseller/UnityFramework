import matplotlib.pyplot as plt

from mlagents_envs.environment import UnityEnvironment


def main():
    print('Connecting to Unity Environment...')
    # This is a non-blocking call that only loads the environment.
    env = UnityEnvironment(file_name=None, seed=1, side_channels=[])

    # Start interacting with the environment.
    env.reset()

    # First Behavior
    behavior_name = list(env.behavior_specs)[0]
    print('Name of first behavior :', behavior_name)

    # Behavior specs
    spec = env.behavior_specs[behavior_name]

    # Observations
    print('Observations properties:', spec.observation_specs)

    # Visual Observations
    vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
    print('Visual observation?', vis_obs)

    # Action
    if spec.action_spec.continuous_size > 0:
        print(f'There are {spec.action_spec.continuous_size} continuous actions')
    if spec.action_spec.is_discrete():
        print(f'There are {spec.action_spec.discrete_size} discrete actions')

    # Number of Discrete Actions
    if spec.action_spec.discrete_size > 0:
        for action, branch_size in enumerate(spec.action_spec.discrete_branches):
            print(f'Action number {action} has {branch_size} different options')

    # Get Step
    """
    This will not move the simulation forward.
    """
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    print('Number of Agents :', len(decision_steps))

    # Set Action
    """
    Need to specify the behavior name and pass a tensor of dimension 2. 
    The first dimension of the action must be equal to the number of Agents that requested a decision during the step.
    """
    env.set_actions(behavior_name, spec.action_spec.empty_action(len(decision_steps)))

    # Step
    env.step()
    """
    Move the simulation forward. 
    The simulation will progress until an Agent requests a decision or terminates.
    """

    # Observations
    """
    DecisionSteps.obs is a tuple containing all of the observations for all of the Agents with the provided Behavior 
    name. 
    Each value in the tuple is an observation tensor containing the observation data for all of the agents.
    """

    for index, obs_spec in enumerate(spec.observation_specs):
        if len(obs_spec.shape) == 3:
            print("Here is the first visual observation")
            plt.imshow(decision_steps.obs[index][0, :, :, :])
            plt.show()

    for index, obs_spec in enumerate(spec.observation_specs):
        if len(obs_spec.shape) == 1:
            print("First vector observations : ", decision_steps.obs[index][0, :])

    # Run Environment
    print('\nRunning Environment ...')
    for episode in range(3):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        tracked_agent = -1  # -1 indicates not yet tracking
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent
        while not done:
            # Track the first agent we see if not tracking
            # Note : len(decision_steps) = [number of agents that requested a decision]
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # Generate an action for all agents
            action = spec.action_spec.random_action(len(decision_steps))

            # Set the actions
            env.set_actions(behavior_name, action)

            # Move the simulation forward
            env.step()

            # Get the new simulation results
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if tracked_agent in decision_steps:  # The agent requested a decision
                episode_rewards += decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps:  # The agent terminated its episode
                episode_rewards += terminal_steps[tracked_agent].reward
                done = True

        print(f"Total rewards for episode {episode} is {episode_rewards}")

    # Close Environment
    env.close()
    print("\nClosed environment")


if __name__ == '__main__':
    main()
