""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task and developed with tflearn + Tensorflow

Original Author: Patrick Emami
Code copied from https://github.com/rcheng805/RL-CBF

The majority of this code is still the same, however, I have made some modifications to add a different form of logging.
Additionally, the neural network architecture has been modified to a smaller network reported on in
https://arxiv.org/abs/1709.06560 and the batch norm layers have been removed to simplify the verification process.
The final learned model is also saved so it can be analyzed afterwards using NNV.
"""
import os
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp
from scipy.io import savemat
import datetime

from replay_buffer import ReplayBuffer


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.compat.v1.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.compat.v1.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(
            map(lambda x: tf.math.divide(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 64, name='relu1', activation='relu')
        # net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64, name='relu2', activation='relu')
        # net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, name='out_layer', activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        self.model = tflearn.DNN(out)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.compat.v1.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.compat.v1.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 64)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64)
        t2 = tflearn.fully_connected(action, 64)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        """
        Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
        based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.compat.v1.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.compat.v1.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.compat.v1.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================
def evaluate(env, actor, episode_length):
    # Reset the environment
    s = env.reset()

    ep_reward = 0

    # Step through each step of the episode
    done = 0
    steps = episode_length
    for i in range(episode_length):
        a = actor.predict(np.reshape(s, (1, actor.s_dim)))
        s2, r, terminal, info = env.step(a[0])

        s = s2
        ep_reward += r

        if terminal:
            done = 1
            steps = i + 1
            break

    # Return the results
    return steps, ep_reward, done


# ===========================
#   Agent Training
# ===========================
def train(sess, env, args, actor, critic, actor_noise, reward_result, log_name):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.compat.v1.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter(args['summary_dir'], sess.graph)

    restorer = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    paths = list()

    # Extract the arguments that will be used repeatedly
    episode_length = int(args['max_episode_len'])
    max_episodes = int(args['max_episodes'])
    num_evals = int(args['num_evals'])

    # Evaluate initial performance
    for j in range(num_evals):
        steps, reward, done = evaluate(env, actor, episode_length)

        # Log the evaluation run
        with open(log_name, "a") as myfile:
            myfile.write(str(0) + ', ' + str(steps) + ', ' + str(reward) + ', ' + str(done) + '\n')

    for i in range(max_episodes):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        obs, action, rewards = [], [], []

        for j in range(episode_length):

            # Added exploration noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            obs.append(s)
            rewards.append(r)
            action.append(a[0])

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(
                    int(ep_reward), i, (ep_ave_max_q / float(j))))
                reward_result[i] = ep_reward
                path = {"Observation": np.concatenate(obs).reshape((200, 3)),
                        "Action": np.concatenate(action),
                        "Reward": np.asarray(rewards)}
                paths.append(path)

                # Evaluate performance of trained model after the episode
                for k in range(num_evals):
                    steps, reward, done = evaluate(env, actor, episode_length)

                    # Log the evaluation run
                    with open(log_name, "a") as myfile:
                        myfile.write(str(i + 1) + ', ' + str(steps) + ', ' + str(reward) + ', ' + str(done) + '\n')

                break

    # Save the final model as a matlab file
    relu1_vars = tflearn.variables.get_layer_variables_by_name('relu1')
    relu2_vars = tflearn.variables.get_layer_variables_by_name('relu2')
    out_vars = tflearn.variables.get_layer_variables_by_name('out_layer')

    weights = [actor.model.get_weights(relu1_vars[0]), actor.model.get_weights(relu2_vars[0]),
               actor.model.get_weights(out_vars[0])]
    biases = [actor.model.get_weights(relu1_vars[1]), actor.model.get_weights(relu2_vars[1]),
              actor.model.get_weights(out_vars[1])]

    savemat(args['log_path'] + '/final_model.mat', mdict={'W': weights, 'b': biases})

    save_path = args['log_path'] + '/final_model.chkp'
    restorer.save(sess, save_path)

    return [summary_ops, summary_vars, paths]


def main(args, reward_result, log_path):
    with tf.Session() as sess:
        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.compat.v1.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        # Create the log files
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        log_save_name = log_path + '/episode_performance.csv'
        f = open(log_save_name, "w+")
        f.write("episode number, steps in evaluation, accumulated reward, done \n")
        f.close()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        [summary_ops, summary_vars, paths] = train(sess, env, args, actor, critic, actor_noise, reward_result,
                                                   log_save_name)

        # Evaluate the final model 100 times to get a better idea of the final model's performance
        f = open(args['log_path'] + '/final_eval.csv', "w+")
        f.write("reward, steps, dones\n")
        episode_length = int(args['max_episode_len'])
        for k in range(100):
            steps, reward, done = evaluate(env, actor, episode_length)
            f.write(str(reward) + ', ' + str(steps) + ', ' + str(done) + '\n')
        f.close()

        return [summary_ops, summary_vars, paths]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1754)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=600)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=200)
    parser.add_argument('--num-evals', help='number of evaluation runs after each episode', default=5)
    parser.add_argument('--render-env', help='render the gym env', action='store_false')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_false')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results2/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results2/tf_ddpg')
    parser.add_argument('--log-path', help='path to directory where a log will be recorded', default='.')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    pp.pprint(args)

    reward_result = np.zeros(int(args['max_episodes']))
    [summary_ops, summary_vars, paths] = main(args, reward_result, args['log_path'])

    # savemat('data4_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',
    #         dict(data=paths, reward=reward_result))
