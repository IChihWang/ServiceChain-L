
import tensorflow as tf
import core as core
#import gym
import numpy as np
import os

from utils import logger
import argparse


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        #self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        # print(self.ptr)
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def is_full(self):
        return self.ptr > self.max_size -10

    def overwrite_last_rew(self, rew):
        self.rew_buf[self.ptr-1] = rew

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        #print(self.ptr)
        #assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.mean_std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]

    def mean_std(self, x):
        x = np.array(x, dtype=np.float32)
        mean = np.mean(x)
        std = np.std(x)
        return mean, std



class Agent():
    def __init__(self, state_dim, action_dim, config_rl, gamma=0.99, train_v_iters=2, steps_per_epoch=1000, lam=0.97):

        self.global_step = tf.train.get_or_create_global_step(graph=None)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pi_lr = config_rl.pi_lr
        self.v_lr = config_rl.v_lr
        print(type(action_dim))
        #exit()
        self.x_ph = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='s0')
        self.a_ph = tf.placeholder(tf.int32, shape=[None], name='a')
        self.train_v_iters = train_v_iters

        self.train_dir = './train_dir'
        self.step_epochs = tf.Variable(0, trainable=False, name='step')
        #self.step_op = tf.assign(self.step_epochs, )

        #self.x_ph, self.a_ph = core.placeholders_from_spaces(state_dim, action_dim)
        self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)

        actor_critic = core.mlp_actor_critic
        self.pi, self.logp, self.logp_pi, self.v, self.entropy = actor_critic(self.x_ph, self.a_ph, action_space=action_dim)
        print("logp", self.logp.shape)
        # Need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

        # Every step, get: action, value, and logprob
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        # Experience buffer
        local_steps_per_epoch = steps_per_epoch #int(steps_per_epoch / num_procs())
        self.buf = VPGBuffer(state_dim, action_dim, local_steps_per_epoch, gamma, lam)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
        #logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # VPG objectives
        self.pi_loss = -tf.reduce_mean(self.logp * self.adv_ph)
        #print((self.logp * self.adv_ph).shape)
        #print((self.pi*self.logp).shape)
        #exit()
        self.pi_loss = -tf.reduce_mean(self.logp * self.adv_ph )  -  0.01*tf.reduce_mean(self.entropy)
        #self.pi_loss = self.logp
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v) ** 2)

        # Info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)  # a sample estimate for KL-divergence, easy to compute
        self.approx_ent = tf.reduce_mean(-self.logp)  # a sample estimate for entropy, also easy to compute

        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr)
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.v_lr)
        self.train_pi = self.actor_optimizer.minimize(self.pi_loss, global_step = self.global_step)
        self.train_v = self.critic_optimizer.minimize(self.v_loss, global_step = self.global_step)

        tf.summary.histogram('log_pi', -self.logp)
        tf.summary.scalar('pi_loss', self.pi_loss)
        tf.summary.scalar('v_loss', self.v_loss)
        tf.summary.scalar('approx_ent', self.approx_ent)
        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=100)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)

    def updat_step_epochs(self, step):

        self.sess.run(tf.assign(self.step_epochs, step))

    def get_step_epochs(self):

        return self.sess.run(self.step_epochs)

    def update(self):
        #[print("buffer shpae:", tmp.shape) for tmp in self.buf.get()]
        #exit()
        inputs = {k: v for k, v in zip(self.all_phs, self.buf.get())}
        #[print(tmp.shape) for tmp in inputs]
        #exit()
        #pi_l_old, v_l_old, ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)

        summary, step = self.sess.run([self.summary_op, self.global_step], feed_dict=inputs)
        self.summary_writer.add_summary(summary, global_step = step)

        #pi_l_old = self.sess.run([self.logp_pi], feed_dict=inputs)

        #exit()
        # Policy gradient step
        self.sess.run(self.train_pi, feed_dict=inputs)

        # Value function learning
        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)

        # Log changes from update
        #pi_l_new, v_l_new, kl = sess.run([self.pi_loss, self.v_loss, self.approx_kl], feed_dict=inputs)
        #logger.store(LossPi=pi_l_old, LossV=v_l_old,
        #             KL=kl, Entropy=ent,
        #             DeltaLossPi=(pi_l_new - pi_l_old),
        #             DeltaLossV=(v_l_new - v_l_old))


    def save_model(self, step=None):
        print("Save model")
        #self.saver.save(self.sess,  os.path.join(self.train_dir, 'model'))
        self.saver.save(self.sess, os.path.join(self.train_dir, 'model'), global_step =step)

    def load_model(self):
        print("Load model")
        #self.saver.restore(self.sess, os.path.join(self.train_dir, 'model'))
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.train_dir))
        print(tf.train.latest_checkpoint(self.train_dir))
        logger.info("Load ckpt: {}".format(tf.train.latest_checkpoint(self.train_dir)))


    def log_tf(self, val, tag=None, step_counter=0):
        summary = tf.Summary()
        summary.value.add(tag= tag, simple_value=val)
        self.summary_writer.add_summary(summary, step_counter)

def train():
    epochs = 150
    episode_counter = 0
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    for epoch in range(epochs):
        print(epoch)
        for t in range(local_steps_per_epoch):
            #print(t, ep_len)
            a, v_t, logp_t = agent.sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})
            #print(a.shape, v_t, logp_t.shape)
            #exit()
            # save and log
            agent.buf.store(o, a, r, v_t, logp_t)
            #logger.store(VVals=v_t)

            o, r, d, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1

            terminal = d
            if terminal or (t == local_steps_per_epoch - 1):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else agent.sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
                agent.buf.finish_path(last_val)
                if terminal:
                    episode_counter += 1
                    print(ep_ret, ep_len)
                    # only save EpRet / EpLen if trajectory finished
                    agent.log_tf(ep_ret, 'Return', episode_counter)
                    #logger.store(EpRet=ep_ret, EpLen=ep_len)



                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        agent.update()

    # save model
    agent.save_model()
    '''
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()
    env.close()
    '''
def eval():
    agent.load_model()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for _ in range(1000):
        env.render()
        a, v_t, logp_t = agent.sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

        # action = env.action_space.sample()  # your agent here (this takes random actions)

        o, r, d, _ = env.step(a[0])

        if d:
            observation = env.reset()
    env.close()

if __name__ == "__main__":

    env = gym.make('CartPole-v1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--pi_lr', type=float, default=1e-3)
    parser.add_argument('--v_lr', type=float, default=1e-3)
    config_rl = parser.parse_args()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(action_dim)
    print(type(env.action_space.n))

    logger.info("test")
    #exit()


    #global_step = tf.train.get_or_create_global_step(graph=None)




    local_steps_per_epoch = 1000

    agent = Agent(state_dim, action_dim, config_rl, steps_per_epoch=local_steps_per_epoch)

    train()
    #eval()
