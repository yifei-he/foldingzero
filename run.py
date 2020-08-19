import random
from collections import deque
from foldenv import foldenv
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from ruct import Node
from hpnet import hpnet
from math import sqrt
import matplotlib.pyplot as plt
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calc_upper(s):
    odd, even = 0, 0
    for ii in range(len(s)):
        if s[ii] == 'H':
            if ii % 2 == 0:
                even += 1
            else:
                odd += 1
    return 2 * min(odd, even)


def make_input(node, inp_env):
    states = []
    prev_node = node
    for ii in range(4):
        if prev_node is not None:
            states.append(np.copy(prev_node.state))
            prev_node = prev_node.parent
        else:
            states.append(np.zeros((4, inp_env.grid_length, inp_env.grid_length), dtype=int))
    inp = np.zeros((17, inp_env.grid_length, inp_env.grid_length), dtype=int)
    for ii in range(16):
        inp[ii] = states[int(ii / 4)][ii % 4]
    if len(inp_env.state) < len(inp_env.seq) and inp_env.seq[len(inp_env.state)] == 'H':
        inp[16] = np.ones((inp_env.grid_length, inp_env.grid_length), dtype=int)
    return inp


def make_input_it(sample, it_seq):
    inputs = torch.zeros(len(sample), 17, grid_length, grid_length)
    states = []
    for k in range(len(sample)):
        states.append(np.copy(sample[k][0]))
        if k - 3 < 0:
            for ii in range(3 - k):
                states.append(np.zeros((4, grid_length, grid_length), dtype=int))
        else:
            for kk in range(1, 4):
                states.append(np.copy(sample[k - kk][0]))
        inp = np.zeros((17, grid_length, grid_length), dtype=int)
        for jj in range(16):
            inp[jj] = states[int(jj / 4)][jj % 4]
        if it_seq[k + 1] == 'H':
            inp[16] = np.ones((grid_length, grid_length), dtype=int)
        inputs[k] = torch.Tensor(np.copy(inp))
    return inputs.to(device)


# hyperparameters
num_episode = 50
sim_round = 100
c_alpha = 1
rewards = []
loss_list = []
inter_steps = deque()
deque_size = 512
sample_size = 32
eval_step = 32
grid_length = 63


def simulate_round(node, sim_env, r, model):
    sim_model = model
    r_upper = r
    if node.is_leaf():
        node_state = make_input(node, sim_env)
        inp = torch.Tensor(node_state).unsqueeze(0).to(device)
        p, v = sim_model(inp)

        # append children nodes if exists
        if node.depth < len(sim_env.seq):
            forward_state = sim_env.imagine_next(0, node_state, sim_env.seq[node.depth])
            left_state = sim_env.imagine_next(1, node_state, sim_env.seq[node.depth])
            right_state = sim_env.imagine_next(2, node_state, sim_env.seq[node.depth])
            # only append possible moves (no collision or trap)
            if forward_state is not None:
                forward_node = Node(forward_state, 0, p[0].item(), node)
                node.children.append(forward_node)
            if left_state is not None:
                left_node = Node(left_state, 1, p[1].item(), node)
                node.children.append(left_node)
            if right_state is not None:
                right_node = Node(right_state, 2, p[2].item(), node)
                node.children.append(right_node)
            if len(node.children) == 0:
                node.backprop(-0.5 * len(sim_env.seq))
            # backprop the reward
            else:
                node.backprop(v[0].item())
        # backprop with the true reward if completely folded
        else:
            final_reward = np.sum(node.state[3])
            node.backprop(final_reward)

    # select which child node to go next
    else:
        a_t = []
        tot_visit = 0
        for ch in node.children:
            tot_visit += ch.visit
        for ch in node.children:
            if r_upper == 0:
                res = c_alpha * ch.prob * sqrt(tot_visit) / (1 + ch.visit)
            else:
                res = ch.mean_reward / r_upper + c_alpha * ch.prob * sqrt(tot_visit) / (1 + ch.visit)
            a_t.append(res)
        # randomly select a children if evaluation value is the same
        m = max(a_t)
        max_res = [ii for ii, jj in enumerate(a_t) if jj == m]
        if len(max_res) == 1:
            next_node = node.children[max_res[0]]
        else:
            next_node = node.children[random.randint(0, len(max_res) - 1)]
        # recursively call the function until it reaches a leaf
        simulate_round(next_node, sim_env, r_upper, sim_model)


def update_model():
    if len(inter_steps) < sample_size:
        return

    sample = random.sample(inter_steps, sample_size)
    for sa in sample:
        v_true = sa[0][2] * torch.ones(len(sa), 1)
        inp = make_input_it(sa, sa[0][3]).to(device)
        p_arr, v_arr = cur_model(inp)
        v_lo = nn.MSELoss()(v_arr.to(device), v_true.to(device))
        p_lo = torch.Tensor([0]).to(device)
        for ii in range(len(sa)):
            p_true, p_pred = sa[ii][1].to(device), p_arr[ii]
            p_lo = p_lo - p_true[0] * torch.log(p_pred[0]) - p_true[1] * torch.log(p_pred[1]) \
                   - p_true[2] * torch.log(p_pred[2])
        total_loss = v_lo + p_lo
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print('loss: ' + str(total_loss))
        loss_list.append(total_loss)


def train_episode(episode_num, root, train_env):
    cur_node = root
    iteration_info = []
    r_upper = calc_upper(train_env.seq)
    while True:
        # env.render()
        for _ in range(sim_round):
            simulate_round(cur_node, train_env, r_upper, cur_model)
        pi = torch.zeros(3)
        total_visit = 0
        for c in cur_node.children:
            total_visit += c.visit

        # deal with situation where not all 3 possible moves are possible
        action_to_idx = {}
        act_idx = 0
        for c in cur_node.children:
            pi[c.action] = c.visit / total_visit
            action_to_idx[c.action] = act_idx
            act_idx += 1

        if len(inter_steps) >= deque_size:
            inter_steps.popleft()
        iteration_info.append([np.copy(cur_node.state), pi])

        # print(pi)
        if len(cur_node.children) > 0:
            cur_node = cur_node.children[action_to_idx[np.argmax(pi).item()]]

        action = cur_node.action
        new_state, reward, done, info = train_env.step(action)

        if done:
            for s in iteration_info:
                s.append(reward)
                s.append(train_env.seq)
            inter_steps.append(iteration_info)
            update_model()
            rewards.append(reward)
            print("reward: " + str(reward))
            # plt.pause(0.001)
            # if episode_num % 20 == 0:
            #     plt.figure(2)
            #     plt.plot(np.array(rewards))
            #     plt.savefig('figure/' + str(episode_num) + '.png')
            # if episode_num % 20 == 0 and len(loss_list) > 10:
            #     plt.figure(3)
            #     plt.plot(np.array(loss_list[10:]))
            #     plt.savefig('figure/loss' + str(episode_num) + '.png')
            # plt.pause(0.001)
            # train_env.render()
            break


def test_episode(test_node, test_env, model):
    cur_node = test_node
    r_upper = calc_upper(test_env.seq)
    while True:
        # env.render()
        for _ in range(sim_round):
            simulate_round(cur_node, test_env, r_upper, model)
        pi = torch.zeros(3)
        total_visit = 0
        for c in cur_node.children:
            total_visit += c.visit

        # deal with situation where not all 3 possible moves are possible
        action_to_idx = {}
        act_idx = 0
        for c in cur_node.children:
            pi[c.action] = c.visit / total_visit
            action_to_idx[c.action] = act_idx
            act_idx += 1

        if len(cur_node.children) > 0:
            cur_node = cur_node.children[action_to_idx[np.argmax(pi).item()]]

        action = cur_node.action
        new_state, reward, done, info = test_env.step(action)

        if done:
            return reward


def main():
    seq_idx = 0
    for seq in train_seq:
        env = foldenv(seq)
        state = env.reset()
        root = Node(np.copy(state))

        for i_episode in range(num_episode):
            print('seq index: ' + str(seq_idx) + ' episode: ' + str(i_episode) + ' seq: ' + str(len(seq)))
            env.reset()
            train_episode(i_episode, root, env)

        # compare old and current model periodically
        if seq_idx >= eval_step and seq_idx % eval_step == 0:
            cur_test_rewards = np.zeros((len(test_seq)))
            old_test_rewards = np.zeros((len(test_seq)))
            te_idx = 0
            for te_seq in test_seq:
                te_cur_env = foldenv(te_seq)
                te_cur_state = te_cur_env.reset()
                te_cur_root = Node(np.copy(te_cur_state))

                te_old_env = foldenv(te_seq)
                te_old_state = te_old_env.reset()
                te_old_root = Node(np.copy(te_old_state))

                cur_test_rewards[te_idx] = test_episode(te_cur_root, te_cur_env, cur_model)
                old_test_rewards[te_idx] = test_episode(te_old_root, te_old_env, old_model)
                cur_win_rate = len(np.where(cur_test_rewards >= old_test_rewards)[0]) / len(test_seq)

                te_idx += 1

                # change the current model if the old model wins, always update old model with the current one
                temp_model = hpnet(grid_length)
                temp_model.load_state_dict(cur_model.state_dict())
                if cur_win_rate < 0.5:
                    cur_model.load_state_dict(old_model.state_dict())
                    print('Current loses! Model updated!')
                else:
                    print('Current wins! Model unchanged!')
                old_model.load_state_dict(temp_model.state_dict())

                to_save = {
                    'iteration': seq_idx,
                    'state_dict': cur_model.state_dict()
                }
                torch.save(to_save, "checkpoint/itr{}.checkpoint.pth.tar".format(seq_idx))

        if seq_idx > 0 and seq_idx % 5 == 0:
            for b in benchmark:
                eval_env = foldenv(b)
                eval_state = eval_env.reset()
                eval_root = Node(np.copy(eval_state))
                eval_res = test_episode(eval_root, eval_env, cur_model)
                with open('eval_int_res.txt', 'a') as file:
                    file.write(str(len(b)) + ': ' + str(eval_res) + '\n')
        seq_idx += 1
        
    for b in benchmark:
        eval_env = foldenv(b)
        eval_state = eval_env.reset()
        eval_root = Node(np.copy(eval_state))
        eval_res = test_episode(eval_root, eval_env, cur_model)
        with open('eval_res.txt', 'a') as file:
            file.write(str(len(b)) + ': ' + str(eval_res) + '\n')


if __name__ == '__main__':
    arr = ['H', 'P']

    # make synthetic training set
    train_seq = []
    for i in range(4):
        for j in range(8):
            seq_len = int(abs(math.floor(np.random.normal(25 + 10 * i, 10))))
            train_seq.append(''.join(np.random.choice(arr, seq_len, p=[0.8, 0.2])))
    for i in range(4):
        for j in range(8):
            seq_len = int(abs(math.floor(np.random.normal(25 + 10 * i, 10))))
            train_seq.append(''.join(np.random.choice(arr, seq_len, p=[0.7, 0.3])))

    # make synthetic test set
    test_seq = []
    for i in range(4):
        for j in range(1):
            seq_len = int(abs(math.floor(np.random.normal(25 + 10 * i, 10))))
            test_seq.append(''.join(np.random.choice(arr, seq_len, p=[0.8, 0.2])))
    for i in range(4):
        for j in range(1):
            seq_len = int(abs(math.floor(np.random.normal(25 + 10 * i, 10))))
            test_seq.append(''.join(np.random.choice(arr, seq_len, p=[0.7, 0.3])))

    # load benchmark
    benchmark = ['hhppppphhppphppphp',
                 'hphphhhppphhhhpphh',
                 'phpphphhhphhphhhhh',
                 'hphpphhphpphphhpphph',
                 'hhhpphphphpphphphpph',
                 'hhpphpphpphpphpphpphpphh',
                 'pphpphhpppphhpppphhpppphh',
                 'ppphhpphhppppphhhhhhhpphhpppphhpphpp',
                 'pphpphhpphhppppphhhhhhhhhhpppppphhpphhpphpphhhhh',
                 'hhphphphphhhhphppphppphpppphppphppphphhhhphphphphh',
                 'pphhhphhhhhhhhppphhhhhhhhhhphppphhhhhhhhhhhhpppphhhhhhphhphp',
                 'hhhhhhhhhhhhphphpphhpphhpphpphhpphhpphpphhpphhpphphphhhhhhhhhhhh',
                 'hhhhpppphhhhhhhhhhhhpppppphhhhhhhhhhhhppphhhhhhhhhhhhppphhhhhhhhhhhhppphpphhpphhpphph',
                 'pppppphphhppppphhhphhhhhphhpppphhpphhphhhhhphhhhhhhhhhphhphhhhhhhppppppppppphhhhhhhpphphhhpppppphphh',
                 'ppphhpphhhhpphhhphhphhphhhhpppppppphhhhhhpphhhhhhppppppppphphhphhhhhhhhhhhpphhhphhphpphphhhpppppphhh']
    for i in range(len(benchmark)):
        benchmark[i] = benchmark[i].upper()

    cur_model = hpnet(grid_length)
    old_model = hpnet(grid_length)
    optimizer = optim.SGD(cur_model.parameters(), weight_decay=4e-5, lr=0.001, momentum=0.9)
    main()

# plt.ioff()
# plt.show()
