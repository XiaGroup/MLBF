# -*- coding: utf-8 -*-
"""
WMMSE algorithm with meta-learning
"""
import numpy as np
from numpy import random
import torch
import torch.nn as nn
from timeit import default_timer as timer
import torch
import torch.nn as nn
import random
from timeit import default_timer as timer
import math
import copy
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import scipy.io as scio
USE_CUDA = False
# if torch.cuda.is_available():
#     USE_CUDA = True
External_iteration = 500
Update_steps=1
Internal_iteration = 1
Epo=1
lr_rate=10e-4
lr_rate_v=1.5*10e-4
# mu_fix=0.04
optimizer_lr_w = lr_rate # changeable
optimizer_lr_u = lr_rate
optimizer_lr_V = lr_rate_v
hidden_size_w=200
hidden_size_u=200
hidden_size_V=500
layer=2
epoch=1
nr_of_training = 50 # used for testing
# nr_of_testing = 100 # used for training
nr_of_iterations = 1 # for WMMSE algorithm in Shi et al.
scheduled_users = [0,1,2,3]
test_rate=0.9
# noise=1
def Adam():
    return torch.optim.Adam()

# Set variables
nr_of_users = 4
nr_of_BS_antennas = 4
selected_users = [0,1,2,3] # array of scheduled users. Note that we schedule all the users.
epsilon = 0.0001 # used to end the iterations of the WMMSE algorithm in Shi et al. when the number of iterations is not fixed (note that the stopping criterion has precendence over the fixed number of iterations)
power_tolerance = 0.0001 # used to end the bisection search in the WMMSE algorithm in Shi et al.
total_power = 1000 # power constraint in the weighted sum rate maximization problem - eq. (4) in our paper
noise_power = 1
# nr_of_batches_training = 10 # used for training
# nr_of_batches_test = 10 # used for testing
# nr_of_samples_per_batch = 10
nr_of_iterations = 10 # for WMMSE algorithm in Shi et al.

M_1=torch.eye(nr_of_users)
M_2=torch.zeros(nr_of_users,nr_of_users)
M_Re=torch.cat((M_1,M_2),dim=0)
# print(M_Re)

M_1=torch.eye(nr_of_users)
M_2=torch.zeros(nr_of_users,nr_of_users)
M_Im=torch.cat((M_2,M_1),dim=0)
# print(M_Im)
# User weights in the weighted sum rate (denoted by alpha in eq. (4) in our paper)
user_weights = np.ones(nr_of_users)
user_weights_for_regular_WMMSE = np.ones(nr_of_users)


def loss_function(receiver_precoder_in,mse_weight,transmitter_precoder_in,channel,noise_power,mu):
    loss=0
    e=0
    # v_complex=transmitter_precoder_in.mm(M_Re.double())+1j*transmitter_precoder_in.mm(M_Im.double())
    # u_complex=receiver_precoder_in.mm(M_Re)+1j*receiver_precoder_in.mm(M_Im)
    # h_com1=channel
    # VVH_complx= v_complex.mm(torch.conj(v_complex).t())
    # VVH_complx=VVH_complx.double()
    # TVV=torch.trace(VVH_complx)
    u_re=receiver_precoder_in.mm(M_Re)
    u_im=receiver_precoder_in.mm(M_Im)
    # u=receiver_precoder_in
    transmitter_precoder_in=transmitter_precoder_in.float()
    V_re=transmitter_precoder_in.mm(M_Re)
    V_im=transmitter_precoder_in.mm(M_Im)
    channel_re=channel.real.float()
    channel_im=channel.imag .float()
    
    
    for user_index_1 in range(nr_of_users):
        u_re1=u_re[:,user_index_1]
        u_im1=u_im[:,user_index_1]
        h_1=channel[user_index_1,:]
        h_1=torch.conj(h_1)
        h_re1=channel_re[user_index_1,:]
        h_re1=h_re1.reshape(1,nr_of_users)
        h_im1=channel_im[user_index_1,:]
        h_im1=-1*h_im1
        h_im1=h_im1.reshape(1,nr_of_users)
        v_re1=V_re[user_index_1,:]
        v_im1=V_im[user_index_1,:]
        v_re1=v_re1.reshape(nr_of_users,1)
        v_im1=v_im1.reshape(nr_of_users,1)
        w_i=mse_weight[user_index_1]
        u_i=u_re1.pow(2)+u_im1.pow(2)#|u_i|^2
        uhvi_Re=u_re1*h_re1.mm(v_re1)-u_re1*h_im1.mm(v_im1)-u_im1*h_im1.mm(v_re1)-u_im1*h_re1.mm(v_im1)
        uhvi_Im=u_re1*h_re1.mm(v_im1)+u_re1*h_im1.mm(v_re1)+u_im1*h_re1.mm(v_re1)-u_im1*h_im1.mm(v_im1)
        uhv_i=(uhvi_Re-1).pow(2)+uhvi_Im.pow(2)#|u_ih^H_iv_i-1|^2
        for user_index_2 in range(nr_of_users):
            if user_index_2 != user_index_1 and user_index_2 in selected_users:
                v_re2=V_re[user_index_2,:]
                v_im2=V_im[user_index_2,:]
                v_re2=v_re2.reshape(nr_of_users,1)
                v_im2=v_im2.reshape(nr_of_users,1)
                uhvj_Re=u_re1*h_re1.mm(v_re2)-u_re1*h_im1.mm(v_im2)-u_im1*h_im1.mm(v_re2)-u_im1*h_re1.mm(v_im2)
                uhvj_Im=u_re1*h_re1.mm(v_im2)+u_re1*h_im1.mm(v_re2)+u_im1*h_re1.mm(v_re2)-u_im1*h_im1.mm(v_im2)
                uhv_j_2=uhvj_Re.pow(2)+uhvj_Im.pow(2)#|u_ih^H_iv_j|^2
                e=e+uhv_j_2
        e=e+uhv_i+u_i+noise_power
        loss=loss+w_i*e-torch.log2(w_i)
    Trace_V=torch.trace(V_re.t().mm(V_re)+V_im.t().mm(V_im))
    loss=loss+mu*Trace_V-mu*total_power#+0.0001*torch.norm(u_im)#+0.001*torch.norm(mse_weight)
    # loss=loss-mu*(total_power-Trace_V)
    return loss




def run_WMMSE(transmitter_precoder, epsilon, channel, selected_users, total_power, noise_power, user_weights, max_nr_of_iterations,
              log=True):
    channel=channel.numpy()
    nr_of_users = np.size(channel, 0)
    nr_of_BS_antennas = np.size(channel, 1)
    WSR = []  # to check if the WSR (our cost function) increases at each iteration of the WMMSE
    break_condition = epsilon + 1  # break condition to stop the WMMSE iterations and exit the while
    receiver_precoder = np.zeros(nr_of_users) + 1j * np.zeros(
        nr_of_users)  # receiver_precoder is "u" in the paper of Shi et al. (it's a an array of complex scalars)
    mse_weights = np.ones(
        nr_of_users)  # mse_weights is "w" in the paper of Shi et al. (it's a an array of real scalars)
    # transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_users,
    #                                                                                    nr_of_BS_antennas))  # transmitter_precoder is "v" in the paper of Shi et al. (it's a complex matrix)

    new_receiver_precoder = np.zeros(nr_of_users) + 1j * np.zeros(nr_of_users)  # for the first iteration
    new_mse_weights = np.zeros(nr_of_users)  # for the first iteration
    new_transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros(
        (nr_of_users, nr_of_BS_antennas))  # for the first iteration

    # Initialization of transmitter precoder (V)
    # for user_index in range(nr_of_users):
    #     if user_index in selected_users:
    #         transmitter_precoder[user_index, :] = channel[user_index, :]
    # transmitter_precoder = transmitter_precoder / np.linalg.norm(transmitter_precoder) * np.sqrt(total_power)

    # Store the WSR obtained with the initialized trasmitter precoder
    WSR.append(compute_weighted_sum_rate_WMMSE(user_weights, channel, transmitter_precoder, noise_power, selected_users))

    # Compute the initial power of the transmitter precoder
    initial_power = 0
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            initial_power = initial_power + (compute_norm_of_complex_array(transmitter_precoder[user_index, :])) ** 2
    if log == True:
        print("Power of the initialized transmitter precoder:", initial_power)

    nr_of_iteration_counter = 0  # to keep track of the number of iteration of the WMMSE

    while break_condition >= epsilon and nr_of_iteration_counter <= max_nr_of_iterations:

        nr_of_iteration_counter = nr_of_iteration_counter + 1
        if log == True:
            print("WMMSE ITERATION: ", nr_of_iteration_counter)

        # Optimize receiver precoder(u) - eq. (5) in the paper of Shi et al.
        for user_index_1 in range(nr_of_users):
            if user_index_1 in selected_users:
                user_interference = 0.0
                for user_index_2 in range(nr_of_users):
                    if user_index_2 in selected_users:
                        user_interference = user_interference + (np.absolute(
                            np.matmul(np.conj(channel[user_index_1, :]), transmitter_precoder[user_index_2, :]))) ** 2

                new_receiver_precoder[user_index_1] = np.matmul(np.conj(channel[user_index_1, :]),
                                                                transmitter_precoder[user_index_1, :]) / (
                                                                  noise_power + user_interference)

        # Optimize mse_weights(w)- eq. (13) in the paper of Shi et al.
        for user_index_1 in range(nr_of_users):
            if user_index_1 in selected_users:

                user_interference = 0  # it includes the channel of all selected users
                inter_user_interference = 0  # it includes the channel of all selected users apart from the current one

                for user_index_2 in range(nr_of_users):
                    if user_index_2 in selected_users:
                        user_interference = user_interference + (np.absolute(
                            np.matmul(np.conj(channel[user_index_1, :]), transmitter_precoder[user_index_2, :]))) ** 2
                for user_index_2 in range(nr_of_users):
                    if user_index_2 != user_index_1 and user_index_2 in selected_users:
                        inter_user_interference = inter_user_interference + (np.absolute(
                            np.matmul(np.conj(channel[user_index_1, :]), transmitter_precoder[user_index_2, :]))) ** 2

                new_mse_weights[user_index_1] = (noise_power + user_interference) / (
                            noise_power + inter_user_interference)

        A = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_BS_antennas, nr_of_BS_antennas))
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                # hh should be an hermitian matrix of size (nr_of_BS_antennas X nr_of_BS_antennas)
                hh = np.matmul(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)),
                               np.conj(np.transpose(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)))))
                A = A + (new_mse_weights[user_index] * user_weights[user_index] * (np.absolute(new_receiver_precoder[user_index])) ** 2) * hh

        Sigma_diag_elements_true, U = np.linalg.eigh(A)
        Sigma_diag_elements = copy.deepcopy(np.real(Sigma_diag_elements_true))
        Lambda = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas)) + 1j * np.zeros(
            (nr_of_BS_antennas, nr_of_BS_antennas))

        for user_index in range(nr_of_users):
            if user_index in selected_users:
                hh = np.matmul(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)),
                               np.conj(np.transpose(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)))))
                Lambda = Lambda + ((user_weights[user_index]) ** 2) * ((new_mse_weights[user_index]) ** 2) * (
                            (np.absolute(new_receiver_precoder[user_index])) ** 2) * hh

        Phi = np.matmul(np.matmul(np.conj(np.transpose(U)), Lambda), U)
        Phi_diag_elements_true = np.diag(Phi)
        Phi_diag_elements = copy.deepcopy(Phi_diag_elements_true)
        Phi_diag_elements = np.real(Phi_diag_elements)

        for i in range(len(Phi_diag_elements)):
            if Phi_diag_elements[i] < np.finfo(float).eps:
                Phi_diag_elements[i] = np.finfo(float).eps
            if (Sigma_diag_elements[i]) < np.finfo(float).eps:
                Sigma_diag_elements[i] = 0

        # Check if mu = 0 is a solution (eq.s (15) and (16) of in the paper of Shi et al.)(mu is the Lagrange multiplier)
        power = 0  # the power of transmitter precoder (i.e. sum of the squared norm)
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                if np.linalg.det(A) != 0:
                    w = np.matmul(np.linalg.inv(A), np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1))) * \
                        user_weights[user_index] * new_mse_weights[user_index] * (new_receiver_precoder[user_index])
                    power = power + (compute_norm_of_complex_array(w)) ** 2

        # If mu = 0 is a solution, then mu_star = 0
        if np.linalg.det(A) != 0 and power <= total_power:
            mu_star = 0
        # If mu = 0 is not a solution then we search for the "optimal" mu by bisection
        else:
            power_distance = []  # list to store the distance from total_power in the bisection algorithm
            mu_low = np.sqrt(1 / total_power * np.sum(Phi_diag_elements))
            mu_high = 0
            # low_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_low)
            # high_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_high)

            obtained_power = total_power + 2 * power_tolerance  # initialization of the obtained power such that we enter the while

            # Bisection search
            while np.absolute(total_power - obtained_power) > power_tolerance:
                mu_new = (mu_high + mu_low) / 2
                obtained_power = compute_P(Phi_diag_elements, Sigma_diag_elements,
                                           mu_new)  # eq. (18) in the paper of Shi et al.
                power_distance.append(np.absolute(total_power - obtained_power))
                if obtained_power > total_power:
                    mu_high = mu_new
                if obtained_power < total_power:
                    mu_low = mu_new
            mu_star = mu_new
            if log == True:
                print("first value:", power_distance[0])
                # plt.title("Distance from the target value in bisection (it should decrease)")
                # plt.plot(power_distance)
                # plt.show()
                #Equation (5c)
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                new_transmitter_precoder[user_index, :] = np.matmul(
                    np.linalg.inv(A + mu_star * np.eye(nr_of_BS_antennas)), channel[user_index, :]) * user_weights[
                                                              user_index] * new_mse_weights[user_index] * (
                                                          new_receiver_precoder[user_index])

                # To select only the weights of the selected users to check the break condition
        mse_weights_selected_users = []
        new_mse_weights_selected_users = []
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                mse_weights_selected_users.append(mse_weights[user_index])
                new_mse_weights_selected_users.append(new_mse_weights[user_index])

        mse_weights = deepcopy(new_mse_weights)
        transmitter_precoder = deepcopy(new_transmitter_precoder)
        receiver_precoder = deepcopy(new_receiver_precoder)

        WSR.append(compute_weighted_sum_rate_WMMSE(user_weights, channel, transmitter_precoder, noise_power, selected_users))
        break_condition = np.absolute(
            np.log2(np.prod(new_mse_weights_selected_users)) - np.log2(np.prod(mse_weights_selected_users)))

    if log == True:
        plt.title("Change of the WSR at each iteration of the WMMSE (it should increase)")
        plt.plot(WSR, 'bo')
        plt.show()

    return transmitter_precoder, receiver_precoder, mse_weights, WSR[-1]
#Equation (3) in Deep unfold
def compute_weighted_sum_rate_WMMSE(user_weights, channel, precoder, noise_power, selected_users):
    result = 0
    nr_of_users = np.size(channel, 0)

    for user_index in range(nr_of_users):
        if user_index in selected_users:
            user_sinr = compute_sinr(channel, precoder, noise_power, user_index, selected_users)
            result = result + user_weights[user_index] * np.log2(1 + user_sinr)

    return result
# Computes a channel realization and returns it in two formats, one for the WMMSE and one for the deep unfolded WMMSE.
# It also returns the initialization value of the transmitter precoder, which is used as input in the computation graph of the deep unfolded WMMSE.
def compute_channel(nr_of_BS_antennas, nr_of_users, total_power):
    channel_WMMSE = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_users, nr_of_BS_antennas))
    for i in range(nr_of_users):
        result_real = np.sqrt(0.5) * np.random.normal(size=(nr_of_BS_antennas, 1))
        result_imag = np.sqrt(0.5) * np.random.normal(size=(nr_of_BS_antennas, 1))
        channel_WMMSE[i, :] = np.reshape(result_real, (1, nr_of_BS_antennas)) + 1j * np.reshape(result_imag,(1, nr_of_BS_antennas))
    return channel_WMMSE

def compute_sinr(channel, precoder, noise_power, user_id, selected_users):
    nr_of_users = np.size(channel, 0)
    numerator = (np.absolute(np.matmul(np.conj(channel[user_id, :]), precoder[user_id, :]))) ** 2

    inter_user_interference = 0
    for user_index in range(nr_of_users):
        if user_index != user_id and user_index in selected_users:
            inter_user_interference = inter_user_interference + (
                np.absolute(np.matmul(np.conj(channel[user_id, :]), precoder[user_index, :]))) ** 2
    denominator = noise_power + inter_user_interference

    result = numerator / denominator
    return result

def compute_weighted_sum_rate(user_weights, channel, precoder_in, noise_power, selected_users):
    result = 0
    nr_of_users = np.size(channel, 0)
    precoder=precoder_in.detach()
    transmitter_precoder=precoder.mm(M_Re)+1j*precoder.mm(M_Im)
    transmitter_precoder=transmitter_precoder.detach().numpy()
    channel=channel.numpy()
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            user_sinr = compute_sinr(channel, transmitter_precoder, noise_power, user_index, selected_users)
            result = result + user_weights[user_index] * np.log2(1 + user_sinr)
            # user_sinr = compute_sinr(channel, receiver_precoder, noise_power, user_index, selected_users)
            # result = result + user_weights[user_index] * np.log2(1 + user_sinr)
    return result


def compute_sinr_V(channel, transmitter_precoder_in, noise_power, user_id, selected_users):
    nr_of_users = np.size(channel, 0)
    h_i=torch.conj(channel[user_id, :])
    h_re=h_i.real.float()
    h_re=h_re.reshape(1,nr_of_users)
    h_im=h_i.imag.float()
    h_im=h_im.reshape(1,nr_of_users)
    h_i=h_i.reshape(1,nr_of_users)
    V_re=transmitter_precoder_in.mm(M_Re)
    V_im=transmitter_precoder_in.mm(M_Im)
    v_re1=V_re[user_id,:]
    v_im1=V_im[user_id,:]
    v_re1=v_re1.reshape(nr_of_users,1)
    v_im1=v_im1.reshape(nr_of_users,1)
    hv_re=h_re.mm(v_re1)-h_im.mm(v_im1)
    hv_im=h_re.mm(v_im1)+h_im.mm(v_re1)
    hv_i=hv_re.pow(2)+hv_im.pow(2)
    numerator = hv_i
    inter_user_interference = 0
    for user_index in range(nr_of_users):
        if user_index != user_id and user_index in selected_users:
            v_re2=V_re[user_index,:]
            v_im2=V_im[user_index,:]
            v_re2=v_re2.reshape(nr_of_users,1)
            v_im2=v_im2.reshape(nr_of_users,1)
            hvj_re=h_re.mm(v_re2)-h_im.mm(v_im2)
            hvj_im=h_im.mm(v_re2)+h_re.mm(v_im2)
            hv_j=hvj_re.pow(2)+hvj_im.pow(2)       
            inter_user_interference = inter_user_interference + hv_j+noise_power
    denominator = noise_power + inter_user_interference
    result = numerator / denominator
    return result

def Compute_Loss_V(user_weights, channel, precoder_in, noise_power, selected_users):
    result = 0
    nr_of_users = np.size(channel, 0)
    transmitter_precoder=precoder_in
    V_re=transmitter_precoder.mm(M_Re)
    V_im=transmitter_precoder.mm(M_Im)
    Trace_V=torch.trace(V_re.t().mm(V_re)+V_im.t().mm(V_im))
    # receiver_precoder=precoder.mm(M_Re)+1j*precoder.mm(M_Im)
    # receiver_precoder=receiver_precoder.t()
    # receiver_precoder=receiver_precoder
    channel=channel
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            user_sinr = compute_sinr_V(channel, transmitter_precoder, noise_power, user_index, selected_users)
            result = result + user_weights[user_index] * torch.log2(1 + user_sinr)
    result=result+mu*(total_power-Trace_V)
    return -result


def compute_norm_of_complex_array(x):
    result = np.sqrt(np.sum((np.absolute(x)) ** 2))
    return result

def compute_P(Phi_diag_elements, Sigma_diag_elements, mu):
    nr_of_BS_antennas = Phi_diag_elements.size
    mu_array = mu * np.ones(Phi_diag_elements.size)
    result = np.divide(Phi_diag_elements, (Sigma_diag_elements + mu_array) ** 2)
    result = np.sum(result)
    return result


def initia_transmitter_precoder(channel_realization):
    # channel_realization = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_users, nr_of_BS_antennas))
    # for i in range(nr_of_users):
    #     result_real = np.sqrt(0.5) * np.random.normal(size=(nr_of_BS_antennas, 1))
    #     result_imag = np.sqrt(0.5) * np.random.normal(size=(nr_of_BS_antennas, 1))
    #     channel_realization[i, :] = np.reshape(result_real, (1, nr_of_BS_antennas)) + 1j * np.reshape(result_imag,(1, nr_of_BS_antennas))
    transmitter_precoder_init = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_users,nr_of_BS_antennas)) 
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            transmitter_precoder_init[user_index, :] = channel_realization[user_index, :]
    transmitter_precoder_initialize = transmitter_precoder_init / np.linalg.norm(transmitter_precoder_init) * np.sqrt(total_power)
    
    transmitter_precoder_init=torch.from_numpy(transmitter_precoder_initialize)
    transmitter_precoder_complex=transmitter_precoder_init
    transmitter_precoder_Re=transmitter_precoder_complex.real
    transmitter_precoder_Im=transmitter_precoder_complex.imag
    transmitter_precoder=torch.cat((transmitter_precoder_Re,transmitter_precoder_Im),dim=1)
    return transmitter_precoder, transmitter_precoder_initialize


def compute_mu(channel,mse_weights_in,receiver_precoder_in):
    channel=channel.numpy()
    mse_weights=mse_weights_in.detach().numpy()    
    # receiver_precoder=receiver_precoder_in.mm(M_Re)+1j*receiver_precoder_in.mm(M_Im)
    receiver_precoder=receiver_precoder_in.t()
    receiver_precoder=receiver_precoder.detach().numpy()
    A = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_BS_antennas, nr_of_BS_antennas))
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            # hh should be an hermitian matrix of size (nr_of_BS_antennas X nr_of_BS_antennas)
            hh = np.matmul(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)),
                           np.conj(np.transpose(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)))))
            A = A + (mse_weights[user_index] * user_weights[user_index] * (
                np.absolute(receiver_precoder[user_index])) ** 2) * hh

    Sigma_diag_elements_true, U = np.linalg.eigh(A)
    Sigma_diag_elements = copy.deepcopy(np.real(Sigma_diag_elements_true))
    Lambda = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas)) + 1j * np.zeros(
        (nr_of_BS_antennas, nr_of_BS_antennas))

    for user_index in range(nr_of_users):
        if user_index in selected_users:
            hh = np.matmul(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)),
                           np.conj(np.transpose(np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1)))))
            Lambda = Lambda + ((user_weights[user_index]) ** 2) * ((mse_weights[user_index]) ** 2) * (
                        (np.absolute(receiver_precoder[user_index])) ** 2) * hh

    Phi = np.matmul(np.matmul(np.conj(np.transpose(U)), Lambda), U)
    Phi_diag_elements_true = np.diag(Phi)
    Phi_diag_elements = copy.deepcopy(Phi_diag_elements_true)
    Phi_diag_elements = np.real(Phi_diag_elements)

    for i in range(len(Phi_diag_elements)):
        if Phi_diag_elements[i] < np.finfo(float).eps:
            Phi_diag_elements[i] = np.finfo(float).eps
        if (Sigma_diag_elements[i]) < np.finfo(float).eps:
            Sigma_diag_elements[i] = 0

    # Check if mu = 0 is a solution (eq.s (15) and (16) of in the paper of Shi et al.)
    power = 0  # the power of transmitter precoder (i.e. sum of the squared norm)
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            if np.linalg.det(A) != 0:
                w = np.matmul(np.linalg.inv(A), np.reshape(channel[user_index, :], (nr_of_BS_antennas, 1))) * \
                    user_weights[user_index] * mse_weights[user_index] * (receiver_precoder[user_index])
                power = power + (compute_norm_of_complex_array(w)) ** 2

    # If mu = 0 is a solution, then mu_star = 0
    if np.linalg.det(A) != 0 and power <= total_power:
        mu_star = 0
    # If mu = 0 is not a solution then we search for the "optimal" mu by bisection
    else:
        power_distance = []  # list to store the distance from total_power in the bisection algorithm
        mu_low = np.sqrt(1 / total_power * np.sum(Phi_diag_elements))
        mu_high = 0
        # low_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_low)
        # high_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_high)

        obtained_power = total_power + 2 * power_tolerance  # initialization of the obtained power such that we enter the while

        # Bisection search
        while np.absolute(total_power - obtained_power) > power_tolerance:
            mu_new = (mu_high + mu_low) / 2
            obtained_power = compute_P(Phi_diag_elements, Sigma_diag_elements,
                                       mu_new)  # eq. (18) in the paper of Shi et al.
            power_distance.append(np.absolute(total_power - obtained_power))
            if obtained_power > total_power:
                mu_high = mu_new
            if obtained_power < total_power:
                mu_low = mu_new
        mu_star = mu_new
    return mu_star


#Dimensions of variables
DIM_u1=nr_of_users*2
DIM_u2=1
DIM_w1=nr_of_users
DIM_w2=1
DIM_V1=nr_of_users*2
DIM_V2=nr_of_BS_antennas
#network settings
input_size_u=DIM_u1
output_size_u=DIM_u1
batchsize_u=DIM_u2

input_size_w=DIM_w2
output_size_w=DIM_w2
batchsize_w=DIM_w1

input_size_V=DIM_V1
output_size_V=DIM_V1
batchsize_V=DIM_V2


#Build optimizee netowrks
class LSTM_Optimizee_u(torch.nn.Module):
    def __init__(self ):
        super(LSTM_Optimizee_u,self).__init__()
        self.lstm=torch.nn.LSTM(input_size_u,hidden_size_u, layer)
        self.out=torch.nn.Linear(hidden_size_u,output_size_u)        
    def forward(self,gradient, state):
            gradient=gradient.unsqueeze(0)
            if state is None:
                state=(torch.zeros(layer,batchsize_u,hidden_size_u),
                torch.zeros(layer,batchsize_u,hidden_size_u))
            if USE_CUDA:
                state = (torch.zeros(layer, batchsize_u, hidden_size_u).cuda(),
                         torch.zeros(layer, batchsize_u, hidden_size_u).cuda())
            update,state=self.lstm(gradient,state)
            update=self.out(update)
            update=update.squeeze(0)
            return update, state


class LSTM_Optimizee_w(torch.nn.Module):
    def __init__(self ):
        super(LSTM_Optimizee_w,self).__init__()
        self.lstm=torch.nn.LSTM(input_size_w,hidden_size_w, layer)
        self.out=torch.nn.Linear(hidden_size_w,output_size_w)        
    def forward(self,gradient, state):
            gradient=gradient.unsqueeze(0)
            if state is None:
                state=(torch.zeros(layer,batchsize_w,hidden_size_w),
                torch.zeros(layer,batchsize_w,hidden_size_w))
            if USE_CUDA:
                state = (torch.zeros(layer, batchsize_w, hidden_size_w).cuda(),
                         torch.zeros(layer, batchsize_w, hidden_size_w).cuda())
            update,state=self.lstm(gradient,state)
            update=self.out(update)
            update=update.squeeze(0)
            return update, state
        
        
class LSTM_Optimizee_V(torch.nn.Module):
    def __init__(self ):
        super(LSTM_Optimizee_V,self).__init__()
        self.lstm=torch.nn.LSTM(input_size_V,hidden_size_V, layer)
        self.out=torch.nn.Linear(hidden_size_V,output_size_V)        
    def forward(self,gradient, state):
            gradient=gradient.unsqueeze(0)
            if state is None:
                state=(torch.zeros(layer,batchsize_V,hidden_size_V),
                torch.zeros(layer,batchsize_w,hidden_size_V))
            if USE_CUDA:
                state = (torch.zeros(layer, batchsize_V, hidden_size_V).cuda(),
                         torch.zeros(layer, batchsize_V, hidden_size_V).cuda())
            update,state=self.lstm(gradient,state)
            update=self.out(update)
            update=update.squeeze(0)
            return update, state
        
#build meta-learners
def meta_learner_u(receiver_precoder_init,optimizee,Internal_iteration,mse_weight,transmitter_precoder,mu,channel, retain_graph_flag=True):
    # receiver_precoder_internal=torch.rand(DIM_u1,DIM_u2)+1j*torch.rand(DIM_u1,DIM_u2)
    # receiver_precoder_internal1=torch.rand(DIM_u2,nr_of_users, dtype=torch.float)
    # receiver_precoder_internal2=torch.zeros(DIM_u2,nr_of_users, dtype=torch.float)
    # receiver_precoder_internal=torch.cat((receiver_precoder_internal1,receiver_precoder_internal2),dim=1)
    receiver_precoder_internal=receiver_precoder_init
    sum_loss_u=0
    state=None
    receiver_precoder_internal.requires_grad=True
    # loss_u=[]
    for internal_index in range(Internal_iteration):
        L=loss_function(receiver_precoder_internal, mse_weight, transmitter_precoder, channel, noise_power, mu)
        # L=Compute_Loss_V(user_weights, channel, transmitter_precoder, noise_power, selected_users)
        L.backward(retain_graph=retain_graph_flag)
        receiver_precoder_update,state=optimizee(receiver_precoder_internal.grad.clone().detach(),state)
        sum_loss_u=L+sum_loss_u
        # loss_u.append(L)
        receiver_precoder_internal=receiver_precoder_internal+receiver_precoder_update
        receiver_precoder_update.retain_grad()
        receiver_precoder_internal.retain_grad()
        if state is not None:
            state = (state[0].detach(),state[1].detach())
    receiver_precoder_star=receiver_precoder_internal
    return L, sum_loss_u, receiver_precoder_star


def meta_learner_w(mse_weight_init,optimizee,Internal_iteration,receiver_precoder,transmitter_precoder,mu,channel, retain_graph_flag=True):
    # mse_weight_internal=abs(torch.rand(DIM_w1,DIM_w2))
    mse_weight_internal=mse_weight_init
    sum_loss_w=0
    state=None
    mse_weight_internal.requires_grad=True
    # loss_w=[]
    for internal_index in range(Internal_iteration):
        L=loss_function(receiver_precoder,mse_weight_internal, transmitter_precoder, channel, noise_power, mu)
        L.backward(retain_graph=retain_graph_flag)
        mse_weight_update,state=optimizee(mse_weight_internal.grad.clone().detach(),state)
        sum_loss_w=L+sum_loss_w
        # loss_w.append(L)
        mse_weight_internal=mse_weight_internal+abs(mse_weight_update)
        mse_weight_update.retain_grad()
        mse_weight_internal.retain_grad()
        if state is not None:
            state = (state[0].detach(),state[1].detach())
    mse_weight_star=mse_weight_internal
    return L, sum_loss_w, mse_weight_star


def meta_learner_V(transmitter_precoder_init,optimizee,Internal_iteration,receiver_precoder,mse_weight,mu,channel, retain_graph_flag=True):
    # R_v=torch.rand(DIM_V2,DIM_V1)
    # transmitter_precoder_internal=initia_transmitter_precoder(channel)
    # transmitter_precoder_internal=transmitter_precoder_internal.float()
    # transmitter_precoder_internal= torch.empty(DIM_V2,DIM_V1)
    # torch.nn.init.uniform_(transmitter_precoder_internal,a=-5,b=5)
    transmitter_precoder_internal=transmitter_precoder_init
    transmitter_precoder_internal=transmitter_precoder_internal.float()
    sum_loss_V=0
    state=None
    transmitter_precoder_internal.requires_grad=True
    # loss_V=[]
    for internal_index in range(Internal_iteration):
        L=loss_function(receiver_precoder,mse_weight, transmitter_precoder_internal, channel, noise_power, mu)
        # L=Compute_Loss_V(user_weights, channel, transmitter_precoder_internal, noise_power, selected_users)
        L.backward(retain_graph=retain_graph_flag)
        transmitter_precoder_update,state=optimizee(transmitter_precoder_internal.grad.clone().detach(),state)
        sum_loss_V=L+sum_loss_V
        # loss_V.append(L)
        transmitter_precoder_internal=transmitter_precoder_internal+transmitter_precoder_update
        
        # V_re=transmitter_precoder_internal.mm(M_Re)
        # V_im=transmitter_precoder_internal.mm(M_Im)
        # Trace_V=torch.trace(V_re.t().mm(V_re)+V_im.t().mm(V_im))
        # Tr=int(Trace_V.detach().numpy())
        # normV=torch.norm(transmitter_precoder_internal)
        # WW=math.sqrt(total_power)/(normV)
        # if Tr>total_power:
        #      transmitter_precoder_internal=transmitter_precoder_internal*WW
        
        transmitter_precoder_update.retain_grad()
        transmitter_precoder_internal.retain_grad()
        if state is not None:
            state = (state[0].detach(),state[1].detach())
    transmitter_precoder_star=transmitter_precoder_internal
    return L, sum_loss_V, transmitter_precoder_star

optimizee_u=LSTM_Optimizee_u()
adam_global_optimizer_u = torch.optim.Adam(optimizee_u.parameters(),lr = optimizer_lr_u)#update optimizee with adam
optimizee_w=LSTM_Optimizee_w()
adam_global_optimizer_w = torch.optim.Adam(optimizee_w.parameters(),lr = optimizer_lr_w)#update optimizee with adam
optimizee_V=LSTM_Optimizee_V()
adam_global_optimizer_V = torch.optim.Adam(optimizee_V.parameters(),lr = optimizer_lr_V)#update optimizee with adam
print(optimizee_u)
print(optimizee_w)
print(optimizee_V)
MSR_record=[]
MSR_WMMSE_record=[]
Loss_record=[]
meta_list=[]
WMMSE_list=[]
epoch_record_loss=0
#For each tranning sample, with given channel, optimize each variable with respect to minimize the 
#accumlated losses. The MSR is reported but not related to the optimization process.
for batch_step in range(epoch):
    epoch_record_meta=0
    epoch_record_WMMSE=0
    WSR_list_per_sample=torch.zeros(nr_of_training,External_iteration)
    WMMSE_list_per_sample=torch.zeros(nr_of_training,External_iteration)
    Loss_v_list_per_sample=torch.zeros(nr_of_training,External_iteration)
    Loss_w_list_per_sample=torch.zeros(nr_of_training,External_iteration)
    for ex_step in range(nr_of_training):
        # initialization
        # mu_fix=0
        # optimizee_V=LSTM_Optimizee_V()
        # adam_global_optimizer_V = torch.optim.Adam(optimizee_V.parameters(),lr = optimizer_lr_V)#update optimizee with adam
        # epoch_record_meta=0
        # epoch_record_WMMSE=0
        channel_realization = compute_channel(nr_of_BS_antennas, nr_of_users, total_power)#得到一个新的channel
        channel_realization = torch.from_numpy(channel_realization)
        channel = channel_realization
        # norm_channel=torch.norm(abs(channel))
        # print(norm_channel)
        # transmitter_precoder=initia_transmitter_precoder(channel_realization)
        transmitter_precoder_init, transmitter_precoder_initialize=initia_transmitter_precoder(channel_realization)
        # transmitter_precoder_init=transmitter_precoder
        transmitter_precoder=transmitter_precoder_init
        mse_weight=torch.rand(DIM_w1,DIM_w2)
        mse_weight_init=mse_weight
        receiver_precoder1=torch.rand(DIM_u2,nr_of_users, dtype=torch.float)
        receiver_precoder2=torch.zeros(DIM_u2,nr_of_users, dtype=torch.float)
        receiver_precoder=torch.cat((receiver_precoder1,receiver_precoder2),dim=1)
        receiver_precoder_init=receiver_precoder
        mu=compute_mu(channel,mse_weight,receiver_precoder)
        # mu=0.1
        # MSR=compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users)
        # print(mu)
        mu=0
        LossAccumulated_u=0
        LossAccumulated_w=0
        LossAccumulated_V=0
        WSR_external=0
        print('\n=======> Epoch', batch_step+1, 'Training sample: {}'.format(ex_step+1))
        start = timer()
        for in_step in range(External_iteration):
            # mse_weight=1*torch.rand(DIM_w1,DIM_w2)
            # mse_weight_init=mse_weight
            # receiver_precoder1=1*torch.rand(DIM_u2,nr_of_users, dtype=torch.float)
            # receiver_precoder2=torch.zeros(DIM_u2,nr_of_users, dtype=torch.float)
            # receiver_precoder=torch.cat((receiver_precoder1,receiver_precoder2),dim=1)
            # receiver_precoder_init=receiver_precoder
            # channel_realization_internal = compute_channel(nr_of_BS_antennas, nr_of_users, total_power)
            # channel_realization_internal = torch.from_numpy(channel_realization_internal)
            # channel_realization_internal = channel_realization_internal
            # transmitter_precoder=initia_transmitter_precoder(channel_realization_internal)
            
            #update each variable by meta-learner networks
            loss_u, sum_loss_u, receiver_precoder = meta_learner_u(receiver_precoder_init,optimizee_u,Internal_iteration,mse_weight.clone().detach(),transmitter_precoder.clone().detach(),mu,channel, retain_graph_flag=True)
            loss_w, sum_loss_w, mse_weight = meta_learner_w(mse_weight_init,optimizee_w,Internal_iteration,receiver_precoder.clone().detach(),transmitter_precoder.clone().detach(),mu,channel, retain_graph_flag=True)
            loss_V, sum_loss_V, transmitter_precoder = meta_learner_V(transmitter_precoder_init,optimizee_V, Internal_iteration, receiver_precoder.clone().detach(), mse_weight.clone().detach(), mu, channel,retain_graph_flag=True)
            # LossAccumulated_u = LossAccumulated_u+loss_u
            # LossAccumulated_w = LossAccumulated_w+loss_w
            
            mu=compute_mu(channel,mse_weight,receiver_precoder)
            # V_re=transmitter_precoder.mm(M_Re)
            # V_im=transmitter_precoder.mm(M_Im)
            # Trace_V=torch.trace(V_re.t().mm(V_re)+V_im.t().mm(V_im))
            V_re=transmitter_precoder.mm(M_Re)
            V_im=transmitter_precoder.mm(M_Im)
            Trace_V=torch.trace(V_re.t().mm(V_re)+V_im.t().mm(V_im))
            Tr=int(Trace_V.detach().numpy())
            normV=torch.norm(transmitter_precoder)
            WW=math.sqrt(total_power)/(normV)
            if Tr>total_power:
                  transmitter_precoder=transmitter_precoder*WW
            loss_V=loss_function(receiver_precoder,mse_weight, transmitter_precoder, channel, noise_power, mu)
            loss_V=Compute_Loss_V(user_weights, channel, transmitter_precoder, noise_power, selected_users)
            LossAccumulated_V = LossAccumulated_V+loss_V
            LossAccumulated_u=LossAccumulated_u+loss_V
            LossAccumulated_w=LossAccumulated_w+loss_V
            # receiver_precoder_init=receiver_precoder.detach()
            # mse_weight_init=mse_weight_init.detach()
            # transmitter_precoder_init=transmitter_precoder.detach()
            MSR=compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users)
            # MSR_V=Compute_Loss_V(user_weights, channel, transmitter_precoder, noise_power, selected_users)
            # print(MSR,MSR_V)
            # MSR_v=compute_weighted_sum_rate_V(user_weights, channel, transmitter_precoder, noise_power, selected_users)
            WSR_external+=MSR
            # V_WMMSE, u_WMMSE, w_WMMSE, WSR_WMMSE_one_sample = run_WMMSE(epsilon, channel_realization, scheduled_users,
            #                                             total_power, noise_power, user_weights_for_regular_WMMSE,
            #                                             nr_of_iterations - 1, log=False)
            V_WMMSE, u_WMMSE, w_WMMSE, WSR_WMMSE_one_sample = run_WMMSE(transmitter_precoder_initialize,epsilon, channel_realization, scheduled_users,
                                                        total_power, noise_power, user_weights_for_regular_WMMSE,
                                                        nr_of_iterations - 1, log=False)
            
            WSR_list_per_sample[ex_step,in_step]=MSR
            WMMSE_list_per_sample[ex_step,in_step]=WSR_WMMSE_one_sample
            Loss_v_list_per_sample[ex_step,in_step]=int(loss_V.clone().detach().numpy())
            Loss_w_list_per_sample[ex_step,in_step]=int(loss_w.clone().detach().numpy())
            #update meta-learner parameters after Update_steps of external iterations
            if (in_step+1)% Update_steps == 0:
                # if ex_step<(test_rate*nr_of_training):
                adam_global_optimizer_u.zero_grad()
                adam_global_optimizer_w.zero_grad()
                adam_global_optimizer_V.zero_grad()
                Average_loss_u=LossAccumulated_u/Update_steps
                Average_loss_w=LossAccumulated_w/Update_steps
                Average_loss_V=LossAccumulated_V/Update_steps
                Average_loss_u.backward(retain_graph=True)
                #Average_loss_w.backward(retain_graph=True)
                #Average_loss_V.backward(retain_graph=True)
                #adam_global_optimizer_u.step()
                #adam_global_optimizer_w.step()
                adam_global_optimizer_V.step()
                time = timer() - start
                MSR=compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users)
                # Tr=int(Trace_V.detach().numpy())
                LossAccumulated_u=0
                LossAccumulated_w=0
                LossAccumulated_V=0
                # V_re=transmitter_precoder.mm(M_Re)
                # V_im=transmitter_precoder.mm(M_Im)
                # Trace_V=torch.trace(V_re.t().mm(V_re)+V_im.t().mm(V_im))
                # Tr=int(Trace_V.detach().numpy())
                
                
                print('->  step :' ,in_step+1,'MSR=','%.2f'%MSR,'WMMSE =','%.2f'%WSR_WMMSE_one_sample,'time=','%.0f'%time,'Trace',Tr)
               
                
               # print('->  step :' ,in_step+1,'MSR=','%.2f'%MSR,'loss =','%.2f'%loss_w.detach().numpy(),'time=','%.0f'%time)
                # mse_weight=1*torch.rand(DIM_w1,DIM_w2)
                # mse_weight_init=mse_weight
                # receiver_precoder1=1*torch.rand(DIM_u2,nr_of_users, dtype=torch.float)
                # receiver_precoder2=torch.zeros(DIM_u2,nr_of_users, dtype=torch.float)
                # receiver_precoder=torch.cat((receiver_precoder1,receiver_precoder2),dim=1)
                # receiver_precoder_init=receiver_precoder
                # channel_realization_internal = compute_channel(nr_of_BS_antennas, nr_of_users, total_power)
                # channel_realization_internal = torch.from_numpy(channel_realization_internal)
                # channel_realization_internal = channel_realization_internal
                # transmitter_precoder=initia_transmitter_precoder(channel_realization_internal)
                
            if (in_step+1)==External_iteration:
                print('->  step :' ,in_step+1,'MSR=','%.2f'%MSR,'WMMSE','%.2f'%WSR_WMMSE_one_sample,'time=','%.0f'%time)
                # print('->  step :' ,in_step+1,'MSR=','%.2f'%MSR,'loss =','%.2f'%loss_V.detach().numpy(),'time=','%.0f'%time,'Trace',Trace_V.detach().numpy())
                # print()
                Mean_WSR=WSR_external/External_iteration

                MSR_record.append(Mean_WSR)
                MSR_WMMSE_record.append(WSR_WMMSE_one_sample)
                Loss_record.append(loss_V.detach().numpy())
                epoch_record_meta+=MSR
                epoch_record_loss+=loss_V.detach().numpy()
                epoch_record_WMMSE+=WSR_WMMSE_one_sample
        if (ex_step+1)%10==0:
            print('epoch_mean_WSR',epoch_record_meta/ex_step)
            print('epoch_mean_WMMSE',epoch_record_WMMSE/ex_step)        
        if (ex_step+1)==nr_of_training:
            meta_list.append(epoch_record_meta/nr_of_training)
            print('epoch_mean_WSR',epoch_record_meta/nr_of_training)
            print('epoch_mean_WMMSE',epoch_record_WMMSE/nr_of_training)
            WMMSE_list.append(epoch_record_loss/nr_of_training)
    metalist=np.array(meta_list)
    # wmmselist=np.array(WMMSE_list)
    losslist=np.array(WMMSE_list)
    WSR_matrix=WSR_list_per_sample.numpy()
    WMMSE_matrix=WMMSE_list_per_sample.numpy()
    lossV_matrix=Loss_v_list_per_sample.numpy()
    lossw_matrix=Loss_w_list_per_sample.numpy()
    datanew1 = './Results_ThreeVariables_5dB.mat'
    scio.savemat(datanew1,{'WSR':metalist,'loss':losslist,'WSRMatrix':WSR_matrix,'WMMSE_matrix':WMMSE_matrix,'lossV_matrix':lossV_matrix,'lossw_matrix':lossw_matrix})


# print('Meta-learning Results:',MSR_record/nr_of_training)
# print('WMMSE Results:',MSR_WMMSE_record/nr_of_training)