import network as netfn    

#=============================================================================
def stim_net(sim_time, stim_time, n_on, on_ind, input_A, input_W,k, vth, r, s, divisor, soften, N, dist, v_rest, t_syn_del, tau_l, N_e, lam, w_e): 
#=============================================================================
    """
    Add input to spiking network and compare its trajectory to a non-perturbed network. 
    
    Inputs:
        sim_time (float): time steps to run simulation
        n_in (int): number of neurons to activate
        on_ind (np array): 1d vector corresponding to indices of neurons to turn on
        input_A (np array): input neurons x network neurons, adjacency matrix
        input_W (np.array): input neurons x network neurons, weight matrix
        k (int): number of edges in network
        v_th (float): spike threshold 
        r (float): weight scaling parameter, defining local vs global scaling
        s (float): weight scaling parameter, defining overall range 
        divisor (float): divisor value for scaling function
        soften (float): degree of exponential softening for scaling function
        N (int): number of neurons in network
        dist (np array): distance matrix
        v_rest (float): resting membrane potential
        t_syn_del (float): synaptic delay
        tau_l (float): time constant
        N_e (int): number of external neurons
        lam (float): Poisson input rate
        w_e (float): weight from poisson inputs onto network
        
    Returns:
        bind (np.array): cells x timepoints, downsampled binarised array of network without input
        bind_inp (np.array): cells x timepoints, downsampled binarised array of network with input
        bind_og (np.array): cells x timepoints, downsampled binarised array of network before input occurs
    
    """

    #Stimulate network
    import brian2 as b2
    from random import sample
    from numpy import random
    import numpy as np

    b2.start_scope()

    #BUILD RECURRENT NET
    # define dynamics for each cell
    lif ="""
    dv/dt = -(v-v_rest) / tau_l : 1 """
    net_dyn = b2.NeuronGroup(
    N, model=lif,
    threshold="v>vth", reset="v = v_rest",
    method="euler")
    net_dyn.v = v_rest #set starting value for voltage

    p_input = b2.PoissonInput(net_dyn, "v", N_e,lam, w_e)

    #Network connectivity + weights
    curr = netfn.ba_netsim(dist).adjmat_generate(k, s, r, divisor, soften, 'directed')
    A = curr.A
    W = curr.adj_mat

    #Build synapses
    net_syn = b2.Synapses(net_dyn, net_dyn, 'w:1', on_pre="v+=w", delay=t_syn_del)
    rows, cols = np.nonzero(A)
    net_syn.connect(i = rows, j = cols)
    net_syn.w = W[rows, cols]

    #Specify input 
    n_input = N
    indices = b2.array(on_ind)
    times = b2.array([stim_time] * n_on)*b2.ms
    input = b2.SpikeGeneratorGroup(n_input, indices, times)

    #Input synapses
    input_syn = b2.Synapses(input, net_dyn, 'w:1', on_pre = "v+=w", delay = 0 * b2.ms)
    input_rows, input_cols = np.nonzero(input_A)
    input_syn.connect(i = input_rows, j = input_cols)
    input_syn.w = input_W[input_rows, input_cols]


    #Store network at steady state
    spike_monitor = b2.SpikeMonitor(net_dyn)
    V = b2.StateMonitor(net_dyn, 'v', record=True)
    sim_time = 400
    b2.run(sim_time*b2.ms)
    spikes_og = spike_monitor.spike_trains()
    bind_og = netfn.bin_data(spikes_og, N, sim_time)
    b2.store()

    #Run network with input
    b2.restore(restore_random_state=True)
    b2.run(sim_time*b2.ms)
    spikes_inp = spike_monitor.spike_trains()
    bind_inp = netfn.bin_data(spikes_inp, N, sim_time*2)

    #Run network without input
    b2.restore(restore_random_state=True)
    input_syn.w = np.zeros(len(input_W[input_rows, input_cols])) #Set weights to 0
    b2.run(sim_time*b2.ms)
    spikes = spike_monitor.spike_trains()
    bind = netfn.bin_data(spikes, N, sim_time*2)
    return(bind, bind_inp, bind_og)


#=============================================================================
def LE(data1, data2):
#=============================================================================
    """
    Calculates the lyapunov exponent by looking at the trajectories of two networks over time. 
    
    Inputs:
        data1 (np array): cells x timepoints, network activity over time
        data2 (np array): cells x timepoints, network activity over time
        
    Returns:
        LE_vec (np array): vector of LE values over time
    """


    LE_vec = np.zeros(data1.shape[1])
    for t in range(LE_vec.shape[0]):
        d0 = np.linalg.norm(data1[:,0] - data2[:,0]) #Distance between trajectories at input

        dt = np.linalg.norm(data1[:,t] - data2[:,t]) #Distance between trajectories at t

        LE_vec[t] = (1/(t+1)) * np.log(np.abs(dt/d0))

    return(LE_vec)

#=============================================================================
def run_LE(sim_time, stim_time, k, vth, r, s, divisor, soften, N, dist, v_rest, t_syn_del, tau_l, N_e, lam, w_e):
#=============================================================================
    """
    Add input to spiking network to calculate lyapunov exponent.
    
    Inputs:
        sim_time (float): time steps to run simulation
        n_in (int): number of neurons to activate
        on_ind (np array): 1d vector corresponding to indices of neurons to turn on
        input_A (np array): input neurons x network neurons, adjacency matrix
        input_W (np.array): input neurons x network neurons, weight matrix
        k (int): number of edges in network
        v_th (float): spike threshold 
        r (float): weight scaling parameter, defining local vs global scaling
        s (float): weight scaling parameter, defining overall range 
        divisor (float): divisor value for scaling function
        soften (float): degree of exponential softening for scaling function
        N (int): number of neurons in network
        dist (np array): distance matrix
        v_rest (float): resting membrane potential
        t_syn_del (float): synaptic delay
        tau_l (float): time constant
        N_e (int): number of external neurons
        lam (float): Poisson input rate
        w_e (float): weight from poisson inputs onto network
        
    Returns:
        le (np array): vector of LE values over time
    
    """
    
    import random

    operate = 'go'
    while operate == 'go':
        n_on = random.sample(list(np.arange(8, 20)), 1)[0]
        n_input = N
        on_ind = random.sample(list(np.arange(0, n_input)), n_on)

        #Define synaptic connections from input - same across each stimulus size across network conditions
        input_A = np.zeros((N,N))
        np.fill_diagonal(input_A, 1)
        input_W = np.ones(input_A.shape)

        bind, bind_inp, bind_og = stim_net(sim_time, stim_time, n_on, on_ind, input_A, input_W, k, vth, r, s, divisor, soften, N, dist, v_rest, t_syn_del, tau_l, N_e, lam, w_e)

        d1 = np.linalg.norm(bind[:,stim_time] - bind_inp[:,stim_time])
        if d1 >0:
            le = LE(bind[:,stim_time:],bind_inp[:,stim_time:])
            if '-inf' not in str(le):
                operate = 'stop'

    return(le)

#=============================================================================
def dyn_range(output_data, input_range):
#=============================================================================
"""
    Estimate the dynamic range of the network, which captures the range of inputs a network can hold with distinct outputs. 
    
    Inputs:
        output_data (np array): 1d vector of output sizes
        input_range (np array): 1d vecotr of input sizes
    
    Returns:
        dyn_r (float): dynamic range
        Smax (float): 90th percentile of input
        Smin (float): 10th percentile of input
        Omax (float): 90th percentile of output
        Omax (float): 10th percentile of output        
"""


    from sklearn.linear_model import LinearRegression
    mean_out = output_data
    range_out = np.max(mean_out) - np.min(mean_out)
    Omax = np.min(mean_out) + (0.9*range_out)
    Omin = np.min(mean_out) + (0.1*range_out)

    X = mean_out.reshape((-1, 1))
    y = input_range
    reg = LinearRegression().fit(X, y)
    Smax = reg.predict([[Omax]])
    Smin = reg.predict([[Omin]])
    dyn_r = 10*np.log10(Smax/Smin)
    return(dyn_r, Smax, Smin, Omax, Omin)


#=============================================================================
def dr_net(on_list, sim_time, stim_time, n_on, on_ind, input_A, input_W,k, vth, r, s, divisor, soften, N, dist, v_rest, t_syn_del, tau_l, N_e, lam, w_e):
#=============================================================================
    """
    Perturb the network across a range of input sizes and calculate the dynamic range. 
    
    Inputs:
        on_list (np array): 1d vector of input sizes to loop through
        sim_time (float): time steps to run simulation
        n_in (int): number of neurons to activate
        on_ind (np array): 1d vector corresponding to indices of neurons to turn on
        input_A (np array): input neurons x network neurons, adjacency matrix
        input_W (np.array): input neurons x network neurons, weight matrix
        k (int): number of edges in network
        v_th (float): spike threshold 
        r (float): weight scaling parameter, defining local vs global scaling
        s (float): weight scaling parameter, defining overall range 
        divisor (float): divisor value for scaling function
        soften (float): degree of exponential softening for scaling function
        N (int): number of neurons in network
        dist (np array): distance matrix
        v_rest (float): resting membrane potential
        t_syn_del (float): synaptic delay
        tau_l (float): time constant
        N_e (int): number of external neurons
        lam (float): Poisson input rate
        w_e (float): weight from poisson inputs onto network
        
    Returns:
        dyn_r (float): dynamic range
        
    """


    import random
    outlist = list(range(len(on_list)))

    for x,n_on in enumerate(on_list):
    #Spike times + indices
        import random
        stim_time = 401
        n_input = N
        on_ind = random.sample(list(np.arange(0, n_input)), n_on)

        #Define synaptic connections from input - same across each stimulus size across network conditions
        input_A = np.zeros((N,N))
        np.fill_diagonal(input_A, 1)
        input_W = np.ones(input_A.shape)

        bind, bind_inp, bind_og = stim_net(sim_time, stim_time, n_on, on_ind, input_A, input_W, k, vth, r, s, divisor, soften, N, dist, v_rest, t_syn_del, tau_l, N_e, lam, w_e)
        sum_ = np.sum(bind_inp[:,stim_time]) - np.sum(bind[:,stim_time]) 
        if sum_ <0:
            outlist[x] = 0
        else:
            outlist[x] = sum_

    dyn_r, Smax, Smin, Omax, Omin = dyn_range(np.array(outlist), on_list)
    

    return(dyn_r)





#=============================================================================
def NMS(diff_list, sim_time, stim_time, n_on, k, vth, r, s, divisor, soften, N, dist, v_rest, t_syn_del, tau_l, N_e, lam, w_e):
#=============================================================================
    """
    Calculate network mediated separation, the ability of a network to separate similar inputs into distinct outputs. 
    
    Inputs:
        diff_list (np array): 1d vector of input size difference for which to calculate NMS
        sim_time (float): time steps to run simulation
        n_in (int): number of neurons to activate
        on_ind (np array): 1d vector corresponding to indices of neurons to turn on
        input_A (np array): input neurons x network neurons, adjacency matrix
        input_W (np.array): input neurons x network neurons, weight matrix
        k (int): number of edges in network
        v_th (float): spike threshold 
        r (float): weight scaling parameter, defining local vs global scaling
        s (float): weight scaling parameter, defining overall range 
        divisor (float): divisor value for scaling function
        soften (float): degree of exponential softening for scaling function
        N (int): number of neurons in network
        dist (np array): distance matrix
        v_rest (float): resting membrane potential
        t_syn_del (float): synaptic delay
        tau_l (float): time constant
        N_e (int): number of external neurons
        lam (float): Poisson input rate
        w_e (float): weight from poisson inputs onto network
        
    Returns:
        data_list (list): NMS across each input pair
        
    """


    import random

    data_list = list(range(len(diff_list)))

    for c,n_add in enumerate(diff_list):
    #Reliability - distance between same input
        n_input = N
        norm_vec = []

        for i in range(50):

            on_ind1 = random.sample(list(np.arange(0, n_input)), n_on)
            input_A = np.zeros((N,N))
            np.fill_diagonal(input_A, 1)
            input_W = np.ones(input_A.shape)
            input_vec1 = np.zeros(N)
            input_vec1[on_ind1] = 1
            bind1, bind_inp1, bind_og1 = stim_net(sim_time, stim_time, n_on, on_ind1, input_A, input_W, k, vth, r, s, divisor, soften, N, dist, v_rest, t_syn_del, tau_l, N_e, lam, w_e)


            on_ind2 = random.sample(list(np.arange(0, n_input)), n_on + n_add)
            input_A = np.zeros((N,N))
            np.fill_diagonal(input_A, 1)
            input_W = np.ones(input_A.shape)
            input_vec2 = np.zeros(N)
            input_vec2[on_ind2] = 1
            bind2, bind_inp2, bind_og2 = stim_net(sim_time, stim_time, n_on + n_add, on_ind2, input_A, input_W, k, vth, r, s, divisor, soften, N, dist, v_rest, t_syn_del, tau_l, N_e, lam, w_e)

            norm_ed = np.linalg.norm(bind_inp1[:,stim_time] - bind_inp2[:,stim_time]) - np.linalg.norm(bind1[:,stim_time] - bind2[:,stim_time])
            norm_vec = np.append(norm_vec, norm_ed)
        data_list[c] = norm_vec
    return(data_list)