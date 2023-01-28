

import copy
import pickle as pkl

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# visualize along the grid cell in 2D topview    
def visualize_discriminator(normalizer, discriminator, initial_state, scatter_states, env_name, aim_input_type, device, savedir_w_name):    
    disc_vis_start_time = time.time()
    assert aim_input_type=='default'
    
    num_test_points = 60 # 30
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 10, num_test_points)
    elif env_name in ['sawyer_peg_push']:
        x = np.linspace(-0.6, 0.6, num_test_points)
        y = np.linspace(0.2, 1.0, num_test_points)
    elif env_name == "PointSpiralMaze-v0":
        x = np.linspace(-10, 10, num_test_points)
        y = np.linspace(-10, 10, num_test_points)
    elif env_name in ["PointNMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 18, num_test_points)
    else:
        return None
    grid_x, grid_y = np.meshgrid(x,y)    
    goal_xy = np.concatenate([np.reshape(grid_x, [-1, 1]), np.reshape(grid_y, [-1, 1])], axis =1) #[num_test_points^2, 2]
    
    if env_name in [ 'sawyer_peg_push']:
        goal_xy = np.concatenate([goal_xy, 0.01457*np.ones([goal_xy.shape[0], 1])], axis=-1) #[num_test_points^2, 3]
    
    num_grid_point = goal_xy.shape[0]
    
    initial_states  = np.tile(initial_state, (num_grid_point, 1)) # [num_test_points^2, dim]
    
    with torch.no_grad():
        observes = torch.as_tensor(np.concatenate([initial_states, goal_xy], axis= -1), device = device).float()# [num_test_points^2, dim*2]
        if normalizer is not None:
            observes = normalizer(observes, env_name)
        aim_outputs = discriminator.forward(observes).detach().cpu().numpy() #[num_test_points^2, 1]
    
    v_min, v_max = aim_outputs.min(), aim_outputs.max()           
    
    aim_outputs = np.reshape(aim_outputs, [num_test_points, num_test_points])
    
    fig, ax = plt.subplots()

    c = ax.pcolormesh(grid_x, grid_y, aim_outputs, cmap='RdBu', vmin=v_min, vmax=v_max)
    
    if scatter_states.ndim==1:
        ax.scatter(scatter_states[0], scatter_states[1], marker="*", c = 'black', s=10, label='Current_position')
    else:
        for t in range(scatter_states.shape[0]):
            ax.scatter(scatter_states[t, 0], scatter_states[t, 1], marker="*", c = str(1.-t/scatter_states.shape[0]) , s=10, label='s_'+str(t))
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name == "PointSpiralMaze-v0":
        obstacle_point_x = np.array([-2, 6, 6, -6, -6, 10, 10, -10, -10, 10, 10, -2, -2])
        obstacle_point_y = np.array([2, 2, 6, 6, -6, -6, -10, -10, 10, 10, -2, -2, 2])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name in ["PointNMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        obstacle_point_x = np.array([10, 2, 2, 10])
        obstacle_point_y = np.array([10, 10, 14, 14])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    else:
        pass

    ax.set_title('aim_discriminator_visualize')        
    ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
    fig.colorbar(c, ax=ax)
    ax.axis('tight')
    plt.legend(loc="best")
    plt.savefig(savedir_w_name+'.jpg')
    plt.close()   
    disc_vis_end_time = time.time()
    # print('aim discriminator visualize time : {}'.format(disc_vis_end_time - disc_vis_start_time))
    
def visualize_discriminator2(normalizer, discriminator, env_name, aim_input_type, device, savedir_w_name, goal = None):   
    disc_vis_start_time = time.time()
    assert aim_input_type=='default'
    
    num_test_points = 60 # 30
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 10, num_test_points)
    elif env_name in [ 'sawyer_peg_push']:
        x = np.linspace(-0.6, 0.6, num_test_points)
        y = np.linspace(0.2, 1.0, num_test_points)
    elif env_name == "PointSpiralMaze-v0":
        x = np.linspace(-10, 10, num_test_points)
        y = np.linspace(-10, 10, num_test_points)
    elif env_name in ["PointNMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 18, num_test_points)
    else:
        return None

    grid_x, grid_y = np.meshgrid(x,y)    
    goal_xy = np.concatenate([np.reshape(grid_x, [-1, 1]), np.reshape(grid_y, [-1, 1])], axis =1) #[num_test_points^2, 2]
    if env_name in [ 'sawyer_peg_push']:
        goal_xy = np.concatenate([goal_xy, 0.01457*np.ones([goal_xy.shape[0], 1])], axis=-1) #[num_test_points^2, 3]

    num_grid_point = goal_xy.shape[0]
    
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        obs_desired_goal = np.array([0., 8.]) 
    elif env_name in [ 'sawyer_peg_push']:
        obs_desired_goal = np.array([-0.3, 0.4, 0.02])
    elif env_name == "PointSpiralMaze-v0":
        obs_desired_goal = np.array([8., -8.]) 
    elif env_name in ["PointNMaze-v0"]:
        obs_desired_goal = np.array([8., 16.]) 

    initial_states  = np.tile(np.array(obs_desired_goal), (num_grid_point, 1)) # [num_test_points^2, dim]
    
    with torch.no_grad():
        observes = torch.as_tensor(np.concatenate([goal_xy, initial_states], axis= -1), device = device).float()# [num_test_points^2, dim*2]
        if normalizer is not None:
            observes = normalizer(observes, env_name)
        aim_outputs = discriminator.forward(observes).detach().cpu().numpy() #[num_test_points^2, 1]
    
    v_min, v_max = aim_outputs.min(), aim_outputs.max()           
    
    aim_outputs = np.reshape(aim_outputs, [num_test_points, num_test_points])
    
    fig, ax = plt.subplots()

    c = ax.pcolormesh(grid_x, grid_y, aim_outputs, cmap='RdBu', vmin=v_min, vmax=v_max)
    # ax.scatter(goal[0], goal[1], marker="*", c = 'black', s=10, label='goal_position')
    ax.scatter(obs_desired_goal[0], obs_desired_goal[1], marker="*", c = 'black', s=10, label='goal_position')
    
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name == "PointSpiralMaze-v0":
        obstacle_point_x = np.array([-2, 6, 6, -6, -6, 10, 10, -10, -10, 10, 10, -2, -2])
        obstacle_point_y = np.array([2, 2, 6, 6, -6, -6, -10, -10, 10, 10, -2, -2, 2])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name in ["PointNMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        obstacle_point_x = np.array([10, 2, 2, 10])
        obstacle_point_y = np.array([10, 10, 14, 14])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    else:
        pass

    ax.set_title('aim_discriminator_visualize')        
    ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
    fig.colorbar(c, ax=ax)
    ax.axis('tight')
    plt.legend(loc="best")
    plt.savefig(savedir_w_name+'.jpg')
    plt.close()   
    disc_vis_end_time = time.time()
    # print('aim discriminator visualize time : {}'.format(disc_vis_end_time - disc_vis_start_time))

# visualize along the grid cell in 2D topview    
def visualize_uncertainty(vf, vf_obs_achieved_goal, scatter_states, env_name, aim_input_type, device, savedir_w_name):    
    disc_vis_start_time = time.time()
    assert aim_input_type=='default'
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0", "PointSpiralMaze-v0", "PointNMaze-v0"]:
        num_test_points = 30
        if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
            x = np.linspace(-2, 10, num_test_points)
            y = np.linspace(-2, 10, num_test_points)
        elif env_name == "PointSpiralMaze-v0":
            x = np.linspace(-10, 10, num_test_points)
            y = np.linspace(-10, 10, num_test_points)
        elif env_name in ["PointNMaze-v0"]:
            x = np.linspace(-2, 10, num_test_points)
            y = np.linspace(-2, 18, num_test_points)
        grid_x, grid_y = np.meshgrid(x,y)    
        goal_xy = np.concatenate([np.reshape(grid_x, [-1, 1]), np.reshape(grid_y, [-1, 1])], axis =1) #[num_test_points^2, 2]
        num_grid_point = goal_xy.shape[0]
        
        vf_obs_achieved_goals  = np.tile(vf_obs_achieved_goal, (num_grid_point, 1)) # [num_test_points^2, dim]
        
        with torch.no_grad():
            observes = torch.as_tensor(np.concatenate([vf_obs_achieved_goals, goal_xy], axis= -1), device = device).float()# [num_test_points^2, dim*2]
            aim_outputs = vf.std(observes).detach().cpu().numpy() #[num_test_points^2, 1]
        
        v_min, v_max = aim_outputs.min(), aim_outputs.max()           
        
        aim_outputs = np.reshape(aim_outputs, [num_test_points, num_test_points])
        
        fig, ax = plt.subplots()

        c = ax.pcolormesh(grid_x, grid_y, aim_outputs, cmap='RdBu', vmin=v_min, vmax=v_max)
        
        if scatter_states.ndim==1:
            ax.scatter(scatter_states[0], scatter_states[1], marker="*", c = 'black', s=10, label='Current_position')
        else:
            raise NotImplementedError

        if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
            obstacle_point_x = np.array([-2, 6, 6, -2])
            obstacle_point_y = np.array([2, 2, 6, 6])  
            ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        elif env_name == "PointSpiralMaze-v0":         
            obstacle_point_x = np.array([-2, 6, 6, -6, -6, 10, 10, -10, -10, 10, 10, -2, -2])
            obstacle_point_y = np.array([2, 2, 6, 6, -6, -6, -10, -10, 10, 10, -2, -2, 2])        
            ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        elif env_name in ["PointNMaze-v0"]:
            obstacle_point_x = np.array([-2, 6, 6, -2])
            obstacle_point_y = np.array([2, 2, 6, 6])        
            ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
            obstacle_point_x = np.array([10, 2, 2, 10])
            obstacle_point_y = np.array([10, 10, 14, 14])        
            ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        

        ax.set_title('uncertainty_visualize')        
        ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        fig.colorbar(c, ax=ax)
        ax.axis('tight')
        plt.legend(loc="best")
        plt.savefig(savedir_w_name+'.jpg')
        plt.close()

    else:
        raise NotImplementedError    
    disc_vis_end_time = time.time()
    # print('aim discriminator visualize time : {}'.format(disc_vis_end_time - disc_vis_start_time))    
    

# visualize along the grid cell in 2D topview    
def visualize_meta_nml(agent, meta_nml_epoch, scatter_states, replay_buffer, goal_env, env_name, aim_input_type, savedir_w_name):    
    disc_vis_start_time = time.time()
    assert aim_input_type=='default'
    
    num_test_points = 60
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 10, num_test_points)
    elif env_name in ['sawyer_peg_push']:
        x = np.linspace(-0.6, 0.6, num_test_points)
        y = np.linspace(0.2, 1.0, num_test_points)
    elif env_name == "PointSpiralMaze-v0":
        x = np.linspace(-10, 10, num_test_points)
        y = np.linspace(-10, 10, num_test_points)
    elif env_name in ["PointNMaze-v0"]:
        x = np.linspace(-2, 10, num_test_points)
        y = np.linspace(-2, 18, num_test_points)
    else:
        return None

    grid_x, grid_y = np.meshgrid(x,y)    
    goal_xy = np.concatenate([np.reshape(grid_x, [-1, 1]), np.reshape(grid_y, [-1, 1])], axis =1) #[num_test_points^2, 2]
    
    if env_name in ['sawyer_peg_push']:
        goal_xy = np.concatenate([goal_xy, 0.01457*np.ones([goal_xy.shape[0], 1])], axis=-1) #[num_test_points^2, 3]

    observes =goal_xy #torch.as_tensor(goal_xy).float()# [num_test_points^2, dim*2]        
    
    outputs = agent.get_prob_by_meta_nml(observes, meta_nml_epoch, replay_buffer=replay_buffer, goal_env=goal_env) # input : [1, dim] output : list of [dim(1)]
    
    v_min, v_max = outputs.min(), outputs.max()           
    
    outputs = np.reshape(outputs, [num_test_points, num_test_points])
    
    use_smoothing=True
    if use_smoothing:
        from scipy.ndimage import gaussian_filter
        outputs = gaussian_filter(outputs, sigma=2)

    fig, ax = plt.subplots()

    c = ax.pcolormesh(grid_x, grid_y, outputs, cmap='RdBu', vmin=v_min, vmax=v_max)
    
    if scatter_states.ndim==1:
        ax.scatter(scatter_states[0], scatter_states[1], marker="*", c = 'black', s=10, label='Current_position')
    else:
        for t in range(scatter_states.shape[0]):
            ax.scatter(scatter_states[t, 0], scatter_states[t, 1], marker="*", c = str(1.-t/scatter_states.shape[0]) , s=30, label='s_'+str(t))
    if env_name in ['AntMazeSmall-v0', "PointUMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name == "PointSpiralMaze-v0":         
            obstacle_point_x = np.array([-2, 6, 6, -6, -6, 10, 10, -10, -10, 10, 10, -2, -2])
            obstacle_point_y = np.array([2, 2, 6, 6, -6, -6, -10, -10, 10, 10, -2, -2, 2])        
            ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    elif env_name in ["PointNMaze-v0"]:
        obstacle_point_x = np.array([-2, 6, 6, -2])
        obstacle_point_y = np.array([2, 2, 6, 6])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
        obstacle_point_x = np.array([10, 2, 2, 10])
        obstacle_point_y = np.array([10, 10, 14, 14])        
        ax.plot(obstacle_point_x, obstacle_point_y, c = 'black')
    else:
        pass

    ax.set_title('meta_nml_prob_visualize')        
    ax.axis([grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
    fig.colorbar(c, ax=ax)
    ax.axis('tight')
    plt.legend(loc="best")
    plt.savefig(savedir_w_name+'.jpg')
    plt.close()   
    disc_vis_end_time = time.time()
    # print('meta nml prob visualize time : {}'.format(disc_vis_end_time - disc_vis_start_time))

