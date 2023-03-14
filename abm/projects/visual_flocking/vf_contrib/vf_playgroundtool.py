from abm.contrib import playgroundtool as pgt


def setup_visflock_playground():
    playground_tool = pgt
    # default_params
    # update default parameters
    playground_tool.default_params["framerate"] = 60
    playground_tool.default_params["N_resc"] = 0
    playground_tool.default_params["patch_radius"] = 0
    playground_tool.default_params["width"] = 900
    playground_tool.default_params["height"] = 700
    # # new default parameters
    playground_tool.default_params["collide_agents"] = False
    playground_tool.default_params["agent_radius"] = 5
    playground_tool.default_params["N"] = 10
    playground_tool.default_params["v_field_res"] = 400
    #playground_tool.default_params["alpha_0"] = 4
    # playground_tool.default_params["phototaxis_theta_step"] = 0.2
    # playground_tool.default_params["detection_range"] = 120
    # playground_tool.default_params["resource_meter_multiplier"] = 1
    # playground_tool.default_params["des_velocity_res"] = 1.5
    # playground_tool.default_params["res_theta_abs"] = 0.2
    # playground_tool.default_params["signalling_cost"] = 0.2
    # playground_tool.default_params["probability_of_starting_signaling"] = 0.5
    # playground_tool.default_params["agent_signaling_rand_event_update"] = 10

    # # update def_params_to_env_vars
    # playground_tool.def_params_to_env_vars[
    #     "collide_agents"] = "AGENT_AGENT_COLLISION"
    # playground_tool.def_params_to_env_vars[
    #     "phototaxis_theta_step"] = "PHOTOTAX_THETA_FAC"
    # playground_tool.def_params_to_env_vars[
    #     "detection_range"] = "DETECTION_RANGE"
    # playground_tool.def_params_to_env_vars[
    #     "resource_meter_multiplier"] = "METER_TO_RES_MULTI"
    # playground_tool.def_params_to_env_vars[
    #     "signalling_cost"] = "SIGNALLING_COST"
    # playground_tool.def_params_to_env_vars[
    #     "probability_of_starting_signaling"] = "SIGNALLING_PROB"
    # playground_tool.def_params_to_env_vars["des_velocity_res"] = "RES_VEL"
    # playground_tool.def_params_to_env_vars["res_theta_abs"] = "RES_THETA"
    # playground_tool.def_params_to_env_vars["agent_signaling_rand_event_update"] = "SIGNAL_PROB_UPDATE_FREQ"

    # # update help_messages
    playground_tool.help_messages["V.Resol."] = '''
        Resolution of Visual Projection Field [px]:

        Desired resolution of visual projection field, i.e. for how many pixel 
        is the retina of the agnt is divided. Ultimately this affects the visual range 
        of agents and their movement precision in very large distances and small resolutions.
        '''
    playground_tool.help_messages["R ag."] = '''
        Agent radius [px]:

        Radius of the agents in pixels.
        '''

    playground_tool.help_messages["Alpha0"] = '''
        Alpha 0 [AU]:

        Sensitivity of agents according to the main algorithm as in 
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7002123/        
        '''

    playground_tool.help_messages["Beta0"] = '''
        Beta 0 [AU]:

        Maneauverability of agents according to the main algorithm as in 
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7002123/        
        '''

    playground_tool.help_messages["Alpha1Beta1"] = '''
        Alpha 1 = Beta 1 [AU]:

        Higher level parameters of the flocking algorithm controlling front-back, and left-right
        equilibrium distances respectively. 
        '''

    playground_tool.help_messages["V0"] = '''
        V0 [px/ts]:

        Preferred/self-propelled speed of the agents in pixels/timestep
        '''

    playground_tool.help_messages["V0"] = '''
        Path length [ts]:

        Path length to save and visualize
        '''

    return playground_tool
