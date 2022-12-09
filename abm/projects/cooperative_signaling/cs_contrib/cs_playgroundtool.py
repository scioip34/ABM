from abm.contrib import playgroundtool as pgt


def setup_coop_sign_playground():
    playground_tool = pgt
    # default_params
    # update default parameters
    playground_tool.default_params["framerate"] = 60
    playground_tool.default_params["N_resc"] = 1
    playground_tool.default_params["patch_radius"] = 1
    # new default parameters
    playground_tool.default_params["collide_agents"] = False
    playground_tool.default_params["phototaxis_theta_step"] = 0.2
    playground_tool.default_params["detection_range"] = 120
    playground_tool.default_params["resource_meter_multiplier"] = 1
    playground_tool.default_params["des_velocity_res"] = 1.5
    playground_tool.default_params["res_theta_abs"] = 0.2
    playground_tool.default_params["signalling_cost"] = 0.2
    playground_tool.default_params["probability_of_starting_signaling"] = 0.5
    playground_tool.default_params["agent_signaling_rand_event_update"] = 10

    # update def_params_to_env_vars
    playground_tool.def_params_to_env_vars[
        "collide_agents"] = "AGENT_AGENT_COLLISION"
    playground_tool.def_params_to_env_vars[
        "phototaxis_theta_step"] = "PHOTOTAX_THETA_FAC"
    playground_tool.def_params_to_env_vars[
        "detection_range"] = "DETECTION_RANGE"
    playground_tool.def_params_to_env_vars[
        "resource_meter_multiplier"] = "METER_TO_RES_MULTI"
    playground_tool.def_params_to_env_vars[
        "signalling_cost"] = "SIGNALLING_COST"
    playground_tool.def_params_to_env_vars[
        "probability_of_starting_signaling"] = "SIGNALLING_PROB"
    playground_tool.def_params_to_env_vars["des_velocity_res"] = "RES_VEL"
    playground_tool.def_params_to_env_vars["res_theta_abs"] = "RES_THETA"
    playground_tool.def_params_to_env_vars["agent_signaling_rand_event_update"] = "SIGNAL_PROB_UPDATE_FREQ"

    # update help_messages
    playground_tool.help_messages["V_RES"] = '''
        Desired Patch Velocity [px/ts]:

        The desired absolute velocity of the resource patch in pixel per 
        timestep. 
        '''
    playground_tool.help_messages["DET_R"] = '''
        Detection Range [px]:

        detection range of agents in pixels. Resource patch is visualized 
        accordingly with the same radius.
        '''
    return playground_tool
