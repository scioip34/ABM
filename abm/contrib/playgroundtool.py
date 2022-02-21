"""parameters for interactive playground simulations"""

default_params = {
                "N": 5,
                "T": 100000,
                "v_field_res": 1200,
                "width": 500,
                "height": 500,
                "framerate": 30,
                "window_pad": 30,
                "with_visualization": True,
                "show_vis_field": False,
                "show_vis_field_return": True,
                "pooling_time": 0,
                "pooling_prob":0,
                "agent_radius": 10,
                "N_resc": 3,
                "min_resc_perpatch": 200,
                "max_resc_perpatch": 201,
                "min_resc_quality": 0.25,
                "max_resc_quality": 0.25,
                "patch_radius": 30,
                "regenerate_patches": True,
                "agent_consumption": 1,
                "teleport_exploit": False,
                "vision_range": 5000,
                "agent_fov": 0.5,
                "visual_exclusion": True,
                "show_vision_range": False,
                "use_ifdb_logging": False,
                "save_csv_files": False,
                "ghost_mode": True,
                "patchwise_exclusion": True,
                "parallel": False
}

help_messages = {
    'framerate': '''
    
                    Framerate [fps]: 
    
                    The framerate is a parameter that defines how often 
                    (per second) is the state of the simulation is
                    updated. In case the simulation has many agents/resources 
                    it might happen that the requested
                    framerate is impossible to keep with the given hardware. 
                    In that case the maximal possible framerate
                    will be kept. 
    
    ''',
    'N':'''
    
                    Number of agents, N [pcs]: 
    
                    The number of agent controls how many individuals the
                    group consists of in terms of foraging agents visualized
                    as colorful circles with a white line showing their ori-
                    entations. The field of view of the agents are shown
                    with a green circle slice.
    
    '''
}