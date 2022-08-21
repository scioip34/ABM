"""parameters for interactive playground simulations"""

VIDEO_SAVE_DIR = "abm/data/videos"

default_params = {
                "N": 5,  # interactive
                "T": 100000,  # interactive
                "v_field_res": 1200,
                "width": 500,
                "height": 500,
                "framerate": 30,  # interactive
                "window_pad": 30,
                "with_visualization": True,
                "show_vis_field": False,
                "show_vis_field_return": True,
                "pooling_time": 0,
                "pooling_prob":0,
                "agent_radius": 10,
                "N_resc": 3,  # interactive
                "min_resc_perpatch": 200,
                "max_resc_perpatch": 201,
                "min_resc_quality": 0.25,
                "max_resc_quality": 0.25,
                "patch_radius": 30,  # interactive
                "regenerate_patches": True,
                "agent_consumption": 1,
                "teleport_exploit": False,
                "vision_range": 5000,
                "agent_fov": 0.5,  # interactive
                "visual_exclusion": True,  # interactive
                "show_vision_range": True,
                "use_ifdb_logging": False,  # interactive
                "use_ram_logging": True,
                "use_zarr": True,
                "save_csv_files": True,
                "ghost_mode": True,  # interactive
                "patchwise_exclusion": True,
                "parallel": False,
                "allow_border_patch_overlap": True
}

# default parameters of playground that are not needed for initialization
def_env_vars = {
    "DEC_TW": "0.5",
    "DEC_EPSW": "2",
    "DEC_GW":"0.085",
    "DEC_BW": "0",
    "DEC_WMAX": "1",
    "DEC_TU": "0.5",
    "DEC_EPSU": "1",
    "DEC_GU": "0.085",
    "DEC_BU": "0",
    "DEC_UMAX": "1",
    "DEC_SWU": "0",
    "DEC_SUW": "0",
    "DEC_TAU": "10",
    "DEC_FN": "1",
    "DEC_FR": "1",
    "MOV_EXP_VEL_MIN": "2",
    "MOV_EXP_VEL_MAX": "2",
    "MOV_EXP_TH_MIN": "-0.5",
    "MOV_EXP_TH_MAX": "0.5",
    "MOV_REL_DES_VEL": "2",
    "MOV_REL_TH_MAX": "0.5",
    "CONS_STOP_RATIO": "0.125"
    }

def_params_to_env_vars = {
                "N": "N",
                "T": "T",
                "v_field_res": "VISUAL_FIELD_RESOLUTION",
                "width": "ENV_WIDTH",
                "height": "ENV_HEIGHT",
                "framerate": "INIT_FRAMERATE",
                "with_visualization": "WITH_VISUALIZATION",
                "show_vis_field": "SHOW_VISUAL_FIELDS",
                "show_vis_field_return": "SHOW_VISUAL_FIELDS_RETURN",
                "pooling_time": "POOLING_TIME",
                "pooling_prob": "POOLING_PROBABILITY",
                "agent_radius": "RADIUS_AGENT",
                "N_resc": "N_RESOURCES",
                "min_resc_perpatch": "MIN_RESOURCE_PER_PATCH",
                "max_resc_perpatch": "MAX_RESOURCE_PER_PATCH",
                "min_resc_quality": "MIN_RESOURCE_QUALITY",
                "max_resc_quality": "MAX_RESOURCE_QUALITY",
                "patch_radius": "RADIUS_RESOURCE",
                "regenerate_patches": "REGENERATE_PATCHES",
                "agent_consumption": "AGENT_CONSUMPTION",
                "teleport_exploit": "TELEPORT_TO_MIDDLE",
                "vision_range": "VISION_RANGE",
                "agent_fov": "AGENT_FOV",
                "visual_exclusion": "VISUAL_EXCLUSION",
                "show_vision_range": "SHOW_VISION_RANGE",
                "use_ifdb_logging": "USE_IFDB_LOGGING",
                "use_ram_logging": "USE_RAM_LOGGING",
                "save_csv_files": "SAVE_CSV_FILES",
                "use_zarr": "USE_ZARR_FORMAT",
                "ghost_mode": "GHOST_WHILE_EXPLOIT",
                "patchwise_exclusion": "PATCHWISE_SOCIAL_EXCLUSION",
                "allow_border_patch_overlap": "PATCH_BORDER_OVERLAP"
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
    
    ''',
    'N_res':'''
    
            Number of resource patches, N_res [pcs]: 

            The number of resource patches in the environment. 
            Each resource patch can be exploited by agents and they
            contain a given amount of resource units. The quality
            of the patch controls how fast (unit/time) can be
            the resource patch exploited by a single agent. Resource
            patches are shown as gray circles on the screen.
    
    ''',
    'FOV': '''

            Field of View, FOV [%]: 

            The amount (in percent) of visible area around individual
            agents. In case this is 0, the agents are blind. In case
            it is 100, the agents have a full 360° FOV.

    ''',
    'RES': '''

            Resource Radius, R [px]: 

            The radius of resource patches in pixels. In case the overall
            covered resource area on the arena exceeds 30% of the total space
            the number of resource patches will be automatically decreased. 

    ''',
    'Epsw': '''

            Social Excitability, E_w [a.U.]: 

            The parameter controls how socially excitable agents are, i.e. how
            much a unit of social information can increase/bias the decision
            process of the agent towards socially guided behavior (Relocation).
            In case this parameter is 0, agents do not integrate social cues
            at all. The larger this parameter is, the faster individuals respond
            to visible exploiting agents and the farther social cues are triggering
            relocation movement.

    ''',
    'Epsu': '''

            Individual Preference Factor, E_u [a.U.]: 
    
            The parameter controls how much a unit of exploited resource biases
            the agent towards further exploitation of resources.
            In case this parameter is low, agents will get picky
            with resource patches, i.e. they stop exploiting low-quality
            patches after an initial sampling period.

    ''',
    'SWU': '''

            W to U Cross-inhibition strength, S_wu [a.U.]: 

            The parameter controls how much socially guided behavior and
            integration of social cues inhibit individual exploitation
            behavior. Note, that this inhibition is only present if the
            social integrator w is above it's decision threshold.

    ''',
    'SUW': '''

            U to W Cross-inhibition strength, S_uw [a.U.]: 

            The parameter controls how much individual exploitation behavior and
            integration of individual information inhibits social
            behavior. Note, that this inhibition is only present if the
            individual integrator u is above it's decision threshold.

    ''',
    'SUMR': '''

            Total number of resource units, SUM_r [a.U.]: 

            The parameter controls how many resource units should be overall
            distributed in the environment. This number can be fixed with the
            "Fix Total Units" action button. In this case changing the number 
            of patches will redistribute the units between the patches so
            that although the ratio between patches stays the same, the
            number of units per patch will change.

    '''
}