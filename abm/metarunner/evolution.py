"""
evolution.py: including the main classes and methods to evolve the system and constrain the evolution
"""
import glob
import json

from abm.contrib.evolution import behave_params_template
import numpy as np
import os
from dotenv import dotenv_values
import warnings
import shutil
from abm import app
from time import sleep
import random
import matplotlib.pyplot as plt
from multiprocessing import Process

EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
envconf = dotenv_values(env_path)

def generate_env_file(env_data, file_name, save_folder):
    """Generating a single env file under save_folder with file_name including env_data as env format"""
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, file_name)
    with open(file_path, "a") as file:
        for k, v in env_data.items():
            file.write(f"{k}={v}\n")

class EvoProtocol:
    """Main class for evolutionary optimization"""
    def __init__(self, num_generations=1, gen_lifetime=100, headless=False,
                 death_rate_limits=(0.1, 0.2), mutation_rates=None, initial_genes=None, continue_evo=True,
                 num_populations=None):
        self.default_envconf = envconf
        self.envconf = self.default_envconf.copy()
        self.headless = headless

        self.num_generations = num_generations
        self.curr_gen = 0
        self.gen_lifetime = gen_lifetime
        self.death_rate_limits = death_rate_limits
        if self.death_rate_limits[0] == 1:
            self.replace_whole_population = True
        else:
            self.replace_whole_population = False

        self.envconf["T"] = self.gen_lifetime
        self.initial_genes = initial_genes
        self.mutation_rates = mutation_rates
        self.temp_dir = f"abm/data/evoexperiment"
        self.continue_evo = continue_evo

        self.num_populations = num_populations

    def prepare_phenotypes(self, initial_genes):
        agent_behave_param_list = []
        # setting up initial phenotypes
        for i in range(int(self.envconf["N"])):
            agent_behave_param_list.append(behave_params_template.copy())

        # randomly distribute requested genes
        for gene, gene_val_list in initial_genes.items():
            random.shuffle(gene_val_list)
            for i, gene_val in enumerate(gene_val_list):
                agent_behave_param_list[i][gene] = gene_val

        return agent_behave_param_list

    def set_env_var(self, key, val):
        """Setting environment variable of the simulations that should remain the same throughout all generations,
        such as the resource environment, etc."""
        self.envconf[key] = val

    def change_summary_folder(self, agent_behave_param_list, save_dir):
        """changing the folder where the simulation will save evo summary according
        to current generation number"""
        for i, behave_params in enumerate(agent_behave_param_list):
            behave_params["evo_summary_path"] = save_dir
            if i == 0:
                os.makedirs(behave_params["evo_summary_path"], exist_ok=True)
        return agent_behave_param_list

    def reproduce_random(self, agent_list, weights, mutation_factor=1):
        """Randomly choosing agents from a list of dictionaries of winner agents
        and reproducing with chance of mutation in all genes resulting in a single baby."""
        # Choosing parents
        parent1 = random.choices(agent_list, weights=weights, k=1)[0]
        parent2 = parent1.copy()
        baby = parent1.copy()

        while parent2 == parent1:
            parent2 = random.choices(agent_list, weights=weights, k=1)[0]

        print(f"Chosen parents id: {parent1['id']} - {parent2['id']}")

        # Cross-over
        for gene, _ in self.initial_genes.items():
            gene_pool = [parent1[gene], parent2[gene]]
            print(f"Gene pool for {gene}: {gene_pool}")
            co_gene = random.choice(gene_pool)  # HEE HEE
            # co_gene = np.random.uniform(gene_pool[0], gene_pool[1])
            print(f"Gene after crossover: {co_gene}")
            baby[gene] = co_gene

        # Mutating genes
        for gene, mutation_dict in self.mutation_rates.items():
            mutation_prob = mutation_dict["prob"] * mutation_factor
            print(f"Current mutation probability in gene {gene}: {mutation_prob}")
            if random.uniform(0, 1) < mutation_prob:
                mutation_extent = np.random.normal(loc=mutation_dict["mean"], scale=mutation_dict["std"])
                print(f"Mutating gene {gene} with {mutation_extent}")
                baby[gene] += mutation_extent
                baby[gene] = max(mutation_dict["min"], baby[gene])
                baby[gene] = min(mutation_dict["max"], baby[gene])

        return baby

    def calculate_theoretical_max_collres(self, agent_behave_param_list):
        """Calculating the theoretical maximum of collected resources on the collective level"""
        agent_consumption_list = [agent["agent_consumption"] for agent in agent_behave_param_list]
        print(f"Agent consumptions: {agent_consumption_list}")
        res_quality = self.envconf["MIN_RESOURCE_QUALITY"]
        print(f"Resource quality: {res_quality}")
        theo_agcons_per_time = [min(res_quality, ag_cons) for ag_cons in agent_consumption_list]
        print(f"Theo agent consumptions: {theo_agcons_per_time}")
        num_timesteps = self.gen_lifetime
        print(num_timesteps)
        theo_max = sum([ag_cons * num_timesteps for ag_cons in theo_agcons_per_time])
        print(f"final: {theo_max}")
        return theo_max


    def reproduction_cycle(self, agent_behave_param_list, parent_weights=None):
        """Choosing the winners of the generation and killing loosers, reproducing and
        mutating winning agents"""
        # Loading summary from previous generation
        evo_sum_path = agent_behave_param_list[0]["evo_summary_path"]
        sum_json = os.path.join(evo_sum_path, "evo_agent_summary.json")
        with open(sum_json, "r") as f:
            agent_sum = json.load(f)

        # Extracting collective return
        collective_return = agent_sum["collected_collective"]
        if collective_return == 0:
            collective_return = 1
        del agent_sum["collected_collective"]

        # Sorting agents according to ind. return
        agent_sum_sorted = []
        for id, behave_params in agent_sum.items():
            behave_params['id'] = int(id)
            agent_sum_sorted.append(behave_params)
        agent_sum_sorted = sorted(agent_sum_sorted, key=lambda x: x['collected_individ'], reverse=True)
        print(f"Agents order according to ind. return: {[ag_d['id'] for ag_d in agent_sum_sorted]}")

        # Killing selected agents
        if not self.replace_whole_population:
            death_rate = np.random.uniform(self.death_rate_limits[0], self.death_rate_limits[1])
            num_agents_to_kill = int(np.floor(len(agent_sum_sorted) * death_rate))
            death_weights = [1 - ((ag['collected_individ']+0.25) / collective_return) for ag in agent_sum_sorted]
            death_weights = [(float(w) - min(death_weights)) / (max(death_weights) - min(death_weights)) for w in death_weights]

            print("Death weights: ", death_weights)
            print(f"Killing {num_agents_to_kill} agents!")
            for i in range(num_agents_to_kill):
                agent_idx_to_kill = random.choices(range(len(agent_sum_sorted)), weights=death_weights, k=1)[0]
                print(f"Killing agent with index {agent_idx_to_kill}")
                del agent_sum_sorted[agent_idx_to_kill]
                del death_weights[agent_idx_to_kill]

        else:
            num_agents_to_kill = len(agent_sum_sorted)

        # Reproducing remaining agents
        babies = []
        if parent_weights is None:
            parent_weights = [(ag['collected_individ']+0.25/collective_return) for ag in agent_sum_sorted]
            parent_weights = [(float(w) - min(parent_weights)) / (max(parent_weights) - min(parent_weights)) for w in
                             parent_weights]
        print("Parent weights: ", parent_weights)

        # theo_max = self.calculate_theoretical_max_collres(agent_behave_param_list)
        # mutation_factor = 1-(collective_return/theo_max)
        # print(f"Generation collected {collective_return} of the theoretical maximum {theo_max} yielding"
        #       f" mutation factor 1 - {collective_return/theo_max} = {mutation_factor}")
        ## temp
        mutation_factor = 1

        for i in range(num_agents_to_kill):
            baby = self.reproduce_random(agent_sum_sorted, parent_weights, mutation_factor=mutation_factor)
            print("New baby has born:")
            for gene, _ in self.initial_genes.items():
                print(f"{gene}: {baby[gene]}")  # HEE HEE
            babies.append(baby)

        # Merging population
        if not self.replace_whole_population:
            agent_behave_param_list = agent_sum_sorted + babies
            print(f"Merged population to length {len(agent_behave_param_list)}")
        else:
            agent_behave_param_list = babies
            print(f"Wiped whole generation, kept babies only with size {len(agent_behave_param_list)}")

        # Cleaning resulting dictionaries and returning result
        for behave_params in agent_behave_param_list:
            del behave_params['id']
            del behave_params['collected_individ']

        return agent_behave_param_list

    def show_evolution_plot(self):
        """Showing evolution through a violin plot"""
        num_genes = len(self.initial_genes)
        fig, ax = plt.subplots(num_genes, 1)

        # Reloading and reorganizing summary files
        final_summary = {}
        coll_returns = []
        for gen_num in range(self.num_generations):
            save_dir = os.path.join("abm/data/simulation_data", f"EVO{EXP_NAME}", f"generation_{gen_num}")
            sum_path = os.path.join(save_dir, "evo_agent_summary.json")
            with open(sum_path, "r") as f:
                summary = json.load(f)
                coll_returns.append(float(summary["collected_collective"]))

            gene_sum = {}
            for gene, _ in self.initial_genes.items():
                gene_sum[gene] = [ag[gene] for ag in list(summary.values()) if isinstance(ag, dict)]
            final_summary[gen_num] = gene_sum

        # Calculating mean values
        gene_means = {}
        for gene, _ in self.initial_genes.items():
            gene_means[gene] = np.zeros(self.num_generations)
            gene_means[f"STD{gene}"] = np.zeros(self.num_generations)
            for gen_num in range(self.num_generations):
                gene_means[gene][gen_num] = np.mean(final_summary[gen_num][gene])
                gene_means[f"STD{gene}"][gen_num] = np.std(final_summary[gen_num][gene])

        # Shoving violinplots
        for i, gene in enumerate(self.initial_genes.keys()):
            if num_genes > 1:
                plt.axes(ax[i])
            # plt.errorbar([i for i in range(self.num_generations)], gene_means[gene], gene_means[f"STD{gene}"], marker='s', mfc='red',
            #          mec='green', ms=2, mew=2)
            plt.violinplot([final_summary[gen_num][gene] for gen_num in range(self.num_generations)], positions=[i for i in range(self.num_generations)],  widths=1.5, showmeans=False, showmedians=True)
            plt.plot(gene_means[gene])
            plt.title(f"Gene {gene}")
            if i == 0:
                if num_genes == 1:
                    ax2 = ax.twinx()
                else:
                    ax2 = ax[0].twinx()
            ax2.plot(coll_returns, c="black")

        plt.legend()
        plt.show()


    def show_metaevolution_plot(self):
        """Showing evolution through a violin plot"""
        num_genes = len(self.initial_genes)
        fig, ax = plt.subplots(num_genes, 1)

        # Reloading and reorganizing summary files
        final_summary = {}
        coll_returns = []
        for gen_num in range(self.num_generations):
            save_dir = os.path.join("abm/data/simulation_data", f"EVO{EXP_NAME}", f"generation_{gen_num}", "winner")
            sum_path = os.path.join(save_dir, "evo_agent_summary.json")
            with open(sum_path, "r") as f:
                summary = json.load(f)
                coll_returns.append(float(summary["collected_collective"]))

            gene_sum = {}
            for gene, _ in self.initial_genes.items():
                gene_sum[gene] = [ag[gene] for ag in list(summary.values()) if isinstance(ag, dict)]
            final_summary[gen_num] = gene_sum

        # Calculating mean values
        gene_means = {}
        for gene, _ in self.initial_genes.items():
            gene_means[gene] = np.zeros(self.num_generations)
            gene_means[f"STD{gene}"] = np.zeros(self.num_generations)
            for gen_num in range(self.num_generations):
                gene_means[gene][gen_num] = np.mean(final_summary[gen_num][gene])
                gene_means[f"STD{gene}"][gen_num] = np.std(final_summary[gen_num][gene])

        # Shoving violinplots
        for i, gene in enumerate(self.initial_genes.keys()):
            if num_genes > 1:
                plt.axes(ax[i])
            # plt.errorbar([i for i in range(self.num_generations)], gene_means[gene], gene_means[f"STD{gene}"], marker='s', mfc='red',
            #          mec='green', ms=2, mew=2)
            plt.violinplot([final_summary[gen_num][gene] for gen_num in range(self.num_generations)], positions=[i for i in range(self.num_generations)],  widths=1.5, showmeans=False, showmedians=True)
            plt.plot(gene_means[gene])
            plt.title(f"Gene {gene}")
            if i == 0:
                if num_genes == 1:
                    ax2 = ax.twinx()
                else:
                    ax2 = ax[0].twinx()
            ax2.plot(coll_returns, c="black")


        plt.legend()
        plt.show()


    def start_evolution(self, with_show=False):
        """Starting evolution for number of generations. In between carrying out evaluation and reproduction
        with mutation"""
        # Preparing environment
        root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        temp_dir = os.path.join(root_abm_dir, self.temp_dir)
        if os.path.isdir(temp_dir):
            warnings.warn("Temporary directory for env files is not empty and will be overwritten")
            shutil.rmtree(temp_dir)

        generate_env_file(self.envconf, f"{EXP_NAME}evoexp_environment.env", temp_dir)
        env_path = os.path.join(temp_dir, f"{EXP_NAME}evoexp_environment.env")
        default_env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
        backup_default_env = os.path.join(root_abm_dir, ".env-orig")
        if os.path.isfile(default_env_path) and not os.path.isfile(backup_default_env):
            shutil.copyfile(default_env_path, backup_default_env)

        # Preparing initial phenotypes
        agent_behave_param_list = self.prepare_phenotypes(self.initial_genes)

        # start evolving the system
        for gen_num in range(self.num_generations):
            # defining save directory
            self.curr_gen = gen_num
            save_dir = os.path.join("abm/data/simulation_data", f"EVO{EXP_NAME}", f"generation_{gen_num}")
            if os.path.isdir(save_dir) and self.continue_evo:
                print(f"Generation along path {save_dir} exists and continue parameter is 1, continue evolution from there...")
                sum_json = os.path.join(save_dir, "evo_agent_summary.json")
                if os.path.isfile(sum_json):
                    with open(sum_json, "r") as f:
                        agent_sum = json.load(f)
                del agent_sum["collected_collective"]
                agent_behave_param_list = list(agent_sum.values())
                for behave_params in agent_behave_param_list:
                    del behave_params['collected_individ']
                print("Loaded previous population. Initiating reproduction cycle")
                agent_behave_param_list = self.reproduction_cycle(agent_behave_param_list)
            else:
                self.envconf["SAVE_ROOT_DIR"] = save_dir
                generate_env_file(self.envconf, f"{EXP_NAME}evoexp_environment.env", temp_dir)
                agent_behave_param_list = self.change_summary_folder(agent_behave_param_list, save_dir)

                # copying and backing up environment
                os.remove(default_env_path)
                shutil.copy(env_path, default_env_path)

                # Run the simulation for 1 generation
                app.start(headless=self.headless, agent_behave_param_list=agent_behave_param_list)
                os.remove(default_env_path)
                shutil.copyfile(backup_default_env, default_env_path)
                sleep(1)

                # Evolving the system further for next generation
                agent_behave_param_list = self.reproduction_cycle(agent_behave_param_list)

        if with_show:
            self.show_evolution_plot()


    def check_if_gen_finished(self, gen_folder):
        """Checking if the given generation with all populations have been finished as these
        were running parallelly. To check it, we check the folder structure and all files"""
        is_finished = True
        pop_pattern = os.path.join(gen_folder, "population*")
        population_folders = [path for path in glob.iglob(pop_pattern)]
        print(population_folders)
        if len(population_folders) == self.num_populations:
            for population_folder in population_folders:
                if not os.path.isfile(os.path.join(population_folder, "evo_agent_summary.json")):
                    is_finished = False
                    break
        else:
            is_finished = False
        return is_finished


    def reproduce_winner_population(self, winner_pop):
        """Creating subpopulations of a successful winner population"""
        populations = []
        for pop_i in range(self.num_populations):
            new_population = self.reproduction_cycle(winner_pop) #, parent_weights=[1 for i in range(len(winner_pop))])
            populations.append(new_population)
        return populations

    def tournament_game(self, gen_folder, mark_winner=False):
        """Choosing winner population as tournament according to collective resources"""
        populations = []
        coll_res = []
        pop_pattern = os.path.join(gen_folder, "population*")
        population_folders = [path for path in glob.iglob(pop_pattern)]

        # filling up populations with evo summaries
        for pop_folder in population_folders:
            sum_path = os.path.join(pop_folder, "evo_agent_summary.json")
            with open(sum_path, "r") as f:
                summary = json.load(f)
                populations.append(summary)
                coll_res.append(summary["collected_collective"])

        print("Population resources: ", coll_res)

        winner_pop_ind = coll_res.index(max(coll_res))
        winner_pop_path = population_folders[winner_pop_ind]
        winner_pop = populations[winner_pop_ind]
        winner_pop_num = winner_pop_path[-1]

        print(f"Population with r={winner_pop['collected_collective']} wins.")
        del winner_pop["collected_collective"]
        winner_pop = list(winner_pop.values())
        for behave_params in winner_pop:
            del behave_params['collected_individ']

        if mark_winner:
            winner_path = os.path.join(gen_folder, "winner")
            shutil.copytree(winner_pop_path, winner_path)
            # os.rename(winner_pop_path, winner_pop_path+"_winner")
            winner_data_path = [path for path in glob.iglob(os.path.join(gen_folder, f"*_pop{winner_pop_num}"))][0]
            # os.rename(winner_data_path, winner_data_path + "_winner")
            shutil.copytree(winner_data_path, os.path.join(winner_path, os.path.basename(winner_data_path)))

        return winner_pop

    def start_meta_evolution(self):
        """Starting evolution for number of generations with number of parallel populations.
        In between carrying out evaluation and reproduction with mutation with population tournament according to
        collective return"""
        # Preparing environment
        root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        temp_dir = os.path.join(root_abm_dir, self.temp_dir)
        if os.path.isdir(temp_dir):
            warnings.warn("Temporary directory for env files is not empty and will be overwritten")
            shutil.rmtree(temp_dir)

        generate_env_file(self.envconf, f"{EXP_NAME}evoexp_environment.env", temp_dir)
        env_path = os.path.join(temp_dir, f"{EXP_NAME}evoexp_environment.env")
        default_env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
        backup_default_env = os.path.join(root_abm_dir, ".env-orig")
        if os.path.isfile(default_env_path) and not os.path.isfile(backup_default_env):
            shutil.copyfile(default_env_path, backup_default_env)

        # Preparing initial phenotypes
        populations = [self.prepare_phenotypes(self.initial_genes) for i in range(self.num_populations)]

        # start evolving the system
        for gen_num in range(self.num_generations):
            # defining save directory
            self.curr_gen = gen_num
            save_gen_dir = os.path.join("abm/data/simulation_data", f"EVO{EXP_NAME}", f"generation_{gen_num}")
            if os.path.isdir(save_gen_dir) and self.continue_evo:
                print(f"Generation along path {save_gen_dir} exists and continue parameter is 1, continue evolution from there...")
                sum_json = os.path.join(save_gen_dir, "winner","evo_agent_summary.json")
                if os.path.isfile(sum_json):
                    with open(sum_json, "r") as f:
                        agent_sum = json.load(f)
                del agent_sum["collected_collective"]
                agent_behave_param_list = list(agent_sum.values())
                for behave_params in agent_behave_param_list:
                    del behave_params['collected_individ']
                print("Loaded previous population. Initiating reproduction cycle")
                populations = self.reproduce_winner_population(agent_behave_param_list)

            else:
                processes = []
                for pop_i in range(self.num_populations):
                    save_pop_dir = os.path.join(save_gen_dir, f'population_{pop_i}')
                    self.envconf["SAVE_ROOT_DIR"] = save_gen_dir
                    generate_env_file(self.envconf, f"{EXP_NAME}evoexp_environment.env", temp_dir)
                    populations[pop_i] = self.change_summary_folder(populations[pop_i], save_pop_dir)
                    for ag_be_params in populations[pop_i]:
                        ag_be_params["population_num"] = pop_i

                    # copying and backing up environment
                    os.remove(default_env_path)
                    shutil.copy(env_path, default_env_path)

                    # Run the simulation for 1 generation
                    proc = Process(target=app.start, kwargs={'headless': self.headless,
                                                             "agent_behave_param_list": populations[pop_i]})
                    # app.start(headless=self.headless, agent_behave_param_list=agent_behave_param_list)
                    proc.start()
                    print(f"Started population {pop_i}")
                    sleep(0.2)

                for i, proc in enumerate(processes):
                    proc.join()
                    print(f"Join process {i}")

                # Waiting for all populations to finish generation
                generation_finished = False
                while not generation_finished:
                    generation_finished = self.check_if_gen_finished(save_gen_dir)
                    print("ISFINISHED: ", generation_finished)
                    sleep(2)

                winner_population = self.tournament_game(save_gen_dir, mark_winner=True)

                populations = self.reproduce_winner_population(winner_population)

        self.show_metaevolution_plot()