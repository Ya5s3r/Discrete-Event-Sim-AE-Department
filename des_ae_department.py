from statistics import mean
import simpy
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np
import math

# font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
# font_manager.findfont("Gill Sans")

"""Lognormal class courtesy of Thomas Monks, Associate Professor of Health Data Science at The University of Exeter."""
class Lognormal:
    """
    Encapsulates a lognormal distirbution
    """
    def __init__(self, mean, stdev, random_seed=None):
        """
        Params:
        -------
        mean = mean of the lognormal distribution
        stdev = standard dev of the lognormal distribution
        """
        self.rand = np.random.default_rng(seed=random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma
        
    def normal_moments_from_lognormal(self, m, v):
        '''
        Returns mu and sigma of normal distribution
        underlying a lognormal with mean m and variance v
        source: https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal
        -data-with-specified-mean-and-variance.html

        Params:
        -------
        m = mean of lognormal distribution
        v = variance of lognormal distribution
                
        Returns:
        -------
        (float, float)
        '''
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2/phi)
        sigma = math.sqrt(math.log(phi**2/m**2))
        return mu, sigma
        
    def sample(self):
        """
        Sample from the normal distribution
        """
        return self.rand.lognormal(self.mu, self.sigma)


"""start of SimPy model"""

# class to hold global parameters - used to alter model dynamics
class p:
    # interarrival mean for exponential distribution sampling
    inter = 5
    # mean and stdev for lognormal function which converts to Mu and Sigma used to sample from lognoral distribution
    mean_doc_consult = 30
    stdev_doc_consult = 10
    mean_nurse_triage = 10
    stdev_nurse_triage = 5

    number_docs = 3
    number_nurses = 2
    ae_cubicles = 7

    # mean time to wait for an inpatient bed if decide to admit
    mean_ip_wait = 90
    
    # simulation run metrics
    warm_up = 120
    sim_duration = 480
    number_of_runs = 250

    # some placeholders used to track wait times for resources
    wait_triage = []
    wait_cubicle = []
    wait_doc = []

    # MIU metrics
    mean_doc_consult_miu = 20
    stdev_doc_consult_miu = 7

    number_docs_miu = 2
    number_nurses_miu = 3
    miu_cubicles = 5

    wait_doc_miu = []


class Tracker: # currently tracking number of triage waiters
    def __init__(self) -> None:
        # some place holders to track number of waiters by points in time
        self.env_time_all = []
        self.waiters = {
            'triage': [],
            'cubicle': [],
            'ae_doc': [],
            'miu_doc': []
        }
        self.waiters_all = {
            'triage': [],
            'cubicle': [],
            'ae_doc': [],
            'miu_doc': []
        }
        # empty df to hold patient level details, including time in system, priority etc
        self.results_df = pd.DataFrame()
        self.results_df["P_ID"] = []
        self.results_df["Priority"] = []
        self.results_df["TriageOutcome"] = []
        self.results_df["TimeInSystem"] = []
        self.results_df.set_index("P_ID", inplace=True)

    def plot_data(self, env_time, type):
        if env_time > p.warm_up:
            self.waiters_all[type].append(len(self.waiters[type]))
            self.env_time_all.append(env_time)

    def mean_priority_wait(self):
        self.priority_means = {}
        for i in range(1, 6):
            try:
                self.priority_means["Priority{0}".format(i)] = mean(self.results_df[self.results_df['Priority'] == i]['TimeInSystem'])
            except:
                self.priority_means["Priority{0}".format(i)] = np.NaN

    def priority_count(self):
        self.priority_counts = {}
        for i in range(1, 6):
            try:
                self.priority_counts["Priority{0}".format(i)] = len(self.results_df[self.results_df['Priority'] == i]['TimeInSystem'])
            except:
                self.priority_counts["Priority{0}".format(i)] = 0

# class representing patients coming in
class AEPatient:
    def __init__(self, p_id) -> None:
        self.p_id = p_id
        self.time_in_system = 0

    def set_priority(self):
        # set priority according to weighted random choices - most are moderate in priority
        self.priority = random.choices([1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.2, 0.1])[0]

    def set_triage_outcome(self):
        # decision tree - if priority 5, go to Minor Injury Unit (MIU) or home. Higher priority go to AE
        if self.priority <5:
            self.triage_outcome = 'AE'
        elif self.priority == 5: # of those who are priority 5, 20% will go home with advice, 80% go to 'MIU'
            self.triage_outcome = random.choices(['home', 'MIU'], [0.2, 0.8])[0]

# class representing AE model
class AEModel:
    # set up simpy env
    def __init__(self) -> None:
        self.env = simpy.Environment()
        self.patient_counter = 0
        # set docs and cubicles as priority resources - urgent patients get seen first
        self.doc = simpy.PriorityResource(self.env, capacity=p.number_docs)
        self.nurse = simpy.Resource(self.env, capacity=p.number_nurses)
        self.cubicle = simpy.PriorityResource(self.env, capacity=p.ae_cubicles)
        # MIU resources - all FIFO
        self.doc_miu = simpy.Resource(self.env, capacity=p.number_docs_miu)
        self.nurse_miu = simpy.Resource(self.env, capacity=p.number_nurses_miu)
        self.cubicle_miu = simpy.Resource(self.env, capacity=p.miu_cubicles)

    # a method that generates AE arrivals
    def generate_ae_arrivals(self):
        while True:
            # add pat
            self.patient_counter += 1

            # create class of AE patient and give ID
            ae_p = AEPatient(self.patient_counter)

            # simpy runs the attend ED methods
            self.env.process(self.attend_ae(ae_p))

            # Randomly sample the time to the next patient arriving to ae.  
            # The mean is stored in the g class.
            sampled_interarrival = random.expovariate(1.0 / p.inter)

            # Freeze this function until that time has elapsed
            yield self.env.timeout(sampled_interarrival)

    def attend_ae(self, patient):
        # this is where we define the pathway through AE
        triage_queue_start = self.env.now
        # track numbers waiting at each point
        track.plot_data(self.env.now, 'triage')
        track.plot_data(self.env.now, 'cubicle')
        track.plot_data(self.env.now, 'ae_doc')
        # request a triage nurse
        with self.nurse.request() as req:
            # append env time
            # track.plot_data(env.now)
            # append to current waiters
            track.waiters['triage'].append(patient)

            # freeze until request can be met
            yield req
            # remove from waiter list (FIFO)
            track.waiters['triage'].pop()
            # track.plot_data(env.now)
            triage_queue_end = self.env.now
            
            if self.env.now > p.warm_up:
                p.wait_triage.append(triage_queue_end - triage_queue_start)

            # sample triage time from lognormal
            lognorm = Lognormal(mean=p.mean_nurse_triage, stdev=p.stdev_nurse_triage)
            sampled_triage_duration = lognorm.sample()
            #sampled_triage_duration = random.expovariate(1.0 / p.mean_nurse_triage)
            # assign the patient a priority
            patient.set_priority()
            
            yield self.env.timeout(sampled_triage_duration)

        # sample chance of being sent home or told to wait for doc
        #proceed_to_doc = random.uniform(0,1)
        # alternative way to select choice
        patient.set_triage_outcome()

        if patient.triage_outcome == 'AE':
            cubicle_queue_start = self.env.now

            with self.cubicle.request(priority = patient.priority) as req_cub: # request cubicle before doctor
                # track cubicle
                track.waiters['cubicle'].append(patient)
                yield req_cub
                track.waiters['cubicle'].pop()
                cubicle_queue_end = self.env.now
                # record AE cubicle wait time
                if self.env.now > p.warm_up:
                        p.wait_cubicle.append(cubicle_queue_end - cubicle_queue_start)
                doc_queue_start = self.env.now

            # request doc if greater than chance sent home
                with self.doc.request(priority = patient.priority) as req_doc:
                    track.waiters['ae_doc'].append(patient)
                    yield req_doc
                    track.waiters['ae_doc'].pop()
                    doc_queue_end = self.env.now
                    if self.env.now > p.warm_up:
                        p.wait_doc.append(doc_queue_end - doc_queue_start)
                    # sample consult time from lognormal
                    lognorm = Lognormal(mean=p.mean_doc_consult, stdev=p.stdev_doc_consult)
                    sampled_consult_duration = lognorm.sample()

                    yield self.env.timeout(sampled_consult_duration)
                # below prob of request for IP bed. AE doc released but not cubicle
                ip_prob = random.uniform(0, 1)
                if ip_prob < 0.3:                    
                    sampled_ip_duration = random.expovariate(1.0 / p.mean_ip_wait) # sample the wait time for an IP bed - exponential dist
                    yield self.env.timeout(sampled_ip_duration)
                # else leave the system
                

        elif patient.triage_outcome == 'MIU':
            miu_attend_start = self.env.now

            with self.cubicle_miu.request() as req_cub:
                yield req_cub
                
                with self.doc_miu.request() as req:
                    yield req

                    miu_doc_queue_end = self.env.now
                    if self.env.now > p.warm_up:
                        p.wait_doc_miu.append(miu_doc_queue_end - miu_attend_start)
                    # sample consult time
                    lognorm = Lognormal(mean=p.mean_doc_consult_miu, stdev=p.stdev_doc_consult_miu)
                    sampled_consult_duration = lognorm.sample()

                    yield self.env.timeout(sampled_consult_duration)
        # else leave the system
        # record time in system
        patient.time_in_system = self.env.now - triage_queue_start
        if self.env.now > p.warm_up:
            df_to_add = pd.DataFrame({"P_ID":[patient.p_id],
                                      "Priority":[patient.priority],
                                      "TriageOutcome":[patient.triage_outcome],
                                      "TimeInSystem":[patient.time_in_system]})
            df_to_add.set_index("P_ID", inplace=True)
            frames = [track.results_df, df_to_add]
            track.results_df = pd.concat(frames)
            

    # method to run sim
    def run(self):
        self.env.process(self.generate_ae_arrivals())
        
        self.env.run(until=p.warm_up + p.sim_duration)
        # print(f"The mean wait for a triage nurse was {mean(p.wait_triage):.1f} minutes")
        # print(f"The mean wait for a AE doctor was {mean(p.wait_doc):.1f} minutes")
        # print(f"The mean wait for a MIU doctor was {mean(p.wait_doc_miu):.1f} minutes")
        # calculate mean waits per priority
        track.mean_priority_wait()
        track.priority_count()
        return mean(p.wait_triage), mean(p.wait_cubicle), mean(p.wait_doc), mean(p.wait_doc_miu)

     

# For the number of runs specified in the g class, create an instance of the
# AEModel class, and call its run method


all_runs_triage_mean = []
all_runs_cubicle_mean = []
all_runs_doc_mean = []
all_runs_miu_doc_mean = []
all_time_in_system = []
all_number_of_patients = []

all_run_time_wait_key = {
    'triage': {},
    'cubicle': {},
    'ae_doc': {},
    'miu_doc': {}
}

all_run_priority_time_in_system = {
    'Priority1': [],
    'Priority2': [],
    'Priority3': [],
    'Priority4': [],
    'Priority5': []
}

all_run_priority_counts = {
    'Priority1': [],
    'Priority2': [],
    'Priority3': [],
    'Priority4': [],
    'Priority5': []
}

for run in range(p.number_of_runs):
    #print (f"Run {run} of {p.number_of_runs}")

    track = Tracker()
    my_ae_model = AEModel()
    triage_mean, cubicle_mean , doc_mean, miu_mean = my_ae_model.run()
    all_runs_triage_mean.append(triage_mean)
    all_runs_cubicle_mean.append(cubicle_mean)
    all_runs_doc_mean.append(doc_mean)
    all_runs_miu_doc_mean.append(miu_mean)
    # number of patients served per run
    all_number_of_patients.append(len(track.results_df))
    # tracking number of waiters in key queues through sim
    for k in all_run_time_wait_key:
        for t, w in zip(track.env_time_all, track.waiters_all[k]):
            #print(t, w)
            all_run_time_wait_key[k].setdefault(round(t), [])
            all_run_time_wait_key[k][round(t)].append(w)
        all_run_time_wait_key[k] = dict(sorted(all_run_time_wait_key[k].items())) # sort items
    all_time_in_system.append(mean(track.results_df['TimeInSystem']))
    # get priority wait times
    for i in range(1, 6):           
        all_run_priority_time_in_system["Priority{0}".format(i)].append(track.priority_means["Priority{0}".format(i)])
    # number of patient per priority
    for i in range(1, 6):           
        all_run_priority_counts["Priority{0}".format(i)].append(track.priority_counts["Priority{0}".format(i)])
    #print ()

print(f"The average number of patients served by the system was {round(mean(all_number_of_patients))}")
print(f"The overall average wait across all runs for a triage nurse was {mean(all_runs_triage_mean):.1f} minutes")
print(f"The overall average wait across all runs for a cubicle was {mean(all_runs_cubicle_mean):.1f} minutes")
print(f"The overall average wait across all runs for a doctor was {mean(all_runs_doc_mean):.1f} minutes")
print(f"The overall average wait across all runs for a MIU doctor was {mean(all_runs_miu_doc_mean):.1f} minutes")
print(f"The mean patient time in the system across all runs was {mean(all_time_in_system):.1f} minutes")
#print(f"The mean patient time in the system across all runs was {mean(list(itertools.chain(*all_time_in_system))):.1f} minutes")


# number of patients per priority
patients_per_priority = []
for k in all_run_priority_counts:
    patients_per_priority.append(round(mean(all_run_priority_counts[k])))
patients_per_priority


wait_means = {
    'triage': [],
    'cubicle': [],
    'ae_doc': [],
    'miu_doc': []
}
# lower quartiles
wait_lq = {
    'triage': [],
    'cubicle': [],
    'ae_doc': [],
    'miu_doc': []
}
# upper quartiles
wait_uq = {
    'triage': [],
    'cubicle': [],
    'ae_doc': [],
    'miu_doc': []
}

for k in all_run_time_wait_key:
    for t in all_run_time_wait_key[k]:
        wait_means[k].append(round(mean(all_run_time_wait_key[k][t]), 2))
        wait_lq[k].append(np.percentile(all_run_time_wait_key[k][t], 25))
        wait_uq[k].append(np.percentile(all_run_time_wait_key[k][t], 75))

all_run_time_wait_key
wait_means
wait_lq

# change the default font family
plt.rcParams.update({'font.family':'Gill Sans'})
# reset the plot configurations to default
#plt.rcdefaults()
#plt
figure_1, ax = plt.subplots()
# Set x axis and y axis labels
ax.set_xlabel('Time', loc='right')
ax.set_ylabel('Mean Number of Waiters', loc='top')
ax.set_title('Mean Number of Patients Waiting per Simulator Time', loc='left')

# Add spines
ax.spines["top"].set(visible = False)
ax.spines["right"].set(visible = False)
# Add grid and axis labels
ax.grid(True, color = "lightgrey", ls = ":") 

# Plot our data (x and y here)
x_time = list(all_run_time_wait_key['triage'].keys())

ax.plot(x_time, wait_means['triage'], label='Triage')
ax.fill_between(x_time, wait_lq['triage'], wait_uq['triage'], alpha=.1)
ax.plot(x_time, wait_means['cubicle'], label='Cubicle')
ax.fill_between(x_time, wait_lq['cubicle'], wait_uq['cubicle'], alpha=.1)
ax.plot(x_time, wait_means['ae_doc'], label='AE Doctor')
ax.fill_between(x_time, wait_lq['ae_doc'], wait_uq['ae_doc'], alpha=.1)
# Create and set up a legend
ax.legend(loc="upper left")

# Show the figure
figure_1.savefig('mean_waiters_fig.png')


figure_2, ax = plt.subplots()
# Set x axis and y axis labels
#ax.set_xlabel('Priority')
ax.set_ylabel('Mean Time In System', fontname="Gill Sans", loc='top')
ax.set_title('Mean Patient Time in System by Priority', loc='left')
#plot
axes_labels = list(all_run_priority_time_in_system.keys())
bar_heights = [np.nanmean(all_run_priority_time_in_system[k]) for k in all_run_priority_time_in_system]
# below calculates the standard error for each priority np.std(data, ddof=1) / np.sqrt(np.size(data))
std_error = [np.nanstd(all_run_priority_time_in_system[k], ddof=1) / np.sqrt(np.size(all_run_priority_time_in_system[k])) 
             for k in all_run_priority_time_in_system]
# and below the standar deviations
stds = [np.nanstd(all_run_priority_time_in_system[k], ddof=1) for k in all_run_priority_time_in_system]

ax.bar(x=axes_labels, height=bar_heights, ec = "black", 
    yerr = stds,
    lw = .75, 
    color = "#005a9b", 
    zorder = 3, 
    width = 0.75)
for x, s in zip(range(0, 5), patients_per_priority):
    ax.text(x=x, y=10, s=s, horizontalalignment='center', color='red')
# Add spines
ax.spines["top"].set(visible = False)
ax.spines["right"].set(visible = False)
# Add grid and axis labels
ax.grid(True, color = "lightgrey", ls = ":")                                                        
figure_2.savefig('mean_time_in_system_priority_fig.png')


# example heat map
# import pandas as pd
# import seaborn as sns
# df_mean_waiters = pd.DataFrame({'Time': all_run_time_wait_key['triage'].keys(),
#               'Triage': wait_means['triage'],
#               'Cubicle': wait_means['cubicle'],
#               'AE Doctor': wait_means['ae_doc']})

# df_mean_waiters.set_index('Time', inplace=True)  

# sns.set()

# ax = sns.heatmap(df_mean_waiters.transpose())
# plt.title("Mean Number of Waiters")
# plt.show()


