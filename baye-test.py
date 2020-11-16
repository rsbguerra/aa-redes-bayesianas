from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

#Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD


election_model = BayesianModel([
                            ('BrokeElectionLaw', 'Indicted'),
                            ('BrokeElectionLaw', 'FoundGuilty'),
                            ('Indicted', 'FoundGuilty'),
                            ('PoliticallyMotivatedProsecutor', 'Indicted'),
                            ('PoliticallyMotivatedProsecutor', 'FoundGuilty'),
                            ('FoundGuilty', 'Jailed')])


cpd_b = TabularCPD(variable='BrokeElectionLaw', variable_card=2,
                      values=[[.9], [0.1]])

cpd_m = TabularCPD(variable='PoliticallyMotivatedProsecutor', variable_card=2,
                       values=[[0.1], [0.9]])

cpd_i = TabularCPD(variable='Indicted', variable_card=2,
                        values=[
                                [0.1, 0.5, 0.5, 0.9], # p(I)
                                [0.9, 0.5, 0.5, 0.1]], # p(~I)
                        evidence=['BrokeElectionLaw', 'PoliticallyMotivatedProsecutor'], 
                        evidence_card=[2,2])

cpd_g = TabularCPD(variable='FoundGuilty', variable_card=2,
                        values=[[0.9, 0.8, 0, 0, 0.2, 0.1, 0, 0],  #p(G)
                                [0.1, 0.2, 1, 1, 0.8, 0.9, 1, 1]], #p(~G)
                        evidence=['BrokeElectionLaw', 'PoliticallyMotivatedProsecutor', 'Indicted'], 
                        evidence_card=[2,2,2])

cpd_j = TabularCPD(variable='Jailed', variable_card=2,
                      values=[[0.9, 0.0], 
                              [0.1, 1.0]],
                      evidence=['FoundGuilty'], evidence_card=[2])

print("Associating the parameters with the model structure\n")
election_model.add_cpds(cpd_b, cpd_i, cpd_m, cpd_g, cpd_j)
print(election_model.get_independencies())


from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator

samples = BayesianModelSampling(election_model).forward_sample(size=int(1e5))
samples.head()

mle = MaximumLikelihoodEstimator(model=election_model, data=samples)

print("\nEstimating the CPD for a single node.\n")
print(mle.estimate_cpd(node='FoundGuilty'))
print(mle.estimate_cpd(node='Jailed'))

# Estimating CPDs for all the nodes in the model
mle.get_parameters()[:10] # Show just the first 10 CPDs in the output

best = BayesianEstimator(model=election_model, data=samples)
print(best.estimate_cpd(node='Jailed', prior_type="K2", equivalent_sample_size=1000))