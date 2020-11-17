
import pylab as plt
import networkx as nx


#pgmpy imports
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling


#Desenho do grafo
election_model = BayesianModel([('BrokeElectionLaw', 'Indicted'),
                            ('BrokeElectionLaw', 'FoundGuilty'),
                            ('Indicted', 'FoundGuilty'),
                            ('PoliticallyMotivatedProsecutor', 'Indicted'),
                            ('PoliticallyMotivatedProsecutor', 'FoundGuilty'),
                            ('FoundGuilty', 'Jailed')])

nx.draw(election_model, with_labels=True)
plt.show()

#Criacao dos nodos
cpd_b = TabularCPD(variable='BrokeElectionLaw', variable_card=2,
                      values=[[.9], [0.1]])

cpd_m = TabularCPD(variable='PoliticallyMotivatedProsecutor', variable_card=2,values=[[0.1], [0.9]])

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


#Associar os model aos nodos
election_model.add_cpds(cpd_b, cpd_i, cpd_m, cpd_g, cpd_j)

#Verificar as independencias
print(election_model.get_independencies())

samples = BayesianModelSampling(election_model).forward_sample(size=int(1e5))
samples.head()

#Mostrar estimativas
mle = MaximumLikelihoodEstimator(model=election_model, data=samples)
print("\nEstimating the CPD for a single node.\n")
print(mle.estimate_cpd(node='BrokeElectionLaw'))
print(mle.estimate_cpd(node='PoliticallyMotivatedProsecutor'))
print(mle.estimate_cpd(node='Indicted'))
print(mle.estimate_cpd(node='FoundGuilty'))
print(mle.estimate_cpd(node='Jailed'))

