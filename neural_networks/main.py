from neural_networks import nn_SCM, nn_EWCM_simple, nn_EWCM_umbrella

print('-'*15, 'SINGLE CLASSIFIER NEURAL NETWORK', '-'*15)
nn_SCM.nn_SCM()
print('-'*15, 'ENSEMBLE CLASSIFIER NEURAL NETWORK (SIMPLE)', '-'*15)
nn_EWCM_simple.nn_EWCM_simple()
print('-'*15, 'ENSEMBLE CLASSIFIER  NEURAL NETWORK (UMBRELLA)', '-'*15)
nn_EWCM_umbrella.nn_EWCM_umbrella()
