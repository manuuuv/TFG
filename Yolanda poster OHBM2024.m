num_input_neurons = 64;      % Dimensionality of the input  [64 EEG channnels)
num_reservoir_neurons = 10;  % Number of reservoir neurons
num_readout_neurons = 1;      % Number of readout neurons 
% this is not so important here

num_training_samples = length(input);   % Number of training samples

% Generate input to reservoir weights
input2res = randn(num_reservoir_neurons, num_input_neurons);
% Generate resevoir recurrent  weights
res2res = randn(num_reservoir_neurons, num_reservoir_neurons);

%put weights columns on a unit circle
for i=1:num_reservoir_neurons
res2res(:,i)= res2res(:,i)./norm(res2res(:,i));
end

% Generate resevoir bias
input_bias = rand(num_reservoir_neurons, 1);

%Initial reservoir states
reservoir_states = randn(num_reservoir_neurons,1);

for i = 1:num_training_samples
reservoir_states(:, i+1) = tanh( (.05*input2res*input(:,i)) + (.95 * res2res * reservoir_states(:, i)) + input_bias);
end

[YUPPER,YLOWER] = envelope(reservoir_states(:,2:end)',10,    'peak');
input_sequence  = YUPPER(1000:end, :)';  % input for clustering, removing the first 1000 point to ensure stability in the reservoir_states
[idx, C] = kmeans(input_sequence', 4);   %assuming 4 states
