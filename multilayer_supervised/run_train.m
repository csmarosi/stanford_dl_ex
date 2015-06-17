% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 296; %TODO:it was 1e6, but that takes too long time.
options.Method = 'lbfgs';
options.useMex = 0;

if exist('neuralParams.oct', 'file')
    options.maxFunEvals = 132;
    load neuralParams.oct
    params = opt_params;
    params = params ./ max(abs(params));
    params += stack2params(stack); %Still want the random thing...
end

kaggleComp = exist('train.csv', 'file')
if kaggleComp
    trainMatrix = csvread('train.csv');
    testMatrix = csvread('test.csv');
    data_train = trainMatrix(2:41000, 2:785)';
    data_train = double(data_train)/255;
    labels_train = trainMatrix(2:41000, 1);
    labels_train = labels_train + 10*(labels_train == 0);
    data_test = trainMatrix(41000:end, 2:785)';
    data_test = double(data_test)/255;
    labels_test = trainMatrix(41000:end, 1);
    labels_test = labels_test + 10*(labels_test == 0);
    data_kaggle = testMatrix(2:end,:)';
end

warning ("off", "Octave:broadcast")
%grad_err = grad_check(@supervised_dnn_cost,...
%    params, 3, ei, data_train(:,1:789), labels_train(1:789))
%% run training
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);

save neuralParams.oct opt_params

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);
[cost, ~, ~] = supervised_dnn_cost( opt_params, ei, data_test, labels_test)

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
[cost, ~, ~] = supervised_dnn_cost( opt_params, ei, data_train, labels_train)

if kaggleComp
    [~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_kaggle, [], true);
    [~,pred] = max(pred);
    pred = mod(pred, 10);
    csvwrite('testOut.csv', [(1:size(pred,2))', pred']);
end
