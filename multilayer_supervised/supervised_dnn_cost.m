function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
layers=numHidden+1;
%% forward prop
z=cell(layers+1, 1);
a=cell(layers+1, 1);
a{1} = data;
for n = 1:layers
    z{n+1} = stack{n}.W * a{n} + stack{n}.b;
    a{n+1} = logisticFun(z{n+1});
end
pred_prob = a{layers+1} ./ sum(a{layers+1});

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
Ind = pred_prob;
for k = 1:size(pred_prob,1);
    Ind(k,:) = (labels == (k + 0*labels));
end
rawCostM = Ind - a{layers+1};
cost = sum(sum(rawCostM .* rawCostM)) / 2;

%% compute gradients using backpropagation
delta=cell(layers+1);
delta{layers+1} = -1 * rawCostM .* logistic1Der(z{layers+1});
for l = layers:-1:2
    delta{l} = (stack{l}.W' * delta{l+1}) .* logistic1Der(z{l});
end

for i = 1:layers
    gradStack{i}.b = sum(delta{i+1},2);
    gradStack{i}.W = delta{i+1}*(a{i}');
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end


function result = logisticFun(z)
    result = 1 ./ ( 1 + exp(-z) );
end
function result = logistic1Der(z)
    result = logisticFun(z) .* ( 1 - logisticFun(z) );
end
