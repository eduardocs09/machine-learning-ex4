function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_recoded = zeros(size(y,1), num_labels);
for index = 1:m
    y_recoded(index, y(index)) = 1;
endfor

X = [ones(size(X, 1), 1) X];
a_2 = sigmoid(X * Theta1');
a_2 = [ones(size(a_2, 1), 1) a_2];
a_3 = sigmoid(a_2 * Theta2');

theta1_reg = Theta1;
theta1_reg(:, [1]) = [];
theta2_reg = Theta2;
theta2_reg(:, [1]) = [];
theta1_reg_sq = theta1_reg .^ 2;
theta2_reg_sq = theta2_reg .^ 2;

cf_vectrorized_difference = (-1 .* y_recoded .* log(a_3)) - (1 .- y_recoded) .* log(1 .- a_3); 
cf_firt_term = (1/m)*sum(cf_vectrorized_difference(:));
cf_second_term = (lambda/(2*m)) * (sum(theta1_reg_sq(:)) + sum(theta2_reg_sq(:)));
J = cf_firt_term + cf_second_term;

d_1_accum = zeros(size(Theta1_grad));
d_2_accum = zeros(size(Theta2_grad));

for t = 1:m
    x_t = X(t,:);
    y_t = y_recoded(t,:);
    
    z_2_t = Theta1 * x_t';
    a_2_t = sigmoid(z_2_t);
    a_2_t = [1; a_2_t];

    a_3_t = sigmoid(Theta2 * a_2_t);

    d_3_t = a_3_t .- y_t';

    d_2_t = (Theta2(:,2:end)' * d_3_t) .* sigmoidGradient(z_2_t);

    d_1_accum = d_1_accum + d_2_t * x_t;
    d_2_accum = d_2_accum + d_3_t * a_2_t';
endfor

theta1_reg_grad = [zeros(size(theta1_reg, 1), 1) theta1_reg];
theta2_reg_grad = [zeros(size(theta2_reg, 1), 1) theta2_reg];

Theta1_grad = (d_1_accum ./ m) + (lambda / m) * theta1_reg_grad;
Theta2_grad = (d_2_accum ./ m) + (lambda / m) * theta2_reg_grad;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
