
using CSV
using DataFrames
using Statistics
using Plots


data= CSV.read("/Users/adrianosilva/Downloads/bank-additional-full.csv")
df= DataFrame(data)
"""
Our Data preparation steps involved the following:
Y was converted to binary 0s and 1s, using a loop.
Apart from the selected attributes, extra dummy attributes were created because to use the one-hot encoding technique
From the original dataset selected attributes were:
AGE,
Education,
loan,
housing,
campain,
poutcome,
emp.var.rate
cons.price.idx
cons.conf.idx
nr.employed
One-hot encoding was performed on Education, housing, loan, and poutcome
"""
i = 1
y=zeros(Int,41188)
"""
The dummy variables where created in order to
implement one-hot encoding
To be able to binarize all the categorical attributes,
firstly attributes containing only zeros were created
then using a loop ones where added where applicable.
"""
#############Education Dummy variables##################################
dummy_basic=zeros(Int,41188) #The basics attributes variables will all be represented by a single variables named basic.
dummy_high_school=zeros(Int,41188)
dummy_illiterate=zeros(Int,41188)
dummy_university_degree=zeros(Int,41188)
dummy_professional_course=zeros(Int,41188)
##################Housing and dummy variables#####################
dummy_yes_housing=zeros(Int,41188)
dummy_no_housing=zeros(Int,41188)
#################Loan Dummy variables####################################
dummy_yes_loan=zeros(Int,41188)
dummy_no_loan=zeros(Int,41188)
##############Poutcome dummy variables##########################
dummy_failure_poutcome=zeros(Int,41188)
dummy_success_poutcome=zeros(Int,41188)

#Variables that went through one-hot encoding required the creation of new
#dummy variables and also to avoid the dummy variable trap some extra created variables
#were dropped(not considered)
for i in 1:41188
if df[21][i]=="yes"
y[i]=1 # Ones were added to the vector Y containing zeros where it was applicable by mapping to the original Y columns
end
if df[4][i]=="basic.4y" || df[4][i]=="basic.6y" || df[4][i]=="basic.9y"
dummy_basic[i]= 1
elseif df[4][i]=="high.school"
      dummy_high_school[i]=1
elseif df[4][i]=="professional.course"
      dummy_professional_course[i]=1
elseif df[4][i]=="university.degree"
      dummy_university_degree[i]=1

elseif df[4][i]=="illiterate"
      dummy_illiterate[i]=1
end
if df[6][i]=="yes"
      dummy_yes_housing[i]=1
elseif  df[6][i]=="no"
      dummy_no_housing[i]=1
end
if df[7][i]=="yes"
      dummy_yes_loan[i]=1
elseif df[7][i]=="no"
      dummy_no_loan[i]=1
end
if df[15][i]=="failure"
      dummy_failure_poutcome[i]=1
elseif df[15][i]=="success"
      dummy_success_poutcome[i]=1
end
end

x = DataFrame([df[1] dummy_basic dummy_high_school dummy_illiterate dummy_university_degree dummy_professional_course dummy_yes_housing dummy_no_housing dummy_yes_loan dummy_no_loan dummy_failure_poutcome dummy_success_poutcome df[16] df[17] df[18] df[19] df[20]])

x_matrix = convert(Matrix,x)
train_data_size= 0.8
dset_size= size(x_matrix)[1]
train_index = trunc(Int,train_data_size * dset_size)
#Divide X into training set and test set
x_train= x_matrix[1:train_index,:]
x_test= x_matrix[train_index+1:end,:]
#Divide Y into Training set and test set
y_train=y[1:train_index]
y_test=y[train_index+1:end]

##########Features normalization and scaling###########################

"""
This functions normalizes using min_max scaling method
"""
function min_max_scaling(x)
x_norm = (x .- extrema(x)[1]) ./ extrema(x)[2] .- extrema(x)[1]
x_mean = mean(x, dims=1)
x_std = std(x,dims=1)
return (x_norm, x_mean, x_std)
end

function test_scale(x,x_mean,x_std)
x_norm = (x .- x_mean) ./ x_std
return (x_norm)
end
"""
This functions normalizes features using Z scoring standardization
"""
function zScore_feature_standardization(x)
x_mean = mean(x, dims=1)
x_std = std(x, dims=1)
x_norm = (x .- x_mean) ./ x_std
return (x_norm, x_mean, x_std);
end

"""
Assign normalized features to the variables
"""
norm_x_train, x_mean, x_std = zScore_feature_standardization(x_train)
norm_x_test = test_scale(x_test,x_mean,x_std)
###############################################END OF DATA preparation############################################

"""
The sigmoid function
"""
function logistic(s)
return 1 ./ (1 .+ exp.(.-s))
end


"""
Cost function
"""
function loss_function(x, y, theta, lambda)
m = length(y)
h = logistic(x * theta)
l1_norm = (lambda/(2*m) * abs(sum(theta[2 : end])))
cost_Vector = (1/m) * ( ((-y)' * log.(h)) - ((1 .- y)' * log.(1 .- h))) + l1_norm
gd = (1/m) * (x') * (h-y) + ((1/m) * (lambda * theta))
gd[1] = (1/m) * (x[:, 1])' * (h-y)
return (cost_Vector, gd)

end



function lmodel(x, y, lambda, fit_intercept=true, η=0.01, max_iter=1000)
# Initialize some useful values
m = length(y); # number of training examples

# Add a constant of 1s if fit_intercept is specified
constant = ones(m, 1)
x = hcat(constant, x)

# Use the number of features to initialise the theta θ vector
n = size(x)[2]
theta = zeros(n)
"""
 Initialise the cost vector
"""
cost_vector = zeros(max_iter)
for iter in range(1, stop=max_iter)

"""
Calcaluate the cost and gradient during iteration
"""
cost_vector[iter], gd = loss_function(x, y, theta, lambda)
"""
Update theta using gradients  for direction and (η) for the magnitude of steps in that direction
"""
theta = theta - (η * gd)
end
return (theta, cost_vector)
end

"""
Use gradient descent to search for the optimal values
"""
theta, 𝐉 = lmodel(norm_x_train, y_train, 0.0001, true, 0.3, 3000);
plot(𝐉, color="red", title="Cost ", legend=false,
     xlabel="iterations", ylabel="Cost")
plot(y_train,norm_x_train, seriestype = :scatter, title = "My Scatter Plot");

"""
"""
function predictor(x, theta, fit_intercept=true)
m = size(x)[1]
if fit_intercept
# Add a constant of 1s if fit_intercept is specified
constant = ones(m, 1)
x = hcat(constant, x)
else
x
end
h = logistic(x * theta)
return h
end


 """
This function applies threshold to the probabilities
 """
function prediction_threshold(proba, threshold=0.5)
return proba .>= threshold
end
"""
Predict values and assign to a variable
"""
predicted_values = prediction_threshold(predictor(norm_x_test, theta))
TP=0
TN=0
FP=0
FN=0
"""
Captures True positives, True negatives, False positive and False negative
"""
for i in 1:8238
if predicted_values[i]==1 && y_test[i]== 1
global TP+=1
elseif  predicted_values[i]==0 && y_test[i]== 0
global TN+=1
elseif predicted_values[i]==1 && y_test[i]==0
    global FP+=1
elseif predicted_values[i]==0 && y_test[i]==1
    global FN+=1
end
end


print("True Positives =",TP)
print("True Negatives =",TN)
print("False Positives =",FP)
print("False Positives =",FN)

"""
CALCULATE PERFORMANCE METRIC
"""
accuracy= (TP + TN) ./ 8238
precision= TP ./ TP + FP
recall=     TP ./ TP + FN

"""
PRINT PERFORMANCE METRIC
"""
print("ACCURACY = ",accuracy)
print("PRECISION = ",precision)
print("RECALL = ",recall)
