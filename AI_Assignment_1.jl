
using CSV
using DataFrames



data= CSV.read("/Users/adrianosilva/Downloads/bank-additional-full.csv")
df= DataFrame(data)

#Our Data preparation steps involved the following:
#Y was converted to binary 0s and 1s, using a loop.
#Apart from the selected attributes, extra dummy attributes were created because to use the one-hot encoding technique
#
#From the original dataset selected attributes were:
#AGE,
#Education,
#loan,
#housing,
#campain,
#poutcome,
#emp.var.rate
#cons.price.idx
#cons.conf.idx

#nr.employed
#
# One-hot encoding was performed on Education, housing, loan, and poutcome

i = 1
y=zeros(Int,41188)
#The dummy variables where created in order to
#implement one-hot encoding
#=
To be able to binarize all the categorical attributes,
firstly attributes containing only zeros were created
then using a loop ones where added where applicable.
=#


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

###############################################END OF DATA preparation############################################
