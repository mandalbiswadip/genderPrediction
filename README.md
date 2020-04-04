# gender_prediction
Gender prediction from Indian names.
The trained model tries to predict the gender of the person given the name of the person. The model has been trained only with Indian names for now.

Model description:
Charecter level embedding(One hot embeddding) has been used and has been fed to a sequential Recurrent Neural Network(bidirectional).
Currently the model is tranied for names with maximum 20 charecters.

I am currently trying to train the model for more than 20 charecters. Also, it will be fun to analyze the patters for recurrent neural networks with a focus on understanding these hidden state dynamics

Few of the predicted results are included in the results file. Other analysis will be uploaded as soon as they are done. 


Sample Results:

For a given name, probablity of being the gender 'male' or 'female' predicted by the model.

name: 'biswadip'
{'male': 0.9898892, 'female': 0.01011086}

name: 'Aadarsh'
{'male': 0.9724812, 'female': 0.02751883}

name: 'Maadhav'
{'male': 0.97186756, 'female': 0.02813238}

name: 'Pabitra'
{'male': 0.270486, 'female': 0.729514}
